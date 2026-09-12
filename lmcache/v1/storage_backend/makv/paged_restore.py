# SPDX-License-Identifier: Apache-2.0

"""Direct MaKV restoration into engine paged KV-cache buffers."""

# Standard
from typing import Any
import time

# Third Party
import torch

# First Party
from lmcache.v1.storage_backend.makv.gpu_restore import (
    _pinned_blob_tensor,
    makv_cuda_op_available,
    payload_tensors_from_gpu_blob,
    payload_tensors_from_obj,
)
from lmcache.v1.storage_backend.makv.memory import MaKVQuantizedMemoryObj
from lmcache.v1.storage_backend.makv.metrics import RESTORE_METRICS


def makv_paged_cuda_op_available() -> bool:
    """Return whether the direct-to-paged MaKV CUDA operator is registered."""
    return makv_cuda_op_available() and hasattr(
        torch.ops.lmcache_makv, "dequantize_scatter_paged_out"
    )


def _validate_bucket_payloads(
    plan: dict[str, Any],
    payloads: dict[int, dict[str, torch.Tensor]],
    metadata: dict[str, Any] | None = None,
) -> None:
    layout = str(plan["importance_layout"])
    layers = int(plan["num_layers"])
    heads = int(plan["num_kv_heads"])
    head_dim = int(plan["head_dim"])
    tokens = int(plan["chunk_length"])
    all_positions: list[torch.Tensor] = []
    for bits in (16, 8, 4, 2):
        item = payloads[bits]
        positions = item["positions"]
        count = positions.numel()
        all_positions.append(positions)
        vectors = (
            count * heads if layout == "layer_kv_token" else count * layers * 2 * heads
        )
        expected_payload = vectors * head_dim
        if bits in (4, 2):
            values_per_byte = 8 // bits
            expected_payload = vectors * (
                (head_dim + values_per_byte - 1) // values_per_byte
            )
        actual_payload = item["payload"].numel()
        entropy_bucket = None
        if metadata is not None:
            entropy = metadata.get("entropy")
            if isinstance(entropy, dict):
                entropy_bucket = entropy.get("buckets", {}).get(str(bits))
        if actual_payload != expected_payload and not (
            actual_payload == 0
            and isinstance(entropy_bucket, dict)
            and int(entropy_bucket.get("vectors", -1)) == vectors
            and int(entropy_bucket.get("head_dim", -1)) == head_dim
            and int(entropy_bucket.get("logical_elements", -1))
            == vectors * head_dim
        ):
            raise ValueError(f"MaKV {bits}-bit payload length mismatch")
        expected_scales = 0 if bits == 16 else vectors
        if item["scales"].numel() != expected_scales:
            raise ValueError(f"MaKV {bits}-bit scale count mismatch")
    positions = torch.cat(all_positions) if all_positions else torch.empty(0)
    expected_count = tokens if layout == "token" else layers * 2 * tokens
    if positions.numel() != expected_count:
        raise ValueError("MaKV bucket positions do not fully cover the chunk")
    sorted_positions = torch.sort(positions.to(torch.int64)).values
    if not torch.equal(sorted_positions, torch.arange(expected_count)):
        raise ValueError("MaKV bucket positions are out of range or duplicated")


def _prepare_cpu_tensor(
    tensor: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, float, float]:
    """Make a correctly typed pinned host tensor before recording H2D events."""
    dtype_started = time.perf_counter()
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    dtype_convert_ms = (time.perf_counter() - dtype_started) * 1000

    # vLLM normally supplies slot_mapping on the target GPU. It is already
    # usable by the fused kernel and cannot be pinned as host memory.
    if tensor.device.type != "cpu":
        return tensor, dtype_convert_ms, 0.0

    pin_started = time.perf_counter()
    if tensor.numel() > 0 and not tensor.is_pinned():
        tensor = tensor.pin_memory()
    pin_ms = (time.perf_counter() - pin_started) * 1000
    return tensor, dtype_convert_ms, pin_ms


def _copy_to_cuda(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Enqueue a copy after the H2D start CUDA event has been recorded."""
    return tensor.to(device=device, non_blocking=True)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def restore_makv_quantized_to_paged(
    memory_obj: MaKVQuantizedMemoryObj,
    *,
    device: torch.device,
    page_ptrs: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_buffer_size: int,
    block_size: int,
    head_size: int,
    engine_kv_format: Any,
    skip_prefix_n_tokens: int = 0,
    require_cuda: bool = True,
    timing_scope: int | None = None,
    h2d_stream: torch.cuda.Stream | None = None,
    max_inflight_tickets: int | None = None,
) -> None:
    """Restore a MaKV object directly into final paged KV-cache buffers."""
    if require_cuda and not makv_paged_cuda_op_available():
        raise RuntimeError("MaKV direct-to-paged CUDA operator is unavailable")
    if device.type != "cuda" or not makv_paged_cuda_op_available():
        raise RuntimeError("MaKV paged restore requires the CUDA operator")
    if max_inflight_tickets is not None and int(max_inflight_tickets) < 1:
        raise ValueError("max_inflight_tickets must be at least one")

    cpu_started = time.perf_counter()
    plan = memory_obj.makv_metadata["plan"]
    try:
        format_value = int(engine_kv_format)
    except TypeError:
        format_value = int(engine_kv_format.value)
    expected_head_size = int(plan["head_dim"]) * (2 if format_value in (12, 13) else 1)
    if int(head_size) != expected_head_size:
        raise ValueError("MaKV head dimension does not match paged KV cache")
    blob_pin_started = time.perf_counter()
    pinned_blob = _pinned_blob_tensor(memory_obj)
    blob_pin_ms = (time.perf_counter() - blob_pin_started) * 1000
    raw_dtype = getattr(torch, str(plan["original_dtype"]).replace("torch.", ""))
    with torch.cuda.device(device):
        compute_stream = torch.cuda.current_stream(device)
        copy_stream = h2d_stream if h2d_stream is not None else compute_stream
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        compute_start = torch.cuda.Event(enable_timing=True)
        kernel_end = torch.cuda.Event(enable_timing=True)
        h2d_start.record(copy_stream)

        view_validate_started = time.perf_counter()
        payloads = payload_tensors_from_obj(
            memory_obj,
            pinned_blob=pinned_blob,
            decode_entropy=False,
        )
        _validate_bucket_payloads(plan, payloads, memory_obj.makv_metadata)
        if slot_mapping.numel() < int(plan["chunk_length"]):
            raise ValueError("slot_mapping is shorter than the MaKV chunk")
        view_validate_ms = (time.perf_counter() - view_validate_started) * 1000

        prepared_slot_mapping, dtype_convert_ms, pin_ms = _prepare_cpu_tensor(
            slot_mapping, torch.int64
        )

        payload_bytes = sum(
            _tensor_nbytes(payloads[bits][name])
            for bits in (16, 8, 4, 2)
            for name in ("positions", "payload", "scales")
        )
        h2d_bytes = _tensor_nbytes(pinned_blob)
        if prepared_slot_mapping.device.type != "cuda":
            h2d_bytes += _tensor_nbytes(prepared_slot_mapping)
        RESTORE_METRICS.add_restore(
            timing_scope,
            makv_restore_cpu_blob_pin_time_ms=blob_pin_ms,
            makv_restore_cpu_view_validate_time_ms=view_validate_ms,
            makv_restore_cpu_dtype_convert_time_ms=dtype_convert_ms,
            makv_restore_cpu_pin_time_ms=pin_ms,
            makv_restore_cpu_prepare_time_ms=(time.perf_counter() - cpu_started)
            * 1000,
        )

        # The object is aligned by the protocol encoder, so one byte-buffer
        # copy supplies zero-copy device views for positions and payloads.
        # Keep all copy-side work on the dedicated stream so the next object
        # can transfer while the compute stream restores the previous one.
        with torch.cuda.stream(copy_stream):
            gpu_blob = _copy_to_cuda(pinned_blob, device)
            prepared_slot_mapping = _copy_to_cuda(prepared_slot_mapping, device)
            gpu_payloads = payload_tensors_from_gpu_blob(
                memory_obj,
                gpu_blob,
                cpu_payloads=payloads,
            )
            # The current CUDA ABI consumes float32 scales. This conversion is
            # limited to the small scale arrays; the large payload remains one
            # contiguous H2D transfer.
            for bits in (8, 4, 2):
                scales = gpu_payloads[bits]["scales"]
                if scales.dtype != torch.float32:
                    gpu_payloads[bits]["scales"] = scales.to(torch.float32)
            h2d_end.record(copy_stream)

        # A paged write may only consume a fully transferred, validated
        # object. This event is the sole cross-stream dependency.
        compute_stream.wait_event(h2d_end)
        compute_start.record(compute_stream)

        (
            raw16,
            int8_payload,
            int4_payload,
            int2_payload,
            pos16,
            pos8,
            pos4,
            pos2,
            int8_scales,
            int4_scales,
            int2_scales,
        ) = (
            gpu_payloads[16]["payload"],
            gpu_payloads[8]["payload"],
            gpu_payloads[4]["payload"],
            gpu_payloads[2]["payload"],
            gpu_payloads[16]["positions"],
            gpu_payloads[8]["positions"],
            gpu_payloads[4]["positions"],
            gpu_payloads[2]["positions"],
            gpu_payloads[8]["scales"],
            gpu_payloads[4]["scales"],
            gpu_payloads[2]["scales"],
        )
        slot_mapping = prepared_slot_mapping

        torch.ops.lmcache_makv.dequantize_scatter_paged_out(
            raw16,
            int8_payload,
            int4_payload,
            int2_payload,
            int8_scales,
            int4_scales,
            int2_scales,
            pos16,
            pos8,
            pos4,
            pos2,
            page_ptrs,
            slot_mapping,
            int(plan["importance_layout"] == "layer_kv_token"),
            int(plan["chunk_length"]),
            int(plan["num_layers"]),
            int(plan["num_kv_heads"]),
            int(plan["head_dim"]),
            int(page_buffer_size),
            int(block_size),
            int(format_value),
            int(raw_dtype == torch.bfloat16),
            int(skip_prefix_n_tokens),
        )
        kernel_end.record(compute_stream)

    RESTORE_METRICS.record_cuda_restore(
        timing_scope,
        h2d_start=h2d_start,
        h2d_end=h2d_end,
        compute_start=compute_start,
        kernel_end=kernel_end,
        payload_bytes=payload_bytes,
        h2d_bytes=h2d_bytes,
        kernel_launch_count=sum(
            int(payloads[bits]["positions"].numel() > 0)
            for bits in (16, 8, 4, 2)
        ),
        max_inflight_tickets=max_inflight_tickets,
        # Hold the host sources and GPU inputs until kernel_end.  This is
        # required for a genuinely asynchronous H2D path.
        keepalive=(
            pinned_blob,
            gpu_blob,
            raw16,
            int8_payload,
            int4_payload,
            int2_payload,
            pos16,
            pos8,
            pos4,
            pos2,
            int8_scales,
            int4_scales,
            int2_scales,
            slot_mapping,
        ),
    )
