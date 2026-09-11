# SPDX-License-Identifier: Apache-2.0
"""Measure MaKV PUT/GET and direct paged restore phases.

The benchmark intentionally uses the real MaKV TCP manager and the production
``dequantize_scatter_paged_out`` path. CUDA Event timing is synchronized only
inside this benchmark, never in the serving restore path.

Example for Qwen3-8B's KV geometry on a GPU selected by CUDA_VISIBLE_DEVICES:

    CUDA_VISIBLE_DEVICES=7 python benchmarks/makv_latency_breakdown.py \
      --layers 36 --kv-heads 8 --head-dim 128 --chunk-tokens 2048 \
      --chunks 8 --iterations 3 --output /tmp/makv-latency.json
"""

from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import asyncio
import importlib
import json
import math
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid

# Third Party
import torch

# First Party
import lmcache.lmcache_native as lmcache_native
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.makv_network_connector import (
    MaKVNetworkConnector,
)
from lmcache.v1.storage_backend.makv.metrics import CLIENT_METRICS, RESTORE_METRICS
from lmcache.v1.storage_backend.makv.paged_restore import (
    makv_paged_cuda_op_available,
    restore_makv_quantized_to_paged,
)
from lmcache.v1.storage_backend.makv.serde import MaKVDeserializer, MaKVSerializer


_RESTORE_TIMING_FIELDS = (
    "makv_restore_calls",
    "makv_restore_payload_bytes",
    "makv_restore_cpu_blob_pin_time_ms",
    "makv_restore_cpu_view_validate_time_ms",
    "makv_restore_cpu_dtype_convert_time_ms",
    "makv_restore_cpu_pin_time_ms",
    "makv_restore_cpu_prepare_time_ms",
    "makv_h2d_bytes",
    "makv_h2d_time_ms",
    "makv_dequant_kernel_time_ms",
    "makv_restore_gpu_total_time_ms",
    "makv_kernel_launch_count",
)


@dataclass(frozen=True)
class _Key:
    index: int
    run_id: str

    def to_string(self) -> str:
        return f"makv-latency:{self.run_id}:{self.index}"


@dataclass
class _ManagedServer:
    url: str
    process: subprocess.Popen[bytes] | None
    temporary_store: tempfile.TemporaryDirectory[str] | None

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.temporary_store is not None:
            self.temporary_store.cleanup()


def _parse_csv(value: str, cast: type[float] | type[int]) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": round(statistics.median(values), 3) if values else 0.0,
        "p95_ms": round(_percentile(values, 95.0), 3),
        "mean_ms": round(statistics.fmean(values), 3) if values else 0.0,
    }


def _allocate_paged_cache(
    *,
    layers: int,
    total_tokens: int,
    block_size: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, int]:
    num_blocks = math.ceil(total_tokens / block_size)
    caches = [
        torch.empty(
            (2, num_blocks, block_size, heads, head_dim),
            dtype=dtype,
            device=device,
        )
        for _ in range(layers)
    ]
    pointers = torch.tensor(
        [cache.data_ptr() for cache in caches], dtype=torch.int64, device=device
    )
    slots = torch.arange(total_tokens, dtype=torch.int64, device=device)
    # The paged op receives raw device pointers, so the tensor objects must
    # remain alive for the entire benchmark rather than returning pointers
    # alone.
    return caches, pointers, slots, num_blocks * block_size


def _make_memory_obj(metadata: LMCacheMetadata) -> TensorMemoryObj:
    shape = metadata.get_shapes()[0]
    tensor = torch.randn(shape, dtype=metadata.kv_dtype)
    return TensorMemoryObj(
        raw_data=tensor,
        metadata=MemoryObjMetadata(
            shape=shape,
            dtype=tensor.dtype,
            address=-1,
            phy_size=tensor.numel() * tensor.element_size(),
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.KV_2LTD,
        ),
        parent_allocator=None,
    )


def _wait_for_server(host: str, port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return
        except OSError as error:
            if process.poll() is not None:
                raise RuntimeError("MaKV manager exited during startup") from error
            time.sleep(0.05)
    process.terminate()
    raise RuntimeError("MaKV manager did not start within 20 seconds")


def _start_manager(args: argparse.Namespace) -> _ManagedServer:
    if args.manager_url:
        if args.storage_url is None:
            if args.storage_backend != "file":
                raise ValueError(
                    "--storage-url is required with --manager-url for non-file storage"
                )
            # The client needs this configuration value for validation only;
            # an externally managed server owns the actual file location.
            args.storage_url = "file:///tmp/lmcache-makv"
        return _ManagedServer(args.manager_url, None, None)

    temporary_store: tempfile.TemporaryDirectory[str] | None = None
    storage_url = args.storage_url
    if storage_url is None:
        if args.storage_backend != "file":
            raise ValueError("--storage-url is required for non-file storage")
        temporary_store = tempfile.TemporaryDirectory(prefix="makv-latency-")
        storage_url = f"file://{temporary_store.name}"
    args.storage_url = storage_url

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    command = [
        sys.executable,
        "-m",
        "lmcache.v1.storage_backend.makv_remote.server",
        "--listen",
        f"127.0.0.1:{port}",
        "--storage-url",
        storage_url,
        "--storage-backend",
        args.storage_backend,
        "--storage-namespace",
        args.storage_namespace,
        "--bucket-ratios",
        args.bucket_ratios,
        "--bucket-bits",
        args.bucket_bits,
        "--memory-cache-gb",
        str(args.memory_cache_gb),
        "--entropy-codec",
        args.entropy_codec,
        "--entropy-backend",
        args.entropy_backend,
        "--residual-dtype",
        args.residual_dtype,
    ]
    if args.trust_validated_objects:
        command.append("--trust-validated-objects")
    if args.entropy_require_cuda:
        command.append("--entropy-require-cuda")
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).parents[1],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_server("127.0.0.1", port, process)
    return _ManagedServer(f"makv://127.0.0.1:{port}", process, temporary_store)


def _make_config(
    args: argparse.Namespace, manager_url: str
) -> LMCacheEngineConfig:
    assert args.storage_url is not None
    return LMCacheEngineConfig.from_defaults(
        chunk_size=args.chunk_tokens,
        local_cpu=True,
        remote_url=manager_url,
        remote_serde="makv",
        extra_config={
            "makv_storage_url": args.storage_url,
            "makv_storage_backend": args.storage_backend,
            "makv_storage_namespace": args.storage_namespace,
            "makv_bucket_ratios": list(_parse_csv(args.bucket_ratios, float)),
            "makv_bucket_bits": list(_parse_csv(args.bucket_bits, int)),
            "makv_protect_prefix_tokens": 0,
            "makv_protect_tail_tokens": 0,
            "makv_require_cuda_dequant": True,
            "makv_fallback": "naive",
            "makv_network_timeout_s": args.timeout_s,
            "makv_entropy_codec": args.entropy_codec,
            "makv_entropy_backend": args.entropy_backend,
            "makv_entropy_require_cuda": args.entropy_require_cuda,
            "makv_residual_dtype": args.residual_dtype,
            "makv_batch_blob": args.batch_blob,
            "makv_streaming_restore": args.streaming_restore,
        },
    )


def _restore_batch(
    objects: list[Any],
    *,
    pointers: torch.Tensor,
    slots: torch.Tensor,
    page_buffer_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    scope = RESTORE_METRICS.begin_restore_scope()
    started = time.perf_counter()
    for index, memory_obj in enumerate(objects):
        start = index * args.chunk_tokens
        end = start + args.chunk_tokens
        restore_makv_quantized_to_paged(
            memory_obj,
            device=device,
            page_ptrs=pointers,
            slot_mapping=slots[start:end],
            page_buffer_size=page_buffer_size,
            block_size=args.block_size,
            head_size=args.head_dim,
            engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            timing_scope=scope,
        )
    torch.cuda.synchronize(device)
    timing = asdict(RESTORE_METRICS.finish_restore_scope(scope))
    return {
        **{field: timing[field] for field in _RESTORE_TIMING_FIELDS},
        "wall_ms": (time.perf_counter() - started) * 1000,
    }


async def _fetch_and_restore_streaming(
    connector: MaKVNetworkConnector,
    keys: list[_Key],
    deserializer: MaKVDeserializer,
    *,
    pointers: torch.Tensor,
    slots: torch.Tensor,
    page_buffer_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Receive one MKVB object at a time and enqueue direct paged restore.

    Each ``await`` for chunk ``n + 1`` leaves the previous chunk's H2D,
    arithmetic decode, and paged scatter executing on the current CUDA stream.
    The sole synchronization is the benchmark's final timing boundary.
    """
    scope = RESTORE_METRICS.begin_restore_scope()
    started = time.perf_counter()
    received = 0
    async for index, value in connector.batched_get_streaming(keys):
        if index != received or value is None:
            raise RuntimeError("MaKV streaming benchmark GET returned a cache miss")
        memory_obj = deserializer.deserialize(value)
        start = index * args.chunk_tokens
        end = start + args.chunk_tokens
        restore_makv_quantized_to_paged(
            memory_obj,
            device=device,
            page_ptrs=pointers,
            slot_mapping=slots[start:end],
            page_buffer_size=page_buffer_size,
            block_size=args.block_size,
            head_size=args.head_dim,
            engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            timing_scope=scope,
        )
        received += 1
    if received != len(keys):
        raise RuntimeError("MaKV streaming benchmark returned too few objects")
    torch.cuda.synchronize(device)
    timing = asdict(RESTORE_METRICS.finish_restore_scope(scope))
    return (
        {
            **{field: timing[field] for field in _RESTORE_TIMING_FIELDS},
            "wall_ms": (time.perf_counter() - started) * 1000,
        },
        asdict(CLIENT_METRICS.snapshot()),
    )


async def _delete_keys(connector: MaKVNetworkConnector, keys: list[_Key]) -> None:
    for key in keys:
        try:
            await connector._request("DELETE", key.to_string())
        except Exception:
            # Cleanup must not hide a completed measurement.
            pass


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    importlib.import_module("lmcache.cuda_ops")
    if not torch.cuda.is_available() or not makv_paged_cuda_op_available():
        raise RuntimeError("This benchmark requires the built MaKV CUDA operator")
    if args.chunks <= 0 or args.iterations <= 0 or args.chunk_tokens <= 0:
        raise ValueError("chunks, iterations and chunk-tokens must be positive")

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    server = _start_manager(args)
    config = _make_config(args, server.url)
    total_tokens = args.chunks * args.chunk_tokens
    metadata = LMCacheMetadata(
        model_name=args.model_name,
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=dtype,
        kv_shape=(args.layers, 2, args.chunk_tokens, args.kv_heads, args.head_dim),
        chunk_size=args.chunk_tokens,
    )
    serializer = MaKVSerializer(config, metadata)
    deserializer = MaKVDeserializer(config, metadata)
    connector = MaKVNetworkConnector(
        server.url, asyncio.get_running_loop(), None, config, metadata
    )
    run_id = uuid.uuid4().hex
    keys = [_Key(index, run_id) for index in range(args.chunks)]
    memory_obj = _make_memory_obj(metadata)
    importance = [float(total_tokens - index) for index in range(total_tokens)]

    try:
        CLIENT_METRICS.reset()
        put_started = time.perf_counter()
        for index, key in enumerate(keys):
            start = index * args.chunk_tokens
            envelope = serializer.serialize(
                memory_obj,
                transfer_spec={
                    "chunk_start": start,
                    "chunk_end": start + args.chunk_tokens,
                    "request_token_count": total_tokens,
                    "makv_importance": importance,
                    "makv_importance_layout": "token",
                    "request_configs": {},
                },
                key=key,
            )
            await connector.put(key, envelope)
        put_wall_ms = (time.perf_counter() - put_started) * 1000
        put_client = asdict(CLIENT_METRICS.snapshot())
        health_after_put = await connector.health()

        async def fetch_objects() -> tuple[list[Any], dict[str, Any]]:
            values = await connector.batched_get(keys)
            if any(value is None for value in values):
                raise RuntimeError("MaKV benchmark GET returned a cache miss")
            objects = [deserializer.deserialize(value) for value in values]
            return objects, asdict(CLIENT_METRICS.snapshot())

        # Warm up both manager hot cache and CUDA allocator/event setup.
        caches, pointers, slots, page_buffer_size = _allocate_paged_cache(
            layers=args.layers,
            total_tokens=total_tokens,
            block_size=args.block_size,
            heads=args.kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            device=device,
        )
        for _ in range(args.warmup):
            CLIENT_METRICS.reset()
            RESTORE_METRICS.reset()
            if args.streaming_restore:
                await _fetch_and_restore_streaming(
                    connector,
                    keys,
                    deserializer,
                    pointers=pointers,
                    slots=slots,
                    page_buffer_size=page_buffer_size,
                    args=args,
                    device=device,
                )
            else:
                warm_objects, _ = await fetch_objects()
                _restore_batch(
                    warm_objects,
                    pointers=pointers,
                    slots=slots,
                    page_buffer_size=page_buffer_size,
                    args=args,
                    device=device,
                )

        network_samples: list[dict[str, float]] = []
        restore_samples: list[dict[str, float]] = []
        full_samples: list[dict[str, float]] = []
        reusable_objects: list[Any] | None = None
        for _ in range(args.iterations):
            CLIENT_METRICS.reset()
            network_started = time.perf_counter()
            objects, client_timing = await fetch_objects()
            network_wall_ms = (time.perf_counter() - network_started) * 1000
            reusable_objects = objects
            network_samples.append(
                {
                    "wall_ms": network_wall_ms,
                    "connect_ms": float(
                        client_timing["makv_client_get_connect_time_ms"]
                    ),
                    "send_ms": float(client_timing["makv_client_get_send_time_ms"]),
                    "first_response_ms": float(
                        client_timing["makv_client_get_first_response_time_ms"]
                    ),
                    "receive_ms": float(
                        client_timing["makv_client_get_receive_time_ms"]
                    ),
                    "tcp_total_ms": float(
                        client_timing["makv_client_get_total_time_ms"]
                    ),
                    "batch_blob_bytes": float(
                        client_timing["makv_client_get_batch_blob_bytes"]
                    ),
                    "batch_blob_frames": float(
                        client_timing["makv_client_get_batch_blob_frames"]
                    ),
                    "deserialize_ms": float(
                        client_timing["makv_client_deserialize_time_ms"]
                    ),
                }
            )

        assert reusable_objects is not None
        for _ in range(args.iterations):
            RESTORE_METRICS.reset()
            restore_samples.append(
                _restore_batch(
                    reusable_objects,
                    pointers=pointers,
                    slots=slots,
                    page_buffer_size=page_buffer_size,
                    args=args,
                    device=device,
                )
            )

        for _ in range(args.iterations):
            CLIENT_METRICS.reset()
            RESTORE_METRICS.reset()
            full_started = time.perf_counter()
            if args.streaming_restore:
                restore_timing, client_timing = await _fetch_and_restore_streaming(
                    connector,
                    keys,
                    deserializer,
                    pointers=pointers,
                    slots=slots,
                    page_buffer_size=page_buffer_size,
                    args=args,
                    device=device,
                )
            else:
                objects, client_timing = await fetch_objects()
                restore_timing = _restore_batch(
                    objects,
                    pointers=pointers,
                    slots=slots,
                    page_buffer_size=page_buffer_size,
                    args=args,
                    device=device,
                )
            full_samples.append(
                {
                    "wall_ms": (time.perf_counter() - full_started) * 1000,
                    "tcp_total_ms": float(
                        client_timing["makv_client_get_total_time_ms"]
                    ),
                    "deserialize_ms": float(
                        client_timing["makv_client_deserialize_time_ms"]
                    ),
                    "cpu_prepare_ms": float(
                        restore_timing["makv_restore_cpu_prepare_time_ms"]
                    ),
                    "h2d_ms": float(restore_timing["makv_h2d_time_ms"]),
                    "kernel_ms": float(
                        restore_timing["makv_dequant_kernel_time_ms"]
                    ),
                }
            )

        health_after_get = await connector.health()
        client_after_get = asdict(CLIENT_METRICS.snapshot())
        raw_bytes = int(health_after_put["metrics"]["makv_raw_input_bytes"])
        stored_bytes = int(health_after_put["metrics"]["makv_stored_bytes"])
        restore_h2d_sum = sum(sample["makv_h2d_time_ms"] for sample in restore_samples)
        restore_kernel_sum = sum(
            sample["makv_dequant_kernel_time_ms"] for sample in restore_samples
        )
        restore_h2d_bytes = sum(sample["makv_h2d_bytes"] for sample in restore_samples)

        # Keep the actual paged allocations alive until every CUDA event has
        # completed. The C++ op intentionally operates on raw page pointers.
        if not caches:
            raise RuntimeError("MaKV latency benchmark allocated no paged KV cache")

        def phase_summary(
            samples: list[dict[str, float]],
        ) -> dict[str, dict[str, float]]:
            return {
                key: _summary([sample[key] for sample in samples])
                for key in samples[0]
            }

        network_summary = phase_summary(network_samples)
        restore_summary = phase_summary(restore_samples)
        full_hit_summary = phase_summary(full_samples)
        put_manager = {
            key: health_after_put["metrics"][key]
            for key in (
                "makv_remote_put_requests",
                "makv_remote_quantize_queue_time_ms",
                "makv_remote_put_decode_time_ms",
                "makv_remote_plan_canonicalize_time_ms",
                "makv_remote_quantize_time_ms",
                "makv_remote_quantize_kernel_time_ms",
                "makv_remote_object_encode_time_ms",
                "makv_remote_object_validate_time_ms",
                "makv_remote_encode_validate_time_ms",
                "makv_remote_storage_put_time_ms",
                "makv_remote_put_total_time_ms",
            )
        }
        put_total_ms = float(put_manager["makv_remote_put_total_time_ms"])
        full_wall_ms = float(full_hit_summary["wall_ms"]["median_ms"])
        full_tcp_ms = float(full_hit_summary["tcp_total_ms"]["median_ms"])
        full_deserialize_ms = float(full_hit_summary["deserialize_ms"]["median_ms"])
        full_cpu_prepare_ms = float(full_hit_summary["cpu_prepare_ms"]["median_ms"])
        full_h2d_ms = float(full_hit_summary["h2d_ms"]["median_ms"])
        full_kernel_ms = float(full_hit_summary["kernel_ms"]["median_ms"])
        full_accounted_ms = (
            full_tcp_ms
            + full_deserialize_ms
            + full_cpu_prepare_ms
            + full_h2d_ms
            + full_kernel_ms
        )

        result = {
            "geometry": {
                "layers": args.layers,
                "kv_heads": args.kv_heads,
                "head_dim": args.head_dim,
                "chunk_tokens": args.chunk_tokens,
                "chunks": args.chunks,
                "total_tokens": total_tokens,
                "dtype": args.dtype,
                "block_size": args.block_size,
            },
            "entropy": {
                "codec": args.entropy_codec,
                "backend": args.entropy_backend,
                "require_cuda": args.entropy_require_cuda,
            },
            "residual_dtype": args.residual_dtype,
            "transport_options": {
                "batch_blob": args.batch_blob,
                "streaming_restore": args.streaming_restore,
                "trust_validated_objects": args.trust_validated_objects,
            },
            "storage": {
                "backend": args.storage_backend,
                "raw_kv_bytes": raw_bytes,
                "remote_stored_bytes": stored_bytes,
                "compression_ratio": raw_bytes / stored_bytes if stored_bytes else 0.0,
            },
            "put": {
                "wall_ms": round(put_wall_ms, 3),
                "client": {
                    key: put_client[key]
                    for key in (
                        "makv_plan_time_ms",
                        "makv_client_plan_build_time_ms",
                        "makv_client_raw_payload_copy_time_ms",
                        "makv_client_envelope_encode_time_ms",
                        "makv_client_serialize_total_time_ms",
                        "makv_client_quantize_calls",
                        "makv_put_raw_bytes",
                        "makv_put_plan_bytes",
                        "makv_client_put_connect_time_ms",
                        "makv_client_put_send_time_ms",
                        "makv_client_put_response_time_ms",
                        "makv_client_put_total_time_ms",
                    )
                },
                "manager": put_manager,
            },
            "network_get": network_summary,
            "transport": {
                "client_pinned_receive_bytes": int(
                    client_after_get["makv_client_pinned_receive_bytes"]
                ),
                "client_pinned_receive_fallbacks": int(
                    client_after_get["makv_client_pinned_receive_fallbacks"]
                ),
            },
            "restore": {
                "phases": restore_summary,
                "h2d_share_of_gpu_pct": round(
                    100 * restore_h2d_sum / (restore_h2d_sum + restore_kernel_sum), 2
                )
                if restore_h2d_sum + restore_kernel_sum
                else 0.0,
                "kernel_share_of_gpu_pct": round(
                    100 * restore_kernel_sum / (restore_h2d_sum + restore_kernel_sum), 2
                )
                if restore_h2d_sum + restore_kernel_sum
                else 0.0,
                "effective_h2d_gbps": round(
                    restore_h2d_bytes / (restore_h2d_sum / 1000) / 1024**3, 3
                )
                if restore_h2d_sum
                else 0.0,
            },
            "full_hit": full_hit_summary,
            "ratios": {
                "remote_put": {
                    "quantize_core_share_of_manager_put_pct": round(
                        100
                        * float(put_manager["makv_remote_quantize_kernel_time_ms"])
                        / put_total_ms,
                        2,
                    )
                    if put_total_ms
                    else 0.0,
                    "quantize_total_share_of_manager_put_pct": round(
                        100
                        * float(put_manager["makv_remote_quantize_time_ms"])
                        / put_total_ms,
                        2,
                    )
                    if put_total_ms
                    else 0.0,
                    "storage_share_of_manager_put_pct": round(
                        100
                        * float(put_manager["makv_remote_storage_put_time_ms"])
                        / put_total_ms,
                        2,
                    )
                    if put_total_ms
                    else 0.0,
                },
                "full_hit_median": {
                    "tcp_share_of_wall_pct": round(100 * full_tcp_ms / full_wall_ms, 2)
                    if full_wall_ms
                    else 0.0,
                    "deserialize_share_of_wall_pct": round(
                        100 * full_deserialize_ms / full_wall_ms, 2
                    )
                    if full_wall_ms
                    else 0.0,
                    "cpu_prepare_share_of_wall_pct": round(
                        100 * full_cpu_prepare_ms / full_wall_ms, 2
                    )
                    if full_wall_ms
                    else 0.0,
                    "h2d_share_of_wall_pct": round(100 * full_h2d_ms / full_wall_ms, 2)
                    if full_wall_ms
                    else 0.0,
                    "fused_kernel_share_of_wall_pct": round(
                        100 * full_kernel_ms / full_wall_ms, 2
                    )
                    if full_wall_ms
                    else 0.0,
                    "other_unattributed_share_of_wall_pct": round(
                        100 * max(0.0, full_wall_ms - full_accounted_ms) / full_wall_ms,
                        2,
                    )
                    if full_wall_ms
                    else 0.0,
                },
            },
            "manager_after_get": health_after_get,
        }
        return result
    finally:
        await _delete_keys(connector, keys)
        server.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MaKV latency phase benchmark")
    parser.add_argument("--model-name", default="Qwen3-8B")
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--manager-url", default=None)
    parser.add_argument("--storage-url", default=None)
    parser.add_argument(
        "--storage-backend", choices=("file", "redis", "mooncake"), default="file"
    )
    parser.add_argument("--storage-namespace", default="lmcache:makv:latency:")
    parser.add_argument("--memory-cache-gb", type=float, default=2.0)
    parser.add_argument(
        "--trust-validated-objects",
        action="store_true",
        help="Skip repeat manager CRC scans for manager-owned immutable objects.",
    )
    parser.add_argument(
        "--batch-blob",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the MKVB single-frame GET_BATCH response (default: enabled).",
    )
    parser.add_argument(
        "--streaming-restore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restore each MKVB object while receiving the next one "
            "(default: enabled)."
        ),
    )
    parser.add_argument("--bucket-ratios", default="0.2,0.3,0.5")
    parser.add_argument("--bucket-bits", default="16,8,4")
    parser.add_argument(
        "--entropy-codec",
        choices=("none", "cachegen_arithmetic"),
        default="none",
    )
    parser.add_argument(
        "--entropy-backend",
        choices=("auto", "cuda", "reference"),
        default="auto",
    )
    parser.add_argument("--entropy-require-cuda", action="store_true")
    parser.add_argument(
        "--residual-dtype",
        choices=("none", "float16", "float32"),
        default="none",
        help="Store manager-side residuals for later precision upgrades.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = asyncio.run(_run(args))
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
