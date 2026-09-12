# SPDX-License-Identifier: Apache-2.0

"""CUDA differential tests for MaKV contiguous and direct paged restore."""

# Standard
import gc

# Third Party
import pytest
import torch

# First Party
import lmcache.cuda_ops as lmc_ops
import lmcache.lmcache_native as lmcache_native
import lmcache.v1.gpu_connector.gpu_connectors as gpu_connectors
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
from lmcache.v1.gpu_connector.makv_restore import (
    begin_makv_restore_timing_scope,
    finish_makv_restore_timing_scope,
    handoff_makv_restore_ready_event,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.makv.config import MaKVConfig
from lmcache.v1.storage_backend.makv.format import (
    decode_makv_object,
    encode_makv_object,
)
from lmcache.v1.storage_backend.makv.entropy import encode_entropy_payloads
from lmcache.v1.storage_backend.makv.gpu_restore import (
    restore_makv_quantized_to_tensor,
)
from lmcache.v1.storage_backend.makv.memory import MaKVQuantizedMemoryObj
from lmcache.v1.storage_backend.makv.metrics import RESTORE_METRICS
from lmcache.v1.storage_backend.makv.paged_restore import (
    makv_paged_cuda_op_available,
    restore_makv_quantized_to_paged,
)
from lmcache.v1.storage_backend.makv.plan import MaKVQuantPlan
from lmcache.v1.storage_backend.makv.plan import build_chunk_quant_plan
from lmcache.v1.storage_backend.makv.quantizer import quantize_canonical_kv

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not makv_paged_cuda_op_available(),
    reason="MaKV CUDA extension is unavailable",
)


def _config(*, entropy: bool = False) -> MaKVConfig:
    return MaKVConfig(
        storage_url="file:///tmp/makv-cuda-test",
        bucket_ratios=(0.1, 0.1, 0.6, 0.2),
        bucket_bits=(16, 8, 4, 2),
        importance_layout="token",
        quant_granularity="per_token_head",
        scale_dtype="float16",
        protect_prefix_tokens=0,
        protect_tail_tokens=0,
        dequant_backend="cuda",
        require_cuda_dequant=True,
        fallback="naive",
        enable_checksum=True,
        entropy_codec="cachegen_arithmetic" if entropy else "none",
        entropy_backend="cuda" if entropy else "auto",
        entropy_require_cuda=entropy,
    )


def _quantized_object(
    dtype: torch.dtype,
    bucket_ids: list[int],
    *,
    layout: str = "token",
    layers: int = 2,
    tokens: int = 11,
    heads: int = 3,
    head_dim: int = 5,
    entropy: bool = False,
) -> MaKVQuantizedMemoryObj:
    plan = MaKVQuantPlan(
        protocol_version=1,
        importance_layout=layout,
        token_count=tokens,
        chunk_start=0,
        chunk_length=tokens,
        bucket_bits=(16, 8, 4, 2),
        bucket_ids=bytes(bucket_ids),
        original_shape=(2, layers, tokens, heads * head_dim),
        original_strides=(
            layers * tokens * heads * head_dim,
            tokens * heads * head_dim,
            heads * head_dim,
            1,
        ),
        original_dtype=str(dtype),
        token_dim=2,
        num_layers=layers,
        num_kv_heads=heads,
        head_dim=head_dim,
        quant_granularity="per_token_head",
        scale_dtype="float16",
        model_fingerprint="model",
        parallel_fingerprint="parallel",
        checksum=0,
    )
    generator = torch.Generator().manual_seed(1234)
    canonical = torch.randn(
        layers,
        2,
        tokens,
        heads,
        head_dim,
        dtype=dtype,
        generator=generator,
    )
    config = _config(entropy=entropy)
    metadata, payloads = quantize_canonical_kv(canonical, plan, config)
    if entropy:
        metadata, payloads = encode_entropy_payloads(
            metadata,
            payloads,
            codec=config.entropy_codec,
            backend=config.entropy_backend,
            require_cuda=config.entropy_require_cuda,
        )
    blob = encode_makv_object(
        object_type="quantized", metadata=metadata, payloads=payloads
    )
    decoded = decode_makv_object(blob, copy_payloads=False)
    return MaKVQuantizedMemoryObj(
        blob,
        metadata_dict=decoded.metadata,
        payloads=decoded.payloads,
    )


def _quantized_kv_separate_object(
    dtype: torch.dtype,
    *,
    scheme: str = "kv_separate_3tier",
) -> MaKVQuantizedMemoryObj:
    """Build a K/V-separated object for the direct-paged CUDA path."""
    layers, tokens, heads, head_dim = 2, 11, 3, 5
    if scheme == "kv_separate_4tier":
        ratios = (0.1, 0.2, 0.5, 0.2)
        bits = (16, 8, 4, 2)
    else:
        ratios = (0.2, 0.3, 0.5)
        bits = (8, 4, 2)
    config = MaKVConfig(
        storage_url="file:///tmp/makv-cuda-kv-separate-test",
        bucket_ratios=ratios,
        bucket_bits=bits,
        importance_layout="token",
        quant_granularity="per_token_head",
        scale_dtype="float16",
        protect_prefix_tokens=0,
        protect_tail_tokens=0,
        dequant_backend="cuda",
        require_cuda_dequant=True,
        fallback="naive",
        enable_checksum=True,
        storage_backend="file",
        precision_scheme=scheme,
    )
    plan = build_chunk_quant_plan(
        importance=list(range(tokens, 0, -1)),
        importance_layout_hint=None,
        chunk_start=0,
        chunk_end=tokens,
        original_shape=(2, layers, tokens, heads * head_dim),
        original_strides=(layers * tokens * heads * head_dim,
                          tokens * heads * head_dim,
                          heads * head_dim, 1),
        original_dtype=str(dtype),
        token_dim=2,
        num_layers=layers,
        num_kv_heads=heads,
        head_dim=head_dim,
        model_name="makv-cuda-kv-separate-test",
        world_size=1,
        worker_id=0,
        config=config,
        request_token_count=tokens,
    )
    generator = torch.Generator().manual_seed(4321)
    canonical = torch.randn(
        layers,
        2,
        tokens,
        heads,
        head_dim,
        dtype=dtype,
        generator=generator,
    )
    metadata, payloads = quantize_canonical_kv(canonical, plan, config)
    blob = encode_makv_object(
        object_type="quantized", metadata=metadata, payloads=payloads
    )
    decoded = decode_makv_object(blob, copy_payloads=False)
    return MaKVQuantizedMemoryObj(
        blob,
        metadata_dict=decoded.metadata,
        payloads=decoded.payloads,
    )


def _cache_shape(
    engine_format: int,
    num_blocks: int,
    block_size: int,
    heads: int,
    head_dim: int,
) -> tuple[int, ...]:
    if engine_format == 1:
        return (2, num_blocks, block_size, heads, head_dim)
    if engine_format == 2:
        return (num_blocks, 2, block_size, heads, head_dim)
    if engine_format == 6:
        return (2, num_blocks, heads, block_size, head_dim)
    if engine_format == 7:
        return (num_blocks, 2, heads, block_size, head_dim)
    if engine_format == 12:
        return (num_blocks, heads, block_size, 2 * head_dim)
    if engine_format == 13:
        return (num_blocks, block_size, heads, 2 * head_dim)
    raise AssertionError("unsupported test format")


def _run_paths(
    memory_obj: MaKVQuantizedMemoryObj,
    dtype: torch.dtype,
    engine_format: int,
    *,
    use_separate_h2d_stream: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    plan = memory_obj.makv_metadata["plan"]
    layers = int(plan["num_layers"])
    tokens = int(plan["chunk_length"])
    heads = int(plan["num_kv_heads"])
    head_dim = int(plan["head_dim"])
    block_size = 4
    num_blocks = 5
    shape = _cache_shape(engine_format, num_blocks, block_size, heads, head_dim)
    path_a = [torch.zeros(shape, dtype=dtype, device="cuda") for _ in range(layers)]
    path_b = [torch.zeros_like(tensor) for tensor in path_a]
    ptrs_a = torch.tensor(
        [tensor.data_ptr() for tensor in path_a],
        dtype=torch.int64,
        device="cuda",
    )
    ptrs_b = torch.tensor(
        [tensor.data_ptr() for tensor in path_b],
        dtype=torch.int64,
        device="cuda",
    )
    slots = torch.randperm(
        num_blocks * block_size,
        generator=torch.Generator(device="cuda").manual_seed(7),
        device="cuda",
    )[:tokens].to(torch.int64)
    stream = torch.cuda.Stream()
    h2d_stream = torch.cuda.Stream() if use_separate_h2d_stream else None
    with torch.cuda.stream(stream):
        contiguous = restore_makv_quantized_to_tensor(
            memory_obj,
            torch.device("cuda"),
            require_cuda=True,
        ).tensor
        assert contiguous is not None
        lmc_ops.multi_layer_kv_transfer(
            contiguous,
            ptrs_a,
            slots,
            torch.device("cuda"),
            num_blocks * block_size,
            lmcache_native.TransferDirection.H2D,
            lmcache_native.EngineKVFormat(engine_format),
            block_size=block_size,
            head_size=head_dim,
        )
        restore_makv_quantized_to_paged(
            memory_obj,
            device=torch.device("cuda"),
            page_ptrs=ptrs_b,
            slot_mapping=slots,
            page_buffer_size=num_blocks * block_size,
            block_size=block_size,
            head_size=head_dim,
            engine_kv_format=lmcache_native.EngineKVFormat(engine_format),
            h2d_stream=h2d_stream,
        )
    stream.synchronize()
    return path_a, path_b


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "bucket_ids",
    (
        [0] * 11,
        [1] * 11,
        [2] * 11,
        [3] * 11,
        [index % 4 for index in range(11)],
    ),
)
def test_direct_paged_matches_contiguous(dtype, bucket_ids):
    memory_obj = _quantized_object(dtype, bucket_ids)
    path_a, path_b = _run_paths(memory_obj, dtype, engine_format=1)
    assert all(
        torch.equal(left, right) for left, right in zip(path_a, path_b, strict=True)
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_direct_paged_separate_h2d_stream_matches_contiguous(dtype):
    """The compute stream must wait for the dedicated H2D stream event."""
    memory_obj = _quantized_object(dtype, [index % 4 for index in range(11)])
    path_a, path_b = _run_paths(
        memory_obj,
        dtype,
        engine_format=1,
        use_separate_h2d_stream=True,
    )
    assert all(
        torch.equal(left, right) for left, right in zip(path_a, path_b, strict=True)
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_direct_paged_arithmetic_matches_plain_makv(dtype):
    """GPU arithmetic decode must preserve the direct paged result exactly."""
    plain_a, _ = _run_paths(
        _quantized_object(dtype, [index % 4 for index in range(11)]),
        dtype,
        engine_format=1,
    )
    entropy_a, entropy_b = _run_paths(
        _quantized_object(
            dtype,
            [index % 4 for index in range(11)],
            entropy=True,
        ),
        dtype,
        engine_format=1,
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(plain_a, entropy_a, strict=True)
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(entropy_a, entropy_b, strict=True)
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_direct_paged_arithmetic_supports_empty_buckets(dtype):
    """Empty arithmetic buckets must retain CUDA device placement."""
    path_a, path_b = _run_paths(
        _quantized_object(dtype, [0] * 11, entropy=True),
        dtype,
        engine_format=1,
    )
    assert all(
        torch.equal(left, right) for left, right in zip(path_a, path_b, strict=True)
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_direct_paged_kv_separate_matches_contiguous(dtype):
    memory_obj = _quantized_kv_separate_object(dtype)
    path_a, path_b = _run_paths(memory_obj, dtype, engine_format=1)
    assert all(
        torch.equal(left, right) for left, right in zip(path_a, path_b, strict=True)
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_direct_paged_kv_separate_4tier_matches_contiguous(dtype):
    memory_obj = _quantized_kv_separate_object(
        dtype,
        scheme="kv_separate_4tier",
    )
    path_a, path_b = _run_paths(memory_obj, dtype, engine_format=1)
    assert all(
        torch.equal(left, right) for left, right in zip(path_a, path_b, strict=True)
    )


@pytest.mark.parametrize("engine_format", (2, 6, 7))
def test_direct_paged_vllm_layouts(engine_format):
    memory_obj = _quantized_object(
        torch.float16,
        [index % 4 for index in range(11)],
    )
    path_a, path_b = _run_paths(memory_obj, torch.float16, engine_format)
    assert all(
        torch.equal(left, right) for left, right in zip(path_a, path_b, strict=True)
    )


@pytest.mark.parametrize("engine_format", (12, 13))
def test_direct_paged_fused_content_layouts(engine_format):
    dtype = torch.bfloat16
    memory_obj = _quantized_object(dtype, [index % 4 for index in range(11)])
    plan = memory_obj.makv_metadata["plan"]
    layers = int(plan["num_layers"])
    tokens = int(plan["chunk_length"])
    heads = int(plan["num_kv_heads"])
    head_dim = int(plan["head_dim"])
    block_size = 4
    num_blocks = 5
    shape = _cache_shape(engine_format, num_blocks, block_size, heads, head_dim)
    expected = [torch.zeros(shape, dtype=dtype, device="cuda") for _ in range(layers)]
    actual = [torch.zeros_like(tensor) for tensor in expected]
    pointers = torch.tensor(
        [tensor.data_ptr() for tensor in actual], dtype=torch.int64, device="cuda"
    )
    slots = torch.randperm(20, device="cuda")[:tokens].to(torch.int64)
    restored = restore_makv_quantized_to_tensor(
        memory_obj, torch.device("cuda"), require_cuda=True
    ).tensor
    assert restored is not None
    for token, slot in enumerate(slots.tolist()):
        block, offset = divmod(slot, block_size)
        values = (
            restored[:, :, token, :]
            .permute(1, 0, 2)
            .reshape(layers, 2, heads, head_dim)
        )
        for layer in range(layers):
            if engine_format == 12:
                expected[layer][block, :, offset, :head_dim] = values[layer, 0]
                expected[layer][block, :, offset, head_dim:] = values[layer, 1]
            else:
                expected[layer][block, offset, :, :head_dim] = values[layer, 0]
                expected[layer][block, offset, :, head_dim:] = values[layer, 1]
    restore_makv_quantized_to_paged(
        memory_obj,
        device=torch.device("cuda"),
        page_ptrs=pointers,
        slot_mapping=slots,
        page_buffer_size=num_blocks * block_size,
        block_size=block_size,
        head_size=2 * head_dim,
        engine_kv_format=lmcache_native.EngineKVFormat(engine_format),
    )
    torch.cuda.synchronize()
    assert all(
        torch.equal(left, right) for left, right in zip(expected, actual, strict=True)
    )


def test_direct_paged_layer_kv_token_layout():
    layers = 2
    tokens = 11
    bucket_ids = [index % 4 for index in range(layers * 2 * tokens)]
    memory_obj = _quantized_object(
        torch.bfloat16,
        bucket_ids,
        layout="layer_kv_token",
    )
    path_a, path_b = _run_paths(memory_obj, torch.bfloat16, engine_format=1)
    assert all(
        torch.equal(left, right) for left, right in zip(path_a, path_b, strict=True)
    )


def test_direct_paged_rejects_duplicate_positions_before_write():
    memory_obj = _quantized_object(
        torch.float16,
        [index % 4 for index in range(11)],
    )
    table = {
        str(entry["name"]): entry
        for entry in memory_obj.makv_metadata["_payload_table"]
    }
    corrupted_blob = bytearray(memory_obj.byte_array)
    positions_4 = int(table["positions_4"]["offset"])
    positions_16 = int(table["positions_16"]["offset"])
    corrupted_blob[positions_4 : positions_4 + 4] = corrupted_blob[
        positions_16 : positions_16 + 4
    ]
    memory_obj.raw_data = corrupted_blob
    cache = torch.zeros((2, 5, 4, 3, 5), dtype=torch.float16, device="cuda")
    ptrs = torch.tensor([cache.data_ptr(), cache.data_ptr()], device="cuda")
    with pytest.raises(ValueError, match="out of range or duplicated"):
        restore_makv_quantized_to_paged(
            memory_obj,
            device=torch.device("cuda"),
            page_ptrs=ptrs,
            slot_mapping=torch.arange(11, device="cuda"),
            page_buffer_size=20,
            block_size=4,
            head_size=5,
            engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        )
    assert torch.count_nonzero(cache).item() == 0


def test_direct_paged_rejects_short_slot_mapping_before_write():
    """Slot bounds are checked before an asynchronous paged write is queued."""
    memory_obj = _quantized_object(
        torch.float16,
        [index % 4 for index in range(11)],
    )
    cache = torch.zeros((2, 5, 4, 3, 5), dtype=torch.float16, device="cuda")
    ptrs = torch.tensor([cache.data_ptr(), cache.data_ptr()], device="cuda")
    with pytest.raises(ValueError, match="slot_mapping is shorter"):
        restore_makv_quantized_to_paged(
            memory_obj,
            device=torch.device("cuda"),
            page_ptrs=ptrs,
            slot_mapping=torch.arange(10, device="cuda"),
            page_buffer_size=20,
            block_size=4,
            head_size=5,
            engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        )
    assert torch.count_nonzero(cache).item() == 0


def test_direct_paged_cuda_events_report_h2d_and_kernel_time():
    """Timing must be based on completed CUDA events, not host launch time."""
    RESTORE_METRICS.reset()
    memory_obj = _quantized_object(
        torch.float16, [index % 3 for index in range(11)]
    )
    layers, tokens, heads, head_dim = 2, 11, 3, 5
    block_size = 4
    cache = [
        torch.zeros(
            (2, 5, block_size, heads, head_dim),
            device="cuda",
            dtype=torch.float16,
        )
        for _ in range(layers)
    ]
    pointers = torch.tensor(
        [tensor.data_ptr() for tensor in cache], dtype=torch.int64, device="cuda"
    )
    restore_makv_quantized_to_paged(
        memory_obj,
        device=torch.device("cuda"),
        page_ptrs=pointers,
        slot_mapping=torch.arange(tokens, device="cuda", dtype=torch.int64),
        page_buffer_size=20,
        block_size=block_size,
        head_size=head_dim,
        engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
    )
    torch.cuda.synchronize()
    timing = RESTORE_METRICS.snapshot()
    assert timing.makv_restore_calls == 1
    assert timing.makv_h2d_bytes > 0
    assert timing.makv_kernel_launch_count == 3
    assert timing.makv_cuda_pending_traces == 0
    assert timing.makv_restore_gpu_total_time_ms >= 0.0
    assert timing.makv_restore_gpu_total_time_ms == pytest.approx(
        timing.makv_h2d_time_ms + timing.makv_dequant_kernel_time_ms,
        abs=0.1,
    )


def test_restore_ready_event_handoff_orders_current_stream_without_host_sync():
    """A consumer-stream event wait makes paged KV visible without sync()."""
    RESTORE_METRICS.reset()
    memory_obj = _quantized_object(
        torch.float16, [index % 4 for index in range(11)]
    )
    layers, tokens, heads, head_dim = 2, 11, 3, 5
    cache = [
        torch.zeros((2, 5, 4, heads, head_dim), device="cuda", dtype=torch.float16)
        for _ in range(layers)
    ]
    ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in cache], dtype=torch.int64, device="cuda"
    )
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    load_stream = torch.cuda.Stream()
    h2d_stream = torch.cuda.Stream()
    consumer_stream = torch.cuda.Stream()
    marker = torch.zeros((), device="cuda", dtype=torch.int32)
    scope = begin_makv_restore_timing_scope()
    with torch.cuda.stream(load_stream):
        restore_makv_quantized_to_paged(
            memory_obj,
            device=torch.device("cuda"),
            page_ptrs=ptrs,
            slot_mapping=slots,
            page_buffer_size=20,
            block_size=4,
            head_size=head_dim,
            engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            h2d_stream=h2d_stream,
            timing_scope=scope,
        )
    with torch.cuda.stream(consumer_stream):
        handoff_makv_restore_ready_event(load_stream, scope)
        marker.add_(1)
    consumer_stream.synchronize()
    timing = finish_makv_restore_timing_scope(scope)

    assert marker.item() == 1
    assert torch.count_nonzero(cache[0]).item() > 0
    assert timing["makv_restore_ready_event_handoffs"] == 1
    assert timing["makv_cuda_pending_traces"] == 0


def test_vllm_v2_connector_uses_direct_paged_restore(monkeypatch):
    layers, tokens, heads, head_dim = 2, 11, 3, 5
    memory_obj = _quantized_object(
        torch.float16, [index % 4 for index in range(tokens)]
    )
    metadata = LMCacheMetadata(
        model_name="makv-vllm-hook",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(layers, 2, tokens, heads, head_dim),
        chunk_size=tokens,
    )
    caches = [
        torch.zeros((2, 5, 4, heads, head_dim), device="cuda", dtype=torch.float16)
        for _ in range(layers)
    ]
    slots = torch.randperm(20, device="cuda")[:tokens].to(torch.int64)
    connector = VLLMPagedMemGPUConnectorV2.from_metadata(
        metadata, use_gpu=False, device=torch.device("cuda")
    )
    real_restore = gpu_connectors.restore_makv_quantized_to_paged
    calls = 0

    def tracked_restore(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_restore(*args, **kwargs)

    monkeypatch.setattr(
        gpu_connectors, "restore_makv_quantized_to_paged", tracked_restore
    )
    connector.to_gpu(memory_obj, 0, tokens, kvcaches=caches, slot_mapping=slots)
    torch.cuda.synchronize()
    assert calls == 1
    assert len(connector._makv_h2d_streams) == 1
    assert next(iter(connector._makv_h2d_streams.values())) is not connector.load_stream
    assert memory_obj.tensor is None
    bad_connector = VLLMPagedMemGPUConnectorV2.from_metadata(
        metadata, use_gpu=False, device=torch.device("cuda")
    )
    bad_caches = [
        torch.zeros((2, 5, 4, heads, head_dim), device="cuda", dtype=torch.float32)
        for _ in range(layers)
    ]
    with pytest.raises(ValueError, match="dtype does not match"):
        bad_connector.to_gpu(
            memory_obj, 0, tokens, kvcaches=bad_caches, slot_mapping=slots
        )


def test_vllm_v2_connector_defers_stream_sync_for_makv_pipeline():
    """Deferred chunk submissions match the normal paged restore result."""
    layers, tokens, heads, head_dim = 2, 11, 3, 5
    memory_obj = _quantized_object(
        torch.float16, [index % 4 for index in range(tokens)]
    )
    metadata = LMCacheMetadata(
        model_name="makv-vllm-deferred-hook",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(layers, 2, tokens, heads, head_dim),
        chunk_size=tokens,
    )
    expected_caches = [
        torch.zeros((2, 5, 4, heads, head_dim), device="cuda", dtype=torch.float16)
        for _ in range(layers)
    ]
    actual_caches = [torch.zeros_like(cache) for cache in expected_caches]
    slots = torch.randperm(20, device="cuda")[:tokens].to(torch.int64)

    expected_connector = VLLMPagedMemGPUConnectorV2.from_metadata(
        metadata, use_gpu=False, device=torch.device("cuda")
    )
    expected_connector.batched_to_gpu(
        [memory_obj], [0], [tokens], kvcaches=expected_caches, slot_mapping=slots
    )

    streamed_connector = VLLMPagedMemGPUConnectorV2.from_metadata(
        metadata, use_gpu=False, device=torch.device("cuda")
    )
    scope = begin_makv_restore_timing_scope()
    assert (
        streamed_connector.batched_to_gpu(
            [memory_obj],
            [0],
            [tokens],
            kvcaches=actual_caches,
            slot_mapping=slots,
            makv_timing_scope=scope,
            makv_defer_synchronize=True,
        )
        is None
    )
    streamed_connector.load_stream.synchronize()
    timing = finish_makv_restore_timing_scope(scope)
    assert timing["makv_restore_calls"] == 1
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(expected_caches, actual_caches, strict=True)
    )


def test_vllm_v2_deferred_restore_keeps_multiple_inputs_alive():
    """Queued H2D/kernel work remains valid after host objects are released."""
    layers, tokens, heads, head_dim = 2, 11, 3, 5
    metadata = LMCacheMetadata(
        model_name="makv-vllm-deferred-lifetime",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(layers, 2, tokens, heads, head_dim),
        chunk_size=tokens,
    )
    objects = [
        _quantized_object(torch.float16, [index % 4 for index in range(tokens)]),
        _quantized_object(torch.float16, [3 - (index % 4) for index in range(tokens)]),
    ]
    slots = torch.randperm(32, device="cuda")[: 2 * tokens].to(torch.int64)
    expected = [
        torch.zeros((2, 8, 4, heads, head_dim), device="cuda", dtype=torch.float16)
        for _ in range(layers)
    ]
    actual = [torch.zeros_like(cache) for cache in expected]

    expected_connector = VLLMPagedMemGPUConnectorV2.from_metadata(
        metadata, use_gpu=False, device=torch.device("cuda")
    )
    expected_connector.batched_to_gpu(
        objects,
        [0, tokens],
        [tokens, 2 * tokens],
        kvcaches=expected,
        slot_mapping=slots,
    )

    streamed_connector = VLLMPagedMemGPUConnectorV2.from_metadata(
        metadata, use_gpu=False, device=torch.device("cuda")
    )
    scope = begin_makv_restore_timing_scope()
    streamed_connector.batched_to_gpu(
        objects,
        [0, tokens],
        [tokens, 2 * tokens],
        kvcaches=actual,
        slot_mapping=slots,
        makv_timing_scope=scope,
        makv_defer_synchronize=True,
    )
    objects.clear()
    gc.collect()
    streamed_connector.load_stream.synchronize()
    timing = finish_makv_restore_timing_scope(scope)

    assert timing["makv_restore_calls"] == 2
    assert timing["makv_cuda_pending_traces"] == 0
    assert all(
        torch.equal(left, right) for left, right in zip(expected, actual, strict=True)
    )
