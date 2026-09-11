# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import hashlib
import json
import socket
import subprocess
import sys
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.makv_network_connector import (
    MaKVNetworkConnector,
)
from lmcache.v1.storage_backend.makv.format import (
    PAYLOAD_ALIGNMENT,
    decode_client_put_envelope,
    decode_makv_object,
    encode_client_put_envelope,
    encode_makv_object,
)
from lmcache.v1.storage_backend.makv.gpu_restore import restore_makv_quantized_to_tensor
from lmcache.v1.storage_backend.makv.metrics import (
    CLIENT_METRICS,
    REMOTE_METRICS,
    RESTORE_METRICS,
)
from lmcache.v1.storage_backend.makv.plan import (
    build_chunk_quant_plan,
    build_chunk_quant_plan_from_precision_plan,
    prompt_token_hash,
)
from lmcache.v1.storage_backend.makv.serde import MaKVDeserializer, MaKVSerializer
from lmcache.v1.storage_backend.makv_remote.manager import MaKVRemoteManager
from lmcache.v1.storage_backend.makv_remote.server import MaKVRemoteServer
from lmcache.v1.storage_backend.makv_remote.protocol import (
    BATCH_BLOB_ENTRY,
    BATCH_BLOB_HEADER,
    BATCH_BLOB_VERSION,
    FRAME_HEADER,
    decode_batch_blob,
    decode_batch_blob_directory,
    encode_batch_blob,
    read_frame,
)
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from tests.v1.utils import (
    close_asyncio_loop,
    dumb_cache_engine_key,
    init_asyncio_loop,
)


@pytest.fixture(autouse=True)
def reset_makv_metrics():
    CLIENT_METRICS.reset()
    REMOTE_METRICS.reset()
    RESTORE_METRICS.reset()
    yield
    CLIENT_METRICS.reset()
    REMOTE_METRICS.reset()
    RESTORE_METRICS.reset()


@pytest.fixture
def small_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="makv-test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(2, 2, 8, 2, 4),
        chunk_size=8,
    )


def _make_config(
    tmp_path, remote_url="makv://127.0.0.1:65432", **extra
) -> LMCacheEngineConfig:
    extra_config = {
        "makv_storage_url": f"file://{tmp_path}",
        "makv_bucket_ratios": [0.25, 0.25, 0.5],
        "makv_bucket_bits": [16, 8, 4],
        "makv_protect_prefix_tokens": 0,
        "makv_protect_tail_tokens": 0,
        "makv_require_cuda_dequant": False,
        "makv_fallback": "naive",
    }
    extra_config.update(extra)
    return LMCacheEngineConfig.from_defaults(
        chunk_size=8,
        local_cpu=True,
        remote_url=remote_url,
        remote_serde="makv",
        extra_config=extra_config,
    )


@pytest.fixture
def makv_server(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lmcache.v1.storage_backend.makv_remote.server",
            "--listen",
            f"127.0.0.1:{port}",
            "--storage-url",
            f"file://{tmp_path}",
            "--queue-depth",
            "4",
            "--workers",
            "2",
            "--memory-cache-gb",
            "0.001",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError as error:
            if process.poll() is not None:
                raise RuntimeError("MaKV manager exited during startup") from error
            time.sleep(0.05)
    else:
        process.terminate()
        raise RuntimeError("MaKV manager did not start")
    try:
        yield f"makv://127.0.0.1:{port}", process
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.fixture
def makv_four_tier_server(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lmcache.v1.storage_backend.makv_remote.server",
            "--listen",
            f"127.0.0.1:{port}",
            "--storage-url",
            f"file://{tmp_path}",
            "--bucket-ratios",
            "0.1,0.1,0.6,0.2",
            "--bucket-bits",
            "16,8,4,2",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError as error:
            if process.poll() is not None:
                raise RuntimeError("four-tier MaKV manager exited") from error
            time.sleep(0.05)
    else:
        process.terminate()
        raise RuntimeError("four-tier MaKV manager did not start")
    try:
        yield f"makv://127.0.0.1:{port}", process
    finally:
        process.terminate()
        process.wait(timeout=10)


def _make_memory_obj(metadata: LMCacheMetadata) -> TensorMemoryObj:
    shape = metadata.get_shapes()[0]
    tensor = torch.arange(shape.numel(), dtype=metadata.kv_dtype).view(shape)
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


def _scoutrank_plan_payload(token_ids: list[int]) -> dict:
    precisions = ["BF16", "K8V4", *("K4V2" for _ in range(6)), "K2V2", "K2V2"]
    blocks = []
    for block_id, precision in enumerate(precisions):
        blocks.append(
            {
                "block_id": block_id,
                "eligible": True,
                "rank": block_id + 1,
                "token_start": block_id * 32,
                "token_end": (block_id + 1) * 32,
                "valid_tokens": 32,
                "stored_tokens": 32,
                "precision": precision,
                "estimated_bytes": 100 + block_id,
                "actual_bytes": 100 + block_id,
            }
        )
    counts = {name: precisions.count(name) for name in ("BF16", "K8V4", "K4V2", "K2V2")}
    total = sum(row["actual_bytes"] for row in blocks)
    operational = {
        "strategy_version": "k2_risk_monotone_four_tier_v1",
        "deployment_status": "shadow",
        "score_precision": "K2V2",
        "proxy_variant": "norm_upper_bound",
        "eligible_block_count": 10,
        "blocks": blocks,
        "estimated_total_bytes": total,
        "actual_total_bytes": total,
    }
    digest = hashlib.sha256(
        json.dumps(operational, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan = {
        **operational,
        "precision_counts": counts,
        "precision_vector": precisions,
        "plan_hash": digest,
    }
    return {
        "schema_version": "scoutrank_block_precision_v1",
        "status": "success",
        "strategy_version": plan["strategy_version"],
        "deployment_status": "shadow",
        "prompt_token_hash": prompt_token_hash(token_ids),
        "token_count": len(token_ids),
        "block_size": 32,
        "plan_hash": digest,
        "actual_bytes": total,
        "estimated_bytes": total,
        "precision_vector": precisions,
        "repeat_exact": True,
        "plan": plan,
    }


def test_makv_config_validation_rejects_bad_ratios(tmp_path):
    with pytest.raises(ValueError, match="sum to 1.0"):
        cfg = LMCacheEngineConfig.from_defaults(
            remote_serde="makv",
            remote_url="makv://127.0.0.1:65432",
            extra_config={
                "makv_storage_url": f"file://{tmp_path}",
                "makv_bucket_ratios": [0.1, 0.1, 0.1],
                "makv_bucket_bits": [16, 8, 4],
            },
        )
        cfg.validate()


def test_build_chunk_quant_plan_is_deterministic_and_protects_prefix_tail():
    metadata = LMCacheMetadata(
        model_name="m",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(2, 2, 5, 2, 4),
    )
    config = _make_config(
        "/tmp", makv_protect_prefix_tokens=1, makv_protect_tail_tokens=1
    )
    importance = [1.0, 1.0, 1.0, float("nan"), 0.0]
    plan = build_chunk_quant_plan(
        importance=importance,
        importance_layout_hint="token",
        chunk_start=0,
        chunk_end=5,
        original_shape=(2, 2, 5, 8),
        original_strides=(80, 40, 8, 1),
        original_dtype="torch.float16",
        token_dim=2,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        model_name=metadata.model_name,
        world_size=metadata.world_size,
        worker_id=metadata.worker_id,
        config=_make_config(
            "/tmp", makv_protect_prefix_tokens=1, makv_protect_tail_tokens=1
        ).extra_config
        and __import__(
            "lmcache.v1.storage_backend.makv.config", fromlist=["get_makv_config"]
        ).get_makv_config(config),
    )
    bucket_ids = list(plan.bucket_ids)
    assert bucket_ids[0] == 0
    assert bucket_ids[3] == 0
    assert bucket_ids[4] == 0
    assert bucket_ids[1] <= bucket_ids[2]


def test_importance_status_forces_uncovered_tokens_to_bf16(tmp_path):
    from lmcache.v1.storage_backend.makv.config import get_makv_config

    config = _make_config(
        tmp_path,
        makv_bucket_ratios=[0.1, 0.2, 0.5, 0.2],
        makv_bucket_bits=[16, 8, 4, 2],
        makv_precision_scheme="kv_separate_4tier",
    )
    status = [
        {"valid_mask": True, "forced_precision": None},
        {"valid_mask": False, "forced_precision": "BF16", "reason": "NO_FUTURE_PROBE"},
        {"valid_mask": True, "forced_precision": None},
        {"valid_mask": True, "forced_precision": None},
    ]
    plan = build_chunk_quant_plan(
        importance=[100.0, -100.0, 1.0, 0.0],
        importance_layout_hint="token",
        chunk_start=0,
        chunk_end=4,
        original_shape=(2, 2, 4, 4),
        original_strides=(32, 16, 4, 1),
        original_dtype="torch.float16",
        token_dim=2,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        model_name="m",
        world_size=1,
        worker_id=0,
        config=get_makv_config(config),
        importance_status=status,
    )
    # The physical layout is layer-major, K/V-major, token-major.  BF16 is
    # bucket zero for both planes, including the uncovered token's K and V.
    for layer in range(2):
        for kv in range(2):
            assert plan.bucket_ids[(layer * 2 + kv) * 4 + 1] == 0


def test_importance_status_rejects_invalid_non_bf16_force(tmp_path):
    from lmcache.v1.storage_backend.makv.config import get_makv_config

    with pytest.raises(ValueError, match="must force BF16"):
        build_chunk_quant_plan(
            importance=[0.0, 1.0],
            importance_layout_hint="token",
            chunk_start=0,
            chunk_end=2,
            original_shape=(1, 2, 2, 4),
            original_strides=(16, 8, 4, 1),
            original_dtype="torch.float16",
            token_dim=2,
            num_layers=1,
            num_kv_heads=1,
            head_dim=4,
            model_name="m",
            world_size=1,
            worker_id=0,
            config=get_makv_config(_make_config(tmp_path)),
            importance_status=[
                {"valid_mask": True, "forced_precision": None},
                {"valid_mask": False, "forced_precision": None},
            ],
        )


def test_build_chunk_quant_plan_layer_kv_layout(tmp_path):
    config = _make_config(tmp_path)
    from lmcache.v1.storage_backend.makv.config import get_makv_config

    importance = torch.arange(2 * 2 * 8, dtype=torch.float32).view(2, 2, 8)
    plan = build_chunk_quant_plan(
        importance=importance,
        importance_layout_hint="layer_kv_token",
        chunk_start=2,
        chunk_end=6,
        original_shape=(2, 2, 8, 8),
        original_strides=(128, 64, 8, 1),
        original_dtype="torch.float16",
        token_dim=2,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        model_name="m",
        world_size=1,
        worker_id=0,
        config=get_makv_config(config),
    )
    assert plan.importance_layout == "layer_kv_token"
    assert len(plan.bucket_ids) == 2 * 2 * 4


def test_scoutrank_block_plan_maps_asymmetric_kv_bits_and_slices_chunks(tmp_path):
    from lmcache.v1.storage_backend.makv.config import get_makv_config

    tokens = list(range(320))
    config = _make_config(
        tmp_path,
        makv_bucket_ratios=[0.1, 0.1, 0.6, 0.2],
        makv_bucket_bits=[16, 8, 4, 2],
        makv_allow_scoutrank_shadow_plan=True,
        # Client-side QDM configuration must not become a production
        # precision/request control field.
        makv_enable_qdm=True,
    )
    plan = build_chunk_quant_plan_from_precision_plan(
        precision_plan=_scoutrank_plan_payload(tokens),
        actual_prompt_token_hash=prompt_token_hash(tokens),
        chunk_start=16,
        chunk_end=80,
        original_shape=(2, 2, 64, 8),
        original_strides=(1024, 512, 8, 1),
        original_dtype="torch.float16",
        token_dim=2,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        model_name="m",
        world_size=1,
        worker_id=0,
        config=get_makv_config(config),
        request_token_count=320,
    )
    assert plan.importance_layout == "layer_kv_token"
    assert plan.bucket_bits == (16, 8, 4, 2)
    one_layer = list(plan.bucket_ids[:128])
    assert one_layer[:64] == [0] * 16 + [1] * 32 + [2] * 16
    assert one_layer[64:] == [0] * 16 + [2] * 32 + [3] * 16


def test_scoutrank_block_plan_rejects_hash_mismatch(tmp_path):
    from lmcache.v1.storage_backend.makv.config import get_makv_config

    tokens = list(range(320))
    config = _make_config(
        tmp_path,
        makv_bucket_ratios=[0.1, 0.1, 0.6, 0.2],
        makv_bucket_bits=[16, 8, 4, 2],
        makv_allow_scoutrank_shadow_plan=True,
    )
    with pytest.raises(ValueError, match="prompt token hash mismatch"):
        build_chunk_quant_plan_from_precision_plan(
            precision_plan=_scoutrank_plan_payload(tokens),
            actual_prompt_token_hash="bad",
            chunk_start=0,
            chunk_end=8,
            original_shape=(2, 2, 8, 8),
            original_strides=(128, 64, 8, 1),
            original_dtype="torch.float16",
            token_dim=2,
            num_layers=2,
            num_kv_heads=2,
            head_dim=4,
            model_name="m",
            world_size=1,
            worker_id=0,
            config=get_makv_config(config),
            request_token_count=320,
        )


def test_makv_format_roundtrip(tmp_path):
    from lmcache.v1.storage_backend.makv.config import get_makv_config

    config = _make_config(tmp_path)
    plan = build_chunk_quant_plan(
        importance=[0.9] * 8,
        importance_layout_hint="token",
        chunk_start=0,
        chunk_end=8,
        original_shape=(2, 2, 8, 8),
        original_strides=(128, 64, 8, 1),
        original_dtype="torch.float16",
        token_dim=2,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        model_name="m",
        world_size=1,
        worker_id=0,
        config=get_makv_config(config),
    )
    blob = encode_client_put_envelope(
        key="abc",
        object_type="raw_with_plan",
        plan=plan,
        raw_kv_payload=b"1234",
    )
    decoded = decode_client_put_envelope(blob)
    assert decoded.key == "abc"
    assert decoded.object_type == "raw_with_plan"
    assert decoded.raw_kv_payload == b"1234"

    object_blob = encode_makv_object(
        object_type="quantized",
        metadata={"cache_key": "abc"},
        payloads={"positions_16": b"123", "payload_16": b"4567"},
    )
    object_decoded = decode_makv_object(object_blob)
    assert all(
        int(entry["offset"]) % PAYLOAD_ALIGNMENT == 0
        for entry in object_decoded.metadata["_payload_table"]
    )


def test_serializer_sends_raw_kv_with_frozen_scoutrank_plan(tmp_path, small_metadata):
    tokens = list(range(320))
    config = _make_config(
        tmp_path,
        makv_bucket_ratios=[0.1, 0.1, 0.6, 0.2],
        makv_bucket_bits=[16, 8, 4, 2],
        makv_allow_scoutrank_shadow_plan=True,
    )
    memory_obj = _make_memory_obj(small_metadata)
    envelope_obj = MaKVSerializer(config, small_metadata).serialize(
        memory_obj,
        transfer_spec={
            "chunk_start": 0,
            "chunk_end": 8,
            "request_token_count": len(tokens),
            "prompt_token_hash": prompt_token_hash(tokens),
            "request_configs": {
                "lmcache.makv_precision_plan": _scoutrank_plan_payload(tokens)
            },
        },
        key=dumb_cache_engine_key(71),
    )
    envelope = decode_client_put_envelope(envelope_obj.byte_array)
    assert envelope.object_type == "raw_with_plan"
    assert envelope.raw_kv_payload == bytes(memory_obj.byte_array)
    assert envelope.metadata["plan"]["source_strategy"] == (
        "k2_risk_monotone_four_tier_v1"
    )
    assert envelope.metadata["plan"]["bucket_bits"] == [16, 8, 4, 2]
    assert envelope.metadata["plan"]["bucket_ids"] == [0] * 32
    assert "qdm_enable" not in envelope.metadata
    assert "qdm_block_size" not in envelope.metadata
    metrics = CLIENT_METRICS.snapshot()
    assert metrics.makv_client_quantize_calls == 0
    assert metrics.makv_client_plan_build_time_ms >= 0.0
    assert metrics.makv_client_raw_payload_copy_time_ms >= 0.0
    assert metrics.makv_client_envelope_encode_time_ms >= 0.0
    assert metrics.makv_client_serialize_total_time_ms >= 0.0


@pytest.mark.parametrize("bucket_ratios", ([1.0, 0.0, 0.0], [0.25, 0.25, 0.5]))
def test_makv_remote_roundtrip_and_restore(
    tmp_path, small_metadata, bucket_ratios, makv_server
):
    manager_url, manager_process = makv_server
    config = _make_config(
        tmp_path, remote_url=manager_url, makv_bucket_ratios=list(bucket_ratios)
    )
    serializer = MaKVSerializer(config, small_metadata)
    deserializer = MaKVDeserializer(config, small_metadata)
    async_loop, async_thread = init_asyncio_loop()
    try:
        connector = MaKVNetworkConnector(
            config.remote_url, async_loop, None, config, small_metadata
        )
        key = dumb_cache_engine_key(7)
        memory_obj = _make_memory_obj(small_metadata)
        transfer_spec = {
            "chunk_start": 0,
            "chunk_end": 8,
            "request_token_count": 8,
            "makv_importance": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            "makv_importance_layout": "token",
            "request_configs": {},
        }
        envelope = serializer.serialize(
            memory_obj, transfer_spec=transfer_spec, key=key
        )
        asyncio.run_coroutine_threadsafe(
            connector.put(key, envelope), async_loop
        ).result()
        stored = asyncio.run_coroutine_threadsafe(
            connector.get(key), async_loop
        ).result()
        assert stored is not None
        assert stored.makv_transport_timing["total_ms"] >= 0.0
        assert stored.makv_server_timing["total_ms"] >= 0.0
        assert stored.makv_server_timing["hot_cache_hit"]
        assert CLIENT_METRICS.snapshot().makv_client_get_batches == 1
        raw_bytes = len(memory_obj.byte_array)
        stored_object = decode_makv_object(stored.byte_array)
        assert "qdm" not in stored_object.metadata
        payload_bytes = sum(len(value) for value in stored_object.payloads.values())
        assert (
            payload_bytes <= raw_bytes
            if bucket_ratios != [1.0, 0.0, 0.0]
            else payload_bytes >= raw_bytes
        )
        quantized_obj = deserializer.deserialize(stored)
        if bucket_ratios == [1.0, 0.0, 0.0]:
            restored = restore_makv_quantized_to_tensor(
                quantized_obj, torch.device("cpu"), require_cuda=False
            )
            assert torch.equal(restored.tensor, memory_obj.tensor)
        else:
            restored = restore_makv_quantized_to_tensor(
                quantized_obj, torch.device("cpu"), require_cuda=False
            )
            assert restored.tensor.shape == memory_obj.tensor.shape
        client_snapshot = CLIENT_METRICS.snapshot()
        health = asyncio.run_coroutine_threadsafe(
            connector.health(), async_loop
        ).result()
        assert client_snapshot.makv_client_quantize_calls == 0
        assert health["pid"] == manager_process.pid
        assert health["pid"] != __import__("os").getpid()
        assert health["quantize_calls"] > 0
        assert health["metrics"]["makv_remote_put_requests"] == 1
        assert health["metrics"]["makv_remote_quantize_time_ms"] >= 0.0
        assert health["metrics"]["makv_remote_plan_canonicalize_time_ms"] >= 0.0
        assert health["metrics"]["makv_remote_quantize_kernel_time_ms"] >= 0.0
        assert health["metrics"]["makv_remote_object_encode_time_ms"] >= 0.0
        assert health["metrics"]["makv_remote_object_validate_time_ms"] >= 0.0
    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_makv_network_batched_get_uses_one_ordered_response_stream(
    tmp_path, small_metadata, makv_server
):
    manager_url, _ = makv_server
    config = _make_config(tmp_path, remote_url=manager_url)
    serializer = MaKVSerializer(config, small_metadata)
    loop, thread = init_asyncio_loop()
    try:
        connector = MaKVNetworkConnector(
            config.remote_url, loop, None, config, small_metadata
        )
        keys = [dumb_cache_engine_key(100 + index) for index in range(3)]
        envelopes = [
            serializer.serialize(
                _make_memory_obj(small_metadata),
                transfer_spec={
                    "chunk_start": 0,
                    "chunk_end": 8,
                    "request_token_count": 8,
                    "makv_importance": [float(8 - index) for index in range(8)],
                    "makv_importance_layout": "token",
                    "request_configs": {},
                },
                key=key,
            )
            for key in keys
        ]
        asyncio.run_coroutine_threadsafe(
            connector.batched_put(keys, envelopes), loop
        ).result()
        values = asyncio.run_coroutine_threadsafe(
            connector.batched_get(keys + [dumb_cache_engine_key(999)]), loop
        ).result()
        assert len(values) == 4
        assert all(value is not None for value in values[:3])
        assert values[3] is None
        assert all(
            decode_makv_object(value.byte_array).object_type == "quantized"
            for value in values[:3]
        )
        client_metrics = CLIENT_METRICS.snapshot()
        assert client_metrics.makv_client_get_batch_blob_frames == 1
        assert client_metrics.makv_client_get_batch_blob_bytes > 0
    finally:
        close_asyncio_loop(loop, thread)


def test_makv_streaming_backend_does_not_require_local_cpu():
    """Direct MaKV restore can stream when no CPU allocator is configured."""

    class Config:
        remote_serde = "makv"

    class StreamingConnection:
        @staticmethod
        def support_batched_get_streaming():
            return True

        @staticmethod
        async def batched_get_streaming(keys):
            del keys
            if False:
                yield 0, None

    backend = RemoteBackend.__new__(RemoteBackend)
    backend.local_cpu_backend = None
    backend.connection = StreamingConnection()
    backend.config = Config()
    backend._mla_worker_id_as0_mode = False

    stream = backend.batched_get_streaming_blocking([dumb_cache_engine_key(199)])
    assert stream is not None


def test_makv_network_batched_get_streams_objects_in_order(
    tmp_path, small_metadata, makv_server
):
    """The stream protocol yields each object in request order."""
    manager_url, _ = makv_server
    config = _make_config(tmp_path, remote_url=manager_url)
    serializer = MaKVSerializer(config, small_metadata)
    loop, thread = init_asyncio_loop()
    try:
        connector = MaKVNetworkConnector(
            config.remote_url, loop, None, config, small_metadata
        )
        keys = [dumb_cache_engine_key(200 + index) for index in range(3)]
        envelopes = [
            serializer.serialize(
                _make_memory_obj(small_metadata),
                transfer_spec={
                    "chunk_start": 0,
                    "chunk_end": 8,
                    "request_token_count": 8,
                    "makv_importance": [float(8 - index) for index in range(8)],
                    "makv_importance_layout": "token",
                    "request_configs": {},
                },
                key=key,
            )
            for key in keys
        ]
        asyncio.run_coroutine_threadsafe(
            connector.batched_put(keys, envelopes), loop
        ).result()

        async def collect_stream() -> list[tuple[int, object | None]]:
            result: list[tuple[int, object | None]] = []
            async for index, value in connector.batched_get_streaming(
                keys + [dumb_cache_engine_key(999)]
            ):
                result.append((index, value))
            return result

        values = asyncio.run_coroutine_threadsafe(collect_stream(), loop).result()
        assert [index for index, _ in values] == [0, 1, 2, 3]
        assert all(value is not None for _, value in values[:3])
        assert values[3][1] is None
        assert all(
            decode_makv_object(value.byte_array).object_type == "quantized"
            for _, value in values[:3]
        )
        client_metrics = CLIENT_METRICS.snapshot()
        assert client_metrics.makv_client_get_stream_requests == 1
        assert client_metrics.makv_client_get_stream_frames == 4
        assert client_metrics.makv_client_get_stream_bytes > 0
        assert client_metrics.makv_client_get_batch_blob_frames == 0
    finally:
        close_asyncio_loop(loop, thread)


def test_makv_stream_server_sends_first_object_before_later_get(
    tmp_path, small_metadata
):
    """Bounded prefetch does not wait for a slow later object."""

    class DelayedManager:
        def __init__(self, values):
            self.values = values
            self.active = 0
            self.max_active = 0

        async def get_with_timing(self, key):
            value, delay = self.values[key]
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(delay)
                return value, {"total_ms": delay * 1000}
            finally:
                self.active -= 1

    loop, thread = init_asyncio_loop()
    server = None
    service = None
    try:
        key_objects = [dumb_cache_engine_key(300 + index) for index in range(3)]
        values = {
            key.to_string(): (payload, delay)
            for key, payload, delay in zip(
                key_objects,
                (b"first-object", b"slow-second", b"third-object"),
                (0.0, 0.2, 0.0),
                strict=True,
            )
        }
        manager = DelayedManager(values)

        async def start_server():
            nonlocal service
            service = MaKVRemoteServer(
                manager,
                queue_depth=1,
                workers=1,
                max_request_bytes=1024,
                batch_stream_prefetch_depth=2,
            )
            return await asyncio.start_server(
                service.handle_client, "127.0.0.1", 0
            )

        server = asyncio.run_coroutine_threadsafe(start_server(), loop).result()
        port = int(server.sockets[0].getsockname()[1])
        config = _make_config(tmp_path, remote_url=f"makv://127.0.0.1:{port}")
        connector = MaKVNetworkConnector(
            config.remote_url, loop, None, config, small_metadata
        )

        async def collect_stream():
            started = time.perf_counter()
            result = []
            async for index, value in connector.batched_get_streaming(key_objects):
                result.append(
                    (
                        index,
                        time.perf_counter() - started,
                        None if value is None else bytes(value.byte_array),
                    )
                )
            return result

        values_received = asyncio.run_coroutine_threadsafe(
            collect_stream(), loop
        ).result()
        assert [item[2] for item in values_received] == [
            b"first-object",
            b"slow-second",
            b"third-object",
        ]
        assert values_received[0][1] < 0.1
        assert values_received[1][1] >= 0.16
        assert manager.max_active <= 2
    finally:
        async def stop_server():
            if server is not None:
                server.close()
                await server.wait_closed()
            if service is not None:
                await service.close()

        if server is not None:
            asyncio.run_coroutine_threadsafe(stop_server(), loop).result()
        close_asyncio_loop(loop, thread)


def test_makv_network_stream_yields_before_later_mkvb_payload(tmp_path, small_metadata):
    """A delayed second segment must not delay delivery of the first object."""
    loop, thread = init_asyncio_loop()
    delay_s = 0.2
    try:
        blob = encode_batch_blob([b"first-object", b"second-object"])
        directory_size = BATCH_BLOB_HEADER.size + 2 * BATCH_BLOB_ENTRY.size
        entries = decode_batch_blob_directory(
            memoryview(blob)[:directory_size],
            payload_length=len(blob),
            expected_count=2,
        )
        assert entries[0] is not None
        first_offset, first_length = entries[0]
        first_segment_end = first_offset + first_length

        async def handler(reader, writer) -> None:
            await read_frame(reader)
            response_header = json.dumps(
                {
                    "status": "ok",
                    "count": 2,
                    "batch_blob_version": BATCH_BLOB_VERSION,
                    "batch_timings": [],
                },
                separators=(",", ":"),
            ).encode("utf-8")
            writer.write(FRAME_HEADER.pack(len(response_header), len(blob)))
            writer.write(response_header)
            writer.write(blob[:first_segment_end])
            await writer.drain()
            await asyncio.sleep(delay_s)
            writer.write(blob[first_segment_end:])
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        async def start_server():
            server = await asyncio.start_server(handler, "127.0.0.1", 0)
            port = int(server.sockets[0].getsockname()[1])
            return server, port

        server, port = asyncio.run_coroutine_threadsafe(start_server(), loop).result()
        config = _make_config(tmp_path, remote_url=f"makv://127.0.0.1:{port}")
        connector = MaKVNetworkConnector(
            config.remote_url, loop, None, config, small_metadata
        )

        async def collect_stream() -> list[tuple[int, float, bytes | None]]:
            started = time.perf_counter()
            result: list[tuple[int, float, bytes | None]] = []
            async for index, value in connector.batched_get_streaming(
                [dumb_cache_engine_key(1), dumb_cache_engine_key(2)]
            ):
                result.append(
                    (
                        index,
                        time.perf_counter() - started,
                        None if value is None else bytes(value.byte_array),
                    )
                )
            return result

        values = asyncio.run_coroutine_threadsafe(collect_stream(), loop).result()
        assert [value[2] for value in values] == [b"first-object", b"second-object"]
        assert values[0][1] < delay_s / 2
        assert values[1][1] >= delay_s * 0.8
        async def stop_server() -> None:
            server.close()
            await server.wait_closed()

        asyncio.run_coroutine_threadsafe(stop_server(), loop).result()
    finally:
        close_asyncio_loop(loop, thread)


def test_makv_batch_blob_is_zero_copy_and_rejects_bad_directory():
    blob = encode_batch_blob([b"first", None, b"third"])
    decoded = decode_batch_blob(blob, expected_count=3)
    assert [None if value is None else bytes(value) for value in decoded] == [
        b"first",
        None,
        b"third",
    ]
    assert decoded[0] is not None
    assert decoded[0].obj is blob
    directory_length = BATCH_BLOB_HEADER.size + 3 * BATCH_BLOB_ENTRY.size
    directory = memoryview(blob)[:directory_length]
    assert decode_batch_blob_directory(
        directory,
        payload_length=len(blob),
        expected_count=3,
    ) == [(128, 5), None, (192, 5)]

    with pytest.raises(ValueError, match="count mismatch"):
        decode_batch_blob(blob, expected_count=2)
    with pytest.raises(ValueError, match="length mismatch"):
        decode_batch_blob(blob[:-1], expected_count=3)

    bad_offset = bytearray(blob)
    BATCH_BLOB_ENTRY.pack_into(
        bad_offset,
        BATCH_BLOB_HEADER.size,
        1,
        len(bad_offset) + 1,
        1,
    )
    with pytest.raises(ValueError, match="out of bounds"):
        decode_batch_blob(bad_offset, expected_count=3)

    bad_missing = bytearray(blob)
    BATCH_BLOB_ENTRY.pack_into(
        bad_missing,
        BATCH_BLOB_HEADER.size + BATCH_BLOB_ENTRY.size,
        0,
        BATCH_BLOB_HEADER.size,
        1,
    )
    with pytest.raises(ValueError, match="missing"):
        decode_batch_blob(bad_missing, expected_count=3)

    bad_overlap = bytearray(blob)
    first_entry_offset = BATCH_BLOB_HEADER.size
    third_entry_offset = BATCH_BLOB_HEADER.size + 2 * BATCH_BLOB_ENTRY.size
    _, first_payload_offset, first_payload_length = BATCH_BLOB_ENTRY.unpack_from(
        bad_overlap, first_entry_offset
    )
    BATCH_BLOB_ENTRY.pack_into(
        bad_overlap,
        third_entry_offset,
        1,
        first_payload_offset,
        first_payload_length,
    )
    with pytest.raises(ValueError, match="overlap"):
        decode_batch_blob(bad_overlap, expected_count=3)


def test_makv_manager_batch_get_uses_adapter_batch_and_preserves_validation(
    tmp_path,
):
    from lmcache.v1.storage_backend.makv.config import get_makv_config

    class BatchStorage:
        def __init__(self, values: dict[str, bytes]) -> None:
            self.values = values
            self.get_calls = 0
            self.get_many_calls = 0

        async def get(self, key: str) -> bytes | None:
            self.get_calls += 1
            return self.values.get(key)

        async def get_many(self, keys: list[str]) -> list[bytes | None]:
            self.get_many_calls += 1
            return [self.values.get(key) for key in keys]

        async def close(self) -> None:
            pass

    keys = ["batch-a", "batch-b", "batch-missing"]
    values = {
        key: encode_makv_object(
            object_type="naive_fallback",
            metadata={"cache_key": key},
            payloads={"raw_kv_payload": key.encode()},
        )
        for key in keys[:2]
    }
    storage = BatchStorage(values)
    manager = MaKVRemoteManager(get_makv_config(_make_config(tmp_path)), storage)

    async def run() -> list[tuple[bytes | None, dict]]:
        return await manager.get_many_with_timing(keys)

    results = asyncio.run(run())
    assert [value for value, _ in results] == [values[keys[0]], values[keys[1]], None]
    assert storage.get_many_calls == 1
    assert storage.get_calls == 0
    assert all("batch_total_ms" in timing for _, timing in results)
    assert results[0][1]["batch_storage_ms"] >= 0.0
    assert results[0][1]["batch_validate_ms"] >= 0.0


def test_makv_manager_trusted_object_path_skips_only_repeat_crc(tmp_path):
    from lmcache.v1.storage_backend.makv.config import get_makv_config

    key = "trusted-object"
    object_bytes = encode_makv_object(
        object_type="naive_fallback",
        metadata={"cache_key": key},
        payloads={"raw_kv_payload": b"payload"},
    )

    class Storage:
        async def get(self, requested_key: str) -> bytes | None:
            return object_bytes if requested_key == key else None

        async def close(self) -> None:
            pass

    manager = MaKVRemoteManager(
        get_makv_config(_make_config(tmp_path)),
        Storage(),
        trust_validated_objects=True,
    )

    async def run() -> None:
        first = await manager.get_many_with_timing([key])
        second = await manager.get_many_with_timing([key])
        assert first[0][0] == object_bytes
        assert second[0][0] == object_bytes

    asyncio.run(run())
    metrics = REMOTE_METRICS.snapshot()
    assert metrics.makv_remote_get_checksum_verifications == 1
    assert metrics.makv_remote_get_checksum_skips == 1


def test_makv_network_receive_buffer_keeps_cpu_fallback_usable():
    connector = object.__new__(MaKVNetworkConnector)

    async def run() -> bytes:
        connector.loop = asyncio.get_running_loop()
        left, right = socket.socketpair()
        left.setblocking(False)
        right.setblocking(False)
        try:
            payload = b"makv-receive-buffer"
            await asyncio.get_running_loop().sock_sendall(right, payload)
            received = await connector._recv_into(
                left, len(payload), pinned=True
            )
            return bytes(received)
        finally:
            left.close()
            right.close()

    assert asyncio.run(run()) == b"makv-receive-buffer"


def test_scoutrank_four_tier_plan_quantizes_only_in_remote_process(
    tmp_path, makv_four_tier_server
):
    manager_url, manager_process = makv_four_tier_server
    metadata = LMCacheMetadata(
        model_name="makv-scoutrank-e2e",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(1, 2, 320, 1, 64),
        chunk_size=320,
    )
    config = _make_config(
        tmp_path,
        remote_url=manager_url,
        makv_bucket_ratios=[0.1, 0.1, 0.6, 0.2],
        makv_bucket_bits=[16, 8, 4, 2],
        makv_allow_scoutrank_shadow_plan=True,
    )
    tokens = list(range(320))
    memory_obj = _make_memory_obj(metadata)
    key = dumb_cache_engine_key(72)
    loop, thread = init_asyncio_loop()
    try:
        connector = MaKVNetworkConnector(
            config.remote_url, loop, None, config, metadata
        )
        envelope = MaKVSerializer(config, metadata).serialize(
            memory_obj,
            transfer_spec={
                "chunk_start": 0,
                "chunk_end": 320,
                "request_token_count": 320,
                "prompt_token_hash": prompt_token_hash(tokens),
                "request_configs": {
                    "lmcache.makv_precision_plan": _scoutrank_plan_payload(tokens)
                },
            },
            key=key,
        )
        asyncio.run_coroutine_threadsafe(connector.put(key, envelope), loop).result()
        stored = asyncio.run_coroutine_threadsafe(connector.get(key), loop).result()
        assert stored is not None
        decoded = decode_makv_object(stored.byte_array)
        assert decoded.object_type == "quantized"
        assert decoded.payloads["payload_2"]
        assert len(stored.byte_array) < len(memory_obj.byte_array)
        health = asyncio.run_coroutine_threadsafe(connector.health(), loop).result()
        assert health["pid"] == manager_process.pid
        assert health["quantize_calls"] > 0
        assert CLIENT_METRICS.snapshot().makv_client_quantize_calls == 0
    finally:
        close_asyncio_loop(loop, thread)


def test_makv_missing_importance_falls_back_to_naive(
    tmp_path, small_metadata, makv_server
):
    manager_url, manager_process = makv_server
    config = _make_config(tmp_path, remote_url=manager_url)
    serializer = MaKVSerializer(config, small_metadata)
    deserializer = MaKVDeserializer(config, small_metadata)
    async_loop, async_thread = init_asyncio_loop()
    try:
        connector = MaKVNetworkConnector(
            config.remote_url, async_loop, None, config, small_metadata
        )
        key = dumb_cache_engine_key(9)
        memory_obj = _make_memory_obj(small_metadata)
        envelope = serializer.serialize(
            memory_obj,
            transfer_spec={"chunk_start": 0, "chunk_end": 8, "request_configs": {}},
            key=key,
        )
        asyncio.run_coroutine_threadsafe(
            connector.put(key, envelope), async_loop
        ).result()
        stored = asyncio.run_coroutine_threadsafe(
            connector.get(key), async_loop
        ).result()
        restored = deserializer.deserialize(stored)
        assert isinstance(restored, TensorMemoryObj)
        assert torch.equal(restored.tensor, memory_obj.tensor)
        health = asyncio.run_coroutine_threadsafe(
            connector.health(), async_loop
        ).result()
        assert health["metrics"]["makv_naive_fallbacks"] == 1
    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_makv_manager_rejects_corrupt_put(tmp_path, small_metadata, makv_server):
    manager_url, _ = makv_server
    config = _make_config(tmp_path, remote_url=manager_url)
    serializer = MaKVSerializer(config, small_metadata)
    loop, thread = init_asyncio_loop()
    try:
        connector = MaKVNetworkConnector(
            config.remote_url, loop, None, config, small_metadata
        )
        key = dumb_cache_engine_key(91)
        envelope = serializer.serialize(
            _make_memory_obj(small_metadata),
            transfer_spec={
                "chunk_start": 0,
                "chunk_end": 8,
                "makv_importance": list(range(8)),
                "request_configs": {},
            },
            key=key,
        )
        corrupt = bytearray(envelope.byte_array)
        corrupt[-1] ^= 0xFF
        from lmcache.v1.memory_management import BytesBufferMemoryObj

        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, BytesBufferMemoryObj(bytes(corrupt))), loop
        )
        with pytest.raises(RuntimeError, match="checksum"):
            future.result()
        assert not connector.exists_sync(key)
    finally:
        close_asyncio_loop(loop, thread)


def test_makv_manager_queue_backpressure():
    class SlowManager:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def put(self, key, payload, queue_ms):
            self.started.set()
            await self.release.wait()
            return len(payload)

    async def run():
        manager = SlowManager()
        service = MaKVRemoteServer(
            manager,
            queue_depth=1,
            workers=1,
            max_request_bytes=1024,
            queue_wait_timeout=1.0,
        )
        await service.start_workers()
        first = asyncio.create_task(service._dispatch("PUT", "a", b"a"))
        await manager.started.wait()
        second = asyncio.create_task(service._dispatch("PUT", "b", b"b"))
        await asyncio.sleep(0)
        third = asyncio.create_task(service._dispatch("PUT", "c", b"c"))
        await asyncio.sleep(0)
        assert not third.done()
        manager.release.set()
        await asyncio.gather(first, second, third)
        await service.close()

    asyncio.run(run())


def test_bytes_buffer_metadata_has_matching_shape_and_dtype_groups():
    from lmcache.v1.memory_management import BytesBufferMemoryObj

    memory_obj = BytesBufferMemoryObj(b"cachegen")
    encoded = RemoteMetadata(
        memory_obj.get_size(),
        memory_obj.get_shapes(),
        memory_obj.get_dtypes(),
        memory_obj.get_memory_format(),
    ).serialize()
    decoded = RemoteMetadata.deserialize(encoded)
    assert decoded.length == len(b"cachegen")
    assert decoded.shapes == [torch.Size([8, 0, 0, 0])]
    assert decoded.dtypes == [None]


def test_batched_put_uses_only_keys_with_serialized_objects(small_metadata):
    class PassthroughSerializer:
        def serialize(self, memory_obj):
            return memory_obj

    class RecordingConnection:
        def __init__(self):
            self.done = threading.Event()
            self.keys = None
            self.memory_objs = None

        def support_batched_put(self):
            return True

        async def batched_put(self, keys, memory_objs):
            self.keys = list(keys)
            self.memory_objs = list(memory_objs)
            self.done.set()

    backend = object.__new__(RemoteBackend)
    backend.connection = RecordingConnection()
    backend.serializer = PassthroughSerializer()
    backend._mla_worker_id_as0_mode = False
    backend.lock = threading.Lock()
    backend.put_tasks = set()
    backend.loop, loop_thread = init_asyncio_loop()
    keys = [dumb_cache_engine_key(index) for index in range(3)]
    memory_objs = [_make_memory_obj(small_metadata) for _ in range(2)]
    try:
        backend.batched_submit_put_task(keys, memory_objs)
        assert backend.connection.done.wait(timeout=2)
        assert backend.connection.keys == keys[:2]
        assert backend.connection.memory_objs == memory_objs
    finally:
        close_asyncio_loop(backend.loop, loop_thread)


def test_request_global_plan_slices_chunk_tensor(tmp_path):
    from lmcache.v1.storage_backend.makv.config import get_makv_config

    config = get_makv_config(_make_config(tmp_path))
    importance = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    common = {
        "importance": importance,
        "importance_layout_hint": "token",
        "chunk_start": 4,
        "chunk_end": 8,
        "original_dtype": "torch.float16",
        "token_dim": 2,
        "num_layers": 2,
        "num_kv_heads": 2,
        "head_dim": 4,
        "model_name": "m",
        "world_size": 1,
        "worker_id": 0,
        "config": config,
    }
    production = build_chunk_quant_plan(
        **common,
        original_shape=(2, 2, 4, 8),
        original_strides=(64, 32, 8, 1),
        request_token_count=8,
    )
    reference = build_chunk_quant_plan(
        **common,
        original_shape=(2, 2, 8, 8),
        original_strides=(128, 64, 8, 1),
    )
    assert production.token_count == 8
    assert production.chunk_length == 4
    assert production.bucket_ids == reference.bucket_ids


def test_makv_streaming_capability_survives_instrumented_connector():
    """The generic connector wrapper must preserve optional MaKV streaming."""
    from lmcache.v1.storage_backend.connector.instrumented_connector import (
        InstrumentedRemoteConnector,
    )

    class FakeConnector:
        def support_batched_get_streaming(self):
            return True

        async def batched_get_streaming(self, keys):
            for index, key in enumerate(keys):
                yield index, key

    async def collect():
        wrapped = InstrumentedRemoteConnector(FakeConnector())
        assert wrapped.support_batched_get_streaming()
        return [item async for item in wrapped.batched_get_streaming(["a", "b"])]

    assert asyncio.run(collect()) == [(0, "a"), (1, "b")]


def test_makv_precision_risk_capability_survives_instrumented_connector():
    """The generic connector wrapper must preserve optional MaKV risk calls."""
    from lmcache.v1.storage_backend.connector.instrumented_connector import (
        InstrumentedRemoteConnector,
    )

    class FakeConnector:
        async def report_precision_risk(self, key, signal):
            return {"accepted": True, "key": key, "signal": signal}

    async def report():
        wrapped = InstrumentedRemoteConnector(FakeConnector())
        return await wrapped.report_precision_risk("key", {"risk": 1.0})

    assert asyncio.run(report()) == {
        "accepted": True,
        "key": "key",
        "signal": {"risk": 1.0},
    }
