# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional MaKV arithmetic layer."""

from typing import Any
import asyncio
from copy import deepcopy
import importlib
import os
import socket
import subprocess
import sys
import time

import pytest
import torch

from lmcache.v1.storage_backend.makv.entropy import (
    decode_entropy_payloads,
    encode_entropy_payloads,
    _pack_low_bit,
)
from lmcache.v1.storage_backend.makv.config import (
    MaKVConfig,
    get_makv_config,
    validate_makv_runtime_config,
)
from lmcache.v1.storage_backend.makv.format import decode_makv_object
from lmcache.v1.storage_backend.makv.memory import MaKVQuantizedMemoryObj
from lmcache.v1.storage_backend.makv.plan import MaKVQuantPlan
from lmcache.v1.storage_backend.makv.gpu_restore import payload_tensors_from_obj
from lmcache.v1.storage_backend.makv_remote.manager import MaKVRemoteManager
from lmcache.v1.storage_backend.makv.quantizer import quantize_canonical_kv
from lmcache.v1.storage_backend.connector.makv_network_connector import (
    MaKVNetworkConnector,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.makv.serde import (
    MaKVDeserializer,
    MaKVSerializer,
)
from lmcache.v1.storage_backend.makv.metrics import CLIENT_METRICS
from tests.v1.utils import close_asyncio_loop, dumb_cache_engine_key, init_asyncio_loop


def _metadata(counts: dict[int, int], *, head_dim: int = 5) -> dict[str, Any]:
    return {
        "plan": {
            "bucket_bits": [16, 8, 4, 2],
            "importance_layout": "token",
            "num_layers": 1,
            "num_kv_heads": 2,
            "head_dim": head_dim,
        },
        "bucket_entries": [
            {"bits": bits, "count": counts.get(bits, 0), "layout": "token"}
            for bits in (16, 8, 4, 2)
        ],
    }


def _source_payloads(
    metadata: dict[str, Any]
) -> tuple[dict[str, bytes], dict[int, bytes]]:
    generator = torch.Generator().manual_seed(1729)
    payloads: dict[str, bytes] = {}
    expected: dict[int, bytes] = {}
    head_dim = int(metadata["plan"]["head_dim"])
    for entry in metadata["bucket_entries"]:
        bits = int(entry["bits"])
        vectors = int(entry["count"]) * 1 * 2 * 2
        if bits == 16:
            continue
        if bits == 8:
            q = torch.randint(
                -127, 128, (vectors, head_dim), generator=generator, dtype=torch.int8
            )
            raw = q.numpy().tobytes()
        else:
            q = torch.randint(
                -((1 << (bits - 1)) - 1),
                ((1 << (bits - 1)) - 1) + 1,
                (vectors, head_dim),
                generator=generator,
                dtype=torch.int8,
            )
            raw = _pack_low_bit(q, head_dim, bits).numpy().tobytes()
        payloads[f"payload_{bits}"] = raw
        expected[bits] = raw
    return payloads, expected


def _getter(payloads: dict[str, bytes], device: torch.device | str = "cpu"):
    def get_payload(name: str, dtype: torch.dtype) -> torch.Tensor:
        raw = payloads.get(name, b"")
        if not raw:
            return torch.empty((0,), dtype=dtype, device=device)
        tensor = torch.frombuffer(raw, dtype=dtype)
        return tensor.to(device=device) if str(device) != "cpu" else tensor

    return get_payload


def test_arithmetic_reference_round_trip_with_odd_head_dim_and_empty_bucket():
    metadata = _metadata({16: 1, 8: 40, 4: 7, 2: 0})
    source, expected = _source_payloads(metadata)
    encoded_metadata, encoded = encode_entropy_payloads(
        metadata, source, codec="cachegen_arithmetic", backend="reference"
    )

    assert encoded_metadata["entropy"]["codec"] == "cachegen_arithmetic_v1"
    assert encoded_metadata["entropy"]["backend"] == "reference"
    assert "payload_8" not in encoded
    assert "payload_4" not in encoded
    assert "payload_2" not in encoded
    assert encoded_metadata["entropy"]["buckets"]["8"]["planes"][0]["stream_count"] > 1

    decoded = decode_entropy_payloads(encoded_metadata, _getter(encoded))
    for bits, raw in expected.items():
        assert decoded[bits].numpy().tobytes() == raw
    assert decoded[2].numel() == 0


def test_arithmetic_codec_none_is_a_zero_cost_identity():
    metadata = _metadata({16: 1, 8: 1, 4: 1, 2: 1})
    payloads, _ = _source_payloads(metadata)
    returned_metadata, returned_payloads = encode_entropy_payloads(
        metadata, payloads, codec="none", backend="reference"
    )
    assert returned_metadata is metadata
    assert returned_payloads is payloads


def test_entropy_runtime_config_rejects_unsupported_combinations():
    base = {
        "makv_storage_url": "file:///tmp/makv-entropy-config-test",
        "makv_bucket_ratios": [0.2, 0.3, 0.5],
        "makv_bucket_bits": [16, 8, 4],
        "makv_require_cuda_dequant": False,
    }

    class _Config:
        remote_serde = "makv"

        def __init__(self, extra_config):
            self.extra_config = extra_config

    invalid_codec = dict(base, makv_entropy_codec="unknown")
    with pytest.raises(ValueError, match="makv_entropy_codec"):
        validate_makv_runtime_config(_Config(invalid_codec))

    invalid_cuda_requirement = dict(
        base,
        makv_entropy_codec="cachegen_arithmetic",
        makv_entropy_backend="reference",
        makv_entropy_require_cuda=True,
    )
    with pytest.raises(ValueError, match="incompatible"):
        validate_makv_runtime_config(_Config(invalid_cuda_requirement))

    valid = dict(
        base,
        makv_entropy_codec="cachegen_arithmetic",
        makv_entropy_backend="reference",
    )
    config = get_makv_config(_Config(valid))
    assert config.entropy_codec == "cachegen_arithmetic"
    assert config.entropy_backend == "reference"


def _manager_config() -> MaKVConfig:
    return MaKVConfig(
        storage_url="file:///tmp/makv-entropy-test",
        bucket_ratios=(0.34, 0.33, 0.33),
        bucket_bits=(16, 8, 4),
        importance_layout="token",
        quant_granularity="per_token_head",
        scale_dtype="float16",
        protect_prefix_tokens=0,
        protect_tail_tokens=0,
        dequant_backend="reference",
        require_cuda_dequant=False,
        fallback="miss",
        enable_checksum=True,
        entropy_codec="cachegen_arithmetic",
        entropy_backend="reference",
    )


def test_remote_manager_encodes_real_quantizer_payloads():
    config = _manager_config()
    layers, tokens, heads, head_dim = 2, 6, 2, 5
    canonical = torch.randn(
        (layers, 2, tokens, heads, head_dim), generator=torch.Generator().manual_seed(7)
    ).to(torch.float16)
    raw = canonical.permute(1, 0, 2, 3, 4).reshape(
        2, layers, tokens, heads * head_dim
    ).contiguous()
    plan = MaKVQuantPlan(
        protocol_version=1,
        importance_layout="token",
        token_count=tokens,
        chunk_start=0,
        chunk_length=tokens,
        bucket_bits=(16, 8, 4),
        bucket_ids=bytes((0, 1, 2, 0, 1, 2)),
        original_shape=tuple(raw.shape),
        original_strides=tuple(raw.stride()),
        original_dtype="torch.float16",
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
    plain_metadata, plain_payloads = quantize_canonical_kv(canonical, plan, config)

    class _Storage:
        async def close(self):
            return None

    manager = MaKVRemoteManager(config, _Storage())
    encoded, _, _, _ = manager._quantize(
        {"key": "entropy-key", "plan": plan.to_dict()}, raw.numpy().tobytes()
    )
    stored = decode_makv_object(encoded)
    assert stored.object_type == "quantized"
    assert stored.metadata["entropy"]["codec"] == "cachegen_arithmetic_v1"
    parsed = MaKVQuantizedMemoryObj(
        encoded,
        metadata_dict=stored.metadata,
        payloads=stored.payloads,
    )
    restored_payloads = payload_tensors_from_obj(parsed)
    assert restored_payloads[16]["payload"].numpy().tobytes() == plain_payloads[
        "payload_16"
    ]
    assert restored_payloads[8]["payload"].numpy().tobytes() == plain_payloads[
        "payload_8"
    ]
    assert restored_payloads[4]["payload"].numpy().tobytes() == plain_payloads[
        "payload_4"
    ]
    assert manager.quantize_calls == 1


def test_entropy_decoder_rejects_invalid_stream_descriptors():
    metadata = _metadata({16: 1, 8: 8, 4: 0, 2: 0})
    source, _ = _source_payloads(metadata)
    encoded_metadata, encoded = encode_entropy_payloads(
        metadata, source, codec="cachegen_arithmetic", backend="reference"
    )

    oversized = deepcopy(encoded_metadata)
    oversized["entropy"]["buckets"]["8"]["planes"][0][
        "symbol_count"
    ] += 256
    with pytest.raises(ValueError, match="stream descriptor"):
        decode_entropy_payloads(oversized, _getter(encoded))

    duplicate = deepcopy(encoded_metadata)
    plane = duplicate["entropy"]["buckets"]["8"]["planes"][0]
    duplicate["entropy"]["buckets"]["8"]["planes"].append(dict(plane))
    with pytest.raises(ValueError, match="duplicate planes"):
        decode_entropy_payloads(duplicate, _getter(encoded))


def test_independent_manager_entropy_network_path(tmp_path):
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
            "--storage-backend",
            "file",
            "--storage-url",
            f"file://{tmp_path}",
            "--bucket-ratios",
            "0.25,0.25,0.5",
            "--bucket-bits",
            "16,8,4",
            "--entropy-codec",
            "cachegen_arithmetic",
            "--entropy-backend",
            "reference",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError(
                        "MaKV entropy manager exited during startup"
                    ) from None
                time.sleep(0.05)
        else:
            raise RuntimeError("MaKV entropy manager did not start")

        metadata = LMCacheMetadata(
            model_name="makv-entropy-network-test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.float16,
            kv_shape=(1, 2, 8, 1, 5),
            chunk_size=8,
        )
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=8,
            local_cpu=True,
            remote_url=f"makv://127.0.0.1:{port}",
            remote_serde="makv",
            extra_config={
                "makv_storage_url": f"file://{tmp_path}",
                "makv_bucket_ratios": [0.25, 0.25, 0.5],
                "makv_bucket_bits": [16, 8, 4],
                "makv_protect_prefix_tokens": 0,
                "makv_protect_tail_tokens": 0,
                "makv_require_cuda_dequant": False,
                "makv_entropy_codec": "cachegen_arithmetic",
                "makv_entropy_backend": "reference",
            },
        )
        shape = metadata.get_shapes()[0]
        tensor = torch.arange(shape.numel(), dtype=torch.float16).view(shape)
        memory_obj = TensorMemoryObj(
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
        serializer = MaKVSerializer(config, metadata)
        deserializer = MaKVDeserializer(config, metadata)
        key = dumb_cache_engine_key(173)
        envelope = serializer.serialize(
            memory_obj,
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
        CLIENT_METRICS.reset()
        loop, thread = init_asyncio_loop()
        try:
            connector = MaKVNetworkConnector(
                config.remote_url, loop, None, config, metadata
            )
            future = asyncio.run_coroutine_threadsafe(
                connector.put(key, envelope), loop
            )
            future.result()
            stored = asyncio.run_coroutine_threadsafe(
                connector.get(key), loop
            ).result()
            assert stored is not None
            stored_object = decode_makv_object(stored.byte_array)
            assert stored_object.metadata["entropy"]["codec"] == (
                "cachegen_arithmetic_v1"
            )
            assert "payload_8" not in stored_object.payloads
            restored = deserializer.deserialize(stored)
            assert isinstance(restored, MaKVQuantizedMemoryObj)
            health = asyncio.run_coroutine_threadsafe(
                connector.health(), loop
            ).result()
            assert health["pid"] == process.pid
            assert health["pid"] != os.getpid()
            assert health["quantize_calls"] > 0
            assert health["metrics"]["makv_remote_entropy_encode_calls"] == 1
            assert CLIENT_METRICS.snapshot().makv_client_quantize_calls == 0
        finally:
            close_asyncio_loop(loop, thread)
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
def test_cuda_cachegen_arithmetic_round_trip():
    c_ops = None
    for module_name in ("lmcache.cuda_ops", "lmcache.c_ops"):
        try:
            c_ops = importlib.import_module(module_name)
        except (ImportError, OSError):
            continue
        break
    if c_ops is None:
        pytest.skip("CacheGen CUDA extension is unavailable")
    if not all(
        hasattr(c_ops, name)
        for name in ("calculate_cdf", "encode_fast_new", "decode_fast_prefsum")
    ):
        pytest.skip("CacheGen arithmetic CUDA functions are not registered")
    metadata = _metadata({16: 1, 8: 8, 4: 5, 2: 3})
    source, expected = _source_payloads(metadata)
    encoded_metadata, encoded = encode_entropy_payloads(
        metadata,
        source,
        codec="cachegen_arithmetic",
        backend="cuda",
        require_cuda=True,
    )
    decoded = decode_entropy_payloads(
        encoded_metadata, _getter(encoded, torch.device("cuda"))
    )
    torch.cuda.synchronize()
    for bits, raw in expected.items():
        assert decoded[bits].cpu().numpy().tobytes() == raw
