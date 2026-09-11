# SPDX-License-Identifier: Apache-2.0

"""MaKV serializer and deserializer."""

# Standard
from typing import Any, Optional
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.makv.config import (
    extract_makv_importance,
    extract_makv_importance_status,
    extract_makv_precision_plan,
    get_makv_config,
)
from lmcache.v1.storage_backend.makv.format import (
    decode_makv_object,
    encode_client_put_envelope,
)
from lmcache.v1.storage_backend.makv.memory import MaKVQuantizedMemoryObj
from lmcache.v1.storage_backend.makv.metrics import CLIENT_METRICS
from lmcache.v1.storage_backend.makv.plan import (
    build_chunk_quant_plan,
    build_chunk_quant_plan_from_precision_plan,
    model_fingerprint,
    parallel_fingerprint,
)
from lmcache.v1.storage_backend.naive_serde.naive_serde import (
    NaiveDeserializer,
    NaiveSerializer,
)
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer, Serializer

logger = init_logger(__name__)


class MaKVSerializer(Serializer):
    """Build a request-scoped quant plan and ship raw KV to the remote manager."""

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheMetadata):
        self.config = config
        self.metadata = metadata
        self.makv_config = get_makv_config(config)
        self.naive = NaiveSerializer()

    @staticmethod
    def _record_put_timing(
        *,
        started: float,
        plan_build_ms: float,
        raw_payload_copy_ms: float,
        envelope_encode_ms: float,
        raw_bytes: int,
        envelope_bytes: int,
    ) -> None:
        """Record client work without attributing memcpy to plan generation."""
        CLIENT_METRICS.add(
            makv_plan_time_ms=plan_build_ms,
            makv_client_plan_build_time_ms=plan_build_ms,
            makv_client_raw_payload_copy_time_ms=raw_payload_copy_ms,
            makv_client_envelope_encode_time_ms=envelope_encode_ms,
            makv_client_serialize_total_time_ms=(time.perf_counter() - started)
            * 1000,
            makv_put_raw_bytes=raw_bytes,
            makv_put_plan_bytes=envelope_bytes - raw_bytes,
        )

    def serialize(
        self,
        memory_obj: MemoryObj,
        transfer_spec: Optional[dict[str, Any]] = None,
        key: Any = None,
    ) -> MemoryObj:
        t0 = time.perf_counter()
        if memory_obj.tensor is None:
            raise ValueError("MaKVSerializer requires a tensor-backed MemoryObj")
        request_configs = None
        if transfer_spec is not None:
            request_configs = transfer_spec.get("request_configs")
        precision_plan = extract_makv_precision_plan(transfer_spec, request_configs)
        importance, importance_layout = extract_makv_importance(
            transfer_spec,
            request_configs,
        )
        importance_status = extract_makv_importance_status(
            transfer_spec,
            request_configs,
        )
        if precision_plan is None and importance is None:
            logger.warning(
                "MaKV PUT is missing importance; request_config_keys=%s "
                "transfer_spec_keys=%s",
                sorted((request_configs or {}).keys()),
                sorted((transfer_spec or {}).keys()),
            )
            if self.makv_config.fallback == "naive":
                raw_started = time.perf_counter()
                raw = bytes(memory_obj.byte_array)
                raw_payload_copy_ms = (time.perf_counter() - raw_started) * 1000
                envelope_started = time.perf_counter()
                envelope = encode_client_put_envelope(
                    key=key.to_string() if key is not None else "",
                    object_type="naive_fallback",
                    plan=None,
                    raw_kv_payload=raw,
                    extra_metadata={
                        "fallback_reason": "missing_importance",
                    },
                )
                self._record_put_timing(
                    started=t0,
                    plan_build_ms=0.0,
                    raw_payload_copy_ms=raw_payload_copy_ms,
                    envelope_encode_ms=(time.perf_counter() - envelope_started) * 1000,
                    raw_bytes=len(raw),
                    envelope_bytes=len(envelope),
                )
                return BytesBufferMemoryObj(envelope)
            raise ValueError("MaKV importance is missing and makv_fallback=miss")

        if transfer_spec is None:
            raise ValueError("MaKVSerializer requires transfer_spec with chunk offsets")
        chunk_start = int(transfer_spec["chunk_start"])
        chunk_end = int(transfer_spec["chunk_end"])
        tensor = memory_obj.tensor
        assert tensor is not None
        common = {
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "original_shape": tuple(int(v) for v in tensor.shape),
            "original_strides": tuple(int(v) for v in tensor.stride()),
            "original_dtype": str(tensor.dtype),
            "token_dim": int(memory_obj.meta.fmt.token_dim()),
            "num_layers": self.metadata.kv_shape[0],
            "num_kv_heads": self.metadata.kv_shape[3],
            "head_dim": self.metadata.kv_shape[4],
            "model_name": self.metadata.model_name,
            "world_size": self.metadata.world_size,
            "worker_id": self.metadata.worker_id,
            "config": self.makv_config,
            "request_token_count": transfer_spec.get("request_token_count"),
        }
        plan_started = time.perf_counter()
        try:
            if precision_plan is not None:
                plan = build_chunk_quant_plan_from_precision_plan(
                    precision_plan=precision_plan,
                    actual_prompt_token_hash=str(
                        transfer_spec.get("prompt_token_hash", "")
                    ),
                    **common,
                )
            else:
                plan = build_chunk_quant_plan(
                    importance=importance,
                    importance_layout_hint=importance_layout,
                    importance_status=importance_status,
                    **common,
                )
        except (KeyError, TypeError, ValueError) as error:
            plan_build_ms = (time.perf_counter() - plan_started) * 1000
            if self.makv_config.fallback != "naive":
                raise
            logger.warning("MaKV plan rejected; storing naive fallback: %s", error)
            raw_started = time.perf_counter()
            raw = bytes(memory_obj.byte_array)
            raw_payload_copy_ms = (time.perf_counter() - raw_started) * 1000
            envelope_started = time.perf_counter()
            envelope = encode_client_put_envelope(
                key=key.to_string() if key is not None else "",
                object_type="naive_fallback",
                plan=None,
                raw_kv_payload=raw,
                extra_metadata={"fallback_reason": f"invalid_plan:{error}"},
            )
            self._record_put_timing(
                started=t0,
                plan_build_ms=plan_build_ms,
                raw_payload_copy_ms=raw_payload_copy_ms,
                envelope_encode_ms=(time.perf_counter() - envelope_started) * 1000,
                raw_bytes=len(raw),
                envelope_bytes=len(envelope),
            )
            return BytesBufferMemoryObj(envelope)
        plan_build_ms = (time.perf_counter() - plan_started) * 1000

        raw_started = time.perf_counter()
        raw = bytes(memory_obj.byte_array)
        raw_payload_copy_ms = (time.perf_counter() - raw_started) * 1000
        envelope_started = time.perf_counter()
        envelope = encode_client_put_envelope(
            key=key.to_string() if key is not None else "",
            object_type="raw_with_plan",
            plan=plan,
            raw_kv_payload=raw,
            extra_metadata={
                "request_token_count": transfer_spec.get("request_token_count"),
            },
        )
        self._record_put_timing(
            started=t0,
            plan_build_ms=plan_build_ms,
            raw_payload_copy_ms=raw_payload_copy_ms,
            envelope_encode_ms=(time.perf_counter() - envelope_started) * 1000,
            raw_bytes=len(raw),
            envelope_bytes=len(envelope),
        )
        return BytesBufferMemoryObj(envelope)


class MaKVDeserializer(Deserializer):
    """Return a delayed-restore MaKV object or decode a fallback envelope."""

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheMetadata):
        self.config = config
        self.metadata = metadata
        self.makv_config = get_makv_config(config)
        self.naive = NaiveDeserializer()

    def deserialize(self, memory_obj: MemoryObj) -> MemoryObj:
        if not isinstance(memory_obj, BytesBufferMemoryObj):
            return memory_obj
        started = time.perf_counter()
        checksum_verified = bool(
            getattr(memory_obj, "makv_checksum_verified", False)
        )
        makv_object = decode_makv_object(
            memory_obj.byte_array,
            copy_payloads=False,
            verify_checksum=not checksum_verified,
        )
        if makv_object.object_type == "naive_fallback":
            raw_bytes = makv_object.payloads["raw_kv_payload"]
            shape = self.metadata.get_shapes()[0]
            tensor = torch.frombuffer(
                bytearray(raw_bytes),
                dtype=self.metadata.kv_dtype,
            ).view(shape)
            result = TensorMemoryObj(
                raw_data=tensor,
                metadata=MemoryObjMetadata(
                    shape=shape,
                    dtype=tensor.dtype,
                    address=-1,
                    phy_size=tensor.numel() * tensor.element_size(),
                    ref_count=1,
                    pin_count=0,
                    fmt=MemoryFormat.KV_MLA_FMT
                    if self.metadata.use_mla
                    else MemoryFormat.KV_2LTD,
                ),
                parent_allocator=None,
            )
            CLIENT_METRICS.add(
                makv_client_deserialize_time_ms=(time.perf_counter() - started)
                * 1000
            )
            return result
        plan = makv_object.metadata.get("plan")
        if not isinstance(plan, dict):
            raise ValueError("MaKV object is missing its quantization plan")
        expected_model = model_fingerprint(
            self.metadata.model_name,
            self.metadata.kv_shape[0],
            self.metadata.kv_shape[3],
            self.metadata.kv_shape[4],
        )
        expected_parallel = parallel_fingerprint(
            self.metadata.world_size, self.metadata.worker_id
        )
        if plan.get("model_fingerprint") != expected_model:
            raise ValueError("MaKV model fingerprint mismatch")
        if plan.get("parallel_fingerprint") != expected_parallel:
            raise ValueError("MaKV parallel fingerprint mismatch")
        if plan.get("original_dtype") != str(self.metadata.kv_dtype):
            raise ValueError("MaKV dtype metadata mismatch")
        if (
            str(plan.get("precision_scheme", "shared"))
            != self.makv_config.precision_scheme
        ):
            raise ValueError("MaKV precision scheme metadata mismatch")
        if (
            tuple(int(value) for value in plan.get("bucket_bits", ()))
            != self.makv_config.bucket_bits
        ):
            raise ValueError("MaKV bucket bit metadata mismatch")
        expected_geometry = (
            self.metadata.kv_shape[0],
            self.metadata.kv_shape[3],
            self.metadata.kv_shape[4],
        )
        actual_geometry = (
            int(plan["num_layers"]),
            int(plan["num_kv_heads"]),
            int(plan["head_dim"]),
        )
        if actual_geometry != expected_geometry:
            raise ValueError("MaKV KV geometry mismatch")
        result = MaKVQuantizedMemoryObj(
            memory_obj.byte_array,
            metadata_dict=makv_object.metadata,
            payloads=makv_object.payloads,
            protocol_version=makv_object.protocol_version,
        )
        # Preserve request-local network/manager timings through delayed GPU
        # restoration. These fields are intentionally optional for legacy and
        # non-network MaKV objects.
        for name in ("makv_server_timing", "makv_transport_timing"):
            value = getattr(memory_obj, name, None)
            if isinstance(value, dict):
                setattr(result, name, dict(value))
        CLIENT_METRICS.add(
            makv_client_deserialize_time_ms=(time.perf_counter() - started) * 1000
        )
        return result
