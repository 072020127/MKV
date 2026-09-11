# SPDX-License-Identifier: Apache-2.0

"""Reference MaKV remote quantizer."""

# Standard
from dataclasses import dataclass
from typing import Any

# Third Party
import torch

# First Party
from lmcache.v1.storage_backend.makv.config import MaKVConfig
from lmcache.v1.storage_backend.makv.plan import MaKVQuantPlan
from lmcache.v1.storage_backend.makv.reference_dequant import (
    dequantize_bucket_vectors,
)
from lmcache.v1.storage_backend.makv.residual import (
    RESIDUAL_SEMANTICS,
    RESIDUAL_VERSION,
    residual_payload_name,
    residual_shape,
    residual_torch_dtype,
)

QUANTIZER_VERSION = "makv_per_token_head_symmetric_narrow_v1"


@dataclass(frozen=True)
class VectorFakeQuantization:
    """One production-semantic per-vector quantize/dequantize result.

    ``quantized`` retains the unpacked signed integer values so callers that
    only need a reconstructed vector do not need to materialize a serialized
    low-bit payload. ``stored_scale`` is the scale after the exact on-wire
    dtype cast; reconstruction deliberately uses this value rather than the
    higher-precision temporary scale.
    """

    quantized: torch.Tensor
    stored_scale: torch.Tensor
    reconstructed: torch.Tensor


def _scale_dtype(scale_dtype: str) -> torch.dtype:
    return torch.float16 if scale_dtype == "float16" else torch.float32


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize tensor storage without relying on NumPy dtype support."""
    return tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes()


def fake_quantize_dequantize_vectors(
    vectors: torch.Tensor,
    bits: int,
    scale_dtype: str,
) -> VectorFakeQuantization:
    """Apply MaKV's production per-token/head quantization arithmetic.

    Args:
        vectors: Tensor whose final dimension is one KV head vector.
        bits: Physical MaKV width, one of 2, 4, 8, or 16.
        scale_dtype: Serialized scale dtype (``float16`` or ``float32``).

    Returns:
        The unpacked signed integers, serialized-scale tensor, and restored
        vectors cast back to ``vectors.dtype``.

    Notes:
        The remote serializer calls this same core before low-bit packing.
        Packing is a lossless transport of ``quantized`` and does not alter
        the fake-quant reconstruction.
    """
    if vectors.ndim < 1 or vectors.shape[-1] <= 0:
        raise ValueError("vectors must have a non-empty final head dimension")
    stored_dtype = _scale_dtype(scale_dtype)
    if bits == 16:
        return VectorFakeQuantization(
            quantized=vectors,
            stored_scale=torch.empty(0, dtype=stored_dtype, device=vectors.device),
            reconstructed=vectors,
        )
    if bits not in (2, 4, 8):
        raise ValueError(f"Unsupported MaKV quantization width: {bits}")
    qmax = {2: 1, 4: 7, 8: 127}[bits]
    max_abs = vectors.abs().amax(dim=-1)
    temporary_scale = torch.where(
        max_abs == 0, torch.ones_like(max_abs), max_abs / qmax
    )
    quantized = (
        torch.round(vectors / temporary_scale.unsqueeze(-1))
        .clamp(-qmax, qmax)
        .to(torch.int8)
    )
    stored_scale = temporary_scale.to(stored_dtype)
    reconstructed = (
        quantized.to(torch.float32) * stored_scale.to(torch.float32).unsqueeze(-1)
    ).to(vectors.dtype)
    return VectorFakeQuantization(
        quantized=quantized,
        stored_scale=stored_scale,
        reconstructed=reconstructed,
    )


def per_token_head_storage_bytes(
    bits: int,
    head_dim: int,
    scale_dtype: str,
    *,
    raw_dtype_bytes: int = 2,
) -> int:
    """Return exact variable payload bytes for one MaKV token/head vector.

    The calculation mirrors the row-local packing in
    ``_quantize_vector_bucket``. It excludes object framing and positions;
    callers that build a whole token plan can add the fixed int32 position
    entry for each K/V plane separately.
    """
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if raw_dtype_bytes <= 0:
        raise ValueError("raw_dtype_bytes must be positive")
    if bits == 16:
        return head_dim * raw_dtype_bytes
    if bits not in (2, 4, 8):
        raise ValueError(f"Unsupported MaKV quantization width: {bits}")
    values_per_byte = 8 // bits
    payload = (head_dim + values_per_byte - 1) // values_per_byte
    return payload + torch.empty((), dtype=_scale_dtype(scale_dtype)).element_size()


def _quantize_vector_bucket(
    vectors: torch.Tensor,
    bits: int,
    scale_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    fake = fake_quantize_dequantize_vectors(vectors, bits, scale_dtype)
    if bits == 16:
        return vectors, fake.stored_scale
    q = fake.quantized
    if bits == 8:
        return q, fake.stored_scale
    # Low-bit values use two's-complement fields packed within each row.
    q_int = q
    flat = q_int.reshape(-1, q_int.shape[-1])
    values_per_byte = 8 // bits
    field_mask = (1 << bits) - 1
    padding = (-flat.shape[-1]) % values_per_byte
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    fields = (flat.to(torch.int16) & field_mask).view(
        flat.shape[0], -1, values_per_byte
    )
    shifts = torch.arange(values_per_byte, dtype=torch.int16) * bits
    shifts = shifts.to(device=flat.device)
    packed_tensor = torch.sum(fields << shifts, dim=-1).to(torch.uint8).flatten()
    return packed_tensor, fake.stored_scale


def quantize_canonical_kv(
    kv_tensor: torch.Tensor,
    plan: MaKVQuantPlan,
    config: MaKVConfig,
    *,
    qdm_observer: Any | None = None,
    enable_qdm: bool = False,
    qdm_block_size: int = 32,
    qdm_quantizer_version: str = QUANTIZER_VERSION,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Quantize canonical KV tensor ``[L,2,T,H,D]`` according to a chunk plan.

    QDM is an optional shadow observer downstream of the existing quantizer.
    With ``enable_qdm=False`` this function follows the original path without
    observer work, even if a caller accidentally supplies an observer.
    """
    # Keep the production path hard-gated.  In particular, an observer object
    # must not be able to opt the path in when the feature is disabled.
    if not enable_qdm:
        qdm_observer = None
    elif qdm_observer is None:
        # Keep the import lazy so the disabled production path has no QDM
        # import or allocation cost.
        from lmcache.v1.storage_backend.makv.qdm import QDMObserver

        qdm_observer = QDMObserver(
            plan,
            block_size=qdm_block_size,
            quantizer_version=qdm_quantizer_version,
        )
    bits_to_bucket = {bit: idx for idx, bit in enumerate(plan.bucket_bits)}
    payloads: dict[str, bytes] = {}
    metadata: dict[str, Any] = {
        "plan": plan.to_dict(),
        "bucket_entries": [],
    }
    residual_dtype_name = str(getattr(config, "residual_dtype", "none"))
    residual_dtype = (
        residual_torch_dtype(residual_dtype_name)
        if residual_dtype_name != "none"
        else None
    )
    residual_entries: list[dict[str, Any]] = []

    def add_residual(
        vectors: torch.Tensor,
        q_payload: torch.Tensor,
        scales: torch.Tensor,
        bits: int,
        count: int,
    ) -> None:
        """Persist the elementwise source-minus-restored error for one bucket."""
        if residual_dtype is None or bits == 16:
            return
        restored = dequantize_bucket_vectors(
            q_payload,
            scales,
            bits,
            vector_shape=tuple(vectors.shape),
            head_dim=plan.head_dim,
            output_dtype=torch.float32,
        )
        residual = (vectors.float() - restored.float()).to(residual_dtype)
        name = residual_payload_name(bits)
        payloads[name] = _tensor_bytes(residual)
        shape = residual_shape(plan.to_dict(), count)
        residual_entries.append(
            {
                "bits": int(bits),
                "count": int(count),
                "layout": plan.importance_layout,
                "shape": list(shape),
                "element_count": int(residual.numel()),
                "payload_name": name,
            }
        )

    def add_empty_residual(bits: int) -> None:
        """Keep the residual table complete when a precision bucket is empty."""
        if residual_dtype is None or bits == 16:
            return
        name = residual_payload_name(bits)
        payloads[name] = b""
        shape = residual_shape(plan.to_dict(), 0)
        residual_entries.append(
            {
                "bits": int(bits),
                "count": 0,
                "layout": plan.importance_layout,
                "shape": list(shape),
                "element_count": 0,
                "payload_name": name,
            }
        )

    if plan.importance_layout == "token":
        bucket_ids = torch.tensor(list(plan.bucket_ids), dtype=torch.uint8)
        for bit in plan.bucket_bits:
            bucket_id = bits_to_bucket[bit]
            positions = torch.nonzero(bucket_ids == bucket_id, as_tuple=False).flatten()
            token_positions = positions.to(torch.int32)
            if token_positions.numel() == 0:
                payloads[f"positions_{bit}"] = b""
                payloads[f"payload_{bit}"] = b""
                payloads[f"scales_{bit}"] = b""
                add_empty_residual(bit)
                metadata["bucket_entries"].append(
                    {"bits": bit, "count": 0, "layout": "token"}
                )
                continue
            vectors = kv_tensor[
                :, :, token_positions.to(device=kv_tensor.device).long(), :, :
            ]
            q_payload, scales = _quantize_vector_bucket(
                vectors, bit, config.scale_dtype
            )
            add_residual(
                vectors,
                q_payload,
                scales,
                bit,
                int(token_positions.numel()),
            )
            if qdm_observer is not None:
                qdm_observer.observe_bucket(
                    vectors=vectors,
                    q_payload=q_payload,
                    scales=scales,
                    bits=bit,
                    positions=token_positions,
                )
            payloads[f"positions_{bit}"] = token_positions.numpy().tobytes()
            payloads[f"payload_{bit}"] = _tensor_bytes(q_payload)
            payloads[f"scales_{bit}"] = _tensor_bytes(scales)
            metadata["bucket_entries"].append(
                {
                    "bits": bit,
                    "count": int(token_positions.numel()),
                    "layout": "token",
                }
            )
    else:
        bucket_ids = torch.tensor(list(plan.bucket_ids), dtype=torch.uint8).view(
            plan.num_layers, 2, plan.chunk_length
        )
        for bit in plan.bucket_bits:
            bucket_id = bits_to_bucket[bit]
            indices = torch.nonzero(bucket_ids == bucket_id, as_tuple=False)
            flat_positions = (
                ((indices[:, 0] * 2 + indices[:, 1]) * plan.chunk_length)
                + indices[:, 2]
            ).to(torch.int32)
            if flat_positions.numel() == 0:
                payloads[f"positions_{bit}"] = b""
                payloads[f"payload_{bit}"] = b""
                payloads[f"scales_{bit}"] = b""
                add_empty_residual(bit)
                metadata["bucket_entries"].append(
                    {"bits": bit, "count": 0, "layout": "layer_kv_token"}
                )
                continue
            indices_device = indices.to(device=kv_tensor.device)
            gathered = kv_tensor[
                indices_device[:, 0],
                indices_device[:, 1],
                indices_device[:, 2],
                :,
                :,
            ]
            q_payload, scales = _quantize_vector_bucket(
                gathered, bit, config.scale_dtype
            )
            add_residual(
                gathered,
                q_payload,
                scales,
                bit,
                int(flat_positions.numel()),
            )
            if qdm_observer is not None:
                qdm_observer.observe_bucket(
                    vectors=gathered,
                    q_payload=q_payload,
                    scales=scales,
                    bits=bit,
                    positions=flat_positions,
                )
            payloads[f"positions_{bit}"] = flat_positions.numpy().tobytes()
            payloads[f"payload_{bit}"] = _tensor_bytes(q_payload)
            payloads[f"scales_{bit}"] = _tensor_bytes(scales)
            metadata["bucket_entries"].append(
                {
                    "bits": bit,
                    "count": int(flat_positions.numel()),
                    "layout": "layer_kv_token",
                }
            )
    if qdm_observer is not None:
        qdm = qdm_observer.finalize()
        metadata["qdm"] = qdm.to_descriptor()
        payloads.update(qdm.to_payloads())
    if residual_dtype is not None:
        metadata["residual"] = {
            "version": RESIDUAL_VERSION,
            "semantics": RESIDUAL_SEMANTICS,
            "dtype": residual_dtype_name,
            "source_dtype": plan.original_dtype,
            "buckets": residual_entries,
        }
    return metadata, payloads
