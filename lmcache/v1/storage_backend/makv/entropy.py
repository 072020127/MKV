# SPDX-License-Identifier: Apache-2.0

"""Optional MaKV arithmetic coding built on CacheGen's AC kernels.

CacheGen's CUDA arithmetic coder accepts a per-channel stream with at most
256 symbols and a CDF with fewer than 64 bins.  MaKV quantized values are
therefore represented as small non-negative alphabets before coding:

* INT2: ``q + 1`` gives three symbols;
* INT4: ``q + 7`` gives fifteen symbols;
* INT8: the unsigned ``q + 127`` byte is split into low/high nibbles.

The resulting CDFs, stream lengths, and byte streams are ordinary MaKV
payloads.  No pickle or CacheGen object serialization is used here.  The
codec is deliberately optional and is disabled by default.
"""

# Standard
from collections.abc import Callable
from typing import Any
import importlib
import warnings

# Third Party
import torch

# First Party
from lmcache.v1.storage_backend.makv.reference_dequant import (
    _unpack_low_bit,
)

CODEC_NAME = "cachegen_arithmetic_v1"
CODEC_VERSION = 1
STREAM_TOKENS = 256
STREAM_BUFFER_BYTES = 256
SUPPORTED_ENTROPY_BITS = (2, 4, 8)


def _tensor_from_bytes(
    data: bytes | memoryview, dtype: torch.dtype, *, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Create a tensor view over a serialized payload, then optionally copy it."""
    if len(data) == 0:
        return torch.empty((0,), dtype=dtype, device=device)
    if len(data) % torch.empty((), dtype=dtype).element_size() != 0:
        raise ValueError("MaKV entropy payload is not aligned to its dtype")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tensor = torch.frombuffer(data, dtype=dtype)
    if str(device) != "cpu":
        tensor = tensor.to(device=device, non_blocking=True)
    return tensor


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().numpy().tobytes()


def _cuda_codec_module() -> Any | None:
    """Return the formal CacheGen extension when it exposes AC kernels.

    ``cuda_ops`` is the current wheel/build target. ``c_ops`` is retained as
    a compatibility fallback for installations produced by older LMCache
    commits, but a stale extension must not prevent the current module from
    being selected.
    """
    required = ("calculate_cdf", "encode_fast_new", "decode_fast_prefsum")
    for module_name in ("lmcache.cuda_ops", "lmcache.c_ops"):
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError):
            continue
        if all(hasattr(module, name) for name in required):
            return module
    return None


def _select_backend(backend: str, require_cuda: bool) -> tuple[str, Any | None]:
    backend = str(backend).strip().lower()
    if backend not in ("auto", "cuda", "reference"):
        raise ValueError(f"unsupported MaKV entropy backend: {backend!r}")
    if backend == "reference" and require_cuda:
        raise ValueError(
            "MaKV entropy backend=reference cannot be used with "
            "require_cuda=true"
        )
    if backend != "reference":
        cuda_module = _cuda_codec_module()
        cuda_available = bool(cuda_module is not None and torch.cuda.is_available())
        if cuda_available:
            return "cuda", cuda_module
        if backend == "cuda" or require_cuda:
            raise RuntimeError(
                "MaKV arithmetic codec requires the CacheGen CUDA extension "
                "and an available CUDA device"
            )
    return "reference", None


def _alphabet_size(bits: int) -> int:
    try:
        return {2: 3, 4: 15, 8: 16}[int(bits)]
    except KeyError as error:
        raise ValueError(
            f"MaKV arithmetic coding does not support {bits} bit"
        ) from error


def _qmax(bits: int) -> int:
    return {2: 1, 4: 7, 8: 127}[int(bits)]


def _pack_low_bit(q: torch.Tensor, head_dim: int, bits: int) -> torch.Tensor:
    """Pack rows using the same little-endian nibble/field order as quantizer."""
    if bits not in (2, 4):
        raise ValueError("only INT2 and INT4 use low-bit packing")
    if q.ndim != 2 or q.shape[1] != head_dim:
        raise ValueError("low-bit quantized rows do not match head_dim")
    values_per_byte = 8 // bits
    padding = (-head_dim) % values_per_byte
    if padding:
        q = torch.nn.functional.pad(q, (0, padding))
    bytes_per_row = (head_dim + values_per_byte - 1) // values_per_byte
    fields = (q.to(torch.int16) & ((1 << bits) - 1)).view(
        q.shape[0], bytes_per_row, values_per_byte
    )
    shifts = (
        torch.arange(values_per_byte, dtype=torch.int16, device=q.device) * bits
    )
    return torch.sum(fields << shifts, dim=-1).to(torch.uint8).flatten()


def _bucket_geometry(
    metadata: dict[str, Any], bits: int
) -> tuple[int, int, int, int]:
    plan = metadata.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("MaKV entropy metadata is missing its plan")
    matching = [
        entry
        for entry in metadata.get("bucket_entries", [])
        if int(entry.get("bits", -1)) == bits
    ]
    if len(matching) != 1:
        raise ValueError(f"MaKV entropy metadata has no unique {bits}-bit bucket")
    count = int(matching[0]["count"])
    layers = int(plan["num_layers"])
    heads = int(plan["num_kv_heads"])
    head_dim = int(plan["head_dim"])
    layout = str(plan["importance_layout"])
    vectors = (
        count * heads
        if layout == "layer_kv_token"
        else count * layers * 2 * heads
    )
    if count < 0 or vectors < 0 or head_dim <= 0:
        raise ValueError("invalid MaKV entropy bucket geometry")
    return count, vectors, head_dim, vectors * head_dim


def _payload_to_symbol_planes(
    payload: bytes | memoryview,
    *,
    bits: int,
    vectors: int,
    head_dim: int,
) -> list[tuple[str, torch.Tensor]]:
    """Convert the production quantized payload into arithmetic symbols."""
    if bits == 8:
        expected = vectors * head_dim
        if len(payload) != expected:
            raise ValueError(
                f"MaKV INT8 payload length mismatch: {len(payload)} != {expected}"
            )
        q = _tensor_from_bytes(payload, torch.int8)
        unsigned = q.to(torch.int16) + _qmax(bits)
        return [
            ("low", (unsigned & 0x0F).to(torch.int8)),
            ("high", (unsigned >> 4).to(torch.int8)),
        ]
    values_per_byte = 8 // bits
    expected = vectors * ((head_dim + values_per_byte - 1) // values_per_byte)
    if len(payload) != expected:
        raise ValueError(
            f"MaKV INT{bits} packed payload length mismatch: "
            f"{len(payload)} != {expected}"
        )
    if vectors == 0:
        return [("value", torch.empty((0,), dtype=torch.int8))]
    packed = _tensor_from_bytes(payload, torch.uint8)
    q = _unpack_low_bit(packed, vectors, head_dim, bits).reshape(-1)
    return [("value", (q.to(torch.int16) + _qmax(bits)).to(torch.int8))]


def _encode_symbol_stream(
    symbols: torch.Tensor,
    *,
    max_bins: int,
    backend: str,
    cuda_module: Any | None,
) -> tuple[bytes, bytes, bytes, int]:
    """Encode one symbol plane and return stream, CDF, lengths, stream count."""
    if symbols.ndim != 1:
        raise ValueError("MaKV entropy symbols must be one-dimensional")
    symbol_count = int(symbols.numel())
    if symbol_count == 0:
        return b"", b"", b"", 0
    if int(symbols.min()) < 0 or int(symbols.max()) >= max_bins:
        raise ValueError("MaKV entropy symbol is outside its CDF alphabet")
    stream_count = (symbol_count + STREAM_TOKENS - 1) // STREAM_TOKENS
    input_cpu = torch.zeros(
        (stream_count, STREAM_TOKENS, 1), dtype=torch.int8, device="cpu"
    )
    input_cpu.view(-1)[:symbol_count].copy_(symbols.to(torch.int8))

    if backend == "cuda":
        assert cuda_module is not None
        device = torch.device("cuda", torch.cuda.current_device())
        input_symbols = input_cpu.to(device=device, non_blocking=True)
        cdf = cuda_module.calculate_cdf(input_symbols, max_bins)
        output_buffer = torch.zeros(
            (stream_count, 1, STREAM_BUFFER_BYTES),
            dtype=torch.uint8,
            device=device,
        )
        output_lengths = torch.zeros(
            (stream_count, 1), dtype=torch.int32, device=device
        )
        cuda_module.encode_fast_new(cdf, input_symbols, output_buffer, output_lengths)
        # The encoded object is CPU-resident.  These copies are on the manager
        # PUT path only; the GET path keeps its decode inputs on the target GPU.
        cdf_cpu = cdf.cpu()
        output_cpu = output_buffer.cpu()
        lengths_cpu = output_lengths.cpu().flatten()
    else:
        from lmcache.v1.platform import torch_ops

        cdf_cpu = torch_ops.calculate_cdf(input_cpu, max_bins)
        output_cpu = torch.zeros(
            (stream_count, 1, STREAM_BUFFER_BYTES), dtype=torch.uint8
        )
        lengths_cpu = torch.zeros((stream_count, 1), dtype=torch.int32)
        torch_ops.encode_fast_new(cdf_cpu, input_cpu, output_cpu, lengths_cpu)
        lengths_cpu = lengths_cpu.flatten()

    lengths = [int(value) for value in lengths_cpu.tolist()]
    if any(length <= 0 or length > STREAM_BUFFER_BYTES for length in lengths):
        raise ValueError(
            "CacheGen arithmetic encoder produced an invalid stream length"
        )
    stream = b"".join(
        output_cpu[index, 0, :length].contiguous().numpy().tobytes()
        for index, length in enumerate(lengths)
    )
    return (
        stream,
        _tensor_bytes(cdf_cpu),
        _tensor_bytes(lengths_cpu.to(torch.int32)),
        stream_count,
    )


def encode_entropy_payloads(
    metadata: dict[str, Any],
    payloads: dict[str, bytes],
    *,
    codec: str = "none",
    backend: str = "auto",
    require_cuda: bool = False,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Arithmetic-code MaKV low-bit payloads without changing the 16-bit data."""
    codec = str(codec).strip().lower().replace("-", "_")
    if codec == "none":
        return metadata, payloads
    if codec != "cachegen_arithmetic":
        raise ValueError(f"unsupported MaKV entropy codec: {codec!r}")
    selected_backend, cuda_module = _select_backend(backend, require_cuda)
    output_payloads = dict(payloads)
    entropy: dict[str, Any] = {
        "codec": CODEC_NAME,
        "version": CODEC_VERSION,
        "backend": selected_backend,
        "stream_tokens": STREAM_TOKENS,
        "buckets": {},
        "input_bytes": 0,
        "output_bytes": 0,
    }
    for bits in (int(value) for value in metadata["plan"]["bucket_bits"]):
        if bits not in SUPPORTED_ENTROPY_BITS:
            continue
        _, vectors, head_dim, logical_elements = _bucket_geometry(metadata, bits)
        raw_name = f"payload_{bits}"
        if raw_name not in output_payloads:
            raise ValueError(f"MaKV entropy input is missing {raw_name}")
        raw_payload = output_payloads.pop(raw_name)
        entropy_bucket: dict[str, Any] = {
            "bits": bits,
            "vectors": vectors,
            "head_dim": head_dim,
            "logical_elements": logical_elements,
            "input_payload_bytes": len(raw_payload),
            "planes": [],
        }
        for plane_index, (plane_name, symbols) in enumerate(
            _payload_to_symbol_planes(
                raw_payload,
                bits=bits,
                vectors=vectors,
                head_dim=head_dim,
            )
        ):
            stream, cdf, lengths, stream_count = _encode_symbol_stream(
                symbols,
                max_bins=_alphabet_size(bits),
                backend=selected_backend,
                cuda_module=cuda_module,
            )
            prefix = f"{bits}_p{plane_index}"
            stream_name = f"entropy_stream_{prefix}"
            cdf_name = f"entropy_cdf_{prefix}"
            lengths_name = f"entropy_lengths_{prefix}"
            output_payloads[stream_name] = stream
            output_payloads[cdf_name] = cdf
            output_payloads[lengths_name] = lengths
            entropy_bucket["planes"].append(
                {
                    "name": plane_name,
                    "stream_name": stream_name,
                    "cdf_name": cdf_name,
                    "lengths_name": lengths_name,
                    "symbol_count": int(symbols.numel()),
                    "stream_count": stream_count,
                    "max_bins": _alphabet_size(bits),
                    "stream_bytes": len(stream),
                    "cdf_bytes": len(cdf),
                    "lengths_bytes": len(lengths),
                }
            )
            entropy["output_bytes"] += len(stream) + len(cdf) + len(lengths)
        entropy["input_bytes"] += len(raw_payload)
        entropy["buckets"][str(bits)] = entropy_bucket
    new_metadata = dict(metadata)
    new_metadata["entropy"] = entropy
    return new_metadata, output_payloads


def _decode_symbol_plane(
    plane: dict[str, Any],
    get_payload: Callable[[str, torch.dtype], torch.Tensor],
) -> torch.Tensor:
    stream_count = int(plane["stream_count"])
    symbol_count = int(plane["symbol_count"])
    max_bins = int(plane["max_bins"])
    stream_bytes = int(plane["stream_bytes"])
    if (
        stream_count < 0
        or symbol_count < 0
        or symbol_count > stream_count * STREAM_TOKENS
        or max_bins <= 0
        or max_bins >= 64
        or stream_bytes < 0
        or stream_bytes > stream_count * STREAM_BUFFER_BYTES
    ):
        raise ValueError("invalid MaKV entropy stream descriptor")
    if stream_count == 0:
        if symbol_count != 0 or stream_bytes != 0:
            raise ValueError("empty MaKV entropy stream has non-empty metadata")
        # Keep empty buckets on the same device as their serialized view. The
        # CUDA MaKV binding validates device placement even for zero-length
        # payloads.
        stream_view = get_payload(str(plane["stream_name"]), torch.uint8)
        return torch.empty((0,), dtype=torch.uint8, device=stream_view.device)
    cdf = get_payload(str(plane["cdf_name"]), torch.int16)
    stream = get_payload(str(plane["stream_name"]), torch.uint8)
    lengths = get_payload(str(plane["lengths_name"]), torch.int32)
    if cdf.numel() != stream_count * (max_bins + 1):
        raise ValueError("MaKV entropy CDF length mismatch")
    if lengths.numel() != stream_count:
        raise ValueError("MaKV entropy stream length table mismatch")
    if stream.numel() != stream_bytes:
        raise ValueError("MaKV entropy bytestream length mismatch")
    if lengths.device.type == "cpu":
        length_values = [int(value) for value in lengths.tolist()]
        if any(value <= 0 or value > STREAM_BUFFER_BYTES for value in length_values):
            raise ValueError("MaKV entropy stream length is out of range")
    cdf = cdf.contiguous().view(stream_count, 1, max_bins + 1)
    stream = stream.contiguous().view(-1)
    lengths_prefix = lengths.to(torch.int64).cumsum(0).view(stream_count, 1)
    output = torch.empty(
        (stream_count, STREAM_TOKENS, 1), dtype=torch.uint8, device=stream.device
    )
    if stream.device.type == "cuda":
        cuda_module = _cuda_codec_module()
        if cuda_module is None:
            raise RuntimeError("MaKV CUDA arithmetic decoder is unavailable")
        cuda_module.decode_fast_prefsum(cdf, stream, lengths_prefix, output)
    else:
        from lmcache.v1.platform import torch_ops

        torch_ops.decode_fast_prefsum(cdf, stream, lengths_prefix, output)
    if symbol_count > output.numel():
        raise ValueError("MaKV entropy symbol count exceeds decoded stream capacity")
    return output.reshape(-1)[:symbol_count]


def decode_entropy_payloads(
    metadata: dict[str, Any],
    get_payload: Callable[[str, torch.dtype], torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Decode arithmetic payloads into compact quantized payload tensors.

    The returned tensors contain INT8 values or the original per-row packed
    UINT8 representation.  They are never expanded to a floating-point KV
    tensor; the existing MaKV dequantize/scatter operator consumes them.
    """
    entropy = metadata.get("entropy")
    if not isinstance(entropy, dict):
        return {}
    if (
        entropy.get("codec") != CODEC_NAME
        or int(entropy.get("version", -1)) != CODEC_VERSION
    ):
        raise ValueError("unsupported MaKV entropy codec version")
    buckets = entropy.get("buckets", {})
    if not isinstance(buckets, dict):
        raise ValueError("invalid MaKV entropy bucket table")
    decoded: dict[int, torch.Tensor] = {}
    for bits_text, bucket in buckets.items():
        bits = int(bits_text)
        if bits not in SUPPORTED_ENTROPY_BITS or not isinstance(bucket, dict):
            raise ValueError("invalid MaKV entropy bucket descriptor")
        vectors = int(bucket["vectors"])
        head_dim = int(bucket["head_dim"])
        logical_elements = int(bucket["logical_elements"])
        if (
            vectors < 0
            or head_dim <= 0
            or logical_elements < 0
            or logical_elements != vectors * head_dim
        ):
            raise ValueError("invalid MaKV entropy bucket geometry")
        planes = bucket.get("planes")
        if not isinstance(planes, list) or not planes:
            raise ValueError("MaKV entropy bucket has no planes")
        plane_values: dict[str, torch.Tensor] = {}
        for plane in planes:
            name = str(plane["name"])
            if name in plane_values:
                raise ValueError("MaKV entropy bucket contains duplicate planes")
            plane_values[name] = _decode_symbol_plane(plane, get_payload)
        if bits == 8:
            if set(plane_values) != {"low", "high"}:
                raise ValueError(
                    "MaKV INT8 entropy bucket must have low/high planes"
                )
            if any(
                value.numel() != logical_elements
                for value in plane_values.values()
            ):
                raise ValueError("MaKV INT8 entropy symbol count mismatch")
            unsigned = plane_values["low"].to(torch.int16) | (
                plane_values["high"].to(torch.int16) << 4
            )
            payload = (unsigned - _qmax(bits)).to(torch.int8).contiguous()
        else:
            if len(plane_values) != 1:
                raise ValueError(f"MaKV INT{bits} entropy bucket must have one plane")
            symbols = next(iter(plane_values.values()))
            if symbols.numel() != logical_elements:
                raise ValueError(f"MaKV INT{bits} entropy symbol count mismatch")
            q = (symbols.to(torch.int16) - _qmax(bits)).to(torch.int8)
            payload = _pack_low_bit(q.view(vectors, head_dim), head_dim, bits)
        expected_packed_bytes = vectors * (
            head_dim if bits == 8 else (head_dim + (8 // bits) - 1) // (8 // bits)
        )
        if payload.numel() != expected_packed_bytes:
            raise ValueError("MaKV entropy decoded payload length mismatch")
        decoded[bits] = payload
    return decoded
