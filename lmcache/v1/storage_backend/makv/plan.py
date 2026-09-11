# SPDX-License-Identifier: Apache-2.0

"""Deterministic MaKV quantization plan generation."""

# Standard
from dataclasses import asdict, dataclass
from typing import Any, Optional
import hashlib
import json
import math

# Third Party
import torch

# First Party
from lmcache.v1.storage_backend.makv.config import MaKVConfig


def _fingerprint(parts: list[str]) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def model_fingerprint(
    model_name: str, num_layers: int, num_kv_heads: int, head_dim: int
) -> str:
    """Return the model-layout compatibility fingerprint."""
    return _fingerprint([model_name, str(num_layers), str(num_kv_heads), str(head_dim)])


def parallel_fingerprint(world_size: int, worker_id: int) -> str:
    """Return the parallel-worker compatibility fingerprint."""
    return _fingerprint([str(world_size), str(worker_id)])


@dataclass(frozen=True)
class MaKVQuantPlan:
    protocol_version: int
    importance_layout: str
    token_count: int
    chunk_start: int
    chunk_length: int
    bucket_bits: tuple[int, ...]
    bucket_ids: bytes
    original_shape: tuple[int, ...]
    original_strides: tuple[int, ...]
    original_dtype: str
    token_dim: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    quant_granularity: str
    scale_dtype: str
    model_fingerprint: str
    parallel_fingerprint: str
    checksum: int
    nan_protected_count: int = 0
    inf_protected_count: int = 0
    source_plan_hash: str = ""
    source_strategy: str = ""
    prompt_token_hash: str = ""
    precision_plan_schema: str = ""
    precision_scheme: str = "shared"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bucket_bits"] = list(self.bucket_bits)
        data["bucket_ids"] = list(self.bucket_ids)
        return data


def compute_quant_plan_checksum(
    plan: MaKVQuantPlan, bucket_ids: bytes | None = None
) -> int:
    """Compute the deterministic checksum for a plan's layout assignment."""
    assigned_bucket_ids = plan.bucket_ids if bucket_ids is None else bytes(bucket_ids)
    checksum_fields = {
        "layout": plan.importance_layout,
        "chunk_start": plan.chunk_start,
        "chunk_end": plan.chunk_start + plan.chunk_length,
        "bucket_bits": list(plan.bucket_bits),
        "bucket_ids": list(assigned_bucket_ids),
        "shape": list(plan.original_shape),
        "strides": list(plan.original_strides),
        "dtype": plan.original_dtype,
        "model_fp": plan.model_fingerprint,
        "parallel_fp": plan.parallel_fingerprint,
    }
    if plan.source_plan_hash or plan.prompt_token_hash:
        # Preserve the checksum schema used by ScoutRank-derived plans.
        checksum_fields.update(
            {
                "source_plan_hash": plan.source_plan_hash,
                "prompt_token_hash": plan.prompt_token_hash,
            }
        )
    else:
        checksum_fields["precision_scheme"] = plan.precision_scheme
    checksum_payload = json.dumps(
        checksum_fields, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(checksum_payload).digest()[:4], "little")


def _normalize_importance_tensor(
    importance: Any,
    layout_hint: Optional[str],
    num_layers: int,
    token_count: int,
) -> tuple[torch.Tensor, str]:
    tensor = torch.as_tensor(importance, dtype=torch.float32)
    if tensor.ndim == 1:
        if tensor.shape[0] != token_count:
            raise ValueError(
                f"MaKV importance length mismatch: expected {token_count}, "
                f"got {tensor.shape[0]}"
            )
        return tensor, "token"
    if tensor.ndim == 3:
        expected = (num_layers, 2, token_count)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                "MaKV importance layout [L,2,T] mismatch: expected "
                f"{expected}, got {tuple(tensor.shape)}"
            )
        return tensor, "layer_kv_token"
    if layout_hint is not None:
        raise ValueError(
            f"Unsupported MaKV importance layout {layout_hint!r} with tensor "
            f"shape {tuple(tensor.shape)}"
        )
    raise ValueError(
        f"MaKV importance must be [T] or [L,2,T]; got {tuple(tensor.shape)}"
    )


def _rank_to_bucket_ids(
    scores: list[float],
    config: MaKVConfig,
    forced_bf16_mask: list[bool] | None = None,
) -> tuple[list[int], int, int]:
    n = len(scores)
    if forced_bf16_mask is not None and len(forced_bf16_mask) != n:
        raise ValueError(
            "MaKV importance status length must match importance length"
        )
    ids = [0] * n
    nan_protected = 0
    inf_protected = 0
    protected = set()
    prefix = min(config.protect_prefix_tokens, n)
    tail = min(config.protect_tail_tokens, max(0, n - prefix))
    for idx in range(prefix):
        protected.add(idx)
    for idx in range(n - tail, n):
        protected.add(idx)
    if forced_bf16_mask is not None:
        protected.update(
            index for index, forced in enumerate(forced_bf16_mask) if forced
        )

    sortable: list[tuple[int, float, int]] = []
    for idx, value in enumerate(scores):
        if idx in protected:
            ids[idx] = 0
            continue
        if math.isnan(value):
            ids[idx] = 0
            nan_protected += 1
            continue
        if math.isinf(value):
            ids[idx] = 0
            inf_protected += 1
            continue
        sortable.append((idx, -float(value), idx))

    sortable.sort(key=lambda item: (item[1], item[2]))
    counts = [0] * len(config.bucket_bits)
    remaining = len(sortable)
    prev_boundary = 0
    cumulative = 0.0
    for bucket_idx, ratio in enumerate(config.bucket_ratios):
        cumulative += ratio
        if bucket_idx == len(config.bucket_ratios) - 1:
            boundary = remaining
        else:
            boundary = int(round(cumulative * remaining))
        counts[bucket_idx] = max(0, boundary - prev_boundary)
        prev_boundary = boundary

    cursor = 0
    for bucket_idx, count in enumerate(counts):
        for idx, _, _ in sortable[cursor : cursor + count]:
            ids[idx] = bucket_idx
        cursor += count

    return ids, nan_protected, inf_protected


def _normalize_importance_status(
    status: Any,
    token_count: int,
) -> list[bool] | None:
    """Validate request token status and return the BF16 protection mask."""
    if status is None:
        return None
    if not isinstance(status, (list, tuple)) or len(status) != token_count:
        raise ValueError(
            "MaKV importance status must be a token-count-sized list"
        )
    forced_bf16: list[bool] = []
    for index, item in enumerate(status):
        if not isinstance(item, dict):
            raise ValueError(
                f"MaKV importance status at token {index} must be an object"
            )
        valid_mask = item.get("valid_mask", True)
        if not isinstance(valid_mask, bool):
            raise ValueError(
                f"MaKV importance status valid_mask at token {index} must be bool"
            )
        forced_precision = item.get("forced_precision")
        if forced_precision not in (None, "BF16"):
            raise ValueError(
                "MaKV importance status can only force BF16 precision"
            )
        if not valid_mask and forced_precision != "BF16":
            raise ValueError(
                "invalid MaKV importance token must force BF16 precision"
            )
        forced_bf16.append(
            (not valid_mask) or forced_precision == "BF16"
        )
    return forced_bf16


KV_SEPARATE_3TIER_PAIR_BITS = ((8, 4), (4, 2), (2, 2))
KV_SEPARATE_3TIER_SCHEME = "kv_separate_3tier"
KV_SEPARATE_4TIER_PAIR_BITS = ((16, 16), (8, 4), (4, 2), (2, 2))
KV_SEPARATE_4TIER_SCHEME = "kv_separate_4tier"


def _kv_separate_bucket_ids(tier_ids: list[int], config: MaKVConfig) -> list[int]:
    """Map score tiers to independent K/V bucket IDs.

    The bucket IDs are the physical payload buckets in ``bucket_bits``.  A
    score tier is first mapped to its ``(K_bits, V_bits)`` pair, then each
    plane is emitted independently in the canonical ``K, V`` order.
    """
    if config.precision_scheme == KV_SEPARATE_3TIER_SCHEME:
        pair_bits = KV_SEPARATE_3TIER_PAIR_BITS
        required_bits = (8, 4, 2)
    elif config.precision_scheme == KV_SEPARATE_4TIER_SCHEME:
        pair_bits = KV_SEPARATE_4TIER_PAIR_BITS
        required_bits = (16, 8, 4, 2)
    else:
        raise ValueError("K/V separate bucket mapping used with another scheme")
    if config.bucket_bits != required_bits:
        raise ValueError(
            f"{config.precision_scheme} requires physical bucket bits {required_bits}"
        )
    bit_to_bucket = {bit: index for index, bit in enumerate(config.bucket_bits)}
    result: list[int] = []
    for tier in tier_ids:
        try:
            k_bits, v_bits = pair_bits[tier]
        except IndexError as error:
            raise ValueError(f"invalid K/V separate score tier {tier}") from error
        result.extend((bit_to_bucket[k_bits], bit_to_bucket[v_bits]))
    return result


def build_chunk_quant_plan(
    *,
    importance: Any,
    importance_layout_hint: Optional[str],
    chunk_start: int,
    chunk_end: int,
    original_shape: tuple[int, ...],
    original_strides: tuple[int, ...],
    original_dtype: str,
    token_dim: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    model_name: str,
    world_size: int,
    worker_id: int,
    config: MaKVConfig,
    request_token_count: Optional[int] = None,
    importance_status: Any = None,
) -> MaKVQuantPlan:
    """Build a deterministic per-chunk MaKV quant plan from request-level importance."""
    token_count = (
        int(request_token_count)
        if request_token_count is not None
        else original_shape[token_dim]
    )
    importance_tensor, layout = _normalize_importance_tensor(
        importance,
        importance_layout_hint,
        num_layers,
        token_count,
    )
    if not (0 <= chunk_start <= chunk_end <= token_count):
        raise ValueError(
            f"Invalid MaKV chunk range [{chunk_start}, {chunk_end}) for token count "
            f"{token_count}"
        )
    forced_bf16_mask = _normalize_importance_status(
        importance_status, token_count
    )

    output_layout = layout
    if config.precision_scheme in (
        KV_SEPARATE_3TIER_SCHEME,
        KV_SEPARATE_4TIER_SCHEME,
    ):
        # A token score is ranked once for the whole request.  The resulting
        # tier is broadcast to every layer, while K and V receive different
        # physical bit widths from the tier's pair.
        flat_ids = []
        nan_count = 0
        inf_count = 0
        if layout == "token":
            tier_ids, nan_count, inf_count = _rank_to_bucket_ids(
                importance_tensor.tolist(), config, forced_bf16_mask
            )
            tier_rows = [tier_ids] * (num_layers * 2)
        else:
            tier_rows = []
            for layer_idx in range(num_layers):
                for kv_idx in range(2):
                    tier_ids, nan_c, inf_c = _rank_to_bucket_ids(
                        importance_tensor[layer_idx, kv_idx].tolist(),
                        config,
                        forced_bf16_mask,
                    )
                    tier_rows.append(tier_ids)
                    nan_count += nan_c
                    inf_count += inf_c
        for row_index, tier_ids in enumerate(tier_rows):
            pair_bucket_ids = _kv_separate_bucket_ids(tier_ids, config)
            kv_index = row_index % 2
            # The wire map is layer-major, then K/V-major, then token-major.
            # Select one plane from the interleaved K/V tier mapping without
            # re-ranking the request or changing the physical layout.
            for token_offset in range(chunk_start, chunk_end):
                flat_ids.append(pair_bucket_ids[2 * token_offset + kv_index])
        chunk_bucket_ids = bytes(flat_ids)
        output_layout = "layer_kv_token"
    elif layout == "token":
        bucket_ids, nan_count, inf_count = _rank_to_bucket_ids(
            importance_tensor.tolist(), config, forced_bf16_mask
        )
        chunk_bucket_ids = bytes(bucket_ids[chunk_start:chunk_end])
    else:
        flat_ids: list[int] = []
        nan_count = 0
        inf_count = 0
        for layer_idx in range(num_layers):
            for kv_idx in range(2):
                ids, nan_c, inf_c = _rank_to_bucket_ids(
                    importance_tensor[layer_idx, kv_idx].tolist(),
                    config,
                    forced_bf16_mask,
                )
                flat_ids.extend(ids[chunk_start:chunk_end])
                nan_count += nan_c
                inf_count += inf_c
        chunk_bucket_ids = bytes(flat_ids)

    model_fp = model_fingerprint(model_name, num_layers, num_kv_heads, head_dim)
    parallel_fp = parallel_fingerprint(world_size, worker_id)
    plan = MaKVQuantPlan(
        protocol_version=1,
        importance_layout=output_layout,
        token_count=token_count,
        chunk_start=chunk_start,
        chunk_length=chunk_end - chunk_start,
        bucket_bits=config.bucket_bits,
        bucket_ids=chunk_bucket_ids,
        original_shape=original_shape,
        original_strides=original_strides,
        original_dtype=original_dtype,
        token_dim=token_dim,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        quant_granularity=config.quant_granularity,
        scale_dtype=config.scale_dtype,
        model_fingerprint=model_fp,
        parallel_fingerprint=parallel_fp,
        checksum=0,
        nan_protected_count=nan_count,
        inf_protected_count=inf_count,
        precision_scheme=config.precision_scheme,
    )
    return replace_quant_plan_checksum(plan)


def replace_quant_plan_checksum(plan: MaKVQuantPlan) -> MaKVQuantPlan:
    """Return a plan with its layout checksum recomputed."""
    from dataclasses import replace

    return replace(plan, checksum=compute_quant_plan_checksum(plan))


SCOUTRANK_PLAN_SCHEMA = "scoutrank_block_precision_v1"
SCOUTRANK_STRATEGY = "k2_risk_monotone_four_tier_v1"
SCOUTRANK_V3_FUNCTIONAL_PLAN_SCHEMA = "scoutrank_v3_functional_token_precision_v1"
SCOUTRANK_V3_FUNCTIONAL_STRATEGY = "scoutrank_v3_functional_disturbance_v1"
SCOUTRANK_PRECISION_BITS = {
    "BF16": (16, 16),
    "K8V4": (8, 4),
    "K4V2": (4, 2),
    "K2V2": (2, 2),
}


def prompt_token_hash(token_ids: list[int]) -> str:
    """Hash token IDs exactly as the frozen ScoutRank scorer does."""
    encoded = ",".join(str(int(value)) for value in token_ids).encode()
    return hashlib.sha256(encoded).hexdigest()


def _scoutrank_operational_payload(plan: dict[str, Any]) -> dict[str, Any]:
    blocks = sorted(plan["blocks"], key=lambda row: int(row["block_id"]))
    return {
        "strategy_version": plan["strategy_version"],
        "deployment_status": plan["deployment_status"],
        "score_precision": plan.get("score_precision", "K2V2"),
        "proxy_variant": plan.get("proxy_variant", "norm_upper_bound"),
        "eligible_block_count": int(plan["eligible_block_count"]),
        "blocks": [
            {
                "block_id": int(row["block_id"]),
                "eligible": bool(row["eligible"]),
                "rank": None if row.get("rank") is None else int(row["rank"]),
                "token_start": int(row["token_start"]),
                "token_end": int(row["token_end"]),
                "valid_tokens": int(row["valid_tokens"]),
                "stored_tokens": int(row["stored_tokens"]),
                "precision": str(row["precision"]),
                "estimated_bytes": int(row["estimated_bytes"]),
                "actual_bytes": int(row["actual_bytes"]),
            }
            for row in blocks
        ],
        "estimated_total_bytes": int(plan["estimated_total_bytes"]),
        "actual_total_bytes": int(plan["actual_total_bytes"]),
    }


def _validate_scoutrank_precision_plan(
    payload: dict[str, Any],
    *,
    token_count: int,
    actual_prompt_token_hash: str,
    config: MaKVConfig,
) -> tuple[list[dict[str, Any]], str]:
    if payload.get("schema_version") != SCOUTRANK_PLAN_SCHEMA:
        raise ValueError("unsupported ScoutRank precision plan schema")
    if payload.get("status") != "success":
        raise ValueError("ScoutRank precision plan status is not success")
    if payload.get("strategy_version") != SCOUTRANK_STRATEGY:
        raise ValueError("unsupported ScoutRank precision plan strategy")
    if payload.get("deployment_status") != "shadow":
        raise ValueError(
            "ScoutRank precision plan must retain deployment_status=shadow"
        )
    if not config.allow_scoutrank_shadow_plan:
        raise ValueError(
            "ScoutRank precision plan is shadow-only; set "
            "makv_allow_scoutrank_shadow_plan=true to consume it explicitly"
        )
    if payload.get("repeat_exact") is not True:
        raise ValueError("ScoutRank precision plan requires repeat_exact=true")
    if int(payload.get("token_count", -1)) != token_count:
        raise ValueError("ScoutRank precision plan token_count mismatch")
    if int(payload.get("block_size", -1)) != 32:
        raise ValueError("frozen ScoutRank precision plan requires block_size=32")
    if not actual_prompt_token_hash:
        raise ValueError("LMCache did not compute a prompt token hash")
    if payload.get("prompt_token_hash") != actual_prompt_token_hash:
        raise ValueError("ScoutRank precision plan prompt token hash mismatch")

    plan = payload.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("ScoutRank precision plan is missing plan object")
    for key in ("strategy_version", "deployment_status"):
        if plan.get(key) != payload.get(key):
            raise ValueError(f"ScoutRank precision plan {key} mismatch")
    blocks = sorted(plan.get("blocks", []), key=lambda row: int(row["block_id"]))
    expected_block_count = math.ceil(token_count / 32)
    if len(blocks) != expected_block_count:
        raise ValueError("ScoutRank block count does not match 32-token geometry")
    precision_vector = payload.get("precision_vector")
    if not isinstance(precision_vector, list) or len(precision_vector) != len(blocks):
        raise ValueError("ScoutRank precision_vector length mismatch")
    if [row.get("precision") for row in blocks] != precision_vector:
        raise ValueError("ScoutRank precision_vector does not match blocks")

    cursor = 0
    eligible = 0
    actual_total = 0
    estimated_total = 0
    counts = {precision: 0 for precision in SCOUTRANK_PRECISION_BITS}
    for expected_id, row in enumerate(blocks):
        if int(row["block_id"]) != expected_id:
            raise ValueError("ScoutRank block IDs must be contiguous from zero")
        start = int(row["token_start"])
        end = int(row["token_end"])
        valid = int(row["valid_tokens"])
        stored = int(row["stored_tokens"])
        precision = str(row["precision"])
        expected_start = expected_id * 32
        expected_valid = min(32, token_count - expected_start)
        if (
            start != cursor
            or start != expected_start
            or end - start != valid
            or valid != expected_valid
            or stored != 32
        ):
            raise ValueError("ScoutRank block token geometry is invalid")
        if precision not in SCOUTRANK_PRECISION_BITS:
            raise ValueError(f"unsupported ScoutRank precision {precision!r}")
        cursor = end
        eligible += int(bool(row["eligible"]))
        actual_total += int(row["actual_bytes"])
        estimated_total += int(row["estimated_bytes"])
        counts[precision] += 1
    if cursor != token_count:
        raise ValueError("ScoutRank blocks do not exactly cover request tokens")
    if eligible != int(plan.get("eligible_block_count", -1)):
        raise ValueError("ScoutRank eligible block count mismatch")
    ranked = sorted(
        (row for row in blocks if bool(row["eligible"])),
        key=lambda row: int(row["rank"]),
    )
    if [int(row["rank"]) for row in ranked] != list(range(1, eligible + 1)):
        raise ValueError("ScoutRank eligible ranks must be contiguous from one")
    cut10 = math.floor(0.10 * eligible)
    cut20 = math.floor(0.20 * eligible)
    cut80 = math.floor(0.80 * eligible)
    for rank, row in enumerate(ranked, start=1):
        expected = (
            "BF16"
            if rank <= cut10
            else "K8V4"
            if rank <= cut20
            else "K4V2"
            if rank <= cut80
            else "K2V2"
        )
        if row["precision"] != expected:
            raise ValueError("ScoutRank precision assignment violates frozen quota")
    if actual_total != int(plan.get("actual_total_bytes", -1)):
        raise ValueError("ScoutRank actual byte total mismatch")
    if estimated_total != int(plan.get("estimated_total_bytes", -1)):
        raise ValueError("ScoutRank estimated byte total mismatch")
    if actual_total != int(payload.get("actual_bytes", -1)):
        raise ValueError("ScoutRank outer actual byte total mismatch")
    if estimated_total != int(payload.get("estimated_bytes", -1)):
        raise ValueError("ScoutRank outer estimated byte total mismatch")
    if plan.get("precision_counts") != counts:
        raise ValueError("ScoutRank precision quota counts mismatch")

    digest = hashlib.sha256(
        json.dumps(
            _scoutrank_operational_payload(plan),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if digest != payload.get("plan_hash") or digest != plan.get("plan_hash"):
        raise ValueError("ScoutRank operational plan hash mismatch")
    return blocks, digest


def _validate_scoutrank_v3_functional_precision_plan(
    payload: dict[str, Any],
    *,
    token_count: int,
    actual_prompt_token_hash: str,
    config: MaKVConfig,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[list[str], str]:
    """Validate an explicit v3 token MCKP assignment without re-ranking it."""
    if payload.get("schema_version") != SCOUTRANK_V3_FUNCTIONAL_PLAN_SCHEMA:
        raise ValueError("unsupported ScoutRank-v3 functional plan schema")
    if payload.get("status") != "success":
        raise ValueError("ScoutRank-v3 functional plan status is not success")
    if payload.get("strategy_version") != SCOUTRANK_V3_FUNCTIONAL_STRATEGY:
        raise ValueError("unsupported ScoutRank-v3 functional plan strategy")
    if payload.get("deployment_status") != "experimental":
        raise ValueError("ScoutRank-v3 functional plan must be experimental")
    if not config.allow_scoutrank_shadow_plan:
        raise ValueError(
            "ScoutRank-v3 functional plan requires "
            "makv_allow_scoutrank_shadow_plan=true"
        )
    if int(payload.get("token_count", -1)) != token_count:
        raise ValueError("ScoutRank-v3 functional plan token_count mismatch")
    if not actual_prompt_token_hash:
        raise ValueError("LMCache did not compute a prompt token hash")
    if payload.get("prompt_token_hash") != actual_prompt_token_hash:
        raise ValueError("ScoutRank-v3 functional plan prompt token hash mismatch")
    if payload.get("makv_scale_dtype") != config.scale_dtype:
        raise ValueError("ScoutRank-v3 functional plan scale_dtype mismatch")
    if payload.get("makv_quant_granularity") != config.quant_granularity:
        raise ValueError(
            "ScoutRank-v3 functional plan quantization granularity mismatch"
        )
    expected_layout = {
        "makv_num_layers": num_layers,
        "makv_num_kv_heads": num_kv_heads,
        "makv_head_dim": head_dim,
        "makv_protect_prefix_tokens": config.protect_prefix_tokens,
        "makv_protect_tail_tokens": config.protect_tail_tokens,
    }
    for field, expected in expected_layout.items():
        if int(payload.get(field, -1)) != expected:
            raise ValueError(
                f"ScoutRank-v3 functional plan {field} mismatch"
            )
    precision_by_token = payload.get("precision_by_token")
    if (
        not isinstance(precision_by_token, list)
        or len(precision_by_token) != token_count
        or any(value not in SCOUTRANK_PRECISION_BITS for value in precision_by_token)
    ):
        raise ValueError("ScoutRank-v3 functional precision_by_token is invalid")
    token_status = payload.get("token_status")
    if token_status is not None:
        if not isinstance(token_status, list) or len(token_status) != token_count:
            raise ValueError("ScoutRank-v3 functional token_status is invalid")
        for token_index, status in enumerate(token_status):
            if not isinstance(status, dict):
                raise ValueError("ScoutRank-v3 functional token_status is invalid")
            if not bool(status.get("valid_mask", True)):
                if status.get("forced_precision") != "BF16":
                    raise ValueError(
                        "ScoutRank-v3 invalid token status must force BF16"
                    )
                if precision_by_token[token_index] != "BF16":
                    raise ValueError(
                        "ScoutRank-v3 invalid token cannot use a quantized precision"
                    )
    operational = {
        "schema_version": SCOUTRANK_V3_FUNCTIONAL_PLAN_SCHEMA,
        "strategy_version": SCOUTRANK_V3_FUNCTIONAL_STRATEGY,
        "token_count": token_count,
        "prompt_token_hash": actual_prompt_token_hash,
        "precision_by_token": precision_by_token,
        "makv_scale_dtype": config.scale_dtype,
        "makv_quant_granularity": config.quant_granularity,
        **expected_layout,
    }
    digest = hashlib.sha256(
        json.dumps(operational, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return [str(value) for value in precision_by_token], digest


def _build_chunk_quant_plan_from_v3_functional_precision_plan(
    *,
    precision_plan: dict[str, Any],
    actual_prompt_token_hash: str,
    chunk_start: int,
    chunk_end: int,
    original_shape: tuple[int, ...],
    original_strides: tuple[int, ...],
    original_dtype: str,
    token_dim: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    model_name: str,
    world_size: int,
    worker_id: int,
    config: MaKVConfig,
    request_token_count: Optional[int] = None,
) -> MaKVQuantPlan:
    """Slice an explicit v3 MCKP assignment into MaKV's physical buckets."""
    token_count = (
        int(request_token_count)
        if request_token_count is not None
        else int(original_shape[token_dim])
    )
    if not (0 <= chunk_start <= chunk_end <= token_count):
        raise ValueError("invalid MaKV chunk range for ScoutRank-v3 functional plan")
    if tuple(config.bucket_bits) != (16, 8, 4, 2):
        raise ValueError(
            "ScoutRank-v3 functional plans require makv_bucket_bits=[16,8,4,2]"
        )
    if config.precision_scheme != KV_SEPARATE_4TIER_SCHEME:
        raise ValueError(
            "ScoutRank-v3 functional plans require "
            "makv_precision_scheme=kv_separate_4tier"
        )
    if config.quant_granularity != "per_token_head":
        raise ValueError(
            "ScoutRank-v3 functional plans require "
            "makv_quant_granularity=per_token_head"
        )
    precision_by_token, source_hash = _validate_scoutrank_v3_functional_precision_plan(
        precision_plan,
        token_count=token_count,
        actual_prompt_token_hash=actual_prompt_token_hash,
        config=config,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    try:
        raw_dtype = getattr(torch, original_dtype.replace("torch.", ""))
        expected_raw_dtype_bytes = torch.empty((), dtype=raw_dtype).element_size()
    except (AttributeError, TypeError) as error:
        raise ValueError(
            f"ScoutRank-v3 functional plan has unsupported dtype {original_dtype!r}"
        ) from error
    if int(precision_plan.get("raw_dtype_bytes", -1)) != expected_raw_dtype_bytes:
        raise ValueError("ScoutRank-v3 functional plan raw dtype byte-width mismatch")
    # Direct token assignments bypass the rank mapper, so apply the same MaKV
    # prefix/tail safety policy before emitting physical bucket IDs.
    prefix = min(config.protect_prefix_tokens, token_count)
    tail = min(config.protect_tail_tokens, max(0, token_count - prefix))
    effective_precision = list(precision_by_token)
    for token_index in range(prefix):
        effective_precision[token_index] = "BF16"
    for token_index in range(token_count - tail, token_count):
        effective_precision[token_index] = "BF16"
    source_hash = hashlib.sha256(
        json.dumps(
            {
                "source_hash": source_hash,
                "protect_prefix_tokens": prefix,
                "protect_tail_tokens": tail,
                "effective_precision_by_token": effective_precision,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    bit_to_bucket = {bit: index for index, bit in enumerate(config.bucket_bits)}
    bucket_ids = bytearray()
    for _layer in range(num_layers):
        for kv_index in range(2):
            for precision in effective_precision[chunk_start:chunk_end]:
                bucket_ids.append(
                    bit_to_bucket[SCOUTRANK_PRECISION_BITS[precision][kv_index]]
                )
    model_fp = model_fingerprint(model_name, num_layers, num_kv_heads, head_dim)
    parallel_fp = parallel_fingerprint(world_size, worker_id)
    checksum_payload = json.dumps(
        {
            "layout": "layer_kv_token",
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "bucket_bits": list(config.bucket_bits),
            "bucket_ids": list(bucket_ids),
            "shape": list(original_shape),
            "strides": list(original_strides),
            "dtype": original_dtype,
            "model_fp": model_fp,
            "parallel_fp": parallel_fp,
            "source_plan_hash": source_hash,
            "prompt_token_hash": actual_prompt_token_hash,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return MaKVQuantPlan(
        protocol_version=1,
        importance_layout="layer_kv_token",
        token_count=token_count,
        chunk_start=chunk_start,
        chunk_length=chunk_end - chunk_start,
        bucket_bits=tuple(config.bucket_bits),
        bucket_ids=bytes(bucket_ids),
        original_shape=original_shape,
        original_strides=original_strides,
        original_dtype=original_dtype,
        token_dim=token_dim,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        quant_granularity=config.quant_granularity,
        scale_dtype=config.scale_dtype,
        model_fingerprint=model_fp,
        parallel_fingerprint=parallel_fp,
        checksum=int.from_bytes(
            hashlib.sha256(checksum_payload).digest()[:4], "little"
        ),
        source_plan_hash=source_hash,
        source_strategy=SCOUTRANK_V3_FUNCTIONAL_STRATEGY,
        prompt_token_hash=actual_prompt_token_hash,
        precision_plan_schema=SCOUTRANK_V3_FUNCTIONAL_PLAN_SCHEMA,
        precision_scheme=config.precision_scheme,
    )


def build_chunk_quant_plan_from_precision_plan(
    *,
    precision_plan: dict[str, Any],
    actual_prompt_token_hash: str,
    chunk_start: int,
    chunk_end: int,
    original_shape: tuple[int, ...],
    original_strides: tuple[int, ...],
    original_dtype: str,
    token_dim: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    model_name: str,
    world_size: int,
    worker_id: int,
    config: MaKVConfig,
    request_token_count: Optional[int] = None,
) -> MaKVQuantPlan:
    """Slice one validated request-level ScoutRank plan for an LMCache chunk."""
    if precision_plan.get("schema_version") == SCOUTRANK_V3_FUNCTIONAL_PLAN_SCHEMA:
        return _build_chunk_quant_plan_from_v3_functional_precision_plan(
            precision_plan=precision_plan,
            actual_prompt_token_hash=actual_prompt_token_hash,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            original_shape=original_shape,
            original_strides=original_strides,
            original_dtype=original_dtype,
            token_dim=token_dim,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            model_name=model_name,
            world_size=world_size,
            worker_id=worker_id,
            config=config,
            request_token_count=request_token_count,
        )
    token_count = int(request_token_count)
    if not (0 <= chunk_start <= chunk_end <= token_count):
        raise ValueError("invalid MaKV chunk range for ScoutRank precision plan")
    blocks, source_hash = _validate_scoutrank_precision_plan(
        precision_plan,
        token_count=token_count,
        actual_prompt_token_hash=actual_prompt_token_hash,
        config=config,
    )
    required_bits = (16, 8, 4, 2)
    if tuple(config.bucket_bits) != required_bits:
        raise ValueError(
            "ScoutRank four-tier plans require makv_bucket_bits=[16,8,4,2]"
        )
    precision_by_token = [""] * token_count
    for row in blocks:
        start = int(row["token_start"])
        end = int(row["token_end"])
        precision_by_token[start:end] = [str(row["precision"])] * (end - start)
    bit_to_bucket = {bit: index for index, bit in enumerate(required_bits)}
    bucket_ids = bytearray()
    for _layer in range(num_layers):
        for kv_index in range(2):
            for precision in precision_by_token[chunk_start:chunk_end]:
                bits = SCOUTRANK_PRECISION_BITS[precision][kv_index]
                bucket_ids.append(bit_to_bucket[bits])

    model_fp = model_fingerprint(model_name, num_layers, num_kv_heads, head_dim)
    parallel_fp = parallel_fingerprint(world_size, worker_id)
    checksum_payload = json.dumps(
        {
            "layout": "layer_kv_token",
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "bucket_bits": list(required_bits),
            "bucket_ids": list(bucket_ids),
            "shape": list(original_shape),
            "strides": list(original_strides),
            "dtype": original_dtype,
            "model_fp": model_fp,
            "parallel_fp": parallel_fp,
            "source_plan_hash": source_hash,
            "prompt_token_hash": actual_prompt_token_hash,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    checksum = int.from_bytes(hashlib.sha256(checksum_payload).digest()[:4], "little")
    return MaKVQuantPlan(
        protocol_version=1,
        importance_layout="layer_kv_token",
        token_count=token_count,
        chunk_start=chunk_start,
        chunk_length=chunk_end - chunk_start,
        bucket_bits=required_bits,
        bucket_ids=bytes(bucket_ids),
        original_shape=original_shape,
        original_strides=original_strides,
        original_dtype=original_dtype,
        token_dim=token_dim,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        quant_granularity=config.quant_granularity,
        scale_dtype=config.scale_dtype,
        model_fingerprint=model_fp,
        parallel_fingerprint=parallel_fp,
        checksum=checksum,
        source_plan_hash=source_hash,
        source_strategy=SCOUTRANK_STRATEGY,
        prompt_token_hash=actual_prompt_token_hash,
        precision_plan_schema=SCOUTRANK_PLAN_SCHEMA,
    )
