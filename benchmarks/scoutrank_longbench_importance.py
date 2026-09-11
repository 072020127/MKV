#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Precompute real ScoutRank token scores for the LongBench runner.

The output is intentionally a small JSON artifact keyed by the exact prompt
token hash. It can be passed to ``longbench_makv_cachegen.py`` through
``--importance-file`` or, for explicit v3 MCKP assignments,
``--precision-plan-file``; it is never inferred from mutable global state.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    # When this file is launched directly, Python puts LMCache/benchmarks on
    # sys.path.  Prefer the sibling module so vllm/benchmarks cannot shadow
    # the LMCache benchmark namespace.
    from longbench_makv_cachegen import (
        load_examples,
        prompt_ids_with_last_user_span,
        prompt_token_hash,
    )
except ImportError:
    from benchmarks.longbench_makv_cachegen import (
        load_examples,
        prompt_ids_with_last_user_span,
        prompt_token_hash,
    )
from experiments.scoutrank_transfer.observer import (  # noqa: E402
    FunctionalDisturbanceObserver,
    FunctionalDisturbanceV31Observer,
    ProductionMaKVErrorObserver,
    VectorizedMaKVErrorObserver,
)
from makv_scoutrank import (  # noqa: E402
    FAST_1_EXACT_SCALAR_D22_SCORING_VERSION,
    FAST_D22_SCORING_VERSION,
    FastExactScalarD22Observer,
    FastD22Observer,
    ScoutForwardAdapter,
    ScoutRankConfig,
    ScoutRankScorer,
    allocate_functional_token_mckp,
    fast_1_exact_scalar_d22_cuda_available,
    fast_d22_cuda_available,
)

_FUNCTIONAL_SCORING_VERSIONS = frozenset(
    {
        "v3-functional-disturbance",
        "v3.1-projected",
        "v3.1-probe-normalized",
    }
)
_FAST_D22_SCORING_VERSIONS = frozenset(
    {FAST_D22_SCORING_VERSION, FAST_1_EXACT_SCALAR_D22_SCORING_VERSION}
)
ATTENTION_NORMALIZED_SCORING_VERSION = "v3.2_attention_normalized"


def _score_semantics(scoring_version: str) -> str:
    """Describe the exact score variant stored in an importance artifact."""
    if scoring_version == ATTENTION_NORMALIZED_SCORING_VERSION:
        return (
            "ScoutRank-v3.2 strict-future attention mass normalized by valid "
            "(layer, head, query) observation count; higher means more important"
        )
    if scoring_version == "v3.1-projected":
        return (
            "ScoutRank-v3.1 projected attention-output damage; "
            "higher means more important"
        )
    if scoring_version == "v3.1-probe-normalized":
        return (
            "ScoutRank-v3.1 probe-normalized attention-output damage; "
            "higher means more important"
        )
    if scoring_version == "v3-functional-disturbance":
        return (
            "ScoutRank-v3.0 D22 functional attention-output damage; "
            "higher means more important"
        )
    if scoring_version == FAST_D22_SCORING_VERSION:
        return (
            "ScoutRank-v3_fast_d22 online K2V2 functional attention-output "
            "damage; higher means more important"
        )
    if scoring_version == FAST_1_EXACT_SCALAR_D22_SCORING_VERSION:
        return (
            "ScoutRank-v3_fast_1_exact_scalar_d22 online K2V2 exact scalar "
            "D22 functional attention-output damage; higher means more important"
        )
    return "ScoutRank damage_22; higher means more important"


def _load_model(path: str, device: torch.device, dtype: torch.dtype) -> Any:
    """Load the frozen ScoutRank model on the requested device."""
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return model.to(device).eval()


def _parse_anchor_layers(
    value: str | None, mode: str, exit_layer: int | None = None
) -> tuple[int, ...]:
    """Parse the anchor policy, using a cheaper two-layer fast default."""
    if value:
        layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        if not layers:
            raise ValueError("--anchor-layers must contain at least one layer")
        return layers
    if exit_layer is not None:
        layers = tuple(layer for layer in (7, 14, 21, 28) if layer <= exit_layer)
        if not layers:
            raise ValueError(
                "--exit-layer must be at least 7 when --anchor-layers is omitted"
            )
        return layers
    if mode == "fast":
        return (14, 28)
    return (7, 14, 21, 28)


def _env_bool(name: str, default: bool) -> bool:
    """Read a conventional 0/1 scout environment switch."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean-like value")


def _build_scout_runtime(
    model: Any,
    *,
    mode: str,
    observer_backend: str,
    observer_token_chunk_size: int,
    anchor_layers: tuple[int, ...],
    exit_layer: int | None,
    scoring_version: str = "v2",
    v3_num_probes: int = 16,
    v3_use_last_user_span: bool = True,
    v3_tail_probes: int = 8,
    v3_probe_selection: str = "priority",
    v3_makv_scale_dtype: str = "float16",
    v3_protect_prefix_tokens: int = 4,
    v3_protect_tail_tokens: int = 16,
    v3_exact_allocator_max_states: int = 200_000,
    return_diagnostics: bool | None = None,
    write_artifact: bool | None = None,
    fast_d22_qk_dtype: str = "float32",
    fast_d22_profile_timing: bool = False,
    fast_d22_use_cuda_kernel: bool = True,
) -> tuple[ScoutForwardAdapter | None, ScoutRankScorer, ScoutRankConfig]:
    """Build reusable ScoutRank state for all prompts in one artifact."""
    fast_d22 = scoring_version in _FAST_D22_SCORING_VERSIONS
    cfg = ScoutRankConfig(
        mode=mode,
        anchor_layers=anchor_layers,
        exit_layer=exit_layer,
        block_size=32,
        include_token_scores=False,
        target_layers=36,
        target_kv_heads=8,
        target_head_dim=128,
        output_attentions=False,
        use_cache=False,
        functional_disturbance_enabled=(
            scoring_version in _FUNCTIONAL_SCORING_VERSIONS or fast_d22
        ),
        scoring_profile="online_d22" if fast_d22 else "default",
        precisions_to_score=("K2V2",)
        if fast_d22
        else ("K2V2", "K4V2", "K8V4"),
        return_diagnostics=(not fast_d22)
        if return_diagnostics is None
        else return_diagnostics,
        write_artifact=(not fast_d22)
        if write_artifact is None
        else write_artifact,
        fast_d22_qk_dtype=fast_d22_qk_dtype,
        fast_d22_profile_timing=fast_d22_profile_timing,
        fast_d22_use_cuda_kernel=fast_d22_use_cuda_kernel,
        v3_num_probes=v3_num_probes,
        v3_use_last_user_span=v3_use_last_user_span,
        v3_tail_probes=v3_tail_probes,
        v3_probe_selection=v3_probe_selection,
        v3_token_chunk_size=observer_token_chunk_size,
        v3_makv_scale_dtype=v3_makv_scale_dtype,
        v3_protect_prefix_tokens=v3_protect_prefix_tokens,
        v3_protect_tail_tokens=v3_protect_tail_tokens,
        exact_allocator_max_states=v3_exact_allocator_max_states,
    )
    if scoring_version == ATTENTION_NORMALIZED_SCORING_VERSION:
        return None, ScoutRankScorer(cfg), cfg
    if scoring_version == "v3-functional-disturbance":
        observer = FunctionalDisturbanceObserver(cfg)
    elif fast_d22:
        model_device = next(model.parameters()).device
        if (
            model_device.type == "cuda"
            and fast_d22_use_cuda_kernel
            and not (
                fast_1_exact_scalar_d22_cuda_available()
                if scoring_version == FAST_1_EXACT_SCALAR_D22_SCORING_VERSION
                else fast_d22_cuda_available()
            )
        ):
            raise RuntimeError(
                f"{scoring_version} CUDA kernel is unavailable; build/import "
                "lmcache.cuda_ops or set fast_d22_use_cuda_kernel=False"
            )
        observer = (
            FastExactScalarD22Observer(cfg)
            if scoring_version == FAST_1_EXACT_SCALAR_D22_SCORING_VERSION
            else FastD22Observer(cfg)
        )
    elif scoring_version in {"v3.1-projected", "v3.1-probe-normalized"}:
        observer = FunctionalDisturbanceV31Observer(cfg)
    elif observer_backend == "vectorized":
        observer = VectorizedMaKVErrorObserver(observer_token_chunk_size)
    elif observer_backend == "production":
        observer = ProductionMaKVErrorObserver()
    else:
        raise ValueError(f"unsupported observer backend: {observer_backend}")
    return ScoutForwardAdapter(model, cfg, observer), ScoutRankScorer(cfg), cfg


def _score_prompt(
    ids: list[int],
    *,
    device: torch.device,
    adapter: ScoutForwardAdapter | None,
    scorer: ScoutRankScorer,
    scoring_version: str = "v2",
    last_user_span: tuple[int, int] | None = None,
    mckp_budget_bytes: int | None = None,
    return_details: bool = False,
    attention_query_chunk_size: int = 512,
    attention_model: Any | None = None,
) -> list[float] | tuple[list[float], dict[str, Any]]:
    """Run one prompt through reusable ScoutRank state."""
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    if scoring_version == ATTENTION_NORMALIZED_SCORING_VERSION:
        if attention_model is None:
            raise RuntimeError("v3.2 attention scoring requires the scout model")
        try:
            from benchmarks.scoutrank_attention_overlap import (
                collect_attention_scores,
            )
        except ImportError:
            from LMCache.benchmarks.scoutrank_attention_overlap import (
                collect_attention_scores,
            )
        variants, attention_metadata = collect_attention_scores(
            attention_model,
            input_ids,
            "future_both",
            query_chunk_size=attention_query_chunk_size,
            layer_mode="last",
        )
        if not isinstance(variants, dict):
            raise RuntimeError("v3.2 attention collection did not return variants")
        scores = torch.tensor(variants["A_norm"], dtype=torch.float32)
        visible_count = [int(value) for value in variants["visible_count"]]
        details = {
            "attention": {
                "A_raw": [float(value) for value in variants["A_raw"]],
                "A_norm": [float(value) for value in variants["A_norm"]],
                "visible_count": visible_count,
                "valid_mask": [count > 0 for count in visible_count],
                "forced_precision_by_token": [
                    "BF16" if count == 0 else None for count in visible_count
                ],
                "reason_by_token": [
                    "NO_FUTURE_PROBE" if count == 0 else None
                    for count in visible_count
                ],
                "token_status": [
                    {
                        "valid_mask": count > 0,
                        "forced_precision": "BF16" if count == 0 else None,
                        "reason": "NO_FUTURE_PROBE" if count == 0 else None,
                    }
                    for count in visible_count
                ],
                "metadata": attention_metadata,
            }
        }
        if scores.numel() != len(ids):
            raise RuntimeError("v3.2 attention score count does not match prompt")
        values = scores.tolist()
        return (values, details) if return_details else values
    if adapter is None:
        raise RuntimeError(f"missing ScoutRank adapter for {scoring_version}")
    forward = adapter.forward_once(
        input_ids,
        last_user_span=last_user_span,
        # v3's primary score intentionally contains no legacy NLL/drift/
        # novelty feature, so avoid this unrelated LM-head work entirely.
        compute_nll=scoring_version == "v2" and adapter.cfg.mode != "fast",
    )
    if forward.summary is None:
        raise RuntimeError("ScoutRank forward did not produce a summary")
    summary = forward.summary
    details: dict[str, Any] = {}
    if scoring_version in _FAST_D22_SCORING_VERSIONS:
        if mckp_budget_bytes is not None:
            raise ValueError("v3_fast_d22 does not support MCKP allocation")
        functional = summary.functional_damage
        if functional is None:
            raise RuntimeError("v3_fast_d22 did not produce functional damage")
        scores = functional.d22.detach().cpu()
        # The fast profile intentionally returns no D42/D84 tensors or
        # per-token artifact payload.
        details = {}
    elif scoring_version in _FUNCTIONAL_SCORING_VERSIONS:
        functional = summary.functional_damage
        if functional is None:
            raise RuntimeError("v3 observer did not produce functional damage")
        if scoring_version == "v3.1-projected":
            scores = functional.d22_projected.detach().cpu()
        elif scoring_version == "v3.1-probe-normalized":
            scores = functional.d22_probe_normalized.detach().cpu()
        else:
            scores = functional.d22.detach().cpu()
        details = {
            "functional_damage": {
                name: tensor.detach().float().cpu().tolist()
                for name, tensor in functional.damage_tensors().items()
            },
            "functional_damage_by_precision": {
                precision: tensor.detach().float().cpu().tolist()
                for precision, tensor in functional.damage_by_precision.items()
            },
            "functional_metadata": dict(functional.metadata),
        }
        if mckp_budget_bytes is not None:
            allocation = allocate_functional_token_mckp(
                functional,
                adapter.cfg,
                budget_bytes=mckp_budget_bytes,
                raw_dtype_bytes=_model_kv_scalar_bytes(adapter.model),
            )
            details["mckp"] = {
                "precision_by_token": list(allocation.precision_by_token),
                "total_bytes": allocation.total_bytes,
                "budget_bytes": allocation.budget_bytes,
                "metadata": allocation.metadata,
            }
            details["precision_plan"] = allocation.to_precision_plan_payload(
                summary.token_ids
            )
    else:
        scores = scorer.token_importance_from_summaries(
            token_ids=summary.token_ids,
            valid_token_mask=summary.valid_token_mask,
            self_information_nll=summary.self_information_nll,
            representation_drift=summary.representation_drift,
            local_novelty=summary.local_novelty,
            task_relevance=summary.task_relevance,
            k_errors=summary.k_errors,
            v_errors=summary.v_errors,
        )
    if scores.numel() != len(ids):
        raise RuntimeError("ScoutRank token score count does not match prompt")
    if (
        scoring_version in _FUNCTIONAL_SCORING_VERSIONS
        or scoring_version in _FAST_D22_SCORING_VERSIONS
    ):
        # Positive infinity is an intentional MaKV safety signal for uncovered
        # or non-finite tokens. _rank_to_bucket_ids maps it to BF16 instead of
        # allowing it into a low-precision bucket.
        if bool(torch.isnan(scores).any() or torch.isneginf(scores).any()):
            raise ValueError("ScoutRank-v3 produced an invalid non-finite damage")
    elif not bool(torch.isfinite(scores).all()):
        raise ValueError("ScoutRank produced a non-finite token score")
    values = scores.tolist()
    return (values, details) if return_details else values


def _synchronize(device: torch.device) -> None:
    """Make GPU timing cover the complete ScoutRank forward/scoring work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    """Capture all CUDA allocator counters for one scored prompt."""
    if device.type != "cuda":
        return {
            "memory_allocated_bytes": 0,
            "peak_memory_allocated_bytes": 0,
            "memory_reserved_bytes": 0,
            "peak_memory_reserved_bytes": 0,
        }
    return {
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "peak_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _model_kv_scalar_bytes(model: Any) -> int:
    """Return the floating scalar width used by the frozen model's KV path."""
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.element_size()
    raise ValueError("ScoutRank model has no floating-point parameters")


def main() -> None:
    probe_selection_default = os.getenv("SCOUT_V3_PROBE_SELECTION", "priority")
    probe_count_from_env = os.getenv("SCOUT_V3_NUM_PROBES")
    tail_probe_count_from_env = os.getenv("SCOUT_V3_TAIL_PROBES")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Qwen3-0.6B ScoutRank model")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--task", default="hotpotqa")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--prompt-run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--mode", choices=("fast", "balanced"), default="balanced")
    parser.add_argument(
        "--scoring-version",
        choices=(
            "v2",
            "v3-functional-disturbance",
            "v3.1-projected",
            "v3.1-probe-normalized",
            FAST_D22_SCORING_VERSION,
            FAST_1_EXACT_SCALAR_D22_SCORING_VERSION,
            ATTENTION_NORMALIZED_SCORING_VERSION,
        ),
        default=os.getenv("SCOUT_SCORING_VERSION", "v2"),
        help=(
            "Keep v2 by default; v3 and the experimental exact-scalar D22 "
            "profile are explicit opt-ins."
        ),
    )
    parser.add_argument(
        "--observer-backend",
        choices=("vectorized", "production"),
        default="vectorized",
        help="Use GPU vectorized production-math errors or the legacy CPU round trip.",
    )
    parser.add_argument(
        "--observer-token-chunk-size",
        type=int,
        default=int(os.getenv("SCOUT_OBSERVER_TOKEN_CHUNK_SIZE", "4096")),
        help="Token chunk size for vectorized observer temporary tensors.",
    )
    parser.add_argument(
        "--attention-query-chunk-size",
        type=int,
        default=int(os.getenv("SCOUT_ATTENTION_QUERY_CHUNK_SIZE", "512")),
        help="Bounded query chunk for v3.2 attention-only scoring.",
    )
    parser.add_argument(
        "--v3-num-probes",
        type=int,
        default=(
            int(probe_count_from_env)
            if probe_count_from_env is not None
            else None
        ),
    )
    parser.add_argument(
        "--v3-use-last-user-span",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("SCOUT_V3_USE_LAST_USER_SPAN", True),
    )
    parser.add_argument(
        "--v3-tail-probes",
        type=int,
        default=(
            int(tail_probe_count_from_env)
            if tail_probe_count_from_env is not None
            else None
        ),
    )
    parser.add_argument(
        "--v3-probe-selection",
        choices=("priority", "mix32"),
        default=probe_selection_default,
        help=(
            "priority preserves the existing selection; mix32 uses 16 "
            "prompt-wide probes and 16 tail probes."
        ),
    )
    parser.add_argument(
        "--v3-mckp-budget-bytes",
        type=int,
        default=None,
        help="Optional token-level v3 MCKP budget; emits precision_by_token.",
    )
    parser.add_argument(
        "--v3-makv-scale-dtype",
        choices=("float16", "float32"),
        default=os.getenv("SCOUT_V3_MAKV_SCALE_DTYPE", "float16"),
        help="Must match makv_scale_dtype for fake-quant parity.",
    )
    parser.add_argument(
        "--v3-protect-prefix-tokens",
        type=int,
        default=int(os.getenv("SCOUT_V3_PROTECT_PREFIX_TOKENS", "4")),
        help="Must match makv_protect_prefix_tokens when using v3 MCKP.",
    )
    parser.add_argument(
        "--v3-protect-tail-tokens",
        type=int,
        default=int(os.getenv("SCOUT_V3_PROTECT_TAIL_TOKENS", "16")),
        help="Must match makv_protect_tail_tokens when using v3 MCKP.",
    )
    parser.add_argument(
        "--v3-exact-allocator-max-states",
        type=int,
        default=int(os.getenv("SCOUT_V3_EXACT_ALLOCATOR_MAX_STATES", "200000")),
        help="Operational exact-MCKP frontier limit; does not alter score or budget.",
    )
    parser.add_argument(
        "--anchor-layers",
        default=None,
        help=(
            "Comma-separated 1-based Scout model anchor layers; fast defaults to 14,28."
        ),
    )
    parser.add_argument(
        "--exit-layer",
        type=int,
        default=None,
        help="Optional Qwen3 early-exit depth; anchors must not exceed it.",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--return-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Return per-token diagnostics; defaults off for online D22 "
            "profiles."
        ),
    )
    parser.add_argument(
        "--write-artifact",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write the JSON artifact; defaults off for online D22 profiles.",
    )
    parser.add_argument(
        "--fast-d22-qk-dtype",
        choices=("float32", "bfloat16"),
        default=os.getenv("SCOUT_FAST_D22_QK_DTYPE", "float32"),
        help="QK accumulation input dtype for the opt-in online D22 profile.",
    )
    parser.add_argument(
        "--fast-d22-profile-timing",
        action="store_true",
        default=_env_bool("SCOUT_FAST_D22_PROFILE_TIMING", False),
        help="Record synchronized fast D22 observer stage CUDA events.",
    )
    parser.add_argument(
        "--fast-d22-use-cuda-kernel",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("SCOUT_FAST_D22_USE_CUDA_KERNEL", True),
        help="Use the prebuilt fused CUDA D22 kernel when available.",
    )
    args = parser.parse_args()
    if args.v3_num_probes is None:
        args.v3_num_probes = 32 if args.v3_probe_selection == "mix32" else 16
    if args.v3_tail_probes is None:
        args.v3_tail_probes = 16 if args.v3_probe_selection == "mix32" else 8

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    examples = load_examples(
        Path(args.dataset_path), args.task, args.limit, args.offset
    )
    device = torch.device(args.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = _load_model(args.model, device, dtype)
    anchor_layers = _parse_anchor_layers(args.anchor_layers, args.mode, args.exit_layer)
    adapter, scorer, cfg = _build_scout_runtime(
        model,
        mode=args.mode,
        observer_backend=args.observer_backend,
        observer_token_chunk_size=args.observer_token_chunk_size,
        anchor_layers=anchor_layers,
        exit_layer=args.exit_layer,
        scoring_version=args.scoring_version,
        v3_num_probes=args.v3_num_probes,
        v3_use_last_user_span=args.v3_use_last_user_span,
        v3_tail_probes=args.v3_tail_probes,
        v3_probe_selection=args.v3_probe_selection,
        v3_makv_scale_dtype=args.v3_makv_scale_dtype,
        v3_protect_prefix_tokens=args.v3_protect_prefix_tokens,
        v3_protect_tail_tokens=args.v3_protect_tail_tokens,
        v3_exact_allocator_max_states=args.v3_exact_allocator_max_states,
        return_diagnostics=args.return_diagnostics,
        write_artifact=args.write_artifact,
        fast_d22_qk_dtype=args.fast_d22_qk_dtype,
        fast_d22_profile_timing=args.fast_d22_profile_timing,
        fast_d22_use_cuda_kernel=args.fast_d22_use_cuda_kernel,
    )
    scores: dict[str, list[float]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    functional_results: dict[str, dict[str, Any]] = {}
    attention_results: dict[str, dict[str, Any]] = {}
    precision_plans: dict[str, dict[str, Any]] = {}
    score_times_ms: list[float] = []
    try:
        for index, example in enumerate(examples, start=1):
            ids, last_user_span = prompt_ids_with_last_user_span(
                tokenizer,
                example,
                args.prompt_run_id,
                enable_thinking=args.enable_thinking,
            )
            key = prompt_token_hash(ids)
            _synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            score_started = time.perf_counter()
            scored = _score_prompt(
                ids,
                device=device,
                adapter=adapter,
                scorer=scorer,
                scoring_version=args.scoring_version,
                last_user_span=last_user_span,
                mckp_budget_bytes=args.v3_mckp_budget_bytes,
                return_details=bool(cfg.return_diagnostics),
                attention_query_chunk_size=args.attention_query_chunk_size,
                attention_model=model,
            )
            if isinstance(scored, tuple):
                values, details = scored
            else:
                # Online D22 deliberately disables diagnostics and returns
                # only the score vector. Keep that low-overhead path distinct
                # from artifact-producing legacy scorers.
                values, details = scored, {}
            _synchronize(device)
            score_time_ms = (time.perf_counter() - score_started) * 1000.0
            memory = _memory_snapshot(device)
            scores[key] = values
            score_times_ms.append(score_time_ms)
            metadata[key] = {
                "example_id": example.example_id,
                "token_count": len(ids),
                "importance_layout": "token",
                "score_semantics": _score_semantics(args.scoring_version),
                "scoutrank_time_ms": score_time_ms,
                **memory,
                # Compatibility alias used by existing LongBench artifacts.
                "peak_memory_bytes": memory["peak_memory_allocated_bytes"],
                "scoring_mode": args.mode,
                "scoring_version": args.scoring_version,
                "allocation_strategy": (
                    "mckp_experimental"
                    if args.v3_mckp_budget_bytes is not None
                    else "d22_fixed"
                    if args.scoring_version in _FUNCTIONAL_SCORING_VERSIONS
                    or args.scoring_version in _FAST_D22_SCORING_VERSIONS
                    or args.scoring_version == ATTENTION_NORMALIZED_SCORING_VERSION
                    else None
                ),
                "mckp_status": (
                    "experimental_opt_in"
                    if args.v3_mckp_budget_bytes is not None
                    else "disabled_by_default"
                    if args.scoring_version in _FUNCTIONAL_SCORING_VERSIONS
                    or args.scoring_version in _FAST_D22_SCORING_VERSIONS
                    or args.scoring_version == ATTENTION_NORMALIZED_SCORING_VERSION
                    else None
                ),
                "observer_backend": args.observer_backend,
                "anchor_layers": list(anchor_layers),
                "exit_layer": args.exit_layer,
                "last_user_span": list(last_user_span)
                if last_user_span is not None
                else None,
            }
            if details:
                if "attention" in details:
                    attention_results[key] = details
                else:
                    functional_results[key] = details
                precision_plan = details.get("precision_plan")
                if isinstance(precision_plan, dict):
                    precision_plans[key] = precision_plan
            print(
                f"[{index}/{len(examples)}] id={example.example_id} "
                f"tokens={len(ids)} "
                f"scoutrank_ms={score_time_ms:.3f} "
                f"score_range=({min(values):.6g},{max(values):.6g})",
                flush=True,
            )
    finally:
        artifact_enabled = bool(cfg.write_artifact)
        del adapter, scorer, cfg, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    score_p95_ms = (
        sorted(score_times_ms)[max(0, (95 * len(score_times_ms) + 99) // 100 - 1)]
        if score_times_ms
        else None
    )
    payload = {
        "schema_version": 1,
        "task": args.task,
        "prompt_run_id": args.prompt_run_id,
        "scout_model": args.model,
        "enable_thinking": args.enable_thinking,
        "scoring_mode": args.mode,
        "scoring_version": args.scoring_version,
        "observer_backend": args.observer_backend,
        "observer_token_chunk_size": args.observer_token_chunk_size,
        "anchor_layers": list(anchor_layers),
        "exit_layer": args.exit_layer,
        "importance_layout": "token",
        "score_semantics": _score_semantics(args.scoring_version),
        "allocation_strategy": (
            "mckp_experimental"
            if args.v3_mckp_budget_bytes is not None
            else "d22_fixed"
            if args.scoring_version in _FUNCTIONAL_SCORING_VERSIONS
            or args.scoring_version in _FAST_D22_SCORING_VERSIONS
            or args.scoring_version == ATTENTION_NORMALIZED_SCORING_VERSION
            else None
        ),
        "mckp_status": (
            "experimental_opt_in"
            if args.v3_mckp_budget_bytes is not None
            else "disabled_by_default"
            if args.scoring_version in _FUNCTIONAL_SCORING_VERSIONS
            or args.scoring_version in _FAST_D22_SCORING_VERSIONS
            or args.scoring_version == ATTENTION_NORMALIZED_SCORING_VERSION
            else None
        ),
        "v3": {
            "num_probes": args.v3_num_probes,
            "use_last_user_span": args.v3_use_last_user_span,
            "tail_probes": args.v3_tail_probes,
            "probe_selection": args.v3_probe_selection,
            "mckp_budget_bytes": args.v3_mckp_budget_bytes,
            "makv_scale_dtype": args.v3_makv_scale_dtype,
            "protect_prefix_tokens": args.v3_protect_prefix_tokens,
            "protect_tail_tokens": args.v3_protect_tail_tokens,
        },
        "timing": {
            "scope": (
                "ScoutRank forward plus scoring, excluding model load and tokenization"
            ),
            "count": len(score_times_ms),
            "scoutrank_time_ms_total": sum(score_times_ms),
            "scoutrank_time_ms_mean": (
                statistics.fmean(score_times_ms) if score_times_ms else None
            ),
            "scoutrank_time_ms_median": (
                statistics.median(score_times_ms) if score_times_ms else None
            ),
            "scoutrank_time_ms_p95": score_p95_ms,
        },
        "metadata": metadata,
        "scores": scores,
    }
    if functional_results:
        payload["functional_results"] = functional_results
    if attention_results:
        payload["attention_results"] = attention_results
    if precision_plans:
        payload["precision_plans"] = precision_plans
    output = Path(args.output)
    if artifact_enabled:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {len(scores)} importance vectors to {output}")
    else:
        print(
            f"Artifact writing disabled for {args.scoring_version}; "
            f"scored {len(scores)} prompts"
        )


if __name__ == "__main__":
    main()
