#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare real-model ``v3_fast_d22`` output with frozen v3 D22.

The comparison uses the same explicit anchors, probes, token chunk size and
model weights.  It reports numerical D22 error, rank overlap, and a
deterministic token-level fixed-plan hash.  This is a validation tool only;
it never changes the production allocator or v3 artifacts.
"""

from __future__ import annotations

# Standard
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Any

# Third Party
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LMCache_ROOT = REPOSITORY_ROOT / "LMCache"
for _path in (REPOSITORY_ROOT, LMCache_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from scoutrank_fast_d22_profile import (
        _gpu_fixed_sort_allocation,
        _parse_layers,
        _parse_positive_ints,
    )
    from scoutrank_longbench_importance import _build_scout_runtime, _load_model
except ImportError:
    from benchmarks.scoutrank_fast_d22_profile import (  # type: ignore
        _gpu_fixed_sort_allocation,
        _parse_layers,
        _parse_positive_ints,
    )
    from benchmarks.scoutrank_longbench_importance import (  # type: ignore
        _build_scout_runtime,
        _load_model,
    )

from makv_scoutrank import FAST_D22_SCORING_VERSION  # noqa: E402


def _top_indices(values: list[float], fraction: float = 0.1) -> set[int]:
    """Return deterministic top-k indices, including fail-closed infinities."""
    count = min(len(values), max(1, int(len(values) * fraction + 0.999999)))
    ordered = sorted(
        range(len(values)), key=lambda index: (-float(values[index]), index)
    )
    return set(ordered[:count])


def _jaccard(left: set[int], right: set[int]) -> float:
    """Compute set Jaccard without a special empty-set convention."""
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _plan_hash(result: Any) -> tuple[str, bytes]:
    """Hash the deterministic benchmark fixed-ratio token assignment."""
    plan = _gpu_fixed_sort_allocation(result).detach().to(device="cpu")
    payload = plan.contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest(), payload


def _run_same_qkv_observers(
    fast_adapter: Any,
    frozen_adapter: Any,
    ids: torch.Tensor,
) -> tuple[Any, Any]:
    """Compare observers on exactly the same real-model Q/K/V tensors.

    A frozen v3 forward replaces every attention layer with its reference
    streaming implementation, while ``online_d22`` intentionally lets
    non-anchor layers use the model's native attention.  Comparing two full
    forwards would therefore include unrelated hidden-state drift before the
    anchor.  Capture the frozen forward's post-RoPE Q/K/V once, then replay
    those tensors through the fast observer.  This is an observer-equivalence
    test, not a claim that two different attention backends produce identical
    hidden states.
    """
    captured: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    frozen_observer = frozen_adapter.kv_observer
    original_observe = frozen_observer.observe_qkv

    def capture_observe(self: Any, **kwargs: Any) -> None:
        captured.append(
            (
                int(kwargs["layer_idx"]),
                kwargs["query_states"].detach().clone(),
                kwargs["key_states"].detach().clone(),
                kwargs["value_states"].detach().clone(),
            )
        )
        original_observe(**kwargs)

    frozen_observer.observe_qkv = types.MethodType(capture_observe, frozen_observer)
    try:
        forward = frozen_adapter.forward_once(ids, compute_nll=False)
    finally:
        frozen_observer.observe_qkv = original_observe
    if forward.summary is None or forward.summary.functional_damage is None:
        raise RuntimeError("frozen adapter did not produce functional damage")
    frozen_result = forward.summary.functional_damage

    fast_observer = fast_adapter.kv_observer
    fast_observer.configure_request(valid_token_mask=None, last_user_span=None)
    for layer_idx, query, key, value in captured:
        fast_observer.observe_qkv(
            layer_idx=layer_idx,
            query_states=query,
            key_states=key,
            value_states=value,
        )
    fast_observer.finalize(device=ids.device)
    fast_result = fast_observer.get_functional_result()
    if fast_result is None:
        raise RuntimeError("fast observer did not produce functional damage")
    return fast_result, frozen_result


def _compare(fast: Any, frozen: Any) -> dict[str, Any]:
    """Compare D22, masks, rankings and fixed token plans."""
    fast_score = fast.d22.float()
    frozen_score = frozen.d22.float()
    finite = torch.isfinite(fast_score) & torch.isfinite(frozen_score)
    difference = (fast_score - frozen_score).abs()
    finite_difference = difference[finite]
    max_abs = (
        float(finite_difference.max().to(device="cpu"))
        if finite_difference.numel()
        else 0.0
    )
    mean_abs = (
        float(finite_difference.mean().to(device="cpu"))
        if finite_difference.numel()
        else 0.0
    )
    relative = difference / (frozen_score.abs() + 1e-8)
    finite_relative = relative[finite]
    max_relative = (
        float(finite_relative.max().to(device="cpu"))
        if finite_relative.numel()
        else 0.0
    )
    fast_values = fast_score.detach().to(device="cpu").tolist()
    frozen_values = frozen_score.detach().to(device="cpu").tolist()
    fast_plan_hash, fast_plan = _plan_hash(fast)
    frozen_plan_hash, frozen_plan = _plan_hash(frozen)
    fast_valid = fast.scoring_valid_token_mask.detach().to(device="cpu")
    frozen_valid = frozen.scoring_valid_token_mask.detach().to(device="cpu")
    return {
        "token_count": len(fast_values),
        "d22_max_abs_error": max_abs,
        "d22_mean_abs_error": mean_abs,
        "d22_max_relative_error": max_relative,
        "finite_token_count": int(finite.sum()),
        "valid_mask_equal": bool(torch.equal(fast_valid, frozen_valid)),
        "top10_jaccard": _jaccard(
            _top_indices(fast_values), _top_indices(frozen_values)
        ),
        "fast_plan_hash": fast_plan_hash,
        "frozen_plan_hash": frozen_plan_hash,
        "plan_hash_equal": fast_plan_hash == frozen_plan_hash,
        "plan_equal": fast_plan == frozen_plan,
        "within_tolerance": max_abs <= 4e-3 and max_relative <= 4e-3,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run all requested token/chunk cases on one real Scout model."""
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real-model correctness requires a usable CUDA runtime")
    model = _load_model(args.model, device, torch.bfloat16)
    rows: list[dict[str, Any]] = []
    try:
        vocab_size = int(model.config.vocab_size)
        anchors = _parse_layers(args.anchor_layers)
        for token_count in _parse_positive_ints(args.tokens):
            ids = (
                torch.arange(token_count, device=device, dtype=torch.long) % vocab_size
            ).unsqueeze(0)
            for chunk_size in _parse_positive_ints(args.chunk_sizes):
                fast_adapter, _fast_scorer, _fast_cfg = _build_scout_runtime(
                    model,
                    mode="fast",
                    observer_backend="vectorized",
                    observer_token_chunk_size=chunk_size,
                    anchor_layers=anchors,
                    exit_layer=None,
                    scoring_version=FAST_D22_SCORING_VERSION,
                    v3_num_probes=args.probes,
                    v3_use_last_user_span=False,
                    v3_tail_probes=0,
                    return_diagnostics=False,
                    write_artifact=False,
                    fast_d22_profile_timing=False,
                    fast_d22_use_cuda_kernel=True,
                )
                frozen_adapter, _frozen_scorer, _frozen_cfg = _build_scout_runtime(
                    model,
                    mode="fast",
                    observer_backend="vectorized",
                    observer_token_chunk_size=chunk_size,
                    anchor_layers=anchors,
                    exit_layer=None,
                    scoring_version="v3-functional-disturbance",
                    v3_num_probes=args.probes,
                    v3_use_last_user_span=False,
                    v3_tail_probes=0,
                    return_diagnostics=False,
                    write_artifact=False,
                )
                fast, frozen = _run_same_qkv_observers(
                    fast_adapter, frozen_adapter, ids
                )
                torch.cuda.synchronize(device)
                row = _compare(fast, frozen)
                row.update(
                    {
                        "token_count": token_count,
                        "chunk_size": chunk_size,
                        "anchors": list(anchors),
                        "probes": args.probes,
                        "fast_kernel_path": fast.metadata.get("kernel_path"),
                    }
                )
                rows.append(row)
                print(
                    f"tokens={token_count} chunk={chunk_size} "
                    f"max_abs={row['d22_max_abs_error']:.6g} "
                    f"jaccard={row['top10_jaccard']:.4f} "
                    f"plan_equal={row['plan_equal']}",
                    flush=True,
                )
                del fast_adapter, frozen_adapter
                torch.cuda.empty_cache()
    finally:
        del model
        torch.cuda.empty_cache()
    return {
        "benchmark": "scoutrank_fast_d22_correctness_v1",
        "status": [
            "V3_D22_FIXED_CANDIDATE",
            "V3_FAST_D22_EXPERIMENTAL",
            "MCKP_DISABLED",
            "TRANSFER_FAIL",
            "NOT_PRODUCTION_READY",
        ],
        "model": args.model,
        "device": str(device),
        "anchors": list(_parse_layers(args.anchor_layers)),
        "probes": args.probes,
        "rows": rows,
        "all_pass": all(
            row["within_tolerance"] and row["plan_equal"] for row in rows
        ),
    }


def main() -> None:
    """Parse arguments and write the correctness artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", default="1024,4096")
    parser.add_argument("--chunk-sizes", default="1,7,4096")
    parser.add_argument("--anchor-layers", default="14,28")
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument(
        "--output", type=Path, default=Path("scoutrank_fast_d22_correctness.json")
    )
    args = parser.parse_args()
    if args.probes <= 0:
        raise ValueError("probes must be positive")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "all_pass": result["all_pass"]}))


if __name__ == "__main__":
    main()
