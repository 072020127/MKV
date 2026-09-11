#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate exact-scalar D22 against frozen v3 on identical real Q/K/V.

This is deliberately a separate artifact-producing harness.  It reuses the
existing same-QKV replay helper so a difference in the model's non-anchor
attention implementation cannot be mistaken for observer error.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LMCache_ROOT = REPOSITORY_ROOT / "LMCache"
for _path in (REPOSITORY_ROOT, LMCache_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from scoutrank_fast_d22_correctness import (
        _compare,
        _gpu_fixed_sort_allocation,
        _parse_layers,
        _parse_positive_ints,
        _run_same_qkv_observers,
    )
    from scoutrank_longbench_importance import _build_scout_runtime, _load_model
except ImportError:
    from benchmarks.scoutrank_fast_d22_correctness import (  # type: ignore
        _compare,
        _gpu_fixed_sort_allocation,
        _parse_layers,
        _parse_positive_ints,
        _run_same_qkv_observers,
    )
    from benchmarks.scoutrank_longbench_importance import (  # type: ignore
        _build_scout_runtime,
        _load_model,
    )

from makv_scoutrank import (  # noqa: E402
    FAST_1_EXACT_SCALAR_D22_SCORING_VERSION,
    makv_functional_token_bytes,
)


_FIXED_BUCKET_PRECISIONS = ("BF16", "K8V4", "K4V2", "K2V2")


def _fixed_plan_stats(result: Any, cfg: Any) -> dict[str, Any]:
    """Return exact variable payload accounting for the benchmark plan.

    The fixed token allocator is the same benchmark primitive used by the
    frozen comparison.  Framing and request-level metadata are constant for a
    pair of plans and are intentionally excluded from this per-token cost.
    """
    plan = _gpu_fixed_sort_allocation(result).detach().to(device="cpu")
    tier_counts = {
        precision: int((plan == bucket).sum().item())
        for bucket, precision in enumerate(_FIXED_BUCKET_PRECISIONS)
    }
    bytes_by_precision = {
        precision: makv_functional_token_bytes(
            precision,  # type: ignore[arg-type]
            num_layers=int(cfg.target_layers),
            num_kv_heads=int(cfg.target_kv_heads),
            head_dim=int(cfg.target_head_dim),
            scale_dtype=str(cfg.v3_makv_scale_dtype),
            raw_dtype_bytes=2,
        )
        for precision in _FIXED_BUCKET_PRECISIONS
    }
    actual_bytes = sum(
        tier_counts[precision] * bytes_by_precision[precision]
        for precision in _FIXED_BUCKET_PRECISIONS
    )
    return {
        "tier_counts": tier_counts,
        "bytes_by_precision_per_token": bytes_by_precision,
        "actual_bytes": actual_bytes,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run exact-scalar/frozen-v3 observer replay cases."""
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real-model correctness requires a usable CUDA runtime")
    model = _load_model(args.model, device, torch.bfloat16)
    rows: list[dict[str, Any]] = []
    anchors = _parse_layers(args.anchor_layers)
    try:
        vocab_size = int(model.config.vocab_size)
        for token_count in _parse_positive_ints(args.tokens):
            ids = (
                torch.arange(token_count, device=device, dtype=torch.long) % vocab_size
            ).unsqueeze(0)
            for chunk_size in _parse_positive_ints(args.chunk_sizes):
                exact_adapter, _exact_scorer, exact_cfg = _build_scout_runtime(
                    model,
                    mode="fast",
                    observer_backend="vectorized",
                    observer_token_chunk_size=chunk_size,
                    anchor_layers=anchors,
                    exit_layer=None,
                    scoring_version=FAST_1_EXACT_SCALAR_D22_SCORING_VERSION,
                    v3_num_probes=args.probes,
                    v3_use_last_user_span=False,
                    v3_tail_probes=0,
                    return_diagnostics=False,
                    write_artifact=False,
                    fast_d22_profile_timing=False,
                    fast_d22_use_cuda_kernel=True,
                    fast_d22_qk_dtype=args.qk_dtype,
                )
                frozen_adapter, _frozen_scorer, frozen_cfg = _build_scout_runtime(
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
                exact, frozen = _run_same_qkv_observers(
                    exact_adapter, frozen_adapter, ids
                )
                torch.cuda.synchronize(device)
                row = _compare(exact, frozen)
                exact_plan_stats = _fixed_plan_stats(exact, exact_cfg)
                frozen_plan_stats = _fixed_plan_stats(frozen, frozen_cfg)
                row.update(
                    {
                        "token_count": token_count,
                        "chunk_size": chunk_size,
                        "anchors": list(anchors),
                        "probes": args.probes,
                        "qk_dtype": args.qk_dtype,
                        "exact_kernel_path": exact.metadata.get("kernel_path"),
                        "exact_scoring_version": exact.metadata.get(
                            "scoring_version"
                        ),
                        "exact_scalar_formula": exact.metadata.get(
                            "exact_scalar_formula"
                        ),
                        "exact_tier_counts": exact_plan_stats["tier_counts"],
                        "frozen_tier_counts": frozen_plan_stats["tier_counts"],
                        "tier_counts_equal": exact_plan_stats["tier_counts"]
                        == frozen_plan_stats["tier_counts"],
                        "exact_actual_bytes": exact_plan_stats["actual_bytes"],
                        "frozen_actual_bytes": frozen_plan_stats["actual_bytes"],
                        "actual_bytes_equal": exact_plan_stats["actual_bytes"]
                        == frozen_plan_stats["actual_bytes"],
                        "bytes_by_precision_per_token": exact_plan_stats[
                            "bytes_by_precision_per_token"
                        ],
                    }
                )
                rows.append(row)
                print(
                    f"tokens={token_count} chunk={chunk_size} "
                    f"max_abs={row['d22_max_abs_error']:.6g} "
                    f"jaccard={row['top10_jaccard']:.4f} "
                    f"plan_equal={row['plan_equal']} "
                    f"tiers={row['exact_tier_counts']} "
                    f"bytes={row['exact_actual_bytes']}",
                    flush=True,
                )
                del exact_adapter, frozen_adapter
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "benchmark": "scoutrank_fast_1_exact_scalar_d22_correctness_v1",
        "status": [
            "V3_D22_FIXED_CANDIDATE",
            "V3_FAST_1_EXPERIMENTAL",
            "MCKP_DISABLED",
            "TRANSFER_FAIL",
            "NOT_PRODUCTION_READY",
        ],
        "model": args.model,
        "device": str(device),
        "anchors": list(anchors),
        "probes": args.probes,
        "rows": rows,
        "all_pass": all(
            row["within_tolerance"]
            and row["plan_equal"]
            and row["tier_counts_equal"]
            and row["actual_bytes_equal"]
            for row in rows
        ),
    }


def main() -> None:
    """Parse arguments and write a correctness artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", default="1024,4096")
    parser.add_argument("--chunk-sizes", default="1,7,4096")
    parser.add_argument("--anchor-layers", default="14,28")
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument(
        "--qk-dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scoutrank_fast_1_exact_scalar_d22_correctness.json"),
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
