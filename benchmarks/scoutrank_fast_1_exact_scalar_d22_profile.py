#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Synchronized 4K/16K/32K profile for frozen v3 and exact-scalar D22.

The two observers are run in a deterministic random alternating order on the
same model and GPU.  CUDA Events are synchronized at every externally
reported boundary.  Artifact JSON serialization is measured separately and is
not included in the reported online ``total``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Callable

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LMCache_ROOT = REPOSITORY_ROOT / "LMCache"
for _path in (REPOSITORY_ROOT, LMCache_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from scoutrank_fast_d22_profile import _gpu_fixed_sort_allocation
    from scoutrank_longbench_importance import _build_scout_runtime, _load_model
except ImportError:
    from benchmarks.scoutrank_fast_d22_profile import (  # type: ignore
        _gpu_fixed_sort_allocation,
    )
    from benchmarks.scoutrank_longbench_importance import (  # type: ignore
        _build_scout_runtime,
        _load_model,
    )

from makv_scoutrank import (  # noqa: E402
    FAST_1_EXACT_SCALAR_D22_SCORING_VERSION,
)

FROZEN_VERSION = "v3-functional-disturbance"
STAGES = (
    "scout_forward",
    "qkv_observer",
    "baseline_attention",
    "fake_quant",
    "delta_z_gemm",
    "scalar_damage",
    "allocator",
    "cpu_transfer",
    "artifact_serialization",
    "total",
    "total_including_artifact",
)


def _synchronize(device: torch.device) -> None:
    """Synchronize at a profiling boundary, never in production scoring."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _event_timed(
    device: torch.device, operation: Callable[[], Any]
) -> tuple[Any, float]:
    """Measure one operation with a synchronized CUDA Event."""
    if device.type != "cuda":
        started = time.perf_counter()
        result = operation()
        return result, (time.perf_counter() - started) * 1000.0
    _synchronize(device)
    stream = torch.cuda.current_stream(device)
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record(stream)
    result = operation()
    finished.record(stream)
    finished.synchronize()
    return result, float(started.elapsed_time(finished))


def _percentile(values: list[float], fraction: float) -> float | None:
    """Return a nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return float(ordered[index])


def _stage_values(result: Any) -> dict[str, float]:
    """Map each observer's stage vocabulary to common report names."""
    stages = {name: 0.0 for name in STAGES}
    metadata = getattr(result, "metadata", {})
    raw_stages = metadata.get("stage_timing_ms", {})
    if not isinstance(raw_stages, dict):
        return stages
    aliases = {
        "baseline_attention": "baseline_attention",
        "baseline_attention_pass": "baseline_attention",
        "fake_quant": "fake_quant",
        "k2_fake_quant": "fake_quant",
        "analytic_update": "scalar_damage",
        "d22_disturbance": "scalar_damage",
        "fused_k2_d22": "scalar_damage",
        "delta_z_gemm": "delta_z_gemm",
        "scalar_damage": "scalar_damage",
        "qkv_observer": "qkv_observer",
    }
    for source, target in aliases.items():
        value = raw_stages.get(source)
        if isinstance(value, (int, float)):
            stages[target] += float(value)
    return stages


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    """Return synchronized CUDA allocator counters."""
    if device.type != "cuda":
        return {
            "allocated_bytes": 0,
            "peak_allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_reserved_bytes": 0,
        }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _profile_one(
    *, adapter: Any, ids: torch.Tensor, device: torch.device, version: str
) -> dict[str, Any]:
    """Profile one online score, GPU allocator, and compatibility copy."""
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started_total = time.perf_counter()
    forward, forward_ms = _event_timed(
        device, lambda: adapter.forward_once(ids, compute_nll=False)
    )
    summary = forward.summary
    if summary is None or summary.functional_damage is None:
        raise RuntimeError(f"{version} produced no functional result")
    result = summary.functional_damage
    stages = _stage_values(result)
    stages["scout_forward"] = forward_ms

    _, allocator_ms = _event_timed(
        device, lambda: _gpu_fixed_sort_allocation(result)
    )
    stages["allocator"] = allocator_ms

    _synchronize(device)
    transfer_started = time.perf_counter()
    host_scores = result.d22.detach().to(device="cpu")
    _synchronize(device)
    stages["cpu_transfer"] = (time.perf_counter() - transfer_started) * 1000.0

    # This is diagnostic overhead only.  It is intentionally after ``total``
    # so artifact creation cannot be claimed as an algorithmic speedup.
    stages["total"] = (time.perf_counter() - started_total) * 1000.0
    serialization_started = time.perf_counter()
    json.dumps(
        {
            "scoring_version": version,
            "token_count": int(host_scores.numel()),
            "metadata": result.metadata,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    stages["artifact_serialization"] = (
        time.perf_counter() - serialization_started
    ) * 1000.0
    stages["total_including_artifact"] = (
        time.perf_counter() - started_total
    ) * 1000.0
    _synchronize(device)
    memory = _memory_snapshot(device)
    return {
        "version": version,
        "token_count": int(host_scores.numel()),
        "stages_ms": stages,
        "memory": memory,
        "kernel_path": result.metadata.get("kernel_path"),
        "qk_dtype": result.metadata.get("qk_dtype"),
        "kernel_launch_count": int(result.metadata.get("kernel_launch_count", 0)),
        "exact_scalar_formula": bool(
            result.metadata.get("exact_scalar_formula", False)
        ),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate formal rows by version and token count."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["version"], row["token_count"]), []).append(row)
    summary: dict[str, Any] = {}
    for (version, token_count), group in grouped.items():
        total_median = statistics.median(
            [float(row["stages_ms"]["total"]) for row in group]
        )
        stage_summary: dict[str, Any] = {}
        for stage in STAGES:
            values = [float(row["stages_ms"][stage]) for row in group]
            median = float(statistics.median(values))
            stage_summary[stage] = {
                "median_ms": median,
                "p95_ms": _percentile(values, 0.95),
                "median_percent_of_online_total": (
                    100.0 * median / total_median if total_median else None
                ),
            }
        summary[f"{version}:{token_count}"] = {
            "version": version,
            "token_count": token_count,
            "formal_iterations": len(group),
            "stages": stage_summary,
            "peak_allocated_bytes": max(
                int(row["memory"]["peak_allocated_bytes"]) for row in group
            ),
            "peak_reserved_bytes": max(
                int(row["memory"]["peak_reserved_bytes"]) for row in group
            ),
            "kernel_launch_count_values": sorted(
                {int(row["kernel_launch_count"]) for row in group}
            ),
            "kernel_path": group[-1]["kernel_path"],
            "qk_dtype": group[-1]["qk_dtype"],
        }
    return summary


def _markdown(result: dict[str, Any]) -> str:
    """Render the measured timing summary."""
    lines = [
        "# ScoutRank v3_fast_1_exact_scalar_d22 profile",
        "",
        "Status: `V3_D22_FIXED_CANDIDATE`, `V3_FAST_1_EXPERIMENTAL`, "
        "`MCKP_DISABLED`, `TRANSFER_FAIL`, `NOT_PRODUCTION_READY`.",
        "",
        "`total` is the synchronized online path through allocator and score "
        "copy; artifact serialization is reported separately.",
        "",
        "| version | tokens | total median ms | total p95 ms | forward median ms | "
        "baseline median ms | delta-z GEMM median ms | scalar damage median ms | "
        "peak allocated | peak reserved | launches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"].values():
        stages = row["stages"]
        lines.append(
            "| {version} | {tokens} | {total:.3f} | {p95:.3f} | "
            "{forward:.3f} | {baseline:.3f} | {delta:.3f} | {damage:.3f} | "
            "{allocated} | {reserved} | {launches} |".format(
                version=row["version"],
                tokens=row["token_count"],
                total=stages["total"]["median_ms"],
                p95=stages["total"]["p95_ms"],
                forward=stages["scout_forward"]["median_ms"],
                baseline=stages["baseline_attention"]["median_ms"],
                delta=stages["delta_z_gemm"]["median_ms"],
                damage=stages["scalar_damage"]["median_ms"],
                allocated=row["peak_allocated_bytes"],
                reserved=row["peak_reserved_bytes"],
                launches=row["kernel_launch_count_values"],
            )
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run warmups and randomized alternating formal iterations."""
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA profiling requires a usable CUDA runtime; no timing values generated"
        )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = _load_model(args.model, device, dtype)
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    try:
        vocab_size = int(model.config.vocab_size)
        for token_count in _parse_positive_ints(args.tokens):
            ids = (
                torch.arange(token_count, device=device, dtype=torch.long) % vocab_size
            ).unsqueeze(0)
            adapters: dict[str, Any] = {}
            for version in (FROZEN_VERSION, FAST_1_EXACT_SCALAR_D22_SCORING_VERSION):
                adapter, _scorer, _cfg = _build_scout_runtime(
                    model,
                    mode="fast",
                    observer_backend="vectorized",
                    observer_token_chunk_size=args.token_chunk_size,
                    anchor_layers=(14, 28),
                    exit_layer=None,
                    scoring_version=version,
                    v3_num_probes=16,
                    v3_use_last_user_span=False,
                    v3_tail_probes=0,
                    return_diagnostics=False,
                    write_artifact=False,
                    fast_d22_qk_dtype=args.qk_dtype,
                    fast_d22_profile_timing=True,
                    fast_d22_use_cuda_kernel=True,
                )
                if adapter is None:
                    raise RuntimeError(f"no adapter for {version}")
                adapters[version] = adapter

            for _ in range(args.warmup):
                for version in (
                    FROZEN_VERSION,
                    FAST_1_EXACT_SCALAR_D22_SCORING_VERSION,
                ):
                    with torch.inference_mode():
                        _profile_one(
                            adapter=adapters[version],
                            ids=ids,
                            device=device,
                            version=version,
                        )
            _synchronize(device)
            for iteration in range(args.iterations):
                order = [FROZEN_VERSION, FAST_1_EXACT_SCALAR_D22_SCORING_VERSION]
                rng.shuffle(order)
                for order_index, version in enumerate(order):
                    with torch.inference_mode():
                        row = _profile_one(
                            adapter=adapters[version],
                            ids=ids,
                            device=device,
                            version=version,
                        )
                    row.update(
                        {
                            "iteration": iteration,
                            "order_index": order_index,
                            "seed": args.seed,
                        }
                    )
                    rows.append(row)
                    print(
                        f"tokens={token_count} iteration={iteration + 1}/"
                        f"{args.iterations} order={order_index} version={version} "
                        f"total_ms={row['stages_ms']['total']:.3f}",
                        flush=True,
                    )
            del adapters
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "benchmark": "scoutrank_fast_1_exact_scalar_d22_profile_v1",
        "status": [
            "V3_D22_FIXED_CANDIDATE",
            "V3_FAST_1_EXPERIMENTAL",
            "MCKP_DISABLED",
            "TRANSFER_FAIL",
            "NOT_PRODUCTION_READY",
        ],
        "model": args.model,
        "device": str(device),
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "tokens": list(_parse_positive_ints(args.tokens)),
        "anchors": [14, 28],
        "probes": 16,
        "token_chunk_size": args.token_chunk_size,
        "warmup": args.warmup,
        "formal_iterations": args.iterations,
        "random_seed": args.seed,
        "randomized_alternation": True,
        "stage_names": list(STAGES),
        "summary": _summarize(rows),
        "rows": rows,
    }


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    """Parse positive comma-separated integers."""
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("expected positive comma-separated integers")
    return parsed


def main() -> None:
    """Parse command-line arguments and write JSON/Markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--tokens", default="4096,16384,32768")
    parser.add_argument("--token-chunk-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--qk-dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scoutrank_fast_1_exact_scalar_d22_profile.json"),
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.token_chunk_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("chunk size must be positive; warmup >= 0; iterations > 0")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    report = args.report or args.output.with_suffix(".md")
    report.write_text(_markdown(result))
    print(json.dumps({"output": str(args.output), "report": str(report)}))


if __name__ == "__main__":
    main()
