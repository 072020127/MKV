#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile the opt-in ``v3_fast_d22`` ScoutRank path on a real Qwen3 model.

The benchmark uses synchronized CUDA events for GPU work and explicit stream
boundaries for host work.  It intentionally writes a compact online result;
the frozen v3 observer is only included when ``--include-frozen-v3`` is set.
No timing value is synthesized when CUDA is unavailable.
"""

from __future__ import annotations

# Standard
import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

# Third Party
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LMCache_ROOT = REPOSITORY_ROOT / "LMCache"
for _path in (REPOSITORY_ROOT, LMCache_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from scoutrank_longbench_importance import _build_scout_runtime, _load_model
except ImportError:
    from benchmarks.scoutrank_longbench_importance import (  # type: ignore
        _build_scout_runtime,
        _load_model,
    )

from makv_scoutrank import FAST_D22_SCORING_VERSION  # noqa: E402


DEFAULT_STAGE_NAMES = (
    "scout_forward",
    "qkv_observer",
    "baseline_attention_pass",
    "k2_fake_quant",
    "d22_disturbance",
    "fused_k2_d22",
    "gpu_sort_allocation",
    "cpu_transfer",
    "artifact_serialization",
    "total",
)


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    """Parse a non-empty comma-separated list of positive integers."""
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("expected positive comma-separated integers")
    return values


def _parse_layers(value: str) -> tuple[int, ...]:
    """Parse and validate sorted 1-based anchor layers."""
    values = _parse_positive_ints(value)
    if values != tuple(sorted(set(values))):
        raise ValueError("anchor layers must be sorted and unique")
    return values


def _synchronize(device: torch.device) -> None:
    """Synchronize only at an externally visible profiling boundary."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _event_timed(
    device: torch.device, operation: Callable[[], Any]
) -> tuple[Any, float]:
    """Run one operation and return a synchronized CUDA-event duration."""
    if device.type != "cuda":
        started = time.perf_counter()
        result = operation()
        return result, (time.perf_counter() - started) * 1000.0
    _synchronize(device)
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream(device)
    started.record(stream)
    result = operation()
    finished.record(stream)
    finished.synchronize()
    return result, float(started.elapsed_time(finished))


def _percentile(values: list[float], percentile: float) -> float | None:
    """Return a deterministic nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return float(ordered[index])


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    """Capture CUDA allocator counters after a synchronized run."""
    if device.type != "cuda":
        return {
            "memory_allocated_bytes": 0,
            "peak_memory_allocated_bytes": 0,
            "memory_reserved_bytes": 0,
            "peak_memory_reserved_bytes": 0,
        }
    return {
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _gpu_fixed_sort_allocation(result: Any) -> torch.Tensor:
    """Measure the GPU-only deterministic D22 fixed-ratio allocation primitive.

    Production block allocation remains in ``D22FixedRatioAllocator``.  This
    compact token-level primitive is deliberately benchmark-only: it measures
    the sort/mask cost that an online GPU planner would pay and does not change
    the frozen allocator or emitted plans.
    """
    scores = result.d22
    valid = result.scoring_valid_token_mask
    forced = getattr(result, "forced_precision_mask", None)
    if forced is None:
        forced = ~valid
    token_count = scores.numel()
    order = torch.argsort(
        torch.where(valid, scores, torch.full_like(scores, float("-inf"))),
        descending=True,
        stable=True,
    )
    eligible = order[~forced.index_select(0, order)]
    eligible_count = eligible.numel()
    bucket_ids = torch.zeros(token_count, dtype=torch.int8, device=scores.device)
    if eligible_count:
        # BF16/K8V4/K4V2/K2V2 = 10/20/50/20, matching the fixed v3 plan.
        boundaries = (
            round(0.10 * eligible_count),
            round(0.30 * eligible_count),
            round(0.80 * eligible_count),
            eligible_count,
        )
        cursor = 0
        for bucket, boundary in enumerate(boundaries):
            bucket_ids[eligible[cursor:boundary]] = bucket
            cursor = boundary
    bucket_ids.masked_fill_(forced, 0)
    return bucket_ids


def _extract_fast_result(forward: Any) -> Any:
    """Extract a fast result without copying score tensors to the host."""
    summary = forward.summary
    if summary is None or summary.functional_damage is None:
        raise RuntimeError("fast D22 forward produced no functional result")
    return summary.functional_damage


def _stage_values(result: Any) -> dict[str, float]:
    """Normalize observer stage metadata to the public profiling names."""
    values = {name: 0.0 for name in DEFAULT_STAGE_NAMES}
    metadata = getattr(result, "metadata", {})
    observer_stages = metadata.get("stage_timing_ms", {})
    if isinstance(observer_stages, dict):
        for name in (
            "qkv_observer",
            "baseline_attention_pass",
            "k2_fake_quant",
            "d22_disturbance",
            "fused_k2_d22",
        ):
            raw = observer_stages.get(name)
            if isinstance(raw, (int, float)):
                values[name] = float(raw)
        # Frozen v3 exposes its historical stage names.  These mappings are
        # reporting aliases only; they do not alter the v3 observer.
        aliases = {
            "baseline_attention": "baseline_attention_pass",
            "fake_quant": "k2_fake_quant",
            "analytic_update": "d22_disturbance",
        }
        for source, target in aliases.items():
            raw = observer_stages.get(source)
            if isinstance(raw, (int, float)):
                values[target] = float(raw)
    if metadata.get("k2_fake_quant_fused"):
        # The fused kernel has no internal event boundary.  Report the exact
        # fused duration once and explicitly leave the decomposed K2 field at
        # zero rather than pretending the kernel's work was separately timed.
        values["fused_k2_d22"] = values["d22_disturbance"]
        values["k2_fake_quant"] = 0.0
    return values


def _profile_one(
    *,
    adapter: Any,
    ids: torch.Tensor,
    device: torch.device,
    version: str,
) -> dict[str, Any]:
    """Profile one request-local score and its compact output boundary."""
    if device.type == "cuda":
        _synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_allocated = (
        int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    )
    total_started = time.perf_counter()
    forward, scout_forward_ms = _event_timed(
        device, lambda: adapter.forward_once(ids, compute_nll=False)
    )
    result = _extract_fast_result(forward)
    stages = _stage_values(result)

    _, gpu_sort_ms = _event_timed(device, lambda: _gpu_fixed_sort_allocation(result))
    stages["gpu_sort_allocation"] = gpu_sort_ms

    # This is deliberately outside the scoring loop and is the only host copy
    # in this benchmark.  It measures the compatibility/API boundary exactly.
    _synchronize(device)
    transfer_started = time.perf_counter()
    host_scores = result.d22.detach().to(device="cpu")
    _synchronize(device)
    stages["cpu_transfer"] = (time.perf_counter() - transfer_started) * 1000.0

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
    _synchronize(device)
    stages["scout_forward"] = scout_forward_ms
    stages["total"] = (time.perf_counter() - total_started) * 1000.0
    memory = _memory_snapshot(device)
    memory["peak_temporary_gpu_bytes"] = max(
        0, memory["peak_memory_allocated_bytes"] - start_allocated
    )
    memory["kernel_launch_count"] = int(result.metadata.get("kernel_launch_count", 0))
    return {
        "version": version,
        "token_count": int(host_scores.numel()),
        "stages_ms": stages,
        "memory": memory,
        "kernel_path": result.metadata.get("kernel_path"),
        "qk_dtype": result.metadata.get("qk_dtype"),
        "probe_count": result.metadata.get("probe_count"),
    }


def _build_adapter(
    model: Any,
    *,
    version: str,
    device: torch.device,
    anchors: tuple[int, ...],
    probes: int,
    token_chunk_size: int,
    qk_dtype: str,
    use_cuda_kernel: bool,
) -> Any:
    """Build one isolated adapter for the selected scoring version."""
    adapter, _scorer, _cfg = _build_scout_runtime(
        model,
        mode="fast",
        observer_backend="vectorized",
        observer_token_chunk_size=token_chunk_size,
        anchor_layers=anchors,
        exit_layer=None,
        scoring_version=version,
        v3_num_probes=probes,
        v3_use_last_user_span=False,
        v3_tail_probes=0,
        v3_protect_prefix_tokens=4,
        v3_protect_tail_tokens=16,
        return_diagnostics=False,
        write_artifact=False,
        fast_d22_qk_dtype=qk_dtype,
        fast_d22_profile_timing=True,
        fast_d22_use_cuda_kernel=use_cuda_kernel,
    )
    if adapter is None:
        raise RuntimeError(f"no adapter was built for {version}")
    del device
    return adapter


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate formal iterations by version and token count."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["version"], row["token_count"]), []).append(row)
    summary: dict[str, Any] = {}
    for (version, token_count), group in grouped.items():
        stage_summary: dict[str, Any] = {}
        for stage in DEFAULT_STAGE_NAMES:
            values = [float(row["stages_ms"][stage]) for row in group]
            median = float(statistics.median(values))
            p95 = _percentile(values, 0.95)
            stage_summary[stage] = {
                "median_ms": median,
                "p95_ms": p95,
                "median_percent_of_total": (
                    100.0 * median / statistics.median(
                        [float(row["stages_ms"]["total"]) for row in group]
                    )
                    if stage != "total"
                    else 100.0
                ),
            }
        peak_temporary = [
            int(row["memory"]["peak_temporary_gpu_bytes"]) for row in group
        ]
        peak_allocated = [
            int(row["memory"]["peak_memory_allocated_bytes"]) for row in group
        ]
        launches = [int(row["memory"]["kernel_launch_count"]) for row in group]
        summary[f"{version}:{token_count}"] = {
            "version": version,
            "token_count": token_count,
            "formal_iterations": len(group),
            "stages": stage_summary,
            "peak_temporary_gpu_bytes": max(peak_temporary),
            "peak_memory_allocated_bytes": max(peak_allocated),
            "kernel_launch_count": max(launches),
            "kernel_launch_count_values": sorted(set(launches)),
            "kernel_path": group[-1]["kernel_path"],
            "qk_dtype": group[-1]["qk_dtype"],
            "probe_count": group[-1]["probe_count"],
        }
    return summary


def _markdown(result: dict[str, Any]) -> str:
    """Render a compact human-readable report from the JSON result."""
    lines = [
        "# ScoutRank v3_fast_d22 profiling",
        "",
        "Status: `V3_D22_FIXED_CANDIDATE`, `V3_FAST_D22_EXPERIMENTAL`, "
        "`MCKP_DISABLED`, `TRANSFER_FAIL`, `NOT_PRODUCTION_READY`.",
        "",
        "The fused row reports K2 fake-quant plus D22 as one kernel; the"
        " decomposed K2 column is therefore zero and is not an inferred split.",
        "",
        "| version | tokens | total median ms | total p95 ms | forward median ms | "
        "fused K2+D22 median ms | peak temporary GPU bytes | launches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"].values():
        stages = row["stages"]
        lines.append(
            "| {version} | {tokens} | {total:.3f} | {p95:.3f} | {forward:.3f} | "
            "{fused:.3f} | {memory} | {launches} |".format(
                version=row["version"],
                tokens=row["token_count"],
                total=stages["total"]["median_ms"],
                p95=stages["total"]["p95_ms"],
                forward=stages["scout_forward"]["median_ms"],
                fused=stages["fused_k2_d22"]["median_ms"],
                memory=row["peak_temporary_gpu_bytes"],
                launches=row["kernel_launch_count"],
            )
        )
    lines.extend(
        [
            "",
            "No quality, LongBench, or production-readiness conclusion is made "
            "by this timing-only benchmark.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Run warmups and formal synchronized measurements."""
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA profiling requires a usable CUDA runtime; no timing values "
            "were generated"
        )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = _load_model(args.model, device, dtype)
    versions = [FAST_D22_SCORING_VERSION]
    if args.include_frozen_v3:
        versions.append("v3-functional-disturbance")
    rows: list[dict[str, Any]] = []
    try:
        vocab_size = int(model.config.vocab_size)
        anchors = _parse_layers(args.anchor_layers)
        for token_count in _parse_positive_ints(args.tokens):
            ids = (
                torch.arange(token_count, device=device, dtype=torch.long) % vocab_size
            ).unsqueeze(0)
            for version in versions:
                adapter = _build_adapter(
                    model,
                    version=version,
                    device=device,
                    anchors=anchors,
                    probes=args.probes,
                    token_chunk_size=args.token_chunk_size,
                    qk_dtype=args.qk_dtype,
                    use_cuda_kernel=args.use_cuda_kernel,
                )
                for _ in range(args.warmup):
                    with torch.inference_mode():
                        _profile_one(
                            adapter=adapter,
                            ids=ids,
                            device=device,
                            version=version,
                        )
                _synchronize(device)
                for iteration in range(args.iterations):
                    with torch.inference_mode():
                        row = _profile_one(
                            adapter=adapter,
                            ids=ids,
                            device=device,
                            version=version,
                        )
                    row["iteration"] = iteration
                    rows.append(row)
                    print(
                        f"version={version} tokens={token_count} "
                        f"iteration={iteration + 1}/{args.iterations} "
                        f"total_ms={row['stages_ms']['total']:.3f}",
                        flush=True,
                    )
                del adapter
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "benchmark": "scoutrank_fast_d22_profile_v1",
        "status": [
            "V3_D22_FIXED_CANDIDATE",
            "V3_FAST_D22_EXPERIMENTAL",
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
        "anchors": list(_parse_layers(args.anchor_layers)),
        "probes": args.probes,
        "token_chunk_size": args.token_chunk_size,
        "warmup": args.warmup,
        "formal_iterations": args.iterations,
        "include_frozen_v3": args.include_frozen_v3,
        "qk_dtype": args.qk_dtype,
        "use_cuda_kernel": args.use_cuda_kernel,
        "stage_names": list(DEFAULT_STAGE_NAMES),
        "summary": _summarize(rows),
        "rows": rows,
    }


def main() -> None:
    """Parse arguments, run the profile, and write JSON/Markdown artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--tokens", default="4096,16384,32768")
    parser.add_argument("--anchor-layers", default="14,28")
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument("--token-chunk-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--qk-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--use-cuda-kernel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--include-frozen-v3", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("scoutrank_fast_d22_profile.json")
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.probes <= 0 or args.token_chunk_size <= 0:
        raise ValueError("probes and token chunk size must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    result = run_profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    report_path = args.report or args.output.with_suffix(".md")
    report_path.write_text(_markdown(result))
    print(json.dumps({"output": str(args.output), "report": str(report_path)}))


if __name__ == "__main__":
    main()
