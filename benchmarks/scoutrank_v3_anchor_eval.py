#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline development-set comparison for fixed ScoutRank-v3 anchor choices.

This entrypoint evaluates supplied artifacts only. It does not select an
anchor, mutate a holdout, or feed evaluation outcomes back into scoring.
"""

from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any
import argparse
import json
import math
import statistics
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# First Party
from experiments.scoutrank_transfer.metrics import spearman  # noqa: E402


def _top_indices(values: list[float], count: int) -> set[int]:
    ordered = sorted(range(len(values)), key=lambda index: (-values[index], index))
    return set(ordered[:count])


def _damage_capture(predicted: list[float], oracle: list[float]) -> float:
    count = min(len(predicted), max(2, math.ceil(0.1 * len(predicted))))
    selected = _top_indices(predicted, count)
    ideal = _top_indices(oracle, count)
    denominator = sum(oracle[index] for index in ideal)
    if denominator == 0:
        return math.nan
    return sum(oracle[index] for index in selected) / denominator


def _topk_jaccard(predicted: list[float], oracle: list[float]) -> float:
    count = min(len(predicted), max(2, math.ceil(0.1 * len(predicted))))
    left = _top_indices(predicted, count)
    right = _top_indices(oracle, count)
    return len(left & right) / len(left | right) if left | right else math.nan


def _extract_scores(artifact: dict[str, Any], prompt_hash: str) -> list[float]:
    functional = artifact.get("functional_results", {}).get(prompt_hash)
    if isinstance(functional, dict):
        damage = functional.get("functional_damage", {})
        values = damage.get("D22", damage.get("K2V2"))
        if isinstance(values, list):
            return [float(value) for value in values]
    values = artifact.get("scores", {}).get(prompt_hash)
    if not isinstance(values, list):
        raise ValueError(f"artifact has no score vector for prompt {prompt_hash}")
    return [float(value) for value in values]


def evaluate_anchor_artifacts(
    *,
    candidates: dict[str, dict[str, Any]],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    """Compare precomputed fixed-anchor v3 artifacts against an oracle artifact."""
    oracle_scores = oracle.get("scores")
    if not isinstance(oracle_scores, dict) or not oracle_scores:
        raise ValueError("oracle artifact must contain non-empty scores")
    result: dict[str, Any] = {
        "evaluation": "scoutrank_v3_anchor_development_eval_v1",
        "selection": "none; caller must choose anchors outside this tool",
        "anchors": {},
    }
    for label, artifact in candidates.items():
        rows: list[dict[str, float | str]] = []
        scorer_times: list[float] = []
        peak_memories: list[float] = []
        metadata = artifact.get("metadata", {})
        for prompt_hash, oracle_values_raw in oracle_scores.items():
            if not isinstance(oracle_values_raw, list):
                raise ValueError("oracle score vectors must be lists")
            predicted = _extract_scores(artifact, str(prompt_hash))
            oracle_values = [float(value) for value in oracle_values_raw]
            if len(predicted) != len(oracle_values) or not predicted:
                raise ValueError(f"score length mismatch for {label}:{prompt_hash}")
            rows.append(
                {
                    "prompt_hash": str(prompt_hash),
                    "spearman": spearman(predicted, oracle_values),
                    "damage_capture": _damage_capture(predicted, oracle_values),
                    "topk_jaccard": _topk_jaccard(predicted, oracle_values),
                }
            )
            row_meta = metadata.get(prompt_hash, {})
            if isinstance(row_meta, dict):
                if isinstance(row_meta.get("scoutrank_time_ms"), (int, float)):
                    scorer_times.append(float(row_meta["scoutrank_time_ms"]))
                if isinstance(row_meta.get("peak_memory_bytes"), (int, float)):
                    peak_memories.append(float(row_meta["peak_memory_bytes"]))
        result["anchors"][label] = {
            "prompt_count": len(rows),
            "spearman_mean": statistics.fmean(float(row["spearman"]) for row in rows),
            "damage_capture_mean": statistics.fmean(
                float(row["damage_capture"]) for row in rows
            ),
            "topk_jaccard_mean": statistics.fmean(
                float(row["topk_jaccard"]) for row in rows
            ),
            "scorer_time_ms_mean": statistics.fmean(scorer_times)
            if scorer_times
            else None,
            "peak_memory_bytes_max": max(peak_memories) if peak_memories else None,
            "rows": rows,
        }
    return result


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH")
    return label, Path(path)


def main() -> None:
    """Run the no-selection development-set evaluation entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=_parse_candidate,
        action="append",
        required=True,
        help="Repeat LABEL=PATH for each fixed anchor artifact.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidates = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in args.candidate
    }
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    result = evaluate_anchor_artifacts(candidates=candidates, oracle=oracle)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
