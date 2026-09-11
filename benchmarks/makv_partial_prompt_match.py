# SPDX-License-Identifier: Apache-2.0

"""Measure MaKV behavior for full, partial-prefix, and non-matching prompts."""

from __future__ import annotations

# Standard
from argparse import Namespace
from pathlib import Path
from typing import Any
import argparse
import json
import math
import os
import re
import time

# Third Party
import requests
from transformers import AutoTokenizer

# First Party
try:
    from longbench_makv_cachegen import (
        LongBenchExample,
        load_examples,
        manager_health,
        prompt_ids,
        request_completion,
    )
except ModuleNotFoundError:
    # Support importing the driver through a PYTHONPATH that exposes the
    # benchmark directory as a namespace package.
    from benchmarks.longbench_makv_cachegen import (
        LongBenchExample,
        load_examples,
        manager_health,
        prompt_ids,
        request_completion,
    )


def _parse_fractions(value: str) -> tuple[float, ...]:
    """Parse and validate partial-prefix fractions."""
    fractions = tuple(
        float(item.strip()) for item in value.split(",") if item.strip()
    )
    if not fractions or any(
        not math.isfinite(item) or not 0.0 < item < 1.0
        for item in fractions
    ):
        raise ValueError("partial fractions must be finite values in (0, 1)")
    return fractions


def _select_longest_prompt(
    tokenizer: Any,
    examples: list[LongBenchExample],
    run_id: str,
    *,
    min_tokens: int,
    max_tokens: int,
    enable_thinking: bool,
) -> tuple[LongBenchExample, list[int]]:
    """Select the longest usable prompt inside the model context limit."""
    candidates: list[tuple[int, LongBenchExample, list[int]]] = []
    for example in examples:
        ids = prompt_ids(
            tokenizer,
            example,
            run_id,
            enable_thinking=enable_thinking,
        )
        if min_tokens <= len(ids) <= max_tokens:
            candidates.append((len(ids), example, ids))
    if not candidates:
        raise RuntimeError(
            f"no prompt in [{min_tokens}, {max_tokens}] tokens; "
            f"scanned {len(examples)} examples"
        )
    _, example, ids = max(candidates, key=lambda item: item[0])
    return example, ids


def _different_token(tokenizer: Any, original: int) -> int:
    """Return a vocabulary token different from ``original``."""
    vocabulary_size = int(getattr(tokenizer, "vocab_size", 0))
    if vocabulary_size < 2:
        raise RuntimeError("tokenizer vocabulary must contain at least two tokens")
    candidate = (int(original) + 1) % vocabulary_size
    if candidate == int(original):
        candidate = (int(original) + 2) % vocabulary_size
    return candidate


def _make_partial_prompt(
    base_ids: list[int], split: int, tokenizer: Any, variant: int
) -> list[int]:
    """Change the first uncached token while preserving prompt length."""
    if not 0 < split < len(base_ids):
        raise ValueError("partial split must be inside the prompt")
    result = list(base_ids)
    replacement = _different_token(tokenizer, base_ids[split])
    # Vary the replacement deterministically across rows without changing the
    # common prefix or the total prompt length.
    if variant % 2:
        replacement = _different_token(tokenizer, replacement)
    result[split] = replacement
    return result


def _request_args(args: argparse.Namespace) -> Namespace:
    """Build the small argument surface consumed by the shared HTTP helper."""
    return Namespace(
        mode="makv",
        scout_overlap=True,
        url=args.url,
        model=args.model,
        max_tokens=args.max_tokens,
        generation_seed=args.generation_seed,
        timeout=args.timeout,
    )


def _health_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Return manager and Scout counter deltas for one request."""
    before_scout = before.get("scout") or {}
    after_scout = after.get("scout") or {}
    before_metrics = before_scout.get("metrics") or {}
    after_metrics = after_scout.get("metrics") or {}
    metric_names = (
        "submitted_jobs",
        "deduplicated_submissions",
        "cross_request_deduplicated_submissions",
        "prompt_hash_cache_hits",
        "completed_jobs",
        "failed_jobs",
        "rejected_jobs",
        "queue_time_ms",
        "score_time_ms",
        "total_job_time_ms",
        "exposed_wait_time_ms",
        "background_wait_time_ms",
        "overlap_hidden_time_ms",
    )
    return {
        "manager_quantize_calls_delta": int(after.get("quantize_calls", 0))
        - int(before.get("quantize_calls", 0)),
        "scout": {
            name: after_metrics.get(name, 0) - before_metrics.get(name, 0)
            for name in metric_names
        },
    }


_LMCacheLogEvent = re.compile(
    r"Reqid: (?P<request_id>[^,]+), Total tokens (?P<total>\d+).*"
    r"LMCache hit tokens: (?P<hit>\d+),"
)


def _read_lmcache_events(log_path: str | None) -> list[dict[str, Any]]:
    """Read per-request cache counters from the vLLM adapter log."""
    if not log_path or not Path(log_path).is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in Path(log_path).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        match = _LMCacheLogEvent.search(line)
        if match is not None:
            events.append(
                {
                    "request_id": match.group("request_id"),
                    "total_tokens": int(match.group("total")),
                    "hit": int(match.group("hit")),
                }
            )
    return events


def _wait_for_request_event(
    args: argparse.Namespace, previous_count: int
) -> dict[str, Any] | None:
    """Wait until vLLM logs the cache lookup for the current request."""
    if not args.vllm_log:
        return None
    deadline = time.monotonic() + args.manager_wait_timeout
    while time.monotonic() < deadline:
        events = _read_lmcache_events(args.vllm_log)
        if len(events) > previous_count:
            return events[previous_count]
        time.sleep(args.manager_wait_interval)
    raise TimeoutError("vLLM did not log the current LMCache lookup")


def _wait_for_local_submit(
    args: argparse.Namespace, request_id: str
) -> None:
    """Wait until vLLM has handed the request's deferred objects to MaKV."""
    marker = f"[req_id={request_id}] Submitted deferred MaKV PUT"
    deadline = time.monotonic() + args.manager_wait_timeout
    last_count = 0
    last_change = time.monotonic()
    while time.monotonic() < deadline:
        log_path = Path(args.vllm_log)
        if log_path.is_file():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            count = text.count(marker)
            if count > last_count:
                last_count = count
                last_change = time.monotonic()
            if count and time.monotonic() - last_change >= args.submit_quiet_seconds:
                return
        time.sleep(args.manager_wait_interval)
    raise TimeoutError(
        f"vLLM did not submit deferred MaKV PUT for request {request_id}"
    )


def _wait_for_idle(
    args: argparse.Namespace, minimum_put_requests: int
) -> dict[str, Any]:
    """Wait for local deferred submission, manager PUT, and Scout completion."""
    deadline = time.monotonic() + args.manager_wait_timeout
    stable = 0
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = manager_health(args.manager_health, args.timeout)
        scout = last.get("scout") or {}
        metrics = last.get("metrics") or {}
        put_requests = int(metrics.get("makv_remote_put_requests", 0))
        idle = (
            put_requests >= minimum_put_requests
            and int(last.get("queue_size", 0)) == 0
            and int(last.get("active_jobs", 0)) == 0
            and int(scout.get("pending_jobs", 0)) == 0
        )
        if idle:
            stable += 1
            if stable >= 3:
                return last
        else:
            stable = 0
        time.sleep(args.manager_wait_interval)
    raise TimeoutError(f"MaKV manager did not become idle: {last}")


def _run_request(
    session: requests.Session,
    args: argparse.Namespace,
    request_args: Namespace,
    *,
    case: str,
    ids: list[int],
    common_prefix_tokens: int,
    expected_cached_tokens: int,
    variant: int = 0,
) -> dict[str, Any]:
    """Send one request and attach manager/Scout observations."""
    before = manager_health(args.manager_health, args.timeout)
    before_event_count = len(_read_lmcache_events(args.vllm_log))
    before_put_requests = int(
        (before.get("metrics") or {}).get("makv_remote_put_requests", 0)
    )
    response = request_completion(
        session,
        args=request_args,
        ids=ids,
        max_tokens=args.max_tokens,
    )
    # Store-side deferred work and the Scout wait are part of the request's
    # dependency boundary. A health read after the response captures them.
    event = _wait_for_request_event(args, before_event_count)
    if event is not None and case != "exact_full_match":
        _wait_for_local_submit(args, str(event["request_id"]))
    minimum_put_requests = before_put_requests + (
        0 if case == "exact_full_match" else 1
    )
    _wait_for_idle(args, minimum_put_requests)
    after = manager_health(args.manager_health, args.timeout)
    cached_tokens = (
        event["hit"] if event is not None else response.get("cached_tokens")
    )
    if event is not None and int(event["total_tokens"]) != len(ids):
        raise RuntimeError(
            f"vLLM logged {event['total_tokens']} tokens for a {len(ids)}-token request"
        )
    manager_delta = _health_delta(before, after)
    submitted_jobs = int(manager_delta["scout"]["submitted_jobs"])
    cacheable_end = (
        len(ids)
        if args.save_unfull_chunk
        else len(ids) // args.chunk_size * args.chunk_size
    )
    score_start = min(max(expected_cached_tokens, 0), cacheable_end)
    score_start = score_start // args.chunk_size * args.chunk_size
    score_end = cacheable_end
    scout_submitted = submitted_jobs > 0
    return {
        "case": case,
        "variant": variant,
        "prompt_tokens": len(ids),
        "common_prefix_tokens": common_prefix_tokens,
        "expected_cached_tokens": expected_cached_tokens,
        "cached_tokens": cached_tokens,
        "cached_tokens_source": (
            "vllm_lmcache_log"
            if event is not None
            else response.get("cached_tokens_source")
        ),
        "cache_match_ok": (
            cached_tokens is not None and int(cached_tokens) >= expected_cached_tokens
        ),
        "ttft_ms": response.get("ttft_ms"),
        "latency_ms": response.get("latency_ms"),
        "vllm_request_id": event["request_id"] if event is not None else None,
        "vllm_logged_prompt_tokens": (
            event["total_tokens"] if event is not None else None
        ),
        "manager_pid": after.get("pid"),
        "manager_quantize_calls": after.get("quantize_calls"),
        "scout_submitted": scout_submitted,
        "scout_score_start": score_start if scout_submitted else None,
        "scout_score_end": score_end if scout_submitted else None,
        "scout_input_tokens": (
            score_end - score_start if scout_submitted else 0
        ),
        "manager_delta": manager_delta,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the partial-prompt matching experiment."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True
    )
    examples = load_examples(
        Path(args.dataset_path), args.task, args.scan_limit, args.offset
    )
    example, base_ids = _select_longest_prompt(
        tokenizer,
        examples,
        args.prompt_run_id,
        min_tokens=args.min_prompt_tokens,
        max_tokens=args.max_prompt_tokens,
        enable_thinking=args.enable_thinking,
    )
    fractions = _parse_fractions(args.partial_fractions)
    chunk_size = int(args.chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")

    request_args = _request_args(args)
    rows: list[dict[str, Any]] = []
    with requests.Session() as session:
        rows.append(
            _run_request(
                session,
                args,
                request_args,
                case="seed_full_prompt",
                ids=base_ids,
                common_prefix_tokens=0,
                expected_cached_tokens=0,
            )
        )
        # The exact repeat is a control: it should load all complete chunks and
        # must not submit a duplicate Scout job merely because the request ID
        # is different.
        rows.append(
            _run_request(
                session,
                args,
                request_args,
                case="exact_full_match",
                ids=base_ids,
                common_prefix_tokens=len(base_ids),
                expected_cached_tokens=(len(base_ids) // chunk_size) * chunk_size,
            )
        )
        for variant, fraction in enumerate(fractions):
            split = int(len(base_ids) * fraction) // chunk_size * chunk_size
            split = max(chunk_size, min(split, len(base_ids) - chunk_size))
            partial_ids = _make_partial_prompt(base_ids, split, tokenizer, variant)
            rows.append(
                _run_request(
                    session,
                    args,
                    request_args,
                    case=f"partial_prefix_{fraction:g}",
                    ids=partial_ids,
                    common_prefix_tokens=split,
                    expected_cached_tokens=split,
                    variant=variant,
                )
            )
        # A fully different prompt validates that the observed hit is due to
        # token-prefix identity rather than a manager-wide warm state.
        no_match = list(base_ids)
        for index in range(min(chunk_size, len(no_match))):
            no_match[index] = _different_token(tokenizer, no_match[index])
        rows.append(
            _run_request(
                session,
                args,
                request_args,
                case="no_prefix_match",
                ids=no_match,
                common_prefix_tokens=0,
                expected_cached_tokens=0,
                variant=len(fractions),
            )
        )

    scout_submissions = [
        int(row["manager_delta"]["scout"]["submitted_jobs"]) for row in rows
    ]
    result = {
        "experiment": "makv_partial_prompt_match",
        "production_path": (
            "partial requests submit only the uncached cacheable suffix to "
            "ScoutRank with an absolute score range"
        ),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "task": args.task,
        "source_example_id": example.example_id,
        "source_prompt_tokens": len(base_ids),
        "chunk_size": chunk_size,
        "partial_fractions": fractions,
        "prompt_run_id": args.prompt_run_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scout_cuda_visible_devices": os.environ.get("SCOUT_CUDA_VISIBLE_DEVICES"),
        "manager_pid": rows[0]["manager_pid"] if rows else None,
        "scout_submissions_per_row": scout_submissions,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--manager-health", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--task", default="hotpotqa")
    parser.add_argument("--scan-limit", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--min-prompt-tokens", type=int, default=10000)
    parser.add_argument("--max-prompt-tokens", type=int, default=24000)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--save-unfull-chunk",
        action="store_true",
        help="Include the final partial chunk in the ScoutRank score range.",
    )
    parser.add_argument("--partial-fractions", default="0.25,0.50,0.75")
    parser.add_argument("--prompt-run-id", default="makv-partial-prompt-match")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--vllm-log",
        default=None,
        help="vLLM log used to recover per-request LMCache hit counts.",
    )
    parser.add_argument("--manager-wait-timeout", type=float, default=900.0)
    parser.add_argument("--manager-wait-interval", type=float, default=0.25)
    parser.add_argument("--submit-quiet-seconds", type=float, default=5.0)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser


def main() -> None:
    """Run the experiment and print its JSON summary."""
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
