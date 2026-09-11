#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a reproducible LongBench cold-vs-cache-hit comparison.

LongBench is multi-task.  The default evaluator is the official LongBench
metric adapter; the legacy substring scorer remains available with
``--evaluator fast`` for regression comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import math
import re
import statistics
import socket
import struct
import time

import requests
from transformers import AutoTokenizer


@dataclass(frozen=True)
class LongBenchExample:
    example_id: str
    context: str
    question: str
    answers: tuple[str, ...]
    task: str
    length: int | None = None
    all_classes: tuple[str, ...] = ()


def _official_api() -> tuple[Any, Any, Any]:
    """Load the optional official evaluator and return its public helpers."""
    try:
        from longbench_official.adapter import (
            OFFICIAL_COMMIT,
            metric_name,
            official_score,
        )
    except ImportError as first_error:
        try:
            from benchmarks.longbench_official.adapter import (
                OFFICIAL_COMMIT,
                metric_name,
                official_score,
            )
        except ImportError as second_error:
            raise RuntimeError(
                "Official LongBench evaluation is unavailable. Install "
                "LMCache/requirements/longbench.txt in the active environment."
            ) from (second_error or first_error)
    return OFFICIAL_COMMIT, metric_name, official_score


def load_examples(
    path: Path, task: str, limit: int, offset: int
) -> list[LongBenchExample]:
    rows: list[LongBenchExample] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < offset:
                continue
            if limit >= 0 and len(rows) >= limit:
                break
            item = json.loads(line)
            raw_length = item.get("length")
            try:
                length = int(raw_length) if raw_length is not None else None
            except (TypeError, ValueError):
                length = None
            if length is not None and length < 0:
                length = None
            raw_classes = item.get("all_classes") or []
            all_classes = (
                tuple(str(value) for value in raw_classes)
                if isinstance(raw_classes, (list, tuple))
                else ()
            )
            rows.append(
                LongBenchExample(
                    example_id=str(item.get("_id", index)),
                    context=str(item.get("context", "")),
                    question=str(item.get("input", "")),
                    answers=tuple(str(answer) for answer in item.get("answers", [])),
                    task=task,
                    length=length,
                    all_classes=all_classes,
                )
            )
    return rows


def _longbench_user_messages(
    example: LongBenchExample,
    run_id: str,
) -> list[dict[str, str]]:
    """Build LongBench's single-user chat message deterministically."""
    case_id = hashlib.sha256(f"{run_id}:{example.example_id}".encode()).hexdigest()
    prompt = (
        f"Evaluation case identifier: {case_id}.\n"
        "Use the context to answer the question. Keep the answer concise.\n\n"
        f"Context:\n{example.context}\n\nQuestion:\n{example.question}\n\n"
        "Answer:"
    )
    return [{"role": "user", "content": prompt}]


def _input_ids_from_chat_encoding(encoded: Any) -> list[int]:
    """Normalize a tokenizer chat-template result to one unbatched ID list."""
    ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("LongBench runner only supports batch size 1")
        ids = ids[0]
    return [int(value) for value in ids]


def prompt_ids(
    tokenizer: Any,
    example: LongBenchExample,
    run_id: str,
    *,
    enable_thinking: bool = False,
) -> list[int]:
    """Encode the exact prompt sent to vLLM.

    Qwen3 enables reasoning in its chat template by default.  LongBench's
    extractive QA metrics expect an answer, not an unfinished reasoning trace,
    so answer-only is the default and reasoning must be explicitly requested.
    """
    encoded = tokenizer.apply_chat_template(
        _longbench_user_messages(example, run_id),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return _input_ids_from_chat_encoding(encoded)


def prompt_ids_with_last_user_span(
    tokenizer: Any,
    example: LongBenchExample,
    run_id: str,
    *,
    enable_thinking: bool = False,
) -> tuple[list[int], tuple[int, int] | None]:
    """Encode LongBench prompt IDs and its exact final user-turn span.

    LongBench constructs one user message. For the Qwen chat template, the
    encoding without the generation prompt must be an exact prefix of the
    generation-ready prompt; otherwise this function returns ``None`` rather
    than guessing a span.
    """
    messages = _longbench_user_messages(example, run_id)
    full_ids = _input_ids_from_chat_encoding(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    )
    user_turn_ids = _input_ids_from_chat_encoding(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
    )
    if user_turn_ids and full_ids[: len(user_turn_ids)] == user_turn_ids:
        return full_ids, (0, len(user_turn_ids))
    return full_ids, None


def importance(token_count: int) -> list[float]:
    """Return the legacy deterministic placeholder for explicit smoke tests."""
    return [float((index * 37) % 1009) / 1009.0 for index in range(token_count)]


def prompt_token_hash(ids: list[int]) -> str:
    """Hash token IDs using the same representation as ScoutRank artifacts."""
    return hashlib.sha256(",".join(str(value) for value in ids).encode()).hexdigest()


# JSON (and the vLLM HTTP request model) cannot carry NaN/Inf.  Keep the
# value inside float32 range so the server-side plan builder sees a finite
# score; the derived status below preserves the fail-closed BF16 assignment.
_JSON_SAFE_NONFINITE_IMPORTANCE = float.fromhex("0x1.fffffep+127")


def _json_safe_importance(value: Any) -> float:
    """Encode a non-finite ScoutRank safety score for the JSON transport."""
    value = float(value)
    return value if math.isfinite(value) else _JSON_SAFE_NONFINITE_IMPORTANCE


def load_importance_file(path: str | None) -> dict[str, list[float]]:
    """Load prompt-hash keyed token importance scores from a JSON artifact."""
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scores = payload.get("scores", payload) if isinstance(payload, dict) else None
    if not isinstance(scores, dict):
        raise ValueError("importance file must contain a hash->score-list map")
    result: dict[str, list[float]] = {}
    for key, values in scores.items():
        if not isinstance(values, list):
            raise ValueError(f"importance scores for {key!r} must be a list")
        result[str(key)] = [_json_safe_importance(value) for value in values]
    return result


def load_importance_status_file(
    path: str | None,
) -> dict[str, list[dict[str, Any]]]:
    """Load v3.2 token validity and forced-BF16 status from an artifact."""
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("importance status file must contain an object")
    direct = payload.get("token_status")
    attention_results = payload.get("attention_results")
    source = direct if isinstance(direct, dict) else attention_results
    if source is None:
        # The exact D22 artifact intentionally represents uncovered tokens as
        # +Inf, but older artifacts do not persist a separate status map.
        # Reconstruct the equivalent fail-closed status before the vector is
        # sent over JSON.  This keeps those tokens out of ratio allocation.
        raw_scores = payload.get("scores")
        if not isinstance(raw_scores, dict):
            return {}
        derived: dict[str, list[dict[str, Any]]] = {}
        for key, values in raw_scores.items():
            if not isinstance(values, list):
                continue
            if not any(not math.isfinite(float(value)) for value in values):
                continue
            derived[str(key)] = [
                {
                    "valid_mask": math.isfinite(float(value)),
                    "forced_precision": (
                        None
                        if math.isfinite(float(value))
                        else "BF16"
                    ),
                    "reason": (
                        None
                        if math.isfinite(float(value))
                        else "NONFINITE_IMPORTANCE"
                    ),
                }
                for value in values
            ]
        return derived
    if not isinstance(source, dict):
        raise ValueError("importance status map must be hash-keyed")
    result: dict[str, list[dict[str, Any]]] = {}
    for key, item in source.items():
        details = item.get("attention", item) if isinstance(item, dict) else None
        if not isinstance(details, dict):
            raise ValueError(f"importance status for {key!r} must be an object")
        statuses = details.get("token_status")
        if statuses is None:
            valid_mask = details.get("valid_mask")
            forced = details.get("forced_precision_by_token")
            reasons = details.get("reason_by_token")
            if not isinstance(valid_mask, list):
                continue
            if forced is None:
                forced = [None] * len(valid_mask)
            if reasons is None:
                reasons = [None] * len(valid_mask)
            if (
                not isinstance(forced, list)
                or not isinstance(reasons, list)
                or len(forced) != len(valid_mask)
                or len(reasons) != len(valid_mask)
            ):
                raise ValueError(f"importance status lengths mismatch for {key!r}")
            statuses = [
                {
                    "valid_mask": bool(valid),
                    "forced_precision": precision,
                    "reason": reason,
                }
                for valid, precision, reason in zip(
                    valid_mask, forced, reasons, strict=True
                )
            ]
        if not isinstance(statuses, list):
            raise ValueError(f"importance status for {key!r} must be a list")
        result[str(key)] = statuses
    return result


def load_precision_plan_file(path: str | None) -> dict[str, dict[str, Any]]:
    """Load prompt-hash keyed explicit MaKV precision plans from JSON.

    A ScoutRank-v3 artifact generated with ``--v3-mckp-budget-bytes`` contains
    a top-level ``precision_plans`` map. The plans remain opaque here because
    MaKV validates their token hash, schema, and deployment gate at PUT time.
    """
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    plans = payload.get("precision_plans") if isinstance(payload, dict) else None
    if not isinstance(plans, dict):
        raise ValueError("precision-plan file must contain a hash->plan map")
    result: dict[str, dict[str, Any]] = {}
    for key, plan in plans.items():
        if not isinstance(plan, dict):
            raise ValueError(f"precision plan for {key!r} must be an object")
        result[str(key)] = plan
    return result


def load_importance_timing(path: str | None) -> dict[str, float]:
    """Load per-prompt ScoutRank timing from a timed importance artifact."""
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(metadata, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in metadata.items():
        if not isinstance(item, dict):
            continue
        value = item.get("scoutrank_time_ms")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value) and value >= 0.0:
                result[str(key)] = value
    return result


def extract_answer_text(text: str) -> str:
    """Remove a Qwen reasoning trace before extractive answer evaluation."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text:
        # An unfinished reasoning trace has no reliable final answer.
        return ""
    return text.strip()


def _nonnegative_int(value: Any) -> int | None:
    """Return a valid token count, treating missing/sentinel values as unknown."""
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def extract_cached_tokens(body: dict[str, Any]) -> tuple[int | None, str | None]:
    """Extract LMCache hit tokens from current and legacy response shapes.

    The latest LMCache adapter returns num_lmcache_cached_tokens directly,
    while the multiprocessing adapter wraps it in cached_token_stats.
    usage.prompt_tokens_details is a last-resort vLLM fallback; prefix
    caching is disabled by the benchmark, so it represents external hits here.
    """
    transfer = body.get("kv_transfer_params")
    if isinstance(transfer, dict):
        stats = transfer.get("cached_token_stats")
        if isinstance(stats, dict):
            value = _nonnegative_int(stats.get("num_lmcache_cached_tokens"))
            if value is not None:
                return value, "lmcache"
        value = _nonnegative_int(transfer.get("num_lmcache_cached_tokens"))
        if value is not None:
            return value, "lmcache"

    usage = body.get("usage")
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            value = _nonnegative_int(details.get("cached_tokens"))
            if value is not None:
                return value, "vllm_usage"
    return None, None


def request_completion(
    session: requests.Session,
    *,
    args: argparse.Namespace,
    ids: list[int],
    stream: bool = True,
    max_tokens: int | None = None,
    importance_values: list[float] | None = None,
    importance_status: list[dict[str, Any]] | None = None,
    precision_plan: dict[str, Any] | None = None,
    risk_token_indices: list[int] | None = None,
    runtime_risk_enabled: bool = True,
) -> dict[str, Any]:
    transfer: dict[str, Any] = {"cached_token_stats": True}
    if args.mode == "makv" and not getattr(args, "scout_overlap", False):
        if precision_plan is not None:
            transfer["lmcache.makv_precision_plan"] = precision_plan
            # MaKV validates the plan against the exact prompt IDs reaching
            # the worker; never trust a hash copied from an artifact.
            # Keep the hash in request configs so LMCache can carry it across
            # chunk alignment, where the stored token slice may be shorter
            # than the complete prompt.
            full_prompt_hash = prompt_token_hash(ids)
            transfer["prompt_token_hash"] = full_prompt_hash
            transfer["lmcache.prompt_token_hash"] = full_prompt_hash
            # The serializer may receive one cache tensor per chunk. Preserve
            # the request-wide length so v3 slices the global assignment.
            transfer["request_token_count"] = len(ids)
        elif importance_values is None:
            if getattr(args, "require_importance_file", False):
                raise ValueError(
                    "MaKV validation requires --importance-file or "
                    "--precision-plan-file; refusing to use the placeholder "
                    "importance vector"
                )
            importance_values = importance(len(ids))
        if precision_plan is None:
            transfer.update(
                {
                    "lmcache.makv_importance": importance_values,
                    "lmcache.makv_importance_layout": "token",
                }
            )
            if importance_status is not None:
                transfer["lmcache.makv_importance_status"] = importance_status
        if getattr(args, "risk_source", "synthetic") == "runtime_conf":
            if not risk_token_indices:
                raise ValueError(
                    "runtime_conf risk source requires explicit risk_token_indices"
                )
            transfer["lmcache.makv.risk_token_indices"] = [
                int(index) for index in risk_token_indices
            ]
            transfer["makv_risk_observer_enabled"] = bool(runtime_risk_enabled)
    started = time.perf_counter()
    response = session.post(
        args.url,
        json={
            "model": args.model,
            "prompt": ids,
            "max_tokens": args.max_tokens if max_tokens is None else max_tokens,
            "temperature": 0.0,
            "seed": args.generation_seed,
            "stream": stream,
            "stream_options": {"include_usage": True} if stream else None,
            "kv_transfer_params": transfer,
        },
        timeout=args.timeout,
        stream=stream,
    )
    if not response.ok:
        detail = response.text[:2000]
        raise requests.HTTPError(
            f"{response.status_code} from {response.url}: {detail}", response=response
        )
    text_parts: list[str] = []
    first_token_at: float | None = None
    cached_tokens: int | None = None
    cached_tokens_source: str | None = None

    def update_cached_tokens(body: dict[str, Any]) -> None:
        nonlocal cached_tokens, cached_tokens_source
        value, source = extract_cached_tokens(body)
        if value is None:
            return
        # Prefer the explicit LMCache field over generic vLLM usage data.
        if cached_tokens_source != "lmcache" or source == "lmcache":
            cached_tokens = value
            cached_tokens_source = source

    if stream:
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            body = json.loads(data)
            choices = body.get("choices") or []
            if choices:
                piece = choices[0].get("text") or ""
                if piece and first_token_at is None:
                    first_token_at = time.perf_counter()
                text_parts.append(piece)
            update_cached_tokens(body)
    else:
        body = response.json()
        text_parts.append(((body.get("choices") or [{}])[0]).get("text") or "")
        update_cached_tokens(body)
    finished = time.perf_counter()
    return {
        "text": "".join(text_parts),
        "latency_ms": (finished - started) * 1000,
        "ttft_ms": (first_token_at - started) * 1000 if first_token_at else None,
        "cached_tokens": cached_tokens,
        "cached_tokens_source": cached_tokens_source,
    }


def attach_scoutrank_timing(
    response: dict[str, Any], scoutrank_time_ms: float
) -> None:
    """Add online-equivalent TTFT with the offline ScoutRank cost included."""
    response["scoutrank_time_ms"] = scoutrank_time_ms
    ttft_ms = response.get("ttft_ms")
    response["ttft_with_scoutrank_ms"] = (
        ttft_ms + scoutrank_time_ms if ttft_ms is not None else None
    )


_MANAGER_FRAME_HEADER = struct.Struct("!IQ")


def _read_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("MaKV manager closed an incomplete health response")
        data.extend(chunk)
    return bytes(data)


def manager_health(address: str, timeout: float) -> dict[str, Any]:
    """Read the independent manager health state without using vLLM."""
    host, port_text = address.rsplit(":", 1)
    header = json.dumps({"op": "HEALTH", "key": ""}, separators=(",", ":")).encode()
    with socket.create_connection(
        (host, int(port_text)), timeout=timeout
    ) as connection:
        connection.sendall(_MANAGER_FRAME_HEADER.pack(len(header), 0) + header)
        raw = _read_exact(connection, _MANAGER_FRAME_HEADER.size)
        header_len, payload_len = _MANAGER_FRAME_HEADER.unpack(raw)
        response = json.loads(_read_exact(connection, header_len).decode("utf-8"))
        if payload_len:
            _read_exact(connection, payload_len)
    if response.get("status") != "ok":
        raise RuntimeError(str(response.get("error", "MaKV manager health failed")))
    return response


def wait_for_manager_idle(args: argparse.Namespace) -> None:
    """Wait until all asynchronous PUTs have been durably processed."""
    if not args.manager_health:
        return
    deadline = time.monotonic() + args.manager_wait_timeout
    last_state: dict[str, Any] | None = None
    stable_idle_observations = 0
    previous_put_requests: int | None = None
    while time.monotonic() < deadline:
        last_state = manager_health(args.manager_health, args.timeout)
        scout = last_state.get("scout")
        scout_pending = (
            int(scout.get("pending_jobs", 0))
            if isinstance(scout, dict) and scout.get("enabled")
            else 0
        )
        metrics = last_state.get("metrics")
        put_requests = (
            int(metrics.get("makv_remote_put_requests", 0))
            if isinstance(metrics, dict)
            else 0
        )
        idle = (
            int(last_state.get("queue_size", 0)) == 0
            and int(last_state.get("active_jobs", 0)) == 0
            and scout_pending == 0
        )
        if idle and put_requests == previous_put_requests:
            stable_idle_observations += 1
            # Deferred local serialization is not visible to manager health.
            # Requiring three stable polls prevents a transient empty remote
            # queue from being mistaken for a durable cache-ready state.
            if stable_idle_observations >= 3:
                return
        else:
            stable_idle_observations = 0
        previous_put_requests = put_requests
        time.sleep(args.manager_wait_interval)
    raise TimeoutError(f"MaKV manager did not become idle: {last_state}")


def _normalize(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def redis_bytes(url: str | None) -> int:
    if not url:
        return 0
    import redis

    client = redis.Redis.from_url(url)
    return sum(
        int(client.strlen(key))
        for key in client.scan_iter(match="*")
        if client.type(key) == b"string"
    )


def score(text: str, answers: tuple[str, ...], task: str) -> bool | None:
    """Answer-hit score for extractive QA/classification tasks.

    LongBench summarization/code-generation tasks are intentionally excluded
    from exact accuracy; their raw outputs remain in the JSONL for ROUGE or
    task-specific official evaluation.
    """
    if task in {"gov_report", "qmsum", "multi_news", "vcsum", "samsum"}:
        return None
    normalized = _normalize(text)
    if not normalized or not answers:
        return False
    normalized_answers = [_normalize(answer) for answer in answers]
    if task in {"trec", "trec_e", "lsht", "passage_count", "passage_count_e"}:
        first_line = _normalize(text.splitlines()[0])
        return any(
            first_line == answer or answer in first_line
            for answer in normalized_answers
        )
    return any(answer and answer in normalized for answer in normalized_answers)


def official_metric_score(
    text: str,
    answers: tuple[str, ...],
    task: str,
    all_classes: tuple[str, ...] = (),
) -> float | None:
    """Return one prediction's official LongBench score in ``[0, 1]``."""
    _, _, scorer = _official_api()
    return scorer(text, answers, task, all_classes)


def official_metric_name(task: str) -> str:
    """Return the official metric name for a LongBench task."""
    _, metric, _ = _official_api()
    return metric(task)


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def summarize(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    evaluator = getattr(args, "evaluator", "fast")
    valid = [row for row in records if row["valid"]]
    scored = [row for row in valid if row["cold"]["correct"] is not None]
    hit_scored = [row for row in valid if row["hit"]["correct"] is not None]
    cold_ttft = [
        row["cold"]["ttft_ms"] for row in valid if row["cold"]["ttft_ms"] is not None
    ]
    hit_ttft = [
        row["hit"]["ttft_ms"] for row in valid if row["hit"]["ttft_ms"] is not None
    ]
    cold_latencies = [
        row["cold"]["latency_ms"]
        for row in valid
        if row["cold"]["latency_ms"] is not None
    ]
    hit_latencies = [
        row["hit"]["latency_ms"]
        for row in valid
        if row["hit"]["latency_ms"] is not None
    ]
    scoutrank_times = [
        float(row.get("scoutrank_time_ms", 0.0))
        for row in valid
        if isinstance(row.get("scoutrank_time_ms", 0.0), (int, float))
    ]
    cold_ttft_with_scoutrank = [
        row["cold"].get("ttft_with_scoutrank_ms")
        for row in valid
        if row["cold"].get("ttft_with_scoutrank_ms") is not None
    ]
    hit_ttft_with_scoutrank = [
        row["hit"].get("ttft_with_scoutrank_ms")
        for row in valid
        if row["hit"].get("ttft_with_scoutrank_ms") is not None
    ]
    raw_bytes = (
        sum(row["prompt_tokens"] for row in valid)
        * args.model_layers
        * 2
        * args.model_kv_heads
        * args.model_head_dim
        * args.model_dtype_bytes
    )
    stored_bytes = 0
    if args.mode in {"cachegen", "naive"} and args.redis_url:
        stored_bytes = redis_bytes(args.redis_url)
    elif args.storage_dir:
        stored_bytes = sum(
            path.stat().st_size
            for path in Path(args.storage_dir).rglob("*")
            if path.is_file()
        )

    def accuracy(rows: list[dict[str, Any]], key: str) -> float | None:
        return (
            sum(bool(row[key]["correct"]) for row in rows) / len(rows) if rows else None
        )

    def official_value(row: dict[str, Any], key: str) -> float | None:
        value = row.get(key, {}).get("official_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if evaluator != "official":
            return None
        return official_metric_score(
            str(row.get(key, {}).get("answer_text", row.get(key, {}).get("text", ""))),
            tuple(str(answer) for answer in row.get("answers", [])),
            args.task,
            tuple(str(value) for value in row.get("all_classes", [])),
        )

    cold_official_scores = [
        value for row in valid if (value := official_value(row, "cold")) is not None
    ]
    hit_official_scores = [
        value for row in valid if (value := official_value(row, "hit")) is not None
    ]
    cold_official = (
        sum(cold_official_scores) / len(cold_official_scores)
        if cold_official_scores
        else None
    )
    hit_official = (
        sum(hit_official_scores) / len(hit_official_scores)
        if hit_official_scores
        else None
    )
    official_degradation = sum(
        official_value(row, "cold") is not None
        and official_value(row, "hit") is not None
        and official_value(row, "cold") > official_value(row, "hit")
        for row in valid
    )
    legacy_cold_accuracy = accuracy(scored, "cold")
    legacy_hit_accuracy = accuracy(hit_scored, "hit")
    primary_cold_accuracy = (
        cold_official if evaluator == "official" else legacy_cold_accuracy
    )
    primary_hit_accuracy = (
        hit_official if evaluator == "official" else legacy_hit_accuracy
    )
    scoutrank_time_mean = statistics.fmean(scoutrank_times) if scoutrank_times else 0.0

    def scoutrank_share(ttft_mean: float | None) -> float:
        if ttft_mean is None:
            return 0.0
        total = ttft_mean + scoutrank_time_mean
        return 100 * scoutrank_time_mean / total if total else 0.0

    summary: dict[str, Any] = {
        "mode": args.mode,
        "task": args.task,
        "total": len(records),
        "valid_complete_hits": len(valid),
        "evaluator": evaluator,
        "scored_samples": (
            len(cold_official_scores) if evaluator == "official" else len(scored)
        ),
        "cold_accuracy": primary_cold_accuracy,
        "hit_accuracy": primary_hit_accuracy,
        "legacy_scored_samples": len(scored),
        "legacy_cold_accuracy": legacy_cold_accuracy,
        "legacy_hit_accuracy": legacy_hit_accuracy,
        "official_metric": (
            official_metric_name(args.task) if evaluator == "official" else None
        ),
        "official_cold_score": cold_official,
        "official_hit_score": hit_official,
        "official_cold_score_percent": (
            round(100 * cold_official, 2) if cold_official is not None else None
        ),
        "official_hit_score_percent": (
            round(100 * hit_official, 2) if hit_official is not None else None
        ),
        # Keep both spellings while downstream benchmark parsers migrate to
        # the clearer ``<side>_official_score_percent`` convention.
        "cold_official_score_percent": (
            round(100 * cold_official, 2) if cold_official is not None else None
        ),
        "hit_official_score_percent": (
            round(100 * hit_official, 2) if hit_official is not None else None
        ),
        "official_scored_samples": len(cold_official_scores),
        "official_hit_scored_samples": len(hit_official_scores),
        "official_cold_to_hit_degradation_samples": official_degradation,
        "cold_correct_to_hit_wrong": sum(
            row["cold"]["correct"] and not row["hit"]["correct"] for row in scored
        ),
        "cold_ttft_mean_ms": statistics.fmean(cold_ttft) if cold_ttft else None,
        "cold_ttft_median_ms": statistics.median(cold_ttft) if cold_ttft else None,
        "cold_ttft_p95_ms": _p95(cold_ttft),
        "hit_ttft_mean_ms": statistics.fmean(hit_ttft) if hit_ttft else None,
        "hit_ttft_median_ms": statistics.median(hit_ttft) if hit_ttft else None,
        "hit_ttft_p95_ms": _p95(hit_ttft),
        "scoutrank_timed_samples": sum(
            float(row.get("scoutrank_time_ms", 0.0)) > 0.0 for row in valid
        ),
        "scoutrank_time_mean_ms": scoutrank_time_mean,
        "scoutrank_time_median_ms": (
            statistics.median(scoutrank_times) if scoutrank_times else 0.0
        ),
        "scoutrank_time_p95_ms": _p95(scoutrank_times) or 0.0,
        "cold_ttft_with_scoutrank_mean_ms": (
            statistics.fmean(cold_ttft_with_scoutrank)
            if cold_ttft_with_scoutrank
            else None
        ),
        "cold_ttft_with_scoutrank_median_ms": (
            statistics.median(cold_ttft_with_scoutrank)
            if cold_ttft_with_scoutrank
            else None
        ),
        "cold_ttft_with_scoutrank_p95_ms": _p95(cold_ttft_with_scoutrank),
        "hit_ttft_with_scoutrank_mean_ms": (
            statistics.fmean(hit_ttft_with_scoutrank)
            if hit_ttft_with_scoutrank
            else None
        ),
        "hit_ttft_with_scoutrank_median_ms": (
            statistics.median(hit_ttft_with_scoutrank)
            if hit_ttft_with_scoutrank
            else None
        ),
        "hit_ttft_with_scoutrank_p95_ms": _p95(hit_ttft_with_scoutrank),
        "scoutrank_share_of_cold_ttft_pct": scoutrank_share(
            statistics.fmean(cold_ttft) if cold_ttft else None
        ),
        "scoutrank_share_of_hit_ttft_pct": scoutrank_share(
            statistics.fmean(hit_ttft) if hit_ttft else None
        ),
        "cold_latency_mean_ms": (
            statistics.fmean(cold_latencies) if cold_latencies else None
        ),
        "cold_latency_median_ms": (
            statistics.median(cold_latencies) if cold_latencies else None
        ),
        "cold_latency_p95_ms": _p95(cold_latencies),
        "hit_latency_mean_ms": (
            statistics.fmean(hit_latencies) if hit_latencies else None
        ),
        "hit_latency_median_ms": (
            statistics.median(hit_latencies) if hit_latencies else None
        ),
        "hit_latency_p95_ms": _p95(hit_latencies),
        "raw_kv_bytes": raw_bytes,
        "remote_stored_bytes": stored_bytes,
        "compression_ratio": raw_bytes / stored_bytes if stored_bytes else None,
    }
    if evaluator == "official":
        summary["official_commit"] = _official_api()[0]
        summary["official_score_delta_percent"] = (
            round(100 * (hit_official - cold_official), 2)
            if cold_official is not None and hit_official is not None
            else None
        )
    return summary


def makv_manager_timing_summary(health: dict[str, Any]) -> dict[str, Any]:
    """Extract PUT-side timing and compression shares from manager health."""
    metrics = health.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}

    def value(name: str) -> float:
        return float(metrics.get(name, 0.0))

    put_total_ms = value("makv_remote_put_total_time_ms")
    raw_bytes = int(metrics.get("makv_raw_input_bytes", 0))
    stored_bytes = int(metrics.get("makv_stored_bytes", 0))
    quantize_core_ms = value("makv_remote_quantize_kernel_time_ms")
    return {
        "put_requests": int(metrics.get("makv_remote_put_requests", 0)),
        "raw_upload_bytes": raw_bytes,
        "stored_bytes": stored_bytes,
        "remote_put_total_ms": round(put_total_ms, 3),
        "remote_put_decode_ms": round(value("makv_remote_put_decode_time_ms"), 3),
        "remote_quantize_total_ms": round(value("makv_remote_quantize_time_ms"), 3),
        # The current remote quantizer is CPU PyTorch. This field is its core
        # quantize/pack interval, not a client CUDA dequantization kernel.
        "remote_quantize_core_ms": round(quantize_core_ms, 3),
        "remote_encode_validate_ms": round(
            value("makv_remote_encode_validate_time_ms"), 3
        ),
        "remote_storage_put_ms": round(value("makv_remote_storage_put_time_ms"), 3),
        "remote_quantize_core_share_of_put_pct": round(
            100 * quantize_core_ms / put_total_ms, 2
        )
        if put_total_ms
        else 0.0,
        "remote_get_requests": int(metrics.get("makv_remote_get_requests", 0)),
        "remote_get_total_ms": round(value("makv_remote_get_total_time_ms"), 3),
        "remote_get_storage_ms": round(value("makv_remote_get_storage_time_ms"), 3),
        "remote_get_validate_ms": round(value("makv_remote_get_validate_time_ms"), 3),
        "compression_ratio": round(raw_bytes / stored_bytes, 4)
        if stored_bytes
        else None,
    }


def run(args: argparse.Namespace) -> None:
    evaluator = getattr(args, "evaluator", "official")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    examples = load_examples(
        Path(args.dataset_path), args.task, args.limit, args.offset
    )
    importance_by_hash = load_importance_file(args.importance_file)
    importance_status_by_hash = load_importance_status_file(args.importance_file)
    precision_plan_file = getattr(args, "precision_plan_file", None)
    precision_plans_by_hash = load_precision_plan_file(precision_plan_file)
    include_scoutrank_time = bool(getattr(args, "include_scoutrank_time", False))
    require_scoutrank_timing = bool(
        getattr(args, "require_scoutrank_timing", False)
    )
    importance_timing_by_hash = (
        load_importance_timing(args.importance_file)
        if include_scoutrank_time and args.mode == "makv"
        else {}
    )
    if args.mode == "makv" and args.importance_file:
        print(
            f"Loaded {len(importance_by_hash)} ScoutRank importance vectors from "
            f"{args.importance_file}",
            flush=True,
        )
        if importance_status_by_hash:
            print(
                f"Loaded {len(importance_status_by_hash)} token-status vectors from "
                f"{args.importance_file}",
                flush=True,
            )
    if args.mode == "makv" and precision_plan_file:
        print(
            f"Loaded {len(precision_plans_by_hash)} explicit MaKV precision plans "
            f"from {precision_plan_file}",
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with requests.Session() as session, output.open("w", encoding="utf-8") as log:
        for index, example in enumerate(examples):
            ids = prompt_ids(
                tokenizer,
                example,
                getattr(args, "prompt_run_id", None) or args.run_id,
                enable_thinking=args.enable_thinking,
            )
            token_hash = prompt_token_hash(ids)
            importance_values = importance_by_hash.get(token_hash)
            importance_status = importance_status_by_hash.get(token_hash)
            precision_plan = precision_plans_by_hash.get(token_hash)
            if args.mode == "makv" and importance_values is not None:
                if len(importance_values) != len(ids):
                    raise ValueError(
                        f"importance length mismatch for {example.example_id}: "
                        f"{len(importance_values)} != {len(ids)}"
                    )
            if args.mode == "makv" and precision_plan is not None:
                if int(precision_plan.get("token_count", -1)) != len(ids):
                    raise ValueError(
                        f"precision-plan length mismatch for {example.example_id}: "
                        f"{precision_plan.get('token_count')} != {len(ids)}"
                    )
            if args.mode == "makv" and importance_status is not None:
                if len(importance_status) != len(ids):
                    raise ValueError(
                        f"importance status length mismatch for {example.example_id}: "
                        f"{len(importance_status)} != {len(ids)}"
                    )
            scoutrank_time_ms = 0.0
            if include_scoutrank_time and args.mode == "makv":
                score_time = importance_timing_by_hash.get(token_hash)
                if score_time is None and require_scoutrank_timing:
                    raise ValueError(
                        "Timed ScoutRank importance is missing for "
                        f"{example.example_id}; regenerate the importance file"
                    )
                if score_time is not None:
                    scoutrank_time_ms = score_time
            cold = request_completion(
                session,
                args=args,
                ids=ids,
                importance_values=importance_values,
                importance_status=importance_status,
                precision_plan=precision_plan,
            )
            attach_scoutrank_timing(cold, scoutrank_time_ms)
            cold["answer_text"] = extract_answer_text(cold["text"])
            cold["correct"] = score(cold["answer_text"], example.answers, args.task)
            if evaluator == "official":
                cold["official_score"] = official_metric_score(
                    cold["answer_text"], example.answers, args.task, example.all_classes
                )
                cold["official_metric"] = official_metric_name(args.task)
            expected = (len(ids) // args.chunk_size) * args.chunk_size
            wait_for_manager_idle(args)
            probe = request_completion(
                session,
                args=args,
                ids=ids,
                stream=False,
                max_tokens=1,
                importance_values=importance_values,
                importance_status=importance_status,
                precision_plan=precision_plan,
            )
            for _ in range(args.hit_retries):
                if (
                    probe["cached_tokens"] is not None
                    and probe["cached_tokens"] >= expected
                ):
                    break
                wait_for_manager_idle(args)
                time.sleep(args.manager_wait_interval)
                probe = request_completion(
                    session,
                    args=args,
                    ids=ids,
                    stream=False,
                    max_tokens=1,
                    importance_values=importance_values,
                    importance_status=importance_status,
                    precision_plan=precision_plan,
                )
            if probe["cached_tokens"] is None:
                raise RuntimeError(
                    "LMCache cache statistics are missing from the vLLM response. "
                    "Add extra_config.enable_cache_usage_details_in_response=true "
                    "or use an adapter that returns cached_token_stats; refusing "
                    "to classify this request as a cache hit."
                )
            ready = probe["cached_tokens"] >= expected
            hit = request_completion(
                session,
                args=args,
                ids=ids,
                importance_values=importance_values,
                importance_status=importance_status,
                precision_plan=precision_plan,
            )
            attach_scoutrank_timing(hit, scoutrank_time_ms)
            hit["answer_text"] = extract_answer_text(hit["text"])
            hit["correct"] = score(hit["answer_text"], example.answers, args.task)
            if evaluator == "official":
                hit["official_score"] = official_metric_score(
                    hit["answer_text"], example.answers, args.task, example.all_classes
                )
                hit["official_metric"] = official_metric_name(args.task)
            hit_cache_tokens = (
                hit["cached_tokens"]
                if hit["cached_tokens"] is not None
                else probe["cached_tokens"]
            )
            hit_cache_hit = (
                hit_cache_tokens is not None and hit_cache_tokens >= expected
            )
            record = {
                "mode": args.mode,
                "task": args.task,
                "example_id": example.example_id,
                "prompt_tokens": len(ids),
                "prompt_token_hash": token_hash,
                "importance_source": (
                    "scoutrank_overlap"
                    if getattr(args, "scout_overlap", False)
                    else "scoutrank_v3_mckp_file"
                    if precision_plan is not None
                    else "scoutrank_file"
                    if importance_values is not None
                    else "placeholder"
                ),
                "importance_status_source": (
                    "scoutrank_attention_v3.2"
                    if importance_status is not None
                    else None
                ),
                "forced_bf16_token_count": (
                    sum(
                        item.get("forced_precision") == "BF16"
                        or item.get("valid_mask") is False
                        for item in importance_status
                    )
                    if importance_status is not None
                    else 0
                ),
                "scoutrank_time_ms": scoutrank_time_ms,
                "answers": list(example.answers),
                "length": example.length,
                "all_classes": list(example.all_classes),
                "evaluator": evaluator,
                "cold": cold,
                "probe": probe,
                "hit": hit,
                "cold_cache_hit": (
                    cold["cached_tokens"] is not None
                    and cold["cached_tokens"] >= expected
                ),
                "hit_cache_hit": hit_cache_hit,
                "valid": ready,
            }
            records.append(record)
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{index + 1}/{len(examples)}] id={example.example_id} "
                f"tokens={len(ids)} cold_correct={cold['correct']} "
                f"hit_correct={hit['correct']} "
                f"cold_official={cold.get('official_score')} "
                f"hit_official={hit.get('official_score')} "
                f"cold_cached={cold['cached_tokens']} "
                f"hit_cached={hit_cache_tokens}/{expected}",
                flush=True,
            )
    wait_for_manager_idle(args)
    summary = summarize(records, args)
    if args.manager_health:
        # Keep manager-side cache/quantization counters next to the latency
        # summary so a benchmark can distinguish disk misses from hot GETs.
        health = manager_health(args.manager_health, args.timeout)
        summary["manager_health"] = health
        if args.mode == "makv":
            summary["makv_manager_timing"] = makv_manager_timing_summary(health)
            scout_health = health.get("scout")
            if isinstance(scout_health, dict):
                summary["makv_scout_overlap"] = scout_health
            # File storage can be measured by walking its directory, while
            # Redis and Mooncake have no local directory to inspect. The
            # manager metric is authoritative for every adapter.
            metrics = health.get("metrics", {})
            stored_bytes = int(metrics.get("makv_stored_bytes", 0))
            if stored_bytes:
                summary["remote_stored_bytes"] = stored_bytes
                raw_bytes = int(summary["raw_kv_bytes"])
                summary["compression_ratio"] = raw_bytes / stored_bytes
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("naive", "cachegen", "makv"), required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8001/v1/completions")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--task", default="hotpotqa")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--prompt-run-id",
        default=None,
        help="Stable prompt seed shared by comparable runs; defaults to --run-id.",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--generation-seed", type=int, default=123)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 reasoning; default is answer-only chat-template encoding.",
    )
    parser.add_argument(
        "--importance-file",
        default=None,
        help="JSON artifact mapping prompt token hash to [T] importance scores.",
    )
    parser.add_argument(
        "--precision-plan-file",
        default=None,
        help=(
            "Optional ScoutRank-v3 artifact with top-level precision_plans; "
            "uses explicit MCKP assignments instead of rank buckets."
        ),
    )
    parser.add_argument(
        "--require-importance-file",
        action="store_true",
        help="Fail MaKV runs instead of using the legacy placeholder importance.",
    )
    parser.add_argument(
        "--include-scoutrank-time",
        action="store_true",
        help="Add timed ScoutRank forward/scoring cost to MaKV TTFT fields.",
    )
    parser.add_argument(
        "--require-scoutrank-timing",
        action="store_true",
        help="Fail if a MaKV importance artifact lacks per-prompt timing.",
    )
    parser.add_argument(
        "--scout-overlap",
        action="store_true",
        help=(
            "Do not send precomputed importance; let LMCache overlap the live "
            "28-layer ScoutRank sidecar with target-model prefill."
        ),
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--manager-health")
    parser.add_argument("--manager-wait-timeout", type=float, default=600)
    parser.add_argument("--manager-wait-interval", type=float, default=1.0)
    parser.add_argument("--hit-retries", type=int, default=2)
    parser.add_argument(
        "--evaluator",
        choices=("official", "fast"),
        default="official",
        help="Scoring backend; official uses the vendored LongBench metrics.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--storage-dir")
    parser.add_argument("--redis-url")
    parser.add_argument("--model-layers", type=int, default=36)
    parser.add_argument("--model-kv-heads", type=int, default=8)
    parser.add_argument("--model-head-dim", type=int, default=128)
    parser.add_argument("--model-dtype-bytes", type=int, default=2)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
