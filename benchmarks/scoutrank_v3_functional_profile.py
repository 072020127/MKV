#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile ScoutRank-v2 and v3 functional disturbance on frozen Qwen3.

The benchmark intentionally reports measured values only. It does not assert
an overhead target or infer a quality conclusion from timing results.
"""

from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any, Callable
import argparse
import gc
import json
import time
from functools import partial
import sys

# Third Party
import torch
from transformers import AutoModelForCausalLM

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from scoutrank_longbench_importance import _build_scout_runtime
except ImportError:
    from benchmarks.scoutrank_longbench_importance import _build_scout_runtime


def _synchronize(device: torch.device) -> None:
    """Synchronize CUDA before an externally visible timing boundary."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _TimedObserver:
    """Benchmark-only proxy that times observer hook work without mutation."""

    def __init__(self, observer: Any) -> None:
        self.observer = observer
        self.last_timing_ms = 0.0
        self.timing_source = "cpu_wall_clock"
        self._wall_seconds = 0.0
        self._events: list[tuple[Any, Any]] = []

    @property
    def functional_disturbance_enabled(self) -> bool:
        """Forward the adapter capability bit from the wrapped observer."""
        return bool(getattr(self.observer, "functional_disturbance_enabled", False))

    def reset(self, valid_token_mask: torch.Tensor | None = None) -> None:
        """Reset the wrapped observer and this request's timing state."""
        self._reset_timing()
        self.observer.reset(valid_token_mask)

    def configure_request(
        self,
        *,
        valid_token_mask: torch.Tensor | None,
        last_user_span: tuple[int, int] | None,
    ) -> None:
        """Forward request setup after clearing prior timing events."""
        self._reset_timing()
        self.observer.configure_request(
            valid_token_mask=valid_token_mask,
            last_user_span=last_user_span,
        )

    def observe_kv(
        self,
        *,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Time a legacy K/V observer hook."""
        self._time(
            key_states.device,
            lambda: self.observer.observe_kv(
                layer_idx=layer_idx,
                key_states=key_states,
                value_states=value_states,
            ),
        )

    def observe_qkv(
        self,
        *,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Time a post-RoPE Q/K/V observer hook."""
        self._time(
            key_states.device,
            lambda: self.observer.observe_qkv(
                layer_idx=layer_idx,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
            ),
        )

    def finalize(
        self, *, device: torch.device
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
        """Finalize wrapped results and attach a synchronized hook duration."""
        key_errors, value_errors, metadata = self.observer.finalize(device=device)
        self.last_timing_ms = self._elapsed_ms(device)
        self.timing_source = (
            "cuda_events" if device.type == "cuda" else "cpu_wall_clock"
        )
        return (
            key_errors,
            value_errors,
            {
                **metadata,
                "profile_observer_timing_ms": self.last_timing_ms,
                "profile_observer_timing_source": self.timing_source,
            },
        )

    def get_functional_result(self) -> Any | None:
        """Forward the optional v3 result."""
        return self.observer.get_functional_result()

    def _reset_timing(self) -> None:
        self.last_timing_ms = 0.0
        self._wall_seconds = 0.0
        self._events = []

    def _time(self, device: torch.device, operation: Callable[[], None]) -> None:
        if device.type == "cuda":
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record(torch.cuda.current_stream(device))
            operation()
            finished.record(torch.cuda.current_stream(device))
            self._events.append((started, finished))
            return
        started = time.perf_counter()
        operation()
        self._wall_seconds += time.perf_counter() - started

    def _elapsed_ms(self, device: torch.device) -> float:
        if device.type != "cuda":
            return self._wall_seconds * 1000.0
        torch.cuda.synchronize(device)
        return float(sum(start.elapsed_time(end) for start, end in self._events))


def _timed(
    device: torch.device, operation: Callable[[], Any]
) -> tuple[Any, float, dict[str, int]]:
    """Run one operation with synchronized CUDA time and four memory metrics."""
    if device.type == "cuda":
        _synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record(torch.cuda.current_stream(device))
        value = operation()
        finished.record(torch.cuda.current_stream(device))
        finished.synchronize()
        return (
            value,
            float(started.elapsed_time(finished)),
            {
                "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "peak_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "peak_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            },
        )
    started = time.perf_counter()
    value = operation()
    return value, (time.perf_counter() - started) * 1000.0, {
        "memory_allocated_bytes": 0,
        "peak_memory_allocated_bytes": 0,
        "memory_reserved_bytes": 0,
        "peak_memory_reserved_bytes": 0,
    }


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    """Capture allocator counters without resetting the current peak."""
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


def _safe_timed(
    device: torch.device, operation: Callable[[], Any]
) -> tuple[Any | None, float, dict[str, int], str | None]:
    """Run one profile stage while preserving later-stage measurements."""
    started = time.perf_counter()
    try:
        value, elapsed_ms, memory = _timed(device, operation)
        return value, elapsed_ms, memory, None
    except Exception as error:
        try:
            _synchronize(device)
        except Exception:
            pass
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        memory = _memory_snapshot(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return (
            None,
            elapsed_ms,
            memory,
            f"{type(error).__name__}: {error}",
        )


def _stage_status(error: str | None) -> str:
    """Normalize a captured stage exception for machine-readable rows."""
    if error is None:
        return "success"
    return "OOM" if "out of memory" in error.lower() else "error"


def _memory_fields(prefix: str, memory: dict[str, int]) -> dict[str, int]:
    """Flatten stage memory metrics while keeping the metric names explicit."""
    return {f"{prefix}_{name}": int(value) for name, value in memory.items()}


def _parse_ints(value: str) -> tuple[int, ...]:
    """Parse one non-empty comma-separated integer list."""
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("values must be positive comma-separated integers")
    return parsed


def _load_model(path: str, device: torch.device, dtype: torch.dtype) -> Any:
    """Load one frozen model for all benchmark cells."""
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return model.to(device).eval()


def _anchors(model: Any, count: int) -> tuple[int, ...]:
    """Select fixed late anchors without searching or tuning them."""
    layers = int(getattr(model.config, "num_hidden_layers", 0))
    if layers <= 0:
        raise ValueError("model config must expose num_hidden_layers")
    if count == 1:
        return (layers,)
    if count == 2:
        return (max(1, layers // 2), layers)
    raise ValueError("anchor counts currently support 1 or 2")


def _output_bytes(functional: Any) -> int:
    """Return the four externally emitted D-tensor bytes."""
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in functional.damage_by_precision.values()
    )


def _base_forward(model: Any, input_ids: torch.Tensor) -> Any:
    """Run the model backbone without either ScoutRank observer."""
    return model.model(
        input_ids=input_ids,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )


def _score_v2(adapter: Any, scorer: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """Run one v2 adapter forward and reduce its legacy token score."""
    forward = adapter.forward_once(input_ids, compute_nll=False)
    if forward.summary is None:
        raise RuntimeError("v2 profile forward produced no summary")
    summary = forward.summary
    return scorer.token_importance_from_summaries(
        token_ids=summary.token_ids,
        valid_token_mask=summary.valid_token_mask,
        self_information_nll=summary.self_information_nll,
        representation_drift=summary.representation_drift,
        local_novelty=summary.local_novelty,
        task_relevance=summary.task_relevance,
        k_errors=summary.k_errors,
        v_errors=summary.v_errors,
    )


def _score_v3(adapter: Any, input_ids: torch.Tensor) -> Any:
    """Run one v3 adapter forward and return its functional tensors."""
    forward = adapter.forward_once(input_ids, compute_nll=False)
    if forward.summary is None or forward.summary.functional_damage is None:
        raise RuntimeError("v3 profile forward produced no functional result")
    return forward.summary.functional_damage


def run_profile(
    *,
    model_path: str,
    device: str,
    dtype: str,
    token_counts: tuple[int, ...],
    probe_counts: tuple[int, ...],
    anchor_counts: tuple[int, ...],
    token_chunk_size: int,
    warmup: int,
) -> dict[str, Any]:
    """Run the requested real model/observer profiling matrix."""
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype_value = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    model = _load_model(model_path, resolved_device, dtype_value)
    rows: list[dict[str, Any]] = []
    try:
        vocab_size = int(model.config.vocab_size)
        for tokens in token_counts:
            ids = (torch.arange(tokens, device=resolved_device) % vocab_size).unsqueeze(
                0
            )
            for anchors_count in anchor_counts:
                anchors = _anchors(model, anchors_count)
                v2_adapter, v2_scorer, _ = _build_scout_runtime(
                    model,
                    mode="fast",
                    observer_backend="vectorized",
                    observer_token_chunk_size=token_chunk_size,
                    anchor_layers=anchors,
                    exit_layer=None,
                )
                v2_observer = _TimedObserver(v2_adapter.kv_observer)
                v2_adapter.kv_observer = v2_observer
                for probes in probe_counts:
                    v3_adapter, _v3_scorer, _ = _build_scout_runtime(
                        model,
                        mode="fast",
                        observer_backend="vectorized",
                        observer_token_chunk_size=token_chunk_size,
                        anchor_layers=anchors,
                        exit_layer=None,
                        scoring_version="v3-functional-disturbance",
                        v3_num_probes=probes,
                        v3_use_last_user_span=True,
                        v3_tail_probes=min(8, probes),
                    )
                    v3_observer = _TimedObserver(v3_adapter.kv_observer)
                    v3_adapter.kv_observer = v3_observer
                    for _ in range(warmup):
                        with torch.inference_mode():
                            _base_forward(model, ids)
                            v2_forward = v2_adapter.forward_once(ids, compute_nll=False)
                            v3_forward = v3_adapter.forward_once(ids, compute_nll=False)
                        del v2_forward, v3_forward
                    _synchronize(resolved_device)

                    _, base_ms, base_memory, base_error = _safe_timed(
                        resolved_device,
                        partial(_base_forward, model, ids),
                    )
                    v2_scores, v2_ms, v2_memory, v2_error = _safe_timed(
                        resolved_device,
                        partial(_score_v2, v2_adapter, v2_scorer, ids),
                    )
                    functional, v3_ms, v3_memory, v3_error = _safe_timed(
                        resolved_device,
                        partial(_score_v3, v3_adapter, ids),
                    )
                    rows.append(
                        {
                            "tokens": tokens,
                            "probes": probes,
                            "anchor_count": anchors_count,
                            "anchor_layers": list(anchors),
                            "base_forward_ms": base_ms,
                            "v2_total_score_ms": v2_ms,
                            "v3_total_score_ms": v3_ms,
                            "v2_observer_plus_score_ms_estimate": max(
                                0.0, v2_ms - base_ms
                            ),
                            "v2_observer_ms": v2_observer.last_timing_ms,
                            "v2_observer_timing_source": v2_observer.timing_source,
                            "v3_observer_plus_score_ms_estimate": max(
                                0.0, v3_ms - base_ms
                            ),
                            "v3_observer_ms": v3_observer.last_timing_ms,
                            "v3_observer_timing_source": v3_observer.timing_source,
                            "stage_status": {
                                "base": _stage_status(base_error),
                                "v2": _stage_status(v2_error),
                                "v3": _stage_status(v3_error),
                            },
                            "stage_errors": {
                                name: error
                                for name, error in (
                                    ("base", base_error),
                                    ("v2", v2_error),
                                    ("v3", v3_error),
                                )
                                if error is not None
                            },
                            **_memory_fields("base", base_memory),
                            **_memory_fields("v2", v2_memory),
                            **_memory_fields("v3", v3_memory),
                            # Compatibility aliases used by earlier profile
                            # artifacts.
                            "base_peak_memory_bytes": base_memory[
                                "peak_memory_allocated_bytes"
                            ],
                            "v2_peak_memory_bytes": v2_memory[
                                "peak_memory_allocated_bytes"
                            ],
                            "v3_peak_memory_bytes": v3_memory[
                                "peak_memory_allocated_bytes"
                            ],
                            "v3_stage_timing_ms": (
                                {}
                                if functional is None
                                else functional.metadata.get("stage_timing_ms", {})
                            ),
                            "v3_stage_timing_source": (
                                None
                                if functional is None
                                else functional.metadata.get("stage_timing_source")
                            ),
                            "v2_output_tensor_bytes": (
                                None
                                if v2_scores is None
                                else v2_scores.numel() * v2_scores.element_size()
                            ),
                            "v3_output_tensor_bytes": (
                                None
                                if functional is None
                                else _output_bytes(functional)
                            ),
                            "v3_probe_count": (
                                None
                                if functional is None
                                else functional.metadata["probe_count"]
                            ),
                        }
                    )
                    del functional, v2_scores, v3_adapter, v3_observer
                    gc.collect()
                    if resolved_device.type == "cuda":
                        torch.cuda.empty_cache()
                del v2_adapter, v2_scorer, v2_observer
    finally:
        del model
        gc.collect()
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "benchmark": "scoutrank_v3_functional_profile_v1",
        "scope": "frozen_qwen3_base_forward_and_single-forward_observers",
        "device": str(resolved_device),
        "dtype": dtype,
        "token_chunk_size": token_chunk_size,
        "warmup": warmup,
        "rows": rows,
    }


def main() -> None:
    """Run the command-line profiling entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--tokens", default="4096,16384,32768")
    parser.add_argument("--probes", default="8,16,32")
    parser.add_argument("--anchor-counts", default="1,2")
    parser.add_argument("--token-chunk-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_profile(
        model_path=args.model,
        device=args.device,
        dtype=args.dtype,
        token_counts=_parse_ints(args.tokens),
        probe_counts=_parse_ints(args.probes),
        anchor_counts=_parse_ints(args.anchor_counts),
        token_chunk_size=args.token_chunk_size,
        warmup=args.warmup,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
