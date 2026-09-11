#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage-by-stage ScoutRank-v3 long-context memory diagnosis.

The command deliberately keeps the input length fixed for every stage. Failed
stages are recorded instead of being treated as evidence that a later stage
also failed. Run it with ``CUDA_VISIBLE_DEVICES=3 --device cuda:0`` when the
physical GPU 3 is the selected device.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from makv_scoutrank import (  # noqa: E402
    FunctionalDisturbanceObserver,
    ScoutForwardAdapter,
    ScoutKVObserver,
    ScoutRankConfig,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_snapshot(device: torch.device) -> dict[str, int]:
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


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def _timed_stage(
    device: torch.device,
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run one stage and retain memory metrics even when it raises OOM."""
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.perf_counter()
    cuda_started: torch.cuda.Event | None = None
    cuda_finished: torch.cuda.Event | None = None
    if device.type == "cuda":
        cuda_started = torch.cuda.Event(enable_timing=True)
        cuda_finished = torch.cuda.Event(enable_timing=True)
        cuda_started.record(torch.cuda.current_stream(device))
    try:
        output = operation()
        status = "success"
        error_text = None
    except BaseException as error:  # Record OOM and continue with other stages.
        output = {}
        status = "OOM" if _is_oom(error) else "ERROR"
        error_text = f"{type(error).__name__}: {error}"
    cuda_time_ms: float | None = None
    if cuda_finished is not None and cuda_started is not None:
        try:
            cuda_finished.record(torch.cuda.current_stream(device))
            cuda_finished.synchronize()
            cuda_time_ms = float(cuda_started.elapsed_time(cuda_finished))
        except BaseException:
            cuda_time_ms = None
    wall_time_ms = (time.perf_counter() - wall_started) * 1000.0
    memory = _memory_snapshot(device)
    result = {
        "status": status,
        "cuda_time_ms": cuda_time_ms,
        "wall_time_ms": wall_time_ms,
        **memory,
        **output,
    }
    if error_text is not None:
        result["error"] = error_text
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


class _NoOpQKVObserver(ScoutKVObserver):
    """Anchor hook probe that retains no tensors and computes no damage."""

    def reset(self, valid_token_mask: torch.Tensor | None = None) -> None:
        del valid_token_mask
        self.observed_layers: list[int] = []

    def observe_qkv(
        self,
        *,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        del query_states, key_states, value_states
        self.observed_layers.append(layer_idx)


def _model_dimensions(model: Any) -> tuple[int, int, int]:
    config = model.config
    layers = int(config.num_hidden_layers)
    kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(
        getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    )
    return layers, kv_heads, head_dim


def _config(
    model: Any,
    *,
    anchors: tuple[int, ...],
    functional: bool,
    probes: int,
    token_chunk_size: int,
) -> ScoutRankConfig:
    layers, kv_heads, head_dim = _model_dimensions(model)
    return ScoutRankConfig(
        mode="fast",
        anchor_layers=anchors,
        exit_layer=None,
        block_size=32,
        include_token_scores=False,
        target_layers=layers,
        target_kv_heads=kv_heads,
        target_head_dim=head_dim,
        output_attentions=False,
        use_cache=False,
        functional_disturbance_enabled=functional,
        v3_num_probes=probes,
        v3_use_last_user_span=True,
        v3_tail_probes=min(8, probes),
        v3_token_chunk_size=token_chunk_size,
    )


def _base_forward(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    output = model.model(
        input_ids=input_ids,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = getattr(output, "last_hidden_state", None)
    return {
        "output_last_hidden_state_shape": (
            list(hidden.shape) if hidden is not None else None
        ),
        "full_logits_generated": False,
        "output_hidden_states_requested": False,
        "output_attentions_requested": False,
        "use_cache_requested": False,
    }


def _causal_lm_forward(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    output = model(
        input_ids=input_ids,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = getattr(output, "logits", None)
    shape = list(logits.shape) if logits is not None else None
    return {
        "logits_shape": shape,
        "full_logits_generated": shape
        == [1, input_ids.shape[1], model.config.vocab_size],
        "output_hidden_states_requested": False,
        "output_attentions_requested": False,
        "use_cache_requested": False,
    }


def _anchor_hook_only(
    model: Any,
    input_ids: torch.Tensor,
    *,
    anchors: tuple[int, ...],
    token_chunk_size: int,
) -> dict[str, Any]:
    cfg = _config(
        model,
        anchors=anchors,
        functional=False,
        probes=1,
        token_chunk_size=token_chunk_size,
    )
    observer = _NoOpQKVObserver()
    adapter = ScoutForwardAdapter(model, cfg, observer)
    forward = adapter.forward_once(input_ids, compute_nll=False)
    return {
        "output_last_hidden_state_shape": list(forward.final_hidden.shape),
        "hook_observed_layers": list(observer.observed_layers),
        "full_logits_generated": False,
        "output_hidden_states_requested": False,
        "output_attentions_requested": False,
        "use_cache_requested": False,
    }


def _v3_forward(
    model: Any,
    input_ids: torch.Tensor,
    *,
    anchors: tuple[int, ...],
    probes: int,
    token_chunk_size: int,
    precision_sequence: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    cfg = _config(
        model,
        anchors=anchors,
        functional=True,
        probes=probes,
        token_chunk_size=token_chunk_size,
    )
    observer = FunctionalDisturbanceObserver(
        cfg,
        diagnostic_precisions=precision_sequence,  # type: ignore[arg-type]
    )
    adapter = ScoutForwardAdapter(model, cfg, observer)
    forward = adapter.forward_once(input_ids, compute_nll=False)
    if forward.summary is None or forward.summary.functional_damage is None:
        raise RuntimeError("v3 stage produced no functional result")
    result = forward.summary.functional_damage
    computed_precisions = result.metadata.get("computed_precisions", [])
    return {
        "output_last_hidden_state_shape": list(forward.final_hidden.shape),
        "hook_observed_layers": list(result.metadata.get("observed_kv_layers", [])),
        "v3_probe_count": result.metadata.get("probe_count"),
        "v3_anchor_count": len(anchors),
        "v3_precision_count": len(computed_precisions),
        "v3_precision_sequence": computed_precisions,
        "v3_complete_precision_set": result.metadata.get("complete_precision_set"),
        "v3_stage_timing_ms": result.metadata.get("stage_timing_ms", {}),
        "full_logits_generated": False,
        "output_hidden_states_requested": False,
        "output_attentions_requested": False,
        "use_cache_requested": False,
        "prompt_sized_attention_matrix_constructed": False,
        "full_gqa_kv_repeat_constructed": False,
    }


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("expected positive comma-separated integers")
    return values


def run_diagnosis(
    *,
    model_path: str,
    device: str,
    dtype: str,
    token_counts: tuple[int, ...],
    probe_counts: tuple[int, ...],
    anchor_counts: tuple[int, ...],
    token_chunk_size: int,
    stages: tuple[str, ...],
) -> dict[str, Any]:
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot use it")
    dtype_value = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype_value,
        low_cpu_mem_usage=True,
    ).to(resolved_device).eval()
    layers = int(model.config.num_hidden_layers)
    vocab_size = int(model.config.vocab_size)
    rows: list[dict[str, Any]] = []
    try:
        for tokens in token_counts:
            input_ids = (
                torch.arange(tokens, device=resolved_device, dtype=torch.long)
                % vocab_size
            ).unsqueeze(0)
            for stage in stages:
                stage_specs: list[
                    tuple[str, tuple[int, ...], int, Callable[[], dict[str, Any]]]
                ] = []
                if stage == "backbone":
                    stage_specs.append(
                        (stage, (), 0, partial(_base_forward, model, input_ids))
                    )
                elif stage == "causal_lm":
                    stage_specs.append(
                        (stage, (), 0, partial(_causal_lm_forward, model, input_ids))
                    )
                elif stage == "anchor_hook":
                    for count in anchor_counts:
                        if count not in (1, 2):
                            raise ValueError("anchor counts currently support 1 or 2")
                        anchors = (
                            (layers,)
                            if count == 1
                            else (max(1, layers // 2), layers)
                        )
                        stage_specs.append(
                            (
                                stage,
                                anchors,
                                0,
                                partial(
                                    _anchor_hook_only,
                                    model,
                                    input_ids,
                                    anchors=anchors,
                                    token_chunk_size=token_chunk_size,
                                ),
                            )
                        )
                elif stage == "v3":
                    for count in anchor_counts:
                        if count not in (1, 2):
                            raise ValueError("anchor counts currently support 1 or 2")
                        anchors = (
                            (layers,)
                            if count == 1
                            else (max(1, layers // 2), layers)
                        )
                        for probes in probe_counts:
                            stage_specs.append(
                                (
                                    stage,
                                    anchors,
                                    probes,
                                    partial(
                                        _v3_forward,
                                        model,
                                        input_ids,
                                        anchors=anchors,
                                        probes=probes,
                                        token_chunk_size=token_chunk_size,
                                    ),
                                )
                            )
                elif stage == "v3_single_precision":
                    for count in anchor_counts:
                        if count not in (1, 2):
                            raise ValueError("anchor counts currently support 1 or 2")
                        anchors = (
                            (layers,)
                            if count == 1
                            else (max(1, layers // 2), layers)
                        )
                        for probes in probe_counts:
                            for precision in ("K2V2", "K4V2", "K8V4"):
                                stage_specs.append(
                                    (
                                        stage,
                                        anchors,
                                        probes,
                                        partial(
                                            _v3_forward,
                                            model,
                                            input_ids,
                                            anchors=anchors,
                                            probes=probes,
                                            token_chunk_size=token_chunk_size,
                                            precision_sequence=(precision,),
                                        ),
                                    )
                                )
                else:
                    raise ValueError(f"unsupported stage {stage!r}")
                for stage_name, anchors, probes, operation in stage_specs:
                    row = _timed_stage(resolved_device, operation)
                    rows.append(
                        {
                            "tokens": tokens,
                            "stage": stage_name,
                            "anchors": list(anchors),
                            "anchor_count": len(anchors),
                            "probes": probes or None,
                            "token_chunk_size": token_chunk_size,
                            **row,
                        }
                    )
                    gc.collect()
                    if resolved_device.type == "cuda":
                        torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "benchmark": "scoutrank_v3_oom_diagnosis_v1",
        "device": str(resolved_device),
        "dtype": dtype,
        "token_counts": list(token_counts),
        "probe_counts": list(probe_counts),
        "anchor_counts": list(anchor_counts),
        "stages": list(stages),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--tokens", default="8192,12288,16384")
    parser.add_argument("--probes", default="1,8,16,32")
    parser.add_argument("--anchor-counts", default="1,2")
    parser.add_argument("--token-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--stages",
        default="backbone,causal_lm,anchor_hook,v3",
        help=(
            "Comma-separated stages: backbone,causal_lm,anchor_hook,v3,"
            "v3_single_precision"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_diagnosis(
        model_path=args.model,
        device=args.device,
        dtype=args.dtype,
        token_counts=_parse_ints(args.tokens),
        probe_counts=_parse_ints(args.probes),
        anchor_counts=_parse_ints(args.anchor_counts),
        token_chunk_size=args.token_chunk_size,
        stages=tuple(item.strip() for item in args.stages.split(",") if item.strip()),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
