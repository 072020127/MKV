#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare ScoutRank token importance with attention-based token scores.

Both methods run on the same tokenized prompt and the same frozen scout model.
ScoutRank uses the existing production importance helper.  The attention
baseline is an offline-only collector: it computes attention scores in bounded
query chunks and releases each chunk before the next one.  It is not imported
by the MaKV production path.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from longbench_makv_cachegen import load_examples, prompt_ids, prompt_token_hash
    from scoutrank_longbench_importance import (
        _build_scout_runtime,
        _parse_anchor_layers,
        _score_prompt,
    )
except ImportError:
    from benchmarks.longbench_makv_cachegen import (
        load_examples,
        prompt_ids,
        prompt_token_hash,
    )
    from benchmarks.scoutrank_longbench_importance import (
        _build_scout_runtime,
        _parse_anchor_layers,
        _score_prompt,
    )

from experiments.scoutrank_transfer.metrics import kendall_tau_b, spearman


DEFAULT_TOPK_RATIOS = (0.01, 0.05, 0.10, 0.20, 0.50)
# Match the current MaKV four-tier production policy: BF16/K8V4/K4V2/K2V2.
# Historical reports retain their original ratios in the artifact itself.
DEFAULT_BUCKET_RATIOS = (0.10, 0.20, 0.50, 0.20)
BUCKET_NAMES = ("BF16", "K8V4", "K4V2", "K2V2")


@dataclass(frozen=True)
class PromptInput:
    """One exact token sequence used by both scoring methods."""

    prompt_id: str
    token_ids: list[int]
    source: str
    metadata: dict[str, Any]


class AttentionMatrixExporter:
    """Stream one complete attention matrix to a raw row-major tensor file."""

    _DTYPES = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    _DTYPE_NAMES = {value: key for key, value in _DTYPES.items()}

    def __init__(
        self,
        output: Path,
        token_count: int,
        requested_dtype: str = "model",
    ) -> None:
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        if requested_dtype not in ("model", *self._DTYPES):
            raise ValueError(f"unsupported attention matrix dtype: {requested_dtype}")
        self.output = output
        self.token_count = token_count
        self.requested_dtype = requested_dtype
        self._part_path = Path(f"{output}.part")
        self._handle: Any | None = None
        self._head_count: int | None = None
        self._storage_dtype: torch.dtype | None = None
        self._chunk_count = 0
        self._bytes_written = 0

    def write(self, attention_weights: torch.Tensor, query_start: int) -> None:
        """Write one [1, heads, query_chunk, key] block into its final offsets."""
        if attention_weights.ndim != 4 or attention_weights.shape[0] != 1:
            raise ValueError(
                "attention matrix blocks must have shape [1, heads, query, key]"
            )
        _, head_count, query_count, key_count = attention_weights.shape
        query_end = query_start + query_count
        if (
            query_start < 0
            or query_end > self.token_count
            or key_count != self.token_count
        ):
            raise ValueError("attention matrix block is outside the prompt bounds")
        storage_dtype = (
            attention_weights.dtype
            if self.requested_dtype == "model"
            else self._DTYPES[self.requested_dtype]
        )
        if storage_dtype not in self._DTYPE_NAMES:
            raise ValueError(f"unsupported attention matrix dtype: {storage_dtype}")
        if self._handle is None:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self._part_path.unlink(missing_ok=True)
            self._head_count = int(head_count)
            self._storage_dtype = storage_dtype
            element_size = torch.empty((), dtype=storage_dtype).element_size()
            self._handle = self._part_path.open("w+b")
            self._handle.truncate(
                self._head_count * self.token_count * self.token_count * element_size
            )
        elif (
            self._head_count != int(head_count) or self._storage_dtype != storage_dtype
        ):
            raise ValueError("attention matrix block shape or dtype changed mid-export")

        values = attention_weights.detach()
        if values.dtype != storage_dtype:
            values = values.to(storage_dtype)
        cpu_values = values[0].contiguous().cpu()
        element_size = cpu_values.element_size()
        for head in range(int(head_count)):
            row_bytes = (
                cpu_values[head].contiguous().view(torch.uint8).numpy().tobytes()
            )
            offset = (
                head * self.token_count * self.token_count
                + query_start * self.token_count
            ) * element_size
            self._handle.seek(offset)
            self._handle.write(row_bytes)
            self._bytes_written += len(row_bytes)
        self._chunk_count += 1

    def finalize(self) -> dict[str, Any]:
        """Atomically publish the completed raw tensor and return its metadata."""
        if (
            self._handle is None
            or self._head_count is None
            or self._storage_dtype is None
        ):
            raise RuntimeError("no attention matrix blocks were exported")
        self._handle.flush()
        self._handle.close()
        self._handle = None
        os.replace(self._part_path, self.output)
        element_size = torch.empty((), dtype=self._storage_dtype).element_size()
        total_bytes = (
            self._head_count * self.token_count * self.token_count * element_size
        )
        return {
            "path": str(self.output),
            "format": "raw little-endian tensor bytes",
            "layout": "[batch, head, query, key] row-major",
            "shape": [1, self._head_count, self.token_count, self.token_count],
            "dtype": self._DTYPE_NAMES[self._storage_dtype],
            "bytes": total_bytes,
            "blocks_written": self._chunk_count,
            "bytes_written": self._bytes_written,
            "complete": self._bytes_written == total_bytes,
        }

    def abort(self) -> None:
        """Close and remove only an incomplete export."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._part_path.unlink(missing_ok=True)


class AttentionScoreCollector:
    """Accumulate a token score from one layer at a time.

    ``incoming_mean`` averages the attention received by a token over the
    causal queries that can see it, then averages over heads and layers.  The
    denominator is explicit so early tokens are not favored merely because
    they are visible to more queries.  ``incoming_sum`` retains that position
    effect and ``last_query`` is the final query's head-average attention.
    """

    def __init__(
        self,
        token_count: int,
        aggregation: str,
        matrix_exporter: AttentionMatrixExporter | None = None,
        valid_token_mask: torch.Tensor | None = None,
    ) -> None:
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        if aggregation not in (
            "incoming_mean",
            "incoming_sum",
            "incoming_both",
            "future_both",
            "last_query",
        ):
            raise ValueError(f"unsupported attention aggregation: {aggregation}")
        self.token_count = token_count
        self.aggregation = aggregation
        self._score: torch.Tensor | None = None
        self._raw_score: torch.Tensor | None = None
        self._layer_score: torch.Tensor | None = None
        self._layer_visible_count: torch.Tensor | None = None
        self._visible_count: torch.Tensor | None = None
        self._last_query_seen = False
        self.matrix_exporter = matrix_exporter
        self.layer_count = 0
        if valid_token_mask is not None:
            if valid_token_mask.ndim != 1 or valid_token_mask.numel() != token_count:
                raise ValueError("valid_token_mask must have shape [token_count]")
            self.valid_token_mask = valid_token_mask.detach().to(dtype=torch.bool)
        else:
            self.valid_token_mask = None
        self.sliding_window: int | None = None

    def begin_layer(self) -> None:
        """Start accumulating one layer, possibly from query-sized chunks."""
        if self._layer_score is not None:
            raise RuntimeError("previous attention layer was not finalized")
        self._last_query_seen = False
        self._layer_visible_count = None

    def set_sliding_window(self, sliding_window: int | None) -> None:
        """Record the model's local-attention window for future visibility."""
        self.sliding_window = sliding_window

    def add_query_chunk(
        self, attention_weights: torch.Tensor, query_start: int
    ) -> None:
        """Consume ``[1, heads, query_chunk, key]`` without retaining the chunk."""
        if attention_weights.ndim != 4 or attention_weights.shape[0] != 1:
            raise ValueError("attention weights must have shape [1, heads, query, key]")
        _, _, query_count, key_count = attention_weights.shape
        query_end = query_start + query_count
        if (
            query_start < 0
            or query_end > self.token_count
            or key_count != self.token_count
        ):
            raise ValueError(
                "attention chunk must cover valid prompt query/key positions"
            )
        weights = torch.nan_to_num(
            attention_weights.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        if self.matrix_exporter is not None:
            self.matrix_exporter.write(attention_weights, query_start)
        if self.aggregation == "last_query":
            if query_end == self.token_count:
                self._layer_score = weights[:, :, -1, :].mean(dim=(0, 1))
                self._last_query_seen = True
            return

        if self.aggregation == "future_both":
            query_positions = torch.arange(
                query_start, query_end, device=weights.device
            )
            key_positions = torch.arange(self.token_count, device=weights.device)
            allowed = key_positions.unsqueeze(0) < query_positions.unsqueeze(1)
            if self.sliding_window is not None:
                allowed &= key_positions.unsqueeze(0) > (
                    query_positions.unsqueeze(1) - self.sliding_window
                )
            if self.valid_token_mask is not None:
                valid = self.valid_token_mask.to(device=weights.device)
                allowed &= valid[query_positions].unsqueeze(1)
                allowed &= valid.unsqueeze(0)
            mask = allowed.view(1, 1, query_end - query_start, self.token_count)
            chunk_score = (weights * mask).sum(dim=(0, 1, 2))
            chunk_count = allowed.sum(dim=0, dtype=torch.int64) * weights.shape[1]
            if self._layer_visible_count is None:
                self._layer_visible_count = chunk_count
            else:
                self._layer_visible_count.add_(chunk_count)
        else:
            incoming = weights.sum(dim=-2)
            if self.aggregation == "incoming_both":
                # Preserve the historical incoming mass semantics. The new
                # future_both branch is deliberately separate because it
                # excludes self-query observations and counts (l,h,j) exactly.
                chunk_score = incoming.sum(dim=(0, 1))
            else:
                chunk_score = incoming.mean(dim=(0, 1))
        if self._layer_score is None:
            self._layer_score = chunk_score
        else:
            self._layer_score.add_(chunk_score)

    def finish_layer(self) -> None:
        """Finalize one layer and apply the selected position normalization."""
        if self._layer_score is None:
            raise RuntimeError("no attention query chunks were collected")
        if self.aggregation == "last_query" and not self._last_query_seen:
            raise RuntimeError("attention chunks did not include the final query")
        layer_score = self._layer_score
        raw_layer_score = layer_score
        self._raw_score = (
            raw_layer_score
            if self._raw_score is None
            else self._raw_score + raw_layer_score
        )
        if self.aggregation == "future_both":
            if self._layer_visible_count is None:
                raise RuntimeError("future attention visibility was not collected")
            self._visible_count = (
                self._layer_visible_count.clone()
                if self._visible_count is None
                else self._visible_count + self._layer_visible_count
            )
        if self.aggregation == "incoming_mean":
            visible_queries = torch.arange(
                self.token_count,
                0,
                -1,
                device=layer_score.device,
                dtype=layer_score.dtype,
            )
            layer_score = layer_score / visible_queries
        self._score = layer_score if self._score is None else self._score + layer_score
        self.layer_count += 1
        self._layer_score = None
        self._layer_visible_count = None
        self._last_query_seen = False

    def result_variants(self) -> dict[str, torch.Tensor]:
        """Return raw incoming mass and its visible-query normalization."""
        if self._raw_score is None or self.layer_count == 0:
            raise RuntimeError("no incoming attention layers were collected")
        raw = self._raw_score / float(self.layer_count)
        visible_queries = torch.arange(
            self.token_count,
            0,
            -1,
            device=raw.device,
            dtype=raw.dtype,
        )
        normalized = raw / visible_queries
        if not torch.isfinite(raw).all() or not torch.isfinite(normalized).all():
            raise FloatingPointError("attention aggregation produced non-finite scores")
        return {"incoming_sum": raw, "incoming_mean": normalized}

    def result_future_variants(self) -> dict[str, torch.Tensor]:
        """Return strict-future mass and exact ``(l,h,j)`` normalization."""
        if (
            self._raw_score is None
            or self._visible_count is None
            or self.layer_count == 0
        ):
            raise RuntimeError("no future attention observations were collected")
        raw = self._raw_score
        visible_count = self._visible_count
        normalized = raw / visible_count.to(dtype=raw.dtype).clamp_min(1)
        if not torch.isfinite(raw).all() or not torch.isfinite(normalized).all():
            raise FloatingPointError(
                "future attention aggregation produced non-finite scores"
            )
        return {
            "A_raw": raw,
            "A_norm": normalized,
            "visible_count": visible_count,
        }

    def add(self, attention_weights: torch.Tensor) -> None:
        """Consume one layer's ``[1, heads, query, key]`` attention tensor."""
        if attention_weights.shape[-2:] != (self.token_count, self.token_count):
            raise ValueError(
                "attention weights must cover the complete prompt without a cache"
            )
        self.begin_layer()
        self.add_query_chunk(attention_weights, 0)
        self.finish_layer()

    def result(self) -> torch.Tensor:
        """Return the layer-averaged score vector on the collector device."""
        if self._score is None or self.layer_count == 0:
            raise RuntimeError("no attention layers were collected")
        result = self._score / float(self.layer_count)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("attention aggregation produced non-finite scores")
        return result


def _install_attention_collector(
    attention: Any,
    layer_index: int,
    collector: AttentionScoreCollector,
    query_chunk_size: int,
) -> Any:
    """Patch one Qwen3 layer with a memory-bounded offline score collector."""
    if (
        attention.__class__.__name__ != "Qwen3Attention"
        or "transformers.models.qwen3" not in attention.__class__.__module__
    ):
        raise TypeError("attention collector only supports Transformers Qwen3Attention")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    del layer_index
    original_forward = attention.forward

    def apply_score_mask(
        logits: torch.Tensor,
        attention_mask: torch.Tensor | None,
        query_start: int,
        query_end: int,
        key_count: int,
        sliding_window: int | None,
    ) -> torch.Tensor:
        """Apply the Qwen causal/padding mask to one query chunk."""
        query_positions = torch.arange(query_start, query_end, device=logits.device)
        key_positions = torch.arange(key_count, device=logits.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        if sliding_window is not None:
            allowed &= key_positions.unsqueeze(0) > (
                query_positions.unsqueeze(1) - sliding_window
            )
        if attention_mask is None:
            return logits.masked_fill(
                ~allowed.view(1, 1, query_end - query_start, key_count),
                torch.finfo(logits.dtype).min,
            )
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError(
                "long attention comparison requires a tensor attention mask; "
                f"got {type(attention_mask).__name__}"
            )
        if attention_mask.ndim == 4:
            mask_chunk = attention_mask[..., query_start:query_end, :key_count]
        elif attention_mask.ndim == 3:
            mask_chunk = attention_mask[:, query_start:query_end, :key_count]
        elif attention_mask.ndim == 2:
            padding = attention_mask[:, :key_count].to(torch.bool)
            mask_chunk = allowed.unsqueeze(0) & padding[:, None, :]
        else:
            raise ValueError(
                "unsupported attention mask rank for chunked attention: "
                f"{attention_mask.ndim}"
            )
        if mask_chunk.dtype == torch.bool:
            return logits.masked_fill(~mask_chunk, torch.finfo(logits.dtype).min)
        return logits + mask_chunk.to(dtype=logits.dtype)

    def collected_forward(
        self: Any,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Run chunked score attention and a memory-efficient model attention."""
        from transformers.models.qwen3 import modeling_qwen3 as qwen3_mod

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = qwen3_mod.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_values is not None:
            raise ValueError(
                "attention overlap requires a prompt without past KV cache"
            )

        token_count = query_states.shape[2]
        if key_states.shape[2] != token_count:
            raise ValueError("attention overlap requires equal query and key lengths")
        collector.begin_layer()
        collector.set_sliding_window(self.sliding_window)
        num_heads = int(
            getattr(self, "num_heads", self.config.num_attention_heads)
        )
        num_key_value_heads = int(
            getattr(self, "num_key_value_heads", self.config.num_key_value_heads)
        )
        try:
            for query_start in range(0, token_count, query_chunk_size):
                query_end = min(query_start + query_chunk_size, token_count)
                query_chunk = query_states[:, :, query_start:query_end, :]
                query_grouped = query_chunk.reshape(
                    1,
                    num_key_value_heads,
                    num_heads // num_key_value_heads,
                    query_end - query_start,
                    self.head_dim,
                )
                logits = torch.einsum(
                    "bngqd,bnkd->bngqk", query_grouped, key_states
                ).reshape(
                    1,
                    num_heads,
                    query_end - query_start,
                    token_count,
                ) * self.scaling
                logits = apply_score_mask(
                    logits,
                    attention_mask,
                    query_start,
                    query_end,
                    token_count,
                    self.sliding_window,
                )
                attention_weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(
                    query_states.dtype
                )
                collector.add_query_chunk(attention_weights, query_start)
                del attention_weights, logits, query_chunk
        finally:
            # Preserve the original exception if projection/setup fails before
            # the first chunk is consumed.
            if collector._layer_score is not None:
                collector.finish_layer()

        attention_interface = qwen3_mod.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, qwen3_mod.eager_attention_forward
        )
        interface_kwargs = dict(kwargs)
        interface_kwargs["output_attentions"] = False
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **interface_kwargs,
        )
        if attn_weights is not None:
            del attn_weights
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    attention.forward = MethodType(collected_forward, attention)

    def restore() -> None:
        """Restore the original layer forward."""
        attention.forward = original_forward

    return restore


def collect_attention_scores(
    model: Any,
    input_ids: torch.Tensor,
    aggregation: str,
    query_chunk_size: int = 256,
    layer_mode: str = "last",
    matrix_output: Path | None = None,
    matrix_dtype: str = "model",
    valid_token_mask: torch.Tensor | None = None,
) -> tuple[list[float] | dict[str, list[float]], dict[str, Any]]:
    """Run one prompt and return attention-based token scores.

    Args:
        model: A Qwen3 causal language model on the target device.
        input_ids: A ``[1, token_count]`` prompt tensor.
        aggregation: ``incoming_mean``, ``incoming_sum``, ``incoming_both``,
            ``future_both``, or ``last_query``. ``incoming_both`` returns its
            historical mass variants; ``future_both`` returns strict-future
            ``A_raw``, ``A_norm`` and integer ``visible_count`` variants.
        query_chunk_size: Number of queries processed by one score block.
        layer_mode: Collect only the final layer (``last``) or every layer
            (``all``).
        matrix_output: Optional raw output path for one complete layer matrix.
        matrix_dtype: Storage dtype for the matrix output or ``model``.
        valid_token_mask: Optional boolean mask for valid query/key tokens.

    Returns:
        A CPU score list and collector metadata.

    Raises:
        ValueError: If the input shape is unsupported.
        RuntimeError: If a layer cannot expose the requested attention path.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, token_count]")
    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise ValueError("Qwen3 base model does not expose model.layers")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if valid_token_mask is not None:
        if valid_token_mask.ndim != 1 or valid_token_mask.numel() != input_ids.shape[1]:
            raise ValueError("valid_token_mask must have shape [token_count]")
        valid_token_mask = valid_token_mask.to(
            device=input_ids.device, dtype=torch.bool
        )
    layer_indices = _attention_layer_indices(len(layers), layer_mode)
    if matrix_output is not None and layer_mode != "first":
        raise ValueError("full attention matrix export requires layer_mode=first")
    matrix_exporter = (
        AttentionMatrixExporter(matrix_output, int(input_ids.shape[1]), matrix_dtype)
        if matrix_output is not None
        else None
    )
    implementation = getattr(base_model.config, "_attn_implementation", None)
    if implementation == "eager" and input_ids.shape[1] > 4096:
        raise ValueError(
            "long attention comparison cannot use eager model attention above 4096 "
            "tokens; load with --model-attention-implementation sdpa"
        )
    collector = AttentionScoreCollector(
        int(input_ids.shape[1]),
        aggregation,
        matrix_exporter=matrix_exporter,
        valid_token_mask=valid_token_mask,
    )
    restores = []
    for index in layer_indices:
        layer = layers[index]
        restores.append(
            _install_attention_collector(
                layer.self_attn, index, collector, query_chunk_size
            )
        )
    device = input_ids.device
    try:
        with torch.inference_mode():
            base_model(
                input_ids=input_ids,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if aggregation == "incoming_both":
            scores = {
                name: value.detach().cpu().tolist()
                for name, value in collector.result_variants().items()
            }
        elif aggregation == "future_both":
            scores = {
                name: value.detach().cpu().tolist()
                for name, value in collector.result_future_variants().items()
            }
        else:
            scores = collector.result().detach().cpu().tolist()
        matrix_metadata = (
            matrix_exporter.finalize() if matrix_exporter is not None else None
        )
    except BaseException:
        if matrix_exporter is not None:
            matrix_exporter.abort()
        raise
    finally:
        for restore in reversed(restores):
            restore()
    metadata = {
        "aggregation": aggregation,
        "score_variants": (
            ["incoming_sum", "incoming_mean"]
            if aggregation == "incoming_both"
            else ["A_raw", "A_norm", "visible_count"]
            if aggregation == "future_both"
            else [aggregation]
        ),
        "raw_mass_formula": (
            "sum_{l,h,j:t<j} a[l,h,j,t]"
            if aggregation == "future_both"
            else "historical incoming causal mass"
        ),
        "normalized_mass_formula": (
            "A_raw / max(visible_count, 1), visible_count=#valid(l,h,j)"
            if aggregation == "future_both"
            else "historical incoming mass / visible_query_count"
        ),
        "valid_mask": (
            [count > 0 for count in scores["visible_count"]]
            if aggregation == "future_both"
            else None
        ),
        "forced_precision_by_token": (
            ["BF16" if count == 0 else None for count in scores["visible_count"]]
            if aggregation == "future_both"
            else None
        ),
        "reason_by_token": (
            [
                "NO_FUTURE_PROBE" if count == 0 else None
                for count in scores["visible_count"]
            ]
            if aggregation == "future_both"
            else None
        ),
        "token_status": (
            [
                {
                    "valid_mask": count > 0,
                    "forced_precision": "BF16" if count == 0 else None,
                    "reason": "NO_FUTURE_PROBE" if count == 0 else None,
                }
                for count in scores["visible_count"]
            ]
            if aggregation == "future_both"
            else None
        ),
        "layer_count": collector.layer_count,
        "attention_weights_retained": False,
        "attention_implementation": "transformers_qwen3_chunked_qk_offline_collector",
        "model_attention_implementation": implementation,
        "query_chunk_size": query_chunk_size,
        "layer_mode": layer_mode,
        "layer_indices_zero_based": list(layer_indices),
        "layer_numbers_one_based": [index + 1 for index in layer_indices],
        "score_tensor_shape": [
            1,
            "num_attention_heads",
            query_chunk_size,
            int(input_ids.shape[1]),
        ],
    }
    if matrix_metadata is not None:
        metadata["matrix_export"] = matrix_metadata
    return scores, metadata


def _attention_layer_indices(layer_count: int, layer_mode: str) -> tuple[int, ...]:
    """Return the layers whose attention scores should be collected."""
    if layer_count <= 0:
        raise ValueError("the model must expose at least one attention layer")
    if layer_mode == "first":
        return (0,)
    if layer_mode == "last":
        return (layer_count - 1,)
    if layer_mode == "all":
        return tuple(range(layer_count))
    raise ValueError(f"unsupported attention layer mode: {layer_mode}")


def _parse_float_list(value: str, name: str) -> tuple[float, ...]:
    """Parse a comma-separated list of finite non-negative floats."""
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not math.isfinite(item) or item < 0.0 for item in values):
        raise ValueError(f"{name} must contain finite non-negative numbers")
    return values


def _validate_ratios(ratios: Iterable[float], name: str) -> tuple[float, ...]:
    """Validate ratios that must sum to one."""
    result = tuple(ratios)
    if not result or not math.isclose(sum(result), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} must sum to 1 within 1e-6")
    return result


def deterministic_order(scores: list[float]) -> list[int]:
    """Return score-descending positions with token index as tie-break."""
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def _finite_analysis_scores(scores: list[float]) -> tuple[list[float], int]:
    """Make a finite copy for auxiliary rank metrics.

    Fast D22 uses non-finite values as a fail-closed safety signal. Those
    values are handled as BF16 by bucket metrics, but legacy rank/correlation
    helpers require finite inputs. Map every non-finite value to one shared
    score above the finite maximum; stable position order remains the tie-break
    and the raw vector is still retained in the report.
    """
    finite = [float(value) for value in scores if math.isfinite(float(value))]
    maximum = max(finite, default=0.0)
    sentinel = maximum + max(1.0, abs(maximum)) * 1e-6
    result = [
        float(value) if math.isfinite(float(value)) else sentinel
        for value in scores
    ]
    return result, len(scores) - len(finite)


def topk_overlap(
    scout_scores: list[float], attention_scores: list[float], ratios: Iterable[float]
) -> dict[str, dict[str, float | int]]:
    """Compute deterministic Top-k intersection, recall, and Jaccard metrics."""
    if len(scout_scores) != len(attention_scores) or not scout_scores:
        raise ValueError("score vectors must have equal non-zero length")
    scout_order = deterministic_order(scout_scores)
    attention_order = deterministic_order(attention_scores)
    n = len(scout_scores)
    result: dict[str, dict[str, float | int]] = {}
    for ratio in ratios:
        if not 0.0 < ratio <= 1.0:
            raise ValueError("Top-k ratios must be in (0, 1]")
        k = min(n, max(1, math.ceil(ratio * n)))
        scout_top = set(scout_order[:k])
        attention_top = set(attention_order[:k])
        overlap = len(scout_top & attention_top)
        union = len(scout_top | attention_top)
        result[f"{ratio:g}"] = {
            "k": k,
            "overlap": overlap,
            "recall": overlap / k,
            "precision": overlap / k,
            "jaccard": overlap / union if union else 1.0,
        }
    return result


def ranking_metrics(
    scout_scores: list[float], attention_scores: list[float]
) -> dict[str, float | None]:
    """Compute rank and absolute-rank-distance metrics for one prompt."""
    if len(scout_scores) != len(attention_scores) or not scout_scores:
        raise ValueError("score vectors must have equal non-zero length")
    scout_order = deterministic_order(scout_scores)
    attention_order = deterministic_order(attention_scores)
    scout_rank = {position: rank for rank, position in enumerate(scout_order, start=1)}
    attention_rank = {
        position: rank for rank, position in enumerate(attention_order, start=1)
    }
    spearman_value = spearman(scout_scores, attention_scores)
    kendall_value = kendall_tau_b(scout_scores, attention_scores)
    values: dict[str, float | None] = {
        "spearman": spearman_value,
        "kendall_tau_b": kendall_value,
        "mean_absolute_rank_difference": statistics.fmean(
            abs(scout_rank[position] - attention_rank[position])
            for position in range(len(scout_scores))
        ),
    }
    return {
        key: value if value is None or math.isfinite(value) else None
        for key, value in values.items()
    }


def _score_distribution(scores: list[float]) -> list[float]:
    """Normalize a finite score vector into a non-negative distribution."""
    if not scores:
        raise ValueError("score vectors must be non-empty")
    deterministic_order(scores)
    minimum = min(scores)
    shifted = [score - minimum if minimum < 0.0 else score for score in scores]
    total = math.fsum(shifted)
    if not math.isfinite(total) or total <= 0.0:
        return [1.0 / len(scores)] * len(scores)
    return [score / total for score in shifted]


def score_similarity_metrics(
    scout_scores: list[float], attention_scores: list[float]
) -> dict[str, float | None]:
    """Compare score values after accounting for their different scales."""
    if len(scout_scores) != len(attention_scores) or not scout_scores:
        raise ValueError("score vectors must have equal non-zero length")
    deterministic_order(scout_scores)
    deterministic_order(attention_scores)
    scout_mean = statistics.fmean(scout_scores)
    attention_mean = statistics.fmean(attention_scores)
    centered_scout = [score - scout_mean for score in scout_scores]
    centered_attention = [score - attention_mean for score in attention_scores]
    covariance = math.fsum(
        left * right
        for left, right in zip(centered_scout, centered_attention, strict=True)
    )
    scout_norm = math.sqrt(math.fsum(value * value for value in centered_scout))
    attention_norm = math.sqrt(math.fsum(value * value for value in centered_attention))
    if scout_norm == 0.0 or attention_norm == 0.0:
        pearson_value = 1.0 if scout_scores == attention_scores else 0.0
    else:
        pearson_value = covariance / (scout_norm * attention_norm)
    raw_scout_norm = math.sqrt(math.fsum(score * score for score in scout_scores))
    raw_attention_norm = math.sqrt(
        math.fsum(score * score for score in attention_scores)
    )
    cosine_value = (
        math.fsum(
            left * right
            for left, right in zip(scout_scores, attention_scores, strict=True)
        )
        / (raw_scout_norm * raw_attention_norm)
        if raw_scout_norm > 0.0 and raw_attention_norm > 0.0
        else (1.0 if scout_scores == attention_scores else 0.0)
    )
    scout_distribution = _score_distribution(scout_scores)
    attention_distribution = _score_distribution(attention_scores)
    midpoint = [
        (left + right) / 2.0
        for left, right in zip(scout_distribution, attention_distribution, strict=True)
    ]
    js_divergence = 0.0
    for left, right, middle in zip(
        scout_distribution, attention_distribution, midpoint, strict=True
    ):
        if left > 0.0:
            js_divergence += 0.5 * left * math.log(left / middle)
        if right > 0.0:
            js_divergence += 0.5 * right * math.log(right / middle)
    normalized_l1_distance = 0.5 * math.fsum(
        abs(left - right)
        for left, right in zip(scout_distribution, attention_distribution, strict=True)
    )
    values: dict[str, float | None] = {
        "pearson": pearson_value,
        "cosine": cosine_value,
        "js_divergence": js_divergence,
        "normalized_l1_distance": normalized_l1_distance,
        "distribution_overlap": 1.0 - normalized_l1_distance,
    }
    return {
        key: value if value is None or math.isfinite(value) else None
        for key, value in values.items()
    }


def _ndcg(candidate_order: list[int], relevance: list[float], k: int) -> float:
    """Compute nDCG for a candidate order against a relevance vector."""
    minimum = min(relevance)
    shifted = [value - minimum if minimum < 0.0 else value for value in relevance]

    def discounted_gain(order: list[int]) -> float:
        return math.fsum(
            shifted[position] / math.log2(rank + 2)
            for rank, position in enumerate(order[:k])
        )

    ideal_order = sorted(
        range(len(shifted)), key=lambda position: (-shifted[position], position)
    )
    ideal_gain = discounted_gain(ideal_order)
    if ideal_gain == 0.0:
        return 1.0
    return discounted_gain(candidate_order) / ideal_gain


def symmetric_ndcg(
    scout_scores: list[float],
    attention_scores: list[float],
    ratios: Iterable[float],
) -> dict[str, dict[str, float | int]]:
    """Compare both rankings using the other score vector as relevance."""
    if len(scout_scores) != len(attention_scores) or not scout_scores:
        raise ValueError("score vectors must have equal non-zero length")
    scout_order = deterministic_order(scout_scores)
    attention_order = deterministic_order(attention_scores)
    n = len(scout_scores)
    result: dict[str, dict[str, float | int]] = {}
    for ratio in ratios:
        if not 0.0 < ratio <= 1.0:
            raise ValueError("Top-k ratios must be in (0, 1]")
        k = min(n, max(1, math.ceil(ratio * n)))
        scout_against_attention = _ndcg(scout_order, attention_scores, k)
        attention_against_scout = _ndcg(attention_order, scout_scores, k)
        result[f"{ratio:g}"] = {
            "k": k,
            "scout_against_attention": scout_against_attention,
            "attention_against_scout": attention_against_scout,
            "symmetric": (scout_against_attention + attention_against_scout) / 2.0,
        }
    return result


def rank_biased_overlap(
    scout_scores: list[float],
    attention_scores: list[float],
    persistence: float = 0.9,
) -> float:
    """Compute a top-weighted rank-biased overlap for two full rankings."""
    if len(scout_scores) != len(attention_scores) or not scout_scores:
        raise ValueError("score vectors must have equal non-zero length")
    if not 0.0 <= persistence < 1.0:
        raise ValueError("persistence must be in [0, 1)")
    scout_order = deterministic_order(scout_scores)
    attention_order = deterministic_order(attention_scores)
    scout_seen: set[int] = set()
    attention_seen: set[int] = set()
    weighted_overlap = 0.0
    overlap = 0
    for depth, (scout_position, attention_position) in enumerate(
        zip(scout_order, attention_order, strict=True), start=1
    ):
        scout_seen.add(scout_position)
        attention_seen.add(attention_position)
        overlap = len(scout_seen & attention_seen)
        weighted_overlap += persistence ** (depth - 1) * overlap / depth
    return (1.0 - persistence) * weighted_overlap + (
        persistence ** len(scout_order)
    ) * overlap / len(scout_order)


def bucket_ids(scores: list[float], ratios: Iterable[float]) -> list[int]:
    """Assign deterministic high-to-low precision bucket IDs by rank."""
    ratios_tuple = _validate_ratios(ratios, "bucket_ratios")
    order = deterministic_order(scores)
    counts = [math.floor(ratio * len(scores)) for ratio in ratios_tuple[:-1]]
    counts.append(len(scores) - sum(counts))
    result = [0] * len(scores)
    cursor = 0
    for bucket, count in enumerate(counts):
        for position in order[cursor : cursor + count]:
            result[position] = bucket
        cursor += count
    if cursor != len(scores):
        raise AssertionError("bucket assignment did not cover all tokens")
    return result


def _bucket_ids_with_nonfinite_safety(
    scores: list[float], ratios: Iterable[float]
) -> tuple[list[int], int]:
    """Assign buckets while applying MaKV's fail-closed BF16 safety rule.

    ``bucket_ids`` deliberately remains strict because it is also used by
    rank metrics. This companion is only for precision-bucket comparison:
    NaN/+Inf/-Inf scores are not rankable and therefore go directly to bucket
    zero (BF16); configured ratios are applied to the remaining finite scores.
    """
    ratios_tuple = _validate_ratios(ratios, "bucket_ratios")
    result = [0] * len(scores)
    finite_positions: list[int] = []
    finite_scores: list[float] = []
    forced_count = 0
    for position, score in enumerate(scores):
        if math.isfinite(float(score)):
            finite_positions.append(position)
            finite_scores.append(float(score))
        else:
            forced_count += 1
    if finite_scores:
        finite_buckets = bucket_ids(finite_scores, ratios_tuple)
        for position, assigned in zip(finite_positions, finite_buckets, strict=True):
            result[position] = assigned
    return result, forced_count


def bucket_agreement(
    scout_scores: list[float], attention_scores: list[float], ratios: Iterable[float]
) -> dict[str, Any]:
    """Compare current four-tier rank assignments and return a confusion matrix."""
    left, scout_nonfinite_count = _bucket_ids_with_nonfinite_safety(
        scout_scores, ratios
    )
    right, attention_nonfinite_count = _bucket_ids_with_nonfinite_safety(
        attention_scores, ratios
    )
    matrix = {name: {other: 0 for other in BUCKET_NAMES} for name in BUCKET_NAMES}
    for scout_bucket, attention_bucket in zip(left, right, strict=True):
        matrix[BUCKET_NAMES[scout_bucket]][BUCKET_NAMES[attention_bucket]] += 1
    sample_count = len(left)
    observed = (
        sum(
            scout_bucket == attention_bucket
            for scout_bucket, attention_bucket in zip(left, right, strict=True)
        )
        / sample_count
    )
    row_totals = {name: sum(row.values()) for name, row in matrix.items()}
    column_totals = {
        name: sum(matrix[row_name][name] for row_name in BUCKET_NAMES)
        for name in BUCKET_NAMES
    }
    expected = sum(row_totals[name] * column_totals[name] for name in BUCKET_NAMES) / (
        sample_count * sample_count
    )
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0
    weighted = 1.0 - statistics.fmean(
        abs(scout_bucket - attention_bucket) / (len(BUCKET_NAMES) - 1)
        for scout_bucket, attention_bucket in zip(left, right, strict=True)
    )
    return {
        "bucket_names": list(BUCKET_NAMES),
        "same_bucket_fraction": observed,
        "weighted_bucket_agreement": weighted,
        "cohen_kappa": kappa,
        "scout_to_attention_confusion": matrix,
        "nonfinite_forced_to_bf16": {
            "scoutrank": scout_nonfinite_count,
            "attention": attention_nonfinite_count,
        },
    }


def bucket_token_id_overlap(
    scout_scores: list[float],
    attention_scores: list[float],
    token_ids: list[int],
    ratios: Iterable[float],
) -> dict[str, Any]:
    """Compare token-ID sets assigned to the same precision bucket.

    This intentionally ignores order within a bucket.  A repeated token ID is
    counted once per prompt and bucket, which answers whether both scorers put
    the same vocabulary token in that precision tier rather than whether they
    chose the same occurrence position.  Position/occurrence counts are kept
    as diagnostics because repeated IDs are common in long prompts.
    """
    if len(scout_scores) != len(attention_scores):
        raise ValueError("score vectors must have equal length")
    if len(token_ids) != len(scout_scores) or not token_ids:
        raise ValueError("token_ids must match non-empty score vectors")
    ratios_tuple = _validate_ratios(ratios, "bucket_ratios")
    if len(ratios_tuple) != len(BUCKET_NAMES):
        raise ValueError(
            f"bucket_token_id_overlap requires {len(BUCKET_NAMES)} bucket ratios"
        )

    scout_bucket_ids, scout_nonfinite_count = _bucket_ids_with_nonfinite_safety(
        scout_scores, ratios_tuple
    )
    attention_bucket_ids, attention_nonfinite_count = (
        _bucket_ids_with_nonfinite_safety(attention_scores, ratios_tuple)
    )
    result: dict[str, Any] = {
        "unit": "unique_token_id_set",
        "order_sensitive": False,
        "duplicate_token_ids_collapsed": True,
        "nonfinite_forced_to_bf16": {
            "scoutrank": scout_nonfinite_count,
            "attention": attention_nonfinite_count,
        },
        "bucket_names": list(BUCKET_NAMES),
        "buckets": {},
    }
    for bucket_index, bucket_name in enumerate(BUCKET_NAMES):
        scout_positions = {
            index
            for index, assigned in enumerate(scout_bucket_ids)
            if assigned == bucket_index
        }
        attention_positions = {
            index
            for index, assigned in enumerate(attention_bucket_ids)
            if assigned == bucket_index
        }
        scout_token_set = {int(token_ids[index]) for index in scout_positions}
        attention_token_set = {
            int(token_ids[index]) for index in attention_positions
        }
        common = scout_token_set & attention_token_set
        union = scout_token_set | attention_token_set
        position_intersection = scout_positions & attention_positions
        result["buckets"][bucket_name] = {
            "scoutrank_position_count": len(scout_positions),
            "attention_position_count": len(attention_positions),
            "scoutrank_unique_token_id_count": len(scout_token_set),
            "attention_unique_token_id_count": len(attention_token_set),
            "same_unique_token_id_count": len(common),
            # The symmetric overlap rate is Jaccard.  Directional values are
            # included because bucket cardinalities can differ after ties or
            # safety handling in future callers.
            "overlap_rate": len(common) / len(union) if union else 1.0,
            "jaccard": len(common) / len(union) if union else 1.0,
            "scoutrank_bucket_recall": (
                len(common) / len(scout_token_set) if scout_token_set else 1.0
            ),
            "attention_bucket_recall": (
                len(common) / len(attention_token_set)
                if attention_token_set
                else 1.0
            ),
            "same_position_count": len(position_intersection),
            "same_position_fraction": (
                len(position_intersection) / len(scout_positions | attention_positions)
                if scout_positions | attention_positions
                else 1.0
            ),
        }
    return result


def block_scores(scores: list[float], block_size: int) -> list[float]:
    """Aggregate token scores into production-aligned block sums."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return [
        sum(scores[start : start + block_size])
        for start in range(0, len(scores), block_size)
    ]


def _top_preview(
    scores: list[float], token_ids: list[int], tokenizer: Any, count: int
) -> list[dict[str, Any]]:
    """Return a compact human-readable top-token preview."""
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    analysis_scores, _ = _finite_analysis_scores(scores)
    return [
        {
            "position": position,
            "token_id": token_ids[position],
            "token": tokens[position],
            "score": scores[position],
        }
        for position in deterministic_order(analysis_scores)[:count]
    ]


def _load_prompts(args: argparse.Namespace, tokenizer: Any) -> list[PromptInput]:
    """Load raw or LongBench prompts and tokenize each exactly once."""
    if args.dataset_path:
        examples = load_examples(
            Path(args.dataset_path), args.task, args.limit, args.offset
        )
        return [
            PromptInput(
                prompt_id=example.example_id,
                token_ids=prompt_ids(
                    tokenizer,
                    example,
                    args.prompt_run_id,
                    enable_thinking=args.enable_thinking,
                ),
                source="longbench",
                metadata={"task": args.task, "example_id": example.example_id},
            )
            for example in examples
        ]
    raw_prompt = args.prompt
    if args.prompt_file:
        raw_prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    if raw_prompt is None:
        raise ValueError(
            "one of --prompt, --prompt-file, or --dataset-path is required"
        )
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    token_ids = list(ids)
    return [
        PromptInput(
            prompt_id="raw-0",
            token_ids=token_ids,
            source="raw_prompt",
            metadata={"prompt_chars": len(raw_prompt)},
        )
    ]


def _dtype(name: str) -> torch.dtype:
    """Map CLI dtype names to torch dtypes."""
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _sync(device: torch.device) -> None:
    """Synchronize only for benchmark measurement boundaries."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean_metric(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    """Average a finite numeric metric over prompt rows."""
    values: list[float] = []
    for row in rows:
        value: Any = row
        for part in path:
            value = value[part]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                values.append(float(value))
    return statistics.fmean(values) if values else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the same-prompt ScoutRank versus attention overlap experiment."""
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, local_files_only=True
    )
    prompts = _load_prompts(args, tokenizer)
    if not prompts:
        raise ValueError("no prompts were loaded")
    if args.attention_matrix_output and len(prompts) != 1:
        raise ValueError(
            "--attention-matrix-output requires exactly one prompt "
            "so files are not overwritten"
        )
    for prompt in prompts:
        if not prompt.token_ids:
            raise ValueError(
                f"prompt {prompt.prompt_id} tokenized to an empty sequence"
            )
        if (
            args.max_attention_tokens > 0
            and len(prompt.token_ids) > args.max_attention_tokens
        ):
            raise ValueError(
                f"prompt {prompt.prompt_id} has {len(prompt.token_ids)} tokens, "
                "exceeding "
                f"--max-attention-tokens={args.max_attention_tokens}; set 0 to allow "
                "the query-chunked score path"
            )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=_dtype(args.dtype),
            low_cpu_mem_usage=True,
            attn_implementation=args.model_attention_implementation,
        )
        .to(device)
        .eval()
    )
    anchor_layers = _parse_anchor_layers(args.anchor_layers, args.mode, args.exit_layer)
    adapter, scorer, cfg = _build_scout_runtime(
        model,
        mode=args.mode,
        observer_backend=args.observer_backend,
        observer_token_chunk_size=args.observer_token_chunk_size,
        anchor_layers=anchor_layers,
        exit_layer=args.exit_layer,
        scoring_version=args.scoring_version,
    )
    topk_ratios = _parse_float_list(args.topk_ratios, "topk_ratios")
    bucket_ratios = _validate_ratios(
        _parse_float_list(args.bucket_ratios, "bucket_ratios"), "bucket_ratios"
    )
    rows: list[dict[str, Any]] = []
    try:
        for index, prompt in enumerate(prompts, start=1):
            ids = prompt.token_ids
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            _sync(device)
            started = time.perf_counter()
            scout_scores = _score_prompt(
                ids,
                device=device,
                adapter=adapter,
                scorer=scorer,
                scoring_version=args.scoring_version,
            )
            _sync(device)
            scout_ms = (time.perf_counter() - started) * 1000.0
            _sync(device)
            started = time.perf_counter()
            attention_scores, attention_meta = collect_attention_scores(
                model,
                input_ids,
                args.attention_aggregation,
                query_chunk_size=args.attention_query_chunk_size,
                layer_mode=args.attention_layer_mode,
                matrix_output=(
                    Path(args.attention_matrix_output)
                    if args.attention_matrix_output
                    else None
                ),
                matrix_dtype=args.attention_matrix_dtype,
            )
            if isinstance(attention_scores, dict):
                if args.attention_aggregation != "future_both":
                    raise RuntimeError(
                        "attention score variants are only expected for future_both"
                    )
                attention_variants = attention_scores
                attention_scores = [
                    float(value) for value in attention_variants["A_norm"]
                ]
                attention_meta = {
                    **attention_meta,
                    "scoring_version": "v3.2_attention_normalized",
                    "selected_variant": "A_norm",
                    "A_raw": [
                        float(value) for value in attention_variants["A_raw"]
                    ],
                    "A_norm": attention_scores,
                    "visible_count": [
                        int(value)
                        for value in attention_variants["visible_count"]
                    ],
                }
            _sync(device)
            attention_ms = (time.perf_counter() - started) * 1000.0
            if len(scout_scores) != len(attention_scores):
                raise RuntimeError("ScoutRank and attention score lengths differ")
            scout_analysis_scores, scout_nonfinite_count = _finite_analysis_scores(
                scout_scores
            )
            attention_analysis_scores, attention_nonfinite_count = (
                _finite_analysis_scores(attention_scores)
            )
            block_scout_scores = block_scores(
                scout_analysis_scores, args.block_size
            )
            block_attention_scores = block_scores(
                attention_analysis_scores, args.block_size
            )
            row = {
                "prompt_id": prompt.prompt_id,
                "source": prompt.source,
                "metadata": prompt.metadata,
                "token_count": len(ids),
                "prompt_token_hash": prompt_token_hash(ids),
                "token_ids": ids,
                "scoutrank_scores": scout_scores,
                "attention_scores": attention_scores,
                "scoutrank_top_preview": _top_preview(
                    scout_scores, ids, tokenizer, args.preview_count
                ),
                "attention_top_preview": _top_preview(
                    attention_scores, ids, tokenizer, args.preview_count
                ),
                "timing_ms": {
                    "scoutrank_forward_and_score": scout_ms,
                    "attention_forward_and_aggregate": attention_ms,
                },
                "analysis_score_nonfinite_count": {
                    "scoutrank": scout_nonfinite_count,
                    "attention": attention_nonfinite_count,
                },
                "ranking": ranking_metrics(
                    scout_analysis_scores, attention_analysis_scores
                ),
                "score_similarity": score_similarity_metrics(
                    scout_analysis_scores, attention_analysis_scores
                ),
                "ranking_similarity": {
                    "symmetric_ndcg": symmetric_ndcg(
                        scout_analysis_scores,
                        attention_analysis_scores,
                        topk_ratios,
                    ),
                    "rbo_p09": rank_biased_overlap(
                        scout_analysis_scores,
                        attention_analysis_scores,
                        persistence=0.9,
                    ),
                },
                "topk": topk_overlap(
                    scout_analysis_scores, attention_analysis_scores, topk_ratios
                ),
                "bucket_agreement": bucket_agreement(
                    scout_scores, attention_scores, bucket_ratios
                ),
                "bucket_token_id_overlap": bucket_token_id_overlap(
                    scout_scores, attention_scores, ids, bucket_ratios
                ),
                "block_32": {
                    "scoutrank_scores": block_scout_scores,
                    "attention_scores": block_attention_scores,
                    "ranking": ranking_metrics(
                        block_scout_scores,
                        block_attention_scores,
                    ),
                    "score_similarity": score_similarity_metrics(
                        block_scout_scores, block_attention_scores
                    ),
                    "ranking_similarity": {
                        "symmetric_ndcg": symmetric_ndcg(
                            block_scout_scores,
                            block_attention_scores,
                            topk_ratios,
                        ),
                        "rbo_p09": rank_biased_overlap(
                            block_scout_scores,
                            block_attention_scores,
                            persistence=0.9,
                        ),
                    },
                    "topk": topk_overlap(
                        block_scout_scores,
                        block_attention_scores,
                        topk_ratios,
                    ),
                },
                "attention": attention_meta,
            }
            rows.append(row)
            top10 = row["topk"].get("0.1", {})
            print(
                f"[{index}/{len(prompts)}] id={prompt.prompt_id} "
                f"tokens={len(ids)} scout_ms={scout_ms:.3f} "
                f"attention_ms={attention_ms:.3f} "
                f"spearman={row['ranking']['spearman']} "
                f"cosine={row['score_similarity']['cosine']} "
                f"rbo_p09={row['ranking_similarity']['rbo_p09']} "
                f"top10_overlap={top10.get('overlap')}/{top10.get('k')}",
                flush=True,
            )
    finally:
        del adapter, scorer, cfg, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    aggregate = {
        "prompt_count": len(rows),
        "token_count_total": sum(row["token_count"] for row in rows),
        "token_count_mean": statistics.fmean(row["token_count"] for row in rows),
        "spearman_mean": _mean_metric(rows, ("ranking", "spearman")),
        "kendall_tau_b_mean": _mean_metric(rows, ("ranking", "kendall_tau_b")),
        "mean_absolute_rank_difference_mean": _mean_metric(
            rows, ("ranking", "mean_absolute_rank_difference")
        ),
        "same_bucket_fraction_mean": _mean_metric(
            rows, ("bucket_agreement", "same_bucket_fraction")
        ),
        "weighted_bucket_agreement_mean": _mean_metric(
            rows, ("bucket_agreement", "weighted_bucket_agreement")
        ),
        "cohen_kappa_mean": _mean_metric(rows, ("bucket_agreement", "cohen_kappa")),
        "bucket_token_id_overlap_mean": {
            bucket_name: {
                metric: _mean_metric(
                    rows,
                    (
                        "bucket_token_id_overlap",
                        "buckets",
                        bucket_name,
                        metric,
                    ),
                )
                for metric in (
                    "overlap_rate",
                    "jaccard",
                    "scoutrank_bucket_recall",
                    "attention_bucket_recall",
                    "same_position_fraction",
                )
            }
            for bucket_name in BUCKET_NAMES
        },
        "score_similarity_mean": {
            metric: _mean_metric(rows, ("score_similarity", metric))
            for metric in (
                "pearson",
                "cosine",
                "js_divergence",
                "normalized_l1_distance",
                "distribution_overlap",
            )
        },
        "symmetric_ndcg_mean": {
            key: _mean_metric(
                rows,
                ("ranking_similarity", "symmetric_ndcg", key, "symmetric"),
            )
            for key in (f"{ratio:g}" for ratio in topk_ratios)
        },
        "rbo_p09_mean": _mean_metric(rows, ("ranking_similarity", "rbo_p09")),
        "scoutrank_time_ms_mean": _mean_metric(
            rows, ("timing_ms", "scoutrank_forward_and_score")
        ),
        "attention_time_ms_mean": _mean_metric(
            rows, ("timing_ms", "attention_forward_and_aggregate")
        ),
        "topk_mean": {
            key: {
                metric: _mean_metric(rows, ("topk", key, metric))
                for metric in ("overlap", "recall", "precision", "jaccard")
            }
            for key in (f"{ratio:g}" for ratio in topk_ratios)
        },
        "block_32_topk_mean": {
            key: {
                metric: _mean_metric(rows, ("block_32", "topk", key, metric))
                for metric in ("overlap", "recall", "precision", "jaccard")
            }
            for key in (f"{ratio:g}" for ratio in topk_ratios)
        },
        "block_32_score_similarity_mean": {
            metric: _mean_metric(rows, ("block_32", "score_similarity", metric))
            for metric in (
                "pearson",
                "cosine",
                "js_divergence",
                "normalized_l1_distance",
                "distribution_overlap",
            )
        },
        "block_32_symmetric_ndcg_mean": {
            key: _mean_metric(
                rows,
                ("block_32", "ranking_similarity", "symmetric_ndcg", key, "symmetric"),
            )
            for key in (f"{ratio:g}" for ratio in topk_ratios)
        },
        "block_32_rbo_p09_mean": _mean_metric(
            rows, ("block_32", "ranking_similarity", "rbo_p09")
        ),
    }
    return {
        "schema_version": 2,
        "method": {
            "scoutrank": f"ScoutRank {args.scoring_version} token importance",
            "attention": (
                "same Qwen3 scout model, chunked QK-softmax attention, "
                f"{args.attention_layer_mode} layer, "
                f"{args.attention_aggregation} aggregation"
            ),
            "same_token_sequence": True,
            "attention_weights_retained": False,
            "production_policy_modified": False,
            "bucket_token_id_metric": (
                "per-bucket unique token-ID set overlap; order within a bucket is ignored"
            ),
            "nonfinite_analysis_policy": (
                "non-finite safety scores stay raw and are forced to BF16 for bucket "
                "metrics; auxiliary rank metrics use a deterministic finite sentinel"
            ),
        },
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "device": str(device),
        "dtype": args.dtype,
        "scoring_mode": args.mode,
        "scoring_version": args.scoring_version,
        "observer_backend": args.observer_backend,
        "anchor_layers": list(anchor_layers),
        "exit_layer": args.exit_layer,
        "attention_layer_mode": args.attention_layer_mode,
        "attention_matrix_output": args.attention_matrix_output,
        "attention_matrix_dtype": args.attention_matrix_dtype,
        "block_size": args.block_size,
        "bucket_ratios": list(bucket_ratios),
        "bucket_names": list(BUCKET_NAMES),
        "topk_ratios": list(topk_ratios),
        "aggregate": aggregate,
        "prompts": rows,
    }


def main() -> None:
    """Parse CLI arguments, run the experiment, and write JSON output."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file")
    source.add_argument("--dataset-path")
    parser.add_argument("--task", default="hotpotqa")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--prompt-run-id", default="scoutrank-attention-overlap")
    parser.add_argument("--model", required=True, help="Qwen3-0.6B scout model")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--mode", choices=("fast", "balanced"), default="balanced")
    parser.add_argument(
        "--scoring-version",
        choices=(
            "v2",
            "v3-functional-disturbance",
            "v3.1-projected",
            "v3.1-probe-normalized",
            "v3_fast_d22",
            "v3_fast_1_exact_scalar_d22",
        ),
        default=os.getenv("SCOUT_SCORING_VERSION", "v2"),
        help=(
            "ScoutRank implementation used for the comparison. The default "
            "v2 preserves the historical offline experiment; use "
            "v3_fast_1_exact_scalar_d22 to validate the current fast scorer."
        ),
    )
    parser.add_argument(
        "--observer-backend", choices=("vectorized", "production"), default="vectorized"
    )
    parser.add_argument("--observer-token-chunk-size", type=int, default=4096)
    parser.add_argument("--anchor-layers", default=None)
    parser.add_argument("--exit-layer", type=int, default=None)
    parser.add_argument(
        "--attention-aggregation",
        choices=("incoming_mean", "incoming_sum", "future_both", "last_query"),
        default="incoming_mean",
    )
    parser.add_argument(
        "--attention-query-chunk-size",
        type=int,
        default=256,
        help=(
            "Number of query rows used by the offline score collector. "
            "Smaller values reduce peak memory for long prompts."
        ),
    )
    parser.add_argument(
        "--attention-layer-mode",
        choices=("first", "last", "all"),
        default="last",
        help=(
            "Collect only the first or final transformer layer by default; "
            "use all for layer averaging."
        ),
    )
    parser.add_argument(
        "--attention-matrix-output",
        default=None,
        help="Optional raw output path for a complete first-layer attention matrix.",
    )
    parser.add_argument(
        "--attention-matrix-dtype",
        choices=("model", "bfloat16", "float16", "float32"),
        default="model",
        help="Storage dtype for the optional full matrix export.",
    )
    parser.add_argument(
        "--model-attention-implementation",
        choices=("sdpa", "eager"),
        default="sdpa",
        help=(
            "Attention implementation used for model outputs; "
            "use sdpa for long prompts."
        ),
    )
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--topk-ratios",
        default=",".join(str(value) for value in DEFAULT_TOPK_RATIOS),
    )
    parser.add_argument(
        "--bucket-ratios",
        default=",".join(str(value) for value in DEFAULT_BUCKET_RATIOS),
    )
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument(
        "--max-attention-tokens",
        type=int,
        default=0,
        help=(
            "Optional prompt-length guard for the offline comparison; "
            "0 allows all lengths because score attention is query-chunked."
        ),
    )
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote attention-overlap report to {output}")


if __name__ == "__main__":
    main()
