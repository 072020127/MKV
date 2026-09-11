# SPDX-License-Identifier: Apache-2.0

"""Client-side submit/wait helpers for ScoutRank/main-model overlap."""

from __future__ import annotations

# Standard
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from urllib.parse import urlparse
import json
import socket
import threading
import time

# First Party
from lmcache.v1.storage_backend.makv.config import (
    IMPORTANCE_LAYOUT_REQUEST_KEY,
    IMPORTANCE_REQUEST_KEY,
    IMPORTANCE_STATUS_REQUEST_KEY,
    SCOUT_SCORE_END_REQUEST_KEY,
    SCOUT_SCORE_START_REQUEST_KEY,
)
from lmcache.v1.storage_backend.makv.metrics import CLIENT_METRICS
from lmcache.v1.storage_backend.makv_remote.protocol import (
    FRAME_HEADER,
    MAX_HEADER_BYTES,
)
from lmcache.v1.storage_backend.makv_remote.scout_protocol import (
    SCOUT_PROTOCOL_VERSION,
    SCOUT_SUFFIX_PROTOCOL_VERSION,
    decode_scores,
    encode_token_ids,
    payload_sha256,
    score_context_sha256,
)


@dataclass(frozen=True)
class ScoutOverlapResult:
    scores: list[float]
    score_time_ms: float
    total_job_time_ms: float
    wait_time_ms: float
    overlap_hidden_time_ms: float
    score_start: int = 0
    score_end: int = 0
    full_token_count: int = 0


_RESULT_CACHE_CAPACITY = 128
_RESULT_CACHE: OrderedDict[str, tuple[int, ScoutOverlapResult]] = OrderedDict()
_RESULT_CACHE_LOCK = threading.Lock()


class ScoutOverlapClient:
    """Stateless synchronous client used by scheduler and worker processes."""

    def __init__(self, url: str, timeout_s: float) -> None:
        parsed = urlparse(url)
        if (
            parsed.scheme != "makv"
            or parsed.hostname is None
            or parsed.port is None
        ):
            raise ValueError("ScoutRank URL must use makv://host:port")
        if timeout_s <= 0:
            raise ValueError("ScoutRank timeout must be positive")
        self.host = parsed.hostname
        self.port = parsed.port
        self.timeout_s = timeout_s

    @staticmethod
    def _recv_exact(connection: socket.socket, size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            chunk = connection.recv(size - len(result))
            if not chunk:
                raise ConnectionError("MaKV manager closed an incomplete frame")
            result.extend(chunk)
        return bytes(result)

    def _request(
        self, header: dict[str, Any], payload: bytes = b""
    ) -> tuple[dict[str, Any], bytes]:
        header_bytes = json.dumps(
            header, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        if len(header_bytes) > MAX_HEADER_BYTES:
            raise ValueError("ScoutRank request header exceeds protocol limit")
        with socket.create_connection(
            (self.host, self.port), self.timeout_s
        ) as connection:
            connection.settimeout(self.timeout_s)
            connection.sendall(
                FRAME_HEADER.pack(len(header_bytes), len(payload))
                + header_bytes
                + payload
            )
            raw_frame = self._recv_exact(connection, FRAME_HEADER.size)
            header_length, payload_length = FRAME_HEADER.unpack(raw_frame)
            if header_length <= 0 or header_length > MAX_HEADER_BYTES:
                raise ValueError("invalid MaKV response header length")
            response = json.loads(
                self._recv_exact(connection, header_length).decode("utf-8")
            )
            response_payload = self._recv_exact(connection, payload_length)
        if response.get("status") != "ok":
            raise RuntimeError(
                str(response.get("error", "ScoutRank manager request failed"))
            )
        return response, response_payload

    def submit(
        self,
        request_id: str,
        token_ids: Sequence[int],
        *,
        score_start: int = 0,
        full_token_count: Optional[int] = None,
    ) -> dict[str, Any]:
        """Submit one request-local score range without waiting.

        Protocol v1 remains the wire representation for a full-prompt score.
        A strict absolute range is added only when the caller submits a
        suffix, so old full-prompt clients and managers remain compatible.
        """
        token_count = len(token_ids)
        if full_token_count is None:
            full_token_count = token_count
        score_end = score_start + token_count
        if not 0 <= score_start <= score_end <= full_token_count:
            raise ValueError("ScoutRank score range is outside the full prompt")
        payload = encode_token_ids(token_ids)
        suffix = score_start != 0 or score_end != full_token_count
        header: dict[str, Any] = {
            "op": "SCOUT_SUBMIT",
            "key": request_id,
            "protocol_version": (
                SCOUT_SUFFIX_PROTOCOL_VERSION
                if suffix
                else SCOUT_PROTOCOL_VERSION
            ),
            "token_count": token_count,
            "token_sha256": payload_sha256(payload),
        }
        if suffix:
            header.update(
                {
                    "score_start": score_start,
                    "score_end": score_end,
                    "full_token_count": full_token_count,
                    "score_context_sha256": score_context_sha256(
                        header["token_sha256"], score_start, full_token_count
                    ),
                }
            )
        response, _ = self._request(
            header,
            payload,
        )
        return response

    def wait(
        self,
        request_id: str,
        token_count: int,
        *,
        deferred: bool = False,
        score_start: Optional[int] = None,
        score_end: Optional[int] = None,
    ) -> ScoutOverlapResult:
        """Fetch scores at the store boundary after overlapped prefill work."""
        if (score_start is None) != (score_end is None):
            raise ValueError("ScoutRank wait requires both score range endpoints")
        suffix = score_start is not None
        if suffix and not 0 <= score_start <= score_end <= token_count:
            raise ValueError("ScoutRank wait score range is outside the full prompt")
        header: dict[str, Any] = {
            "op": "SCOUT_WAIT",
            "key": request_id,
            "protocol_version": (
                SCOUT_SUFFIX_PROTOCOL_VERSION
                if suffix
                else SCOUT_PROTOCOL_VERSION
            ),
            "token_count": token_count,
            "timeout_s": self.timeout_s,
            "deferred": deferred,
        }
        if suffix:
            header.update(
                {
                    "score_start": score_start,
                    "score_end": score_end,
                    "full_token_count": token_count,
                }
            )
        response, payload = self._request(header)
        scored_count = int(response.get("scored_token_count", token_count))
        result_start = int(response.get("score_start", 0))
        result_end = int(response.get("score_end", result_start + scored_count))
        result_full_count = int(
            response.get("full_token_count", token_count)
        )
        if (
            result_full_count != token_count
            or result_end < result_start
            or result_end - result_start != scored_count
        ):
            raise ValueError("ScoutRank response score range is invalid")
        if suffix and (result_start != score_start or result_end != score_end):
            raise ValueError("ScoutRank response score range does not match wait")
        return ScoutOverlapResult(
            scores=decode_scores(payload, scored_count),
            score_time_ms=float(response["score_time_ms"]),
            total_job_time_ms=float(response["total_job_time_ms"]),
            wait_time_ms=float(response["wait_time_ms"]),
            overlap_hidden_time_ms=float(response["overlap_hidden_time_ms"]),
            score_start=result_start,
            score_end=result_end,
            full_token_count=result_full_count,
        )


def scout_overlap_enabled(config: Any) -> bool:
    """Gate all overlap behavior behind an explicit MaKV-only switch."""
    if getattr(config, "remote_serde", None) != "makv":
        return False
    extra = getattr(config, "extra_config", None) or {}
    return bool(extra.get("makv_scout_overlap_enabled", False))


def _client_from_config(config: Any) -> ScoutOverlapClient:
    extra = getattr(config, "extra_config", None) or {}
    url = str(extra.get("makv_scout_url") or getattr(config, "remote_url", ""))
    timeout_s = float(extra.get("makv_scout_timeout_s", 60.0))
    return ScoutOverlapClient(url, timeout_s)


def scout_suffix_only_enabled(config: Any) -> bool:
    """Return whether MaKV overlap should score only the uncached suffix."""
    extra = getattr(config, "extra_config", None) or {}
    return bool(
        extra.get(
            "makv_scout_suffix_only", getattr(config, "scout_suffix_only", True)
        )
    )


def get_scout_score_range(
    config: Any,
    *,
    token_count: int,
    cached_tokens: int,
    request_configs: Optional[dict[str, Any]],
) -> Optional[tuple[int, int]]:
    """Return the absolute request range that ScoutRank must score.

    The range ends at the last cacheable token when partial chunks are not
    stored.  This prevents spending scorer work on a tail that cannot become a
    MaKV object.  ``None`` means no score is needed, or explicit importance is
    already present in the request.
    """
    if not scout_overlap_enabled(config):
        return None
    if request_configs and request_configs.get(IMPORTANCE_REQUEST_KEY) is not None:
        return None
    if token_count < 0:
        raise ValueError("ScoutRank token_count must be non-negative")
    chunk_size = int(getattr(config, "chunk_size", 1))
    if chunk_size <= 0:
        raise ValueError("LMCache chunk_size must be positive")
    save_unfull_chunk = bool(getattr(config, "save_unfull_chunk", False))
    cacheable_tokens = (
        token_count
        if save_unfull_chunk
        else token_count // chunk_size * chunk_size
    )
    if cacheable_tokens <= 0:
        return None
    hit_tokens = min(max(int(cached_tokens), 0), cacheable_tokens)
    # External lookup results should be chunk aligned.  Rounding down is the
    # safe behavior if a custom lookup client reports a partial hit.
    hit_tokens = hit_tokens // chunk_size * chunk_size
    if hit_tokens >= cacheable_tokens:
        return None
    if not scout_suffix_only_enabled(config):
        return 0, token_count
    return hit_tokens, cacheable_tokens


def submit_scout_if_needed(
    config: Any,
    *,
    request_id: str,
    token_ids: Sequence[int],
    request_configs: Optional[dict[str, Any]],
    cached_tokens: int,
) -> bool:
    """Start ScoutRank before prefill when this request will store new KV."""
    if not scout_overlap_enabled(config):
        return False
    if request_configs and request_configs.get(IMPORTANCE_REQUEST_KEY) is not None:
        return False
    score_range = get_scout_score_range(
        config,
        token_count=len(token_ids),
        cached_tokens=cached_tokens,
        request_configs=request_configs,
    )
    if score_range is None:
        return False
    score_start, score_end = score_range
    started = time.perf_counter()
    client = _client_from_config(config)
    if score_start == 0 and score_end == len(token_ids):
        # Keep the v1 call shape for full prompts and simple test doubles.
        client.submit(request_id, token_ids)
    else:
        client.submit(
            request_id,
            token_ids[score_start:score_end],
            score_start=score_start,
            full_token_count=len(token_ids),
        )
    CLIENT_METRICS.add(
        makv_scout_submit_calls=1,
        makv_scout_submit_time_ms=(time.perf_counter() - started) * 1000.0,
    )
    return True


def resolve_scout_importance(
    config: Any,
    *,
    request_id: Optional[str],
    token_count: Optional[int],
    request_configs: Optional[dict[str, Any]],
    deferred: bool = False,
    score_start: Optional[int] = None,
    score_end: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """Wait for scores and return a request-local config copy for serialization."""
    if not scout_overlap_enabled(config):
        return request_configs
    if request_configs and request_configs.get(IMPORTANCE_REQUEST_KEY) is not None:
        return request_configs
    if not request_id or token_count is None or token_count < 0:
        raise ValueError("ScoutRank overlap requires request_id and prompt token count")
    if (score_start is None) != (score_end is None):
        raise ValueError("ScoutRank resolve requires both score range endpoints")
    if score_start is not None and not 0 <= score_start <= score_end <= token_count:
        raise ValueError("ScoutRank resolve score range is outside the full prompt")
    with _RESULT_CACHE_LOCK:
        cached = _RESULT_CACHE.get(request_id)
        if cached is not None:
            if cached[0] != token_count:
                raise ValueError(
                    "cached ScoutRank result token count does not match request"
                )
            _RESULT_CACHE.move_to_end(request_id)
            result = cached[1]
        else:
            result = None
    if result is None:
        started = time.perf_counter()
        client = _client_from_config(config)
        if score_start is None:
            result = client.wait(request_id, token_count, deferred=deferred)
        else:
            result = client.wait(
                request_id,
                token_count,
                deferred=deferred,
                score_start=score_start,
                score_end=score_end,
            )
        client_wait_ms = (time.perf_counter() - started) * 1000.0
        CLIENT_METRICS.add(
            makv_scout_wait_calls=1,
            makv_scout_wait_time_ms=client_wait_ms,
            makv_scout_score_time_ms=result.score_time_ms,
            makv_scout_overlap_hidden_time_ms=result.overlap_hidden_time_ms,
        )
        with _RESULT_CACHE_LOCK:
            _RESULT_CACHE[request_id] = (token_count, result)
            _RESULT_CACHE.move_to_end(request_id)
            while len(_RESULT_CACHE) > _RESULT_CACHE_CAPACITY:
                _RESULT_CACHE.popitem(last=False)
    else:
        client_wait_ms = 0.0
    scores, status = _expand_scout_scores(
        result,
        token_count,
        requested_start=score_start,
        requested_end=score_end,
    )
    resolved = dict(request_configs or {})
    resolved[IMPORTANCE_REQUEST_KEY] = scores
    resolved[IMPORTANCE_LAYOUT_REQUEST_KEY] = "token"
    if status is not None:
        effective_start = result.score_start if score_start is None else score_start
        effective_end = result.score_end if score_end is None else score_end
        resolved[IMPORTANCE_STATUS_REQUEST_KEY] = status
        resolved[SCOUT_SCORE_START_REQUEST_KEY] = effective_start
        resolved[SCOUT_SCORE_END_REQUEST_KEY] = effective_end
    resolved["lmcache.makv_scoutrank_timing"] = {
        "score_time_ms": result.score_time_ms,
        "total_job_time_ms": result.total_job_time_ms,
        "manager_wait_time_ms": result.wait_time_ms,
        "client_wait_time_ms": client_wait_ms,
        "overlap_hidden_time_ms": result.overlap_hidden_time_ms,
    }
    return resolved


def _expand_scout_scores(
    result: ScoutOverlapResult,
    token_count: int,
    *,
    requested_start: Optional[int] = None,
    requested_end: Optional[int] = None,
) -> tuple[list[float], Optional[list[dict[str, Any]]]]:
    """Map a range-scored result back to absolute request token positions."""
    if result.full_token_count == 0:
        # Results constructed by older callers/tests predate range metadata and
        # are necessarily full-prompt results.
        result_full = token_count
        result_start = 0
        result_end = token_count
    else:
        result_full = result.full_token_count
        result_start = result.score_start
        result_end = result.score_end
    if result_full != token_count:
        raise ValueError("ScoutRank result full token count does not match request")
    if not 0 <= result_start <= result_end <= token_count:
        raise ValueError("ScoutRank result score range is invalid")
    if result_end - result_start != len(result.scores):
        raise ValueError("ScoutRank result score count does not match its range")
    if (requested_start is None) != (requested_end is None):
        raise ValueError("ScoutRank requested range is incomplete")

    target_start = result_start if requested_start is None else requested_start
    target_end = result_end if requested_end is None else requested_end
    if not 0 <= target_start <= target_end <= token_count:
        raise ValueError("ScoutRank requested score range is invalid")
    if target_start < result_start or target_end > result_end:
        raise ValueError("ScoutRank result does not cover requested score range")

    if target_start == 0 and target_end == token_count:
        return list(result.scores), None

    source_offset = target_start - result_start
    scores = [0.0] * token_count
    scores[target_start:target_end] = result.scores[
        source_offset : source_offset + target_end - target_start
    ]
    status = []
    for index in range(token_count):
        if target_start <= index < target_end:
            status.append({"valid_mask": True, "forced_precision": None})
        else:
            status.append(
                {
                    "valid_mask": False,
                    "forced_precision": "BF16",
                    "reason": "SCOUT_SUFFIX_NOT_SCORED",
                }
            )
    return scores, status
