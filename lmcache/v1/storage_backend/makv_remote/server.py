# SPDX-License-Identifier: Apache-2.0

"""Command-line TCP server for the independent MaKV Remote Manager."""

# Standard
from dataclasses import dataclass
from typing import Any
import argparse
import asyncio
import json
import math
import socket
import time

# First Party
from lmcache.v1.storage_backend.makv.config import (
    DEFAULT_MAKV_STORAGE_URL,
    SUPPORTED_BUCKET_BITS,
    SUPPORTED_ENTROPY_BACKENDS,
    SUPPORTED_ENTROPY_CODECS,
    SUPPORTED_PRECISION_SCHEMES,
    SUPPORTED_RESIDUAL_DTYPES,
    MaKVConfig,
    default_makv_bucket_ratios,
)
from lmcache.v1.storage_backend.makv.metrics import REMOTE_METRICS
from lmcache.v1.storage_backend.makv_remote.manager import MaKVRemoteManager
from lmcache.v1.storage_backend.makv_remote.protocol import (
    BATCH_BLOB_VERSION,
    BATCH_STREAM_VERSION,
    DEFAULT_MAX_PAYLOAD_BYTES,
    PRECISION_RISK_OPERATION,
    batch_blob_size,
    read_frame,
    write_batch_blob_frame,
    write_frame,
)
from lmcache.v1.storage_backend.makv_remote.scout_protocol import (
    SCOUT_PROTOCOL_VERSION,
    SCOUT_SUFFIX_PROTOCOL_VERSION,
    decode_token_ids,
    encode_scores,
    payload_sha256,
    score_context_sha256,
)
from lmcache.v1.storage_backend.makv_remote.scout_service import ScoutJobService
from lmcache.v1.storage_backend.makv_remote.storage_adapter import (
    SUPPORTED_STORAGE_BACKENDS,
    create_storage_adapter,
    infer_storage_backend,
)


# Keep GET_BATCH streaming while avoiding one writer.drain() per LMCache
# chunk.  A bounded threshold prevents a large prompt from accumulating the
# entire response in the asyncio transport buffer.
_BATCH_DRAIN_BYTES = 64 * 1024 * 1024
_DEFAULT_SOCKET_BUFFER_BYTES = 16 * 1024 * 1024
_DEFAULT_BATCH_STREAM_PREFETCH_DEPTH = 4
_MAX_LIST_KEY_HASHES = 4096

# These are live-manager defaults only.  Keep ScoutRankConfig's frozen v3
# defaults unchanged so offline/reproducibility workflows retain their prior
# behavior.
_DEFAULT_LIVE_SCOUT_SCORING_VERSION = "v3_fast_1_exact_scalar_d22"
_DEFAULT_LIVE_SCOUT_NUM_PROBES = 32
_DEFAULT_LIVE_SCOUT_TAIL_PROBES = 16
_DEFAULT_LIVE_SCOUT_PROBE_SELECTION = "mix32"


def _configure_client_socket(
    writer: asyncio.StreamWriter, buffer_bytes: int
) -> None:
    """Tune the accepted socket for large response batches."""
    if buffer_bytes <= 0:
        return
    sock = writer.get_extra_info("socket")
    if sock is None:
        return
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_bytes)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_bytes)
        if hasattr(socket, "TCP_NODELAY"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        # The kernel may cap or reject buffer tuning. This is optional and
        # must never change the protocol behavior.
        return


@dataclass
class PutJob:
    key: str
    payload: bytes
    enqueued_at: float
    result: asyncio.Future[int]


class MaKVRemoteServer:
    """Serve MaKV storage operations across a real TCP process boundary."""

    def __init__(
        self,
        manager: MaKVRemoteManager,
        *,
        queue_depth: int,
        workers: int,
        max_request_bytes: int,
        queue_wait_timeout: float = 600.0,
        socket_buffer_bytes: int = _DEFAULT_SOCKET_BUFFER_BYTES,
        batch_stream_prefetch_depth: int = _DEFAULT_BATCH_STREAM_PREFETCH_DEPTH,
        scout_service: ScoutJobService | None = None,
        scout_max_wait_timeout: float = 600.0,
    ) -> None:
        if (
            queue_depth <= 0
            or workers <= 0
            or queue_wait_timeout <= 0
            or socket_buffer_bytes < 0
            or batch_stream_prefetch_depth <= 0
            or scout_max_wait_timeout <= 0
        ):
            raise ValueError(
                "queue_depth, workers and queue_wait_timeout must be positive; "
                "socket_buffer_bytes must be non-negative"
            )
        self.manager = manager
        self.queue: asyncio.Queue[PutJob | None] = asyncio.Queue(queue_depth)
        self.worker_count = workers
        self.max_request_bytes = max_request_bytes
        self.queue_wait_timeout = queue_wait_timeout
        self.socket_buffer_bytes = socket_buffer_bytes
        self.batch_stream_prefetch_depth = batch_stream_prefetch_depth
        self.scout_service = scout_service
        self.scout_max_wait_timeout = scout_max_wait_timeout
        self.active_jobs = 0
        self.worker_tasks: list[asyncio.Task[None]] = []

    async def start_workers(self) -> None:
        """Start bounded quantization workers."""
        self.worker_tasks = [
            asyncio.create_task(self._worker(), name=f"makv-worker-{index}")
            for index in range(self.worker_count)
        ]

    async def close(self) -> None:
        """Stop all quantization workers."""
        for _ in self.worker_tasks:
            await self.queue.put(None)
        await asyncio.gather(*self.worker_tasks)
        self.worker_tasks.clear()
        if self.scout_service is not None:
            self.scout_service.close()

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            try:
                if job is None:
                    return
                self.active_jobs += 1
                queue_ms = (time.perf_counter() - job.enqueued_at) * 1000
                size = await self.manager.put(job.key, job.payload, queue_ms)
                if not job.result.done():
                    job.result.set_result(size)
            except Exception as error:
                if job is not None and not job.result.done():
                    job.result.set_exception(error)
            finally:
                if job is not None:
                    self.active_jobs -= 1
                self.queue.task_done()

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle one request and close the connection."""
        _configure_client_socket(writer, self.socket_buffer_bytes)
        try:
            header, payload = await read_frame(
                reader, max_payload_bytes=self.max_request_bytes
            )
            operation = str(header.get("op", "")).upper()
            key = str(header.get("key", ""))
            if operation == "GET_BATCH":
                batch_format = header.get("batch_format")
                if batch_format == "stream_v1":
                    await self._dispatch_batch_stream(
                        writer, header.get("keys")
                    )
                elif batch_format == "blob_v1":
                    response, response_values = await self._dispatch_batch_blob(
                        header.get("keys")
                    )
                    await write_batch_blob_frame(
                        writer,
                        {"status": "ok", **response},
                        response_values,
                    )
                else:
                    # Older connectors expect one response frame per key. Keep
                    # that format available when the client did not negotiate
                    # the batch blob explicitly.
                    responses = await self._dispatch_batch(header.get("keys"))
                    buffered_bytes = 0
                    for index, (response, response_payload) in enumerate(responses):
                        await write_frame(
                            writer,
                            {"status": "ok", "index": index, **response},
                            response_payload,
                            drain=False,
                        )
                        buffered_bytes += len(response_payload)
                        if buffered_bytes >= _BATCH_DRAIN_BYTES:
                            await writer.drain()
                            buffered_bytes = 0
                    if buffered_bytes:
                        await writer.drain()
                return
            response, response_payload = await self._dispatch(
                operation, key, payload, header
            )
            await write_frame(writer, {"status": "ok", **response}, response_payload)
        except (asyncio.IncompleteReadError, ConnectionError, BrokenPipeError):
            return
        except asyncio.QueueFull:
            await write_frame(writer, {"status": "busy", "error": "manager queue full"})
        except Exception as error:
            await write_frame(
                writer,
                {"status": "error", "error": f"{type(error).__name__}: {error}"},
            )
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, BrokenPipeError):
                pass

    async def _get_batch_values(
        self, keys: Any
    ) -> tuple[list[str], list[tuple[bytes | None, dict[str, Any]]]]:
        """Load and validate a batch once for either wire response format."""
        keys = self._validate_batch_keys(keys)
        get_many = getattr(self.manager, "get_many_with_timing", None)
        if callable(get_many):
            values = await get_many(keys)
        else:
            values = await asyncio.gather(
                *(self.manager.get_with_timing(key) for key in keys)
            )
        return keys, values

    @staticmethod
    def _validate_batch_keys(keys: Any) -> list[str]:
        """Validate and normalize a GET_BATCH key list."""
        if not isinstance(keys, list) or not all(
            isinstance(key, str) for key in keys
        ):
            raise ValueError("GET_BATCH requires a list of string keys")
        if len(keys) > 4096:
            raise ValueError("GET_BATCH supports at most 4096 keys")
        return keys

    @staticmethod
    def _filter_list_keys(keys: list[str], key_hashes: Any) -> list[str]:
        """Filter diagnostic LIST output by CacheEngineKey chunk hashes."""
        if key_hashes is None:
            return keys
        if not isinstance(key_hashes, list) or not all(
            isinstance(value, str) and value for value in key_hashes
        ):
            raise ValueError("LIST key_hashes must be a list of non-empty strings")
        if len(key_hashes) > _MAX_LIST_KEY_HASHES:
            raise ValueError(
                f"LIST supports at most {_MAX_LIST_KEY_HASHES} key hashes"
            )
        expected = {value.lower() for value in key_hashes}
        matched: list[str] = []
        for key in keys:
            parts = key.rsplit("@", 2)
            if len(parts) == 3 and parts[1].lower() in expected:
                matched.append(key)
        return matched

    async def _dispatch_batch_stream(
        self, writer: asyncio.StreamWriter, keys: Any
    ) -> None:
        """Fetch and send a bounded, ordered stream of complete objects.

        Unlike ``blob_v1``, this method never calls ``get_many_with_timing``.
        A small number of single-key reads are prefetched, but the first
        response is written as soon as its own read and validation finish.
        ``writer.drain`` applies TCP backpressure before the next response is
        exposed to the client.
        """
        keys = self._validate_batch_keys(keys)
        count = len(keys)
        started = time.perf_counter()
        first_object_ms = 0.0
        send_ms = 0.0
        sent = 0
        scheduled = 0
        pending: dict[
            int, asyncio.Task[tuple[bytes | None, dict[str, Any]]]
        ] = {}

        def schedule(index: int) -> None:
            pending[index] = asyncio.create_task(
                self.manager.get_with_timing(keys[index]),
                name=f"makv-get-stream-{index}",
            )

        try:
            prefetch = min(self.batch_stream_prefetch_depth, count)
            while scheduled < prefetch:
                schedule(scheduled)
                scheduled += 1

            while sent < count:
                task = pending.pop(sent)
                value, timing = await task
                # Do not replace the completed task until it has released its
                # storage resources; this keeps active GETs <= prefetch.
                if scheduled < count:
                    schedule(scheduled)
                    scheduled += 1
                payload = value or b""
                response = {
                    "status": "ok",
                    "index": sent,
                    "count": count,
                    "batch_stream_version": BATCH_STREAM_VERSION,
                    "found": value is not None,
                    "checksum_verified": value is not None,
                    "makv_server_timing": timing,
                }
                send_started = time.perf_counter()
                await write_frame(writer, response, payload)
                send_ms += (time.perf_counter() - send_started) * 1000
                sent += 1
                if sent == 1:
                    first_object_ms = (time.perf_counter() - started) * 1000
        finally:
            for task in pending.values():
                task.cancel()
            if pending:
                await asyncio.gather(*pending.values(), return_exceptions=True)
            REMOTE_METRICS.add(
                makv_remote_get_stream_requests=1,
                makv_remote_get_stream_objects=sent,
                makv_remote_get_stream_first_object_time_ms=first_object_ms,
                makv_remote_get_stream_send_time_ms=send_ms,
                makv_remote_get_stream_total_time_ms=(
                    (time.perf_counter() - started) * 1000
                ),
            )

    async def _dispatch_batch(
        self, keys: Any
    ) -> list[tuple[dict[str, Any], bytes]]:
        """Fetch several objects using the legacy one-frame-per-key format."""
        keys, values = await self._get_batch_values(keys)
        return [
            (
                {
                    "found": value is not None,
                    "checksum_verified": value is not None,
                    "makv_server_timing": timing,
                },
                value or b"",
            )
            for value, timing in values
        ]

    async def _dispatch_batch_blob(
        self, keys: Any
    ) -> tuple[dict[str, Any], list[bytes | None]]:
        """Fetch objects for one directory plus contiguous payload blob."""
        keys, values = await self._get_batch_values(keys)
        object_values = [value for value, _ in values]
        # Timing is diagnostic only. Compact positional arrays keep the JSON
        # header bounded while preserving the fields used by TTFT reports.
        timings = (
            [
                [
                    int(bool(timing.get("hot_cache_hit", False))),
                    float(timing.get("hot_cache_ms", 0.0)),
                    float(timing.get("storage_ms", 0.0)),
                    float(timing.get("validate_ms", 0.0)),
                    float(timing.get("total_ms", 0.0)),
                    float(timing.get("batch_storage_ms", 0.0)),
                    float(timing.get("batch_validate_ms", 0.0)),
                    float(timing.get("batch_total_ms", 0.0)),
                ]
                for _, timing in values
            ]
            if len(keys) <= 256
            else []
        )
        REMOTE_METRICS.add(
            makv_remote_get_batch_blob_requests=1,
            makv_remote_get_batch_blob_bytes=batch_blob_size(object_values),
        )
        return {
            "batch_blob_version": BATCH_BLOB_VERSION,
            "count": len(keys),
            "batch_timings": timings,
        }, object_values

    async def _dispatch(
        self,
        operation: str,
        key: str,
        payload: bytes,
        request_header: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bytes]:
        request_header = request_header or {}
        if operation == "PUT":
            if not key:
                raise ValueError("PUT requires key")
            result: asyncio.Future[int] = asyncio.get_running_loop().create_future()
            await asyncio.wait_for(
                self.queue.put(PutJob(key, payload, time.perf_counter(), result)),
                timeout=self.queue_wait_timeout,
            )
            return {"stored_bytes": await result}, b""
        if operation == "GET":
            data, timing = await self.manager.get_with_timing(key)
            return {
                "found": data is not None,
                "checksum_verified": data is not None,
                "makv_server_timing": timing,
            }, data or b""
        if operation == PRECISION_RISK_OPERATION:
            if not key:
                raise ValueError("PRECISION_RISK requires key")
            try:
                signal = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("PRECISION_RISK payload must be JSON") from error
            return await self.manager.apply_precision_risk(key, signal), b""
        if operation == "EXISTS":
            return {"exists": await self.manager.storage.exists(key)}, b""
        if operation == "DELETE":
            return {"deleted": await self.manager.delete(key)}, b""
        if operation == "LIST":
            keys = await self.manager.storage.list_keys()
            return {
                "keys": self._filter_list_keys(
                    keys, request_header.get("key_hashes")
                )
            }, b""
        if operation == "HEALTH":
            health = await self.manager.health()
            return {
                **health,
                "queue_size": self.queue.qsize(),
                "queue_depth": self.queue.maxsize,
                "active_jobs": self.active_jobs,
                "workers": self.worker_count,
                "scout": (
                    self.scout_service.health()
                    if self.scout_service is not None
                    else {"enabled": False}
                ),
            }, b""
        if operation in ("SCOUT_SUBMIT", "SCOUT_WAIT"):
            if self.scout_service is None:
                raise RuntimeError("ScoutRank service is not enabled")
            protocol_version = int(request_header.get("protocol_version", -1))
            if protocol_version not in (
                SCOUT_PROTOCOL_VERSION,
                SCOUT_SUFFIX_PROTOCOL_VERSION,
            ):
                raise ValueError("unsupported ScoutRank protocol version")
            token_count = int(request_header.get("token_count", -1))
            if operation == "SCOUT_SUBMIT":
                token_sha256 = str(request_header.get("token_sha256", ""))
                if not token_sha256 or payload_sha256(payload) != token_sha256:
                    raise ValueError("ScoutRank token payload checksum mismatch")
                token_ids = decode_token_ids(payload, token_count)
                if protocol_version == SCOUT_PROTOCOL_VERSION:
                    score_start = 0
                    full_token_count = token_count
                    score_context = token_sha256
                else:
                    score_start = int(request_header.get("score_start", -1))
                    full_token_count = int(
                        request_header.get("full_token_count", -1)
                    )
                    score_end = score_start + token_count
                    declared_score_end = int(
                        request_header.get("score_end", -1)
                    )
                    if (
                        declared_score_end != score_end
                        or not 0 <= score_start <= score_end <= full_token_count
                    ):
                        raise ValueError("invalid ScoutRank suffix score range")
                    score_context = str(
                        request_header.get("score_context_sha256", "")
                    )
                    if score_context != score_context_sha256(
                        token_sha256, score_start, full_token_count
                    ):
                        raise ValueError(
                            "ScoutRank score context checksum mismatch"
                        )
                return self.scout_service.submit(
                    key,
                    token_ids,
                    token_sha256,
                    score_start=score_start,
                    full_token_count=full_token_count,
                    score_context_sha256=score_context,
                ), b""
            if payload:
                raise ValueError("SCOUT_WAIT does not accept a payload")
            requested_timeout = float(
                request_header.get("timeout_s", self.scout_max_wait_timeout)
            )
            full_token_count = int(
                request_header.get("full_token_count", token_count)
            )
            expected_score_start = None
            expected_score_end = None
            if protocol_version == SCOUT_SUFFIX_PROTOCOL_VERSION:
                expected_score_start = int(
                    request_header.get("score_start", -1)
                )
                expected_score_end = int(request_header.get("score_end", -1))
            result, timing = await self.scout_service.wait(
                key,
                full_token_count,
                min(requested_timeout, self.scout_max_wait_timeout),
                deferred=bool(request_header.get("deferred", False)),
                score_start=expected_score_start,
                score_end=expected_score_end,
            )
            return {
                "token_count": full_token_count,
                "scored_token_count": len(result.scores),
                "score_start": result.score_start,
                "score_end": result.score_end,
                "full_token_count": result.full_token_count,
                "queue_time_ms": result.queue_time_ms,
                "score_time_ms": result.score_time_ms,
                "total_job_time_ms": result.total_time_ms,
                **timing,
            }, encode_scores(result.scores)
        raise ValueError(f"unsupported MaKV operation {operation!r}")


def _parse_list(value: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    """Build the manager command-line parser."""
    parser = argparse.ArgumentParser(description="MaKV Remote Manager")
    parser.add_argument("--listen", default="0.0.0.0:65432")
    parser.add_argument("--storage-url", default=DEFAULT_MAKV_STORAGE_URL)
    parser.add_argument(
        "--storage-backend",
        choices=SUPPORTED_STORAGE_BACKENDS,
        default=None,
        help="Backend implementation; omitted means infer it from --storage-url.",
    )
    parser.add_argument(
        "--storage-namespace",
        default="lmcache:makv:",
        help="Redis key namespace; ignored by file and Mooncake adapters.",
    )
    parser.add_argument(
        "--mooncake-config",
        default=None,
        help="Mooncake JSON setup file when --storage-backend=mooncake.",
    )
    parser.add_argument("--bucket-ratios", default=None)
    parser.add_argument("--bucket-bits", default=None)
    parser.add_argument(
        "--precision-scheme",
        choices=SUPPORTED_PRECISION_SCHEMES,
        default="shared",
        help="Score-to-bucket policy used to validate incoming plans.",
    )
    parser.add_argument(
        "--scale-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument(
        "--enable-qdm",
        action="store_true",
        help=(
            "Opt in to the shadow QDM observer; it does not affect precision "
            "planning or production quantization decisions."
        ),
    )
    parser.add_argument("--qdm-block-size", type=int, default=32)
    parser.add_argument(
        "--entropy-codec",
        choices=SUPPORTED_ENTROPY_CODECS,
        default="none",
        help=(
            "Optional MaKV entropy codec; cachegen_arithmetic reuses "
            "CacheGen arithmetic-coding kernels."
        ),
    )
    parser.add_argument(
        "--entropy-backend",
        choices=SUPPORTED_ENTROPY_BACKENDS,
        default="auto",
        help="Arithmetic codec implementation selected by the manager.",
    )
    parser.add_argument(
        "--entropy-require-cuda",
        action="store_true",
        help=(
            "Fail PUT instead of using the reference codec when the CUDA "
            "arithmetic codec is unavailable."
        ),
    )
    parser.add_argument(
        "--residual-dtype",
        choices=SUPPORTED_RESIDUAL_DTYPES,
        default="none",
        help=(
            "Store original-minus-dequantized residuals for later precision "
            "upgrades; float16 is cheaper, float32 is more accurate."
        ),
    )
    parser.add_argument(
        "--risk-upgrade-threshold",
        type=float,
        default=0.8,
        help="Upgrade an object when the received risk is at least this value.",
    )
    parser.add_argument(
        "--risk-upgrade-policy",
        choices=("next", "full"),
        default="next",
        help="Promote one configured tier or directly to the highest tier.",
    )
    parser.add_argument(
        "--risk-window-tokens",
        type=int,
        default=16,
        help=(
            "Keep promoted risk tokens active for this many logical decode "
            "steps."
        ),
    )
    parser.add_argument(
        "--risk-window-ttl-s",
        type=float,
        default=0.0,
        help=(
            "Optional wall-clock expiry for a precision window; zero uses "
            "logical decode steps only."
        ),
    )
    parser.add_argument("--fallback", choices=("naive", "miss"), default="naive")
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--queue-wait-timeout", type=float, default=600.0)
    parser.add_argument(
        "--socket-buffer-bytes",
        type=int,
        default=_DEFAULT_SOCKET_BUFFER_BYTES,
        help=(
            "TCP send/receive buffer target for bulk GET responses; "
            "0 disables tuning."
        ),
    )
    parser.add_argument(
        "--batch-stream-prefetch-depth",
        type=int,
        default=_DEFAULT_BATCH_STREAM_PREFETCH_DEPTH,
        help="Maximum number of single-key GETs prefetched by stream_v1.",
    )
    parser.add_argument(
        "--memory-cache-gb",
        type=float,
        default=0.0,
        help="Bounded hot-object cache in the manager process; 0 disables it.",
    )
    parser.add_argument(
        "--trust-validated-objects",
        action="store_true",
        help=(
            "Skip repeat CRC scans for objects validated by this manager. "
            "Structural and bounds validation still runs on every GET."
        ),
    )
    parser.add_argument(
        "--allow-scoutrank-shadow-plan",
        action="store_true",
        help="Allow explicitly supplied experimental ScoutRank-v3 token plans.",
    )
    parser.add_argument(
        "--redis-socket-timeout",
        type=float,
        default=600.0,
        help="Redis read/write socket timeout in seconds for large MaKV blobs.",
    )
    parser.add_argument(
        "--max-request-bytes", type=int, default=DEFAULT_MAX_PAYLOAD_BYTES
    )
    parser.add_argument(
        "--scout-model",
        default=None,
        help="Enable overlap with this local 28-layer ScoutRank model.",
    )
    parser.add_argument("--scout-device", default="cuda:0")
    parser.add_argument(
        "--scout-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--scout-mode", choices=("fast", "balanced"), default="fast"
    )
    parser.add_argument(
        "--scout-scoring-version",
        default=_DEFAULT_LIVE_SCOUT_SCORING_VERSION,
        help=(
            "ScoutRank scoring implementation used by live overlap. The "
            "default is the exact scalar D22 fast path."
        ),
    )
    parser.add_argument("--scout-anchor-layers", default="14,28")
    parser.add_argument("--scout-observer-token-chunk-size", type=int, default=4096)
    parser.add_argument("--scout-expected-layers", type=int, default=28)
    parser.add_argument(
        "--scout-v3-num-probes",
        type=int,
        default=_DEFAULT_LIVE_SCOUT_NUM_PROBES,
        help="Total live D22 probes; mix32 requires 32.",
    )
    parser.add_argument(
        "--scout-v3-tail-probes",
        type=int,
        default=_DEFAULT_LIVE_SCOUT_TAIL_PROBES,
        help="Tail probes reserved by the live D22 policy; mix32 requires 16.",
    )
    parser.add_argument(
        "--scout-v3-probe-selection",
        choices=("priority", "mix32"),
        default=_DEFAULT_LIVE_SCOUT_PROBE_SELECTION,
        help="Live D22 probe policy; mix32 uses 16 prompt-wide and 16 tail probes.",
    )
    parser.add_argument(
        "--scout-context-policy",
        choices=("sliding_window", "error"),
        default="sliding_window",
        help=(
            "How to handle prompts longer than the Scout model context: "
            "bounded overlapping windows or an explicit error."
        ),
    )
    parser.add_argument(
        "--scout-context-window",
        type=int,
        default=None,
        help="Optional safe Scout window size; defaults to model config limit.",
    )
    parser.add_argument(
        "--scout-context-left-tokens",
        type=int,
        default=None,
        help="History tokens retained before each long-prompt score region.",
    )
    parser.add_argument(
        "--scout-context-right-tokens",
        type=int,
        default=None,
        help="Future tokens retained for probes after each score region.",
    )
    parser.add_argument("--scout-queue-depth", type=int, default=64)
    parser.add_argument("--scout-result-ttl-s", type=float, default=600.0)
    parser.add_argument(
        "--scout-prompt-cache-capacity",
        type=int,
        default=128,
        help=(
            "Maximum number of completed token-prompt hash entries retained "
            "for cross-request reuse."
        ),
    )
    parser.add_argument("--scout-max-wait-timeout", type=float, default=600.0)
    return parser


async def run_server(args: argparse.Namespace) -> None:
    """Run the manager until cancelled."""
    if args.bucket_ratios is None:
        default_ratios = ",".join(
            str(value) for value in default_makv_bucket_ratios(args.precision_scheme)
        )
    else:
        default_ratios = args.bucket_ratios
    ratios = _parse_list(default_ratios, float)
    default_bits = (
        "8,4,2"
        if args.precision_scheme == "kv_separate_3tier"
        else "16,8,4,2"
        if args.precision_scheme == "kv_separate_4tier"
        else "16,8,4"
    )
    bits = _parse_list(args.bucket_bits or default_bits, int)
    if (
        len(ratios) != len(bits)
        or any(value < 0.0 for value in ratios)
        or abs(sum(ratios) - 1.0) > 1e-6
        or any(value not in SUPPORTED_BUCKET_BITS for value in bits)
        or len(set(bits)) != len(bits)
    ):
        raise ValueError("bucket ratios/bits are invalid")
    if args.precision_scheme == "kv_separate_3tier" and bits != (8, 4, 2):
        raise ValueError(
            "--precision-scheme=kv_separate_3tier requires --bucket-bits 8,4,2"
        )
    if args.precision_scheme == "kv_separate_4tier" and bits != (16, 8, 4, 2):
        raise ValueError(
            "--precision-scheme=kv_separate_4tier requires "
            "--bucket-bits 16,8,4,2"
        )
    if args.memory_cache_gb < 0:
        raise ValueError("--memory-cache-gb must be non-negative")
    if not math.isfinite(args.risk_upgrade_threshold) or not 0.0 <= (
        args.risk_upgrade_threshold
    ) <= 1.0:
        raise ValueError("--risk-upgrade-threshold must be in [0, 1]")
    if args.risk_window_tokens <= 0:
        raise ValueError("--risk-window-tokens must be positive")
    if not math.isfinite(args.risk_window_ttl_s) or args.risk_window_ttl_s < 0.0:
        raise ValueError(
            "--risk-window-ttl-s must be finite and non-negative"
        )
    if args.redis_socket_timeout <= 0:
        raise ValueError("--redis-socket-timeout must be positive")
    if args.enable_qdm and args.qdm_block_size <= 0:
        raise ValueError("--qdm-block-size must be positive")
    config = MaKVConfig(
        storage_url=args.storage_url,
        bucket_ratios=ratios,
        bucket_bits=bits,
        importance_layout="token",
        quant_granularity="per_token_head",
        scale_dtype=args.scale_dtype,
        protect_prefix_tokens=0,
        protect_tail_tokens=0,
        dequant_backend="cuda",
        require_cuda_dequant=True,
        fallback=args.fallback,
        enable_checksum=True,
        allow_scoutrank_shadow_plan=args.allow_scoutrank_shadow_plan,
        storage_backend=args.storage_backend
        or infer_storage_backend(args.storage_url),
        storage_namespace=args.storage_namespace,
        mooncake_config_path=args.mooncake_config,
        precision_scheme=args.precision_scheme,
        enable_qdm=args.enable_qdm,
        qdm_block_size=args.qdm_block_size,
        entropy_codec=args.entropy_codec,
        entropy_backend=args.entropy_backend,
        entropy_require_cuda=args.entropy_require_cuda,
        residual_dtype=args.residual_dtype,
        risk_upgrade_threshold=args.risk_upgrade_threshold,
        risk_upgrade_policy=args.risk_upgrade_policy,
        risk_window_tokens=args.risk_window_tokens,
        risk_window_ttl_s=args.risk_window_ttl_s,
    )
    host, port_text = args.listen.rsplit(":", 1)
    manager = MaKVRemoteManager(
        config,
        create_storage_adapter(
            args.storage_url,
            backend=args.storage_backend,
            namespace=args.storage_namespace,
            redis_socket_timeout=args.redis_socket_timeout,
            mooncake_config_path=args.mooncake_config,
        ),
        memory_cache_bytes=int(args.memory_cache_gb * 1024**3),
        trust_validated_objects=args.trust_validated_objects,
    )
    scout_service = None
    if args.scout_model:
        if (
            args.scout_observer_token_chunk_size <= 0
            or args.scout_expected_layers <= 0
        ):
            raise ValueError(
                "ScoutRank layer count and observer chunk must be positive"
            )
        # Keep transformers and CUDA initialization optional for CPU-only
        # LMCache deployments that do not enable overlap.
        from makv_scoutrank.runtime import ScoutRankRuntime

        scout_runtime = ScoutRankRuntime(
            args.scout_model,
            device=args.scout_device,
            dtype=args.scout_dtype,
            mode=args.scout_mode,
            anchor_layers=_parse_list(args.scout_anchor_layers, int),
            observer_token_chunk_size=args.scout_observer_token_chunk_size,
            expected_layers=args.scout_expected_layers,
            scoring_version=args.scout_scoring_version,
            v3_num_probes=args.scout_v3_num_probes,
            v3_tail_probes=args.scout_v3_tail_probes,
            v3_probe_selection=args.scout_v3_probe_selection,
            context_policy=args.scout_context_policy,
            context_window=args.scout_context_window,
            context_left_tokens=args.scout_context_left_tokens,
            context_right_tokens=args.scout_context_right_tokens,
        )
        scout_service = ScoutJobService(
            scout_runtime,
            max_pending_jobs=args.scout_queue_depth,
            result_ttl_s=args.scout_result_ttl_s,
            max_cached_prompts=args.scout_prompt_cache_capacity,
        )
    service = MaKVRemoteServer(
        manager,
        queue_depth=args.queue_depth,
        workers=args.workers,
        max_request_bytes=args.max_request_bytes,
        queue_wait_timeout=args.queue_wait_timeout,
        socket_buffer_bytes=args.socket_buffer_bytes,
        batch_stream_prefetch_depth=args.batch_stream_prefetch_depth,
        scout_service=scout_service,
        scout_max_wait_timeout=args.scout_max_wait_timeout,
    )
    try:
        await service.start_workers()
        server = await asyncio.start_server(
            service.handle_client, host, int(port_text)
        )
        async with server:
            await server.serve_forever()
    finally:
        await service.close()
        await manager.close()


def main() -> None:
    """Run the MaKV Remote Manager CLI."""
    try:
        asyncio.run(run_server(build_parser().parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
