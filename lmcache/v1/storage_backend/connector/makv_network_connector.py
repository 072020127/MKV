# SPDX-License-Identifier: Apache-2.0

"""Network client connector for the independent MaKV Remote Manager."""

from __future__ import annotations


# Standard
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Mapping, Optional
from urllib.parse import urlparse
import asyncio
import json
import socket
import time

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import BytesBufferMemoryObj, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.makv.metrics import CLIENT_METRICS
from lmcache.v1.storage_backend.makv_remote.protocol import (
    BATCH_BLOB_ENTRY,
    BATCH_BLOB_HEADER,
    BATCH_BLOB_VERSION,
    BATCH_STREAM_VERSION,
    DEFAULT_MAX_PAYLOAD_BYTES,
    FRAME_HEADER,
    MAX_HEADER_BYTES,
    PRECISION_RISK_OPERATION,
    decode_batch_blob,
    decode_batch_blob_directory,
    read_frame,
    write_frame,
)

_MAX_BATCH_KEYS = 256
_DEFAULT_SOCKET_BUFFER_BYTES = 16 * 1024 * 1024
_DEFAULT_STREAM_RECEIVE_PREFETCH_DEPTH = 2


@dataclass(frozen=True)
class _StreamReceiveFailure:
    """Transport error forwarded from a background stream receiver."""

    error: Exception


_STREAM_RECEIVE_DONE = object()


def _decode_batch_timing(value: Any) -> dict[str, Any] | None:
    """Expand the compact server timing tuple carried by batch blob v1."""
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 8:
        raise RuntimeError("invalid MaKV batch timing entry")
    try:
        return {
            "hot_cache_hit": bool(int(value[0])),
            "hot_cache_ms": float(value[1]),
            "storage_ms": float(value[2]),
            "validate_ms": float(value[3]),
            "total_ms": float(value[4]),
            "batch_storage_ms": float(value[5]),
            "batch_validate_ms": float(value[6]),
            "batch_total_ms": float(value[7]),
        }
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("invalid MaKV batch timing value") from error


def _configure_socket(
    sock: Any,
    *,
    buffer_bytes: int,
    tcp_nodelay: bool,
) -> None:
    """Apply bulk-transfer socket options when the platform exposes them."""
    if sock is None:
        return
    try:
        if buffer_bytes > 0:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_bytes)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_bytes)
        if tcp_nodelay and hasattr(socket, "TCP_NODELAY"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        # Socket buffer limits are controlled by the host kernel. The protocol
        # remains correct when a platform rejects an optional tuning knob.
        return


class MaKVNetworkConnector(RemoteConnector):
    """Send MaKV operations to a separate TCP manager process."""

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: Any,
        config: Optional[LMCacheEngineConfig],
        metadata: Optional[LMCacheMetadata],
    ) -> None:
        del local_cpu_backend
        assert config is not None
        assert metadata is not None
        super().__init__(config, metadata)
        parsed = urlparse(url)
        if parsed.hostname is None or parsed.port is None:
            raise ValueError("makv:// URL must include host and port")
        self.host = parsed.hostname
        self.port = parsed.port
        self.timeout = float(
            (config.extra_config or {}).get("makv_network_timeout_s", 30)
        )
        self.socket_buffer_bytes = max(
            0,
            int(
                (config.extra_config or {}).get(
                    "makv_socket_buffer_bytes", _DEFAULT_SOCKET_BUFFER_BYTES
                )
            ),
        )
        tcp_nodelay = (config.extra_config or {}).get("makv_tcp_nodelay", True)
        if isinstance(tcp_nodelay, str):
            tcp_nodelay = tcp_nodelay.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.tcp_nodelay = bool(tcp_nodelay)
        batch_blob = (config.extra_config or {}).get("makv_batch_blob", True)
        if isinstance(batch_blob, str):
            batch_blob = batch_blob.strip().lower() in {"1", "true", "yes", "on"}
        self.batch_blob = bool(batch_blob)
        self.pinned_receive = bool(
            (config.extra_config or {}).get("makv_pinned_receive", True)
        )
        self.pinned_receive_min_bytes = max(
            0,
            int(
                (config.extra_config or {}).get(
                    "makv_pinned_receive_min_bytes", 1 << 20
                )
            ),
        )
        self.stream_receive_prefetch_depth = max(
            1,
            int(
                (config.extra_config or {}).get(
                    "makv_stream_receive_prefetch_depth",
                    _DEFAULT_STREAM_RECEIVE_PREFETCH_DEPTH,
                )
            ),
        )
        self.loop = loop
        self.put_concurrency = max(
            1, int((config.extra_config or {}).get("makv_put_concurrency", 4))
        )
        self._put_semaphore = asyncio.Semaphore(self.put_concurrency)

    async def _request(
        self, operation: str, key: str = "", payload: bytes = b""
    ) -> tuple[dict[str, Any], bytes]:
        header, response_payload, timing = await self._request_with_timing(
            operation, key, payload
        )
        if operation == "PUT":
            CLIENT_METRICS.add(
                makv_client_put_connect_time_ms=timing["connect_ms"],
                makv_client_put_send_time_ms=timing["send_ms"],
                makv_client_put_response_time_ms=timing["receive_ms"],
                makv_client_put_total_time_ms=timing["total_ms"],
            )
        return header, response_payload

    async def _request_with_timing(
        self, operation: str, key: str = "", payload: bytes = b""
    ) -> tuple[dict[str, Any], bytes, dict[str, float]]:
        started = time.perf_counter()
        connect_started = started
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port), self.timeout
        )
        _configure_socket(
            writer.get_extra_info("socket"),
            buffer_bytes=self.socket_buffer_bytes,
            tcp_nodelay=self.tcp_nodelay,
        )
        connect_ms = (time.perf_counter() - connect_started) * 1000
        try:
            send_started = time.perf_counter()
            await write_frame(writer, {"op": operation, "key": key}, payload)
            send_ms = (time.perf_counter() - send_started) * 1000
            receive_started = time.perf_counter()
            header, response_payload = await asyncio.wait_for(
                read_frame(reader), self.timeout
            )
            receive_ms = (time.perf_counter() - receive_started) * 1000
            if header.get("status") != "ok":
                raise RuntimeError(
                    str(header.get("error", "MaKV manager request failed"))
                )
            return header, response_payload, {
                "connect_ms": connect_ms,
                "send_ms": send_ms,
                "first_response_ms": receive_ms,
                "receive_ms": receive_ms,
                "total_ms": (time.perf_counter() - started) * 1000,
            }
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, BrokenPipeError):
                pass

    async def exists(self, key: Any) -> bool:
        """Check the key through the remote manager."""
        header, _ = await self._request("EXISTS", key.to_string())
        return bool(header["exists"])

    def exists_sync(self, key: Any) -> bool:
        """Synchronously check the key through the remote manager."""
        header, _ = self._request_sync("EXISTS", key.to_string())
        return bool(header["exists"])

    async def get(self, key: Any) -> Optional[MemoryObj]:
        """Download the stored quantized object without remote dequantization."""
        header, payload, timing = await self._request_with_timing(
            "GET", key.to_string()
        )
        CLIENT_METRICS.add(
            makv_client_get_batches=1,
            makv_client_get_objects=1,
            makv_client_get_connect_time_ms=timing["connect_ms"],
            makv_client_get_send_time_ms=timing["send_ms"],
            makv_client_get_first_response_time_ms=timing["first_response_ms"],
            makv_client_get_receive_time_ms=timing["receive_ms"],
            makv_client_get_total_time_ms=timing["total_ms"],
        )
        return self._memory_obj_from_response(
            header, payload, transport_timing=timing
        )

    @staticmethod
    def _memory_obj_from_response(
        header: dict[str, Any],
        payload: bytes | bytearray | memoryview,
        *,
        transport_timing: dict[str, float] | None = None,
    ) -> Optional[MemoryObj]:
        if not bool(header.get("found", False)):
            return None
        CLIENT_METRICS.add(makv_get_quantized_bytes=len(payload))
        memory_obj = BytesBufferMemoryObj(payload)
        # The independent manager validates the object before sending it. The
        # deserializer can therefore skip a second full-object CRC pass.
        memory_obj.makv_checksum_verified = bool(
            header.get("checksum_verified", False)
        )
        server_timing = header.get("makv_server_timing")
        if isinstance(server_timing, dict):
            memory_obj.makv_server_timing = dict(server_timing)
        if transport_timing is not None:
            memory_obj.makv_transport_timing = dict(transport_timing)
        return memory_obj

    @classmethod
    def _append_legacy_batch_result(
        cls,
        results: list[Optional[MemoryObj]],
        header: dict[str, Any],
        payload: bytes | bytearray | memoryview,
        *,
        expected_index: int,
    ) -> None:
        """Parse one response from the pre-MKVB streaming batch protocol."""
        if header.get("status") != "ok":
            raise RuntimeError(
                str(header.get("error", "MaKV batch GET failed"))
            )
        if int(header.get("index", -1)) != expected_index:
            raise RuntimeError("MaKV batch GET response order mismatch")
        results.append(cls._memory_obj_from_response(header, payload))

    async def put(self, key: Any, memory_obj: MemoryObj) -> None:
        """Upload the client raw-KV envelope to the manager."""
        async with self._put_semaphore:
            await self._request("PUT", key.to_string(), bytes(memory_obj.byte_array))

    async def report_precision_risk(
        self, key: Any, signal: Mapping[str, Any] | Any
    ) -> dict[str, Any]:
        """Send one output-side risk signal to the remote precision policy."""
        if hasattr(signal, "as_dict"):
            signal = signal.as_dict()
        if not isinstance(signal, Mapping):
            raise TypeError("MaKV precision risk signal must be a mapping")
        payload = json.dumps(
            dict(signal), separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        header, _ = await self._request(
            PRECISION_RISK_OPERATION, key.to_string(), payload
        )
        return header

    async def list(self) -> List[str]:
        """List manager storage object digests."""
        header, _ = await self._request("LIST")
        return [str(item) for item in header["keys"]]

    async def health(self) -> dict[str, Any]:
        """Return manager process identity and metrics."""
        header, _ = await self._request("HEALTH")
        return header

    def support_ping(self) -> bool:
        """Return that manager health checks are supported."""
        return True

    async def ping(self) -> int:
        """Return zero when the manager responds to HEALTH."""
        await self.health()
        return 0

    def remove_sync(self, key: Any) -> bool:
        """Synchronously delete one remote object."""
        header, _ = self._request_sync("DELETE", key.to_string())
        return bool(header["deleted"])

    async def close(self) -> None:
        """Close the stateless connector."""

    def support_batched_put(self) -> bool:
        return True

    def support_batched_contains(self) -> bool:
        return True

    def batched_contains(self, keys: list[Any]) -> int:
        async def _contains() -> list[bool]:
            return await asyncio.gather(*(self.exists(key) for key in keys))

        future = asyncio.run_coroutine_threadsafe(_contains(), self.loop)
        results = future.result(self.timeout)
        for index, found in enumerate(results):
            if not found:
                return index
        return len(results)

    async def batched_put(self, keys: list[Any], memory_objs: list[MemoryObj]) -> None:
        await asyncio.gather(
            *(
                self.put(key, memory_obj)
                for key, memory_obj in zip(keys, memory_objs, strict=True)
            )
        )

    def support_batched_get(self) -> bool:
        return True

    def support_batched_get_streaming(self) -> bool:
        """Return whether GET_BATCH can yield each object before the batch ends."""
        return True

    async def batched_get_streaming(
        self, keys: list[Any]
    ) -> AsyncIterator[tuple[int, Optional[MemoryObj]]]:
        """Yield GET_BATCH values in key order as soon as each frame arrives.

        New managers use ``stream_v1`` with bounded server-side prefetch. The
        client also accepts the older MKVB and per-frame response formats so a
        rolling manager upgrade does not turn a hit into a cache miss.
        """
        for start in range(0, len(keys), _MAX_BATCH_KEYS):
            async for index, value in self._batched_get_chunk_streaming(
                keys[start : start + _MAX_BATCH_KEYS]
            ):
                yield start + index, value

    async def _batched_get_chunk_streaming(
        self, keys: list[Any]
    ) -> AsyncIterator[tuple[int, Optional[MemoryObj]]]:
        """Read one GET_BATCH response one independently validated object at a time."""
        started = time.perf_counter()
        sock, connect_ms = await self._open_batch_socket()
        receive_started: float | None = None
        send_ms = 0.0
        first_response_ms = 0.0
        first_result: Optional[MemoryObj] = None
        completed = False
        try:
            request_header = {
                "op": "GET_BATCH",
                "keys": [key.to_string() for key in keys],
                "batch_format": "stream_v1",
            }
            send_started = time.perf_counter()
            await asyncio.wait_for(
                self._send_socket_frame(sock, request_header), self.timeout
            )
            send_ms = (time.perf_counter() - send_started) * 1000
            receive_started = time.perf_counter()

            first_header, payload_length = await asyncio.wait_for(
                self._read_socket_frame_header(sock), self.timeout
            )
            first_response_ms = (time.perf_counter() - receive_started) * 1000
            if first_header.get("status") != "ok":
                raise RuntimeError(
                    str(first_header.get("error", "MaKV batch GET failed"))
                )

            if first_header.get("batch_stream_version") == BATCH_STREAM_VERSION:
                if int(first_header.get("count", -1)) != len(keys):
                    raise RuntimeError("MaKV batch stream response count mismatch")
                async for index, value in self._consume_stream_v1_prefetched(
                    sock,
                    keys,
                    first_header=first_header,
                    first_payload_length=payload_length,
                ):
                    if value is not None and first_result is None:
                        first_result = value
                    yield index, value
            elif first_header.get("batch_blob_version") == BATCH_BLOB_VERSION:
                if int(first_header.get("count", -1)) != len(keys):
                    raise RuntimeError("MaKV batch blob response count mismatch")
                if payload_length < BATCH_BLOB_HEADER.size:
                    raise RuntimeError("truncated MaKV batch blob header")
                prefix = await self._recv_into(sock, BATCH_BLOB_HEADER.size)
                _, _, count, _ = BATCH_BLOB_HEADER.unpack_from(prefix, 0)
                directory_size = BATCH_BLOB_HEADER.size + count * BATCH_BLOB_ENTRY.size
                if directory_size > payload_length:
                    raise RuntimeError("MaKV batch blob directory is out of bounds")
                directory = bytearray(directory_size)
                directory[: BATCH_BLOB_HEADER.size] = prefix
                if directory_size > BATCH_BLOB_HEADER.size:
                    directory[BATCH_BLOB_HEADER.size :] = await self._recv_into(
                        sock, directory_size - BATCH_BLOB_HEADER.size
                    )
                entries = decode_batch_blob_directory(
                    directory,
                    payload_length=payload_length,
                    expected_count=len(keys),
                )
                timings = first_header.get("batch_timings", [])
                if not isinstance(timings, list) or len(timings) not in (0, len(keys)):
                    raise RuntimeError("MaKV batch blob timing count mismatch")

                cursor = directory_size
                for index, entry in enumerate(entries):
                    timing = _decode_batch_timing(
                        timings[index] if timings else None
                    )
                    if entry is None:
                        yield index, None
                        continue
                    offset, length = entry
                    if offset < cursor:
                        raise RuntimeError(
                            "MaKV batch blob entries are not wire ordered"
                        )
                    await self._discard_socket_bytes(sock, offset - cursor)
                    payload = await self._recv_into(
                        sock,
                        length,
                        pinned=(
                            self.pinned_receive
                            and length >= self.pinned_receive_min_bytes
                        ),
                    )
                    cursor = offset + length
                    object_header: dict[str, Any] = {
                        "found": True,
                        "checksum_verified": True,
                    }
                    if timing is not None:
                        object_header["makv_server_timing"] = timing
                    memory_obj = self._memory_obj_from_response(object_header, payload)
                    assert memory_obj is not None
                    if first_result is None:
                        first_result = memory_obj
                    yield index, memory_obj
                await self._discard_socket_bytes(sock, payload_length - cursor)
                CLIENT_METRICS.add(
                    makv_client_get_batch_blob_frames=1,
                    makv_client_get_batch_blob_bytes=payload_length,
                )
            else:
                # Legacy managers send independent frames.  The first frame
                # header has already been consumed, so stream its payload and
                # then retain the old ordering checks for the remaining keys.
                first_payload = await self._recv_into(sock, payload_length)
                legacy_results: list[Optional[MemoryObj]] = []
                self._append_legacy_batch_result(
                    legacy_results, first_header, first_payload, expected_index=0
                )
                first = legacy_results[0]
                if first is not None:
                    first_result = first
                yield 0, first
                for index in range(1, len(keys)):
                    header, payload = await asyncio.wait_for(
                        self._read_socket_frame(sock), self.timeout
                    )
                    legacy_results = []
                    self._append_legacy_batch_result(
                        legacy_results, header, payload, expected_index=index
                    )
                    value = legacy_results[0]
                    yield index, value
            completed = True
        finally:
            now = time.perf_counter()
            receive_ms = (
                (now - receive_started) * 1000 if receive_started is not None else 0.0
            )
            timing = {
                "connect_ms": connect_ms,
                "send_ms": send_ms,
                "first_response_ms": first_response_ms,
                "receive_ms": receive_ms,
                "total_ms": (now - started) * 1000,
            }
            if first_result is not None:
                # RemoteBackend records the batch timing once after its
                # synchronous iterator is exhausted.  The result object stays
                # alive through that point even though it was yielded earlier.
                first_result.makv_transport_timing = timing
            if completed:
                CLIENT_METRICS.add(
                    makv_client_get_batches=1,
                    makv_client_get_objects=len(keys),
                    makv_client_get_connect_time_ms=timing["connect_ms"],
                    makv_client_get_send_time_ms=timing["send_ms"],
                    makv_client_get_first_response_time_ms=timing[
                        "first_response_ms"
                    ],
                    makv_client_get_receive_time_ms=timing["receive_ms"],
                    makv_client_get_total_time_ms=timing["total_ms"],
                )
            sock.close()

    async def _consume_stream_v1_prefetched(
        self,
        sock: socket.socket,
        keys: list[Any],
        *,
        first_header: dict[str, Any],
        first_payload_length: int,
    ) -> AsyncIterator[tuple[int, Optional[MemoryObj]]]:
        """Consume complete stream_v1 objects through a bounded receive queue.

        ``Queue.put`` blocks before another socket read when the consumer falls
        behind. This intentionally lets the server's existing ``writer.drain``
        observe TCP backpressure instead of accumulating whole MaKV objects in
        Python memory.
        """
        queue: asyncio.Queue[
            tuple[int, Optional[MemoryObj]] | _StreamReceiveFailure | object
        ] = asyncio.Queue(maxsize=self.stream_receive_prefetch_depth)
        producer = asyncio.create_task(
            self._receive_stream_v1_objects(
                sock,
                keys,
                first_header=first_header,
                first_payload_length=first_payload_length,
                queue=queue,
            ),
            name="makv-stream-receiver",
        )
        try:
            while True:
                item = await queue.get()
                if item is _STREAM_RECEIVE_DONE:
                    await producer
                    return
                if isinstance(item, _StreamReceiveFailure):
                    await producer
                    raise item.error
                yield item
        finally:
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    async def _receive_stream_v1_objects(
        self,
        sock: socket.socket,
        keys: list[Any],
        *,
        first_header: dict[str, Any],
        first_payload_length: int,
        queue: asyncio.Queue[
            tuple[int, Optional[MemoryObj]] | _StreamReceiveFailure | object
        ],
    ) -> None:
        """Read independently framed objects until the bounded queue is full."""
        stream_frame_count = 0
        stream_bytes = 0
        queue_peak = 0
        backpressure_waits = 0

        async def enqueue(
            index: int, header: dict[str, Any], payload_length: int
        ) -> None:
            nonlocal stream_frame_count, stream_bytes, queue_peak, backpressure_waits
            if (
                header.get("batch_stream_version") != BATCH_STREAM_VERSION
                or int(header.get("count", -1)) != len(keys)
            ):
                raise RuntimeError("MaKV batch stream response metadata mismatch")
            payload = await asyncio.wait_for(
                self._recv_into(sock, payload_length), self.timeout
            )
            results: list[Optional[MemoryObj]] = []
            self._append_legacy_batch_result(
                results, header, payload, expected_index=index
            )
            if queue.full():
                backpressure_waits += 1
            await queue.put((index, results[0]))
            queue_peak = max(queue_peak, queue.qsize())
            stream_frame_count += 1
            stream_bytes += len(payload)

        try:
            await enqueue(0, first_header, first_payload_length)
            for index in range(1, len(keys)):
                header, payload_length = await asyncio.wait_for(
                    self._read_socket_frame_header(sock), self.timeout
                )
                await enqueue(index, header, payload_length)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await queue.put(_StreamReceiveFailure(error))
            return

        CLIENT_METRICS.add(
            makv_client_get_stream_requests=1,
            makv_client_get_stream_frames=stream_frame_count,
            makv_client_get_stream_bytes=stream_bytes,
            makv_client_get_stream_prefetch_peak=queue_peak,
            makv_client_get_stream_prefetch_backpressure_waits=backpressure_waits,
        )
        await queue.put(_STREAM_RECEIVE_DONE)

    async def batched_get(self, keys: list[Any]) -> list[Optional[MemoryObj]]:
        if not keys:
            return []
        results: list[Optional[MemoryObj]] = []
        # Keep the JSON request header comfortably below the protocol limit for
        # long prompts with many LMCache chunks.
        for start in range(0, len(keys), _MAX_BATCH_KEYS):
            results.extend(
                await self._batched_get_chunk(
                    keys[start : start + _MAX_BATCH_KEYS]
                )
            )
        return results

    async def _batched_get_chunk(
        self, keys: list[Any]
    ) -> list[Optional[MemoryObj]]:
        started = time.perf_counter()
        sock, connect_ms = await self._open_batch_socket()
        try:
            request_header = {
                "op": "GET_BATCH",
                "keys": [key.to_string() for key in keys],
            }
            if self.batch_blob:
                request_header["batch_format"] = "blob_v1"
            send_started = time.perf_counter()
            await asyncio.wait_for(
                self._send_socket_frame(sock, request_header), self.timeout
            )
            send_ms = (time.perf_counter() - send_started) * 1000
            receive_started = time.perf_counter()
            results: list[Optional[MemoryObj]] = []
            first_response_ms: float | None = None
            first_header, first_payload = await asyncio.wait_for(
                self._read_socket_frame(sock), self.timeout
            )
            if first_response_ms is None:
                first_response_ms = (time.perf_counter() - receive_started) * 1000
            if first_header.get("status") != "ok":
                raise RuntimeError(
                    str(first_header.get("error", "MaKV batch GET failed"))
                )
            if first_header.get("batch_blob_version") == BATCH_BLOB_VERSION:
                if int(first_header.get("count", -1)) != len(keys):
                    raise RuntimeError("MaKV batch blob response count mismatch")
                payloads = decode_batch_blob(
                    first_payload, expected_count=len(keys)
                )
                timings = first_header.get("batch_timings", [])
                if not isinstance(timings, list) or len(timings) not in (0, len(keys)):
                    raise RuntimeError("MaKV batch blob timing count mismatch")
                for index, payload in enumerate(payloads):
                    timing = _decode_batch_timing(
                        timings[index] if timings else None
                    )
                    object_header = {
                        "found": payload is not None,
                        "checksum_verified": payload is not None,
                    }
                    if timing is not None:
                        object_header["makv_server_timing"] = timing
                    results.append(
                        self._memory_obj_from_response(
                            object_header,
                            payload if payload is not None else b"",
                        )
                    )
                CLIENT_METRICS.add(
                    makv_client_get_batch_blob_frames=1,
                    makv_client_get_batch_blob_bytes=len(first_payload),
                )
            else:
                # Compatibility with managers predating batch blob v1. The
                # first response has already been consumed; read the rest.
                self._append_legacy_batch_result(
                    results, first_header, first_payload, expected_index=0
                )
                for index in range(1, len(keys)):
                    header, payload = await asyncio.wait_for(
                        self._read_socket_frame(sock), self.timeout
                    )
                    self._append_legacy_batch_result(
                        results, header, payload, expected_index=index
                    )
            timing = {
                "connect_ms": connect_ms,
                "send_ms": send_ms,
                "first_response_ms": first_response_ms or 0.0,
                "receive_ms": (time.perf_counter() - receive_started) * 1000,
                "total_ms": (time.perf_counter() - started) * 1000,
            }
            CLIENT_METRICS.add(
                makv_client_get_batches=1,
                makv_client_get_objects=len(keys),
                makv_client_get_connect_time_ms=timing["connect_ms"],
                makv_client_get_send_time_ms=timing["send_ms"],
                makv_client_get_first_response_time_ms=timing["first_response_ms"],
                makv_client_get_receive_time_ms=timing["receive_ms"],
                makv_client_get_total_time_ms=timing["total_ms"],
            )
            # The TCP batch timing belongs to the batch, not every object. Put
            # it on the first returned object so request-level aggregation does
            # not over-count it by the number of LMCache chunks.
            for result in results:
                if result is not None:
                    result.makv_transport_timing = timing
                    break
            return results
        finally:
            sock.close()

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def batched_get_non_blocking(
        self, lookup_id: str, keys: list[Any]
    ) -> list[MemoryObj]:
        del lookup_id
        values = await self.batched_get(keys)
        result: list[MemoryObj] = []
        for value in values:
            if value is None:
                break
            result.append(value)
        return result

    def _request_sync(
        self, operation: str, key: str = ""
    ) -> tuple[dict[str, Any], bytes]:
        header_bytes = json.dumps(
            {"op": operation, "key": key}, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        with socket.create_connection(
            (self.host, self.port), self.timeout
        ) as connection:
            connection.sendall(FRAME_HEADER.pack(len(header_bytes), 0) + header_bytes)
            frame = self._recv_exact(connection, FRAME_HEADER.size)
            response_header_len, response_payload_len = FRAME_HEADER.unpack(frame)
            response_header = json.loads(
                self._recv_exact(connection, response_header_len).decode("utf-8")
            )
            payload = self._recv_exact(connection, response_payload_len)
        if response_header.get("status") != "ok":
            raise RuntimeError(str(response_header.get("error", "request failed")))
        return response_header, payload

    async def _send_socket_frame(
        self, sock: socket.socket, header: dict[str, Any]
    ) -> None:
        """Send a header-only request without creating an asyncio stream."""
        header_bytes = json.dumps(
            header, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        if len(header_bytes) > MAX_HEADER_BYTES:
            raise ValueError("MaKV frame header exceeds configured limit")
        frame = FRAME_HEADER.pack(len(header_bytes), 0) + header_bytes
        await self.loop.sock_sendall(sock, frame)

    async def _open_batch_socket(self) -> tuple[socket.socket, float]:
        """Resolve and connect one nonblocking socket for a batch GET."""
        started = time.perf_counter()
        addresses = await asyncio.wait_for(
            self.loop.getaddrinfo(
                self.host,
                self.port,
                type=socket.SOCK_STREAM,
            ),
            self.timeout,
        )
        last_error: OSError | None = None
        for family, socktype, proto, _, address in addresses:
            sock = socket.socket(family, socktype, proto)
            sock.setblocking(False)
            _configure_socket(
                sock,
                buffer_bytes=self.socket_buffer_bytes,
                tcp_nodelay=self.tcp_nodelay,
            )
            try:
                await asyncio.wait_for(
                    self.loop.sock_connect(sock, address), self.timeout
                )
                return sock, (time.perf_counter() - started) * 1000
            except (OSError, asyncio.TimeoutError) as error:
                last_error = error if isinstance(error, OSError) else OSError(error)
                sock.close()
        if last_error is not None:
            raise last_error
        raise OSError(f"unable to resolve MaKV manager address {self.host!r}")

    async def _read_socket_frame(
        self, sock: socket.socket
    ) -> tuple[dict[str, Any], bytes | bytearray]:
        """Read one frame directly into its final payload bytearray.

        The stream-based protocol helper is retained for ordinary requests.
        Batch GETs are large and benefit from avoiding the intermediate
        StreamReader buffer and its second payload copy.
        """
        header, payload_length = await self._read_socket_frame_header(sock)
        payload = await self._recv_into(
            sock,
            payload_length,
            pinned=(
                self.pinned_receive
                and payload_length >= self.pinned_receive_min_bytes
            ),
        )
        return header, payload

    async def _read_socket_frame_header(
        self, sock: socket.socket
    ) -> tuple[dict[str, Any], int]:
        """Read the frame prefix while leaving its payload on the socket."""
        raw_header = await self._recv_into(sock, FRAME_HEADER.size)
        header_length, payload_length = FRAME_HEADER.unpack(raw_header)
        if header_length <= 0 or header_length > MAX_HEADER_BYTES:
            raise ValueError("invalid MaKV frame header length")
        if payload_length > DEFAULT_MAX_PAYLOAD_BYTES:
            raise ValueError("MaKV frame payload exceeds configured limit")
        header_bytes = await self._recv_into(sock, header_length)
        header = json.loads(bytes(header_bytes).decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError("MaKV frame header must be an object")
        return header, payload_length

    async def _discard_socket_bytes(self, sock: socket.socket, size: int) -> None:
        """Consume alignment padding without retaining a batch-sized buffer."""
        while size > 0:
            chunk_size = min(size, 64 * 1024)
            await self._recv_into(sock, chunk_size)
            size -= chunk_size

    async def _recv_into(
        self, sock: socket.socket, size: int, *, pinned: bool = False
    ) -> bytes | bytearray | memoryview:
        """Receive exactly ``size`` bytes into one host-side buffer.

        Large batch payloads use a pinned PyTorch byte tensor when available,
        so the subsequent async H2D copy does not first copy the entire blob
        into another pinned allocation. Allocation failures are expected on
        CPU-only builds and fall back to the original bytearray path.
        """
        if size == 0:
            return b""
        pinned_result: memoryview | None = None
        if pinned:
            try:
                import torch

                # PyTorch tensors do not expose the Python buffer protocol in
                # all supported versions. numpy() is a zero-copy view and
                # keeps the tensor allocation alive through the view object.
                pinned_result = memoryview(
                    torch.empty(size, dtype=torch.uint8, pin_memory=True).numpy()
                )
            except (ImportError, RuntimeError, TypeError, NotImplementedError):
                CLIENT_METRICS.add(makv_client_pinned_receive_fallbacks=1)
        result: bytearray | memoryview = (
            pinned_result if pinned_result is not None else bytearray(size)
        )
        if pinned_result is not None:
            CLIENT_METRICS.add(makv_client_pinned_receive_bytes=size)
        view = memoryview(result)
        try:
            while view:
                received = await self.loop.sock_recv_into(sock, view)
                if received == 0:
                    raise ConnectionError(
                        "MaKV manager closed an incomplete response"
                    )
                view = view[received:]
        finally:
            view.release()
        return result

    @staticmethod
    def _recv_exact(connection: socket.socket, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            data = connection.recv(size - len(chunks))
            if not data:
                raise ConnectionError("MaKV manager closed an incomplete response")
            chunks.extend(data)
        return bytes(chunks)
