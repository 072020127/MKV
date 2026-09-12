# SPDX-License-Identifier: Apache-2.0

"""In-process MaKV metrics used by tests and benchmarks.

CUDA work is asynchronous, so host ``perf_counter`` intervals around a
``Tensor.to(..., non_blocking=True)`` call only measure submission overhead.
The restore accumulator therefore retains CUDA events and folds them into the
snapshot only after their terminal event has completed. Production callers use
``query()``; the explicit benchmark path may request an event wait.
"""

# Standard
from dataclasses import dataclass
from typing import Any
import threading
import time


DEFAULT_MAX_INFLIGHT_RESTORE_TICKETS = 64


@dataclass
class MaKVMetricsSnapshot:
    makv_plan_time_ms: float = 0.0
    # ``makv_plan_time_ms`` is retained for compatibility and now measures
    # only deterministic importance/precision-plan construction. Payload
    # materialization and binary-envelope copies are reported separately.
    makv_client_plan_build_time_ms: float = 0.0
    makv_client_raw_payload_copy_time_ms: float = 0.0
    makv_client_envelope_encode_time_ms: float = 0.0
    makv_client_serialize_total_time_ms: float = 0.0
    makv_put_raw_bytes: int = 0
    makv_put_plan_bytes: int = 0
    makv_client_quantize_calls: int = 0
    makv_scout_submit_calls: int = 0
    makv_scout_submit_time_ms: float = 0.0
    makv_scout_wait_calls: int = 0
    makv_scout_wait_time_ms: float = 0.0
    makv_scout_score_time_ms: float = 0.0
    makv_scout_overlap_hidden_time_ms: float = 0.0
    makv_remote_quantize_time_ms: float = 0.0
    makv_remote_quantize_queue_time_ms: float = 0.0
    makv_remote_residual_bytes: int = 0
    makv_remote_risk_signals: int = 0
    makv_remote_precision_upgrades: int = 0
    makv_remote_precision_upgrade_failures: int = 0
    makv_remote_residual_upgrade_time_ms: float = 0.0
    makv_remote_precision_window_activations: int = 0
    makv_remote_precision_window_refreshes: int = 0
    makv_remote_precision_window_expirations: int = 0
    makv_remote_precision_window_hits: int = 0
    makv_remote_precision_window_restores: int = 0
    makv_raw_input_bytes: int = 0
    makv_stored_bytes: int = 0
    makv_quantize_failures: int = 0
    makv_naive_fallbacks: int = 0
    makv_get_quantized_bytes: int = 0
    makv_memory_cache_hits: int = 0
    makv_memory_cache_misses: int = 0
    makv_remote_put_requests: int = 0
    makv_remote_put_decode_time_ms: float = 0.0
    makv_remote_plan_canonicalize_time_ms: float = 0.0
    makv_remote_quantize_kernel_time_ms: float = 0.0
    makv_remote_entropy_encode_calls: int = 0
    makv_remote_entropy_encode_time_ms: float = 0.0
    makv_remote_entropy_input_bytes: int = 0
    makv_remote_entropy_output_bytes: int = 0
    makv_remote_object_encode_time_ms: float = 0.0
    makv_remote_object_validate_time_ms: float = 0.0
    makv_remote_encode_validate_time_ms: float = 0.0
    makv_remote_storage_put_time_ms: float = 0.0
    makv_remote_put_total_time_ms: float = 0.0
    makv_remote_get_requests: int = 0
    makv_remote_get_hot_cache_time_ms: float = 0.0
    makv_remote_get_storage_time_ms: float = 0.0
    makv_remote_get_validate_time_ms: float = 0.0
    makv_remote_get_total_time_ms: float = 0.0
    makv_remote_get_checksum_verifications: int = 0
    makv_remote_get_checksum_skips: int = 0
    makv_remote_get_batch_requests: int = 0
    makv_remote_get_batch_objects: int = 0
    makv_remote_get_batch_storage_time_ms: float = 0.0
    makv_remote_get_batch_validate_time_ms: float = 0.0
    makv_remote_get_batch_total_time_ms: float = 0.0
    makv_remote_get_batch_blob_requests: int = 0
    makv_remote_get_batch_blob_bytes: int = 0
    makv_remote_get_stream_requests: int = 0
    makv_remote_get_stream_objects: int = 0
    makv_remote_get_stream_first_object_time_ms: float = 0.0
    makv_remote_get_stream_send_time_ms: float = 0.0
    makv_remote_get_stream_total_time_ms: float = 0.0
    makv_client_put_connect_time_ms: float = 0.0
    makv_client_put_send_time_ms: float = 0.0
    makv_client_put_response_time_ms: float = 0.0
    makv_client_put_total_time_ms: float = 0.0
    makv_client_get_batches: int = 0
    makv_client_get_objects: int = 0
    makv_client_get_connect_time_ms: float = 0.0
    makv_client_get_send_time_ms: float = 0.0
    makv_client_get_first_response_time_ms: float = 0.0
    makv_client_get_receive_time_ms: float = 0.0
    makv_client_get_total_time_ms: float = 0.0
    makv_client_get_batch_blob_frames: int = 0
    makv_client_get_batch_blob_bytes: int = 0
    makv_client_get_stream_requests: int = 0
    makv_client_get_stream_frames: int = 0
    makv_client_get_stream_bytes: int = 0
    makv_client_get_stream_prefetch_peak: int = 0
    makv_client_get_stream_prefetch_backpressure_waits: int = 0
    makv_client_pinned_receive_bytes: int = 0
    makv_client_pinned_receive_fallbacks: int = 0
    makv_client_deserialize_time_ms: float = 0.0
    makv_restore_calls: int = 0
    makv_restore_payload_bytes: int = 0
    makv_restore_cpu_blob_pin_time_ms: float = 0.0
    makv_restore_cpu_view_validate_time_ms: float = 0.0
    makv_restore_cpu_dtype_convert_time_ms: float = 0.0
    makv_restore_cpu_pin_time_ms: float = 0.0
    makv_restore_cpu_prepare_time_ms: float = 0.0
    makv_h2d_bytes: int = 0
    makv_h2d_time_ms: float = 0.0
    makv_dequant_kernel_time_ms: float = 0.0
    makv_entropy_decode_calls: int = 0
    makv_entropy_decode_time_ms: float = 0.0
    makv_entropy_decode_bytes: int = 0
    makv_restore_gpu_total_time_ms: float = 0.0
    makv_restore_total_time_ms: float = 0.0
    makv_kernel_launch_count: int = 0
    makv_cuda_pending_traces: int = 0
    makv_restore_ticket_reaped: int = 0
    makv_restore_ticket_backpressure_waits: int = 0
    makv_restore_ticket_peak: int = 0
    makv_restore_ready_event_handoffs: int = 0
    # Streaming-restore host timeline. These fields intentionally measure
    # submission boundaries only; CUDA event timing remains in the fields
    # above and is never inferred from host wall-clock time.
    makv_restore_pipeline_streamed_objects: int = 0
    makv_restore_pipeline_receive_to_submit_time_ms: float = 0.0
    makv_restore_pipeline_first_submit_delay_ms: float = 0.0
    makv_restore_pipeline_enqueue_span_ms: float = 0.0
    makv_restore_pipeline_scope_wall_time_ms: float = 0.0


@dataclass
class MaKVRestoreTicket:
    """One asynchronous paged restore and the buffers it still owns."""

    scope_id: int | None
    h2d_start: Any
    h2d_end: Any
    compute_start: Any
    kernel_end: Any
    # Non-blocking H2D copies must retain their CPU sources until this event
    # completes.  Keeping the GPU inputs here also prevents allocator reuse
    # before the paged scatter has consumed them.
    keepalive: tuple[Any, ...]


@dataclass
class _RestoreScopeTimeline:
    """Host submission timeline for one streamed restore request."""

    started_ns: int
    first_submit_ns: int | None = None
    last_submit_ns: int | None = None


class MaKVMetrics:
    """Thread-safe metrics accumulator."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = MaKVMetricsSnapshot()
        self._pending_cuda_traces: list[MaKVRestoreTicket] = []
        self._scopes: dict[int, MaKVMetricsSnapshot] = {}
        self._scope_timelines: dict[int, _RestoreScopeTimeline] = {}
        self._next_scope_id = 1

    def reset(self) -> None:
        """Reset counters without releasing buffers owned by unfinished work."""
        with self._lock:
            # A reset is exceptional (tests/explicit instrumentation reset),
            # so it is safe to wait ticket-by-ticket. Dropping keepalive here
            # would let pinned H2D sources be reclaimed before CUDA consumes
            # them. This is deliberately not cudaDeviceSynchronize().
            for ticket in self._pending_cuda_traces:
                ticket.kernel_end.synchronize()
            self._collect_ready_cuda_locked()
            self._snapshot = MaKVMetricsSnapshot()
            self._pending_cuda_traces.clear()
            self._scopes.clear()
            self._scope_timelines.clear()
            self._next_scope_id = 1

    def snapshot(self) -> MaKVMetricsSnapshot:
        with self._lock:
            self._collect_ready_cuda_locked()
            return MaKVMetricsSnapshot(**self._snapshot.__dict__)

    def add(self, **kwargs) -> None:
        with self._lock:
            self._add_locked(self._snapshot, kwargs)

    def begin_restore_scope(self) -> int:
        """Create a request-local scope for a batched paged restore."""
        with self._lock:
            scope_id = self._next_scope_id
            self._next_scope_id += 1
            self._scopes[scope_id] = MaKVMetricsSnapshot()
            self._scope_timelines[scope_id] = _RestoreScopeTimeline(
                started_ns=time.perf_counter_ns()
            )
            return scope_id

    def record_stream_restore_submission(
        self,
        scope_id: int,
        *,
        restore_ready_ns: int,
        submitted_ns: int | None = None,
    ) -> None:
        """Record when one complete object becomes eligible for GPU restore.

        The connector only calls this after the full object has passed framing
        and deserialization. This keeps incomplete network payloads outside
        the restore timeline and makes the metric safe for cache-miss paths.
        """
        if submitted_ns is None:
            submitted_ns = time.perf_counter_ns()
        with self._lock:
            scope = self._scopes.get(scope_id)
            timeline = self._scope_timelines.get(scope_id)
            if scope is None or timeline is None:
                return
            values = {
                "makv_restore_pipeline_streamed_objects": 1,
                "makv_restore_pipeline_receive_to_submit_time_ms": max(
                    0.0, (submitted_ns - restore_ready_ns) / 1_000_000
                ),
            }
            self._add_locked(self._snapshot, values)
            self._add_locked(scope, values)
            if timeline.first_submit_ns is None:
                timeline.first_submit_ns = submitted_ns
                first_submit_delay_ms = max(
                    0.0, (submitted_ns - timeline.started_ns) / 1_000_000
                )
                first_submit_values = {
                    "makv_restore_pipeline_first_submit_delay_ms": (
                        first_submit_delay_ms
                    )
                }
                self._add_locked(self._snapshot, first_submit_values)
                self._add_locked(scope, first_submit_values)
            timeline.last_submit_ns = submitted_ns

    def add_restore(self, scope_id: int | None, **kwargs) -> None:
        """Add CPU-side restore work to the global and request-local totals."""
        with self._lock:
            self._add_locked(self._snapshot, kwargs)
            if scope_id is not None and scope_id in self._scopes:
                self._add_locked(self._scopes[scope_id], kwargs)

    def record_cuda_restore(
        self,
        scope_id: int | None,
        *,
        h2d_start: Any,
        h2d_end: Any,
        compute_start: Any,
        kernel_end: Any,
        payload_bytes: int,
        h2d_bytes: int,
        kernel_launch_count: int,
        keepalive: tuple[Any, ...] = (),
        max_inflight_tickets: int | None = None,
    ) -> None:
        """Submit a bounded asynchronous restore ticket.

        Completion is normally polled without synchronization. If a caller
        exceeds the ticket cap, only the oldest outstanding ticket is waited
        on; this bounds pinned/GPU input ownership without a device-wide sync.
        """
        ticket_limit = (
            DEFAULT_MAX_INFLIGHT_RESTORE_TICKETS
            if max_inflight_tickets is None
            else int(max_inflight_tickets)
        )
        if ticket_limit < 1:
            raise ValueError("max_inflight_tickets must be at least one")
        values = {
            "makv_restore_calls": 1,
            "makv_restore_payload_bytes": int(payload_bytes),
            "makv_h2d_bytes": int(h2d_bytes),
            "makv_kernel_launch_count": int(kernel_launch_count),
        }
        while True:
            with self._lock:
                self._collect_ready_cuda_locked()
                if len(self._pending_cuda_traces) < ticket_limit:
                    self._add_locked(self._snapshot, values)
                    scope = self._scopes.get(scope_id)
                    if scope is not None:
                        self._add_locked(scope, values)
                    self._pending_cuda_traces.append(
                        MaKVRestoreTicket(
                            scope_id=scope_id,
                            h2d_start=h2d_start,
                            h2d_end=h2d_end,
                            compute_start=compute_start,
                            kernel_end=kernel_end,
                            keepalive=keepalive,
                        )
                    )
                    pending_count = len(self._pending_cuda_traces)
                    self._snapshot.makv_cuda_pending_traces = pending_count
                    self._snapshot.makv_restore_ticket_peak = max(
                        self._snapshot.makv_restore_ticket_peak,
                        pending_count,
                    )
                    if scope is not None:
                        scope.makv_restore_ticket_peak = max(
                            scope.makv_restore_ticket_peak,
                            pending_count,
                        )
                    return
                oldest_event = self._pending_cuda_traces[0].kernel_end
                backpressure = {"makv_restore_ticket_backpressure_waits": 1}
                self._add_locked(self._snapshot, backpressure)
                scope = self._scopes.get(scope_id)
                if scope is not None:
                    self._add_locked(scope, backpressure)
            oldest_event.synchronize()

    def finish_restore_scope(
        self, scope_id: int, *, wait: bool = False
    ) -> MaKVMetricsSnapshot:
        """Collect and return one batched restore's timing delta.

        ``wait`` is intended only for an explicit benchmark. Normal V2/V3
        callers invoke this after their existing load-stream synchronize, so
        no extra device synchronization is introduced here.
        """
        if wait:
            with self._lock:
                events = [
                    trace.kernel_end
                    for trace in self._pending_cuda_traces
                    if trace.scope_id == scope_id
                ]
            for event in events:
                event.synchronize()
        with self._lock:
            self._collect_ready_cuda_locked()
            scope = self._scopes.pop(scope_id, MaKVMetricsSnapshot())
            timeline = self._scope_timelines.pop(scope_id, None)
            if timeline is not None:
                finished_ns = time.perf_counter_ns()
                values = {
                    "makv_restore_pipeline_scope_wall_time_ms": max(
                        0.0, (finished_ns - timeline.started_ns) / 1_000_000
                    ),
                    "makv_restore_pipeline_enqueue_span_ms": (
                        0.0
                        if timeline.first_submit_ns is None
                        or timeline.last_submit_ns is None
                        else max(
                            0.0,
                            (timeline.last_submit_ns - timeline.first_submit_ns)
                            / 1_000_000,
                        )
                    ),
                }
                self._add_locked(self._snapshot, values)
                self._add_locked(scope, values)
            return MaKVMetricsSnapshot(**scope.__dict__)

    @staticmethod
    def _add_locked(snapshot: MaKVMetricsSnapshot, values: dict[str, Any]) -> None:
        for key, value in values.items():
            current = getattr(snapshot, key)
            setattr(snapshot, key, current + value)

    def _collect_ready_cuda_locked(self) -> None:
        pending: list[MaKVRestoreTicket] = []
        for trace in self._pending_cuda_traces:
            try:
                complete = bool(trace.kernel_end.query())
            except RuntimeError:
                # Preserve the trace for a later poll when a stream is still
                # being initialized or the event has not reached the device.
                complete = False
            if not complete:
                pending.append(trace)
                continue
            values = {
                "makv_restore_ticket_reaped": 1,
                "makv_h2d_time_ms": trace.h2d_start.elapsed_time(trace.h2d_end),
                "makv_dequant_kernel_time_ms": trace.compute_start.elapsed_time(
                    trace.kernel_end
                ),
                "makv_restore_gpu_total_time_ms": trace.h2d_start.elapsed_time(
                    trace.kernel_end
                ),
                # Kept for compatibility with existing dashboards. With a
                # dedicated copy stream this is end-to-end GPU span, including
                # any compute-stream queueing before the paged scatter.
                "makv_restore_total_time_ms": trace.h2d_start.elapsed_time(
                    trace.kernel_end
                ),
            }
            self._add_locked(self._snapshot, values)
            if trace.scope_id is not None and trace.scope_id in self._scopes:
                self._add_locked(self._scopes[trace.scope_id], values)
        self._pending_cuda_traces = pending
        self._snapshot.makv_cuda_pending_traces = len(pending)


CLIENT_METRICS = MaKVMetrics()
REMOTE_METRICS = MaKVMetrics()
RESTORE_METRICS = MaKVMetrics()
