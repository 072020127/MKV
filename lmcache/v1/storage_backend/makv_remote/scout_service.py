# SPDX-License-Identifier: Apache-2.0

"""Bounded asynchronous ScoutRank jobs hosted by the MaKV manager."""

from __future__ import annotations

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol
import asyncio
import time


class ScoutRuntime(Protocol):
    """Minimal interface implemented by the reusable 28-layer scorer."""

    def score_token_ids(self, token_ids: list[int]) -> list[float]: ...


@dataclass(frozen=True)
class ScoutScoreResult:
    scores: list[float]
    queue_time_ms: float
    score_time_ms: float
    total_time_ms: float
    score_start: int = 0
    score_end: int = 0
    full_token_count: int = 0


@dataclass
class ScoutJob:
    request_id: str
    token_count: int
    token_sha256: str
    score_context_sha256: str
    score_start: int
    score_end: int
    full_token_count: int
    submitted_at: float
    future: Future[ScoutScoreResult]
    metrics_recorded: bool = False


@dataclass
class ScoutServiceMetrics:
    submitted_jobs: int = 0
    deduplicated_submissions: int = 0
    cross_request_deduplicated_submissions: int = 0
    prompt_hash_cache_hits: int = 0
    prompt_hash_cache_evictions: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    rejected_jobs: int = 0
    wait_calls: int = 0
    queue_time_ms: float = 0.0
    score_time_ms: float = 0.0
    total_job_time_ms: float = 0.0
    exposed_wait_time_ms: float = 0.0
    background_wait_time_ms: float = 0.0
    overlap_hidden_time_ms: float = 0.0


class ScoutJobService:
    """Run one persistent Scout model behind a bounded, idempotent queue."""

    def __init__(
        self,
        runtime: ScoutRuntime,
        *,
        max_pending_jobs: int = 64,
        result_ttl_s: float = 600.0,
        max_cached_prompts: int = 128,
    ) -> None:
        if max_pending_jobs <= 0 or result_ttl_s <= 0 or max_cached_prompts <= 0:
            raise ValueError(
                "Scout queue depth, result TTL, and prompt cache capacity "
                "must be positive"
            )
        self.runtime = runtime
        self.max_pending_jobs = max_pending_jobs
        self.result_ttl_s = result_ttl_s
        self.max_cached_prompts = max_cached_prompts
        # A single worker preserves model state safety and predictable GPU use.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="makv-scout"
        )
        # Request IDs are short-lived aliases. The prompt hash is the stable
        # identity that allows different requests to share an in-flight or
        # completed importance result.
        self._jobs: dict[str, ScoutJob] = {}
        self._prompt_jobs: dict[str, ScoutJob] = {}
        self.metrics = ScoutServiceMetrics()

    def _unique_jobs(self) -> tuple[ScoutJob, ...]:
        """Return jobs once even when several request IDs share one job."""
        unique: dict[int, ScoutJob] = {}
        for job in self._jobs.values():
            unique[id(job)] = job
        return tuple(unique.values())

    def _drop_job(self, job: ScoutJob) -> None:
        """Remove a job and all request aliases pointing at it."""
        for request_id, candidate in tuple(self._jobs.items()):
            if candidate is job:
                del self._jobs[request_id]
        if self._prompt_jobs.get(job.score_context_sha256) is job:
            del self._prompt_jobs[job.score_context_sha256]

    def _evict_completed(self) -> None:
        """Bound prompt-hash lookup entries without evicting request jobs."""
        while True:
            if len(self._prompt_jobs) <= self.max_cached_prompts:
                return
            completed = [
                job
                for job in self._prompt_jobs.values()
                if job.future.done() and not job.future.cancelled()
            ]
            if not completed:
                # Running work is never discarded to make room for a cache
                # entry. It will be considered after the next submission or
                # health check once the future has completed.
                return
            oldest = min(completed, key=lambda job: job.submitted_at)
            if self._prompt_jobs.get(oldest.score_context_sha256) is oldest:
                # Keep request_id aliases alive until their normal TTL. This
                # prevents capacity eviction from making an outstanding
                # SCOUT_WAIT fail; only future cross-request lookups miss.
                del self._prompt_jobs[oldest.score_context_sha256]
            self.metrics.prompt_hash_cache_evictions += 1

    def _cleanup(self, now: float) -> None:
        for job in self._unique_jobs():
            if not job.future.done():
                continue
            if job.future.cancelled() or job.future.exception() is not None:
                # Failed results must not poison the prompt hash. A later
                # request may retry the score computation safely.
                self._drop_job(job)
                continue
            if now - job.submitted_at >= self.result_ttl_s:
                self._drop_job(job)
        self._evict_completed()

    def _run(
        self,
        token_ids: list[int],
        submitted_at: float,
        score_start: int,
        score_end: int,
        full_token_count: int,
    ) -> ScoutScoreResult:
        score_started = time.perf_counter()
        scores = self.runtime.score_token_ids(token_ids)
        completed_at = time.perf_counter()
        if len(scores) != len(token_ids):
            raise ValueError("ScoutRank score count does not match prompt token count")
        return ScoutScoreResult(
            scores=scores,
            queue_time_ms=(score_started - submitted_at) * 1000.0,
            score_time_ms=(completed_at - score_started) * 1000.0,
            total_time_ms=(completed_at - submitted_at) * 1000.0,
            score_start=score_start,
            score_end=score_end,
            full_token_count=full_token_count,
        )

    def submit(
        self,
        request_id: str,
        token_ids: list[int],
        token_sha256: str,
        *,
        score_start: int = 0,
        full_token_count: int | None = None,
        score_context_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Enqueue a request, returning immediately after executor submission."""
        if not request_id:
            raise ValueError("ScoutRank submission requires request_id")
        if not token_sha256:
            raise ValueError("ScoutRank submission requires token_sha256")
        if full_token_count is None:
            full_token_count = len(token_ids)
        score_end = score_start + len(token_ids)
        if not (
            0 <= score_start <= score_end <= full_token_count
        ):
            raise ValueError("ScoutRank score range is outside the full prompt")
        score_context = score_context_sha256 or token_sha256
        now = time.perf_counter()
        self._cleanup(now)
        existing = self._jobs.get(request_id)
        if existing is not None:
            if (
                existing.token_count != len(token_ids)
                or existing.token_sha256 != token_sha256
                or existing.score_context_sha256 != score_context
                or existing.score_start != score_start
                or existing.score_end != score_end
                or existing.full_token_count != full_token_count
            ):
                raise ValueError("ScoutRank request_id was reused for another prompt")
            self.metrics.deduplicated_submissions += 1
            return {"accepted": True, "deduplicated": True}

        prompt_job = self._prompt_jobs.get(score_context)
        if prompt_job is not None:
            if (
                prompt_job.token_count != len(token_ids)
                or prompt_job.score_start != score_start
                or prompt_job.score_end != score_end
                or prompt_job.full_token_count != full_token_count
            ):
                raise ValueError(
                    "ScoutRank score context was reused for another prompt range"
                )
            # This alias is deliberately installed only after the server has
            # verified the payload hash. The service then shares both pending
            # execution and the retained completed result across requests.
            self._jobs[request_id] = prompt_job
            self.metrics.deduplicated_submissions += 1
            self.metrics.cross_request_deduplicated_submissions += 1
            self.metrics.prompt_hash_cache_hits += 1
            return {
                "accepted": True,
                "deduplicated": True,
                "dedup_scope": "prompt_hash",
            }

        pending = sum(not job.future.done() for job in self._unique_jobs())
        if pending >= self.max_pending_jobs:
            self.metrics.rejected_jobs += 1
            raise asyncio.QueueFull
        future = self._executor.submit(
            self._run,
            token_ids,
            now,
            score_start,
            score_end,
            full_token_count,
        )
        self._jobs[request_id] = ScoutJob(
            request_id=request_id,
            token_count=len(token_ids),
            token_sha256=token_sha256,
            score_context_sha256=score_context,
            score_start=score_start,
            score_end=score_end,
            full_token_count=full_token_count,
            submitted_at=now,
            future=future,
        )
        self._prompt_jobs[score_context] = self._jobs[request_id]
        self.metrics.submitted_jobs += 1
        self._evict_completed()
        return {"accepted": True, "deduplicated": False}

    async def wait(
        self,
        request_id: str,
        token_count: int,
        timeout_s: float,
        deferred: bool = False,
        score_start: int | None = None,
        score_end: int | None = None,
    ) -> tuple[ScoutScoreResult, dict[str, float]]:
        """Wait only at the MaKV store dependency boundary."""
        if timeout_s <= 0:
            raise ValueError("ScoutRank wait timeout must be positive")
        job = self._jobs.get(request_id)
        if job is None:
            raise KeyError(f"ScoutRank job not found for request {request_id!r}")
        if job.full_token_count != token_count:
            raise ValueError(
                "ScoutRank wait full token_count does not match submission"
            )
        if score_start is not None and score_start != job.score_start:
            raise ValueError("ScoutRank wait score_start does not match submission")
        if score_end is not None and score_end != job.score_end:
            raise ValueError("ScoutRank wait score_end does not match submission")
        wait_started = time.perf_counter()
        self.metrics.wait_calls += 1
        try:
            deadline = wait_started + timeout_s
            while not job.future.done():
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("ScoutRank job timed out")
                # Polling a concurrent Future avoids depending on an asyncio
                # cross-thread callback, which is unreliable in some serving
                # launchers. The event loop remains free to serve storage I/O.
                await asyncio.sleep(min(0.002, remaining))
            result = job.future.result()
        except Exception:
            if job.future.done() and not job.future.cancelled():
                self.metrics.failed_jobs += 1
            raise
        wait_ms = (time.perf_counter() - wait_started) * 1000.0
        hidden_ms = (
            result.total_time_ms
            if deferred
            else max(0.0, result.total_time_ms - wait_ms)
        )
        if deferred:
            self.metrics.background_wait_time_ms += wait_ms
        else:
            self.metrics.exposed_wait_time_ms += wait_ms
        if not job.metrics_recorded:
            self.metrics.completed_jobs += 1
            self.metrics.queue_time_ms += result.queue_time_ms
            self.metrics.score_time_ms += result.score_time_ms
            self.metrics.total_job_time_ms += result.total_time_ms
            self.metrics.overlap_hidden_time_ms += hidden_ms
            job.metrics_recorded = True
        return result, {
            "wait_time_ms": wait_ms,
            "overlap_hidden_time_ms": hidden_ms,
        }

    def health(self) -> dict[str, Any]:
        """Return queue state and cumulative overlap timings."""
        now = time.perf_counter()
        self._cleanup(now)
        jobs = self._unique_jobs()
        return {
            "enabled": True,
            "pending_jobs": sum(not job.future.done() for job in jobs),
            "retained_jobs": len(jobs),
            "prompt_hash_cache_entries": len(self._prompt_jobs),
            "prompt_hash_cache_capacity": self.max_cached_prompts,
            "queue_depth": self.max_pending_jobs,
            "metrics": dict(self.metrics.__dict__),
        }

    def close(self) -> None:
        """Finish submitted work and release the executor."""
        self._executor.shutdown(wait=True, cancel_futures=False)
