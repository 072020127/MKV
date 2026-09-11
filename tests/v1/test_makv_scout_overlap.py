# SPDX-License-Identifier: Apache-2.0

"""Tests for full-depth ScoutRank/main-model overlap plumbing."""

# Standard
from types import SimpleNamespace
import asyncio
import multiprocessing as mp
import os
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.makv.config import IMPORTANCE_REQUEST_KEY
from lmcache.v1.storage_backend.makv.metrics import CLIENT_METRICS
from lmcache.v1.storage_backend.makv.scout_overlap import (
    ScoutOverlapClient,
    ScoutOverlapResult,
    get_scout_score_range,
    resolve_scout_importance,
    submit_scout_if_needed,
)
from lmcache.v1.storage_backend.makv_remote.scout_protocol import (
    SCOUT_PROTOCOL_VERSION,
    SCOUT_SUFFIX_PROTOCOL_VERSION,
    decode_scores,
    decode_token_ids,
    encode_scores,
    encode_token_ids,
    payload_sha256,
    score_context_sha256,
)
from lmcache.v1.storage_backend.makv_remote.scout_service import ScoutJobService
from lmcache.v1.storage_backend.makv_remote.server import MaKVRemoteServer
from lmcache.v1.cache_engine import LMCacheEngine


class _BlockingRuntime:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def score_token_ids(self, token_ids: list[int]) -> list[float]:
        self.calls += 1
        self.started.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError("test did not release ScoutRank runtime")
        return [float(value) / 10.0 for value in token_ids]


class _TcpScoutRuntime:
    def score_token_ids(self, token_ids: list[int]) -> list[float]:
        return [float(value) / 10.0 for value in token_ids]


class _TcpScoutManager:
    async def health(self):
        return {"pid": os.getpid(), "metrics": {}}


def _run_tcp_scout_manager(ready, stop) -> None:
    """Run a minimal real Manager process for the TCP protocol test."""

    async def run() -> None:
        jobs = ScoutJobService(
            _TcpScoutRuntime(), result_ttl_s=30.0, max_cached_prompts=8
        )
        service = MaKVRemoteServer(
            _TcpScoutManager(),
            queue_depth=1,
            workers=1,
            max_request_bytes=1024 * 1024,
            scout_service=jobs,
        )
        await service.start_workers()
        listener = await asyncio.start_server(
            service.handle_client, "127.0.0.1", 0
        )
        ready.put(listener.sockets[0].getsockname()[1])
        try:
            await asyncio.to_thread(stop.wait)
        finally:
            listener.close()
            await listener.wait_closed()
            await service.close()

    asyncio.run(run())


def _config(serde: str = "makv") -> SimpleNamespace:
    return SimpleNamespace(
        remote_serde=serde,
        remote_url="makv://127.0.0.1:65432",
        chunk_size=256,
        save_unfull_chunk=False,
        extra_config={
            "makv_scout_overlap_enabled": True,
            "makv_scout_timeout_s": 1.0,
        },
    )


def test_scout_binary_payload_round_trip() -> None:
    token_ids = [0, 1, 151935, 0xFFFFFFFF]
    token_payload = encode_token_ids(token_ids)
    assert decode_token_ids(token_payload, len(token_ids)) == token_ids
    assert payload_sha256(token_payload) == payload_sha256(token_payload)

    scores = [-1.25, 0.0, 0.5, 100.0]
    restored = decode_scores(encode_scores(scores), len(scores))
    assert restored == pytest.approx(scores)
    with pytest.raises(ValueError, match="token_count"):
        decode_scores(encode_scores(scores), len(scores) - 1)


def test_scout_service_overlaps_and_deduplicates() -> None:
    async def run() -> None:
        runtime = _BlockingRuntime()
        service = ScoutJobService(runtime, max_pending_jobs=1, result_ttl_s=10)
        token_ids = [4, 5, 6]
        checksum = payload_sha256(encode_token_ids(token_ids))
        submit_started = time.perf_counter()
        accepted = service.submit("req-1", token_ids, checksum)
        assert (time.perf_counter() - submit_started) < 0.1
        assert accepted == {"accepted": True, "deduplicated": False}
        assert service.submit("req-1", token_ids, checksum)["deduplicated"]
        with pytest.raises(ValueError, match="reused"):
            service.submit("req-1", [7], payload_sha256(encode_token_ids([7])))
        with pytest.raises(asyncio.QueueFull):
            service.submit("req-2", [8], payload_sha256(encode_token_ids([8])))

        assert runtime.started.wait(timeout=1)
        await asyncio.sleep(0.02)
        runtime.release.set()
        result, timing = await service.wait("req-1", len(token_ids), 1.0)
        assert result.scores == pytest.approx([0.4, 0.5, 0.6])
        assert timing["overlap_hidden_time_ms"] > 0
        assert runtime.calls == 1
        assert service.health()["metrics"]["completed_jobs"] == 1
        service.close()

    asyncio.run(run())


def test_scout_service_reuses_prompt_hash_across_requests() -> None:
    async def run() -> None:
        runtime = _BlockingRuntime()
        service = ScoutJobService(
            runtime, max_pending_jobs=1, result_ttl_s=10.0, max_cached_prompts=4
        )
        token_ids = [21, 22, 23, 24]
        checksum = payload_sha256(encode_token_ids(token_ids))

        assert service.submit("request-a", token_ids, checksum) == {
            "accepted": True,
            "deduplicated": False,
        }
        # A second request joins the same in-flight job and does not consume
        # another queue slot.
        assert service.submit("request-b", token_ids, checksum) == {
            "accepted": True,
            "deduplicated": True,
            "dedup_scope": "prompt_hash",
        }
        with pytest.raises(asyncio.QueueFull):
            service.submit(
                "request-c",
                [25],
                payload_sha256(encode_token_ids([25])),
            )

        assert runtime.started.wait(timeout=1)
        runtime.release.set()
        result_a, _ = await service.wait("request-a", len(token_ids), 1.0)
        result_b, _ = await service.wait("request-b", len(token_ids), 1.0)
        assert result_a.scores == pytest.approx(result_b.scores)
        assert runtime.calls == 1

        # The completed result is also reusable by a later request.
        assert service.submit("request-d", token_ids, checksum)["dedup_scope"] == (
            "prompt_hash"
        )
        result_d, _ = await service.wait("request-d", len(token_ids), 1.0)
        assert result_d.scores == pytest.approx(result_a.scores)
        assert runtime.calls == 1

        health = service.health()
        assert health["pending_jobs"] == 0
        assert health["retained_jobs"] == 1
        assert health["prompt_hash_cache_entries"] == 1
        metrics = health["metrics"]
        assert metrics["cross_request_deduplicated_submissions"] == 2
        assert metrics["prompt_hash_cache_hits"] == 2
        service.close()

    asyncio.run(run())


def test_scout_service_failed_prompt_hash_can_retry() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.calls = 0

        def score_token_ids(self, token_ids: list[int]) -> list[float]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient ScoutRank failure")
            return [float(value) for value in token_ids]

    async def run() -> None:
        runtime = Runtime()
        service = ScoutJobService(runtime, result_ttl_s=10.0)
        token_ids = [31, 32]
        checksum = payload_sha256(encode_token_ids(token_ids))
        service.submit("failed-request", token_ids, checksum)
        with pytest.raises(RuntimeError, match="transient"):
            await service.wait("failed-request", len(token_ids), 1.0)

        # Failed jobs are removed on the next submission cleanup and cannot
        # poison the cross-request cache.
        assert service.submit("retry-request", token_ids, checksum) == {
            "accepted": True,
            "deduplicated": False,
        }
        result, _ = await service.wait("retry-request", len(token_ids), 1.0)
        assert result.scores == pytest.approx(token_ids)
        assert runtime.calls == 2
        service.close()

    asyncio.run(run())


def test_scout_service_bounds_completed_prompt_cache() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.calls = 0

        def score_token_ids(self, token_ids: list[int]) -> list[float]:
            self.calls += 1
            return [float(value) for value in token_ids]

    async def run() -> None:
        runtime = Runtime()
        service = ScoutJobService(
            runtime, result_ttl_s=10.0, max_cached_prompts=1
        )

        async def score(request_id: str, token_ids: list[int]) -> None:
            checksum = payload_sha256(encode_token_ids(token_ids))
            service.submit(request_id, token_ids, checksum)
            await service.wait(request_id, len(token_ids), 1.0)

        await score("request-a", [41])
        await score("request-b", [42])
        assert service.health()["prompt_hash_cache_entries"] == 1
        assert service.health()["metrics"]["prompt_hash_cache_evictions"] == 1

        # The first result was evicted, so it is recomputed rather than
        # incorrectly treated as a cache hit.
        token_ids = [41]
        checksum = payload_sha256(encode_token_ids(token_ids))
        assert service.submit("request-a-again", token_ids, checksum)[
            "deduplicated"
        ] is False
        await service.wait("request-a-again", len(token_ids), 1.0)
        assert runtime.calls == 3
        service.close()

    asyncio.run(run())


def test_manager_dispatches_scout_protocol() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.calls = 0

        def score_token_ids(self, token_ids: list[int]) -> list[float]:
            self.calls += 1
            return [float(value) for value in token_ids]

    class Manager:
        async def health(self):
            return {"pid": 1, "metrics": {}}

    async def run() -> None:
        jobs = ScoutJobService(Runtime())
        server = MaKVRemoteServer(
            Manager(),
            queue_depth=1,
            workers=1,
            max_request_bytes=1024,
            scout_service=jobs,
        )
        token_ids = [11, 12, 13]
        payload = encode_token_ids(token_ids)
        submit_header = {
            "protocol_version": SCOUT_PROTOCOL_VERSION,
            "token_count": len(token_ids),
            "token_sha256": payload_sha256(payload),
        }
        response, response_payload = await server._dispatch(
            "SCOUT_SUBMIT", "req-wire", payload, submit_header
        )
        assert response["accepted"]
        assert response_payload == b""
        wait_header = {
            "protocol_version": SCOUT_PROTOCOL_VERSION,
            "token_count": len(token_ids),
            "timeout_s": 1.0,
        }
        response, response_payload = await server._dispatch(
            "SCOUT_WAIT", "req-wire", b"", wait_header
        )
        assert decode_scores(response_payload, len(token_ids)) == pytest.approx(
            token_ids
        )
        assert response["score_time_ms"] >= 0
        second_submit, _ = await server._dispatch(
            "SCOUT_SUBMIT", "req-wire-second", payload, submit_header
        )
        assert second_submit == {
            "accepted": True,
            "deduplicated": True,
            "dedup_scope": "prompt_hash",
        }
        second_wait, second_payload = await server._dispatch(
            "SCOUT_WAIT", "req-wire-second", b"", wait_header
        )
        assert decode_scores(second_payload, len(token_ids)) == pytest.approx(
            token_ids
        )
        assert second_wait["score_time_ms"] >= 0
        assert jobs.runtime.calls == 1
        health, _ = await server._dispatch("HEALTH", "", b"")
        assert health["scout"]["enabled"] is True
        assert health["scout"]["prompt_hash_cache_entries"] == 1
        await server.close()

    asyncio.run(run())


def test_manager_dispatches_suffix_range_without_rescoring_prefix() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.seen: list[list[int]] = []

        def score_token_ids(self, token_ids: list[int]) -> list[float]:
            self.seen.append(list(token_ids))
            return [float(value) for value in token_ids]

    class Manager:
        async def health(self):
            return {"pid": 1, "metrics": {}}

    async def run() -> None:
        runtime = Runtime()
        jobs = ScoutJobService(runtime)
        server = MaKVRemoteServer(
            Manager(),
            queue_depth=1,
            workers=1,
            max_request_bytes=1024,
            scout_service=jobs,
        )
        token_ids = [71, 72]
        payload = encode_token_ids(token_ids)
        token_sha = payload_sha256(payload)
        submit_header = {
            "protocol_version": SCOUT_SUFFIX_PROTOCOL_VERSION,
            "token_count": len(token_ids),
            "token_sha256": token_sha,
            "score_start": 4,
            "score_end": 6,
            "full_token_count": 6,
            "score_context_sha256": score_context_sha256(token_sha, 4, 6),
        }
        response, _ = await server._dispatch(
            "SCOUT_SUBMIT", "req-suffix-wire", payload, submit_header
        )
        assert response["accepted"] is True
        wait_header = {
            "protocol_version": SCOUT_SUFFIX_PROTOCOL_VERSION,
            "token_count": 6,
            "score_start": 4,
            "score_end": 6,
            "full_token_count": 6,
            "timeout_s": 1.0,
        }
        response, response_payload = await server._dispatch(
            "SCOUT_WAIT", "req-suffix-wire", b"", wait_header
        )
        assert decode_scores(response_payload, 2) == pytest.approx([71.0, 72.0])
        assert (response["score_start"], response["score_end"]) == (4, 6)
        assert runtime.seen == [token_ids]
        await server.close()

    asyncio.run(run())


@pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(),
    reason="real TCP Scout manager test requires fork",
)
def test_scout_prompt_hash_reuse_over_real_tcp_multiprocess() -> None:
    context = mp.get_context("fork")
    ready = context.Queue()
    stop = context.Event()
    process = context.Process(
        target=_run_tcp_scout_manager, args=(ready, stop), daemon=True
    )
    process.start()
    try:
        port = ready.get(timeout=5.0)
        client = ScoutOverlapClient(f"makv://127.0.0.1:{port}", timeout_s=5.0)
        token_ids = [51, 52, 53, 54]
        first = client.submit("tcp-request-a", token_ids)
        second = client.submit("tcp-request-b", token_ids)
        assert first["deduplicated"] is False
        assert second == {
            "status": "ok",
            "accepted": True,
            "deduplicated": True,
            "dedup_scope": "prompt_hash",
        }

        first_result = client.wait("tcp-request-a", len(token_ids))
        second_result = client.wait("tcp-request-b", len(token_ids))
        assert first_result.scores == pytest.approx(second_result.scores)

        health, payload = client._request({"op": "HEALTH"})
        assert payload == b""
        assert health["scout"]["pending_jobs"] == 0
        assert health["scout"]["prompt_hash_cache_entries"] == 1
        assert health["scout"]["metrics"]["submitted_jobs"] == 1
        assert health["scout"]["metrics"]["completed_jobs"] == 1
        assert (
            health["scout"]["metrics"]["cross_request_deduplicated_submissions"]
            == 1
        )
        assert health["scout"]["metrics"]["prompt_hash_cache_hits"] == 1
        assert health["pid"] == process.pid
        assert process.pid != os.getpid()
    finally:
        stop.set()
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        ready.close()
        ready.join_thread()


@pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(),
    reason="real TCP Scout manager test requires fork",
)
def test_scout_suffix_round_trip_over_real_tcp_multiprocess() -> None:
    context = mp.get_context("fork")
    ready = context.Queue()
    stop = context.Event()
    process = context.Process(
        target=_run_tcp_scout_manager, args=(ready, stop), daemon=True
    )
    process.start()
    try:
        port = ready.get(timeout=5.0)
        client = ScoutOverlapClient(f"makv://127.0.0.1:{port}", timeout_s=5.0)
        response = client.submit(
            "tcp-suffix-request",
            [61, 62],
            score_start=4,
            full_token_count=6,
        )
        assert response["accepted"] is True
        result = client.wait(
            "tcp-suffix-request",
            6,
            score_start=4,
            score_end=6,
        )
        assert result.scores == pytest.approx([6.1, 6.2])
        assert (result.score_start, result.score_end, result.full_token_count) == (
            4,
            6,
            6,
        )
        health, _ = client._request({"op": "HEALTH"})
        assert health["scout"]["metrics"]["submitted_jobs"] == 1
    finally:
        stop.set()
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        ready.close()
        ready.join_thread()


def test_submit_and_join_are_makv_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeClient:
        def submit(self, request_id, token_ids):
            calls.append(("submit", (request_id, list(token_ids))))
            return {"accepted": True}

        def wait(self, request_id, token_count, *, deferred=False):
            assert deferred is False
            calls.append(("wait", (request_id, token_count)))
            return ScoutOverlapResult(
                scores=[0.1, 0.2, 0.3],
                score_time_ms=40.0,
                total_job_time_ms=42.0,
                wait_time_ms=10.0,
                overlap_hidden_time_ms=32.0,
            )

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.makv.scout_overlap._client_from_config",
        lambda config: FakeClient(),
    )
    CLIENT_METRICS.reset()
    config = _config()
    config.save_unfull_chunk = True
    assert submit_scout_if_needed(
        config,
        request_id="req-3",
        token_ids=[1, 2, 3],
        request_configs=None,
        cached_tokens=0,
    )
    resolved = resolve_scout_importance(
        _config(),
        request_id="req-3",
        token_count=3,
        request_configs={"unrelated": True},
    )
    assert resolved is not None
    assert resolved[IMPORTANCE_REQUEST_KEY] == pytest.approx([0.1, 0.2, 0.3])
    assert resolved["unrelated"] is True
    cached_resolved = resolve_scout_importance(
        config,
        request_id="req-3",
        token_count=3,
        request_configs=None,
    )
    assert cached_resolved is not None
    assert cached_resolved[IMPORTANCE_REQUEST_KEY] == pytest.approx(
        [0.1, 0.2, 0.3]
    )
    assert [name for name, _ in calls] == ["submit", "wait"]
    metrics = CLIENT_METRICS.snapshot()
    assert metrics.makv_scout_submit_calls == 1
    assert metrics.makv_scout_wait_calls == 1
    assert metrics.makv_scout_overlap_hidden_time_ms == 32.0

    calls.clear()
    assert not submit_scout_if_needed(
        _config("cachegen"),
        request_id="req-native",
        token_ids=[1, 2, 3],
        request_configs=None,
        cached_tokens=0,
    )
    assert calls == []


def test_partial_hit_submits_only_uncached_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, list[int], dict[str, int]]] = []

    class FakeClient:
        def submit(self, request_id, token_ids, **kwargs):
            calls.append((request_id, list(token_ids), dict(kwargs)))
            return {"accepted": True}

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.makv.scout_overlap._client_from_config",
        lambda config: FakeClient(),
    )
    config = _config()
    token_ids = list(range(1024))
    assert get_scout_score_range(
        config,
        token_count=len(token_ids),
        cached_tokens=512,
        request_configs=None,
    ) == (512, 1024)
    assert submit_scout_if_needed(
        config,
        request_id="req-suffix-submit",
        token_ids=token_ids,
        request_configs=None,
        cached_tokens=512,
    )
    assert calls == [
        (
            "req-suffix-submit",
            token_ids[512:],
            {"score_start": 512, "full_token_count": 1024},
        )
    ]


def test_partial_result_expands_to_absolute_scores_and_masks_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def wait(
            self,
            request_id,
            token_count,
            *,
            deferred=False,
            score_start=None,
            score_end=None,
        ):
            assert request_id == "req-suffix-resolve"
            assert token_count == 1024
            assert deferred is False
            assert (score_start, score_end) == (512, 1024)
            return ScoutOverlapResult(
                scores=[float(index) for index in range(512)],
                score_time_ms=1.0,
                total_job_time_ms=1.0,
                wait_time_ms=1.0,
                overlap_hidden_time_ms=0.0,
                score_start=512,
                score_end=1024,
                full_token_count=1024,
            )

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.makv.scout_overlap._client_from_config",
        lambda config: FakeClient(),
    )
    resolved = resolve_scout_importance(
        _config(),
        request_id="req-suffix-resolve",
        token_count=1024,
        request_configs=None,
        score_start=512,
        score_end=1024,
    )
    assert resolved is not None
    scores = resolved[IMPORTANCE_REQUEST_KEY]
    assert scores[:512] == [0.0] * 512
    assert scores[512:520] == pytest.approx(list(range(8)))
    status = resolved["lmcache.makv_importance_status"]
    assert all(item["valid_mask"] is False for item in status[:512])
    assert all(item["forced_precision"] == "BF16" for item in status[:512])
    assert all(item["valid_mask"] is True for item in status[512:])
    assert resolved["lmcache.makv_scout_score_start"] == 512
    assert resolved["lmcache.makv_scout_score_end"] == 1024


def test_suffix_only_can_be_disabled_for_legacy_full_prompt_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[int]] = []

    class FakeClient:
        def submit(self, request_id, token_ids):
            del request_id
            calls.append(list(token_ids))
            return {"accepted": True}

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.makv.scout_overlap._client_from_config",
        lambda config: FakeClient(),
    )
    config = _config()
    config.extra_config["makv_scout_suffix_only"] = False
    token_ids = list(range(512))
    assert submit_scout_if_needed(
        config,
        request_id="req-full-legacy",
        token_ids=token_ids,
        request_configs=None,
        cached_tokens=256,
    )
    assert calls == [token_ids]


def test_explicit_importance_bypasses_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.makv.scout_overlap._client_from_config",
        lambda config: pytest.fail("overlap client must not be created"),
    )
    request_configs = {IMPORTANCE_REQUEST_KEY: [0.5, 0.4]}
    assert not submit_scout_if_needed(
        _config(),
        request_id="req-explicit",
        token_ids=[1, 2],
        request_configs=request_configs,
        cached_tokens=0,
    )
    assert (
        resolve_scout_importance(
            _config(),
            request_id="req-explicit",
            token_count=2,
            request_configs=request_configs,
        )
        is request_configs
    )


def test_complete_aligned_hit_does_not_submit_tail_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.makv.scout_overlap._client_from_config",
        lambda config: pytest.fail("aligned complete hit must not submit"),
    )
    assert not submit_scout_if_needed(
        _config(),
        request_id="req-hit",
        token_ids=list(range(279)),
        request_configs=None,
        cached_tokens=256,
    )


def test_deferred_store_joins_before_batched_put(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Memory:
        refs = 1

        def ref_count_down(self):
            self.refs -= 1

    class StorageManager:
        def batched_put(self, keys, memory_objs, *, transfer_spec, location):
            events.append("put")
            assert keys == ["key"]
            assert transfer_spec["request_configs"][IMPORTANCE_REQUEST_KEY] == [
                0.1,
                0.2,
            ]
            assert transfer_spec["chunk_starts"] == [0]
            assert transfer_spec["chunk_ends"] == [2]
            assert location == "RemoteBackend"
            for memory_obj in memory_objs:
                memory_obj.ref_count_down()

    def resolve(*args, **kwargs):
        assert kwargs["deferred"] is True
        events.append("join")
        return {IMPORTANCE_REQUEST_KEY: [0.1, 0.2]}

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.makv.scout_overlap.resolve_scout_importance",
        resolve,
    )
    engine = object.__new__(LMCacheEngine)
    engine.config = _config()
    engine.storage_manager = StorageManager()
    engine.store_location = "RemoteBackend"
    memory = Memory()
    engine._deferred_makv_put(
        ["key"],
        [memory],
        [0],
        [2],
        None,
        None,
        2,
        [1, 2],
        "req-deferred",
    )
    assert events == ["join", "put"]
    assert memory.refs == 0
