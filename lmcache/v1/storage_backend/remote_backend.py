# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future, TimeoutError
from typing import Any, Callable, Iterator, List, Optional, Sequence, Set
import asyncio
import threading
import time

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.exceptions import IrrecoverableException
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.connector import CreateConnector
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.naive_serde import CreateSerde

logger = init_logger(__name__)


class RemoteBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: Optional[LocalCPUBackend],
        dst_device: str = torch_device_type,
        plugin_name: Optional[str] = None,
    ):
        super().__init__(dst_device=dst_device)
        self.put_tasks: Set[CacheEngineKey] = set()
        self.lock = threading.Lock()

        self.plugin_name = plugin_name

        # Determine if we're using legacy remote_url or new plugin-based approach
        if plugin_name is not None:
            # Using plugin-based approach
            self.remote_url = f"plugin://{plugin_name}"
            logger.info("Creating RemoteBackend for plugin: %s", plugin_name)
        else:
            # Legacy remote_url approach
            if config.remote_url is None:
                raise ValueError(
                    "remote_url must be provided when not using plugin_name"
                )
            self.remote_url = config.remote_url

        self.local_cpu_backend = local_cpu_backend

        self.loop = loop
        self.config = config
        self.metadata = metadata

        # Re-establish connection only when the connection
        # has been lost for 10 secs
        self.connection: Optional[RemoteConnector] = None
        self.min_reconnect_interval = 10
        self.failure_time = -1000000.0
        self.init_connection()

        assert config.remote_serde is not None
        self.serializer, self.deserializer = CreateSerde(
            config.remote_serde, metadata, config
        )

        # Precompute MLA mode status
        self._mla_worker_id_as0_mode = (
            config.get_extra_config_value(
                "remote_enable_mla_worker_id_as0", metadata.use_mla
            )
            and metadata.use_mla
            and metadata.world_size > 1
            and metadata.worker_id != 0
        )
        logger.info("metadata=%s", metadata)
        logger.info(
            "Connected to remote storage at %s, remote_mla_worker_id_as_0 mode: %s",
            config.remote_url,
            self._mla_worker_id_as0_mode,
        )

        # TODO(Jiayi): If we want to have cache admission policies,
        # we must make decision (whether to send or not) at the local side

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        # NOTE: Health monitoring is now handled at the LMCacheEngine level
        # through HealthMonitor. RemoteBackend no longer manages its own
        # health monitoring. The HealthMonitor in LMCacheEngine will
        # register RemoteBackendHealthCheck for each RemoteBackend.

        self._get_blocking_failed_count = 0
        self._put_failed_count = 0

        self._setup_metrics()

    def _setup_metrics(self) -> None:
        prometheus_logger = PrometheusLogger.GetOrCreate(
            self.metadata,
            config=self.config,
        )
        prometheus_logger.remote_put_task_num.set_function(lambda: len(self.put_tasks))
        prometheus_logger.get_blocking_failed_count.set_function(
            lambda: self._get_blocking_failed_count
        )
        prometheus_logger.put_failed_count.set_function(lambda: self._put_failed_count)

    def __str__(self):
        return self.__class__.__name__

    def init_connection(self):
        # Initialize connection
        if self.connection is not None:
            return
        if (time.time() - self.failure_time) < self.min_reconnect_interval:
            logger.warning(
                "Connection will not be re-established yet "
                "since it has not been long enough since "
                "the last failure"
            )
            return
        try:
            # Determine the URL to use for connection
            if self.plugin_name is not None:
                # Using plugin-based approach
                # Create a virtual URL that the adapter can recognize
                url = f"plugin://{self.plugin_name}"
                logger.info("Creating connector for plugin: %s", self.plugin_name)
            else:
                # Legacy remote_url approach
                if self.config.remote_url is None:
                    raise ValueError(
                        "remote_url must be provided when not using plugin_name"
                    )
                url = self.config.remote_url

            self.connection = CreateConnector(
                url,
                self.loop,
                self.local_cpu_backend,
                self.config,
                self.metadata,
                plugin_name=self.plugin_name,
            )
            logger.info("Connection initialized/re-established at %s", url)
        except IrrecoverableException:
            logger.error("Irrecoverable error during connection initialization")
            raise
        except Exception as e:
            with self.lock:
                self.failure_time = time.time()
            logger.warning("Failed to initialize/re-establish remote connection: %s", e)
            self.connection = None

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        if self.connection is None:
            logger.warning("Connection is None in contains, returning False")
            return False

        # For MLA worker id as 0 mode, use worker_id 0
        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)

        try:
            if self.config.extra_config is not None and self.config.extra_config.get(
                "use_exists_sync", False
            ):
                return self.connection.exists_sync(key)
            else:
                future = asyncio.run_coroutine_threadsafe(
                    self.connection.exists(key), self.loop
                )
                res = future.result()
                return res
        except Exception as e:
            logger.warning("Remote connection failed in contains: %s", e)
            logger.warning("Returning False")
            return False

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if self.connection is None:
            logger.warning("Connection is None in batched_contains, returning 0")
            return 0

        if not self.connection.support_batched_contains():
            return super().batched_contains(keys, pin)

        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]

        try:
            return self.connection.batched_contains(keys)
        except Exception as e:
            logger.warning("Remote connection failed in batched_contains: %s", e)
            return 0

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.lock:
            return key in self.put_tasks

    def put_callback(self, future: Future, key: CacheEngineKey):
        with self.lock:
            self.put_tasks.discard(key)
        try:
            future.result()
        except Exception as e:
            self._put_failed_count += 1
            logger.error("Put task failed for key %s: %s", key, e)

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Future:
        """
        Submit a put task to store KV cache to remote storage asynchronously.

        :param on_complete_callback: Optional callback invoked after the remote
            write completes. Callback exceptions are caught and logged.
        """

        def create_immediate_empty_future() -> Future:
            f: Future = Future()
            f.set_result(None)
            return f

        if self.connection is None:
            logger.warning("Connection is None in submit_put_task, returning None")
            return create_immediate_empty_future()

        # If MLA worker id as 0 mode is enabled, skip put tasks
        if self._mla_worker_id_as0_mode:
            return create_immediate_empty_future()

        if self.exists_in_put_tasks(key):
            return create_immediate_empty_future()

        memory_obj.ref_count_up()

        with self.lock:
            self.put_tasks.add(key)

        is_makv = getattr(getattr(self, "config", None), "remote_serde", None) == "makv"
        if is_makv:
            compressed_memory_obj = self.serializer.serialize(
                memory_obj,
                transfer_spec=None,
                key=key,
            )
        else:
            compressed_memory_obj = self.serializer.serialize(memory_obj)
        memory_obj.ref_count_down()

        def put_done_callback(f: Future) -> None:
            self.put_callback(f, key)
            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning("on_complete_callback failed for key %s: %s", key, e)

        # NOTE: No need to do error handling here
        # since the `future` is never waited
        future = asyncio.run_coroutine_threadsafe(
            self.connection.put(key, compressed_memory_obj), self.loop
        )
        future.add_done_callback(put_done_callback)
        return future

    def report_precision_risk(
        self, key: CacheEngineKey, signal: Any
    ) -> dict[str, Any]:
        """Forward one MaKV runtime risk signal without affecting other serdes.

        The method is intentionally a no-op response for non-MaKV backends so
        the runtime vLLM hook can share a connector boundary without adding a
        risk protocol to ``naive`` or ``cachegen``.
        """
        if self.config.remote_serde != "makv":
            return {"accepted": False, "reason": "remote_serde_is_not_makv"}
        if self.connection is None:
            return {"accepted": False, "reason": "remote_connection_unavailable"}

        report = getattr(self.connection, "report_precision_risk", None)
        if not callable(report):
            return {"accepted": False, "reason": "connector_does_not_support_risk"}
        future = asyncio.run_coroutine_threadsafe(report(key, signal), self.loop)
        try:
            result = future.result(self.config.blocking_timeout_secs)
        except Exception:
            future.cancel()
            raise
        if isinstance(result, dict):
            return result
        return {"accepted": bool(result)}

    def batched_put_callback(self, future: Future, keys: List[CacheEngineKey]):
        """
        Callback function for batched put tasks.
        """
        with self.lock:
            self.put_tasks.difference_update(keys)

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Submit batched put tasks to store KV caches to remote storage.

        :param on_complete_callback: Optional callback invoked once per key
            after that key's write completes (not once per batch).
        """
        if self.connection is None:
            logger.warning(
                "Connection is None in batched_submit_put_task, returning None"
            )
            return
        if self.connection.support_batched_put():
            if self._mla_worker_id_as0_mode:
                return

            # First, increment reference counts for all objects
            for memory_obj in memory_objs:
                memory_obj.ref_count_up()

            compressed_memory_objs = []
            serialized_keys: list[CacheEngineKey] = []
            try:
                is_makv = (
                    getattr(getattr(self, "config", None), "remote_serde", None)
                    == "makv"
                )
                chunk_starts = (
                    None
                    if not is_makv or transfer_spec is None
                    else transfer_spec.get("chunk_starts")
                )
                chunk_ends = (
                    None
                    if not is_makv or transfer_spec is None
                    else transfer_spec.get("chunk_ends")
                )
                for idx, (batch_key, memory_obj) in enumerate(
                    zip(keys, memory_objs, strict=False)
                ):
                    if is_makv:
                        item_transfer_spec = dict(transfer_spec or {})
                        if chunk_starts is not None and idx < len(chunk_starts):
                            item_transfer_spec["chunk_start"] = int(chunk_starts[idx])
                        if chunk_ends is not None and idx < len(chunk_ends):
                            item_transfer_spec["chunk_end"] = int(chunk_ends[idx])
                        compressed_memory_objs.append(
                            self.serializer.serialize(
                                memory_obj,
                                transfer_spec=item_transfer_spec,
                                key=batch_key,
                            )
                        )
                    else:
                        compressed_memory_objs.append(
                            self.serializer.serialize(memory_obj)
                        )
                    serialized_keys.append(batch_key)
            finally:
                # Always decrement reference counts for all objects,
                # regardless of whether serialization succeeded or failed
                for memory_obj in memory_objs:
                    memory_obj.ref_count_down()

            def batched_done_callback(f: Future) -> None:
                self.batched_put_callback(f, serialized_keys)
                # Invoke per-key callback for each key in the batch
                if on_complete_callback is not None:
                    for key in serialized_keys:
                        try:
                            on_complete_callback(key)
                        except Exception as e:
                            logger.warning(
                                "on_complete_callback failed for key %s: %s", key, e
                            )

            future = asyncio.run_coroutine_threadsafe(
                self.connection.batched_put(serialized_keys, compressed_memory_objs),  # type: ignore
                self.loop,
            )
            future.add_done_callback(batched_done_callback)
        else:
            for key, memory_obj in zip(keys, memory_objs, strict=False):
                self.submit_put_task(
                    key, memory_obj, on_complete_callback=on_complete_callback
                )

    @_lmcache_nvtx_annotate
    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Blocking get function.
        """
        # Check if local_cpu_backend is available (required for memory allocation)
        if self.local_cpu_backend is None:
            logger.warning(
                "local_cpu_backend is None in get_blocking "
                "(likely scheduler role), returning None"
            )
            return None

        if self.connection is None:
            logger.warning("Connection is None in get_blocking, returning None")
            return None
        # For MLA worker id as 0 mode, use worker_id 0
        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)
        t1 = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(self.connection.get(key), self.loop)

        try:
            memory_obj = future.result(self.config.blocking_timeout_secs)
        except Exception as e:
            if isinstance(e, TimeoutError):
                logger.warning("get blocking timeout, trigger cancel the future task")
                future.cancel()
            logger.warning("Error occurred in get_blocking: %s, return None", e)
            memory_obj = None

        t2 = time.perf_counter()
        self.stats_monitor.update_interval_remote_time_to_get_sync((t2 - t1) * 1000)
        if memory_obj is None:
            self._get_blocking_failed_count += 1
            return None
        deserialize_started = time.perf_counter()
        try:
            decompressed_memory_obj = self.deserializer.deserialize(memory_obj)
        except Exception as error:
            logger.warning("Remote object deserialization failed: %s", error)
            self._get_blocking_failed_count += 1
            return None
        t3 = time.perf_counter()
        self._record_makv_get_timing(
            [memory_obj], (t3 - deserialize_started) * 1000
        )
        logger.debug(
            "Get takes %.6f msec, deserialization takes %.6f msec",
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
        )
        return decompressed_memory_obj

    @property
    def get_blocking_failed_count(self):
        return self._get_blocking_failed_count

    @property
    def put_failed_count(self):
        return self._put_failed_count

    def _record_makv_get_timing(
        self, memory_objs: Sequence[Optional[MemoryObj]], deserialize_ms: float
    ) -> None:
        """Attach optional MaKV transport/manager timing to this retrieval.

        Object attributes avoid an eager MaKV import in the common backend, so
        non-MaKV serde modes keep their existing import and execution paths.
        """
        if getattr(getattr(self, "config", None), "remote_serde", None) != "makv":
            return
        if not any(
            isinstance(getattr(memory_obj, "makv_server_timing", None), dict)
            or isinstance(getattr(memory_obj, "makv_transport_timing", None), dict)
            for memory_obj in memory_objs
            if memory_obj is not None
        ):
            return
        retrieve_stats = self.stats_monitor.get_current_retrieve_stats()
        if retrieve_stats is None:
            return
        timing = retrieve_stats.detailed_metrics.setdefault(
            "makv_latency",
            {
                "tcp_batches": 0,
                "tcp_connect_ms": 0.0,
                "tcp_send_ms": 0.0,
                "tcp_first_response_ms": 0.0,
                "tcp_receive_ms": 0.0,
                "tcp_total_ms": 0.0,
                "tcp_download_bytes": 0,
                "manager_requests": 0,
                "manager_hot_cache_hits": 0,
                "manager_hot_cache_ms": 0.0,
                "manager_storage_ms": 0.0,
                "manager_validate_ms": 0.0,
                "manager_total_ms": 0.0,
                "manager_hot_cache_ms_max": 0.0,
                "manager_storage_ms_max": 0.0,
                "manager_validate_ms_max": 0.0,
                "manager_total_ms_max": 0.0,
                "manager_batch_storage_ms_max": 0.0,
                "manager_batch_validate_ms_max": 0.0,
                "manager_batch_total_ms_max": 0.0,
                "deserialize_ms": 0.0,
            },
        )
        timing["deserialize_ms"] += deserialize_ms
        for memory_obj in memory_objs:
            if memory_obj is None:
                continue
            transport = getattr(memory_obj, "makv_transport_timing", None)
            if isinstance(transport, dict):
                timing["tcp_batches"] += 1
                timing["tcp_connect_ms"] += float(transport.get("connect_ms", 0.0))
                timing["tcp_send_ms"] += float(transport.get("send_ms", 0.0))
                timing["tcp_first_response_ms"] += float(
                    transport.get("first_response_ms", 0.0)
                )
                timing["tcp_receive_ms"] += float(transport.get("receive_ms", 0.0))
                timing["tcp_total_ms"] += float(transport.get("total_ms", 0.0))
                delattr(memory_obj, "makv_transport_timing")
            timing["tcp_download_bytes"] += len(memory_obj.byte_array)
            server = getattr(memory_obj, "makv_server_timing", None)
            if isinstance(server, dict):
                hot_cache_ms = float(server.get("hot_cache_ms", 0.0))
                storage_ms = float(server.get("storage_ms", 0.0))
                validate_ms = float(server.get("validate_ms", 0.0))
                total_ms = float(server.get("total_ms", 0.0))
                batch_storage_ms = float(server.get("batch_storage_ms", 0.0))
                batch_validate_ms = float(server.get("batch_validate_ms", 0.0))
                batch_total_ms = float(server.get("batch_total_ms", 0.0))
                timing["manager_requests"] += 1
                timing["manager_hot_cache_hits"] += int(
                    bool(server.get("hot_cache_hit", False))
                )
                timing["manager_hot_cache_ms"] += hot_cache_ms
                timing["manager_storage_ms"] += storage_ms
                timing["manager_validate_ms"] += validate_ms
                timing["manager_total_ms"] += total_ms
                timing["manager_hot_cache_ms_max"] = max(
                    timing["manager_hot_cache_ms_max"], hot_cache_ms
                )
                timing["manager_storage_ms_max"] = max(
                    timing["manager_storage_ms_max"], storage_ms
                )
                timing["manager_validate_ms_max"] = max(
                    timing["manager_validate_ms_max"], validate_ms
                )
                timing["manager_total_ms_max"] = max(
                    timing["manager_total_ms_max"], total_ms
                )
                timing["manager_batch_storage_ms_max"] = max(
                    timing["manager_batch_storage_ms_max"], batch_storage_ms
                )
                timing["manager_batch_validate_ms_max"] = max(
                    timing["manager_batch_validate_ms_max"], batch_validate_ms
                )
                timing["manager_batch_total_ms_max"] = max(
                    timing["manager_batch_total_ms_max"], batch_total_ms
                )

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        # Check if local_cpu_backend is available (required for memory allocation)
        if self.local_cpu_backend is None:
            logger.warning(
                "local_cpu_backend is None in batched_get_blocking "
                "(likely scheduler role), returning None list"
            )
            return [None] * len(keys)

        if self.connection is None:
            logger.warning("Connection is None in batched_get_blocking, returning None")
            return [None] * len(keys)

        # For MLA worker id as 0 mode, use worker_id 0
        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]

        t1 = time.perf_counter()
        # batched get
        if self.connection.support_batched_get():
            future = asyncio.run_coroutine_threadsafe(
                self.connection.batched_get(keys), self.loop
            )
            try:
                memory_objs = future.result(self.config.blocking_timeout_secs)
            except Exception as e:
                if isinstance(e, TimeoutError):
                    logger.warning(
                        "batched get blocking timeout, trigger cancel the future task"
                    )
                    future.cancel()
                else:
                    logger.warning(
                        "Error occurred in batched_get_blocking: %s, "
                        "returning None list",
                        e,
                    )
                memory_objs = [None] * len(keys)
        else:
            remote_backend_individual_get_stats: dict[
                CacheEngineKey, dict[str, float]
            ] = {}
            retrieve_stats = self.stats_monitor.get_current_retrieve_stats()
            if retrieve_stats is not None:
                retrieve_stats.detailed_metrics[
                    "remote_backend_individual_get_stats"
                ] = remote_backend_individual_get_stats

            futures = [
                asyncio.run_coroutine_threadsafe(self.connection.get(key), self.loop)
                for key in keys
            ]
            memory_objs = []
            failed = False
            for fut in futures:
                if not failed:
                    try:
                        memory_obj = fut.result(self.config.blocking_timeout_secs)
                    except Exception as e:
                        failed = True
                        if isinstance(e, TimeoutError):
                            logger.warning(
                                "get blocking timeout, trigger cancel the future task"
                            )
                            fut.cancel()
                        else:
                            logger.warning(
                                "Error occurred in get_blocking: %s, returning None", e
                            )
                        memory_obj = None
                    memory_objs.append(memory_obj)
                else:
                    memory_objs.append(None)
                    fut.cancel()

        t2 = time.perf_counter()
        duration = t2 - t1
        self.stats_monitor.update_interval_remote_time_to_get_sync(duration * 1000)

        retrieve_stats = self.stats_monitor.get_current_retrieve_stats()
        if retrieve_stats is not None:
            retrieve_stats.detailed_metrics[
                "remote_backend_batched_get_blocking_time"
            ] = (
                retrieve_stats.detailed_metrics.get(
                    "remote_backend_batched_get_blocking_time", 0.0
                )
                + duration
            )
        deserialize_started = time.perf_counter()
        decompressed_memory_objs: list[Optional[MemoryObj]] = []
        error_happened = False
        for memory_obj in memory_objs:
            if memory_obj is None:
                error_happened = True
                decompressed_memory_objs.append(None)
            else:
                try:
                    decompressed_memory_objs.append(
                        self.deserializer.deserialize(memory_obj)
                    )
                except Exception as error:
                    logger.warning("Remote object deserialization failed: %s", error)
                    error_happened = True
                    decompressed_memory_objs.append(None)
        if error_happened:
            self._get_blocking_failed_count += 1

        self._record_makv_get_timing(
            memory_objs, (time.perf_counter() - deserialize_started) * 1000
        )

        assert len(decompressed_memory_objs) == len(keys), (
            f"keys length: {len(keys)}, "
            f"decompressed memory objs length: {len(decompressed_memory_objs)}"
        )
        return decompressed_memory_objs

    def batched_get_streaming_blocking(
        self, keys: List[CacheEngineKey]
    ) -> Optional[Iterator[Optional[MemoryObj]]]:
        """Return an ordered MaKV iterator that does not materialize GET_BATCH.

        This is intentionally an opt-in connector capability rather than a new
        abstract backend requirement.  Existing remote serde modes continue to
        use ``batched_get_blocking`` unchanged.
        """
        # MaKV streaming returns a quantized MemoryObj that the GPU connector
        # restores directly. Unlike the ordinary blocking path, it does not
        # need a LocalCPUBackend allocator.
        if self.connection is None:
            return None
        if self.config.remote_serde != "makv":
            return None
        support = getattr(self.connection, "support_batched_get_streaming", None)
        stream_get = getattr(self.connection, "batched_get_streaming", None)
        if not callable(support) or not support() or not callable(stream_get):
            return None
        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]
        return self._iter_streaming_makv_get(keys, stream_get)

    def _iter_streaming_makv_get(
        self,
        keys: List[CacheEngineKey],
        stream_get: Callable[[List[CacheEngineKey]], Any],
    ) -> Iterator[Optional[MemoryObj]]:
        """Bridge the connector's async generator to the cache-engine thread."""
        started = time.perf_counter()
        raw_memory_objs: list[Optional[MemoryObj]] = []
        deserialize_ms = 0.0
        failed = False
        stream = stream_get(keys)
        next_index = 0
        try:
            while next_index < len(keys):
                future = asyncio.run_coroutine_threadsafe(stream.__anext__(), self.loop)
                try:
                    index, memory_obj = future.result(self.config.blocking_timeout_secs)
                except StopAsyncIteration:
                    logger.warning("MaKV streaming GET ended before all keys arrived")
                    failed = True
                    yield None
                    break
                except Exception as error:
                    if isinstance(error, TimeoutError):
                        future.cancel()
                    logger.warning("MaKV streaming GET failed: %s", error)
                    failed = True
                    yield None
                    break
                if int(index) != next_index:
                    logger.warning(
                        "MaKV streaming GET response order mismatch: "
                        "expected %d, got %s",
                        next_index,
                        index,
                    )
                    failed = True
                    yield None
                    break

                raw_memory_objs.append(memory_obj)
                if memory_obj is None:
                    failed = True
                    yield None
                    break
                deserialize_started = time.perf_counter()
                try:
                    result = self.deserializer.deserialize(memory_obj)
                except Exception as error:
                    logger.warning("Remote object deserialization failed: %s", error)
                    failed = True
                    result = None
                deserialize_ms += (time.perf_counter() - deserialize_started) * 1000
                if result is not None:
                    # The GPU restore timeline begins only after the complete
                    # framed object has been deserialized and validated.
                    result.makv_restore_ready_ns = time.perf_counter_ns()
                yield result
                if result is None:
                    break
                next_index += 1
        finally:
            try:
                close_future = asyncio.run_coroutine_threadsafe(
                    stream.aclose(), self.loop
                )
                close_future.result(self.config.blocking_timeout_secs)
            except Exception as error:
                logger.debug("MaKV streaming GET close failed: %s", error)
            duration = time.perf_counter() - started
            self.stats_monitor.update_interval_remote_time_to_get_sync(duration * 1000)
            retrieve_stats = self.stats_monitor.get_current_retrieve_stats()
            if retrieve_stats is not None:
                retrieve_stats.detailed_metrics[
                    "remote_backend_batched_get_blocking_time"
                ] = (
                    retrieve_stats.detailed_metrics.get(
                        "remote_backend_batched_get_blocking_time", 0.0
                    )
                    + duration
                )
            self._record_makv_get_timing(raw_memory_objs, deserialize_ms)
            if failed:
                self._get_blocking_failed_count += 1

    async def support_batched_async_contains(self) -> bool:
        return (
            self.connection is not None
            and self.connection.support_batched_async_contains()
        )

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if self.connection is None:
            logger.warning("Connection is None in batched_async_contains, returning 0")
            return 0
        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]

        try:
            assert self.connection.support_batched_async_contains(), (
                f"Connector {self.connection} does not support batched async contains"
            )
            # warning, this timeout will not actually stop the
            # scheduler from waiting for the result
            return await asyncio.wait_for(
                self.connection.batched_async_contains(lookup_id, keys, pin),
                self.config.blocking_timeout_secs,
            )
        except asyncio.TimeoutError:
            logger.warning("batched_async_contains timed out")
            return 0
        except Exception as e:
            logger.warning("Error occurred in batched_async_contains: %s", e)
            return 0

    async def support_batched_get_non_blocking(self) -> bool:
        return (
            self.connection is not None
            and self.connection.support_batched_get_non_blocking()
        )

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> List[MemoryObj]:
        # Check if local_cpu_backend is available (required for memory allocation)
        if self.local_cpu_backend is None:
            logger.warning(
                "local_cpu_backend is None in batched_get_non_blocking "
                "(likely scheduler role), returning empty list"
            )
            return []

        if self.connection is None:
            logger.warning(
                "Connection is None in batched_get_non_blocking, returning empty list"
            )
            return []
        try:
            # warning, this timeout will not actually stop the
            # scheduler from waiting for the result
            return await asyncio.wait_for(
                self.connection.batched_get_non_blocking(lookup_id, keys),
                self.config.blocking_timeout_secs,
            )
        except asyncio.TimeoutError:
            logger.warning("batched_get_non_blocking timed out")
            return []
        except Exception as e:
            logger.warning("Error occurred in batched_get_non_blocking: %s", e)
            return []

    def pin(self, key: CacheEngineKey) -> bool:
        logger.debug(
            "Remote backend does not support pin. "
            "This method is a no-op and will return True."
        )
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        logger.debug(
            "Remote backend does not support unpin. "
            "This method is a no-op and will return True."
        )
        return True

    def remove(self, key, force=True):
        if self.connection is None:
            logger.warning("Connection is None in remove, returning False")
            return False

        try:
            return self.connection.remove_sync(key)
        except Exception as e:
            logger.exception(
                "Failed to remove key %s from remote backend, error: %s", key, e
            )
            return False

    def get_allocator_backend(self):
        assert self.local_cpu_backend is not None, (
            "local_cpu_backend is required for get_allocator_backend, "
            "should not be called in scheduler role"
        )
        return self.local_cpu_backend

    def close(self):
        try:
            assert self.connection is not None
            future = asyncio.run_coroutine_threadsafe(
                self.connection.close(), self.loop
            )
            future.result()
            logger.info("Remote backend closed.")
        except Exception as e:
            logger.warning("Error occurred when closing remote connection: %s", e)
