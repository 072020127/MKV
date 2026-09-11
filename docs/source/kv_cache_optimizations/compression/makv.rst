MaKV
====

MaKV adds an importance-aware remote KV serde mode alongside ``naive`` and
``cachegen``:

.. code-block:: yaml

   remote_serde: makv
   remote_url: "makv://127.0.0.1:65432"
   extra_config:
     makv_storage_backend: "file"
     makv_storage_url: "file:///tmp/lmcache-makv"
     makv_storage_namespace: "lmcache:makv:"
     makv_mooncake_config: ""
     makv_bucket_ratios: [0.20, 0.30, 0.50]
     makv_bucket_bits: [16, 8, 4]
     makv_precision_scheme: shared
     makv_importance_layout: token
     makv_quant_granularity: per_token_head
     makv_scale_dtype: float16
     makv_protect_prefix_tokens: 4
     makv_protect_tail_tokens: 16
     makv_dequant_backend: cuda
     makv_require_cuda_dequant: true
     makv_fallback: naive
     makv_enable_checksum: true
     makv_entropy_codec: none
     makv_entropy_backend: auto
     makv_entropy_require_cuda: false
     makv_residual_dtype: none
     makv_risk_upgrade_threshold: 0.80
     makv_risk_upgrade_policy: next
     makv_streaming_restore: true

For the frozen ScoutRank four-tier block plan, use four buckets and opt in to
the shadow policy explicitly:

.. code-block:: yaml

   extra_config:
     makv_bucket_ratios: [0.10, 0.10, 0.60, 0.20]
     makv_bucket_bits: [16, 8, 4, 2]
     makv_allow_scoutrank_shadow_plan: true

K/V-separated three-tier scheme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To assign different widths to K and V while retaining one request-level score
tier per token, select the opt-in three-tier scheme:

.. code-block:: yaml

   extra_config:
     makv_precision_scheme: kv_separate_3tier
     makv_bucket_ratios: [0.20, 0.30, 0.50]
     makv_bucket_bits: [8, 4, 2]

The highest-scoring 20% of tokens use ``K8V4``, the next 30% use ``K4V2``,
and the remaining 50% use ``K2V2``. ``[T]`` importance is ranked once over
the complete request, then broadcast to each layer and expanded into separate
K/V positions. ``[L,2,T]`` importance is ranked independently for each layer
and K/V plane. In the latter case the physical K and V buckets are still
independent, so a token can receive different K and V tiers when its scores
differ.

This scheme has no 16-bit bucket; prefix/tail protection promotes positions to
the highest tier, ``K8V4``. The remote manager must be started with the same
scheme and bucket bits. The default ``shared`` scheme and the frozen ScoutRank
four-tier plan are unchanged.

K/V-separated four-tier scheme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To retain an original-precision bucket while using asymmetric K/V widths,
select the four-tier scheme:

.. code-block:: yaml

   extra_config:
     makv_precision_scheme: kv_separate_4tier
     makv_bucket_ratios: [0.10, 0.20, 0.50, 0.20]
     makv_bucket_bits: [16, 8, 4, 2]

The score-ranked tiers map to ``K16V16`` (top 10%), ``K8V4`` (next 20%),
``K4V2`` (next 50%), and ``K2V2`` (bottom 20%). The 16-bit bucket preserves
the original FP16/BF16 values and does not create scales. This is the default
ratio for ``kv_separate_4tier``; the existing three-tier and ``shared``
defaults are unchanged. Ratio sweeps can use
``tests/run_longbench_makv_bucket_sweep.sh`` with the same four physical bits.

Architecture
------------

MaKV uses a single PUT compression location: the independent Remote Manager.
The client computes a deterministic ``MaKVQuantPlan`` from prompt-token
importance and uploads:

- raw FP16/BF16 KV bytes
- plan metadata
- cache key and layout metadata
- protocol version and checksum

The remote MaKV manager is the only place that quantizes data during PUT. It
validates the plan, canonicalizes the KV layout to ``[L, 2, T, H, D]``, and
stores a self-describing ``MaKVObject`` with mixed 16-bit / INT8 / INT4 / INT2
payloads plus per-token-per-head scales.

The manager persistence layer is selected independently from the MaKV object
format. ``file://`` uses atomic local files, ``redis://``/``rediss://`` uses
namespaced Redis binary values, and ``mooncake://`` uses the optional
Mooncake Python SDK. Changing the adapter does not move quantization to the
client or change the LMCache ``makv://`` network protocol.

GET returns the stored ``MaKVObject`` bytes without remote dequantization. The
client deserializer keeps the payload compressed until restore time.

The sender never quantizes or entropy-codes the KV payload. The manager performs
the configured quantization, optional CacheGen arithmetic coding, and optional
residual generation before atomically storing the object. Set
``makv_residual_dtype`` to ``float16`` or ``float32`` when a later risk-guided
precision upgrade is required. ``none`` deliberately disables that upgrade
path. Missing importance uses the configured explicit ``naive`` fallback; it
never silently turns all tokens into INT4.

Importance formats
------------------

MaKV currently accepts two importance layouts:

- ``[T]`` / ``token``: one score per prompt token, broadcast to every layer and
  K/V plane.
- ``[L, 2, T]`` / ``layer_kv_token``: independent scores per layer, K/V plane,
  and token.

The bucketing pass is request-global before chunk slicing. Tokens are never
re-ranked independently inside each LMCache chunk. Supply importance through
``transfer_spec["makv_importance"]`` and optionally
``transfer_spec["makv_importance_layout"]``; request configuration also accepts
``lmcache.makv_importance`` and ``lmcache.makv_importance_layout``.

Frozen ScoutRank block plan
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``k2_risk_monotone_four_tier_v1`` produces a final block precision plan, not
token importance scores. Pass it through ``lmcache.makv_precision_plan``. Do
not broadcast its block risk and run the ratio bucketer again: that would lose
the frozen quota and asymmetric K/V precision. The mapping is:

- ``BF16`` -> K16 / V16
- ``K8V4`` -> K8 / V4
- ``K4V2`` -> K4 / V2
- ``K2V2`` -> K2 / V2

Use ``VllmLmcachePlanAdapter.to_makv_request_configs`` to build the request
configuration from ``MonotoneFourTierPlan`` and the complete prompt token IDs.
LMCache independently hashes the actual store tokens using the scorer's
canonical SHA-256 encoding before accepting the plan. It also validates the
strategy and shadow status, 32-token block geometry, complete token coverage,
precision vector, cumulative 10/20/80 quota, byte totals, operational
``plan_hash``, and ``repeat_exact=true`` before slicing the request-level plan
into LMCache chunks.

The strategy remains shadow-only. ``makv_allow_scoutrank_shadow_plan`` defaults
to ``false`` and must be enabled deliberately. A rejected plan follows
``makv_fallback`` and is never converted to zero importance or re-ranked by the
legacy token bucketer.

Bucketing rules
---------------

- Ratios must be non-negative and sum to 1.
- Supported bucket bit-widths are ``16``, ``8``, ``4``, and ``2``.
- ``kv_separate_3tier`` requires physical buckets ``[8, 4, 2]`` and maps them
  to the asymmetric pairs ``K8V4``, ``K4V2``, and ``K2V2``.
- ``kv_separate_4tier`` requires physical buckets ``[16, 8, 4, 2]`` and maps
  them to ``K16V16``, ``K8V4``, ``K4V2``, and ``K2V2``.
- Ties are broken deterministically by original token index.
- ``NaN`` / ``+Inf`` / ``-Inf`` scores are promoted to the highest-precision
  bucket.
- ``makv_protect_prefix_tokens`` and ``makv_protect_tail_tokens`` force tokens
  into the highest-precision bucket.
- Missing or invalid importance can fall back to ``naive`` when
  ``makv_fallback: naive`` is configured.

Object format
-------------

MaKV stores one self-describing binary object per cache key. The current format
uses:

- magic ``MAKV`` for stored objects
- magic ``MKVP`` for client PUT envelopes
- JSON metadata with a payload table
- CRC32 checksum over the complete object with the checksum field zeroed during
  calculation
- 64-byte-aligned payload offsets for one contiguous pinned-to-GPU transfer

The alignment is a layout optimization, not a new protocol version: the
payload table remains authoritative, and objects written by older v1 encoders
are accepted with a per-segment copy fallback.

The stored metadata includes:

- original dtype, shape, and stride information
- chunk start and chunk length
- bucket definitions and bucket assignments
- model and parallel fingerprints
- quantization granularity and scale dtype

Runtime behavior
----------------

Client PUT
~~~~~~~~~~

``MaKVSerializer`` builds the quantization plan but does not quantize KV.
``makv_client_quantize_calls`` should remain zero.

Remote PUT
~~~~~~~~~~

``MaKVNetworkConnector`` sends the raw-KV envelope over a length-delimited TCP
protocol to an independent ``MaKVRemoteManager`` process. Only that process
imports and invokes ``quantize_canonical_kv``. The manager selects its storage
adapter with ``--storage-backend`` and ``--storage-url``; the client only
needs matching metadata in ``extra_config``.

Client GET
~~~~~~~~~~

``MaKVDeserializer`` returns either:

- ``MaKVQuantizedMemoryObj`` for quantized objects
- ``TensorMemoryObj`` for explicit naive fallbacks

The restore hook is inside the V2/V3 vLLM GPU connectors. Quantized payloads,
scales, and positions are copied asynchronously to the target GPU, then
``lmcache_makv::dequantize_scatter_paged_out`` writes FP16/BF16 values directly
into the final paged K/V cache using ``slot_mapping``. The production path does
not allocate a full ``[2, L, T, H * D]`` restored tensor. The contiguous CUDA
op and PyTorch reference implementation remain correctness-test paths only.

For a normal multi-chunk retrieve, the network connector uses one ``GET_BATCH``
TCP connection. With ``makv_streaming_restore: true`` (the default), it
negotiates ``stream_v1``: the manager keeps a bounded number of single-key
reads in flight and sends each complete MaKV object as soon as its own read and
validation finish. Every response frame is drained before the next frame is
exposed, so client-side H2D, optional CUDA arithmetic decode, and paged scatter
can overlap later storage/network work without accumulating the complete batch.
The client only synchronizes the GPU load stream once after the hit prefix is
complete; it never materializes a full restored KV tensor.

``blob_v1`` remains the default for ordinary non-streaming ``batched_get`` and
is retained for compatibility. Older managers that do not recognize
``stream_v1`` fall back to their legacy per-object response behavior. The
manager option ``--batch-stream-prefetch-depth`` (default ``4``) bounds the
number of complete objects held while a slow response is being sent.

Set ``makv_streaming_restore: false`` to retain the previous whole-batch read
path for A/B measurement or diagnosis. Missing, truncated, malformed, or
fallback objects stop the pipeline at the same contiguous prefix boundary as a
normal cache miss. The manager can keep a bounded in-process hot-object LRU
cache; ``--memory-cache-gb 2`` is used by the LongBench helper and is especially
useful when the durable ``file://`` adapter is on a slower filesystem.

The legacy response path buffers batch writes up to 64 MiB before awaiting
transport backpressure. ``stream_v1`` instead awaits transport backpressure
after each object so its bounded prefetch guarantee also applies to slow
clients.

For ``blob_v1``, Redis and Mooncake use the adapter's native batch read
(``MGET`` for Redis and ``batch_get_buffer`` for Mooncake). ``stream_v1`` uses
bounded single-key reads so the first object can be sent before later reads
finish. File storage retains concurrent single file reads. The manager still
validates every returned object's structure and payload table before the client
can consume it.

The optional manager flag ``--trust-validated-objects`` skips repeat CRC scans
for objects that this manager has already validated and successfully stored,
or validated during an earlier cold read. Length, protocol, metadata, offset,
and overlap checks still run on every GET. Keep the flag disabled when the
storage namespace can be modified by processes other than the MaKV manager;
the default remains full checksum verification.

Large TCP GET payloads are received directly into pinned host memory when the
PyTorch runtime supports pinned allocations. Set
``makv_pinned_receive: false`` to disable this optimization or adjust
``makv_pinned_receive_min_bytes`` for small objects. CPU-only runtimes fall
back to bytearray reception automatically.

For aligned objects, paged restore copies the complete serialized blob once to
the target GPU and creates device byte views for positions and quantized
payloads. The existing CUDA operator still launches at most one kernel per
non-empty precision bucket and writes directly to the final paged cache. Scale
arrays are normalized to the CUDA ABI's FP32 representation; they are much
smaller than the payload and do not require another full-object copy.

CacheGen arithmetic entropy coding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MaKV can optionally reuse CacheGen's existing CUDA arithmetic coder for the
INT8/INT4/INT2 payloads. It is disabled by default so ``entropy_codec: none``
keeps the original MaKV object format and latency path unchanged. Enable it on
both the independent manager and the LMCache client configuration:

.. code-block:: yaml

   extra_config:
     makv_entropy_codec: cachegen_arithmetic
     makv_entropy_backend: cuda
     makv_entropy_require_cuda: true

The corresponding manager command is:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=7 ../.venv/bin/python -m \
     lmcache.v1.storage_backend.makv_remote.server \
     --listen 0.0.0.0:65432 \
     --storage-backend redis \
     --storage-url redis://127.0.0.1:6379/0 \
     --entropy-codec cachegen_arithmetic \
     --entropy-backend cuda \
     --entropy-require-cuda

The remote manager calls the arithmetic encoder during PUT and the client still
uploads raw FP16/BF16 KV and its plan. The encoder uses the existing
``calculate_cdf``/``encode_fast_new`` CacheGen CUDA APIs, splitting each
quantized stream into bounded 256-symbol streams. INT8 is represented by two
small symbol planes; INT4 and INT2 use one plane. CDFs, stream lengths, and
encoded bytes remain inside the self-describing MaKV blob. On GET, the client
uses ``decode_fast_prefsum`` on the target GPU and feeds the reconstructed
compact integer payload directly to the MaKV dequantize-scatter kernel; it does
not create a full floating-point restore tensor before paged restore.

``makv_entropy_backend: auto`` selects the CUDA implementation when the
compiled ``lmcache.c_ops`` extension and a CUDA device are available, and
otherwise uses the reference arithmetic implementation. Set
``makv_entropy_backend: reference`` explicitly for CPU tests. With
``makv_entropy_require_cuda: true``, missing CUDA support is an explicit error,
not a silent CPU fallback. The arithmetic layer is independent of CacheGen's
serialization format and does not use pickle.

Optional precision-risk residuals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``makv_residual_dtype`` to ``float16`` or ``float32`` on the remote manager
to retain the elementwise difference between the original KV and the manager's
low-bit dequantized value:

.. code-block:: yaml

   extra_config:
     makv_residual_dtype: float16
     makv_risk_upgrade_threshold: 0.80
     makv_risk_upgrade_policy: next
     makv_risk_window_tokens: 16
     makv_risk_window_ttl_s: 0

The client still sends only raw FP16/BF16 KV plus its QuantPlan; residuals are
computed after quantization in the manager and stored in the same self-contained
``MaKVObject`` as ``residual_8``, ``residual_4`` and ``residual_2`` payloads.
The default ``none`` preserves the legacy object size and behavior. ``float16``
has lower storage overhead; ``float32`` gives a more accurate source
reconstruction for an upgrade but can substantially reduce compression.

The manager exposes a ``PRECISION_RISK`` request, and the network connector
provides ``report_precision_risk(key, signal)``. After strict validation of the
frozen ``PrecisionRiskObserver`` contract, a risk at or above the threshold
promotes only the selected token's physical entries either one available tier
(``next``) or directly to the highest tier (``full``). The optional signal
fields ``token_index`` and ``window_tokens`` identify an absolute request/KV
token and its logical window; without ``token_index``, ``step`` is used as a
deterministic compatibility fallback.

The manager serializes promotions per key and keeps the upgraded object only
as a process-local temporary view. The canonical object in storage is never
overwritten, so the next GET after ``[step, step + window_tokens)`` expires
returns the original precision without another reconstruction. An optional
``makv_risk_window_ttl_s`` provides wall-clock expiry when decode-step signals
are not continuous. Residuals, ``risk_upgrade`` and window metadata are
manager-only and are stripped from normal GET and GET_BATCH responses; normal
GET therefore remains the quantized direct-to-paged GPU path. Missing
residuals, invalid objects and failed upgrades return an explicit no-op/failure
reason.

Start a manager with residual retention enabled using:

.. code-block:: bash

   PYTHONPATH=LMCache python -m lmcache.v1.storage_backend.makv_remote.server \
     --listen 0.0.0.0:65432 \
     --storage-url redis://127.0.0.1:6379/0 \
     --residual-dtype float16 \
     --risk-upgrade-threshold 0.80 \
     --risk-upgrade-policy next \
     --risk-window-tokens 16

When CUDA restore is required but the op is unavailable, configuration validation
fails clearly. Corrupt or incompatible GET objects are treated as cache misses;
partially restored entries are not returned.

Current limitations
-------------------

- Direct restore supports vLLM paged formats 0, 1, 2, 6, 7, 12, and 13,
  including fused-content HND and NHD layouts. MLA, SGLang, and TRT-LLM
  layouts fail explicitly.
- V3 direct restore currently requires one homogeneous layer group.
- The file, Redis, and Mooncake adapters store complete blobs; they do not
  provide cross-adapter migration or a shared object namespace automatically.
- A file-backed manager is slower than an in-memory/Redis baseline when its hot
  cache is disabled; size ``--memory-cache-gb`` for the working set or select
  ``redis``/``mooncake`` explicitly.
- Mooncake support requires a build exposing ``put_parts``,
  ``batch_get_buffer``, and ``is_exist``. The manager reports a clear startup
  error when the optional SDK is absent; Mooncake does not provide portable
  key enumeration for ``LIST``.
- Scale H2D is normalized to FP32 before the CUDA kernel.
- Stored objects support protocol versions 1 and 2; unknown versions are cache misses rather than negotiated.

Build
-----

The MaKV CUDA op is compiled as part of ``lmcache.c_ops`` when the CUDA build
profile is selected.

.. code-block:: bash

   cd LMCache
   export CUDA_HOME=/usr/local/cuda
   export PATH="$CUDA_HOME/bin:$PATH"
   export TORCH_CUDA_ARCH_LIST="12.0"  # select targets for the deployment
   python setup.py build_ext --inplace
   python -c "import lmcache.c_ops, torch; print(torch.ops.lmcache_makv.dequantize_scatter_paged_out)"

Requirements:

- a working ``nvcc`` in ``PATH``
- a PyTorch build compatible with the target CUDA toolkit
- a working GPU runtime/driver stack

CPU-only environments can still import LMCache when MaKV is not enabled. If
``makv_require_cuda_dequant: true`` is configured and the built op is missing,
startup fails instead of selecting the CPU reference path. Runtime JIT compilation
is not used.

Remote manager
--------------

Start the independent manager before LMCache:

.. code-block:: bash

   python -m lmcache.v1.storage_backend.makv_remote.server \
     --listen 0.0.0.0:65432 \
     --storage-backend file \
     --storage-url file:///tmp/lmcache-makv \
     --bucket-ratios 0.10,0.10,0.60,0.20 \
     --bucket-bits 16,8,4,2 \
     --queue-depth 64 --workers 2

Redis uses atomic ``SET`` replacement and a namespace to avoid collisions with
other Redis users:

.. code-block:: bash

   python -m lmcache.v1.storage_backend.makv_remote.server \
     --listen 0.0.0.0:65432 \
     --storage-backend redis \
     --storage-url redis://127.0.0.1:6379/1 \
     --storage-namespace lmcache:makv:experiment-1:

Mooncake is optional and reads the same JSON setup format used by the existing
LMCache Mooncake connector:

.. code-block:: bash

   python -m lmcache.v1.storage_backend.makv_remote.server \
     --listen 0.0.0.0:65432 \
     --storage-backend mooncake \
     --storage-url mooncake:// \
     --mooncake-config /path/to/mooncake.json

The protocol implements PUT, GET, EXISTS, DELETE, LIST, and HEALTH. PUT uses a
bounded queue for backpressure. GET and GET_BATCH return the stored object
unchanged and never dequantize on the manager. ``GET_BATCH`` uses an ``MKVB``
directory plus aligned object segments, so the manager avoids a second
concatenated blob while the client can stream objects in order.

Testing
-------

Targeted MaKV tests:

.. code-block:: bash

   cd LMCache
   PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0 \
     BUILD_WITH_CUDA=1 ../.venv/bin/python setup.py build_ext --inplace
   ../.venv/bin/python -m pytest -q tests/v1/test_makv.py
   ../.venv/bin/python -m pytest -q tests/v1/test_makv_entropy.py
   ../.venv/bin/python -m pytest -q tests/v1/test_makv_storage_adapter.py
   CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python -m pytest -q tests/v1/test_makv_cuda.py

Benchmark entry:

.. code-block:: bash

   cd LMCache
   ../.venv/bin/python -m pytest tests/benchmarks/test_makv.py --benchmark-only

For a phase-level profile using the real TCP manager and the production direct
paged CUDA restore path:

.. code-block:: bash

   cd LMCache
   CUDA_VISIBLE_DEVICES=7 ../.venv/bin/python benchmarks/makv_latency_breakdown.py \
     --model-name Qwen3-8B --layers 36 --kv-heads 8 --head-dim 128 \
     --chunk-tokens 2048 --chunks 8 --block-size 16 \
     --warmup 1 --iterations 3 --memory-cache-gb 2 \
     --output /tmp/makv-latency-qwen3-8b.json

Use ``--no-streaming-restore`` for the whole-batch receive/restore control.
The ``full_hit`` section is the A/B value: with streaming enabled it measures
the overlapped TCP receive plus direct paged restore wall time, rather than the
sum of their independent phase timings.

The JSON reports client plan construction, raw-payload copying, envelope
encoding, manager PUT decode/canonicalize/quantize/encode/validation/storage,
TCP GET, deserialize, CPU payload preparation, H2D, and fused paged
dequantization. It generates random KV values with the supplied model geometry;
it measures the MaKV data path rather than model forward time. The command uses
the real TCP manager and direct-to-paged CUDA operator, while the temporary
file store is automatically cleaned after the run.

``makv_remote_quantize_kernel_time_ms`` is a legacy metric name for the exact
``quantize_canonical_kv`` interval in the manager. The manager currently runs
this PyTorch quantize/pack work on CPU; it is not the client CUDA restore
kernel. The CUDA restore work is reported separately as
``makv_dequant_kernel_time_ms``. In a normal vLLM request, CacheEngine emits an
``MaKV latency breakdown`` JSON log after a successful retrieve with the
request-local TCP, CPU prepare, H2D, and fused-kernel components.

Metrics
-------

Current MaKV metrics include:

- client: ``makv_plan_time_ms`` / ``makv_client_plan_build_time_ms``,
  ``makv_client_raw_payload_copy_time_ms``,
  ``makv_client_envelope_encode_time_ms``,
  ``makv_client_serialize_total_time_ms``, ``makv_put_raw_bytes``,
  ``makv_put_plan_bytes``, and ``makv_client_quantize_calls``. The last metric
  must remain zero because PUT quantization is manager-side.
- remote: ``makv_remote_quantize_time_ms``, ``makv_raw_input_bytes``,
  ``makv_stored_bytes`` and HEALTH ``compression_ratio``,
  ``makv_quantize_failures``, ``makv_naive_fallbacks``,
  ``makv_memory_cache_hits`` and ``makv_memory_cache_misses``. HEALTH also
  includes ``makv_remote_put_decode_time_ms``,
  ``makv_remote_plan_canonicalize_time_ms``,
  ``makv_remote_quantize_kernel_time_ms``,
  ``makv_remote_object_encode_time_ms``,
  ``makv_remote_object_validate_time_ms``,
  ``makv_remote_encode_validate_time_ms``, and
  ``makv_remote_storage_put_time_ms``.
- restore: ``makv_get_quantized_bytes``, ``makv_h2d_time_ms``,
  ``makv_dequant_kernel_time_ms``, ``makv_restore_cpu_prepare_time_ms``,
  ``makv_restore_gpu_total_time_ms``, and ``makv_restore_total_time_ms``

Residual/control-plane metrics include ``makv_remote_residual_bytes``,
``makv_remote_risk_signals``, ``makv_remote_precision_upgrades``,
``makv_remote_precision_upgrade_failures`` and
``makv_remote_residual_upgrade_time_ms``. Window lifecycle metrics include
``makv_remote_precision_window_activations``,
``makv_remote_precision_window_refreshes``,
``makv_remote_precision_window_hits``,
``makv_remote_precision_window_expirations`` and
``makv_remote_precision_window_restores``.

The manager HEALTH response also reports ``makv_remote_get_batch_*`` timings,
``makv_remote_get_checksum_verifications``, and
``makv_remote_get_checksum_skips``. The client metrics include
``makv_client_pinned_receive_bytes`` and
``makv_client_pinned_receive_fallbacks``.

ScoutRank low-latency planning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Long-context experiments should use the vectorized fast planning policy:

.. code-block:: bash

   SCOUT_MODE=fast \
   SCOUT_OBSERVER_BACKEND=vectorized \
   SCOUT_ANCHOR_LAYERS=14,28 \
   ./tests/run_longbench_scoutrank_timing.sh

``fast`` omits the per-token LM-head NLL pass. ``vectorized`` computes the same
per-token/head MaKV residual math on the current GPU without serializing each
32-token observer block through the CPU. Importance generation uses the
lightweight ``damage_22`` API instead of materializing a full ``TokenScore``
object for every token. The legacy baseline remains available with
``SCOUT_MODE=balanced SCOUT_OBSERVER_BACKEND=production``; this is useful for
quality regression, but is not the recommended long-context TTFT setting.

QDM shadow diagnostics
----------------------

The MaKV Quantization Drift Meter (QDM) is a shadow/diagnostic observer and is
disabled by default. ``enable_qdm=False`` leaves the production quantizer,
precision plan, request metadata, stored payloads, and restore path unchanged.
The serializer does not transmit QDM control fields; an explicitly enabled
remote-manager shadow observer is the only production-side opt-in.

QDM witness metadata and payloads are optional and remain backward compatible
with objects that do not contain them. The reference witness, exact-drift
oracle, downstream-sensitivity oracle, and validation artifacts remain under
the QDM validation modules. The four risk states are diagnostic labels only;
they are not precision-controller decisions or calibrated production
thresholds.

Fallbacks
---------

``makv_fallback: naive`` is used for:

- missing importance
- invalid importance length
- remote quantization failure
- unsupported protocol situations handled on PUT

The remote object type is explicit, so a naive fallback object is not parsed as a
quantized MaKV payload.
