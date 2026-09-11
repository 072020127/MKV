# SPDX-License-Identifier: Apache-2.0

"""MaKV configuration helpers."""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
import json
import importlib
import math
import os
from urllib.parse import urlparse

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    pass

logger = init_logger(__name__)

SUPPORTED_BUCKET_BITS = (16, 8, 4, 2)
SUPPORTED_PRECISION_SCHEMES = (
    "shared",
    "kv_separate_3tier",
    "kv_separate_4tier",
)
SUPPORTED_STORAGE_BACKENDS = ("file", "redis", "mooncake")
SUPPORTED_ENTROPY_CODECS = ("none", "cachegen_arithmetic")
SUPPORTED_ENTROPY_BACKENDS = ("auto", "cuda", "reference")
SUPPORTED_RESIDUAL_DTYPES = ("none", "float16", "float32")
DEFAULT_MAKV_STORAGE_URL = "redis://127.0.0.1:6379/0"
IMPORTANCE_REQUEST_KEY = "lmcache.makv_importance"
IMPORTANCE_LAYOUT_REQUEST_KEY = "lmcache.makv_importance_layout"
IMPORTANCE_STATUS_REQUEST_KEY = "lmcache.makv_importance_status"
PRECISION_PLAN_REQUEST_KEY = "lmcache.makv_precision_plan"
SCOUT_SCORE_START_REQUEST_KEY = "lmcache.makv_scout_score_start"
SCOUT_SCORE_END_REQUEST_KEY = "lmcache.makv_scout_score_end"


@dataclass(frozen=True)
class MaKVConfig:
    storage_url: str
    bucket_ratios: tuple[float, ...]
    bucket_bits: tuple[int, ...]
    importance_layout: str
    quant_granularity: str
    scale_dtype: str
    protect_prefix_tokens: int
    protect_tail_tokens: int
    dequant_backend: str
    require_cuda_dequant: bool
    fallback: str
    enable_checksum: bool
    allow_scoutrank_shadow_plan: bool = False
    storage_backend: str = "redis"
    storage_namespace: str = "lmcache:makv:"
    mooncake_config_path: Optional[str] = None
    precision_scheme: str = "shared"
    # Shadow/diagnostic observer only; never consumed by precision planning.
    enable_qdm: bool = False
    qdm_block_size: int = 32
    qdm_quantizer_version: str = "makv_per_token_head_symmetric_narrow_v1"
    scout_overlap_enabled: bool = False
    scout_suffix_only: bool = True
    scout_url: Optional[str] = None
    scout_timeout_s: float = 60.0
    # Optional arithmetic coding built on CacheGen's existing CUDA kernels.
    # Keep this disabled by default so existing MaKV objects are unchanged.
    entropy_codec: str = "none"
    entropy_backend: str = "auto"
    entropy_require_cuda: bool = False
    # Optional elementwise quantization residual used by the remote precision
    # upgrade policy. Disabled by default to preserve existing object sizes.
    residual_dtype: str = "none"
    risk_upgrade_threshold: float = 0.8
    risk_upgrade_policy: str = "next"
    # A risk promotion is a logical decode-step window. The durable object is
    # never replaced; the manager keeps an upgraded view only while active.
    risk_window_tokens: int = 16
    # Optional wall-clock expiry for deployments that do not report every
    # decode step. Zero keeps the logical-step-only behavior.
    risk_window_ttl_s: float = 0.0


def _normalize_importance_layout(value: Any) -> str:
    if value is None:
        return "token"
    layout = str(value).strip().lower().replace("-", "_")
    if layout not in ("token", "layer_kv_token"):
        raise ValueError(
            f"makv_importance_layout must be 'token' or 'layer_kv_token', got {value!r}"
        )
    return layout


def _normalize_scale_dtype(value: Any) -> str:
    if value is None:
        return "float16"
    dtype = str(value).strip().lower()
    if dtype not in ("float16", "float32"):
        raise ValueError(
            f"makv_scale_dtype must be 'float16' or 'float32', got {value!r}"
        )
    return dtype


def _normalize_backend(value: Any) -> str:
    if value is None:
        return "cuda"
    backend = str(value).strip().lower()
    if backend not in ("cuda", "reference"):
        raise ValueError(
            f"makv_dequant_backend must be 'cuda' or 'reference', got {value!r}"
        )
    return backend


def _normalize_fallback(value: Any) -> str:
    if value is None:
        return "naive"
    fallback = str(value).strip().lower()
    if fallback not in ("naive", "miss"):
        raise ValueError(f"makv_fallback must be 'naive' or 'miss', got {value!r}")
    return fallback


def _normalize_entropy_codec(value: Any) -> str:
    if value is None:
        return "none"
    codec = str(value).strip().lower().replace("-", "_")
    if codec not in SUPPORTED_ENTROPY_CODECS:
        raise ValueError(
            "makv_entropy_codec must be one of "
            f"{', '.join(SUPPORTED_ENTROPY_CODECS)}, got {value!r}"
        )
    return codec


def _normalize_entropy_backend(value: Any) -> str:
    if value is None:
        return "auto"
    backend = str(value).strip().lower().replace("-", "_")
    if backend not in SUPPORTED_ENTROPY_BACKENDS:
        raise ValueError(
            "makv_entropy_backend must be one of "
            f"{', '.join(SUPPORTED_ENTROPY_BACKENDS)}, got {value!r}"
        )
    return backend


def _normalize_residual_dtype(value: Any) -> str:
    if value is None:
        return "none"
    dtype = str(value).strip().lower()
    if dtype not in SUPPORTED_RESIDUAL_DTYPES:
        raise ValueError(
            "makv_residual_dtype must be one of "
            f"{', '.join(SUPPORTED_RESIDUAL_DTYPES)}, got {value!r}"
        )
    return dtype


def _normalize_risk_upgrade_policy(value: Any) -> str:
    if value is None:
        return "next"
    policy = str(value).strip().lower().replace("-", "_")
    if policy not in ("next", "full"):
        raise ValueError(
            "makv_risk_upgrade_policy must be 'next' or 'full', "
            f"got {value!r}"
        )
    return policy


def _normalize_risk_window_tokens(value: Any) -> int:
    if value is None:
        return 16
    if isinstance(value, bool):
        raise ValueError("makv_risk_window_tokens must be a positive integer")
    try:
        tokens = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "makv_risk_window_tokens must be a positive integer"
        ) from error
    if tokens <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError("makv_risk_window_tokens must be a positive integer")
    return tokens


def _normalize_risk_window_ttl(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        ttl = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "makv_risk_window_ttl_s must be finite and non-negative"
        ) from error
    if not math.isfinite(ttl) or ttl < 0.0:
        raise ValueError("makv_risk_window_ttl_s must be finite and non-negative")
    return ttl


def normalize_makv_precision_scheme(value: Any) -> str:
    """Normalize the precision assignment policy used by MaKV plans."""
    if value is None:
        return "shared"
    scheme = str(value).strip().lower().replace("-", "_")
    aliases = {
        "kv_separate": "kv_separate_3tier",
        "k8v4_k4v2_k2v2": "kv_separate_3tier",
        "kv_separate_with_bf16": "kv_separate_4tier",
        "k16v16_k8v4_k4v2_k2v2": "kv_separate_4tier",
    }
    scheme = aliases.get(scheme, scheme)
    if scheme not in SUPPORTED_PRECISION_SCHEMES:
        raise ValueError(
            "makv_precision_scheme must be one of "
            f"{', '.join(SUPPORTED_PRECISION_SCHEMES)}, got {value!r}"
        )
    return scheme


def default_makv_bucket_ratios(precision_scheme: str) -> tuple[float, ...]:
    """Return defaults matching each supported precision scheme."""
    if precision_scheme == "kv_separate_3tier":
        return (0.20, 0.30, 0.50)
    if precision_scheme == "kv_separate_4tier":
        return (0.10, 0.20, 0.50, 0.20)
    return (0.20, 0.30, 0.50)


def default_makv_bucket_bits(precision_scheme: str) -> tuple[int, ...]:
    """Return the physical payload buckets for a precision scheme."""
    if precision_scheme == "kv_separate_3tier":
        return (8, 4, 2)
    if precision_scheme == "kv_separate_4tier":
        return (16, 8, 4, 2)
    return (16, 8, 4)


def _normalize_storage_backend(value: Any, storage_url: str) -> str:
    """Normalize and cross-check the selected MaKV persistence backend."""
    scheme = urlparse(storage_url).scheme.lower()
    inferred = {
        "file": "file",
        "redis": "redis",
        "rediss": "redis",
        "mooncake": "mooncake",
    }.get(scheme)
    if inferred is None:
        raise ValueError(
            f"Unsupported makv_storage_url scheme {scheme!r}; use file://, "
            "redis://, rediss://, or mooncake://"
        )
    backend = inferred if value is None else str(value).strip().lower()
    if backend not in SUPPORTED_STORAGE_BACKENDS:
        raise ValueError(
            f"makv_storage_backend must be one of "
            f"{', '.join(SUPPORTED_STORAGE_BACKENDS)}, got {value!r}"
        )
    if backend != inferred:
        raise ValueError(
            f"makv_storage_backend={backend!r} does not match "
            f"makv_storage_url scheme {scheme!r}"
        )
    return backend


def _coerce_importance_from_request_config(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict)):
        return value
    if torch.is_tensor(value):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return json.loads(stripped)
    raise ValueError(
        f"Unsupported MaKV importance payload type in request_configs: {type(value)!r}"
    )


def validate_makv_runtime_config(config: Any) -> None:
    """Validate MaKV-specific config when ``remote_serde=makv``."""
    if getattr(config, "remote_serde", None) != "makv":
        return

    extra = getattr(config, "extra_config", None) or {}
    precision_scheme = normalize_makv_precision_scheme(
        extra.get("makv_precision_scheme")
    )
    default_ratios = default_makv_bucket_ratios(precision_scheme)
    ratios_raw = extra.get("makv_bucket_ratios", default_ratios)
    default_bits = default_makv_bucket_bits(precision_scheme)
    bits_raw = extra.get("makv_bucket_bits")
    if bits_raw is None:
        bits_raw = default_bits
    ratios = [float(v) for v in ratios_raw]
    bits = [int(v) for v in bits_raw]

    if len(ratios) != len(bits):
        raise ValueError(
            "makv_bucket_ratios and makv_bucket_bits must have the same length"
        )
    if not ratios:
        raise ValueError("makv_bucket_ratios must not be empty")
    if any(v < 0.0 for v in ratios):
        raise ValueError("makv_bucket_ratios must be non-negative")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("makv_bucket_ratios must sum to 1.0")
    if any(bit not in SUPPORTED_BUCKET_BITS for bit in bits):
        raise ValueError(
            "makv_bucket_bits only supports 16/8/4/2 in the current implementation"
        )
    if len(set(bits)) != len(bits):
        raise ValueError("makv_bucket_bits must not contain duplicate widths")
    if precision_scheme == "kv_separate_3tier" and tuple(bits) != (8, 4, 2):
        raise ValueError(
            "makv_precision_scheme=kv_separate_3tier requires "
            "makv_bucket_bits=[8,4,2]"
        )
    if precision_scheme == "kv_separate_4tier" and tuple(bits) != (16, 8, 4, 2):
        raise ValueError(
            "makv_precision_scheme=kv_separate_4tier requires "
            "makv_bucket_bits=[16,8,4,2]"
        )
    if extra.get("makv_quant_granularity", "per_token_head") != "per_token_head":
        raise ValueError(
            "makv_quant_granularity currently only supports 'per_token_head'"
        )

    _normalize_importance_layout(extra.get("makv_importance_layout"))
    _normalize_scale_dtype(extra.get("makv_scale_dtype"))
    _normalize_backend(extra.get("makv_dequant_backend"))
    _normalize_fallback(extra.get("makv_fallback"))
    entropy_codec = _normalize_entropy_codec(extra.get("makv_entropy_codec"))
    entropy_backend = _normalize_entropy_backend(
        extra.get("makv_entropy_backend")
    )
    if bool(extra.get("makv_entropy_require_cuda", False)) and entropy_codec != "none":
        if entropy_backend == "reference":
            raise ValueError(
                "makv_entropy_require_cuda=true is incompatible with "
                "makv_entropy_backend=reference"
            )
    _normalize_residual_dtype(extra.get("makv_residual_dtype"))
    risk_threshold = float(extra.get("makv_risk_upgrade_threshold", 0.8))
    if not math.isfinite(risk_threshold) or not 0.0 <= risk_threshold <= 1.0:
        raise ValueError("makv_risk_upgrade_threshold must be in [0, 1]")
    _normalize_risk_upgrade_policy(extra.get("makv_risk_upgrade_policy"))
    _normalize_risk_window_tokens(extra.get("makv_risk_window_tokens"))
    _normalize_risk_window_ttl(extra.get("makv_risk_window_ttl_s"))
    storage_url = str(
        extra.get("makv_storage_url", DEFAULT_MAKV_STORAGE_URL)
    )
    storage_backend = _normalize_storage_backend(
        extra.get("makv_storage_backend"), storage_url
    )
    namespace = str(extra.get("makv_storage_namespace", "lmcache:makv:"))
    if storage_backend == "redis" and not namespace:
        raise ValueError("makv_storage_namespace must not be empty for Redis")

    if int(extra.get("makv_protect_prefix_tokens", 4)) < 0:
        raise ValueError("makv_protect_prefix_tokens must be >= 0")
    if int(extra.get("makv_protect_tail_tokens", 16)) < 0:
        raise ValueError("makv_protect_tail_tokens must be >= 0")

    scout_overlap_enabled = bool(
        extra.get("makv_scout_overlap_enabled", False)
    )
    if scout_overlap_enabled:
        scout_url = str(
            extra.get("makv_scout_url") or getattr(config, "remote_url", "")
        )
        parsed_scout_url = urlparse(scout_url)
        if (
            parsed_scout_url.scheme != "makv"
            or parsed_scout_url.hostname is None
            or parsed_scout_url.port is None
        ):
            raise ValueError(
                "makv_scout_overlap_enabled=true requires makv_scout_url "
                "or remote_url in makv://host:port form"
            )
        if float(extra.get("makv_scout_timeout_s", 60.0)) <= 0:
            raise ValueError("makv_scout_timeout_s must be positive")

    qdm_enabled = bool(
        extra.get("makv_enable_qdm", extra.get("enable_qdm", False))
    )
    qdm_block_size = int(
        extra.get("makv_qdm_block_size", extra.get("qdm_block_size", 32))
    )
    if qdm_enabled and qdm_block_size <= 0:
        raise ValueError("makv_qdm_block_size must be positive")

    require_cuda = bool(extra.get("makv_require_cuda_dequant", True))
    backend = _normalize_backend(extra.get("makv_dequant_backend"))
    if require_cuda and backend == "cuda":
        extension_error: Exception | None = None
        for module_name in ("lmcache.cuda_ops", "lmcache.c_ops"):
            try:
                importlib.import_module(module_name)
                extension_error = None
                break
            except (ImportError, OSError) as error:
                extension_error = error
        if extension_error is not None:
            raise RuntimeError(
                "makv_require_cuda_dequant=true but no usable LMCache CUDA "
                "extension (lmcache.cuda_ops or lmcache.c_ops) is available"
            ) from extension_error
        from lmcache.v1.storage_backend.makv.paged_restore import (
            makv_paged_cuda_op_available,
        )

        if not makv_paged_cuda_op_available():
            raise RuntimeError(
                "makv_require_cuda_dequant=true but "
                "lmcache_makv.dequantize_scatter_paged_out is unavailable"
            )


def get_makv_config(config: Any) -> MaKVConfig:
    """Build a normalized runtime MaKV config."""
    validate_makv_runtime_config(config)
    extra = getattr(config, "extra_config", None) or {}
    qdm_enabled = bool(
        extra.get("makv_enable_qdm", extra.get("enable_qdm", False))
    )
    precision_scheme = normalize_makv_precision_scheme(
        extra.get("makv_precision_scheme")
    )
    storage_url = str(
        extra.get("makv_storage_url", DEFAULT_MAKV_STORAGE_URL)
    )
    storage_backend = _normalize_storage_backend(
        extra.get("makv_storage_backend"), storage_url
    )
    mooncake_config_path = (
        extra.get("makv_mooncake_config")
        or extra.get("mooncake_config_path")
        or os.getenv("MOONCAKE_CONFIG_PATH")
    )
    default_ratios = default_makv_bucket_ratios(precision_scheme)
    default_bits = default_makv_bucket_bits(precision_scheme)
    bits_value = extra.get("makv_bucket_bits")
    if bits_value is None:
        bits_value = default_bits
    entropy_codec = _normalize_entropy_codec(extra.get("makv_entropy_codec"))
    entropy_backend = _normalize_entropy_backend(
        extra.get("makv_entropy_backend")
    )
    return MaKVConfig(
        storage_url=storage_url,
        bucket_ratios=tuple(
            float(v) for v in extra.get("makv_bucket_ratios", default_ratios)
        ),
        bucket_bits=tuple(int(v) for v in bits_value),
        importance_layout=_normalize_importance_layout(
            extra.get("makv_importance_layout")
        ),
        quant_granularity=str(extra.get("makv_quant_granularity", "per_token_head")),
        scale_dtype=_normalize_scale_dtype(extra.get("makv_scale_dtype")),
        protect_prefix_tokens=int(extra.get("makv_protect_prefix_tokens", 4)),
        protect_tail_tokens=int(extra.get("makv_protect_tail_tokens", 16)),
        dequant_backend=_normalize_backend(extra.get("makv_dequant_backend")),
        require_cuda_dequant=bool(extra.get("makv_require_cuda_dequant", True)),
        fallback=_normalize_fallback(extra.get("makv_fallback")),
        enable_checksum=bool(extra.get("makv_enable_checksum", True)),
        allow_scoutrank_shadow_plan=bool(
            extra.get("makv_allow_scoutrank_shadow_plan", False)
        ),
        storage_backend=storage_backend,
        storage_namespace=str(
            extra.get("makv_storage_namespace", "lmcache:makv:")
        ),
        mooncake_config_path=(
            str(mooncake_config_path) if mooncake_config_path else None
        ),
        precision_scheme=precision_scheme,
        enable_qdm=qdm_enabled,
        qdm_block_size=int(
            extra.get("makv_qdm_block_size", extra.get("qdm_block_size", 32))
        ),
        qdm_quantizer_version=str(
            extra.get(
                "makv_qdm_quantizer_version",
                "makv_per_token_head_symmetric_narrow_v1",
            )
        ),
        scout_overlap_enabled=bool(
            extra.get("makv_scout_overlap_enabled", False)
        ),
        scout_suffix_only=bool(extra.get("makv_scout_suffix_only", True)),
        scout_url=(
            str(extra.get("makv_scout_url") or getattr(config, "remote_url", ""))
            if extra.get("makv_scout_overlap_enabled", False)
            else None
        ),
        scout_timeout_s=float(extra.get("makv_scout_timeout_s", 60.0)),
        entropy_codec=entropy_codec,
        entropy_backend=entropy_backend,
        entropy_require_cuda=bool(extra.get("makv_entropy_require_cuda", False)),
        residual_dtype=_normalize_residual_dtype(
            extra.get("makv_residual_dtype")
        ),
        risk_upgrade_threshold=float(
            extra.get("makv_risk_upgrade_threshold", 0.8)
        ),
        risk_upgrade_policy=_normalize_risk_upgrade_policy(
            extra.get("makv_risk_upgrade_policy")
        ),
        risk_window_tokens=_normalize_risk_window_tokens(
            extra.get("makv_risk_window_tokens")
        ),
        risk_window_ttl_s=_normalize_risk_window_ttl(
            extra.get("makv_risk_window_ttl_s")
        ),
    )


def extract_makv_precision_plan(
    transfer_spec: Optional[dict[str, Any]],
    request_configs: Optional[dict[str, Any]],
) -> Any:
    """Extract a frozen ScoutRank precision plan without interpreting scores."""
    value = None
    if transfer_spec is not None:
        value = transfer_spec.get("makv_precision_plan")
    if value is None and request_configs is not None:
        value = request_configs.get(PRECISION_PLAN_REQUEST_KEY)
    if value is None or isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        return json.loads(value) if value else None
    raise ValueError(
        "MaKV precision plan must be a JSON object or encoded JSON string, got "
        f"{type(value)!r}"
    )


def extract_makv_importance(
    transfer_spec: Optional[dict[str, Any]],
    request_configs: Optional[dict[str, Any]],
) -> tuple[Any, Optional[str]]:
    """Extract MaKV importance payload from transfer or request context."""
    if transfer_spec is not None:
        if "makv_importance" in transfer_spec:
            return transfer_spec["makv_importance"], transfer_spec.get(
                "makv_importance_layout"
            )
    if request_configs is not None and IMPORTANCE_REQUEST_KEY in request_configs:
        return (
            _coerce_importance_from_request_config(
                request_configs.get(IMPORTANCE_REQUEST_KEY)
            ),
            request_configs.get(IMPORTANCE_LAYOUT_REQUEST_KEY),
        )
    return None, None


def extract_makv_importance_status(
    transfer_spec: Optional[dict[str, Any]],
    request_configs: Optional[dict[str, Any]],
) -> Any:
    """Extract optional per-token validity/forced-precision status.

    The status is independent from the floating-point importance vector. This
    lets an uncovered token remain a finite score while staying fail-closed in
    the allocator.
    """
    value = None
    if transfer_spec is not None:
        value = transfer_spec.get(IMPORTANCE_STATUS_REQUEST_KEY)
    if value is None and request_configs is not None:
        value = request_configs.get(IMPORTANCE_STATUS_REQUEST_KEY)
    return _coerce_importance_from_request_config(value)


def has_makv_importance(
    transfer_spec: Optional[dict[str, Any]],
    request_configs: Optional[dict[str, Any]],
) -> bool:
    """Return whether MaKV importance is present in the current request context."""
    importance, _ = extract_makv_importance(transfer_spec, request_configs)
    return importance is not None
