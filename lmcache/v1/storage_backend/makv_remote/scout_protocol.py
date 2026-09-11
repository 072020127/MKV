# SPDX-License-Identifier: Apache-2.0

"""Binary payload helpers for asynchronous ScoutRank manager jobs."""

# Standard
from array import array
from collections.abc import Iterable, Sequence
import hashlib
import sys

# Version 1 is the original full-prompt protocol.  Version 2 adds an
# explicit absolute score range so a client can submit only an uncached
# suffix while retaining request-level layout metadata.
SCOUT_PROTOCOL_VERSION = 1
SCOUT_SUFFIX_PROTOCOL_VERSION = 2


def _network_order_array(typecode: str, values: Iterable[int | float]) -> bytes:
    result = array(typecode, values)
    if result.itemsize != 4:
        raise RuntimeError(f"array({typecode!r}) does not use 32-bit elements")
    if sys.byteorder == "little":
        result.byteswap()
    return result.tobytes()


def _decode_network_order_array(typecode: str, payload: bytes) -> array:
    if len(payload) % 4:
        raise ValueError("ScoutRank payload length must be a multiple of four")
    result = array(typecode)
    result.frombytes(payload)
    if result.itemsize != 4:
        raise RuntimeError(f"array({typecode!r}) does not use 32-bit elements")
    if sys.byteorder == "little":
        result.byteswap()
    return result


def encode_token_ids(token_ids: Sequence[int]) -> bytes:
    """Encode prompt token IDs as unsigned network-order 32-bit integers."""
    values = [int(value) for value in token_ids]
    if any(value < 0 or value > 0xFFFFFFFF for value in values):
        raise ValueError("ScoutRank token IDs must fit in uint32")
    return _network_order_array("I", values)


def decode_token_ids(payload: bytes, token_count: int) -> list[int]:
    """Decode and length-check one ScoutRank prompt payload."""
    if token_count < 0 or len(payload) != token_count * 4:
        raise ValueError("ScoutRank token payload length does not match token_count")
    return [int(value) for value in _decode_network_order_array("I", payload)]


def encode_scores(scores: Sequence[float]) -> bytes:
    """Encode request-level token scores as network-order float32."""
    return _network_order_array("f", (float(value) for value in scores))


def decode_scores(payload: bytes, token_count: int) -> list[float]:
    """Decode and length-check request-level float32 token scores."""
    if token_count < 0 or len(payload) != token_count * 4:
        raise ValueError("ScoutRank score payload length does not match token_count")
    return [float(value) for value in _decode_network_order_array("f", payload)]


def payload_sha256(payload: bytes) -> str:
    """Return the stable content identity used for idempotent submissions."""
    return hashlib.sha256(payload).hexdigest()


def score_context_sha256(
    token_sha256: str, score_start: int, full_token_count: int
) -> str:
    """Hash a score payload together with its absolute request range.

    The token payload alone is not a safe cross-request identity for
    suffix-only scoring: the same suffix at different absolute positions can
    represent different scorer inputs.  Including the range prevents an
    incorrect prompt-hash cache reuse.
    """
    context = f"{token_sha256}:{int(score_start)}:{int(full_token_count)}"
    return hashlib.sha256(context.encode("ascii")).hexdigest()
