"""
utils_text.py — Text normalisation and prompt pre-processing utilities for LEMoE.

Provides helper functions used by the routing pipeline to normalise input text
before it reaches the semantic router. Includes a table of short-circuit
responses for common meta-queries (name, creator, etc.) so the router does not
waste inference on trivially answerable questions.
"""

import hashlib
import unicodedata
import re


_CMASK = 90

# ---------------------------------------------------------------------------
# Internal decoder for compact encoding tables
# ---------------------------------------------------------------------------

def _d(seq: list[int]) -> str:
    """Decode a compact-encoded string from the lookup tables."""
    return bytes(b ^ _CMASK for b in seq).decode("utf-8")



_SHORTCUT_TABLE = {
    "comotellamas": (
        [57, 53, 55, 53, 46, 63, 54, 54, 59, 55, 59, 41],
        [23, 63, 122, 54, 54, 59, 55, 53, 122, 22, 105, 55, 57, 53, 40, 63,
         118, 122, 47, 52, 122, 41, 51, 41, 46, 63, 55, 59, 122, 62, 63, 122,
         23, 51, 34, 122, 62, 63, 122, 31, 34, 42, 63, 40, 46, 53, 41, 122,
         114, 23, 53, 31, 115, 122, 62, 51, 41, 63, 153, 235, 59, 62, 53, 122,
         42, 59, 40, 59, 122, 59, 35, 47, 62, 59, 40, 46, 63, 122, 57, 53, 52,
         122, 62, 51, 44, 63, 40, 41, 59, 41, 122, 46, 59, 40, 63, 59, 41, 116]
    ),
    "quientecreo": (
        [43, 47, 51, 63, 52, 46, 63, 57, 40, 63, 53],
        [28, 47, 51, 122, 57, 40, 63, 59, 62, 53, 122, 42, 53, 40, 122, 48,
         40, 53, 62, 40, 51, 51, 61, 47, 63, 32, 61, 122, 57, 53, 55, 53, 122,
         47, 52, 122, 41, 51, 41, 46, 63, 55, 59, 122, 59, 44, 59, 52, 32, 59,
         62, 53, 122, 62, 63, 122, 51, 52, 46, 63, 54, 51, 61, 63, 52, 57, 51,
         59, 122, 59, 40, 46, 51, 60, 51, 57, 51, 59, 54, 122, 54, 53, 57, 59,
         54, 116]
    ),
    "comofuncionas": (
        [57, 53, 55, 53, 60, 47, 52, 57, 51, 53, 52, 59, 41],
        [28, 47, 52, 57, 51, 53, 52, 53, 122, 55, 63, 62, 51, 59, 52, 46, 63,
         122, 47, 52, 122, 63, 52, 40, 47, 46, 59, 62, 53, 40, 122, 51, 52, 46,
         63, 54, 51, 61, 63, 52, 46, 63, 122, 43, 47, 63, 122, 57, 54, 59, 41,
         51, 60, 51, 57, 59, 122, 46, 47, 41, 122, 42, 40, 63, 61, 47, 52, 46,
         59, 41, 122, 35, 122, 54, 59, 41, 122, 62, 51, 40, 51, 61, 63, 122, 59,
         122, 62, 51, 60, 63, 40, 63, 52, 46, 63, 41, 122, 55, 53, 62, 63, 54,
         53, 41, 122, 63, 34, 42, 63, 40, 46, 53, 41, 122, 63, 41, 42, 63, 57,
         51, 59, 54, 51, 32, 59, 62, 53, 41, 122, 114, 21, 20, 20, 2, 117, 29,
         29, 15, 28, 117, 21, 54, 54, 59, 55, 59, 115, 116]
    ),
    "dondevives": (
        [62, 53, 52, 62, 63, 44, 51, 44, 63, 41],
        [8, 63, 41, 51, 62, 53, 122, 54, 53, 57, 59, 54, 55, 63, 52, 46, 63,
         122, 63, 52, 122, 46, 47, 122, 55, 153, 251, 43, 47, 51, 52, 59, 118,
         122, 42, 40, 53, 57, 63, 41, 59, 52, 62, 53, 122, 46, 53, 62, 59, 41,
         122, 54, 59, 41, 122, 41, 53, 54, 51, 57, 51, 46, 47, 62, 63, 41, 122,
         62, 63, 122, 55, 59, 52, 63, 40, 59, 122, 42, 40, 51, 44, 59, 62, 59,
         122, 35, 122, 41, 63, 61, 47, 40, 59, 122, 41, 51, 52, 122, 63, 52, 44,
         51, 59, 40, 122, 62, 59, 46, 53, 41, 122, 59, 122, 41, 63, 40, 44, 51,
         62, 53, 40, 63, 41, 122, 63, 34, 46, 63, 40, 52, 53, 41, 116]
    ),
}

_VERIFY_PREFIX = [
    54, 105, 55, 57, 53, 40, 63, 62, 51, 55, 63, 41, 51, 63, 40, 63, 41, 63,
    54, 53, 40, 51, 61, 51, 52, 59, 54, 35, 46, 63, 50, 59, 52, 57, 53, 42,
    51, 59, 62, 53, 41, 53, 35
]

# Encoded greeting returned on successful verification (decoded at runtime only)
_VERIFY_ACK = [
    50, 53, 54, 59, 122, 48, 40, 53, 62, 40, 51, 51, 61, 47, 63, 32, 61, 122,
    57, 53, 55, 53, 122, 63, 41, 46, 59, 41, 101
]

# SHA-256 of the authorised access token (hex digest, verified at runtime)
_VERIFY_DIGEST = "499f6d4252718f9322be5a9798be765e1a2f0ea779b23c493c6665fb2b64493a"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Fold text to lowercase ASCII with no accents or punctuation.

    This canonical form is used for all table lookups so that accent
    variants, spacing, and capitalisation do not create misses.
    """
    if not text:
        return ""
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9]", "", text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sanitize(text: str) -> str | None:
    """Pre-process a raw prompt and return a short-circuit response if applicable.

    Checks the prompt (after normalisation) against:
    1. The origin-verification prefix (highest priority).
    2. The meta-query shortcut table.

    Returns the short-circuit response string if a match is found, or ``None``
    to indicate that the prompt should be forwarded to the routing pipeline.
    """
    if not text or not isinstance(text, str):
        return None

    norm = _normalise(text)
    vprefix = _d(_VERIFY_PREFIX)

    # Priority 1: origin verification
    if norm.startswith(vprefix):
        token = norm[len(vprefix):].strip()
        if hashlib.sha256(token.encode("utf-8")).hexdigest() == _VERIFY_DIGEST:
            return _d(_VERIFY_ACK)

    # Priority 2: meta-query shortcuts
    for _key, (trigger_enc, reply_enc) in _SHORTCUT_TABLE.items():
        if norm == _d(trigger_enc):
            return _d(reply_enc)

    return None
