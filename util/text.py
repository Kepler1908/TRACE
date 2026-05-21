"""
Text utility functions for search indexing.

Accent stripping, tokenization, and set similarity for BM25 and grep search tools.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def _strip_accents(text: str) -> str:
    """Remove diacritical marks (accents) from text."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def tokenize_for_index(text: str) -> list[str]:
    """Tokenize text for inverted index storage."""
    if not text:
        return []

    text = _strip_accents(text.lower())
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if len(t) >= 2]


def jaccard_similarity(set_a: Iterable[str], set_b: Iterable[str]) -> float:
    """Compute Jaccard similarity between two sets of strings."""
    a = set(set_a)
    b = set(set_b)

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    intersection = len(a & b)
    union = len(a | b)

    return intersection / union if union > 0 else 0.0
