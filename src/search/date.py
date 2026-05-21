"""
Date search tool: temporal proximity and date-range matching.

Searches a date index for documents near a target date (exponential decay)
or within a date range. Chunk-aware with max-pooling per parent doc_id.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import CorpusDoc, SearchHit

_PER_CHANNEL_MAX = 50


def search_date(
    date_str: str,
    date_index: dict[str, list[str]],
    window_days: int = 7,
    max_results: int = _PER_CHANNEL_MAX,
    chunked_docs: dict | None = None,
    candidate_ids: list[str] | None = None,
) -> list[SearchHit]:
    """Search documents by temporal proximity with exponential decay scoring.

    When date_index maps to chunk_keys, resolves parent doc_ids via chunked_docs.
    Max-pools scores per parent doc_id.
    """
    from ..types import SearchHit

    if not date_str:
        return []
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return []
    candidate_set = set(candidate_ids) if candidate_ids else None
    window = max(int(window_days), 0)
    raw_hits: list[tuple[str, float, str]] = []  # (key, score, snippet)
    for offset in range(-window, window + 1):
        d = target + timedelta(days=offset)
        key_str = d.strftime("%Y-%m-%d")
        for entry_key in date_index.get(key_str, []):
            if candidate_set:
                doc = chunked_docs.get(entry_key) if chunked_docs else None
                parent = doc.doc_id if doc else entry_key
                if parent not in candidate_set:
                    continue
            score = math.exp(-0.15 * abs(offset))
            raw_hits.append((entry_key, score, f"date: {key_str}"))

    # Max-pool per parent doc_id if chunked
    if chunked_docs:
        doc_best: dict[str, tuple[float, str, str]] = {}  # parent → (score, chunk_key, snippet)
        for entry_key, score, snippet in raw_hits:
            doc = chunked_docs.get(entry_key)
            parent = doc.doc_id if doc else entry_key
            if score > doc_best.get(parent, (-1.0, "", ""))[0]:
                doc_best[parent] = (score, entry_key, snippet)
        hits = [SearchHit(parent, score, snippet, chunk_key=ck)
                for parent, (score, ck, snippet) in doc_best.items()]
    else:
        hits = [SearchHit(k, s, sn) for k, s, sn in raw_hits]

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:max_results]


def search_date_range(
    date_start: str,
    date_end: str,
    date_index: dict[str, list[str]],
    max_results: int = _PER_CHANNEL_MAX,
    chunked_docs: dict | None = None,
    candidate_ids: list[str] | None = None,
) -> list[SearchHit]:
    """Search documents within [date_start, date_end]. Chunk-aware with max-pool."""
    from ..types import SearchHit

    if not date_start or not date_end:
        return []
    try:
        start = datetime.strptime(date_start, "%Y-%m-%d")
        end = datetime.strptime(date_end, "%Y-%m-%d")
    except ValueError:
        return []
    if end < start:
        start, end = end, start
    candidate_set = set(candidate_ids) if candidate_ids else None
    raw: list[tuple[str, float, str]] = []
    current = start
    while current <= end:
        key = current.strftime("%Y-%m-%d")
        for entry_key in date_index.get(key, []):
            if candidate_set:
                doc = chunked_docs.get(entry_key) if chunked_docs else None
                parent = doc.doc_id if doc else entry_key
                if parent not in candidate_set:
                    continue
            raw.append((entry_key, 1.0, f"date: {key}"))
        current += timedelta(days=1)

    if chunked_docs:
        doc_best: dict[str, tuple[float, str, str]] = {}
        for entry_key, score, snippet in raw:
            doc = chunked_docs.get(entry_key)
            parent = doc.doc_id if doc else entry_key
            if score > doc_best.get(parent, (-1.0, "", ""))[0]:
                doc_best[parent] = (score, entry_key, snippet)
        hits = [SearchHit(parent, score, snippet, chunk_key=ck)
                for parent, (score, ck, snippet) in doc_best.items()]
    else:
        hits = [SearchHit(k, s, sn) for k, s, sn in raw]

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:max_results]
