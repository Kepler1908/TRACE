"""
Grep tool: case-insensitive, accent-normalized substring search.

Searches chunk text for exact substrings. Falls back to token-overlap
ranking when exact substring match yields fewer than 3 results.
Max-pools scores per parent doc_id and returns best chunk_key.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ...util.text import _strip_accents, jaccard_similarity

if TYPE_CHECKING:
    from ..types import CorpusDoc, SearchHit


def _norm(text: str) -> str:
    return _strip_accents(text.lower())


def _simple_tokens(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", _norm(text)) if t]


def _index_text(doc: CorpusDoc) -> str:
    return doc.text_original or ""


def _snippet(doc: CorpusDoc, length: int = 150) -> str:
    return (doc.text_original or "")[:length].replace("\n", " ")


def tool_grep(
    query: str,
    docs: dict[str, CorpusDoc],
    max_results: int = 10,
    candidate_ids: list[str] | None = None,
) -> list[SearchHit]:
    """
    Substring search across corpus chunks (chunk-aware, max-pooled).

    Searches text_original for exact substrings (accent/case insensitive).
    Returns the best chunk score per parent doc_id (max-pooling).
    """
    from ..types import SearchHit

    q_norm = _norm(query)
    q_tokens = set(_simple_tokens(query))

    if candidate_ids:
        parent_set = set(candidate_ids)
        scope = [k for k, d in docs.items() if d.doc_id in parent_set]
    else:
        scope = list(docs.keys())

    chunk_scores: list[tuple[str, str, float]] = []  # (chunk_key, parent, score)
    for chunk_key in scope:
        doc = docs.get(chunk_key)
        if doc is None:
            continue
        text = doc.text_original or ""
        if not text or q_norm not in _norm(text):
            continue
        doc_tokens = set(_simple_tokens(_index_text(doc)))
        bonus = jaccard_similarity(q_tokens, doc_tokens)
        chunk_scores.append((chunk_key, doc.doc_id, 1.0 + bonus))

    # If fewer than 3 exact matches, augment with token-overlap fallback
    if len(chunk_scores) < 3:
        seen = {cs[0] for cs in chunk_scores}
        for chunk_key in scope:
            if chunk_key in seen:
                continue
            doc = docs.get(chunk_key)
            if doc is None:
                continue
            doc_tokens = set(_simple_tokens(_index_text(doc)))
            score = jaccard_similarity(q_tokens, doc_tokens)
            if score > 0:
                chunk_scores.append((chunk_key, doc.doc_id, score))

    # Max-pool per parent doc_id, track best chunk_key
    doc_best: dict[str, tuple[float, str]] = {}  # parent → (score, chunk_key)
    for chunk_key, parent, score in chunk_scores:
        if score > doc_best.get(parent, (-1.0, ""))[0]:
            doc_best[parent] = (score, chunk_key)

    sorted_docs = sorted(doc_best.items(), key=lambda x: -x[1][0])[:max_results]

    hits: list[SearchHit] = []
    for parent_id, (score, chunk_key) in sorted_docs:
        doc = docs.get(chunk_key)
        snippet = _snippet(doc) if doc else ""
        hits.append(SearchHit(parent_id, score, snippet, chunk_key=chunk_key))
    return hits
