"""
BM25 tool: term-frequency ranking with IDF weighting.

Custom BM25 implementation using tokenize_for_index() from the storage
utilities. Supports chunk-based indexing where multiple chunks share the
same parent doc_id — returns the best chunk score per document (max-pooling).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...util.text import tokenize_for_index

if TYPE_CHECKING:
    from ..types import CorpusDoc, SearchHit


def _index_text(doc: CorpusDoc) -> str:
    return doc.text_original or ""


def _snippet(doc: CorpusDoc, length: int = 150) -> str:
    return (doc.text_original or "")[:length].replace("\n", " ")


@dataclass
class BM25Index:
    """Precomputed BM25 index over a (possibly chunked) corpus.

    When built from a chunked corpus, ``chunk_to_parent`` maps each internal
    chunk key to its parent doc_id.  ``tool_bm25`` max-pools scores per
    parent so callers always receive parent doc_ids.
    """

    doc_ids: list[str]
    tf: dict[str, Counter]        # key -> term -> count
    df: dict[str, int]            # term -> doc count
    doc_len: dict[str, int]       # key -> token count
    avgdl: float
    chunk_to_parent: dict[str, str] = field(default_factory=dict)
    k1: float = 1.5
    b: float = 0.75


def build_bm25(docs: dict[str, CorpusDoc]) -> BM25Index:
    """Build a BM25 index from corpus documents (chunk-aware).

    If the dict contains chunk keys (``doc_id__chunk_N``), each chunk is
    indexed independently and ``chunk_to_parent`` maps chunk keys to the
    CorpusDoc.doc_id (parent).
    """
    doc_ids: list[str] = []
    tf: dict[str, Counter] = {}
    df: defaultdict[str, int] = defaultdict(int)
    doc_len: dict[str, int] = {}
    chunk_to_parent: dict[str, str] = {}

    for key, doc in docs.items():
        doc_ids.append(key)
        chunk_to_parent[key] = doc.doc_id
        tokens = tokenize_for_index(_index_text(doc))
        counts = Counter(tokens)
        tf[key] = counts
        doc_len[key] = len(tokens)
        for term in counts:
            df[term] += 1

    total = sum(doc_len.values())
    avgdl = total / len(doc_len) if doc_len else 0.0
    return BM25Index(
        doc_ids=doc_ids, tf=tf, df=dict(df), doc_len=doc_len, avgdl=avgdl,
        chunk_to_parent=chunk_to_parent,
    )


def _bm25_score(query_tokens: list[str], idx: BM25Index, doc_id: str) -> float:
    """Score a single document against query tokens using BM25."""
    if not query_tokens:
        return 0.0
    N = len(idx.doc_ids)
    dl = idx.doc_len.get(doc_id, 0)
    norm = idx.k1 * (1.0 - idx.b + idx.b * dl / (idx.avgdl or 1.0))
    score = 0.0
    doc_tf = idx.tf.get(doc_id, Counter())
    for term in query_tokens:
        df = idx.df.get(term, 0)
        if df == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        f = doc_tf.get(term, 0)
        if f == 0:
            continue
        score += idf * (f * (idx.k1 + 1.0)) / (f + norm)
    return score


def tool_bm25(
    query: str,
    index: BM25Index,
    docs: dict[str, CorpusDoc],
    max_results: int = 10,
    candidate_ids: list[str] | None = None,
) -> list[SearchHit]:
    """
    BM25 keyword search over the corpus (chunk-aware, max-pooled).

    Scores each chunk independently and returns the best score per parent
    doc_id. When ``candidate_ids`` are provided, they are parent doc_ids —
    all chunks belonging to those parents are searched.

    Returns:
        Sorted list of SearchHit with parent doc_ids (highest BM25 first).
    """
    from ..types import SearchHit

    q_tokens = tokenize_for_index(query)

    if candidate_ids:
        # Expand parent IDs to chunk keys
        parent_set = set(candidate_ids)
        scope = [k for k in index.doc_ids
                 if index.chunk_to_parent.get(k, k) in parent_set]
    else:
        scope = index.doc_ids

    # Score chunks, max-pool per parent doc_id, track best chunk_key
    doc_best: dict[str, tuple[float, str]] = {}  # parent → (score, chunk_key)
    for key in scope:
        s = _bm25_score(q_tokens, index, key)
        if s <= 0:
            continue
        parent = index.chunk_to_parent.get(key, key)
        if s > doc_best.get(parent, (-1.0, ""))[0]:
            doc_best[parent] = (s, key)

    sorted_docs = sorted(doc_best.items(), key=lambda x: -x[1][0])[:max_results]

    hits: list[SearchHit] = []
    for parent_id, (score, chunk_key) in sorted_docs:
        doc = docs.get(parent_id)
        snippet = _snippet(doc) if doc else ""
        hits.append(SearchHit(parent_id, score, snippet, chunk_key=chunk_key))
    return hits
