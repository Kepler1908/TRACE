"""
BM25 tool: term-frequency ranking with IDF weighting.

Two-tier implementation:
1. If ``bm25s`` is installed AND ``USE_BM25S`` is enabled, scoring goes
   through the C-backed sparse-matrix library (~100× faster on large
   corpora; same Okapi BM25 formula).
2. Otherwise a custom implementation runs with a postings-list inverted
   index (df-keyed), so query cost scales with ``|query_terms| · df`` not
   with the full corpus size.

Both tiers are chunk-aware: ``chunk_to_parent`` maps chunk keys to parent
doc_ids and ``tool_bm25`` max-pools scores per parent.
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

    ``postings`` is the inverted index used by the custom scorer:
    term -> list[(chunk_key, tf)]. ``bm25s_obj`` holds an optional
    ``bm25s.BM25`` instance (with parallel ``bm25s_corpus_keys`` listing
    the chunk_key per row) used when the library is available.
    """

    doc_ids: list[str]
    tf: dict[str, Counter]        # key -> term -> count
    df: dict[str, int]            # term -> doc count
    doc_len: dict[str, int]       # key -> token count
    avgdl: float
    chunk_to_parent: dict[str, str] = field(default_factory=dict)
    postings: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    bm25s_obj: object | None = None
    bm25s_corpus_keys: list[str] = field(default_factory=list)
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
    postings: dict[str, list[tuple[str, int]]] = defaultdict(list)

    for key, doc in docs.items():
        doc_ids.append(key)
        chunk_to_parent[key] = doc.doc_id
        tokens = tokenize_for_index(_index_text(doc))
        counts = Counter(tokens)
        tf[key] = counts
        doc_len[key] = len(tokens)
        for term, count in counts.items():
            df[term] += 1
            postings[term].append((key, count))

    total = sum(doc_len.values())
    avgdl = total / len(doc_len) if doc_len else 0.0

    bm25s_obj = None
    bm25s_corpus_keys: list[str] = []
    try:
        from ..types import USE_BM25S
    except Exception:
        USE_BM25S = False  # type: ignore[assignment]

    if USE_BM25S:
        try:
            import bm25s  # type: ignore

            corpus_tokens = [list(tf[k].elements()) for k in doc_ids]
            retriever = bm25s.BM25(k1=1.5, b=0.75)
            retriever.index(corpus_tokens)
            bm25s_obj = retriever
            bm25s_corpus_keys = list(doc_ids)
        except Exception:
            # bm25s missing or failed at index time — silently fall back
            # to the postings-list scorer. The custom path is still correct.
            bm25s_obj = None
            bm25s_corpus_keys = []

    return BM25Index(
        doc_ids=doc_ids, tf=tf, df=dict(df), doc_len=doc_len, avgdl=avgdl,
        chunk_to_parent=chunk_to_parent,
        postings=dict(postings),
        bm25s_obj=bm25s_obj,
        bm25s_corpus_keys=bm25s_corpus_keys,
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


def _score_via_postings(
    q_tokens: list[str],
    index: BM25Index,
    scope_set: set[str] | None,
) -> dict[str, float]:
    """Score using the postings inverted index.

    Only documents that contain at least one query term are visited — O(sum
    of df) instead of O(N corpus). For each candidate, we apply the full
    Okapi BM25 formula on the union of query terms it actually contains.
    """
    if not q_tokens or index.avgdl <= 0.0:
        return {}
    N = len(index.doc_ids)
    k1, b = index.k1, index.b

    candidate_terms: dict[str, list[tuple[str, int, float]]] = {}
    for term in set(q_tokens):
        df = index.df.get(term, 0)
        if df == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        for key, freq in index.postings.get(term, ()):
            if scope_set is not None and key not in scope_set:
                continue
            candidate_terms.setdefault(key, []).append((term, freq, idf))

    scores: dict[str, float] = {}
    for key, entries in candidate_terms.items():
        dl = index.doc_len.get(key, 0)
        norm = k1 * (1.0 - b + b * dl / index.avgdl)
        s = 0.0
        for _term, freq, idf in entries:
            s += idf * (freq * (k1 + 1.0)) / (freq + norm)
        if s > 0.0:
            scores[key] = s
    return scores


def _score_via_bm25s(
    q_tokens: list[str],
    index: BM25Index,
    scope_set: set[str] | None,
    k_pool: int,
) -> dict[str, float]:
    if index.bm25s_obj is None or not q_tokens:
        return {}
    try:
        import numpy as np

        topn = min(k_pool, len(index.bm25s_corpus_keys))
        # bm25s expects a tokenised query (list of strings).
        ranked_idx, ranked_scores = index.bm25s_obj.retrieve(
            [q_tokens], k=topn, return_as="tuple",
        )
        ids = np.asarray(ranked_idx[0])
        scores = np.asarray(ranked_scores[0])
    except Exception:
        return {}

    out: dict[str, float] = {}
    for idx, score in zip(ids.tolist(), scores.tolist()):
        if score <= 0:
            continue
        key = index.bm25s_corpus_keys[int(idx)]
        if scope_set is not None and key not in scope_set:
            continue
        out[key] = float(score)
    return out


def tool_bm25(
    query: str,
    index: BM25Index,
    docs: dict[str, CorpusDoc],
    max_results: int = 10,
    candidate_ids: list[str] | None = None,
) -> list[SearchHit]:
    from ..types import SearchHit

    q_tokens = tokenize_for_index(query)
    if not q_tokens:
        return []

    if candidate_ids:
        parent_set = set(candidate_ids)
        scope_set: set[str] | None = {
            k for k in index.doc_ids
            if index.chunk_to_parent.get(k, k) in parent_set
        }
    else:
        scope_set = None

    # Try bm25s first (faster on big corpora). Over-fetch so post-filter
    # and per-parent max-pool still yield ``max_results`` parent docs.
    k_pool = max(max_results * 6, 200)
    chunk_scores = _score_via_bm25s(q_tokens, index, scope_set, k_pool)
    if not chunk_scores:
        chunk_scores = _score_via_postings(q_tokens, index, scope_set)

    # Max-pool per parent doc_id, track best chunk_key
    doc_best: dict[str, tuple[float, str]] = {}
    for key, s in chunk_scores.items():
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
