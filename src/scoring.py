# scoring.py
"""Phase 1 scoring: RRF fusion, corpus allocation."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any

from .types import (
    CorpusDoc, MetadataEntry, SearchHit, FusedCandidate,
    PER_CHANNEL_MAX, RRF_K, LLMParseError,
)
from .search.bm25 import BM25Index, tool_bm25
from .search.cosine import EmbeddingIndex, tool_cosine
from .search.date import search_date


_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_query(text: str) -> str:
    """Unicode-NFKC + whitespace collapse + lowercase.

    Used to dedupe agent searches that only differ in punctuation, accents
    or whitespace — the prior strict-lower compare let near-duplicates slip.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE_RE.sub(" ", normalized).strip()


# ---------------------------------------------------------------------------
# RRF score fusion
# ---------------------------------------------------------------------------

def _merge_hits_max_score(hits: list[SearchHit]) -> list[SearchHit]:
    """Deduplicate hits by doc_id, keeping max score."""
    best: dict[str, SearchHit] = {}
    for h in hits:
        if h.doc_id not in best or h.score > best[h.doc_id].score:
            best[h.doc_id] = h
    return sorted(best.values(), key=lambda h: h.score, reverse=True)


def run_score_fusion(
    query: str,
    *,
    full_bm25: BM25Index,
    full_docs: dict[str, CorpusDoc],
    full_emb: EmbeddingIndex | None,
    metadata_index: dict[str, MetadataEntry],
    date_index: dict[str, list[str]],
    top_k: int = 20,
    rrf_k: int = RRF_K,
    doc_index: Any = None,
) -> tuple[list[FusedCandidate], dict[str, list[str]]]:
    """3-channel Reciprocal Rank Fusion: BM25, date/metadata, dense embedding."""
    channel_hits: dict[str, list[SearchHit]] = {}

    # Channel 1: BM25
    channel_hits["bm25"] = tool_bm25(query, full_bm25, full_docs, max_results=PER_CHANNEL_MAX)

    # Channel 2: Date/metadata
    dates = _DATE_RE.findall(query)
    meta_hits: list[SearchHit] = []
    if dates:
        for d in dates:
            meta_hits.extend(search_date(d, date_index, window_days=7,
                                         max_results=PER_CHANNEL_MAX,
                                         chunked_docs=full_docs))
        meta_hits = _merge_hits_max_score(meta_hits)[:PER_CHANNEL_MAX]
    channel_hits["metadata"] = meta_hits

    # Channel 3: Dense embeddings
    if full_emb is not None:
        channel_hits["dense"] = tool_cosine(query, full_emb, max_results=PER_CHANNEL_MAX)


    # RRF fusion
    scores: dict[str, float] = defaultdict(float)
    channels_map: dict[str, list[str]] = defaultdict(list)
    snippets: dict[str, str] = {}
    # Best chunk_key per parent doc_id, picked by RRF-rank (cross-channel
    # comparable) with dense > bm25 > metadata as tie-breaker since dense
    # spans contain the semantically relevant excerpt for review.
    chunk_keys: dict[str, str] = {}
    chunk_key_rank: dict[str, float] = {}
    _CHANNEL_PRIORITY = {"dense": 0, "bm25": 1, "metadata": 2}

    for ch_name, hits in channel_hits.items():
        if not hits:
            continue
        seen: dict[str, SearchHit] = {}
        for h in hits:
            if h.doc_id not in seen or h.score > seen[h.doc_id].score:
                seen[h.doc_id] = h
        ranked = sorted(seen.values(), key=lambda h: h.score, reverse=True)

        for rank, h in enumerate(ranked):
            rrf_contrib = 1.0 / (rrf_k + rank + 1)
            scores[h.doc_id] += rrf_contrib
            channels_map[h.doc_id].append(ch_name)
            if h.doc_id not in snippets:
                snippets[h.doc_id] = h.snippet
            if h.chunk_key:
                ch_prio = _CHANNEL_PRIORITY.get(ch_name, 9)
                # Prefer best RRF rank; ties broken by channel priority.
                key_score = -rank * 10 - ch_prio
                if (h.doc_id not in chunk_keys
                        or key_score > chunk_key_rank[h.doc_id]):
                    chunk_keys[h.doc_id] = h.chunk_key
                    chunk_key_rank[h.doc_id] = key_score

    # Sort by RRF score
    sorted_all = sorted(scores.keys(), key=lambda d: (-scores[d], d))

    fused = [
        FusedCandidate(
            doc_id=doc_id,
            rrf_score=scores[doc_id],
            channels=channels_map[doc_id],
            snippet=snippets.get(doc_id, ""),
            chunk_key=chunk_keys.get(doc_id, doc_id),
        )
        for doc_id in sorted_all[:top_k]
    ]

    return fused, dict(channels_map)


# ---------------------------------------------------------------------------
# Corpus allocation
# ---------------------------------------------------------------------------

def allocate_by_corpus(
    fused: list[FusedCandidate],
    metadata_index: dict[str, MetadataEntry],
    corpus_constraint: str,
    max_total: int = 20,
) -> list[FusedCandidate]:
    """Allocate RRF candidates proportionally by corpus, respecting constraint."""
    if corpus_constraint == "any":
        return fused[:max_total]

    # Filter to matching corpus
    filtered = []
    for c in fused:
        meta = metadata_index.get(c.chunk_key) or metadata_index.get(c.doc_id)
        if meta and (corpus_constraint == meta.corpus or corpus_constraint == meta.corpus_type):
            filtered.append(c)

    return filtered[:max_total]


# ---------------------------------------------------------------------------
# Utility functions used by agent/pipeline
# ---------------------------------------------------------------------------

def filter_hits_by_corpus(
    hits: list[SearchHit],
    metadata_index: dict[str, MetadataEntry],
    corpus_filter: str | None,
) -> list[SearchHit]:
    """Filter search hits by corpus name or corpus_type."""
    if not corpus_filter:
        return hits
    return [
        h for h in hits
        if (m := metadata_index.get(h.chunk_key) or metadata_index.get(h.doc_id))
        and (corpus_filter == m.corpus or corpus_filter == m.corpus_type)
    ]


def is_repeated_query(
    query: str,
    call_history: list[tuple[str, str, str | int | None]],
    corpus_filter: str | None = None,
    action_key: str = "",
) -> bool:
    """Check if a query was already made (avoid repeated searches).

    Comparison uses NFKC-normalized casefolded text with whitespace collapse,
    so trivial paraphrases ("Hugo, Victor" vs "  hugo victor  ") are caught.
    """
    query_norm = _normalize_query(query)
    for prev_action, prev_query, prev_filter in call_history:
        if action_key and prev_action != action_key:
            continue
        if _normalize_query(prev_query) == query_norm:
            if corpus_filter == prev_filter or (not corpus_filter and not prev_filter):
                return True
    return False


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from LLM response, stripping fences.

    Raises ``LLMParseError`` (a typed exception) on failure so callers can
    distinguish parse problems from genuine LLM/network errors.
    """
    import json
    raw = (text or "").strip()
    stripped = raw
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LLMParseError(
            f"could not parse JSON object: {exc.msg} (preview: {raw[:160]!r})"
        ) from exc


def truncate_stage_memory(text: str, max_chars: int = 1500) -> str:
    """Truncate stage memory to max_chars, preserving whole lines."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars - 20].rsplit("\n", 1)[0]
    return cut + "\n...(truncated)"


def as_list(val: Any) -> list:
    """Convert value to list. Returns empty list for None."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


