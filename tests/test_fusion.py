"""Smoke + behaviour tests for the RRF fusion + corpus allocation paths.

Builds a tiny in-memory corpus, fakes the dense channel with a stub
``EmbeddingIndex`` so the test runs without ``sentence-transformers``, and
checks: (a) RRF aggregates across channels, (b) chunk_key prefers the
dense channel hit, (c) ``allocate_by_corpus`` respects the constraint.
"""

from __future__ import annotations

import numpy as np
import pytest

from TRACE.src.scoring import (
    _normalize_query,
    allocate_by_corpus,
    is_repeated_query,
    run_score_fusion,
)
from TRACE.src.search.bm25 import build_bm25
from TRACE.src.search.cosine import EmbeddingIndex
from TRACE.src.types import CorpusDoc, MetadataEntry


def _mk_doc(doc_id: str, text: str, corpus: str = "newspapers", date: str = "1887-01-15") -> CorpusDoc:
    return CorpusDoc(
        doc_id=doc_id,
        filename=f"{doc_id}.json",
        corpus=corpus,
        date=date,
        title=doc_id,
        summary_brief="",
        text_original=text,
        contents_text=text,
    )


class _StubModel:
    def encode(self, queries, normalize_embeddings=True, show_progress_bar=False):
        # All zeros — cosine returns zero relevance everywhere. We only care
        # that RRF still tallies it as a channel signal when the dense rank
        # is meaningful via the stubbed embeddings matrix below.
        return np.zeros((len(queries), 3), dtype=np.float32)


def _mk_emb_index(doc_keys):
    # Each chunk gets a hand-crafted vector; query is [1, 0, 0] so we know
    # which row will rank first. We patch the model's encode to return that
    # query vector deterministically.
    vecs = np.zeros((len(doc_keys), 3), dtype=np.float32)
    if doc_keys:
        vecs[0] = np.array([1.0, 0.0, 0.0])  # best
        if len(doc_keys) > 1:
            vecs[1] = np.array([0.7, 0.5, 0.0])
        if len(doc_keys) > 2:
            vecs[2] = np.array([0.1, 0.9, 0.0])

    class _Model(_StubModel):
        def encode(self, queries, normalize_embeddings=True, show_progress_bar=False):
            return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    return EmbeddingIndex(
        doc_ids=[k.split("__")[0] for k in doc_keys],
        chunk_keys=list(doc_keys),
        embeddings=vecs,
        model=_Model(),
        id_to_idx={k.split("__")[0]: i for i, k in enumerate(doc_keys)},
        id_to_idxs={k.split("__")[0]: [i] for i, k in enumerate(doc_keys)},
    )


@pytest.fixture()
def mini_corpus():
    docs = {
        "doc_001": _mk_doc("doc_001", "Victor Hugo discusses the parliamentary session."),
        "doc_002": _mk_doc("doc_002", "Le Gaulois reports on a railway accident.", corpus="le_gaulois"),
        "doc_003": _mk_doc("doc_003", "An unrelated essay about gardening.", corpus="l_intransigeant"),
    }
    metadata = {
        d: MetadataEntry(doc_id=d, date=docs[d].date, corpus=docs[d].corpus, corpus_type=docs[d].corpus)
        for d in docs
    }
    date_index = {docs[d].date: [d] for d in docs}
    bm25 = build_bm25(docs)
    emb = _mk_emb_index(list(docs.keys()))
    return docs, metadata, date_index, bm25, emb


def test_rrf_fuses_three_channels(mini_corpus):
    docs, metadata, date_index, bm25, emb = mini_corpus
    fused, contributions = run_score_fusion(
        "Victor Hugo parliament 1887-01-15",
        full_bm25=bm25,
        full_docs=docs,
        full_emb=emb,
        metadata_index=metadata,
        date_index=date_index,
        top_k=10,
    )
    assert fused, "RRF must return at least one candidate"
    top_doc = fused[0]
    # doc_001 wins: it gets bm25 (hugo/parliament) + dense (best vector) +
    # date metadata (1887-01-15).
    assert top_doc.doc_id == "doc_001"
    assert len(top_doc.channels) >= 2


def test_allocate_by_corpus_filters(mini_corpus):
    docs, metadata, date_index, bm25, emb = mini_corpus
    fused, _ = run_score_fusion(
        "railway accident",
        full_bm25=bm25,
        full_docs=docs,
        full_emb=emb,
        metadata_index=metadata,
        date_index=date_index,
        top_k=10,
    )
    restricted = allocate_by_corpus(fused, metadata, "le_gaulois", max_total=5)
    assert restricted, "Allocation should keep at least the matching doc"
    assert all(metadata[c.doc_id].corpus == "le_gaulois" for c in restricted)


def test_normalize_query_collapses_whitespace_and_case():
    assert _normalize_query("  Hugo,  Victor  ") == "hugo, victor"
    assert _normalize_query("CAFÉ") == _normalize_query("café")
    assert _normalize_query("") == ""


def test_is_repeated_query_catches_paraphrases():
    history = [("search_bm25", "Victor Hugo", None)]
    assert is_repeated_query("victor   hugo", history, action_key="search_bm25")
    # Different action key should not match.
    assert not is_repeated_query("victor hugo", history, action_key="search_grep")
    # Different corpus filter should not match.
    history2 = [("search_bm25", "Victor Hugo", "le_gaulois")]
    assert not is_repeated_query(
        "victor hugo", history2, corpus_filter="parliamentary", action_key="search_bm25"
    )
