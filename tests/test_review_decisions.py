"""Decision-table tests for ``process_review_decisions``.

Covers the four canonical outcomes (accept / hold / reject / undecided-auto)
and the chunk-vs-parent normalisation that drives the accept/hold/reject
bookkeeping.
"""

from __future__ import annotations

import pytest

from TRACE.src.agent import process_review_decisions
from TRACE.src.types import CorpusDoc, RetrievalState


def _mk_chunked_corpus():
    return {
        "doc_a__chunk_0": CorpusDoc(
            doc_id="doc_a", filename="a.json", corpus="newspapers",
            date="1887-01-15", title="A", summary_brief="",
            text_original="...", contents_text="...",
        ),
        "doc_a__chunk_1": CorpusDoc(
            doc_id="doc_a", filename="a.json", corpus="newspapers",
            date="1887-01-15", title="A", summary_brief="",
            text_original="...", contents_text="...",
        ),
        "doc_b__chunk_0": CorpusDoc(
            doc_id="doc_b", filename="b.json", corpus="parliamentary",
            date="1887-01-16", title="B", summary_brief="",
            text_original="...", contents_text="...",
        ),
    }


@pytest.fixture()
def state_with_pending():
    state = RetrievalState(query="q", sub_question_index=0)
    state.pending_review = ["doc_a__chunk_0", "doc_b__chunk_0"]
    state.phase = "PENDING_REVIEW"
    return state


def test_accept_promotes_parent(state_with_pending):
    docs = _mk_chunked_corpus()
    n_acc, n_rej, n_held = process_review_decisions(
        state_with_pending,
        [{"doc_id": "doc_a", "accept": True, "reason": "matches"}],
        ["doc_a__chunk_0", "doc_b__chunk_0"],
        step_n=1,
        chunked_docs=docs,
        allow_hold=True,
    )
    accepted_parents = {a.doc_id for a in state_with_pending.accepted}
    assert accepted_parents == {"doc_a"}
    assert n_acc == 1
    # doc_b was undecided → auto-held.
    held_parents = {docs[h].doc_id for h in state_with_pending.held}
    assert held_parents == {"doc_b"}


def test_reject_chunk_only_not_parent(state_with_pending):
    docs = _mk_chunked_corpus()
    process_review_decisions(
        state_with_pending,
        [{"doc_id": "doc_a", "accept": False, "hold": False, "reason": "no"}],
        ["doc_a__chunk_0", "doc_b__chunk_0"],
        step_n=1,
        chunked_docs=docs,
        allow_hold=True,
    )
    assert "doc_a__chunk_0" in state_with_pending.rejected
    # Sibling chunk doc_a__chunk_1 is not in rejected (other chunks of the
    # same parent must remain discoverable later).
    assert "doc_a__chunk_1" not in state_with_pending.rejected


def test_hold_keeps_chunk_key(state_with_pending):
    docs = _mk_chunked_corpus()
    process_review_decisions(
        state_with_pending,
        [{"doc_id": "doc_b", "hold": True, "reason": "maybe"}],
        ["doc_a__chunk_0", "doc_b__chunk_0"],
        step_n=1,
        chunked_docs=docs,
        allow_hold=True,
    )
    # doc_b held under its chunk key; doc_a undecided → also held.
    assert "doc_b__chunk_0" in state_with_pending.held
    assert any(h.startswith("doc_a") for h in state_with_pending.held)


def test_accept_then_reject_other_chunk_keeps_parent(state_with_pending):
    docs = _mk_chunked_corpus()
    # Pretend doc_a__chunk_1 was previously rejected; an accept on
    # doc_a__chunk_0 must clear that rejection at the parent level.
    state_with_pending.rejected.add("doc_a__chunk_1")
    process_review_decisions(
        state_with_pending,
        [{"doc_id": "doc_a", "accept": True, "reason": "ok"}],
        ["doc_a__chunk_0", "doc_b__chunk_0"],
        step_n=2,
        chunked_docs=docs,
        allow_hold=True,
    )
    # The sibling-chunk rejection should be lifted now that the parent is
    # accepted.
    rejected_parents = {docs[ck].doc_id for ck in state_with_pending.rejected if ck in docs}
    assert "doc_a" not in rejected_parents
