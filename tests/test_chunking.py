"""Chunking behaviour: overlap is added, sizes respect the token budget."""

from __future__ import annotations

from TRACE.src.data import _chunk_text_by_sentences, _count_tokens


_SENTS = [
    "Alpha bravo charlie delta echo foxtrot golf hotel.",
    "India juliet kilo lima mike november oscar papa.",
    "Quebec romeo sierra tango uniform victor whiskey xray.",
    "Yankee zulu one two three four five six seven.",
    "Aardvark badger cobra dingo elk frog gazelle hyena.",
]


def test_short_text_returns_single_chunk():
    text = "Short text."
    chunks = _chunk_text_by_sentences(text, max_tokens=100, overlap_ratio=0.1)
    assert chunks == ["Short text."]


def test_long_text_is_split():
    text = " ".join(_SENTS)
    chunks = _chunk_text_by_sentences(text, max_tokens=12, overlap_ratio=0.0)
    assert len(chunks) >= 2
    # No chunk should grossly exceed the budget (small slack for sentence
    # boundaries — ±1 sentence).
    for chunk in chunks:
        assert _count_tokens(chunk) <= 24


def test_overlap_carries_trailing_sentence():
    text = " ".join(_SENTS)
    no_overlap = _chunk_text_by_sentences(text, max_tokens=12, overlap_ratio=0.0)
    with_overlap = _chunk_text_by_sentences(text, max_tokens=12, overlap_ratio=0.3)
    # With overlap, the second chunk should start with content from the
    # tail of the first.
    if len(with_overlap) >= 2 and len(no_overlap) >= 2:
        # The second overlap chunk should contain at least one token that
        # also appears in the first chunk.
        first_tokens = set(no_overlap[0].split())
        second_tokens = set(with_overlap[1].split())
        assert first_tokens & second_tokens, "overlap must carry context across boundary"
