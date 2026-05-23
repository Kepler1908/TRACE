# data.py
"""Data loading and index building from JSONL files."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .types import (
    CorpusDoc, MetadataEntry, QuestionEntry,
    CORPUS_JSONL, QUESTION_JSONL, CORPUS_DIR,
    CHUNK_MAX_TOKENS, CHUNK_OVERLAP_RATIO,
)

# Optional real tokenizer — fall back to whitespace if tiktoken not installed.
try:
    import tiktoken
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # ImportError or runtime model fetch failure
    _TIKTOKEN_ENC = None


def _count_tokens(text: str) -> int:
    """Token count for chunking budget.

    Uses cl100k_base when tiktoken is available (close to BPE behaviour of
    most modern encoders, including the SentenceTransformer models we
    typically use); falls back to whitespace word count otherwise.
    """
    if _TIKTOKEN_ENC is not None:
        try:
            return len(_TIKTOKEN_ENC.encode(text, disallowed_special=()))
        except Exception:
            pass
    return len(text.split())

# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def load_corpus_jsonl(path: str | Path | None = None) -> list[dict]:
    """Load corpus documents from JSONL.

    Expected fields per line:
      doc_id, corpus, date, title, text_original
    Optional: filename, summary_brief, contents_text
    """
    p = Path(path) if path else CORPUS_JSONL
    records = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_questions_jsonl(path: str | Path | None = None) -> list[QuestionEntry]:
    """Load questions from JSONL as QuestionEntry objects.

    Expected fields per line:
      question_id, question_type, question, gold_ids
    """
    p = Path(path) if path else QUESTION_JSONL
    entries = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            entries.append(QuestionEntry(
                question_id=raw.get("question_id", ""),
                question_type=raw.get("question_type", ""),
                question=raw.get("question", ""),
                gold_ids=raw.get("gold_ids", []),
            ))
    return entries


# ---------------------------------------------------------------------------
# Index building from JSONL
# ---------------------------------------------------------------------------

def build_full_corpus(
    path: str | Path | None = None,
    cache: dict[str, CorpusDoc] | None = None,
) -> dict[str, CorpusDoc]:
    """Build doc_id → CorpusDoc dict from JSONL corpus.

    Each JSONL line must contain at minimum: doc_id, text_original.
    """
    entries = load_corpus_jsonl(path)
    docs = {}
    for entry in entries:
        doc_id = entry["doc_id"]
        text = entry.get("text_original", "")
        doc = CorpusDoc(
            doc_id=doc_id,
            filename=entry.get("filename", f"{doc_id}.json"),
            corpus=entry.get("corpus", ""),
            date=entry.get("date", ""),
            title=entry.get("title", ""),
            summary_brief=entry.get("summary_brief", "")[:300],
            text_original=text,
            contents_text=entry.get("contents_text", text),
        )
        docs[doc_id] = doc
        if cache is not None:
            cache[doc_id] = doc
    return docs


# ---------------------------------------------------------------------------
# Sentence-boundary chunking
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r"(?<=[.!?…»\"])\s+")


def _chunk_text_by_sentences(
    text: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_ratio: float = CHUNK_OVERLAP_RATIO,
) -> list[str]:
    """Split text into chunks of ~max_tokens preserving sentence boundaries.

    Uses ``_count_tokens`` (tiktoken when available, whitespace otherwise) so
    chunk sizes correlate with what the embedding model actually sees,
    instead of the prior raw word count which over-shot on CJK/inflected
    text.

    A trailing tail of size ``max_tokens * overlap_ratio`` is carried into
    the next chunk so a proper noun cut at a chunk boundary remains visible
    in both sides.
    """
    sentences = _SENT_RE.split(text.strip())
    if not sentences:
        return [text.strip()] if text.strip() else []

    overlap_budget = max(0, int(max_tokens * max(0.0, overlap_ratio)))
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    current_sent_tokens: list[int] = []

    def _flush_with_overlap() -> None:
        nonlocal current, current_tokens, current_sent_tokens
        if not current:
            return
        chunks.append(" ".join(current))
        if overlap_budget <= 0:
            current, current_tokens, current_sent_tokens = [], 0, []
            return
        # Carry trailing sentences whose tokens fit the overlap budget.
        carry_sents: list[str] = []
        carry_tokens: list[int] = []
        running = 0
        for sent, tok in zip(reversed(current), reversed(current_sent_tokens)):
            if running + tok > overlap_budget and carry_sents:
                break
            carry_sents.append(sent)
            carry_tokens.append(tok)
            running += tok
            if running >= overlap_budget:
                break
        current = list(reversed(carry_sents))
        current_sent_tokens = list(reversed(carry_tokens))
        current_tokens = sum(current_sent_tokens)

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_tokens = _count_tokens(sent)

        if sent_tokens >= max_tokens:
            if current:
                chunks.append(" ".join(current))
                current, current_tokens, current_sent_tokens = [], 0, []
            chunks.append(sent)
            continue

        if current_tokens + sent_tokens > max_tokens and current:
            _flush_with_overlap()

        current.append(sent)
        current_sent_tokens.append(sent_tokens)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text.strip()]


def build_chunked_corpus(
    full_docs: dict[str, CorpusDoc],
    max_tokens: int = CHUNK_MAX_TOKENS,
) -> dict[str, CorpusDoc]:
    """Build chunk-level corpus for BM25/embedding indexing.

    Short documents (<=max_tokens words) are kept as-is.
    Long documents are split at sentence boundaries into ~max_tokens-word chunks.
    Each chunk's CorpusDoc.doc_id stays as the parent doc_id (for gold matching).
    Dict keys use ``{doc_id}__chunk_{N}`` for internal uniqueness.
    """
    chunked: dict[str, CorpusDoc] = {}
    for doc_id, doc in full_docs.items():
        text = doc.text_original or doc.contents_text or ""
        token_count = _count_tokens(text)

        if token_count <= max_tokens:
            chunked[doc_id] = doc
            continue

        parts = _chunk_text_by_sentences(text, max_tokens)
        for i, chunk_text in enumerate(parts):
            chunk_key = f"{doc_id}__chunk_{i}"
            chunked[chunk_key] = CorpusDoc(
                doc_id=doc_id,  # parent ID — for gold matching
                filename=doc.filename,
                corpus=doc.corpus,
                date=doc.date,
                title=doc.title,
                summary_brief=doc.summary_brief,
                text_original=chunk_text,
                contents_text=chunk_text,
            )
    return chunked


def build_chunked_date_index(
    chunked_docs: dict[str, CorpusDoc],
) -> dict[str, list[str]]:
    """Build date → [chunk_keys] index from chunked corpus."""
    date_index: dict[str, list[str]] = {}
    for chunk_key, doc in chunked_docs.items():
        if doc.date:
            date_index.setdefault(doc.date, []).append(chunk_key)
    return date_index


def build_chunked_metadata_index(
    chunked_docs: dict[str, CorpusDoc],
) -> dict[str, MetadataEntry]:
    """Build chunk_key → MetadataEntry from chunked corpus."""
    metadata_index: dict[str, MetadataEntry] = {}
    for chunk_key, doc in chunked_docs.items():
        metadata_index[chunk_key] = MetadataEntry(
            doc_id=doc.doc_id,
            date=doc.date,
            corpus=doc.corpus,
            corpus_type=doc.corpus,
        )
    return metadata_index


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def get_embedding_model(model_name: str | None = None):
    """Load the embedding model on CPU for query-time encoding.

    Corpus embeddings are pre-cached (build_embeddings.py), so the model
    is only needed to encode short query strings at search time. Loading
    on CPU avoids GPU OOM on cards where the model barely fits.

    Uses EMBEDDING_MODEL from .env if model_name is not provided.
    """
    if model_name is None:
        from ..util.config import get_settings
        model_name = get_settings().embedding.model
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            model_name,
            device="cpu",
            model_kwargs={"torch_dtype": torch.float16},
        )
        return model
    except Exception:
        return None


def load_corpus_embeddings(
    docs: dict[str, CorpusDoc],
    embedding_model: Any,
    corpus_dir: Path | None = None,
):
    """Load pre-computed embeddings from cache (.npy + doc_ids.json)."""
    if embedding_model is None:
        return None

    from .search.cosine import EmbeddingIndex
    import numpy as np
    from collections import defaultdict

    cache = corpus_dir or CORPUS_DIR
    emb_path = cache / "embeddings_cache.npy"
    ids_path = cache / "embeddings_doc_ids.json"

    if not emb_path.exists() or not ids_path.exists():
        return None

    try:
        cached_ids = json.loads(ids_path.read_text("utf-8"))
        cached_emb = np.load(str(emb_path))
    except Exception:
        return None

    if len(cached_ids) != cached_emb.shape[0]:
        return None

    # Build index mappings (chunk-aware: id_to_idxs for max-pooling)
    id_to_idx: dict[str, int] = {}
    id_to_idxs: dict[str, list[int]] = defaultdict(list)
    # Reconstruct chunk_keys: for parent with N rows, keys are
    # {parent}__chunk_0 .. {parent}__chunk_{N-1}. Single-row = parent itself.
    parent_count: dict[str, int] = defaultdict(int)
    chunk_keys: list[str] = []
    for i, did in enumerate(cached_ids):
        id_to_idx.setdefault(did, i)
        id_to_idxs[did].append(i)
        idx_within = parent_count[did]
        parent_count[did] += 1
        if len(id_to_idxs[did]) == 1:
            # First occurrence — might be single-row (set key = did)
            chunk_keys.append(did)
        else:
            # Multi-chunk: fix previous entry if needed, then add current
            if idx_within == 1:
                # Second occurrence — retroactively fix the first
                first_idx = id_to_idxs[did][0]
                chunk_keys[first_idx] = f"{did}__chunk_0"
            chunk_keys.append(f"{did}__chunk_{idx_within}")

    return EmbeddingIndex(
        doc_ids=cached_ids,
        chunk_keys=chunk_keys,
        embeddings=cached_emb,
        model=embedding_model,
        id_to_idx=id_to_idx,
        id_to_idxs=dict(id_to_idxs),
    )


