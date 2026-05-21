"""
Cosine similarity tool: semantic embedding search.

Supports chunk-based embeddings where multiple rows map to the
same doc_id — returns the best chunk score per document (max-pooling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..types import SearchHit


@dataclass
class EmbeddingIndex:
    """Precomputed embedding index over a corpus (chunk-based).

    doc_ids may contain duplicates when multiple chunks belong to the same
    document. id_to_idxs maps each unique doc_id to all its row indices.
    """

    doc_ids: list[str]       # parent doc_id per row
    chunk_keys: list[str]    # chunk key per row (for text lookup)
    embeddings: Any          # np.ndarray (N, dim)
    model: Any               # SentenceTransformer
    id_to_idx: dict[str, int] = field(default_factory=dict)
    id_to_idxs: dict[str, list[int]] = field(default_factory=dict)


def tool_cosine(
    query: str,
    emb_index: EmbeddingIndex,
    max_results: int = 10,
    candidate_ids: list[str] | None = None,
) -> list[SearchHit]:
    """
    Semantic similarity search over the corpus using pre-built embeddings.

    Computes similarity for all chunks and returns the maximum score per
    document (max-pooling over chunks).
    """
    from ..types import SearchHit

    if not query:
        return []

    q_emb = emb_index.model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    q_vec = np.asarray(q_emb[0], dtype=np.float32)

    # Determine which row indices to search
    if candidate_ids:
        if emb_index.id_to_idxs:
            idxs = []
            for c in candidate_ids:
                idxs.extend(emb_index.id_to_idxs.get(c, []))
        else:
            idxs = [emb_index.id_to_idx[c] for c in candidate_ids
                     if c in emb_index.id_to_idx]
    else:
        idxs = list(range(len(emb_index.doc_ids)))

    if not idxs:
        return []

    mat = emb_index.embeddings[idxs]
    scores = mat @ q_vec

    # Max-pool per document: keep best chunk score + chunk_key
    doc_best: dict[str, tuple[float, str]] = {}  # parent → (score, chunk_key)
    for i, row_idx in enumerate(idxs):
        doc_id = emb_index.doc_ids[row_idx]
        s = float(scores[i])
        chunk_key = (emb_index.chunk_keys[row_idx]
                     if emb_index.chunk_keys else doc_id)
        if s > doc_best.get(doc_id, (-1.0, ""))[0]:
            doc_best[doc_id] = (s, chunk_key)

    # Sort by score descending
    sorted_docs = sorted(doc_best.items(), key=lambda x: -x[1][0])[:max_results]

    hits: list[SearchHit] = []
    for doc_id, (score, chunk_key) in sorted_docs:
        hits.append(SearchHit(doc_id, score, "", chunk_key=chunk_key))
    return hits
