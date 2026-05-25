"""
Cosine similarity tool: semantic embedding search.

Supports chunk-based embeddings where multiple rows map to the
same doc_id — returns the best chunk score per document (max-pooling).

Approximate-nearest-neighbour acceleration: if ``usearch`` or ``faiss-cpu``
is available *and* ``USE_ANN_INDEX`` is enabled, an HNSW index is built
once at first use and reused for every query. Falls back to numpy brute
force when neither is installed or for the constrained ``candidate_ids``
path (where the ANN index would only see a subset).
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

    ``_ann`` holds a lazily-built HNSW index (usearch or faiss). It is
    bypassed when callers pass ``candidate_ids`` — the ANN index is built
    over the full corpus only.
    """

    doc_ids: list[str]       # parent doc_id per row
    chunk_keys: list[str]    # chunk key per row (for text lookup)
    embeddings: Any          # np.ndarray (N, dim)
    model: Any               # SentenceTransformer
    id_to_idx: dict[str, int] = field(default_factory=dict)
    id_to_idxs: dict[str, list[int]] = field(default_factory=dict)
    _ann: Any = None
    _ann_backend: str = ""  # "usearch" | "faiss" | "" (none)
    _ann_attempted: bool = False


def _try_build_ann(emb_index: EmbeddingIndex) -> None:
    """Best-effort build of an HNSW index. Sets _ann/_ann_backend on success.

    Called once per EmbeddingIndex; subsequent queries reuse it. Failures
    (lib missing, OOM, dim mismatch) are swallowed and the brute-force path
    is used instead.
    """
    if emb_index._ann_attempted:
        return
    emb_index._ann_attempted = True

    try:
        from ..types import USE_ANN_INDEX
    except Exception:
        return
    if not USE_ANN_INDEX:
        return

    mat = np.ascontiguousarray(emb_index.embeddings, dtype=np.float32)
    n, dim = mat.shape
    if n < 1000:
        # Brute force is already < a few ms; skip the build cost.
        return

    # Try usearch first (lighter dep, pure-Python install).
    try:
        from usearch.index import Index as _UsearchIndex

        ann = _UsearchIndex(ndim=dim, metric="cos", dtype="f32")
        ann.add(np.arange(n, dtype=np.int64), mat)
        emb_index._ann = ann
        emb_index._ann_backend = "usearch"
        return
    except Exception:
        pass

    try:
        import faiss  # type: ignore

        # Cosine over L2-normalised vectors == inner-product, which HNSWFlat
        # supports directly. Embeddings written by build_embeddings.py are
        # already L2-normalised, so no extra normalisation step is needed.
        ann = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        ann.hnsw.efConstruction = 80
        ann.hnsw.efSearch = 64
        ann.add(mat)
        emb_index._ann = ann
        emb_index._ann_backend = "faiss"
        return
    except Exception:
        return


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
        idxs = None  # full-corpus path may use ANN

    # Fast path: full-corpus query with an HNSW index. We over-fetch
    # (max_results * 4 chunks) before max-pooling, then trim to max_results
    # parent docs — empirically this preserves recall vs. brute force on
    # the BGE / E5 / Qwen embedding families.
    use_ann = idxs is None
    if use_ann:
        _try_build_ann(emb_index)

    if use_ann and emb_index._ann is not None:
        k = max(max_results * 4, max_results + 10)
        try:
            if emb_index._ann_backend == "usearch":
                matches = emb_index._ann.search(q_vec, count=k)
                row_idxs = list(matches.keys)
                row_scores = [float(d) for d in matches.distances]
                # usearch cos-metric returns DISTANCE (1 - sim) → flip.
                row_scores = [1.0 - s for s in row_scores]
            else:  # faiss
                D, I = emb_index._ann.search(q_vec.reshape(1, -1), k)
                row_idxs = [int(x) for x in I[0] if x != -1]
                row_scores = [float(s) for s in D[0][: len(row_idxs)]]
        except Exception:
            # ANN failed at query time — fall back to brute force.
            row_idxs, row_scores = [], []
            use_ann = False

        if row_idxs:
            doc_best: dict[str, tuple[float, str]] = {}
            for row_idx, s in zip(row_idxs, row_scores):
                if row_idx < 0 or row_idx >= len(emb_index.doc_ids):
                    continue
                doc_id = emb_index.doc_ids[row_idx]
                chunk_key = (emb_index.chunk_keys[row_idx]
                             if emb_index.chunk_keys else doc_id)
                if s > doc_best.get(doc_id, (-1.0, ""))[0]:
                    doc_best[doc_id] = (s, chunk_key)
            sorted_docs = sorted(doc_best.items(), key=lambda x: -x[1][0])[:max_results]
            return [SearchHit(d, sc, "", chunk_key=ck)
                    for d, (sc, ck) in sorted_docs]

    # Brute-force path (constrained candidate set or ANN unavailable).
    if idxs is None:
        idxs = list(range(len(emb_index.doc_ids)))
    if not idxs:
        return []

    mat = emb_index.embeddings[idxs]
    scores = mat @ q_vec

    doc_best = {}
    for i, row_idx in enumerate(idxs):
        doc_id = emb_index.doc_ids[row_idx]
        s = float(scores[i])
        chunk_key = (emb_index.chunk_keys[row_idx]
                     if emb_index.chunk_keys else doc_id)
        if s > doc_best.get(doc_id, (-1.0, ""))[0]:
            doc_best[doc_id] = (s, chunk_key)

    sorted_docs = sorted(doc_best.items(), key=lambda x: -x[1][0])[:max_results]
    return [SearchHit(doc_id, score, "", chunk_key=chunk_key)
            for doc_id, (score, chunk_key) in sorted_docs]
