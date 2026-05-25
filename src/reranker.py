"""Cross-encoder reranker.

Used by the final-rerank stage when ``USE_CROSS_ENCODER_RERANK`` is enabled
and ``sentence_transformers.CrossEncoder`` is installed. Falls back to the
LLM-based ``rerank_accepted`` otherwise.

We use ``BAAI/bge-reranker-v2-m3`` by default — multilingual, ~3M params for
the base, trained on diverse query/doc pairs; good fit for a 19th-century
French corpus where the LLM reranker would otherwise burn ~50× the cost.
"""

from __future__ import annotations

from typing import Any

from ..util.logging import get_logger
from .types import CROSS_ENCODER_MODEL, USE_CROSS_ENCODER_RERANK

logger = get_logger(__name__)

_CROSS_ENCODER: Any = None
_LOAD_ATTEMPTED = False


def _load_cross_encoder() -> Any:
    """Lazy-load the cross-encoder once per process.

    Returns ``None`` on any failure (missing dep, model fetch error). Callers
    are expected to fall back to the LLM rerank when this happens.
    """
    global _CROSS_ENCODER, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _CROSS_ENCODER
    _LOAD_ATTEMPTED = True

    if not USE_CROSS_ENCODER_RERANK:
        return None

    try:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("cross_encoder.loaded", model=CROSS_ENCODER_MODEL)
    except Exception as exc:
        logger.warning(
            "cross_encoder.load_failed",
            model=CROSS_ENCODER_MODEL,
            error=str(exc)[:200],
        )
        _CROSS_ENCODER = None
    return _CROSS_ENCODER


def cross_encoder_available() -> bool:
    return _load_cross_encoder() is not None


def rerank_with_cross_encoder(
    question: str,
    candidates: list[str],
    snippets: dict[str, str],
) -> list[str] | None:
    """Rerank candidates by cross-encoder relevance score.

    ``snippets`` is doc_id -> short passage shown to the encoder. Returns
    None on failure so the caller can fall back without further branching.
    """
    if not candidates:
        return []
    encoder = _load_cross_encoder()
    if encoder is None:
        return None
    pairs = [(question, snippets.get(doc_id, "")) for doc_id in candidates]
    try:
        scores = encoder.predict(pairs)
    except Exception as exc:
        logger.warning("cross_encoder.predict_failed", error=str(exc)[:200])
        return None
    paired = sorted(
        zip(candidates, [float(s) for s in scores]),
        key=lambda x: x[1],
        reverse=True,
    )
    return [doc_id for doc_id, _ in paired]
