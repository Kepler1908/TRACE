# types.py
"""Dataclasses and constants for the retrieval pipeline. Self-contained."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .search.bm25 import BM25Index
from .search.cosine import EmbeddingIndex


# ---------------------------------------------------------------------------
# Typed exceptions — replace bare `except Exception` so silent failures
# surface as warnings instead of masquerading as empty results.
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """Base class for TRACE pipeline errors."""


class LLMParseError(PipelineError, ValueError):
    """LLM returned content that could not be parsed as the expected JSON shape.

    Also inherits from ``ValueError`` so legacy ``except ValueError`` blocks
    continue to work unchanged.
    """


class LLMSchemaError(PipelineError, ValueError):
    """LLM output parsed as JSON but failed schema validation."""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBQ_MAX_STEPS = 13
MAX_SUB_QUESTIONS = 4
ENABLE_SEARCH_EXPANSION = True

PER_CHANNEL_MAX = 50
PENDING_REVIEW_CAP = 10
HARD_MAX_STEPS = 15
EMPTY_SEARCH_WARN_THRESHOLD = 3
REVIEW_BATCH_SIZE = 8
DEFAULT_LIMIT = 5

# Chunking
CHUNK_MAX_TOKENS = 1000            # sentence-boundary chunks for BM25/embeddings
CHUNK_OVERLAP_RATIO = 0.12         # 10–15% overlap between consecutive chunks
EMBEDDING_BATCH_SIZE = 64          # build_embeddings batch (1 → 64 is the unlock)

# Feature flags (toggle implementations introduced in the 2026-05 review)
USE_BM25S = True                   # use bm25s lib if available, fall back to inverted-list custom impl
USE_ANN_INDEX = True               # use usearch/faiss HNSW if available, fall back to numpy brute force
USE_CROSS_ENCODER_RERANK = True    # use cross-encoder reranker if available, fall back to LLM rerank
USE_ANTHROPIC_PROMPT_CACHING = True
HELD_REEVAL_SECOND_PASS_THRESHOLD = 0.5  # retrigger held re-eval if |accepted| moved by > 50%
PLANNER_VALIDATION_RETRIES = 2
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"

# Per-call max_tokens caps (avoids wasting decode budget)
AGENT_SEARCH_MAX_TOKENS = 768
AGENT_REVIEW_MAX_TOKENS = 1536
PLANNER_MAX_TOKENS = 2048
RERANK_MAX_TOKENS = 1024
BRIEFING_MAX_TOKENS = 1024
REPLAN_MAX_TOKENS = 1024
HELD_REEVAL_MAX_TOKENS = 1024

# Provider-aware agent limits
_LOCAL_PROVIDERS = {"llama_cpp", "vllm", "sglang"}


@dataclass(frozen=True)
class AgentLimits:
    """Provider-aware token/batch limits."""
    review_batch_size: int = REVIEW_BATCH_SIZE
    agent_search_max_tokens: int = AGENT_SEARCH_MAX_TOKENS
    agent_review_max_tokens: int = AGENT_REVIEW_MAX_TOKENS
    max_parse_errors: int = 5
    # Post-agent pipeline stages
    held_reeval_max_tokens: int = HELD_REEVAL_MAX_TOKENS
    briefing_max_tokens: int = BRIEFING_MAX_TOKENS
    replan_max_tokens: int = REPLAN_MAX_TOKENS
    rerank_max_tokens: int = RERANK_MAX_TOKENS
    narrative_max_tokens: int = 1024


DEFAULT_AGENT_LIMITS = AgentLimits()

LOCAL_AGENT_LIMITS = AgentLimits(
    review_batch_size=6,
    agent_search_max_tokens=1024,
    agent_review_max_tokens=3072,
    max_parse_errors=2,
    held_reeval_max_tokens=2048,
    briefing_max_tokens=2048,
    replan_max_tokens=2048,
    rerank_max_tokens=2048,
    narrative_max_tokens=2048,
)


def get_agent_limits(provider: str | None = None) -> AgentLimits:
    if provider in _LOCAL_PROVIDERS:
        return LOCAL_AGENT_LIMITS
    return DEFAULT_AGENT_LIMITS


PHASE1_TOP_K_RATIO = 0.05
PHASE1_TOP_K_MIN = 20
RRF_K = 60

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "corpus"
CORPUS_JSONL = DATA_DIR / "corpus_dataset.jsonl"
QUESTION_JSONL = DATA_DIR / "question_dataset.jsonl"
CORPUS_DIR = DATA_DIR


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CorpusDoc:
    doc_id: str
    filename: str
    corpus: str
    date: str
    title: str
    summary_brief: str
    text_original: str
    contents_text: str


@dataclass
class MetadataEntry:
    doc_id: str
    date: str
    corpus: str
    corpus_type: str


@dataclass
class SearchHit:
    doc_id: str
    score: float
    snippet: str
    chunk_key: str = ""  # best matching chunk key (for review text lookup)


@dataclass
class FusedCandidate:
    doc_id: str
    rrf_score: float
    channels: list[str]
    snippet: str = ""
    chunk_key: str = ""  # best matching chunk key


@dataclass
class AcceptedDoc:
    doc_id: str
    reason: str
    accepted_at_step: int
    chunk_key: str = ""  # chunk key (for text lookup in chunked_docs)


@dataclass
class QuestionEntry:
    question_id: str
    question_type: str
    question: str
    gold_ids: list[str]


@dataclass
class RetrievalResult:
    question_id: str
    gold_ids: list[str]
    candidate_ids: list[str]
    steps: list[dict[str, Any]] = field(default_factory=list)
    narrative: str = ""
    accepted_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

@dataclass
class RetrievalState:
    """Mutable state for a sub-question's Phase 2 agent loop.

    accepted: parent doc_ids (accept chunk = accept parent doc).
    rejected: chunk_keys (reject chunk ≠ reject parent; other chunks remain discoverable).
    held: chunk_keys (for deferred re-evaluation).
    """
    query: str
    sub_question_index: int
    accepted: list[AcceptedDoc] = field(default_factory=list)
    pending_review: list[str] = field(default_factory=list)
    pending_queue: list[str] = field(default_factory=list)
    rejected: set[str] = field(default_factory=set)
    held: list[str] = field(default_factory=list)
    phase: str = "READY"
    call_history: list[tuple[str, str, str | int | None]] = field(default_factory=list)
    search_plan: dict[str, Any] = field(default_factory=dict)
    step_count: int = 0
    max_steps: int = SUBQ_MAX_STEPS


@dataclass
class SubQuestionResult:
    """Result from one sub-question's retrieval."""
    sub_question: str
    target_corpus: str
    accepted_ids: list[str]
    held_ids: list[str]
    briefing: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    confirmed: bool = False
    confirmed_ids: list[str] = field(default_factory=list)
    accepted_docs: list[AcceptedDoc] = field(default_factory=list)
    spawned: bool = False
