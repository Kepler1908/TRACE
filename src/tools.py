# tools.py
"""Unified tools interface for the retrieval pipeline.

Re-exports from data, scoring, metrics, and search modules so that
agent.py, pipeline.py, and cli.py can import everything from one place.
"""

# Data loading + index building
from .data import (
    load_corpus_jsonl,
    load_questions_jsonl,
    build_full_corpus,
    get_embedding_model,
    load_corpus_embeddings,
)

# Search tools
from .search import (
    BM25Index,
    EmbeddingIndex,
    tool_bm25 as search_bm25,
    tool_cosine as search_semantic,
    tool_grep as search_grep,
    search_date,
    search_date_range,
    build_bm25,
)

# Scoring + Phase 1 RRF
from .scoring import (
    run_score_fusion,
    allocate_by_corpus,
    filter_hits_by_corpus,
    is_repeated_query,
    parse_json_object,
    truncate_stage_memory,
    as_list,
)

# Metrics
from .metrics import (
    recall_at_k,
    gold_rank,
    precision_in_accepted,
    recall_in_accepted,
    compute_metrics,
    print_results_table,
)

# Types + constants
from .types import (
    CORPUS_DIR,
    CORPUS_JSONL,
    QUESTION_JSONL,
    PER_CHANNEL_MAX,
    PENDING_REVIEW_CAP,
    EMPTY_SEARCH_WARN_THRESHOLD,
    HARD_MAX_STEPS,
    PHASE1_TOP_K_MIN,
    PHASE1_TOP_K_RATIO,
    REVIEW_BATCH_SIZE,
    DEFAULT_LIMIT,
)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

from .types import CorpusDoc, MetadataEntry, SearchHit, AcceptedDoc


def load_review_batch(
    chunk_keys: list[str],
    chunked_docs: dict[str, CorpusDoc],
    metadata_index: dict[str, MetadataEntry],
) -> str:
    """Load chunk text for a review batch.

    Chunk-native: shows the chunk text directly (~3000 chars per chunk).
    No truncation needed — chunks are already the right granularity.
    Header shows parent doc_id (the agent never sees chunk keys).
    """
    lines = []
    for key in chunk_keys:
        doc = chunked_docs.get(key)
        if not doc:
            lines.append(f"=== {key} ===\n(not found)")
            continue
        meta = metadata_index.get(key)
        meta_str = f"{meta.date or '?'} {meta.corpus or '?'}" if meta else ""
        lines.append(f"=== {doc.doc_id} | {meta_str} ===\n{doc.text_original}")
    return "\n\n".join(lines)


def merge_subquestion_results(
    sub_results: list,
) -> tuple[list[str], list[str]]:
    """Merge accepted picks from all sub-questions.

    Returns (merged_ids sorted by cross-sub-Q count, removed_ids).
    """
    from collections import defaultdict

    doc_counts: dict[str, int] = defaultdict(int)
    confirmed_set: set[str] = set()
    removed_ids: list[str] = []

    for r in sub_results:
        if r.confirmed:
            for doc_id in r.confirmed_ids:
                doc_counts[doc_id] += 1
                confirmed_set.add(doc_id)
        else:
            removed_ids.extend(r.accepted_ids)

    merged = sorted(confirmed_set, key=lambda d: (-doc_counts[d], d))
    return merged, removed_ids


async def generate_narrative(
    entry,
    accepted_docs: list[AcceptedDoc],
    full_docs: dict[str, CorpusDoc],
    metadata_index: dict[str, MetadataEntry],
    tracker,
    quiet: bool,
    limits=None,
) -> str:
    """Generate a narrative answer from accepted documents.

    Uses full_docs (not chunks) for complete document context.
    Splits into batches to stay within context window, then synthesizes.
    """
    from ..util.llm_client import LLMClient
    from ..util.config import get_settings
    from .prompts import (
        NARRATIVE_SYSTEM, NARRATIVE_SINGLE_BATCH,
        NARRATIVE_EXTRACT_BATCH, NARRATIVE_SYNTHESIZE,
    )

    if not accepted_docs:
        return ""

    doc_entries: list[tuple[str, str]] = []
    for i, a in enumerate(accepted_docs[:10], 1):
        doc = full_docs.get(a.doc_id)
        if doc:
            text = doc.text_original or ""
            meta = metadata_index.get(a.chunk_key) or metadata_index.get(a.doc_id)
            meta_str = f"{meta.date} {meta.corpus}" if meta else ""
            header = f"Document {i} ({a.doc_id}, {meta_str})"
            doc_entries.append((header, text))

    if not doc_entries:
        return ""

    from . import types as types_mod
    _limits = limits or types_mod.get_agent_limits(get_settings().llm.provider)
    client = LLMClient(get_settings().llm)

    MAX_BATCH_CHARS = 100_000
    batches: list[list[tuple[str, str]]] = []
    current_batch: list[tuple[str, str]] = []
    current_chars = 0
    for header, text in doc_entries:
        entry_chars = len(header) + len(text) + 50
        if current_batch and current_chars + entry_chars > MAX_BATCH_CHARS:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append((header, text))
        current_chars += entry_chars
    if current_batch:
        batches.append(current_batch)

    if len(batches) == 1:
        doc_texts = [f"{h}:\n{t}" for h, t in batches[0]]
        prompt = NARRATIVE_SINGLE_BATCH.format(
            question=entry.question,
            doc_block="\n\n---\n\n".join(doc_texts),
        )
        try:
            response = await client.complete(
                prompt, system=NARRATIVE_SYSTEM,
                max_tokens=_limits.narrative_max_tokens,
            )
            tracker.record(response.usage)
            if not quiet:
                print(f"  [Narrative] {len(response.content)} chars")
            return response.content
        except Exception:
            if not quiet:
                print("  [Narrative] generation failed; empty narrative")
            return ""

    batch_summaries: list[str] = []
    for bi, batch in enumerate(batches, 1):
        doc_texts = [f"{h}:\n{t}" for h, t in batch]
        prompt = NARRATIVE_EXTRACT_BATCH.format(
            question=entry.question,
            batch_idx=bi,
            batch_total=len(batches),
            doc_block="\n\n---\n\n".join(doc_texts),
        )
        try:
            response = await client.complete(
                prompt, system=NARRATIVE_SYSTEM,
                max_tokens=_limits.narrative_max_tokens,
            )
            tracker.record(response.usage)
            batch_summaries.append(response.content)
        except Exception:
            if not quiet:
                print(f"  [Narrative] batch {bi} failed")

    if not batch_summaries:
        return ""

    synth_prompt = NARRATIVE_SYNTHESIZE.format(
        question=entry.question,
        batch_block="\n\n---\n\n".join(
            f"Batch {i+1}:\n{s}" for i, s in enumerate(batch_summaries)
        ),
    )
    try:
        response = await client.complete(
            synth_prompt, system=NARRATIVE_SYSTEM,
            max_tokens=_limits.narrative_max_tokens,
        )
        tracker.record(response.usage)
        if not quiet:
            print(f"  [Narrative] {len(response.content)} chars ({len(batches)} batches)")
        return response.content
    except Exception:
        if not quiet:
            print("  [Narrative] synthesis failed; using first batch")
        return batch_summaries[0]
