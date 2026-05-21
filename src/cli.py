# cli.py
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from ..util import llm_client as llm_client_mod
from ..util.llm_client import TokenTracker
from ..util.config import get_settings

from . import tools
from . import types as types_mod
from .pipeline import run_retrieval

async def main_async(args: argparse.Namespace) -> None:
    if args.thinking:
        llm_client_mod.DEEPSEEK_THINKING_ENABLED = True
        types_mod.PLANNER_MAX_TOKENS = 65536
        types_mod.DEFAULT_AGENT_LIMITS = types_mod.AgentLimits(
            agent_search_max_tokens=65536,
            agent_review_max_tokens=65536,
            rerank_max_tokens=65536,
            briefing_max_tokens=65536,
            replan_max_tokens=65536,
            held_reeval_max_tokens=65536,
        )
    t0 = time.monotonic()

    from .data import (
        build_full_corpus, build_chunked_corpus,
        build_chunked_date_index, build_chunked_metadata_index,
        get_embedding_model, load_corpus_embeddings, load_questions_jsonl,
    )

    print("Building corpus from JSONL...")
    full_docs = build_full_corpus()
    print(f"  {len(full_docs)} documents loaded")

    print("Building chunked corpus + indexes...")
    chunked_docs = build_chunked_corpus(full_docs)
    from .search.bm25 import build_bm25
    full_bm25 = build_bm25(chunked_docs)
    date_index = build_chunked_date_index(chunked_docs)
    metadata_index = build_chunked_metadata_index(chunked_docs)
    print(f"  {len(full_docs)} docs → {len(chunked_docs)} chunks, "
          f"{len(date_index)} dates, {len(metadata_index)} metadata")

    print("Loading embedding model...")
    embedding_model = get_embedding_model()
    if embedding_model:
        print("  Embedding model loaded")
    else:
        print("  Embedding model unavailable -- dense channel disabled")

    print("Loading embedding cache...")
    full_emb = load_corpus_embeddings(full_docs, embedding_model)
    if full_emb:
        print(f"  {len(full_emb.doc_ids)} embeddings loaded from cache")
    else:
        print("  Embedding cache not available -- dense channel disabled")

    print("Loading questions...")
    all_questions = load_questions_jsonl()

    answerable = [
        q for q in all_questions
        if q.gold_ids and q.question.strip()
    ]

    if args.type:
        answerable = [q for q in answerable if q.question_type == args.type]

    if not args.all:
        rng = random.Random(args.seed)
        n = args.limit if args.limit is not None else tools.DEFAULT_LIMIT
        answerable = rng.sample(answerable, min(n, len(answerable)))
    elif args.limit is not None:
        answerable = answerable[: args.limit]

    type_map = {q.question_id: q.question_type for q in answerable}
    settings = get_settings()

    phase1_top_k = max(tools.PHASE1_TOP_K_MIN, int(len(full_docs) * tools.PHASE1_TOP_K_RATIO))

    print("\nRAG RETRIEVAL TEST")
    print("=" * 60)
    print(
        f"Corpus: {len(full_docs)} documents | "
        f"Questions: {len(answerable)} answerable / {len(all_questions)} total"
    )
    thinking_label = "ON (3x token budget)" if args.thinking else "OFF"
    print(f"Model: {settings.llm.provider}/{settings.llm.model} | Thinking: {thinking_label}")
    print(
        f"Phase 1: top-{phase1_top_k} | Max steps/subQ: {args.max_steps} | "
        f"Max sub-questions: {args.max_subquestions} | Seed: {args.seed}"
    )
    print(
        f"BM25: {len(chunked_docs)} chunks | "
        f"Dense: {'yes' if full_emb else 'no'}"
    )
    print("=" * 60)

    tracker = TokenTracker()
    concurrency = args.concurrency
    sem = asyncio.Semaphore(concurrency)

    checkpoint_path = Path(args.json_out) if getattr(args, "json_out", None) else None
    checkpoint_data: list[dict] = []
    completed_keys: set[tuple[str, str]] = set()
    if args.resume and checkpoint_path and checkpoint_path.exists():
        checkpoint_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        completed_keys = {(r["question_id"], r["question"]) for r in checkpoint_data}
        print(f"Resuming: {len(completed_keys)} questions already completed, skipping them")
        answerable = [q for q in answerable if (q.question_id, q.question) not in completed_keys]
        if not answerable:
            print("All questions already completed!")
            return

    _checkpoint_lock = asyncio.Lock()
    _n_done = [0]
    _n_failed = [0]
    _total = len(answerable)

    def _make_record(result: types_mod.RetrievalResult, q: types_mod.QuestionEntry) -> dict:
        diag = getattr(result, "_diagnostics", {})
        r10 = tools.recall_at_k(result.gold_ids, result.candidate_ids, 10)
        rank = tools.gold_rank(result.gold_ids, result.candidate_ids)
        return {
            "question_id": result.question_id,
            "question_type": type_map.get(result.question_id, ""),
            "question": q.question,
            "gold_ids": result.gold_ids,
            "candidate_ids": result.candidate_ids[:15],
            "accepted_ids": diag.get("accepted_ids", []),
            "rejected_ids": diag.get("rejected_ids", []),
            "narrative": result.narrative,
            "recall_at_10": r10,
            "gold_rank": rank,
            "precision_in_accepted": diag.get("precision_in_accepted", 0.0),
            "recall_in_accepted": diag.get("recall_in_accepted", 0.0),
            "n_accepted": diag.get("n_accepted", 0),
            "search_plan": diag.get("search_plan", {}),
            "sub_questions": diag.get("sub_questions", []),
            "removed_subq_picks": diag.get("removed_subq_picks", []),
            "steps": result.steps,
        }

    async def _save_checkpoint(record: dict) -> None:
        if not checkpoint_path:
            return
        async with _checkpoint_lock:
            key = (record["question_id"], record["question"])
            if key in completed_keys:
                return
            completed_keys.add(key)
            checkpoint_data.append(record)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps(checkpoint_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    async def _process_one(i: int, q: types_mod.QuestionEntry):
        async with sem:
            if not args.quiet:
                print(f"\n[Q {i}/{_total}] {q.question_type} -- {q.question_id}")
                print(f"  Question: {q.question[:120]}")
                print(f"  Gold IDs: {q.gold_ids}")

            try:
                result = await run_retrieval(
                    entry=q,
                    chunked_docs=chunked_docs,
                    full_docs=full_docs,
                    metadata_index=metadata_index,
                    date_index=date_index,
                    full_bm25=full_bm25,
                    full_emb=full_emb,
                    phase1_top_k=phase1_top_k,
                    max_steps=args.max_steps,
                    max_subquestions=args.max_subquestions,
                    tracker=tracker,
                    quiet=args.quiet,
                    skip_narrative=not getattr(args, "with_narrative", False),
                )
            except Exception as e:
                _n_failed[0] += 1
                print(f"  ERROR [{q.question_id}]: {type(e).__name__}: {e}")
                return i, None

            _n_done[0] += 1
            if not args.quiet:
                r3 = tools.recall_at_k(result.gold_ids, result.candidate_ids, 3)
                r5 = tools.recall_at_k(result.gold_ids, result.candidate_ids, 5)
                r10 = tools.recall_at_k(result.gold_ids, result.candidate_ids, 10)
                rank = tools.gold_rank(result.gold_ids, result.candidate_ids)
                rank_str = str(rank) if rank else "-"
                diag = getattr(result, "_diagnostics", {})
                p_acc = diag.get("precision_in_accepted", 0.0)
                r_acc = diag.get("recall_in_accepted", 0.0)
                n_acc = diag.get("n_accepted", 0)
                n_subq = len(diag.get("sub_questions", []))
                print(
                    f"  Recall: @3={r3:.2f} @5={r5:.2f} @10={r10:.2f} "
                    f"| Gold rank: {rank_str}"
                )
                print(
                    f"  P_acc={p_acc:.2f} R_acc={r_acc:.2f} "
                    f"#accepted={n_acc} #sub_questions={n_subq}"
                )
            elif _n_done[0] % 10 == 0:
                print(f"  Progress: {_n_done[0]}/{_total} done, {_n_failed[0]} failed")

            await _save_checkpoint(_make_record(result, q))
            return i, result

    tasks = [_process_one(i, q) for i, q in enumerate(answerable, 1)]
    completed = await asyncio.gather(*tasks)

    results: list[types_mod.RetrievalResult] = []
    for _, result in sorted(completed, key=lambda x: x[0]):
        if result is not None:
            results.append(result)

    metrics = tools.compute_metrics(results, type_map)
    tools.print_results_table(metrics)

    if checkpoint_path:
        print(f"\nJSON report saved to {checkpoint_path} ({len(checkpoint_data)} entries)")
    if _n_failed[0]:
        print(f"  {_n_failed[0]} questions failed (use --resume to retry them)")

    elapsed = time.monotonic() - t0
    usage = tracker.totals()
    print(
        f"\nTokens: prompt={usage['prompt_tokens']} "
        f"completion={usage['completion_tokens']} | "
        f"Time: {elapsed:.1f}s"
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval test framework"
    )
    parser.add_argument("--limit", type=int, default=None, help="Number of questions")
    parser.add_argument("--all", action="store_true", help="Evaluate all answerable questions")
    parser.add_argument("--type", type=str, default=None, help="Filter by question type")
    parser.add_argument(
        "--max-steps", type=int, default=types_mod.SUBQ_MAX_STEPS,
        help="Max Phase 2 steps per sub-question",
    )
    parser.add_argument(
        "--max-subquestions", type=int, default=types_mod.MAX_SUB_QUESTIONS,
        help="Max sub-questions per question",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    parser.add_argument("--quick", action="store_true", help="Run 1 question only")
    parser.add_argument(
        "--json-out", type=str, default=None,
        help="Save per-question JSON report to this file",
    )
    parser.add_argument(
        "--with-narrative", action="store_true",
        help="Enable narrative generation (disabled by default to save ~28%% API tokens)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Number of concurrent questions (default: 10)",
    )
    parser.add_argument(
        "--thinking", action="store_true",
        help="Enable DeepSeek v4 thinking mode (default: disabled)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint: skip questions already in --json-out file",
    )
    args = parser.parse_args()

    if args.quick:
        args.limit = 1
    if args.max_steps > tools.HARD_MAX_STEPS:
        args.max_steps = tools.HARD_MAX_STEPS

    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
