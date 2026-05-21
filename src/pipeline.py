# pipeline.py
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from ..util.llm_client import LLMClient, TokenTracker
from ..util.config import get_settings

from .types import SubQuestionResult, RetrievalState, RetrievalResult, AcceptedDoc
from . import types as types_mod
from . import prompts
from . import tools
from . import agent
from typing import Callable, Awaitable

ReviewCallback = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


async def _no_review(_review_type: str, data: dict[str, Any]) -> dict[str, Any]:
    return data

async def run_planner(
    entry: types_mod.QuestionEntry,
    tracker: TokenTracker,
    max_subq: int = types_mod.MAX_SUB_QUESTIONS,
    quiet: bool = False,
) -> dict[str, Any]:
    client = LLMClient(get_settings().llm)
    system = (prompts.PLANNER_SYSTEM
              .replace("{max_subq}", str(max_subq))
              .replace("{corpus_lang}", prompts.CORPUS_LANG)
              .replace("{corpora}", prompts.CORPORA_STR)
              .replace("{corpus_description}", prompts.CORPUS_DESCRIPTION))
    prompt = f"Question: {entry.question}"
    response = await client.complete(
        prompt, system=system, json_mode=True,
        max_tokens=types_mod.PLANNER_MAX_TOKENS,
    )
    tracker.record(response.usage)
    try:
        parsed = tools.parse_json_object(response.content)
    except ValueError:
        parsed = {}
    parsed.setdefault("query_analysis", "")
    parsed.setdefault("sub_questions", [])
    parsed.setdefault("entities_to_find", [])
    parsed.setdefault("dates_mentioned", [])
    if len(parsed["sub_questions"]) > max_subq:
        parsed["sub_questions"] = parsed["sub_questions"][:max_subq]
    if not quiet:
        n_sq = len(parsed["sub_questions"])
        print(f"  [Planner] sub_questions={n_sq}")
        print(f"  [Planner] analysis: {parsed.get('query_analysis', '')[:120]}")
    return parsed

async def rerank_accepted(
    question: str,
    accepted_ids: list[str],
    metadata_index: dict[str, types_mod.MetadataEntry],
    doc_counts: dict[str, int],
    doc_reasons: dict[str, list[str]],
    tracker: TokenTracker,
    quiet: bool = False,
    limits: types_mod.AgentLimits | None = None,
) -> list[str]:
    """Rerank by sub-question confirmation count, tie-breaking within
    same-count groups using the agent's own acceptance reasons."""
    if len(accepted_ids) <= 2:
        return list(accepted_ids)

    groups: dict[int, list[str]] = defaultdict(list)
    for doc_id in accepted_ids:
        groups[doc_counts.get(doc_id, 1)].append(doc_id)

    _limits = limits or types_mod.get_agent_limits(get_settings().llm.provider)
    client = LLMClient(get_settings().llm)

    reordered: list[str] = []
    n_tiebreaks = 0
    for count in sorted(groups.keys(), reverse=True):
        group = groups[count]
        if len(group) <= 1:
            reordered.extend(group)
            continue

        doc_lines: list[str] = []
        for i, doc_id in enumerate(group, 1):
            meta = metadata_index.get(doc_id)
            date = meta.date if meta else "?"
            corpus = meta.corpus if meta else "?"
            reasons = [r for r in doc_reasons.get(doc_id, []) if r]
            reasons_str = "; ".join(f'"{r}"' for r in reasons) or "(no reason recorded)"
            doc_lines.append(
                f"{i}. {doc_id} | {date} | {corpus}\n   Reasons: {reasons_str}"
            )

        prompt = (
            f"Question: {question}\n\n"
            f"These {len(group)} documents were each accepted by {count} "
            f"sub-question(s). Rank them by relevance to the question, "
            f"using the investigator's reasons as evidence.\n\n"
            + "\n".join(doc_lines)
            + "\n\nReturn a JSON array of doc_ids, most relevant first."
        )

        try:
            response = await client.complete(
                prompt, system=prompts.RERANK_SYSTEM, json_mode=True,
                max_tokens=_limits.rerank_max_tokens,
            )
            tracker.record(response.usage)
            parsed = json.loads(response.content)
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
            if not isinstance(parsed, list):
                reordered.extend(group)
                continue
            valid = set(group)
            group_order = [d for d in parsed if d in valid]
            seen = set(group_order)
            for d in group:
                if d not in seen:
                    group_order.append(d)
            reordered.extend(group_order)
            n_tiebreaks += 1
        except Exception:
            reordered.extend(group)

    if not quiet:
        print(
            f"    [Rerank] {len(accepted_ids)} docs in {len(groups)} "
            f"count-groups, {n_tiebreaks} tiebreak calls"
        )
    return reordered

async def run_subquestion_pipeline(
    sub_question: str,
    sub_index: int,
    target_corpus: str,
    key_entities: list[str],
    *,
    replan_context: dict[str, Any] | None = None,
    original_question: str | None = None,
    chunked_docs: dict[str, types_mod.CorpusDoc],
    metadata_index: dict[str, types_mod.MetadataEntry],
    date_index: dict[str, list[str]],
    full_bm25: types_mod.BM25Index,
    full_emb: types_mod.EmbeddingIndex | None,
    phase1_top_k: int,
    max_steps: int,
    tracker: TokenTracker,
    quiet: bool,
) -> SubQuestionResult:
    if not quiet:
        print(f"\n  --- Sub-question {sub_index + 1}: {sub_question[:100]}")
        print(f"      target_corpus={target_corpus}, entities={key_entities}")
        if replan_context:
            print(f"      replan_context: {list(replan_context.keys())}")

    fused, contributions = tools.run_score_fusion(
        sub_question,
        doc_index=chunked_docs,
        full_bm25=full_bm25,
        full_docs=chunked_docs,
        full_emb=full_emb,
        metadata_index=metadata_index,
        date_index=date_index,
        top_k=phase1_top_k,
    )
    corpus_constraint = target_corpus if target_corpus != "any" else "any"
    fused = tools.allocate_by_corpus(
        fused, metadata_index, corpus_constraint, max_total=20,
    )

    if not quiet:
        corpus_label = target_corpus if target_corpus != "any" else "any"
        print(f"      [Phase1] {len(fused)} candidates (constraint={corpus_label})")

    limits = types_mod.get_agent_limits(get_settings().llm.provider)
    # Use chunk_keys for pending (for text lookup in chunked_docs)
    all_chunk_keys = [h.chunk_key or h.doc_id for h in fused]
    first_batch = all_chunk_keys[:limits.review_batch_size]
    remaining_queue = all_chunk_keys[limits.review_batch_size:]

    state = RetrievalState(
        query=sub_question,
        sub_question_index=sub_index,
        pending_review=first_batch,
        pending_queue=remaining_queue,
        phase="PENDING_REVIEW" if first_batch else "READY",
        max_steps=min(max_steps, tools.HARD_MAX_STEPS),
    )

    initial_stage_memory = agent.init_stage_memory(
        sub_question, key_entities, replan_context=replan_context,
    )

    accepted_ids, held_ids, briefing, phase2_steps, accepted_docs = await agent.run_phase2(
        sub_question, state,
        initial_stage_memory=initial_stage_memory,
        chunked_docs=chunked_docs,
        full_bm25=full_bm25,
        full_emb=full_emb,
        metadata_index=metadata_index,
        date_index=date_index,
        tracker=tracker,
        quiet=quiet,
        key_entities=key_entities,
        original_question=original_question,
    )

    confirmed = True
    confirmed_ids = accepted_ids
    verification: dict[str, Any] = {}

    return SubQuestionResult(
        sub_question=sub_question,
        target_corpus=target_corpus,
        accepted_ids=accepted_ids,
        held_ids=held_ids,
        briefing=briefing,
        steps=phase2_steps,
        verification=verification,
        confirmed=confirmed,
        confirmed_ids=confirmed_ids,
        accepted_docs=accepted_docs,
    )

async def run_retrieval(
    entry: types_mod.QuestionEntry,
    *,
    chunked_docs: dict[str, types_mod.CorpusDoc],
    full_docs: dict[str, types_mod.CorpusDoc],
    metadata_index: dict[str, types_mod.MetadataEntry],
    date_index: dict[str, list[str]],
    full_bm25: types_mod.BM25Index,
    full_emb: types_mod.EmbeddingIndex | None,
    phase1_top_k: int,
    max_steps: int,
    max_subquestions: int,
    tracker: TokenTracker,
    quiet: bool,
    review_callback: ReviewCallback | None = None,
    progress_callback: Any = None,
    skip_narrative: bool = True,
) -> RetrievalResult:
    query = entry.question
    if review_callback is None:
        review_callback = _no_review
    limits = types_mod.get_agent_limits(get_settings().llm.provider)

    def _log(step: str, detail: str = "") -> None:
        if progress_callback:
            progress_callback(step, detail)

    _log("Planner", "Decomposing question...")
    plan = await run_planner(entry, tracker, max_subq=max_subquestions, quiet=quiet)
    n_subq = len(plan.get("sub_questions") or [])
    _log("Planner", f"Generated {n_subq} sub-questions")

    # PAUSE 1: human reviews planner output
    plan = await review_callback("planner", plan)
    sub_questions_raw = plan.get("sub_questions") or []
    common_kwargs = dict(
        original_question=query,
        chunked_docs=chunked_docs,
        metadata_index=metadata_index,
        date_index=date_index,
        full_bm25=full_bm25,
        full_emb=full_emb,
        phase1_top_k=phase1_top_k,
        max_steps=max_steps,
        tracker=tracker,
        quiet=quiet,
    )

    if len(sub_questions_raw) <= 1:
        sub_q_text = query
        target_corpus = "any"
        key_entities: list[str] = []
        if sub_questions_raw and isinstance(sub_questions_raw[0], dict):
            sq = sub_questions_raw[0]
            sub_q_text = str(sq.get("question", query))
            target_corpus = str(sq.get("target_corpus", "any"))
            key_entities = tools.as_list(sq.get("key_entities"))

        _log("Sub-Q 1/1", f"{sub_q_text[:80]}...")
        sub_result = await run_subquestion_pipeline(
            sub_q_text, 0, target_corpus, key_entities,
            **common_kwargs,
        )

        final_ids = sub_result.accepted_ids
        all_steps = sub_result.steps
        sub_results = [sub_result]
    else:
        sub_results: list[SubQuestionResult] = []
        all_steps: list[dict[str, Any]] = []
        replan_client = LLMClient(get_settings().llm)

        remaining_sqs: list[dict[str, Any]] = [
            sq for sq in sub_questions_raw
            if isinstance(sq, dict) and str(sq.get("question", "")).strip()
        ]

        completed_count = 0
        completed_summaries: list[str] = []
        prior_briefing: dict[str, Any] | None = None
        i = 0
        while i < len(remaining_sqs):
            replan_context: dict[str, Any] | None = None
            if i > 0 and prior_briefing:
                completed_block = "\n".join(
                    f"  [{j+1}] {s}" for j, s in enumerate(completed_summaries)
                )
                remaining_block = "\n".join(
                    f"  [{completed_count + 1 + k}] {json.dumps(sq, ensure_ascii=False)}"
                    for k, sq in enumerate(remaining_sqs[i:])
                )
                replan_prompt = (
                    f"Original question: {query}\n\n"
                    f"=== PLAN OVERVIEW ===\n"
                    f"Total sub-questions allowed: {max_subquestions}\n"
                    f"Completed ({completed_count}):\n{completed_block}\n\n"
                    f"Just completed (sub-Q {completed_count}):\n"
                    f"  Briefing:\n{json.dumps(prior_briefing, ensure_ascii=False)}\n\n"
                    f"Remaining ({len(remaining_sqs) - i}):\n{remaining_block}\n\n"
                    "Revise the remaining plan. You may add new sub-questions if gaps exist. "
                    "Return JSON with revised_sub_questions and reasoning."
                )
                replan_system = (
                    prompts.REPLAN_SYSTEM
                    .replace("{corpus_lang}", prompts.CORPUS_LANG)
                    .replace("{corpora}", prompts.CORPORA_STR)
                    .replace("{max_subquestions}", str(max_subquestions))
                )
                try:
                    replan_resp = await replan_client.complete(
                        replan_prompt,
                        system=replan_system,
                        json_mode=True,
                        max_tokens=limits.replan_max_tokens,
                    )
                    tracker.record(replan_resp.usage)
                    parsed_replan = json.loads(replan_resp.content)
                    revised = parsed_replan.get("revised_sub_questions", [])
                    if isinstance(revised, list) and revised:
                        revised = [sq for sq in revised if isinstance(sq, dict)]
                        expected_count = len(remaining_sqs) - i
                        if revised and len(revised) >= expected_count and (completed_count + len(revised)) <= max_subquestions:
                            n_spawned = len(revised) - expected_count
                            for sq in revised[expected_count:]:
                                sq["spawned"] = True
                            remaining_sqs = remaining_sqs[:i] + revised
                            replan_context = parsed_replan
                            if not quiet:
                                reasoning = parsed_replan.get("reasoning", "")[:100]
                                spawn_note = f" (+{n_spawned} spawned)" if n_spawned else ""
                                print(f"  [Replan] Updated {len(revised)} sub-Qs{spawn_note}: {reasoning}")
                        elif not quiet:
                            print(f"  [Replan] Rejected: got {len(revised)} sub-Qs, expected >={expected_count}, cap={max_subquestions - completed_count}")
                except Exception:
                    if not quiet:
                        print(f"  [Replan] parse error, keeping original plan")

                # PAUSE 2: human reviews briefing + re-planner's revision
                review_data = await review_callback("replan", {
                    "briefing": prior_briefing,
                    "revised_sub_questions": remaining_sqs[i:],
                    "sub_q_index": i,
                    "total_sub_qs": len(remaining_sqs),
                })
                # Apply user's edits (may have modified sub-Q text, entities, etc.)
                edited_sqs = review_data.get("revised_sub_questions")
                if edited_sqs and isinstance(edited_sqs, list):
                    remaining_sqs = remaining_sqs[:i] + edited_sqs

            sq_raw = remaining_sqs[i]
            sq_text = str(sq_raw.get("question", "")).strip()
            if not sq_text:
                i += 1
                continue
            target_corpus = str(sq_raw.get("target_corpus", "any"))
            key_entities = tools.as_list(sq_raw.get("key_entities"))
            is_spawned = bool(sq_raw.get("spawned", False))

            _log(f"Sub-Q {i+1}/{len(remaining_sqs)}", f"[{target_corpus}] {sq_text[:80]}...")
            r = await run_subquestion_pipeline(
                sq_text, i, target_corpus, key_entities,
                replan_context=replan_context,
                **common_kwargs,
            )
            r.spawned = is_spawned
            sub_results.append(r)
            all_steps.extend(r.steps)
            _log(f"Sub-Q {i+1} done", f"{len(r.accepted_ids)} accepted, {len(r.held_ids)} held")

            prior_briefing = r.briefing
            completed_count += 1
            completed_summaries.append(
                f"[{target_corpus}] {sq_text[:60]} → {len(r.accepted_ids)} accepted"
            )
            i += 1

        _log("Merge", f"Merging {len(sub_results)} sub-question results")
        final_ids, removed_ids = tools.merge_subquestion_results(sub_results)

    agent_accepted_ids = list(final_ids)

    if len(final_ids) > 2:
        _log("Rerank", f"Reranking {len(final_ids)} candidates")
        doc_counts: dict[str, int] = defaultdict(int)
        doc_reasons: dict[str, list[str]] = defaultdict(list)
        for sr in sub_results:
            for doc_id in sr.confirmed_ids:
                doc_counts[doc_id] += 1
            for a in sr.accepted_docs:
                if a.reason:
                    doc_reasons[a.doc_id].append(a.reason)

        final_ids = await rerank_accepted(
            entry.question, final_ids, metadata_index,
            doc_counts, doc_reasons, tracker, quiet, limits=limits,
        )

    if skip_narrative:
        narrative = ""
    else:
        _log("Narrative", "Generating answer...")
        accepted_docs = [AcceptedDoc(d, "merged", 0, chunk_key=d) for d in final_ids]
        narrative = await tools.generate_narrative(
            entry, accepted_docs, full_docs,
            metadata_index, tracker, quiet, limits=limits,
        )

    result = RetrievalResult(
        question_id=entry.question_id,
        gold_ids=entry.gold_ids,
        candidate_ids=final_ids,
        steps=all_steps,
        narrative=narrative,
        accepted_ids=agent_accepted_ids,
    )

    sub_q_diagnostics = []
    for sr in sub_results:
        diag: dict[str, Any] = {
            "question": sr.sub_question,
            "target_corpus": sr.target_corpus,
            "accepted_ids": sr.accepted_ids,
            "held_ids": sr.held_ids,
            "briefing": sr.briefing,
            "verification": sr.verification,
            "confirmed": sr.confirmed,
            "confirmed_ids": sr.confirmed_ids,
            "steps": sr.steps,
        }
        if sr.spawned:
            diag["spawned"] = True
        sub_q_diagnostics.append(diag)

    removed_subq_picks = []
    for sr in sub_results:
        if sr.confirmed:
            confirmed_set = set(sr.confirmed_ids)
            removed_subq_picks.extend(
                [d for d in sr.accepted_ids if d not in confirmed_set]
            )
        else:
            removed_subq_picks.extend(sr.accepted_ids)

    result._diagnostics = {  # type: ignore[attr-defined]
        "search_plan": plan,
        "sub_questions": sub_q_diagnostics,
        "removed_subq_picks": removed_subq_picks,
        "accepted_ids": agent_accepted_ids,
        "rejected_ids": [],
        "n_accepted": len(agent_accepted_ids),
        "precision_in_accepted": tools.precision_in_accepted(entry.gold_ids, agent_accepted_ids),
        "recall_in_accepted": tools.recall_in_accepted(entry.gold_ids, agent_accepted_ids),
    }

    return result
