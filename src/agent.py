# agent.py
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

from ..util.llm_client import LLMClient, TokenTracker
from ..util.config import get_settings

from .types import RetrievalState, AcceptedDoc
from . import types as types_mod
from . import prompts
from . import tools



def init_stage_memory(
    sub_question: str,
    key_entities: list[str],
    replan_context: dict[str, Any] | None = None,
) -> dict[str, str]:
    mem: dict[str, str] = {}
    if replan_context:
        parts: list[str] = []
        strategy = replan_context.get("search_strategy")
        if strategy:
            parts.append(f"Search strategy:\n{strategy}")
        key_facts = replan_context.get("key_facts")
        if key_facts:
            if isinstance(key_facts, list):
                facts = "\n".join(f"- {f}" for f in key_facts)
            else:
                facts = str(key_facts)
            parts.append(f"Key facts:\n{facts}")
        key_dates = replan_context.get("key_dates")
        if key_dates:
            if isinstance(key_dates, list):
                dates = ", ".join(str(d) for d in key_dates)
            else:
                dates = str(key_dates)
            parts.append(f"Dates:\n{dates}")
        if parts:
            mem["prior_findings"] = "\n\n".join(parts)
    mem["question_analysis"] = sub_question
    mem["evidence_gathered"] = "(none yet)"
    mem["gaps_remaining"] = "(to be determined after first review)"
    if key_entities:
        ent_str = ", ".join(key_entities)
        mem["next_steps"] = f"1. Search for entities: {ent_str}\n2. Review Phase 1 candidates"
    else:
        mem["next_steps"] = "1. Review Phase 1 candidates"
    return mem

def apply_memory_delta(current: dict[str, str] | str, delta: Any) -> dict[str, str] | str:
    if isinstance(delta, dict) and delta:
        base = current if isinstance(current, dict) else {}
        updated = dict(base)
        for k, v in delta.items():
            if v is None:
                updated.pop(k, None)
            else:
                updated[k] = str(v)
        return updated
    if isinstance(delta, str) and delta.strip():
        return delta.strip()
    return current

_MEMORY_SECTION_ORDER = [
    ("prior_findings", "Prior Findings"),
    ("question_analysis", "Question Analysis"),
    ("evidence_gathered", "Evidence Gathered"),
    ("gaps_remaining", "Gaps Remaining"),
    ("next_steps", "Next Steps"),
]

def format_structured_memory(mem: dict[str, str] | str) -> str:
    if isinstance(mem, str):
        return tools.truncate_stage_memory(mem.strip()) if mem.strip() else "(empty)"
    if not mem:
        return "(empty)"
    parts: list[str] = []
    seen: set[str] = set()
    for key, label in _MEMORY_SECTION_ORDER:
        if key in mem:
            parts.append(f"## {label}\n{mem[key]}")
            seen.add(key)
    for key in sorted(mem.keys()):
        if key not in seen:
            parts.append(f"## {key.replace('_', ' ').title()}\n{mem[key]}")
    return tools.truncate_stage_memory("\n\n".join(parts))

def format_state_for_llm(
    query: str,
    state: RetrievalState,
    step_n: int,
    stage_memory: dict[str, str] | str,
    action_log: list[str],
    metadata_index: dict[str, types_mod.MetadataEntry] | None = None,
    tool_output: str | None = None,
    original_question: str | None = None,
    show_memory: bool = True,
) -> str:
    parts = []
    if original_question and original_question != query:
        parts.append(f"Original research question: {original_question}")
        parts.append(f"Current sub-question: {query}")
    else:
        parts.append(f"Question: {query}")
    parts.append(f"Step {step_n}/{state.max_steps}")

    if state.accepted:
        acc_lines = []
        for i, a in enumerate(state.accepted):
            meta_str = ""
            if metadata_index:
                meta = metadata_index.get(a.chunk_key) or metadata_index.get(a.doc_id)
                if meta:
                    meta_str = f"{meta.date or '?'} {meta.corpus or '?'}"
            acc_lines.append(f"  {i+1}. {a.doc_id} | {meta_str} | {a.reason}")
        parts.append(f"Accepted candidates ({len(state.accepted)} docs):")
        parts.append("\n".join(acc_lines))
    else:
        parts.append("Accepted candidates: (none)")

    if state.phase == "PENDING_REVIEW" and state.pending_review:
        # Show parent doc_ids, not chunk_keys (agent should never see chunk_keys)
        if metadata_index:
            pending_parents = []
            for ck in state.pending_review:
                m = metadata_index.get(ck)
                pending_parents.append(m.doc_id if m else ck)
            parts.append(
                f"Pending review ({len(state.pending_review)} docs): "
                + ", ".join(pending_parents)
            )
        else:
            parts.append(
                f"Pending review ({len(state.pending_review)} docs)"
            )

    parts.append(f"Rejected: {len(state.rejected)} | Held: {len(state.held)}")

    if action_log:
        parts.append("Action log:")
        parts.append("\n".join(action_log[-8:]))

    if show_memory:
        parts.append("Stage memory:")
        parts.append(format_structured_memory(stage_memory))

    if tool_output:
        parts.append("Last tool output:")
        parts.append(tool_output)

    if show_memory:
        parts.append('Include "memory_update" as a JSON object with only changed sections.')

    if step_n >= state.max_steps:
        parts.append(
            ">>> LAST STEP: You MUST call finish now to return your accepted candidates."
        )

    return "\n\n".join(parts)

def process_review_decisions(
    state: RetrievalState,
    decisions: list[dict[str, Any]],
    pending_chunk_keys: list[str],
    step_n: int,
    chunked_docs: dict[str, types_mod.CorpusDoc] | None = None,
    allow_hold: bool = True,
) -> tuple[int, int, int]:
    """Process review decisions. pending_chunk_keys are chunk keys.

    The LLM's decisions reference parent doc_ids (shown in review header).
    We map them back to chunk_keys via the pending list.
    Accepted set tracks parent doc_ids (accept chunk = accept parent).
    Rejected set tracks chunk_keys (reject chunk ≠ reject parent).
    Held list stores chunk_keys for text lookup.
    """
    # Build bidirectional mappings from pending list
    parent_to_chunk: dict[str, str] = {}
    chunk_to_parent: dict[str, str] = {}
    for ck in pending_chunk_keys:
        if chunked_docs and ck in chunked_docs:
            pid = chunked_docs[ck].doc_id
            parent_to_chunk[pid] = ck
            chunk_to_parent[ck] = pid
        else:
            parent_to_chunk[ck] = ck
            chunk_to_parent[ck] = ck

    decided_parents: set[str] = set()
    accepted_set = {a.doc_id for a in state.accepted}
    held_parents = set()
    if chunked_docs:
        held_parents = {chunked_docs[h].doc_id if h in chunked_docs else h
                        for h in state.held}
    else:
        held_parents = set(state.held)
    n_accepted = 0
    n_rejected = 0
    n_held = 0

    for d in decisions:
        if not isinstance(d, dict):
            continue
        raw_id = d.get("doc_id", "")
        if not raw_id:
            continue
        # Resolve: LLM may return parent doc_id OR chunk_key
        # Normalize to (parent doc_id, chunk_key) pair
        if raw_id in chunk_to_parent:
            doc_id = chunk_to_parent[raw_id]
            chunk_key = raw_id
        elif raw_id in parent_to_chunk:
            doc_id = raw_id
            chunk_key = parent_to_chunk[raw_id]
        else:
            # Unknown ID — resolve via chunked_docs if possible
            if chunked_docs and raw_id in chunked_docs:
                doc_id = chunked_docs[raw_id].doc_id
                chunk_key = raw_id
            else:
                doc_id = raw_id
                chunk_key = raw_id
        decided_parents.add(doc_id)

        if d.get("accept", False):
            if doc_id not in accepted_set:
                state.accepted.append(
                    AcceptedDoc(doc_id, d.get("reason", ""), step_n,
                                chunk_key=chunk_key)
                )
                accepted_set.add(doc_id)
                n_accepted += 1
            # Accept parent → remove all rejected chunks of this parent
            state.rejected = {ck for ck in state.rejected
                              if (chunked_docs[ck].doc_id if chunked_docs and ck in chunked_docs else ck) != doc_id}
            # Remove from held if present — resolve each held chunk_key to parent
            state.held = [h for h in state.held
                          if (chunked_docs[h].doc_id if chunked_docs and h in chunked_docs else h) != doc_id]
            held_parents.discard(doc_id)
        elif d.get("hold", False) and allow_hold:
            if doc_id not in accepted_set and doc_id not in held_parents:
                state.held.append(chunk_key)
                held_parents.add(doc_id)
                n_held += 1
        else:
            # Reject this chunk only (not the whole parent)
            state.held = [h for h in state.held if h != chunk_key]
            if chunk_key not in state.rejected:
                state.rejected.add(chunk_key)
                n_rejected += 1

    # Undecided pending → auto-hold or auto-reject
    for ck in pending_chunk_keys:
        parent = chunked_docs[ck].doc_id if chunked_docs and ck in chunked_docs else ck
        if parent in decided_parents or parent in accepted_set or ck in state.rejected or parent in held_parents:
            continue
        if allow_hold:
            state.held.append(ck)
            held_parents.add(parent)
            n_held += 1
        else:
            state.rejected.add(ck)
            n_rejected += 1

    return n_accepted, n_rejected, n_held

async def run_held_reevaluation(
    query: str,
    state: RetrievalState,
    chunked_docs: dict[str, types_mod.CorpusDoc],
    metadata_index: dict[str, types_mod.MetadataEntry],
    tracker: TokenTracker,
    quiet: bool,
    limits: types_mod.AgentLimits | None = None,
) -> list[str]:
    """Re-evaluate held chunk_keys. Returns list of promoted chunk_keys."""
    if not state.held:
        return []

    def _summary(chunk_key: str) -> str:
        doc = chunked_docs.get(chunk_key)
        if not doc:
            return ""
        if doc.summary_brief and len(doc.summary_brief) > 10:
            return doc.summary_brief[:200]
        if doc.title:
            return doc.title[:200]
        if doc.contents_text:
            return doc.contents_text[:150]
        return ""

    acc_lines: list[str] = []
    for i, a in enumerate(state.accepted, 1):
        meta = metadata_index.get(a.chunk_key) or metadata_index.get(a.doc_id)
        meta_str = f"{meta.date or '?'} {meta.corpus or '?'}" if meta else "?"
        acc_lines.append(f"{i}. {a.doc_id} | {meta_str} | {a.reason}\n   {_summary(a.chunk_key)}")

    # state.held stores chunk_keys; show parent doc_id to LLM
    held_lines: list[str] = []
    for i, chunk_key in enumerate(state.held, 1):
        doc = chunked_docs.get(chunk_key)
        parent = doc.doc_id if doc else chunk_key
        meta = metadata_index.get(chunk_key)
        meta_str = f"{meta.date or '?'} {meta.corpus or '?'}" if meta else "?"
        held_lines.append(f"{i}. {parent} | {meta_str}\n   {_summary(chunk_key)}")

    prompt = (
        f"Question: {query}\n\n"
        f"Accepted evidence ({len(state.accepted)} docs):\n"
        + "\n".join(acc_lines or ["(none)"])
        + f"\n\nHeld documents to re-evaluate ({len(state.held)}):\n"
        + "\n".join(held_lines)
        + "\n\nReturn a JSON array of held doc_ids to promote."
    )

    _limits = limits or types_mod.get_agent_limits(get_settings().llm.provider)
    client = LLMClient(get_settings().llm)
    try:
        response = await client.complete(
            prompt, system=prompts.HELD_REEVAL_SYSTEM, json_mode=True,
            max_tokens=_limits.held_reeval_max_tokens,
        )
        tracker.record(response.usage)
        parsed = json.loads(response.content)
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
        if not isinstance(parsed, list):
            return []
        # Map parent doc_ids → chunk_keys in state.held
        # LLM returns parent doc_ids; state.held stores chunk_keys
        parent_to_held_ck: dict[str, str] = {}
        for ck in state.held:
            doc = chunked_docs.get(ck)
            pid = doc.doc_id if doc else ck
            parent_to_held_ck[pid] = ck
            parent_to_held_ck[ck] = ck  # also match if LLM returns chunk_key
        promoted = [parent_to_held_ck[d] for d in parsed if d in parent_to_held_ck]
        if not quiet and promoted:
            print(f"    [Held Re-eval] Promoted {len(promoted)}: {promoted}")
        return promoted
    except Exception:
        if not quiet:
            print("    [Held Re-eval] parse error; no promotions")
        return []

async def run_phase2(
    query: str,
    state: RetrievalState,
    *,
    initial_stage_memory: dict[str, str],
    chunked_docs: dict[str, types_mod.CorpusDoc],
    full_bm25: types_mod.BM25Index,
    full_emb: types_mod.EmbeddingIndex | None,
    metadata_index: dict[str, types_mod.MetadataEntry],
    date_index: dict[str, list[str]],
    tracker: TokenTracker,
    quiet: bool,
    key_entities: list[str] | None = None,
    original_question: str | None = None,
) -> tuple[list[str], list[str], dict[str, Any], list[dict[str, Any]], list[AcceptedDoc]]:
    client = LLMClient(get_settings().llm)
    limits = types_mod.get_agent_limits(get_settings().llm.provider)
    steps: list[dict[str, Any]] = []
    consecutive_repeats = 0
    consecutive_empty_searches = 0
    parse_error_count = 0

    stage_memory: dict[str, str] | str = initial_stage_memory
    action_log: list[str] = []
    pending_tool_output: str | None = None

    # Single-shot calls for all providers (system + user per call).
    system_base = prompts.AGENT_SYSTEM

    async def _do_finish(
        step_label: int | None = None,
    ) -> tuple[list[str], list[str], dict[str, Any], list[dict[str, Any]], list[AcceptedDoc]]:
        promoted = await run_held_reevaluation(
                query, state, chunked_docs, metadata_index, tracker, quiet,
                limits=limits,
            )
        if promoted:
            for chunk_key in promoted:
                doc = chunked_docs.get(chunk_key)
                parent = doc.doc_id if doc else chunk_key
                if not any(a.doc_id == parent for a in state.accepted):
                    state.accepted.append(
                        AcceptedDoc(parent, "promoted_from_hold",
                                    step_label or state.step_count,
                                    chunk_key=chunk_key)
                    )
            state.held = [h for h in state.held if h not in set(promoted)]
            obs = f"held_reeval -> {len(promoted)} promoted (total accepted: {len(state.accepted)})"
            steps.append({
                "phase": 2, "step": step_label or state.step_count,
                "thought": "held reevaluation", "action": "held_reeval",
                "args": {"promoted": promoted},
                "observation": obs,
            })
            action_log.append(f"[Re-eval] {obs}")

        accepted_ids = [a.doc_id for a in state.accepted]
        held_ids = list(state.held)
        accepted_docs = list(state.accepted)

        def _doc_summary(chunk_key: str) -> str:
            doc = chunked_docs.get(chunk_key)
            if not doc:
                return ""
            if doc.summary_brief and len(doc.summary_brief) > 10:
                return doc.summary_brief[:200]
            if doc.title:
                return doc.title[:200]
            return ""

        acc_lines: list[str] = []
        for i, a in enumerate(state.accepted, 1):
            meta = metadata_index.get(a.chunk_key) or metadata_index.get(a.doc_id)
            meta_str = f"{meta.date or '?'} {meta.corpus or '?'}" if meta else "?"
            acc_lines.append(f"{i}. {a.doc_id} | {meta_str} | {a.reason}\n   {_doc_summary(a.chunk_key)}")

        stage_text = format_structured_memory(stage_memory)
        briefing: dict[str, Any] = {}
        briefing_prompt = (
            f"Question investigated: {query}\n\n"
            f"Stage memory:\n{stage_text}\n\n"
            f"Accepted evidence ({len(accepted_ids)} docs):\n"
            + "\n".join(acc_lines or ["(none)"])
            + "\n\nProduce a JSON briefing for the next investigator."
        )
        try:
            resp = await client.complete(
                briefing_prompt,
                system=prompts.BRIEFING_SYSTEM.replace("{corpus_lang}", prompts.CORPUS_LANG),
                json_mode=True,
                max_tokens=limits.briefing_max_tokens,
            )
            tracker.record(resp.usage)
            parsed_b = json.loads(resp.content)
            if isinstance(parsed_b, dict):
                briefing = parsed_b
        except Exception:
            if not quiet:
                print("    [Briefing] parse error; empty briefing")

        if not quiet and briefing:
            summary = briefing.get("summary", "")[:100]
            print(f"    [Briefing] {summary}...")

        return accepted_ids, held_ids, briefing, steps, accepted_docs

    # Phase 1 review pre-loop
    if state.pending_review:
        batch_text = tools.load_review_batch(
            state.pending_review, chunked_docs, metadata_index,
        )
        total_pending = len(state.pending_review) + len(state.pending_queue)
        pending_tool_output = (
            f"Phase 1 produced {total_pending} candidates for your review.\n"
            f"Showing batch 1 ({len(state.pending_review)} docs).\n"
            f"Full text of each document:\n\n{batch_text}"
        )

    phase1_n = 0
    while state.phase == "PENDING_REVIEW" and state.pending_review:
        phase1_n += 1

        prompt = format_state_for_llm(
            query, state, phase1_n, stage_memory, action_log,
            metadata_index, tool_output=pending_tool_output,
            original_question=original_question,
            show_memory=True,
        )
        pending_tool_output = None

        system_prompt = system_base + prompts.TOOLS_REVIEW
        response = await client.complete(
            prompt, system=system_prompt, json_mode=True,
            max_tokens=limits.agent_review_max_tokens,
        )
        tracker.record(response.usage)

        try:
            parsed = tools.parse_json_object(response.content)
        except ValueError:
            parse_error_count += 1
            if not quiet:
                print(f"    [Phase1 R{phase1_n}] parse_error ({parse_error_count})")
            process_review_decisions(state, [], state.pending_review, phase1_n,
                                    chunked_docs=chunked_docs,
                                    allow_hold=True)
            state.pending_review = []
            if parse_error_count > limits.max_parse_errors:
                state.pending_queue = []
                state.phase = "READY"
                break
            # Try next batch from queue
            if state.pending_queue:
                next_batch = state.pending_queue[:limits.review_batch_size]
                state.pending_queue = state.pending_queue[limits.review_batch_size:]
                state.pending_review = next_batch
                batch_text = tools.load_review_batch(
                    next_batch, chunked_docs, metadata_index,
                )
                pending_tool_output = (
                    f"Previous batch parse error (docs auto-held). "
                    f"Next batch ({len(next_batch)} docs):\n\n{batch_text}"
                )
                continue
            else:
                state.phase = "READY"
                break

        thought = parsed.get("thought", "")
        args = parsed.get("args", {}) if isinstance(parsed.get("args"), dict) else {}

        memory_update = parsed.get("memory_update", {})
        stage_memory = apply_memory_delta(stage_memory, memory_update)

        decisions = args.get("decisions", [])
        if not isinstance(decisions, list):
            decisions = []

        n_accepted, n_rejected, n_held = process_review_decisions(
            state, decisions, state.pending_review, phase1_n,
            chunked_docs=chunked_docs,
            allow_hold=True,
        )

        state.pending_review = []

        obs = (
            f"review -> {n_accepted} accepted, {n_rejected} rejected, "
            f"{n_held} held (total accepted: {len(state.accepted)})"
        )

        if state.pending_queue:
            next_batch = state.pending_queue[:limits.review_batch_size]
            state.pending_queue = state.pending_queue[limits.review_batch_size:]
            state.pending_review = next_batch
            state.phase = "PENDING_REVIEW"

            batch_text = tools.load_review_batch(
                next_batch, chunked_docs, metadata_index,
            )
            remaining_count = len(state.pending_queue)
            pending_tool_output = (
                f"{obs}\n\n"
                f"Next batch ({len(next_batch)} docs, "
                f"{remaining_count} remaining after this):\n\n{batch_text}"
            )
            obs += f" | next batch: {len(next_batch)} docs"
        else:
            state.phase = "READY"

        step_rec: dict[str, Any] = {
            "phase": 1, "step": phase1_n,
            "thought": thought, "action": "review",
            "args": args, "stage_memory": stage_memory,
            "observation": obs,
        }
        steps.append(step_rec)
        action_log.append(f"[P1-R{phase1_n}] {obs}")
        if not quiet:
            print(f"    [Phase1 R{phase1_n}] {obs}")

    if phase1_n > 0:
        pending_tool_output = (
            f"Phase 1 judgment complete: {len(state.accepted)} accepted, "
            f"{len(state.rejected)} rejected, {len(state.held)} held."
        )

    def _corpus_candidate_ids(cf: str | None) -> list[str] | None:
        if not cf:
            return None
        ids = list({
            m.doc_id for m in metadata_index.values()
            if cf == m.corpus or cf == m.corpus_type
        })
        return ids or None

    effective_steps = 0
    total_iterations = 0
    max_iterations = state.max_steps * 2

    while effective_steps < state.max_steps and total_iterations < max_iterations:
        total_iterations += 1
        step_n = effective_steps + 1
        state.step_count = step_n

        if effective_steps == state.max_steps - 1:
            accepted_ids = [a.doc_id for a in state.accepted]
            step_rec: dict[str, Any] = {
                "phase": 2, "step": step_n,
                "thought": "max steps reached",
                "action": "finish", "args": {},
                "observation": f"forced finish (max steps) -> {len(accepted_ids)} accepted",
            }
            steps.append(step_rec)
            if not quiet:
                print(f"    [Step {step_n}] FORCED FINISH (max steps) -> {len(accepted_ids)} accepted")
            return await _do_finish(step_n)

        prompt = format_state_for_llm(
            query, state, step_n, stage_memory, action_log,
            metadata_index, tool_output=pending_tool_output,
            original_question=original_question,
            show_memory=True,
        )
        pending_tool_output = None

        max_tok = (
            limits.agent_review_max_tokens
            if state.phase == "PENDING_REVIEW"
            else limits.agent_search_max_tokens
        )

        tools_section = (
            prompts.TOOLS_REVIEW if state.phase == "PENDING_REVIEW"
            else prompts.TOOLS_READY
        )
        system_prompt = system_base + tools_section
        response = await client.complete(
            prompt, system=system_prompt, json_mode=True,
            max_tokens=max_tok,
        )
        tracker.record(response.usage)

        try:
            parsed = tools.parse_json_object(response.content)
        except ValueError:
            parse_error_count += 1
            if state.pending_review:
                process_review_decisions(state, [], state.pending_review, step_n,
                                        chunked_docs=chunked_docs,
                                        allow_hold=True)
                state.pending_review = []
                state.phase = "READY"
            steps.append({
                "phase": 2, "step": step_n,
                "thought": "", "action": "parse_error",
                "args": {"raw": response.content[:500]},
                "observation": f"JSON parse failed ({parse_error_count}/{limits.max_parse_errors})",
            })
            if not quiet:
                print(f"    [Step {step_n}] parse_error ({parse_error_count}/{limits.max_parse_errors})")
            if parse_error_count > limits.max_parse_errors:
                break
            # Feed error context for retry
            pending_tool_output = (
                "Your previous response was not valid JSON. "
                'Respond with a single JSON object: {"thought": ..., "action": ..., "args": ...}'
            )
            action_log.append(f"[S{step_n}] parse_error -> retry")
            continue

        parse_error_count = 0
        thought = parsed.get("thought", "")
        action = parsed.get("action", "")
        args = parsed.get("args", {}) if isinstance(parsed.get("args"), dict) else {}

        memory_update = parsed.get("memory_update", {})
        stage_memory = apply_memory_delta(stage_memory, memory_update)

        step_rec = {
            "phase": 2, "step": step_n,
            "thought": thought, "action": action,
            "args": args, "stage_memory": stage_memory,
        }

        if not quiet:
            print(f"    [Step {step_n}] Thought: {thought}")
            print(f"             Action: {action}({json.dumps(args, ensure_ascii=False)})")

        if not action:
            action = "finish"
            step_rec["action"] = "finish"
            step_rec["observation"] = "auto-corrected: no action returned"

        if state.phase == "PENDING_REVIEW" and action.startswith("search_"):
            original_action = action
            action = "review"
            step_rec["action"] = "review"
            step_rec["observation"] = f"auto-corrected: {original_action}->review"
            if not quiet:
                print(f"             Auto-corrected to review (was {original_action})")

        if action == "review" and state.phase != "PENDING_REVIEW":
            step_rec["observation"] = "skipped (nothing to review)"
            steps.append(step_rec)
            action_log.append(f"[S{step_n}] review -> skipped")
            continue

        if action == "finish":
            if state.phase == "PENDING_REVIEW":
                state.pending_review = []
                state.pending_queue = []
                state.phase = "READY"

            if not state.accepted:
                pending_tool_output = (
                    "ERROR: Cannot finish with 0 accepted documents. "
                    "Search for relevant documents first."
                )
                step_rec["observation"] = "blocked (finish with 0 accepted)"
                steps.append(step_rec)
                action_log.append(f"[S{step_n}] finish -> BLOCKED (0 accepted)")
                if not quiet:
                    print(f"             BLOCKED: finish with 0 accepted")
                continue

            accepted_ids = [a.doc_id for a in state.accepted]
            step_rec["observation"] = f"finish -> {len(accepted_ids)} accepted"
            steps.append(step_rec)
            action_log.append(f"[S{step_n}] finish -> {len(accepted_ids)} accepted")
            if not quiet:
                print(f"             Observation: finish -> {len(accepted_ids)} accepted")
            return await _do_finish(step_n)

        if action == "review":
            decisions = args.get("decisions", [])
            if not isinstance(decisions, list):
                decisions = []

            n_accepted, n_rejected, n_held = process_review_decisions(
                state, decisions, state.pending_review, step_n,
                chunked_docs=chunked_docs,
                allow_hold=True,
            )

            state.pending_review = []
            consecutive_empty_searches = 0

            obs = (
                f"review -> {n_accepted} accepted, {n_rejected} rejected, "
                f"{n_held} held (total accepted: {len(state.accepted)})"
            )

            if state.pending_queue:
                next_batch = state.pending_queue[:limits.review_batch_size]
                state.pending_queue = state.pending_queue[limits.review_batch_size:]
                state.pending_review = next_batch
                state.phase = "PENDING_REVIEW"

                batch_text = tools.load_review_batch(
                    next_batch, chunked_docs, metadata_index,
                )
                remaining_count = len(state.pending_queue)
                pending_tool_output = (
                    f"{obs}\n\n"
                    f"Next batch ({len(next_batch)} docs, "
                    f"{remaining_count} remaining after this):\n\n{batch_text}"
                )
                obs += f" | next batch: {len(next_batch)} docs"
            else:
                state.phase = "READY"

            step_rec["observation"] = obs
            steps.append(step_rec)
            action_log.append(f"[S{step_n}] {obs}")
            if not quiet:
                print(f"             Observation: {obs}")
            effective_steps += 1
            continue

        search_handlers = {
            "search_bm25", "search_semantic",
            "search_date", "search_grep",
        }

        if action in search_handlers:
            hits: list[types_mod.SearchHit] = []
            obs_prefix = action

            if action == "search_bm25":
                new_query = args.get("query", "")
                corpus_filter = args.get("corpus_filter")
                if not new_query:
                    step_rec["observation"] = "skipped (empty query)"
                    steps.append(step_rec)
                    action_log.append(f"[S{step_n}] search_bm25 -> skipped (empty)")
                    continue
                if tools.is_repeated_query(new_query, state.call_history, corpus_filter=corpus_filter, action_key="search_bm25"):
                    consecutive_repeats += 1
                    step_rec["observation"] = f"skipped (already tried search_bm25({new_query!r}, {corpus_filter}) — use different terms or tool)"
                    steps.append(step_rec)
                    action_log.append(f"[S{step_n}] search_bm25({new_query!r}) -> skipped (repetitive)")
                    if consecutive_repeats >= 2:
                        return await _do_finish(state.step_count or None)
                    continue
                consecutive_repeats = 0
                state.call_history.append(("search_bm25", new_query, corpus_filter))
                hits = tools.search_bm25(new_query, full_bm25, chunked_docs, max_results=tools.PER_CHANNEL_MAX,
                                         candidate_ids=_corpus_candidate_ids(corpus_filter))
                hits = tools.filter_hits_by_corpus(hits, metadata_index, corpus_filter)
                filter_note = f" [{corpus_filter}]" if corpus_filter else ""
                obs_prefix = f"search_bm25({new_query!r}{filter_note})"

            elif action == "search_semantic":
                new_query = args.get("query", "")
                corpus_filter = args.get("corpus_filter")
                if not new_query:
                    step_rec["observation"] = "skipped (empty query)"
                    steps.append(step_rec)
                    action_log.append(f"[S{step_n}] search_semantic -> skipped (empty)")
                    continue
                if full_emb is None:
                    step_rec["observation"] = "skipped (dense embeddings unavailable)"
                    steps.append(step_rec)
                    action_log.append(f"[S{step_n}] search_semantic -> skipped (no embeddings)")
                    continue
                if tools.is_repeated_query(new_query, state.call_history, corpus_filter=corpus_filter, action_key="search_semantic"):
                    consecutive_repeats += 1
                    step_rec["observation"] = f"skipped (already tried search_semantic({new_query!r}, {corpus_filter}) — use different terms or tool)"
                    steps.append(step_rec)
                    action_log.append(f"[S{step_n}] search_semantic({new_query!r}) -> skipped (repetitive)")
                    if consecutive_repeats >= 2:
                        return await _do_finish(state.step_count or None)
                    continue
                consecutive_repeats = 0
                state.call_history.append(("search_semantic", new_query, corpus_filter))
                hits = tools.search_semantic(new_query, full_emb, max_results=tools.PER_CHANNEL_MAX,
                                            candidate_ids=_corpus_candidate_ids(corpus_filter))
                hits = tools.filter_hits_by_corpus(hits, metadata_index, corpus_filter)
                filter_note = f" [{corpus_filter}]" if corpus_filter else ""
                obs_prefix = f"search_semantic({new_query!r}{filter_note})"

            elif action == "search_date":
                date_start = args.get("date_start")
                date_end = args.get("date_end")
                date_str = str(args.get("date", "")).strip()
                window_days = args.get("window_days", 7)
                corpus_filter = args.get("corpus_filter")

                use_range = bool(
                    date_start and date_end
                    and str(date_start).strip() and str(date_end).strip()
                )

                if use_range:
                    date_start = str(date_start).strip()
                    date_end = str(date_end).strip()
                    if (
                        not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_start)
                        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_end)
                    ):
                        step_rec["observation"] = "skipped (invalid date range)"
                        steps.append(step_rec)
                        action_log.append(f"[S{step_n}] search_date -> skipped (invalid)")
                        continue
                    dedup_key = f"{date_start}|{date_end}"
                else:
                    if not date_str or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
                        step_rec["observation"] = "skipped (invalid date)"
                        steps.append(step_rec)
                        action_log.append(f"[S{step_n}] search_date -> skipped (invalid)")
                        continue
                    try:
                        window_days = max(1, min(int(window_days), 60))
                    except (TypeError, ValueError):
                        window_days = 7
                    dedup_key = f"{date_str}|{window_days}"

                corpus_filter_norm = corpus_filter if corpus_filter else None
                is_dup = any(
                    key == "search_date" and prev_q == dedup_key
                    and prev_f == corpus_filter_norm
                    for key, prev_q, prev_f in state.call_history
                )
                if is_dup:
                    consecutive_repeats += 1
                    step_rec["observation"] = f"skipped (already tried search_date({dedup_key}, {corpus_filter}) — try different dates or tool)"
                    steps.append(step_rec)
                    action_log.append(f"[S{step_n}] search_date -> skipped (repetitive)")
                    if consecutive_repeats >= 2:
                        return await _do_finish(state.step_count or None)
                    continue
                consecutive_repeats = 0
                state.call_history.append(("search_date", dedup_key, corpus_filter_norm))
                date_cids = _corpus_candidate_ids(corpus_filter)

                if use_range:
                    hits = tools.search_date_range(
                        date_start, date_end, date_index,
                        max_results=tools.PER_CHANNEL_MAX,
                        chunked_docs=chunked_docs,
                        candidate_ids=date_cids,
                    )
                    obs_prefix = f"search_date({date_start}..{date_end}"
                else:
                    hits = tools.search_date(
                        date_str, date_index, window_days=window_days,
                        max_results=tools.PER_CHANNEL_MAX,
                        chunked_docs=chunked_docs,
                        candidate_ids=date_cids,
                    )
                    obs_prefix = f"search_date({date_str!r}, +/-{window_days}d"

                hits = tools.filter_hits_by_corpus(hits, metadata_index, corpus_filter)

                if types_mod.ENABLE_SEARCH_EXPANSION:
                    base_ids = {h.doc_id for h in hits}
                    try:
                        if use_range:
                            exp_start = (datetime.strptime(date_start, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
                            exp_end = (datetime.strptime(date_end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                            exp_hits = tools.search_date_range(
                                exp_start, exp_end, date_index,
                                max_results=tools.PER_CHANNEL_MAX,
                                chunked_docs=chunked_docs,
                                candidate_ids=date_cids,
                            )
                        else:
                            exp_hits = tools.search_date(
                                date_str, date_index, window_days=window_days + 1,
                                max_results=tools.PER_CHANNEL_MAX,
                                chunked_docs=chunked_docs,
                                candidate_ids=date_cids,
                            )
                        exp_hits = tools.filter_hits_by_corpus(exp_hits, metadata_index, corpus_filter)
                        accepted_set_for_exp = {a.doc_id for a in state.accepted}
                        held_parents_for_exp = set()
                        for hk in state.held:
                            hd = chunked_docs.get(hk)
                            held_parents_for_exp.add(hd.doc_id if hd else hk)
                        for h in exp_hits:
                            if (h.doc_id not in base_ids
                                    and h.doc_id not in accepted_set_for_exp
                                    and (h.chunk_key or h.doc_id) not in state.rejected
                                    and h.doc_id not in held_parents_for_exp):
                                ck = h.chunk_key or h.doc_id
                                state.held.append(ck)
                                held_parents_for_exp.add(h.doc_id)
                    except (ValueError, Exception):
                        pass

                filter_note = f" [{corpus_filter}]" if corpus_filter else ""
                obs_prefix += f"{filter_note})"

            elif action == "search_grep":
                from .tools import search_grep as tool_grep
                query_text = args.get("query", "")
                corpus_filter = args.get("corpus_filter")
                if not query_text:
                    step_rec["observation"] = "skipped (empty query)"
                    steps.append(step_rec)
                    action_log.append(f"[S{step_n}] search_grep -> skipped (empty)")
                    continue
                if tools.is_repeated_query(query_text, state.call_history, corpus_filter=corpus_filter, action_key="search_grep"):
                    consecutive_repeats += 1
                    step_rec["observation"] = f"skipped (already tried search_grep({query_text!r}, {corpus_filter}) — use different terms or tool)"
                    steps.append(step_rec)
                    action_log.append(f"[S{step_n}] search_grep({query_text!r}) -> skipped (repetitive)")
                    if consecutive_repeats >= 2:
                        return await _do_finish(state.step_count or None)
                    continue
                consecutive_repeats = 0
                state.call_history.append(("search_grep", query_text, corpus_filter))
                grep_candidate_ids = None
                if corpus_filter:
                    grep_candidate_ids = list({
                        m.doc_id for m in metadata_index.values()
                        if corpus_filter == m.corpus or corpus_filter == m.corpus_type
                    })
                hits = tool_grep(query_text, chunked_docs, max_results=tools.PER_CHANNEL_MAX,
                                 candidate_ids=grep_candidate_ids)
                hits = tools.filter_hits_by_corpus(hits, metadata_index, corpus_filter)
                filter_note = f" [{corpus_filter}]" if corpus_filter else ""
                obs_prefix = f"search_grep({query_text!r}{filter_note})"

            # Dedup: skip accepted parents, rejected chunks, held parents
            accepted_ids_set = {a.doc_id for a in state.accepted}
            held_parents = set()
            for hk in state.held:
                hd = chunked_docs.get(hk)
                held_parents.add(hd.doc_id if hd else hk)
            new_hits = [
                h for h in hits
                if h.doc_id not in accepted_ids_set
                and (h.chunk_key or h.doc_id) not in state.rejected
                and h.doc_id not in held_parents
            ]
            new_hits = new_hits[:tools.PENDING_REVIEW_CAP]

            if not new_hits:
                consecutive_empty_searches += 1
                obs = f"{obs_prefix}: 0 new docs (all already accepted/rejected)"
                step_rec["observation"] = obs
                steps.append(step_rec)
                action_log.append(f"[S{step_n}] {obs}")

                if consecutive_empty_searches >= tools.EMPTY_SEARCH_WARN_THRESHOLD:
                    pending_tool_output = (
                        f"WARNING: {consecutive_empty_searches} consecutive searches "
                        "returned 0 new documents. Consider calling 'finish'."
                    )
                else:
                    pending_tool_output = f"{obs_prefix}: (no new results)"

                if not quiet:
                    print(f"             Observation: {obs}")
                effective_steps += 1
                continue

            consecutive_empty_searches = 0

            # Use chunk_keys for pending (for text lookup), fall back to doc_id
            all_chunk_keys = [h.chunk_key or h.doc_id for h in new_hits]
            first_batch = all_chunk_keys[:limits.review_batch_size]
            remaining = all_chunk_keys[limits.review_batch_size:]

            state.pending_review = first_batch
            state.pending_queue = remaining
            state.phase = "PENDING_REVIEW"

            obs = f"{obs_prefix}: {len(new_hits)} new docs for review"
            step_rec["observation"] = obs
            steps.append(step_rec)
            action_log.append(f"[S{step_n}] {obs}")

            batch_text = tools.load_review_batch(
                first_batch, chunked_docs, metadata_index,
            )
            if remaining:
                pending_tool_output = (
                    f"{obs_prefix}: {len(new_hits)} new documents for review.\n"
                    f"Showing batch 1 ({len(first_batch)} docs, "
                    f"{len(remaining)} remaining).\n"
                    f"Full text of each document:\n\n{batch_text}"
                )
            else:
                pending_tool_output = (
                    f"{obs_prefix}: {len(new_hits)} new documents for review.\n"
                    f"Full text of each document:\n\n{batch_text}"
                )

            if not quiet:
                print(f"             Observation: {obs}")
            effective_steps += 1
            continue

        step_rec["observation"] = f"unknown action: {action}"
        steps.append(step_rec)
        action_log.append(f"[S{step_n}] unknown action: {action}")
        if not quiet:
            print(f"             Unknown action: {action}")
        continue

    return await _do_finish(state.step_count or None)
