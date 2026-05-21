# metrics.py
"""Evaluation metrics for retrieval results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .types import RetrievalResult


def recall_at_k(gold_ids: list[str], candidate_ids: list[str], k: int) -> float:
    """Fraction of gold docs found in top-k candidates."""
    if not gold_ids:
        return 0.0
    top_k = set(candidate_ids[:k])
    return len(set(gold_ids) & top_k) / len(gold_ids)


def gold_rank(gold_ids: list[str], candidate_ids: list[str]) -> int | None:
    """Position of first gold doc in candidates (1-indexed). None if not found."""
    gold_set = set(gold_ids)
    for i, cid in enumerate(candidate_ids, 1):
        if cid in gold_set:
            return i
    return None


def precision_in_accepted(gold_ids: list[str], accepted_ids: list[str]) -> float:
    """Fraction of accepted docs that are gold."""
    if not accepted_ids:
        return 0.0
    return len(set(gold_ids) & set(accepted_ids)) / len(accepted_ids)


def recall_in_accepted(gold_ids: list[str], accepted_ids: list[str]) -> float:
    """Fraction of gold docs found in accepted set."""
    if not gold_ids:
        return 0.0
    return len(set(gold_ids) & set(accepted_ids)) / len(gold_ids)


def compute_metrics(
    results: list[RetrievalResult],
    type_map: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Compute aggregate metrics by question type."""
    by_type: dict[str, list[RetrievalResult]] = defaultdict(list)
    for r in results:
        qtype = type_map.get(r.question_id, "Unknown")
        by_type[qtype].append(r)
    by_type["All"] = list(results)

    metrics: dict[str, dict[str, float]] = {}
    for qtype, type_results in by_type.items():
        n = len(type_results)
        if n == 0:
            continue
        total_r3 = total_r5 = total_r10 = total_mrr = 0.0
        total_pacc = total_racc = total_acc = 0.0

        for r in type_results:
            total_r3 += recall_at_k(r.gold_ids, r.candidate_ids, 3)
            total_r5 += recall_at_k(r.gold_ids, r.candidate_ids, 5)
            total_r10 += recall_at_k(r.gold_ids, r.candidate_ids, 10)
            rank = gold_rank(r.gold_ids, r.candidate_ids)
            if rank:
                total_mrr += 1.0 / rank
            acc = r.accepted_ids or []
            total_pacc += precision_in_accepted(r.gold_ids, acc)
            total_racc += recall_in_accepted(r.gold_ids, acc)
            total_acc += len(acc)

        metrics[qtype] = {
            "n": n,
            "r3": total_r3 / n,
            "r5": total_r5 / n,
            "r10": total_r10 / n,
            "mrr": total_mrr / n,
            "p_acc": total_pacc / n,
            "r_acc": total_racc / n,
            "avg_acc": total_acc / n,
        }

    return metrics


def print_results_table(metrics: dict[str, dict[str, float]]) -> None:
    """Print a formatted results table."""
    print("RESULTS")
    print("=" * 90)
    print(f"{'':>26} {'R@3':>5} {'R@5':>5} {'R@10':>5} {'MRR':>5} {'P_acc':>5} {'R_acc':>5} {'#Acc':>5} {'N':>5}")
    print("-" * 90)
    type_order = ["SH", "MH - Generic", "MH - BridgeEntity", "MH - Comparative", "All"]
    for qtype in type_order:
        if qtype not in metrics:
            continue
        m = metrics[qtype]
        print(
            f"{qtype:>26} {m['r3']:>5.2f} {m['r5']:>5.2f} {m['r10']:>5.2f} "
            f"{m['mrr']:>5.2f} {m['p_acc']:>5.2f} {m['r_acc']:>5.2f} "
            f"{m['avg_acc']:>5.1f} {m['n']:>5.0f}"
        )
    print("=" * 90)
