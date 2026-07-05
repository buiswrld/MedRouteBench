"""
Evaluation metrics for staged reasoning traces.
"""
from collections import Counter
from typing import Dict, List, Optional

from .schema import FINAL_ALLOWED, NONFINAL_ALLOWED


# ── individual metrics ───────────────────────────────────────────────────────

def action_accuracy_by_stage(traces: List[dict]) -> dict:
    """
    Returns two views:
      by_absolute_stage: stage index → accuracy (cases that reached that index).
      by_role:           "nonfinal" / "final" → accuracy.
    Oracle: nonfinal → NONFINAL_ALLOWED; final → FINAL_ALLOWED.
    """
    per_abs: Dict[int, List[int]] = {}
    per_role: Dict[str, List[int]] = {"nonfinal": [0, 0], "final": [0, 0]}
    for t in traces:
        for s in t["stages"]:
            role = "final" if s["is_final"] else "nonfinal"
            allowed = FINAL_ALLOWED if role == "final" else NONFINAL_ALLOWED
            per_abs.setdefault(s["stage"], [0, 0])
            per_abs[s["stage"]][1] += 1
            per_role[role][1]      += 1
            if s["parsed"]["action"] in allowed:
                per_abs[s["stage"]][0] += 1
                per_role[role][0]      += 1
    return {
        "by_absolute_stage": {k: (c / n if n else None) for k, (c, n) in sorted(per_abs.items())},
        "by_role":           {k: (c / n if n else None) for k, (c, n) in per_role.items()},
    }


def final_answer_accuracy(traces: List[dict], gt: dict) -> dict:
    correct = total = 0
    for t in traces:
        ans = t["stages"][-1]["parsed"]["answer"]
        if ans in {"yes", "no", "maybe"}:
            total += 1
            if ans == gt.get(t["pmid"]):
                correct += 1
    return {
        "accuracy": (correct / total if total else 0.0),
        "committed": total,
        "n": len(traces),
    }


def revision_correctness(traces: List[dict], gt: dict) -> dict:
    """
    A 'revision' occurs when consecutive stages have different non-null answers.
    Correctness = fraction of revisions that moved the answer closer to GT.
    """
    rev = corr = 0
    for t in traces:
        for i in range(1, len(t["stages"])):
            prev_a = t["stages"][i - 1]["parsed"]["answer"]
            new_a  = t["stages"][i]["parsed"]["answer"]
            if prev_a and new_a and prev_a != new_a:
                rev += 1
                if new_a == gt.get(t["pmid"]):
                    corr += 1
    return {"correctness": (corr / rev if rev else None), "n_revisions": rev}


def premature_answer_rate(traces: List[dict], gt: dict) -> float:
    """Fraction of cases where the agent committed a wrong answer at a non-final stage."""
    prem = 0
    for t in traces:
        for s in t["stages"][:-1]:
            if s["parsed"]["action"] == "ANSWER" and s["parsed"]["answer"] != gt.get(t["pmid"]):
                prem += 1
                break
    return prem / len(traces) if traces else 0.0


def missed_revision_rate(traces: List[dict], gt: dict) -> dict:
    """
    Eligible = final stage had a wrong penultimate answer.
    Missed   = agent did not use REVISE_ANSWER (or kept the same answer).
    """
    missed = eligible = 0
    for t in traces:
        stages = t["stages"]
        if len(stages) < 2:
            continue
        prev = stages[-2]["parsed"]["answer"]
        final = stages[-1]["parsed"]
        if prev and prev != gt.get(t["pmid"]):
            eligible += 1
            if final["action"] != "REVISE_ANSWER" or final["answer"] == prev:
                missed += 1
    return {"rate": (missed / eligible if eligible else None), "n_eligible": eligible}


def abstention_rate(traces: List[dict]) -> float:
    if not traces:
        return 0.0
    return sum(1 for t in traces if t["stages"][-1]["parsed"]["action"] == "ABSTAIN") / len(traces)


# ── report builder ────────────────────────────────────────────────────────────

def build_report(
    traces: List[dict],
    gt: dict,
    model: str,
    sample_counts: Optional[dict] = None,
) -> dict:
    """Assemble the full metrics report dict."""
    return {
        "model": model,
        "n_cases": len(traces),
        "sample_label_counts": sample_counts,
        "action_accuracy_by_stage": action_accuracy_by_stage(traces),
        "final_answer_accuracy":    final_answer_accuracy(traces, gt),
        "revision_correctness":     revision_correctness(traces, gt),
        "premature_answer_rate":    premature_answer_rate(traces, gt),
        "missed_revision_rate":     missed_revision_rate(traces, gt),
        "abstention_rate":          abstention_rate(traces),
    }
