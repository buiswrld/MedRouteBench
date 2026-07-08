"""
Evaluation metrics for staged reasoning traces.
"""
from typing import Dict, List, Optional

from .oracle import FINAL_ANSWERS, expected_action_for_trace


def _gt_for_trace(trace: dict, gt: Optional[dict]) -> Optional[str]:
    if gt is not None:
        return gt.get(trace["pmid"], trace.get("gt"))
    return trace.get("gt")


# ── individual metrics ───────────────────────────────────────────────────────

def action_accuracy_by_stage(traces: List[dict], gt: Optional[dict] = None) -> dict:
    """
    Returns two views:
      by_absolute_stage: stage index → accuracy (cases that reached that index).
      by_role:           "nonfinal" / "final" → accuracy.
    Oracle: nonfinal -> RETRIEVE_EVIDENCE; final -> ANSWER,
    REVISE_ANSWER, or ABSTAIN depending on gold label and prior answer.
    """
    per_abs: Dict[int, List[int]] = {}
    per_role: Dict[str, List[int]] = {"nonfinal": [0, 0], "final": [0, 0]}
    for t in traces:
        for i, s in enumerate(t["stages"]):
            role = "final" if s["is_final"] else "nonfinal"
            expected = expected_action_for_trace(t, i, gt)
            per_abs.setdefault(s["stage"], [0, 0])
            per_abs[s["stage"]][1] += 1
            per_role[role][1]      += 1
            if s["parsed"]["action"] == expected:
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
        gt_label = _gt_for_trace(t, gt)
        if ans in FINAL_ANSWERS:
            total += 1
            if ans == gt_label:
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
        gt_label = _gt_for_trace(t, gt)
        for i in range(1, len(t["stages"])):
            prev_a = t["stages"][i - 1]["parsed"]["answer"]
            new_a  = t["stages"][i]["parsed"]["answer"]
            if prev_a and new_a and prev_a != new_a:
                rev += 1
                if new_a == gt_label:
                    corr += 1
    return {"correctness": (corr / rev if rev else None), "n_revisions": rev}


def premature_answer_rate(traces: List[dict], gt: dict) -> float:
    """Fraction of cases where the agent finalized with ANSWER before the final stage."""
    prem = 0
    for t in traces:
        for s in t["stages"][:-1]:
            if s["parsed"]["action"] == "ANSWER":
                prem += 1
                break
    return prem / len(traces) if traces else 0.0


def missed_revision_rate(traces: List[dict], gt: dict) -> dict:
    """
    Eligible = the final-stage oracle expects REVISE_ANSWER.
    Missed   = agent did not revise to the gold answer.
    """
    missed = eligible = 0
    for t in traces:
        stages = t["stages"]
        if len(stages) < 2:
            continue
        gt_label = _gt_for_trace(t, gt)
        final = stages[-1]["parsed"]
        expected = expected_action_for_trace(t, len(stages) - 1, gt)
        if expected == "REVISE_ANSWER":
            eligible += 1
            if final["action"] != "REVISE_ANSWER" or final["answer"] != gt_label:
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
        "action_accuracy_by_stage": action_accuracy_by_stage(traces, gt),
        "final_answer_accuracy":    final_answer_accuracy(traces, gt),
        "revision_correctness":     revision_correctness(traces, gt),
        "premature_answer_rate":    premature_answer_rate(traces, gt),
        "missed_revision_rate":     missed_revision_rate(traces, gt),
        "abstention_rate":          abstention_rate(traces),
    }
