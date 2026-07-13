"""Metrics for the fixed two-stage PubMedQA revision experiment."""

from collections import Counter
from typing import List, Optional


FINAL_ANSWERS = {"yes", "no", "maybe"}


def _ratio(numerator: int, denominator: int) -> dict:
    return {
        "rate": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def _scorable(traces: List[dict]) -> List[dict]:
    """Return selected traces with a valid official gold label."""
    return [
        trace
        for trace in traces
        if trace.get("pubmedqa_gold_label") in FINAL_ANSWERS
        and not str(trace.get("status", "")).startswith("skipped_")
    ]


def _completed(traces: List[dict]) -> List[dict]:
    return [trace for trace in traces if trace.get("status") == "completed"]


def _stage1_answer(trace: dict):
    output = trace.get("stage1_model_output") or {}
    parsed = output.get("parsed") or {}
    if not output.get("valid"):
        return None
    return parsed.get("answer")


def _stage2_answer_action(trace: dict):
    if trace.get("status") != "completed":
        return None, None
    output = trace.get("stage2_model_output") or {}
    parsed = output.get("parsed") or {}
    if not output.get("valid"):
        return None, None
    return parsed.get("answer"), parsed.get("action")


def _valid_stage1(traces: List[dict]) -> List[dict]:
    return [
        trace
        for trace in _scorable(traces)
        if _stage1_answer(trace) in FINAL_ANSWERS
    ]


def stage1_answer_accuracy(traces: List[dict]) -> dict:
    """Accuracy across all selected scorable cases; invalid output is incorrect."""
    eligible = _scorable(traces)
    correct = sum(
        _stage1_answer(trace) == trace["pubmedqa_gold_label"] for trace in eligible
    )
    return _ratio(correct, len(eligible))


def final_answer_accuracy(traces: List[dict]) -> dict:
    """Final accuracy on the same cohort as Stage 1; invalid/abstain is incorrect."""
    eligible = _scorable(traces)
    correct = sum(
        _stage2_answer_action(trace)[0] == trace["pubmedqa_gold_label"]
        for trace in eligible
    )
    return _ratio(correct, len(eligible))


def successful_revision_rate(traces: List[dict]) -> dict:
    """Correction rate among cases with a valid but wrong Stage 1 answer."""
    eligible = [
        trace
        for trace in _valid_stage1(traces)
        if _stage1_answer(trace) != trace["pubmedqa_gold_label"]
    ]
    successful = sum(
        _stage2_answer_action(trace)[1] == "REVISE_ANSWER"
        and _stage2_answer_action(trace)[0] == trace["pubmedqa_gold_label"]
        for trace in eligible
    )
    return _ratio(successful, len(eligible))


def missed_revision_rate(traces: List[dict]) -> dict:
    """Failure-to-correct rate among wrong Stage 1 cases, excluding abstentions."""
    eligible = [
        trace
        for trace in _valid_stage1(traces)
        if _stage1_answer(trace) != trace["pubmedqa_gold_label"]
        and _stage2_answer_action(trace)[1] != "ABSTAIN"
    ]
    missed = sum(
        _stage2_answer_action(trace)[0] != trace["pubmedqa_gold_label"]
        for trace in eligible
    )
    return _ratio(missed, len(eligible))


def overreaction_rate(traces: List[dict]) -> dict:
    """Rate of changing a correct Stage 1 answer to an incorrect final answer."""
    eligible = [
        trace
        for trace in _valid_stage1(traces)
        if _stage1_answer(trace) == trace["pubmedqa_gold_label"]
    ]
    overreactions = sum(
        _stage2_answer_action(trace)[1] == "REVISE_ANSWER"
        and _stage2_answer_action(trace)[0] != trace["pubmedqa_gold_label"]
        for trace in eligible
    )
    return _ratio(overreactions, len(eligible))


def kept_correct_rate(traces: List[dict]) -> dict:
    """Rate of explicitly keeping a correct Stage 1 answer."""
    eligible = [
        trace
        for trace in _valid_stage1(traces)
        if _stage1_answer(trace) == trace["pubmedqa_gold_label"]
    ]
    kept = sum(
        _stage2_answer_action(trace)[1] == "KEEP_ANSWER"
        and _stage2_answer_action(trace)[0] == trace["pubmedqa_gold_label"]
        for trace in eligible
    )
    return _ratio(kept, len(eligible))


def final_abstention_rate(traces: List[dict]) -> dict:
    """Final ABSTAIN rate across all selected scorable cases."""
    eligible = _scorable(traces)
    abstentions = sum(
        _stage2_answer_action(trace)[1] == "ABSTAIN" for trace in eligible
    )
    return _ratio(abstentions, len(eligible))


def maintenance_rate(traces: List[dict]) -> dict:
    """Answer agreement after Stage 1, with invalid Stage 2 counted as failure."""
    eligible = [
        trace
        for trace in _valid_stage1(traces)
        if _stage2_answer_action(trace)[1] != "ABSTAIN"
    ]
    maintained = sum(
        _stage2_answer_action(trace)[0] == _stage1_answer(trace)
        for trace in eligible
    )
    return _ratio(maintained, len(eligible))


def build_report(
    traces: List[dict],
    model: str,
    sample_counts: Optional[dict] = None,
    dataset_counts: Optional[dict] = None,
    *,
    backend: Optional[str] = None,
    run_id: Optional[str] = None,
    provenance: Optional[dict] = None,
) -> dict:
    """Assemble metrics plus denominators and run provenance."""
    completed = _completed(traces)
    return {
        "run_id": run_id,
        "model": model,
        "backend": backend,
        "provenance": provenance,
        "n_selected_cases": len(traces),
        "n_completed_cases": len(completed),
        "n_invalid_cases": sum(trace.get("label") == "invalid" for trace in traces),
        "dataset_counts": dataset_counts,
        "sample_label_counts": sample_counts,
        "trace_label_counts": dict(Counter(trace.get("label") for trace in traces)),
        "stage1_answer_accuracy": stage1_answer_accuracy(traces),
        "final_answer_accuracy": final_answer_accuracy(traces),
        "successful_revision_rate": successful_revision_rate(traces),
        "missed_revision_rate": missed_revision_rate(traces),
        "overreaction_rate": overreaction_rate(traces),
        "kept_correct_rate": kept_correct_rate(traces),
        "final_abstention_rate": final_abstention_rate(traces),
        "maintenance_rate": maintenance_rate(traces),
    }
