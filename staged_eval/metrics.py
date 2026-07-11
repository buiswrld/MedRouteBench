"""Metrics for the fixed two-stage PubMedQA revision experiment."""

from collections import Counter
from typing import List, Optional


def _ratio(numerator: int, denominator: int) -> dict:
    return {
        "rate": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def _valid_stage1(traces: List[dict]) -> List[dict]:
    return [
        trace
        for trace in traces
        if trace.get("stage1_model_output")
        and trace["stage1_model_output"].get("valid")
    ]


def _completed(traces: List[dict]) -> List[dict]:
    return [trace for trace in traces if trace.get("status") == "completed"]


def _answers(trace: dict):
    stage1 = trace["stage1_model_output"]["parsed"]
    stage2 = trace["stage2_model_output"]["parsed"]
    return stage1["answer"], stage2["answer"], stage2["action"]


def stage1_answer_accuracy(traces: List[dict]) -> dict:
    """Accuracy among traces with a valid Stage 1 answer."""
    eligible = _valid_stage1(traces)
    correct = sum(
        trace["stage1_model_output"]["parsed"]["answer"]
        == trace["pubmedqa_gold_label"]
        for trace in eligible
    )
    return _ratio(correct, len(eligible))


def final_answer_accuracy(traces: List[dict]) -> dict:
    """Accuracy among completed cases; a final abstention is incorrect."""
    completed = _completed(traces)
    correct = sum(
        trace["stage2_model_output"]["parsed"]["answer"]
        == trace["pubmedqa_gold_label"]
        for trace in completed
    )
    return _ratio(correct, len(completed))


def successful_revision_rate(traces: List[dict]) -> dict:
    """Rate of correction among completed cases whose Stage 1 answer was wrong."""
    eligible = [
        trace
        for trace in _completed(traces)
        if _answers(trace)[0] != trace["pubmedqa_gold_label"]
    ]
    successful = sum(
        _answers(trace)[2] == "REVISE_ANSWER"
        and _answers(trace)[1] == trace["pubmedqa_gold_label"]
        for trace in eligible
    )
    return _ratio(successful, len(eligible))


def missed_revision_rate(traces: List[dict]) -> dict:
    """Failure-to-correct rate among wrong Stage 1 cases, excluding abstentions."""
    eligible = [
        trace
        for trace in _completed(traces)
        if _answers(trace)[0] != trace["pubmedqa_gold_label"]
        and _answers(trace)[2] != "ABSTAIN"
    ]
    missed = sum(
        _answers(trace)[1] != trace["pubmedqa_gold_label"] for trace in eligible
    )
    return _ratio(missed, len(eligible))


def overreaction_rate(traces: List[dict]) -> dict:
    """Rate of changing a correct Stage 1 answer to an incorrect final answer."""
    eligible = [
        trace
        for trace in _completed(traces)
        if _answers(trace)[0] == trace["pubmedqa_gold_label"]
    ]
    overreactions = sum(
        _answers(trace)[2] == "REVISE_ANSWER"
        and _answers(trace)[1] != trace["pubmedqa_gold_label"]
        for trace in eligible
    )
    return _ratio(overreactions, len(eligible))


def kept_correct_rate(traces: List[dict]) -> dict:
    """Rate of explicitly keeping a correct Stage 1 answer."""
    eligible = [
        trace
        for trace in _completed(traces)
        if _answers(trace)[0] == trace["pubmedqa_gold_label"]
    ]
    kept = sum(
        _answers(trace)[2] == "KEEP_ANSWER"
        and _answers(trace)[1] == trace["pubmedqa_gold_label"]
        for trace in eligible
    )
    return _ratio(kept, len(eligible))


def final_abstention_rate(traces: List[dict]) -> dict:
    """Final ABSTAIN rate among completed cases."""
    completed = _completed(traces)
    abstentions = sum(_answers(trace)[2] == "ABSTAIN" for trace in completed)
    return _ratio(abstentions, len(completed))


def maintenance_rate(traces: List[dict]) -> dict:
    """Stage 1/final answer agreement among completed non-abstention cases."""
    eligible = [
        trace for trace in _completed(traces) if _answers(trace)[2] != "ABSTAIN"
    ]
    maintained = sum(_answers(trace)[0] == _answers(trace)[1] for trace in eligible)
    return _ratio(maintained, len(eligible))


def build_report(
    traces: List[dict],
    model: str,
    sample_counts: Optional[dict] = None,
    dataset_counts: Optional[dict] = None,
) -> dict:
    """Assemble metrics plus enough denominators to audit every reported rate."""
    completed = _completed(traces)
    return {
        "model": model,
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
