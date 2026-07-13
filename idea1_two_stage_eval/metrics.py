"""
Evaluation metrics for the fixed two-stage revision experiment.
"""
from typing import List, Optional


def _rate(count: int, total: int) -> Optional[float]:
    return count / total if total else None


def _completed_cases(traces: List[dict]) -> List[dict]:
    return [trace for trace in traces if trace.get("completed")]


def _stage1_eligible(traces: List[dict]) -> List[dict]:
    return [trace for trace in traces if trace.get("stage1_valid")]


def stage1_answer_accuracy(traces: List[dict]) -> dict:
    eligible = _stage1_eligible(traces)
    correct = sum(
        1 for trace in eligible
        if trace["stage1_model_output"]["answer"] == trace["gold_label"]
    )
    return {
        "accuracy": _rate(correct, len(eligible)),
        "n_valid": len(eligible),
        "n_total": len(traces),
    }


def final_answer_accuracy(traces: List[dict]) -> dict:
    completed = _completed_cases(traces)
    correct = sum(1 for trace in completed if trace["final_answer"] == trace["gold_label"])
    return {
        "accuracy": _rate(correct, len(completed)),
        "n_completed": len(completed),
        "n_total": len(traces),
    }


def successful_revision_rate(traces: List[dict]) -> dict:
    completed = [
        trace for trace in _completed_cases(traces)
        if trace["stage1_model_output"]["answer"] != trace["gold_label"]
    ]
    count = sum(1 for trace in completed if trace["label"] == "successful_revision")
    return {"rate": _rate(count, len(completed)), "n_eligible": len(completed)}


def missed_revision_rate(traces: List[dict]) -> dict:
    eligible = [
        trace for trace in _completed_cases(traces)
        if trace["stage1_model_output"]["answer"] != trace["gold_label"]
        and trace["label"] != "abstention"
    ]
    count = sum(1 for trace in eligible if trace["label"] == "missed_revision")
    return {"rate": _rate(count, len(eligible)), "n_eligible": len(eligible)}


def overreaction_rate(traces: List[dict]) -> dict:
    completed = [
        trace for trace in _completed_cases(traces)
        if trace["stage1_model_output"]["answer"] == trace["gold_label"]
    ]
    count = sum(1 for trace in completed if trace["label"] == "overreaction")
    return {"rate": _rate(count, len(completed)), "n_eligible": len(completed)}


def kept_correct_rate(traces: List[dict]) -> dict:
    completed = [
        trace for trace in _completed_cases(traces)
        if trace["stage1_model_output"]["answer"] == trace["gold_label"]
    ]
    count = sum(1 for trace in completed if trace["label"] == "kept_correct")
    return {"rate": _rate(count, len(completed)), "n_eligible": len(completed)}


def final_abstention_rate(traces: List[dict]) -> dict:
    completed = _completed_cases(traces)
    count = sum(
        1 for trace in completed
        if trace["stage2_model_output"]["action"] == "ABSTAIN"
    )
    return {"rate": _rate(count, len(completed)), "n_completed": len(completed)}


def maintenance_rate(traces: List[dict]) -> dict:
    completed = [
        trace for trace in _completed_cases(traces)
        if trace["stage2_model_output"]["action"] != "ABSTAIN"
    ]
    count = sum(
        1 for trace in completed
        if trace["stage1_model_output"]["answer"] == trace["final_answer"]
    )
    return {"rate": _rate(count, len(completed)), "n_eligible": len(completed)}


def build_report(
    traces: List[dict],
    *,
    model: str,
    sample_counts: Optional[dict] = None,
    split_counts: Optional[dict] = None,
    n_skipped_ineligible: int = 0,
) -> dict:
    """Assemble the full metrics report dict."""
    return {
        "model": model,
        "n_cases": len(traces),
        "n_completed": len(_completed_cases(traces)),
        "n_skipped_ineligible": n_skipped_ineligible,
        "sample_label_counts": sample_counts,
        "split_strategy_counts": split_counts,
        "stage1_answer_accuracy": stage1_answer_accuracy(traces),
        "final_answer_accuracy": final_answer_accuracy(traces),
        "successful_revision_rate": successful_revision_rate(traces),
        "missed_revision_rate": missed_revision_rate(traces),
        "overreaction_rate": overreaction_rate(traces),
        "kept_correct_rate": kept_correct_rate(traces),
        "final_abstention_rate": final_abstention_rate(traces),
        "maintenance_rate": maintenance_rate(traces),
    }
