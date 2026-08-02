"""Aggregate reference-path metrics for MedCTA controlled replay."""

from collections import Counter
from typing import Optional

from .answer_scoring import strict_answer_match


def _ratio(numerator: int, denominator: int) -> dict:
    return {
        "rate": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def _steps(traces: list[dict]):
    for trace in traces:
        yield from trace.get("steps") or []


def _evaluable_traces(traces: list[dict]) -> list[dict]:
    """Exclude cases where no routing evaluation could be completed.

    Premature finalization and other model decisions remain evaluable. Only
    provider/infrastructure failures are excluded from rate cohorts; those
    failures are surfaced separately in the report diagnostics.
    """
    return [trace for trace in traces if trace.get("status") != "inference_failure"]


def _valid_actual(step: dict):
    output = step.get("model_output") or {}
    if not output.get("valid"):
        return None
    return output.get("parsed")


def next_tool_accuracy(traces: list[dict]) -> dict:
    traces = _evaluable_traces(traces)
    # Count only steps where the reference expected a tool call and the model
    # was actually evaluated (i.e., a step_trace exists for that position).
    evaluated_tool_steps = [
        step for step in _steps(traces)
        if step.get("expected_action") == "CALL_TOOL"
    ]
    denominator = len(evaluated_tool_steps)
    numerator = sum(step.get("reference_tool_match") is True for step in evaluated_tool_steps)
    return _ratio(numerator, denominator)


def trajectory_step_accuracy(traces: list[dict]) -> dict:
    traces = _evaluable_traces(traces)
    denominator = sum(int(trace.get("reference_step_count") or 0) for trace in traces)
    numerator = sum(bool(step.get("action_match")) for step in _steps(traces))
    return _ratio(numerator, denominator)


def premature_finalization_rate(traces: list[dict]) -> dict:
    traces = _evaluable_traces(traces)
    numerator = sum(trace.get("status") == "premature_finalization" for trace in traces)
    return _ratio(numerator, len(traces))


def missed_finalization_rate(traces: list[dict]) -> dict:
    traces = _evaluable_traces(traces)
    reached_valid_final = [
        step
        for step in _steps(traces)
        if step.get("expected_action") == "FINAL_ANSWER" and _valid_actual(step)
    ]
    numerator = sum(
        _valid_actual(step)["action"] == "CALL_TOOL" for step in reached_valid_final
    )
    return _ratio(numerator, len(reached_valid_final))


def tool_precision(traces: list[dict]) -> dict:
    traces = _evaluable_traces(traces)
    calls = [
        step
        for step in _steps(traces)
        if _valid_actual(step) and _valid_actual(step)["action"] == "CALL_TOOL"
    ]
    numerator = sum(step.get("reference_tool_match") is True for step in calls)
    return _ratio(numerator, len(calls))


def unnecessary_tool_rate(traces: list[dict]) -> dict:
    traces = _evaluable_traces(traces)
    calls = [
        step
        for step in _steps(traces)
        if _valid_actual(step) and _valid_actual(step)["action"] == "CALL_TOOL"
    ]
    numerator = sum(step.get("expected_action") == "FINAL_ANSWER" for step in calls)
    return _ratio(numerator, len(calls))


def trajectory_exact_match_rate(traces: list[dict]) -> dict:
    traces = _evaluable_traces(traces)
    numerator = sum(bool(trace.get("trajectory_exact_match")) for trace in traces)
    return _ratio(numerator, len(traces))


def strict_final_answer_match_rate(traces: list[dict]) -> dict:
    traces = _evaluable_traces(traces)
    numerator = sum(
        strict_answer_match(
            trace.get("final_answer"),
            trace.get("accepted_ground_truth_answers") or [],
        )
        for trace in traces
    )
    return _ratio(numerator, len(traces))


def invalid_action_rate(traces: list[dict]) -> dict:
    traces = _evaluable_traces(traces)
    responded = [
        step
        for step in _steps(traces)
        if (step.get("model_output") or {}).get("initial_response_received")
    ]
    numerator = sum(
        bool(step["model_output"].get("initial_invalid")) for step in responded
    )
    return _ratio(numerator, len(responded))


def build_report(
    traces: list[dict],
    model: str,
    *,
    backend: Optional[str] = None,
    run_id: Optional[str] = None,
    provenance: Optional[dict] = None,
) -> dict:
    evaluable_traces = _evaluable_traces(traces)
    inference_failure_count = sum(
        bool((step.get("model_output") or {}).get("inference_failure"))
        for step in _steps(traces)
    )
    return {
        "run_id": run_id,
        "model": model,
        "backend": backend,
        "provenance": provenance,
        "reference_matching_note": (
            "Routing metrics measure agreement with one MedCTA reference trajectory, "
            "not absolute clinical correctness or uniqueness of the tool path."
        ),
        "answer_scoring_note": (
            "The inference run records normalized exact-whitelist matching only. "
            "Semantic answer correctness is produced by the separate, versioned "
            "judge_eval pipeline and must be human-validated."
        ),
        "n_selected_cases": len(traces),
        "n_evaluable_cases": len(evaluable_traces),
        "n_completed_cases": sum(trace.get("status") == "completed" for trace in traces),
        "trace_status_counts": dict(Counter(trace.get("status") for trace in traces)),
        "next_tool_accuracy": next_tool_accuracy(traces),
        "trajectory_step_accuracy": trajectory_step_accuracy(traces),
        "premature_finalization_rate": premature_finalization_rate(traces),
        "missed_finalization_rate": missed_finalization_rate(traces),
        "tool_precision": tool_precision(traces),
        "unnecessary_tool_rate": unnecessary_tool_rate(traces),
        "trajectory_exact_match_rate": trajectory_exact_match_rate(traces),
        "strict_final_answer_match_rate": strict_final_answer_match_rate(traces),
        "invalid_action_rate": invalid_action_rate(traces),
        "inference_failure_count": inference_failure_count,
        "inference_failure_case_count": sum(
            trace.get("status") == "inference_failure" for trace in traces
        ),
        "repair_count": sum(
            bool((step.get("model_output") or {}).get("repaired"))
            for step in _steps(traces)
        ),
    }
