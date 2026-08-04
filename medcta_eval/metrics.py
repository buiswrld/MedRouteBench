"""Aggregate reference-path metrics for MedCTA controlled replay.

Naming convention: every metric here consumes fields that were already
thresholded upstream in runner.py (current_answer_correct, final_answer_match,
answer_changed, etc. — all booleans, compared against
FINAL_ACCURACY_CONFIDENCE_THRESHOLD once at the source) and returns
`_ratio(numerator, denominator)` -> {"rate", "numerator", "denominator"}.

The lone exception is final_answer_mean_score, which reports the mean of the
*raw, untresholded* 0-1 judge score instead of a pass/fail rate — it returns
a differently-shaped dict ({"mean_score", "n_scored", "n_evaluable"}) as a
visual flag that it isn't a _ratio() rate. Compare it against
final_answer_accuracy (the thresholded rate over the same underlying scores)
to see the score distribution behind the pass/fail cutoff.
"""

from collections import Counter
from typing import Optional

from shared.pipeline_utils import ratio as _ratio


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


def _scored_steps(traces: list[dict]) -> list[dict]:
    return [
        step for step in _steps(_evaluable_traces(traces))
        if step.get("current_answer") is not None
    ]


def _transitions(traces: list[dict]):
    for trace in _evaluable_traces(traces):
        yield from trace.get("stage_transitions") or []


def next_tool_accuracy(traces: list[dict]) -> dict:
    traces = _evaluable_traces(traces)
    # Denominator is all reference CALL_TOOL steps, including those never reached
    # due to premature finalization. Unreached steps score zero per the MedCTA paper.
    denominator = sum(
        len(trace.get("reference_tool_sequence") or []) for trace in traces
    )
    evaluated_tool_steps = [
        step for step in _steps(traces)
        if step.get("expected_action") == "CALL_TOOL"
    ]
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


def final_answer_accuracy(traces: list[dict]) -> dict:
    traces = _evaluable_traces(traces)
    numerator = sum(bool(trace.get("final_answer_match")) for trace in traces)
    return _ratio(numerator, len(traces))


def final_answer_mean_score(traces: list[dict]) -> dict:
    """Mean RAW (un-thresholded) LLM judge score, 0-1, not a pass/fail rate.

    Contrast with final_answer_accuracy, which reports the fraction of cases
    where this same score cleared FINAL_ACCURACY_CONFIDENCE_THRESHOLD.
    """
    traces = _evaluable_traces(traces)
    scored = [
        trace for trace in traces
        if isinstance(trace.get("final_answer_score"), (int, float))
    ]
    if not scored:
        return {"mean_score": None, "n_scored": len(scored), "n_evaluable": len(traces)}
    mean = sum(t["final_answer_score"] for t in scored) / len(scored)
    return {"mean_score": round(mean, 4), "n_scored": len(scored), "n_evaluable": len(traces)}


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


# ── Revision metrics ──────────────────────────────────────────────────────
# Track the model's stage-by-stage current-best answer as new tool evidence
# arrives, distinct from the routing metrics above.

def stage_answer_accuracy(traces: list[dict]) -> dict:
    steps = _scored_steps(traces)
    numerator = sum(bool(step.get("current_answer_correct")) for step in steps)
    return _ratio(numerator, len(steps))


def stage_answer_accuracy_by_step(traces: list[dict]) -> dict:
    """Bonus breakdown: accuracy at each specific reference step_index."""
    by_index: dict = {}
    for step in _scored_steps(traces):
        by_index.setdefault(step.get("step_index"), []).append(step)
    return {
        str(index): _ratio(
            sum(bool(s.get("current_answer_correct")) for s in group), len(group)
        )
        for index, group in sorted(by_index.items())
    }


def successful_revision_rate(traces: list[dict]) -> dict:
    transitions = [t for t in _transitions(traces) if t.get("prev_correct") is False]
    numerator = sum(t.get("label") == "successful_revision" for t in transitions)
    return _ratio(numerator, len(transitions))


def missed_revision_rate(traces: list[dict]) -> dict:
    transitions = [t for t in _transitions(traces) if t.get("prev_correct") is False]
    numerator = sum(t.get("label") == "missed_revision" for t in transitions)
    return _ratio(numerator, len(transitions))


def overreaction_rate(traces: list[dict]) -> dict:
    transitions = [t for t in _transitions(traces) if t.get("prev_correct") is True]
    numerator = sum(t.get("label") == "overreaction" for t in transitions)
    return _ratio(numerator, len(transitions))


def kept_correct_rate(traces: list[dict]) -> dict:
    transitions = [t for t in _transitions(traces) if t.get("prev_correct") is True]
    numerator = sum(t.get("label") == "kept_correct" for t in transitions)
    return _ratio(numerator, len(transitions))


def answer_change_rate(traces: list[dict]) -> dict:
    transitions = list(_transitions(traces))
    numerator = sum(bool(t.get("answer_changed")) for t in transitions)
    return _ratio(numerator, len(transitions))


def maintenance_rate(traces: list[dict]) -> dict:
    """Complement of answer_change_rate — reported explicitly for symmetry."""
    transitions = list(_transitions(traces))
    numerator = sum(not t.get("answer_changed") for t in transitions)
    return _ratio(numerator, len(transitions))


def maintained_wrong_rate(traces: list[dict]) -> dict:
    transitions = [t for t in _transitions(traces) if t.get("prev_correct") is False]
    numerator = sum(bool(t.get("maintained_wrong")) for t in transitions)
    return _ratio(numerator, len(transitions))


def answer_stability(traces: list[dict]) -> dict:
    applicable = [
        trace for trace in _evaluable_traces(traces)
        if trace.get("answer_stable_throughout") is not None
    ]
    numerator = sum(trace.get("answer_stable_throughout") is True for trace in applicable)
    return _ratio(numerator, len(applicable))


# ── Stopping / confidence metrics ─────────────────────────────────────────
# A secondary, distinct set from the revision metrics above: these study
# when the model decides it has gathered enough evidence to stop, rather
# than whether its answer improves as evidence arrives.

def premature_finalization_wrong(traces: list[dict]) -> dict:
    traces = [
        trace for trace in _evaluable_traces(traces)
        if trace.get("status") == "premature_finalization"
    ]
    numerator = sum(trace.get("final_answer_match") is False for trace in traces)
    return _ratio(numerator, len(traces))


def early_correct_finalization(traces: list[dict]) -> dict:
    traces = [
        trace for trace in _evaluable_traces(traces)
        if trace.get("status") == "premature_finalization"
    ]
    numerator = sum(trace.get("final_answer_match") is True for trace in traces)
    return _ratio(numerator, len(traces))


def unnecessary_tool_calls(traces: list[dict]) -> dict:
    """Steps with a correct current answer, backed by evidence, but chose CALL_TOOL anyway.

    Requires evidence_shown to be non-empty, excluding each trace's step 0
    (before any reference observation has been revealed): a correct answer
    formed with zero evidence is a lucky guess, not a signal the model was
    ready to stop, and counting it would flag a model that dutifully follows
    the full reference tool sequence as "unnecessary" purely because it
    happened to guess right before gathering anything.
    """
    steps = [
        s for s in _steps(_evaluable_traces(traces))
        if s.get("current_answer_correct") is True and s.get("evidence_shown")
    ]

    def _action(step):
        actual = _valid_actual(step)
        return actual["action"] if actual else None

    numerator = sum(_action(step) == "CALL_TOOL" for step in steps)
    return _ratio(numerator, len(steps))


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
        "final_answer_accuracy": final_answer_accuracy(traces),
        "final_answer_mean_score": final_answer_mean_score(traces),
        "invalid_action_rate": invalid_action_rate(traces),
        "stage_metrics_note": (
            "unnecessary_tool_calls (correctness-based: the model already had a "
            "correct current answer, backed by at least one revealed piece of "
            "evidence, but still called a tool) is distinct from "
            "unnecessary_tool_rate above (reference-position-based: the model "
            "called a tool when the reference trajectory expected FINAL_ANSWER). "
            "A zero-evidence step-0 guess never counts toward "
            "unnecessary_tool_calls, even if it happens to be correct."
        ),
        "stage_answer_accuracy": stage_answer_accuracy(traces),
        "stage_answer_accuracy_by_step": stage_answer_accuracy_by_step(traces),
        "successful_revision_rate": successful_revision_rate(traces),
        "missed_revision_rate": missed_revision_rate(traces),
        "overreaction_rate": overreaction_rate(traces),
        "kept_correct_rate": kept_correct_rate(traces),
        "answer_change_rate": answer_change_rate(traces),
        "maintenance_rate": maintenance_rate(traces),
        "maintained_wrong_rate": maintained_wrong_rate(traces),
        "answer_stability": answer_stability(traces),
        "premature_finalization_wrong": premature_finalization_wrong(traces),
        "early_correct_finalization": early_correct_finalization(traces),
        "unnecessary_tool_calls": unnecessary_tool_calls(traces),
        "inference_failure_count": inference_failure_count,
        "inference_failure_case_count": sum(
            trace.get("status") == "inference_failure" for trace in traces
        ),
        "repair_count": sum(
            bool((step.get("model_output") or {}).get("repaired"))
            for step in _steps(traces)
        ),
    }
