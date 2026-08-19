"""Aggregate metrics for MedCTA forced-continuation ("golden path") replay.

Unlike medcta_eval, this package never terminates a case early because the
model deviated from the reference trajectory -- every case reaches the
reference's true final step (except on inference failure or persistent
invalidity). There is therefore no next_tool_accuracy / tool_precision /
missed_finalization_rate / trajectory_exact_match_rate etc. here: those
routing-fidelity metrics are medcta_eval's job. This module reports only the
metrics that make sense once routing is forced: final-answer quality,
revision behavior as evidence arrives, and how often/how far the model tried
to bail before the reference trajectory allowed it to.

final_answer_mean_score, first_step_answer_score, and
mean_score_before_premature_finalization report means of *raw,
unthresholded* 0-1 judge scores, not pass/fail rates -- they return
{"mean_score", "n_scored", "n_evaluable"} rather than a _ratio() dict.
mean_score_improvement is the same shape but reports a mean of a per-case
per-tool-call *delta* of two such scores. premature_finalization_progress is
a third shape ({"mean_rate", "n_cases"}): a mean of a per-case rate (how
much of the reference tool sequence was completed before the model's first
attempted early exit), not a judge score.
"""

from collections import Counter
from typing import Optional

from shared.pipeline_utils import ratio as _ratio


def _evaluable_traces(traces: list[dict]) -> list[dict]:
    """Exclude cases where no evaluation could be completed at all.

    Only provider/infrastructure failures are excluded from the metric
    cohorts; those failures are surfaced separately in the report
    diagnostics.
    """
    return [trace for trace in traces if trace.get("status") != "inference_failure"]


def _transitions(traces: list[dict]):
    for trace in _evaluable_traces(traces):
        yield from trace.get("stage_transitions") or []


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


def first_step_answer_score(traces: list[dict]) -> dict:
    """Mean RAW judge score of each case's very first current-best answer
    (step 0, before any evidence). Pairs with final_answer_mean_score as the
    start/end of the score trajectory -- deliberately NOT bucketed by
    step_index, since reference-trajectory length varies per case and a
    fixed high step_index would average over a shrinking, non-random subset
    of (likely longer/harder) cases.
    """
    traces = _evaluable_traces(traces)
    scored = []
    for trace in traces:
        steps = trace.get("steps") or []
        if not steps:
            continue
        score = steps[0].get("current_answer_score")
        if isinstance(score, (int, float)):
            scored.append(score)
    if not scored:
        return {"mean_score": None, "n_scored": 0, "n_evaluable": len(traces)}
    mean = sum(scored) / len(scored)
    return {"mean_score": round(mean, 4), "n_scored": len(scored), "n_evaluable": len(traces)}


def mean_score_improvement(traces: list[dict]) -> dict:
    """Mean per-tool-call score delta: (final_answer_score - first_answer_score) / tools_called.

    Weighs every case equally (simple mean of per-case deltas, not pooled).
    A case is included only when it has both a first-step score and a final
    score, and made at least one tool call -- division-by-zero cases are
    excluded, not counted as zero.

    Unlike medcta_eval's version of this metric, tools_called here is in
    practice always len(reference_tool_sequence(case)): the path is forced
    to completion, so every evaluable, fully-scored case calls exactly the
    reference's tools, and this denominator is fixed per case rather than
    model-dependent. The two frameworks' mean_score_improvement numbers are
    therefore not measuring identical things -- see the README.
    """
    traces = _evaluable_traces(traces)
    deltas = []
    for trace in traces:
        steps = trace.get("steps") or []
        if not steps:
            continue
        first_score = steps[0].get("current_answer_score")
        final_score = trace.get("final_answer_score")
        tools_called = len(trace.get("model_tool_sequence") or [])
        if first_score is None or final_score is None or tools_called == 0:
            continue
        deltas.append((final_score - first_score) / tools_called)
    if not deltas:
        return {"mean_score": None, "n_scored": 0, "n_evaluable": len(traces)}
    mean = sum(deltas) / len(deltas)
    return {"mean_score": round(mean, 4), "n_scored": len(deltas), "n_evaluable": len(traces)}


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


def premature_finalization_rate(traces: list[dict]) -> dict:
    """Fraction of evaluable cases where the model attempted at least one
    early exit before the reference trajectory allowed it (case-level, same
    per-case definition medcta_eval uses for its own premature_finalization
    cohort) -- regardless of the fact that the attempt was overridden and
    the case was forced to continue.
    """
    traces = _evaluable_traces(traces)
    numerator = sum(bool(trace.get("attempted_premature_finalization")) for trace in traces)
    return _ratio(numerator, len(traces))


def premature_finalization_progress(traces: list[dict]) -> dict:
    """Mean fraction of the reference tool sequence completed before the
    model's first attempted early exit, over evaluable cases that attempted
    one. 0.0 = attempted immediately (step 0); close to 1.0 = attempted just
    short of the reference's last tool. Reads runner.py's
    premature_finalization_progress trace field directly -- unlike
    medcta_eval, model_tool_sequence always equals the *full* reference
    sequence here (the path is forced to completion), so this can't be
    derived after the fact the way medcta_eval derives its equivalent.
    """
    traces = _evaluable_traces(traces)
    rates = [
        t["premature_finalization_progress"] for t in traces
        if isinstance(t.get("premature_finalization_progress"), (int, float))
    ]
    if not rates:
        return {"mean_rate": None, "n_cases": 0}
    return {"mean_rate": round(sum(rates) / len(rates), 4), "n_cases": len(rates)}


def mean_score_before_premature_finalization(traces: list[dict]) -> dict:
    """Mean current_answer_score at each case's *first* attempted early
    exit, over evaluable cases that attempted one. This is the metric
    genuinely comparable to medcta_eval's premature_finalization_mean_score:
    both restrict to the cohort of cases that attempted a premature exit,
    and per case the two numbers should match (same judge, same answer
    text, identical prior context up to the first deviation point) --
    unlike either framework's final_answer_mean_score, which spans all
    evaluable cases including ones that never attempted an early exit.
    """
    traces = [
        t for t in _evaluable_traces(traces)
        if t.get("attempted_premature_finalization")
    ]
    scored = []
    for trace in traces:
        first_attempt = next(
            (s for s in trace.get("steps") or [] if s.get("attempted_early_exit")),
            None,
        )
        if first_attempt is None:
            continue
        score = first_attempt.get("current_answer_score")
        if isinstance(score, (int, float)):
            scored.append(score)
    if not scored:
        return {"mean_score": None, "n_scored": 0, "n_evaluable": len(traces)}
    mean = sum(scored) / len(scored)
    return {"mean_score": round(mean, 4), "n_scored": len(scored), "n_evaluable": len(traces)}


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
        bool(((step.get("model_output") or {})).get("inference_failure"))
        for trace in traces
        for step in trace.get("steps") or []
    )
    return {
        "run_id": run_id,
        "model": model,
        "backend": backend,
        "provenance": provenance,
        "forced_continuation_note": (
            "Every evaluable case here was forced through the full MedCTA "
            "reference tool sequence regardless of what the model declared; "
            "premature_finalization_rate/_progress and "
            "mean_score_before_premature_finalization measure attempted, "
            "not executed, early exits."
        ),
        "n_selected_cases": len(traces),
        "n_evaluable_cases": len(evaluable_traces),
        "trace_status_counts": dict(Counter(trace.get("status") for trace in traces)),
        "final_answer_accuracy": final_answer_accuracy(traces),
        "final_answer_mean_score": final_answer_mean_score(traces),
        "first_step_answer_score": first_step_answer_score(traces),
        "mean_score_improvement": mean_score_improvement(traces),
        "successful_revision_rate": successful_revision_rate(traces),
        "missed_revision_rate": missed_revision_rate(traces),
        "overreaction_rate": overreaction_rate(traces),
        "kept_correct_rate": kept_correct_rate(traces),
        "premature_finalization_rate": premature_finalization_rate(traces),
        "premature_finalization_progress": premature_finalization_progress(traces),
        "mean_score_before_premature_finalization": mean_score_before_premature_finalization(traces),
        "inference_failure_count": inference_failure_count,
        "inference_failure_case_count": sum(
            trace.get("status") == "inference_failure" for trace in traces
        ),
        "repair_count": sum(
            bool((step.get("model_output") or {}).get("repaired"))
            for trace in traces
            for step in trace.get("steps") or []
        ),
    }
