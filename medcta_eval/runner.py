"""Per-case controlled replay over a MedCTA reference trajectory."""

from dataclasses import asdict
from typing import Callable, Optional

from .data import reference_tool_sequence, tool_names
from .prompts import REPAIR_TEMPLATE, SYSTEM_PROMPT, build_user_prompt
from .config import (
    ANSWER_EQUIVALENCE_CONFIDENCE_THRESHOLD,
    FINAL_ACCURACY_CONFIDENCE_THRESHOLD,
)
from .schema import safe_json_loads, validate


def _default_call_json(system: str, user: str, image_url: str | None) -> str:
    from .llm import call_json

    return call_json(system, user, image_url)


def _judge_answer_correctness(
    answer: str | None,
    accepted_answers: list[str],
) -> float | None:
    """Score `answer` against every accepted gold answer via one merged judge.

    Uses the single llm.judge_answer function/prompt for every step,
    whether `answer` is a genuine FINAL_ANSWER or an intermediate
    current-best hypothesis, so identical answer text always scores
    identically regardless of which stage it came from — stage-transition
    labels can only change because the answer actually changed, not
    because of a judge framing difference across the FINAL_ANSWER
    boundary. All accepted answers are passed through (matching any one is
    sufficient), not just the first.
    """
    from .llm import judge_answer

    if not accepted_answers or not answer:
        return None
    return judge_answer(accepted_answers, answer)


def _normalize_answer_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def _judge_answer_correctness_cached(
    cache: dict[str, float | None],
    answer: str | None,
    accepted_answers: list[str],
) -> float | None:
    """Memoized wrapper around _judge_answer_correctness.

    Keyed by normalized answer text alone — scoped to one case's cache
    dict, so keying without the gold answer is safe within a single run.
    Since every step uses the same judge function/prompt regardless of
    whether it's an intermediate current-best hypothesis or the genuine
    FINAL_ANSWER, identical answer text always shares one cache entry and
    one score, anywhere in the trajectory.
    """
    if not answer:
        return _judge_answer_correctness(answer, accepted_answers)
    key = _normalize_answer_text(answer)
    if key in cache:
        return cache[key]
    score = _judge_answer_correctness(answer, accepted_answers)
    cache[key] = score
    return score


def _judge_answer_equivalence(previous: str | None, current: str | None) -> float | None:
    """Score whether two consecutive current-best answers mean the same thing.

    Identical text (after whitespace/case normalization) is trivially
    unchanged and skips the judge call entirely to save cost. Otherwise
    uses judge_equivalence, a symmetric same-conclusion check — not
    judge_answer, which grades correctness against a gold answer and is
    asymmetrically biased toward scoring refinements as unchanged.
    """
    from .llm import judge_equivalence

    if not previous or not current:
        return None
    if _normalize_answer_text(previous) == _normalize_answer_text(current):
        return 1.0
    return judge_equivalence(previous, current)


def _exception_summary(exc: Exception) -> dict:
    return {"type": type(exc).__name__, "message": str(exc)}


def _call_stage(
    call_fn: Callable,
    system_prompt: str,
    user_prompt: str,
    image_url: str | None,
    available_tools: list[str],
    repair_template: str,
) -> dict:
    """Call once, repair once when invalid, and retain full provenance."""
    try:
        initial_raw = call_fn(system_prompt, user_prompt, image_url)
    except Exception as exc:
        return {
            "initial_response_received": False,
            "initial_raw": None,
            "raw": None,
            "parsed": None,
            "valid": False,
            "repaired": False,
            "initial_invalid": False,
            "validation_error": None,
            "repair_prompt": None,
            "inference_failure": True,
            "inference_error": _exception_summary(exc),
        }

    parsed_raw, json_error = safe_json_loads(initial_raw)
    parsed, validation_error = validate(
        parsed_raw,
        available_tools=available_tools,
    )
    error = json_error or validation_error
    if error is None:
        return {
            "initial_response_received": True,
            "initial_raw": initial_raw,
            "raw": initial_raw,
            "parsed": asdict(parsed),
            "valid": True,
            "repaired": False,
            "initial_invalid": False,
            "validation_error": None,
            "repair_prompt": None,
            "inference_failure": False,
            "inference_error": None,
        }

    repair_prompt = repair_template.format(
        ERROR=error,
        RESPONSE=initial_raw,
        ORIGINAL=user_prompt,
    )
    try:
        repair_raw = call_fn(system_prompt, repair_prompt, image_url)
    except Exception as exc:
        return {
            "initial_response_received": True,
            "initial_raw": initial_raw,
            "raw": None,
            "parsed": None,
            "valid": False,
            "repaired": True,
            "initial_invalid": True,
            "validation_error": error,
            "repair_prompt": repair_prompt,
            "inference_failure": True,
            "inference_error": _exception_summary(exc),
        }

    repaired_raw, repaired_json_error = safe_json_loads(repair_raw)
    repaired, repaired_validation_error = validate(
        repaired_raw,
        available_tools=available_tools,
    )
    repaired_error = repaired_json_error or repaired_validation_error
    return {
        "initial_response_received": True,
        "initial_raw": initial_raw,
        "raw": repair_raw,
        "parsed": asdict(repaired) if repaired_error is None else None,
        "valid": repaired_error is None,
        "repaired": True,
        "initial_invalid": True,
        "validation_error": repaired_error,
        "repair_prompt": repair_prompt,
        "inference_failure": False,
        "inference_error": None,
    }


def run_case(
    case: dict,
    *,
    call_fn: Optional[Callable] = None,
    system_prompt: Optional[str] = None,
    repair_template: Optional[str] = None,
) -> dict:
    """Run one case using controlled reference-observation replay."""
    call = call_fn if call_fn is not None else _default_call_json
    system = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    repair = repair_template if repair_template is not None else REPAIR_TEMPLATE
    available_tool_names = tool_names(case)
    reference_sequence = reference_tool_sequence(case)
    accepted_answers = case["ground_truth"]["accepted_answers"]

    prior_actions: list[dict] = []
    prior_observations: list[dict] = []
    model_tool_sequence: list[str] = []
    step_traces: list[dict] = []
    answer_score_cache: dict[str, float | None] = {}

    trace = {
        "case_id": case["case_id"],
        "question": case["question"],
        "image_reference": case.get("image_reference"),
        "available_tools": case["available_tools"],
        "accepted_ground_truth_answers": accepted_answers,
        "reference_tool_sequence": reference_sequence,
        "reference_step_count": len(case["reference_steps"]),
        "model_tool_sequence": model_tool_sequence,
        "model_actions": prior_actions,
        "steps": step_traces,
        "status": "pending",
        "termination_reason": None,
        "completed_reference_finalization": False,
        "trajectory_exact_match": False,
        "final_answer": None,
        "final_answer_match": False,
        "final_answer_match_method": f"llm_judge_threshold_{FINAL_ACCURACY_CONFIDENCE_THRESHOLD}",
        "final_answer_score": None,
    }

    for expected in case["reference_steps"]:
        user_prompt = build_user_prompt(case, prior_actions, prior_observations)
        result = _call_stage(
            call,
            system,
            user_prompt,
            case.get("image_reference"),
            available_tool_names,
            repair,
        )
        step_trace = {
            "step_index": expected["step_index"],
            "expected_action": expected["action"],
            "expected_tool_name": expected.get("tool_name"),
            "system_prompt": system,
            "user_prompt": user_prompt,
            "image_reference": case.get("image_reference"),
            "model_output": result,
            "action_match": False,
            "reference_tool_match": None,
            "reference_observation_revealed": None,
            "current_answer": None,
            "reasoning": None,
            "current_answer_score": None,
            "current_answer_correct": False,
            "evidence_shown": list(prior_observations),
        }
        step_traces.append(step_trace)

        if result["inference_failure"]:
            trace["status"] = "inference_failure"
            trace["termination_reason"] = "inference failed after retries"
            break
        if not result["valid"]:
            trace["status"] = "invalid_action"
            trace["termination_reason"] = "action remained invalid after one repair"
            break

        actual = result["parsed"]
        action_record = {
            "step_index": expected["step_index"],
            "action": actual["action"],
            "tool_name": actual["tool_name"],
            "answer": actual["answer"],
            "reasoning": actual["reasoning"],
        }
        prior_actions.append(action_record)

        # current_answer_score: raw 0-1 judge score (or None on judge failure).
        # current_answer_correct: that score thresholded at
        # FINAL_ACCURACY_CONFIDENCE_THRESHOLD; None defaults to False (not
        # "unknown") so a failed judge call can't silently count as correct.
        # Every step's answer -- CALL_TOOL or FINAL_ANSWER alike -- is graded
        # by the same judge_answer function/prompt, so identical answer text
        # always scores identically regardless of which stage it came from.
        score = _judge_answer_correctness_cached(
            answer_score_cache,
            actual["answer"],
            accepted_answers,
        )
        correct = (
            score >= FINAL_ACCURACY_CONFIDENCE_THRESHOLD if score is not None else False
        )
        step_trace["current_answer"] = actual["answer"]
        step_trace["reasoning"] = actual["reasoning"]
        step_trace["current_answer_score"] = score
        step_trace["current_answer_correct"] = correct

        if expected["action"] == "CALL_TOOL":
            if actual["action"] == "FINAL_ANSWER":
                step_trace["reference_tool_match"] = False
                trace["final_answer"] = actual["answer"]
                trace["final_answer_score"] = score
                trace["final_answer_match"] = correct
                trace["status"] = "premature_finalization"
                trace["termination_reason"] = "model finalized before reference tools ended"
                break

            model_tool_sequence.append(actual["tool_name"])
            matched = actual["tool_name"] == expected["tool_name"]
            step_trace["action_match"] = matched
            step_trace["reference_tool_match"] = matched
            revealed = {
                "step_index": expected["step_index"],
                "reference_tool_name": expected["tool_name"],
                "content": expected["reference_observation"],
            }
            prior_observations.append(revealed)
            step_trace["reference_observation_revealed"] = revealed
            continue

        if actual["action"] == "CALL_TOOL":
            model_tool_sequence.append(actual["tool_name"])
            trace["status"] = "missed_finalization"
            trace["termination_reason"] = "model called a tool at reference finalization"
            break

        step_trace["action_match"] = True
        trace["final_answer"] = actual["answer"]
        trace["final_answer_score"] = score
        trace["final_answer_match"] = correct
        trace["completed_reference_finalization"] = True
        trace["status"] = "completed"
        trace["termination_reason"] = "model finalized at the reference final step"
        break

    if trace["status"] == "pending":
        trace["status"] = "invalid_reference_trajectory"
        trace["termination_reason"] = "reference trajectory ended without finalization"

    trace["trajectory_exact_match"] = (
        model_tool_sequence == reference_sequence
        and trace["completed_reference_finalization"]
    )

    scored_steps = [s for s in step_traces if s.get("current_answer") is not None]
    stage_transitions = []
    for prev, nxt in zip(scored_steps, scored_steps[1:]):
        prev_correct = bool(prev["current_answer_correct"])
        new_correct = bool(nxt["current_answer_correct"])
        if not prev_correct and new_correct:
            label = "successful_revision"
        elif not prev_correct and not new_correct:
            label = "missed_revision"
        elif prev_correct and not new_correct:
            label = "overreaction"
        else:
            label = "kept_correct"
        # answer_equivalence_score: raw 0-1 "same conclusion?" judge score (or
        # None on judge failure). answer_changed: that score thresholded at
        # ANSWER_EQUIVALENCE_CONFIDENCE_THRESHOLD, but inverted-default from
        # current_answer_correct above: None defaults to answer_changed=True,
        # since assuming a change we can't verify is the conservative failure
        # mode for answer_stability (vs. defaulting to False and silently
        # claiming stability).
        equivalence_score = _judge_answer_equivalence(
            prev["current_answer"], nxt["current_answer"]
        )
        answer_changed = not (
            equivalence_score is not None
            and equivalence_score >= ANSWER_EQUIVALENCE_CONFIDENCE_THRESHOLD
        )
        stage_transitions.append(
            {
                "prev_step_index": prev["step_index"],
                "next_step_index": nxt["step_index"],
                "prev_current_answer": prev["current_answer"],
                "new_current_answer": nxt["current_answer"],
                "prev_correct": prev_correct,
                "new_correct": new_correct,
                "label": label,
                "answer_equivalence_score": equivalence_score,
                "answer_changed": answer_changed,
                "maintained_wrong": label == "missed_revision" and not answer_changed,
            }
        )
    trace["stage_transitions"] = stage_transitions
    trace["answer_stable_throughout"] = (
        None
        if not stage_transitions
        else all(not t["answer_changed"] for t in stage_transitions)
    )

    trace["matched_reference_steps"] = sum(
        bool(step["action_match"]) for step in step_traces
    )
    trace["attempted_steps"] = len(step_traces)
    trace["initial_invalid_steps"] = sum(
        bool(step["model_output"]["initial_invalid"]) for step in step_traces
    )
    trace["inference_failure_steps"] = sum(
        bool(step["model_output"]["inference_failure"]) for step in step_traces
    )
    return trace
