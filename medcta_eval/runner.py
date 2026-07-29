"""Per-case controlled replay over a MedCTA reference trajectory."""

from dataclasses import asdict
from typing import Callable, Optional

from .data import reference_tool_sequence, tool_names
from .prompts import REPAIR_TEMPLATE, SYSTEM_PROMPT, build_user_prompt
from .schema import answer_matches, safe_json_loads, validate


def _default_call_json(system: str, user: str, image_url: str | None) -> str:
    from .llm import call_json

    return call_json(system, user, image_url)


def _exception_summary(exc: Exception) -> dict:
    return {"type": type(exc).__name__, "message": str(exc)}


def _response_metadata(response) -> Optional[dict]:
    metadata = getattr(response, "metadata", None)
    return metadata if isinstance(metadata, dict) else None


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
            "initial_response_metadata": None,
            "raw": None,
            "repair_response_metadata": None,
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
    initial_response_metadata = _response_metadata(initial_raw)
    parsed, validation_error = validate(
        parsed_raw,
        available_tools=available_tools,
    )
    error = json_error or validation_error
    if error is None:
        return {
            "initial_response_received": True,
            "initial_raw": initial_raw,
            "initial_response_metadata": initial_response_metadata,
            "raw": initial_raw,
            "repair_response_metadata": None,
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
            "initial_response_metadata": initial_response_metadata,
            "raw": None,
            "repair_response_metadata": None,
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
    repair_response_metadata = _response_metadata(repair_raw)
    repaired, repaired_validation_error = validate(
        repaired_raw,
        available_tools=available_tools,
    )
    repaired_error = repaired_json_error or repaired_validation_error
    return {
        "initial_response_received": True,
        "initial_raw": initial_raw,
        "initial_response_metadata": initial_response_metadata,
        "raw": repair_raw,
        "repair_response_metadata": repair_response_metadata,
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
        "final_answer_match_method": "normalized_exact_whitelist",
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
        }
        prior_actions.append(action_record)

        if expected["action"] == "CALL_TOOL":
            if actual["action"] == "FINAL_ANSWER":
                step_trace["reference_tool_match"] = False
                trace["final_answer"] = actual["answer"]
                trace["final_answer_match"] = answer_matches(
                    actual["answer"], accepted_answers
                )
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
        trace["final_answer_match"] = answer_matches(actual["answer"], accepted_answers)
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
