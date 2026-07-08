"""
Per-case fixed two-stage revision runner.

`run_case` accepts an injectable `call_fn` so it can be driven by a stub in
tests without touching the network.
"""
from dataclasses import asdict
from typing import Callable, Optional

from .schema import AgentOutput, safe_json_loads, validate
from .data import split_revision_evidence
from .prompts import (
    STAGE1_SYSTEM_PROMPT as _DEFAULT_STAGE1_SYSTEM_PROMPT,
    STAGE2_SYSTEM_PROMPT as _DEFAULT_STAGE2_SYSTEM_PROMPT,
    REPAIR_TEMPLATE as _DEFAULT_REPAIR_TEMPLATE,
    build_stage1_user_prompt,
    build_stage2_user_prompt,
)
from .llm import call_json as _default_call_json


def _invalid_output(error: str) -> dict:
    return {
        "action": "INVALID",
        "answer": None,
        "confidence": 0.0,
        "reason_for_action": f"parse_error: {error}",
    }


def _call_and_validate(
    *,
    stage: int,
    call_fn: Callable,
    system_prompt: str,
    user_prompt: str,
    repair_template: str,
    prior_answer: Optional[str] = None,
) -> dict:
    raw = call_fn(system_prompt, user_prompt)
    parsed_raw, _ = safe_json_loads(raw)
    parsed, err = validate(parsed_raw, stage=stage, prior_answer=prior_answer)
    parse_error = False

    if err:
        raw2 = call_fn(system_prompt, repair_template.format(ERROR=err, ORIGINAL=user_prompt))
        parsed_raw2, _ = safe_json_loads(raw2)
        parsed, err2 = validate(parsed_raw2, stage=stage, prior_answer=prior_answer)
        raw = raw2
        err = err2
        if err2:
            parse_error = True

    return {
        "raw": raw,
        "parsed": asdict(parsed) if isinstance(parsed, AgentOutput) else _invalid_output(err or "unknown"),
        "is_valid": not parse_error and isinstance(parsed, AgentOutput),
        "validation_error": err,
    }


def _classify_case(gold_label: Optional[str], stage1_answer: str, stage2_output: dict) -> tuple[Optional[str], Optional[str]]:
    action = stage2_output["action"]
    final_answer = None if action == "ABSTAIN" else stage2_output["answer"]

    if action == "ABSTAIN":
        return final_answer, "abstention"
    if stage1_answer == gold_label:
        if final_answer == stage1_answer:
            return final_answer, "kept_correct"
        return final_answer, "overreaction"
    if final_answer == gold_label and action == "REVISE_ANSWER":
        return final_answer, "successful_revision"
    return final_answer, "missed_revision"


def run_case(
    case: dict,
    gt_label: Optional[str],
    *,
    call_fn: Optional[Callable] = None,
    stage1_system_prompt: Optional[str] = None,
    stage2_system_prompt: Optional[str] = None,
    repair_template: Optional[str] = None,
) -> dict:
    """
    Run one case through the fixed two-stage experiment and return a trace dict.

    Parameters
    ----------
    case         : Prepared PubMedQA case dict or a raw case that can be split.
    gt_label     : Ground-truth label ("yes" / "no" / "maybe" / None).
    call_fn      : LLM call function ``(system: str, user: str) -> str``.
                   Defaults to ``llm.call_json``.
    stage1_system_prompt: Override the Stage 1 system prompt.
    stage2_system_prompt: Override the Stage 2 system prompt.
    repair_template: Override the default REPAIR_TEMPLATE.
    """
    _call = call_fn if call_fn is not None else _default_call_json
    _stage1_system = (
        stage1_system_prompt
        if stage1_system_prompt is not None
        else _DEFAULT_STAGE1_SYSTEM_PROMPT
    )
    _stage2_system = (
        stage2_system_prompt
        if stage2_system_prompt is not None
        else _DEFAULT_STAGE2_SYSTEM_PROMPT
    )
    _repair = repair_template if repair_template is not None else _DEFAULT_REPAIR_TEMPLATE

    prepared = case if "stage1_evidence" in case else split_revision_evidence(case)
    trace = {
        "pmid": case["pmid"],
        "gold_label": gt_label,
        "question": case["QUESTION"],
        "split_strategy": prepared["split_strategy"] if prepared else None,
        "completed": False,
        "stage1_evidence_shown": [],
        "stage1_model_output": None,
        "stage1_raw_output": None,
        "stage1_valid": False,
        "stage1_validation_error": None,
        "stage2_added_evidence": [],
        "stage2_full_context_shown": [],
        "stage2_model_output": None,
        "stage2_raw_output": None,
        "stage2_valid": False,
        "stage2_validation_error": None,
        "final_answer": None,
        "label": None,
    }

    if prepared is None:
        trace["stage1_validation_error"] = "unable to create a non-empty two-stage evidence split"
        return trace

    trace["stage1_evidence_shown"] = [
        {"label": label, "context": context}
        for label, context in prepared["stage1_evidence"]
    ]
    trace["stage2_added_evidence"] = [
        {"label": label, "context": context}
        for label, context in prepared["stage2_added_evidence"]
    ]
    trace["stage2_full_context_shown"] = [
        {"label": label, "context": context}
        for label, context in prepared["stage2_full_context"]
    ]

    stage1_prompt = build_stage1_user_prompt(prepared)
    stage1_result = _call_and_validate(
        stage=1,
        call_fn=_call,
        system_prompt=_stage1_system,
        user_prompt=stage1_prompt,
        repair_template=_repair,
    )
    trace["stage1_model_output"] = stage1_result["parsed"]
    trace["stage1_raw_output"] = stage1_result["raw"]
    trace["stage1_valid"] = stage1_result["is_valid"]
    trace["stage1_validation_error"] = stage1_result["validation_error"]

    if not stage1_result["is_valid"]:
        return trace

    stage1_output = stage1_result["parsed"]
    stage2_prompt = build_stage2_user_prompt(prepared, stage1_output)
    stage2_result = _call_and_validate(
        stage=2,
        call_fn=_call,
        system_prompt=_stage2_system,
        user_prompt=stage2_prompt,
        repair_template=_repair,
        prior_answer=stage1_output["answer"],
    )
    trace["stage2_model_output"] = stage2_result["parsed"]
    trace["stage2_raw_output"] = stage2_result["raw"]
    trace["stage2_valid"] = stage2_result["is_valid"]
    trace["stage2_validation_error"] = stage2_result["validation_error"]

    if not stage2_result["is_valid"]:
        return trace

    final_answer, case_label = _classify_case(gt_label, stage1_output["answer"], stage2_result["parsed"])
    trace["completed"] = True
    trace["final_answer"] = final_answer
    trace["label"] = case_label

    return trace
