"""Per-case runner for the fixed two-stage revision experiment."""

from dataclasses import asdict
from typing import Callable, Optional

from .data import FINAL_ANSWERS, split_evidence
from .prompts import (
    REPAIR_TEMPLATE as _DEFAULT_REPAIR_TEMPLATE,
    SYSTEM_PROMPT as _DEFAULT_SYSTEM_PROMPT,
    build_user_prompt,
)
from .schema import safe_json_loads, validate


def _default_call_json(system: str, user: str) -> str:
    """Load the built-in provider only for direct run_case calls without a backend."""
    from .llm import call_json

    return call_json(system, user)


def _call_stage(
    call_fn: Callable,
    system_prompt: str,
    user_prompt: str,
    repair_template: str,
    *,
    stage: int,
    prior_answer: Optional[str] = None,
) -> dict:
    """Call once, repair once if needed, and preserve validation provenance."""
    initial_raw = call_fn(system_prompt, user_prompt)
    parsed_raw, json_error = safe_json_loads(initial_raw)
    parsed, validation_error = validate(
        parsed_raw,
        stage=stage,
        prior_answer=prior_answer,
    )
    error = json_error or validation_error
    if error is None:
        return {
            "raw": initial_raw,
            "parsed": asdict(parsed),
            "valid": True,
            "repaired": False,
            "validation_error": None,
        }

    repair_prompt = repair_template.format(
        ERROR=error,
        RESPONSE=initial_raw,
        ORIGINAL=user_prompt,
    )
    repair_raw = call_fn(system_prompt, repair_prompt)
    repaired_raw, repaired_json_error = safe_json_loads(repair_raw)
    repaired, repaired_validation_error = validate(
        repaired_raw,
        stage=stage,
        prior_answer=prior_answer,
    )
    repaired_error = repaired_json_error or repaired_validation_error
    return {
        "initial_raw": initial_raw,
        "raw": repair_raw,
        "parsed": asdict(repaired) if repaired_error is None else None,
        "valid": repaired_error is None,
        "repaired": True,
        "validation_error": repaired_error,
    }


def _classify(stage1_answer: str, stage2_output: dict, gold_label: str) -> str:
    action = stage2_output["action"]
    final_answer = stage2_output["answer"]
    if action == "ABSTAIN":
        return "abstention"
    if stage1_answer != gold_label:
        return "successful_revision" if final_answer == gold_label else "missed_revision"
    return "kept_correct" if final_answer == gold_label else "overreaction"


def run_case(
    case: dict,
    gt_label: Optional[str],
    *,
    call_fn: Optional[Callable] = None,
    system_prompt: Optional[str] = None,
    repair_template: Optional[str] = None,
) -> dict:
    """Run exactly two stages and return the requested per-case trace."""
    call = call_fn if call_fn is not None else _default_call_json
    system = system_prompt if system_prompt is not None else _DEFAULT_SYSTEM_PROMPT
    repair = repair_template if repair_template is not None else _DEFAULT_REPAIR_TEMPLATE
    evidence_split = split_evidence(case)

    trace = {
        "pmid": str(case["pmid"]),
        "pubmedqa_gold_label": gt_label,
        "split_strategy": evidence_split["strategy"] if evidence_split else None,
        "stage1_evidence_shown": evidence_split["stage1_evidence"] if evidence_split else [],
        "stage1_model_output": None,
        "stage2_added_evidence": evidence_split["stage2_added_evidence"] if evidence_split else [],
        "stage2_full_context_shown": evidence_split["full_context"] if evidence_split else [],
        "stage2_model_output": None,
        "status": "pending",
        "label": None,
    }

    if gt_label not in FINAL_ANSWERS:
        trace.update(status="skipped_missing_gold", label="skipped")
        return trace
    if evidence_split is None:
        trace.update(status="skipped_unsplittable", label="skipped")
        return trace

    stage1_prompt = build_user_prompt(case, 1, evidence_split)
    stage1_result = _call_stage(
        call,
        system,
        stage1_prompt,
        repair,
        stage=1,
    )
    trace["stage1_model_output"] = stage1_result
    if not stage1_result["valid"]:
        trace.update(status="invalid_stage1", label="invalid")
        return trace

    stage1_parsed = stage1_result["parsed"]
    stage2_prompt = build_user_prompt(case, 2, evidence_split, stage1_parsed)
    stage2_result = _call_stage(
        call,
        system,
        stage2_prompt,
        repair,
        stage=2,
        prior_answer=stage1_parsed["answer"],
    )
    trace["stage2_model_output"] = stage2_result
    if not stage2_result["valid"]:
        trace.update(status="invalid_stage2", label="invalid")
        return trace

    trace.update(
        status="completed",
        label=_classify(stage1_parsed["answer"], stage2_result["parsed"], gt_label),
    )
    return trace
