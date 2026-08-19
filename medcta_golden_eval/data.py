"""Loading and structural validation for the adapted MedCTA subset."""

import json
from pathlib import Path
from typing import Optional

from .config import DATA_PATH


def resolve_data_path(path=None) -> Path:
    resolved = Path(path) if path is not None else DATA_PATH
    if not resolved.is_file():
        raise FileNotFoundError(f"MedCTA adapted subset does not exist: {resolved}")
    return resolved.resolve()


def validate_dataset(payload: dict) -> None:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("MedCTA subset must use schema_version 1")
    dataset = payload.get("dataset")
    cases = payload.get("cases")
    selected_ids = payload.get("selected_case_ids")
    if not isinstance(dataset, dict) or not dataset.get("source_revision"):
        raise ValueError("MedCTA subset lacks dataset provenance")
    if not isinstance(cases, list) or not cases:
        raise ValueError("MedCTA subset must contain cases")
    if selected_ids != [case.get("case_id") for case in cases]:
        raise ValueError("selected_case_ids must align with cases")

    for case in cases:
        case_id = case.get("case_id")
        tools = case.get("available_tools")
        steps = case.get("reference_steps")
        accepted = (case.get("ground_truth") or {}).get("accepted_answers")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("every MedCTA case requires a string case_id")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"case {case_id}: missing question")
        if not isinstance(case.get("image_reference"), str):
            raise ValueError(f"case {case_id}: missing image_reference")
        if not isinstance(tools, list) or not tools:
            raise ValueError(f"case {case_id}: missing available tools")
        tool_names = [tool.get("name") for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError(f"case {case_id}: duplicate tools")
        if not isinstance(steps, list) or len(steps) < 2:
            raise ValueError(f"case {case_id}: incomplete reference trajectory")
        final_indexes = [
            index for index, step in enumerate(steps)
            if step.get("action") == "FINAL_ANSWER"
        ]
        if final_indexes != [len(steps) - 1]:
            raise ValueError(
                f"case {case_id}: requires exactly one terminal FINAL_ANSWER"
            )
        for index, step in enumerate(steps):
            if step.get("step_index") != index:
                raise ValueError(f"case {case_id}: non-contiguous step indexes")
            if step.get("action") == "CALL_TOOL":
                if step.get("tool_name") not in tool_names:
                    raise ValueError(f"case {case_id}: step uses unavailable tool")
                if not isinstance(step.get("reference_observation"), str):
                    raise ValueError(f"case {case_id}: tool step lacks observation")
            elif step.get("action") == "FINAL_ANSWER":
                if not isinstance(step.get("reference_answer"), str):
                    raise ValueError(f"case {case_id}: final step lacks answer")
            else:
                raise ValueError(f"case {case_id}: invalid reference action order")
        if not isinstance(accepted, list) or not accepted:
            raise ValueError(f"case {case_id}: missing accepted answers")


def load_dataset(path=None) -> dict:
    resolved = resolve_data_path(path)
    with open(resolved, encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_dataset(payload)
    return payload


def load_cases(path=None, limit: Optional[int] = None) -> list[dict]:
    payload = load_dataset(path)
    cases = payload["cases"]
    return cases[:limit] if limit is not None else cases


def tool_names(case: dict) -> list[str]:
    return [tool["name"] for tool in case["available_tools"]]


def reference_tool_sequence(case: dict) -> list[str]:
    return [
        step["tool_name"]
        for step in case["reference_steps"]
        if step["action"] == "CALL_TOOL"
    ]
