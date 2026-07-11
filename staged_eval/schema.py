"""Strict action and answer validation for the fixed two-stage experiment."""

import json
from dataclasses import dataclass
from typing import Optional, Tuple


ACTIONS = ["ANSWER", "KEEP_ANSWER", "REVISE_ANSWER", "ABSTAIN"]
STAGE1_ALLOWED = {"ANSWER"}
STAGE2_ALLOWED = {"KEEP_ANSWER", "REVISE_ANSWER", "ABSTAIN"}
VALID_ANSWERS = {"yes", "no", "maybe"}


@dataclass
class AgentOutput:
    action: str
    answer: Optional[str]


def safe_json_loads(value: str) -> Tuple[Optional[dict], Optional[str]]:
    """Parse a JSON string without raising."""
    try:
        return json.loads(value), None
    except Exception as exc:
        return None, f"json_parse: {exc}"


def validate(
    raw,
    *,
    stage: int,
    prior_answer: Optional[str] = None,
) -> Tuple[Optional[AgentOutput], Optional[str]]:
    """Validate one output, including all action/answer consistency rules."""
    if not isinstance(raw, dict):
        return None, "root is not object"
    for key in ("action", "answer"):
        if key not in raw:
            return None, f"missing key '{key}'"

    action = str(raw["action"]).strip().upper()
    if action not in ACTIONS:
        return None, f"invalid action '{action}'"

    answer = raw["answer"]
    if isinstance(answer, str):
        answer = answer.strip().lower() or None
    if answer is not None and answer not in VALID_ANSWERS:
        return None, f"invalid answer '{answer}'"

    if stage == 1:
        if action != "ANSWER":
            return None, "Stage 1 action must be ANSWER"
        if answer is None:
            return None, "Stage 1 ANSWER requires yes, no, or maybe"
    elif stage == 2:
        if prior_answer not in VALID_ANSWERS:
            return None, "Stage 2 requires a valid Stage 1 prior answer"
        if action not in STAGE2_ALLOWED:
            return None, "Stage 2 action must be KEEP_ANSWER, REVISE_ANSWER, or ABSTAIN"
        if action == "KEEP_ANSWER" and answer != prior_answer:
            return None, "KEEP_ANSWER requires answer to match the Stage 1 answer"
        if action == "REVISE_ANSWER":
            if answer is None:
                return None, "REVISE_ANSWER requires yes, no, or maybe"
            if answer == prior_answer:
                return None, "REVISE_ANSWER requires an answer different from Stage 1"
        if action == "ABSTAIN" and answer is not None:
            return None, "ABSTAIN requires answer null"
    else:
        return None, f"invalid stage '{stage}'"

    return AgentOutput(action=action, answer=answer), None
