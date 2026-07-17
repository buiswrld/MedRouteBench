"""Action validation and answer matching for MedCTA replay."""

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


ACTIONS = ("CALL_TOOL", "FINAL_ANSWER")
REQUIRED_KEYS = {"action", "tool_name", "answer"}


@dataclass(frozen=True)
class AgentOutput:
    action: str
    tool_name: Optional[str]
    answer: Optional[str]


def safe_json_loads(value: str) -> Tuple[Optional[dict], Optional[str]]:
    """Parse one JSON object without raising."""
    try:
        parsed = json.loads(value)
    except Exception as exc:
        return None, f"json_parse: {exc}"
    if not isinstance(parsed, dict):
        return None, "root is not object"
    return parsed, None


def validate(
    raw,
    *,
    available_tools: Sequence[str],
) -> Tuple[Optional[AgentOutput], Optional[str]]:
    """Validate the exact CALL_TOOL / FINAL_ANSWER response contract."""
    if not isinstance(raw, dict):
        return None, "root is not object"
    keys = set(raw)
    if keys != REQUIRED_KEYS:
        missing = sorted(REQUIRED_KEYS - keys)
        extra = sorted(keys - REQUIRED_KEYS)
        return None, f"response keys mismatch; missing={missing}, extra={extra}"

    action = str(raw["action"]).strip().upper()
    if action not in ACTIONS:
        return None, f"invalid action '{action}'"

    tool_name = raw["tool_name"]
    if isinstance(tool_name, str):
        tool_name = tool_name.strip() or None
    elif tool_name is not None:
        return None, "tool_name must be a string or null"

    answer = raw["answer"]
    if isinstance(answer, str):
        answer = answer.strip() or None
    elif answer is not None:
        return None, "answer must be a string or null"

    if action == "CALL_TOOL":
        if tool_name not in available_tools:
            return None, f"invalid or unavailable tool_name '{tool_name}'"
        if answer is not None:
            return None, "CALL_TOOL requires answer null"
    else:
        if tool_name is not None:
            return None, "FINAL_ANSWER requires tool_name null"
        if answer is None:
            return None, "FINAL_ANSWER requires a nonempty answer"

    return AgentOutput(action=action, tool_name=tool_name, answer=answer), None


def normalize_answer(value: str) -> str:
    """Normalize a short answer for transparent whitelist exact matching."""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return re.sub(r"[\s\.,;:!?]+$", "", normalized)


def answer_matches(answer: Optional[str], accepted_answers: Sequence[str]) -> bool:
    if not isinstance(answer, str) or not answer.strip():
        return False
    candidate = normalize_answer(answer)
    return any(candidate == normalize_answer(gold) for gold in accepted_answers)
