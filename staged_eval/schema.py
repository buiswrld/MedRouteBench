"""
Action ontology, output schema, and JSON validator.
"""
import json
from dataclasses import dataclass
from typing import Optional, Tuple

# ── ontology ─────────────────────────────────────────────────────────────────
ACTIONS = ["FOLLOW_UP", "REVISE_ANSWER", "ANSWER", "ABSTAIN"]

# Permissive action oracle used by metrics:
NONFINAL_ALLOWED = {"FOLLOW_UP", "REVISE_ANSWER", "ANSWER"}  # ABSTAIN forbidden non-final
FINAL_ALLOWED    = {"REVISE_ANSWER", "ANSWER", "ABSTAIN"}     # FOLLOW_UP forbidden final

VALID_ANSWERS = {"yes", "no", "maybe", None}


# ── output schema ─────────────────────────────────────────────────────────────
@dataclass
class AgentOutput:
    action: str
    answer: Optional[str]
    confidence: float
    reason_for_action: str
    needed_information: Optional[str]


# ── helpers ──────────────────────────────────────────────────────────────────
def safe_json_loads(s: str) -> Tuple[Optional[dict], Optional[str]]:
    """Parse JSON string; return (obj, None) or (None, error_str). Never raises."""
    try:
        return json.loads(s), None
    except Exception as exc:
        return None, f"json_parse: {exc}"


def validate(raw) -> Tuple[Optional[AgentOutput], Optional[str]]:
    """Return (AgentOutput, None) on success or (None, error_str) on failure. Never raises."""
    if not isinstance(raw, dict):
        return None, "root is not object"
    required = ["action", "answer", "confidence", "reason_for_action", "needed_information"]
    for k in required:
        if k not in raw:
            return None, f"missing key '{k}'"
    action = str(raw["action"]).strip().upper()
    if action not in ACTIONS:
        return None, f"invalid action '{action}'"
    ans = raw["answer"]
    if isinstance(ans, str):
        ans = ans.strip().lower() or None
    if ans not in VALID_ANSWERS:
        return None, f"invalid answer '{ans}'"
    try:
        conf = float(raw["confidence"])
    except Exception:
        return None, "confidence not a number"
    conf = max(0.0, min(1.0, conf))
    reason = raw["reason_for_action"]
    if not isinstance(reason, str) or not reason.strip():
        return None, "empty reason_for_action"
    need = raw["needed_information"]
    if need is not None and not isinstance(need, str):
        return None, "needed_information not str/null"
    return AgentOutput(action, ans, conf, reason.strip(), need), None
