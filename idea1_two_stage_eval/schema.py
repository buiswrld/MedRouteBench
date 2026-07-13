"""
Action ontology, output schema, and JSON validator.
"""
import json
from dataclasses import dataclass
from typing import Optional, Tuple

# ── ontology ─────────────────────────────────────────────────────────────────
ACTIONS = ["ANSWER", "KEEP_ANSWER", "REVISE_ANSWER", "ABSTAIN"]
STAGE1_ALLOWED = {"ANSWER"}
STAGE2_ALLOWED = {"KEEP_ANSWER", "REVISE_ANSWER", "ABSTAIN"}

VALID_ANSWERS = {"yes", "no", "maybe", None}


# ── output schema ─────────────────────────────────────────────────────────────
@dataclass
class AgentOutput:
    action: str
    answer: Optional[str]
    confidence: float
    reason_for_action: str


# ── helpers ──────────────────────────────────────────────────────────────────
def safe_json_loads(s: str) -> Tuple[Optional[dict], Optional[str]]:
    """Parse JSON string; return (obj, None) or (None, error_str). Never raises."""
    try:
        return json.loads(s), None
    except Exception as exc:
        return None, f"json_parse: {exc}"


def validate(
    raw,
    *,
    stage: int,
    prior_answer: Optional[str] = None,
) -> Tuple[Optional[AgentOutput], Optional[str]]:
    """Return (AgentOutput, None) on success or (None, error_str) on failure. Never raises."""
    if not isinstance(raw, dict):
        return None, "root is not object"
    required = ["action", "answer", "confidence", "reason_for_action"]
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

    if stage == 1:
        if action not in STAGE1_ALLOWED:
            return None, f"stage 1 requires action in {sorted(STAGE1_ALLOWED)}"
        if ans not in {"yes", "no", "maybe"}:
            return None, "stage 1 answer must be yes/no/maybe"
    elif stage == 2:
        if action not in STAGE2_ALLOWED:
            return None, f"stage 2 requires action in {sorted(STAGE2_ALLOWED)}"
        if prior_answer not in {"yes", "no", "maybe"}:
            return None, "stage 2 requires a valid stage 1 answer"
        if action == "ABSTAIN":
            if ans is not None:
                return None, "ABSTAIN requires answer=null"
        else:
            if ans not in {"yes", "no", "maybe"}:
                return None, "non-abstain stage 2 answer must be yes/no/maybe"
            if action == "KEEP_ANSWER" and ans != prior_answer:
                return None, "KEEP_ANSWER must repeat the stage 1 answer"
            if action == "REVISE_ANSWER" and ans == prior_answer:
                return None, "REVISE_ANSWER must differ from the stage 1 answer"
    else:
        return None, f"unsupported stage '{stage}'"

    return AgentOutput(action, ans, conf, reason.strip()), None
