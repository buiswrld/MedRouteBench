"""
System prompt, per-stage user prompt builder, and repair template.
"""
import json
from typing import Optional

from .data import revealed_pairs


SYSTEM_PROMPT = """You are a careful biomedical reasoning agent answering a PubMedQA question in stages. New CONTEXTS are revealed one LABEL at a time.

At every stage you MUST respond with a SINGLE JSON object and NOTHING else — no
prose, no markdown, no code fences.

Required keys:
  "action": one of ["FOLLOW_UP","REVISE_ANSWER","ANSWER","ABSTAIN"]
  "answer": "yes" | "no" | "maybe" | null
  "confidence": float in [0.0, 1.0]
  "reason_for_action": short string (<= 240 chars)
  "needed_information": short string or null

Action semantics:
- FOLLOW_UP: you need the next case LABEL revealed
- REVISE_ANSWER: change your prior answer based on new information
- ANSWER: give an answer
- ABSTAIN: refuse when evidence is genuinely insufficient

Stage-gated action rules (STRICT):
- On any NON-FINAL stage, FOLLOW_UP, REVISE_ANSWER, and ANSWER are allowed. If you choose FOLLOW_UP, you must leave "answer" as null. Use REVISE_ANSWER only when your prior stage's answer differs from your new answer; otherwise use ANSWER. ABSTAIN is FORBIDDEN until the final stage.
- On the FINAL stage (all case LABELs revealed) you MUST choose exactly one of
  {ANSWER, REVISE_ANSWER, ABSTAIN}. If you choose ANSWER or REVISE_ANSWER, the "answer" key MUST be filled with "yes", "no", or "maybe". If you choose ABSTAIN, you must leave the key as null. FOLLOW_UP is FORBIDDEN on the final stage.
"""


def build_user_prompt(
    case: dict,
    stage: int,
    total_stages: int,
    prior_output: Optional[dict],
) -> str:
    is_final = stage == total_stages - 1
    lines = [
        f"STAGE {stage} of {total_stages - 1} "
        f"(0 = question only, {total_stages - 1} = all LABELs revealed).",
        f"QUESTION:\n{case['QUESTION']}",
    ]
    pairs = revealed_pairs(case, stage)
    if pairs:
        lines.append("REVEALED SO FAR:")
        for lbl, ctx in pairs:
            lines.append(f"LABEL {lbl}: {ctx}")
    else:
        lines.append("REVEALED SO FAR: (none — question only)")
    if prior_output:
        lines.append(f"YOUR PRIOR OUTPUT: {json.dumps(prior_output)}")
    if is_final:
        lines.append(
            "THIS IS THE FINAL STAGE. You MUST choose action ∈ "
            "{ANSWER, REVISE_ANSWER, ABSTAIN}. If not ABSTAIN, `answer` MUST be "
            '"yes", "no", or "maybe" — never null. FOLLOW_UP is FORBIDDEN now.'
        )
    lines.append("Respond with the required JSON object only.")
    return "\n\n".join(lines)


REPAIR_TEMPLATE = (
    "Your previous response failed schema validation with error: {ERROR}. "
    "Return a corrected JSON object with the required keys. No prose.\n\n"
    "Original prompt:\n{ORIGINAL}"
)
