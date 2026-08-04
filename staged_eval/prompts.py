"""Prompts for the fixed PubMedQA commit-then-revise experiment."""

import json
from typing import Optional


STAGE1_SYSTEM_PROMPT = """You are a careful biomedical reasoning agent in a fixed two-stage PubMedQA experiment.

Use only the question and labelled evidence supplied in the user message. Never
claim access to a gold answer, long answer, hidden decision, or external source.

Return one JSON object and nothing else. The object must contain exactly:
  "action": "ANSWER"
  "answer": "yes", "no", or "maybe"
  "confidence": float in [0.0, 1.0]
  "reason_for_action": short string (<= 240 chars)

Stage 1 rules:
- action MUST be "ANSWER"
- answer MUST be "yes", "no", or "maybe"
"""


STAGE2_SYSTEM_PROMPT = """You are a careful biomedical reasoning agent in the revision stage of a fixed two-stage PubMedQA experiment.

Use only the question, your shown Stage 1 output, and the labelled evidence
supplied in the user message. Never claim access to a gold answer, long answer,
hidden decision, or external source.

Return one JSON object and nothing else. The object must contain exactly:
  "action": "KEEP_ANSWER", "REVISE_ANSWER", or "ABSTAIN"
  "answer": "yes", "no", "maybe", or null
  "confidence": float in [0.0, 1.0]
  "reason_for_action": short string (<= 240 chars)

Stage 2 rules:
- "KEEP_ANSWER": answer MUST exactly match the Stage 1 answer
- "REVISE_ANSWER": answer MUST be yes/no/maybe and differ from Stage 1
- "ABSTAIN": answer MUST be null
"""


def _format_evidence(items: list) -> str:
    return "\n\n".join(
        f"LABEL: {item['label']}\nCONTEXT: {item['context']}" for item in items
    )


def build_user_prompt(
    case: dict,
    stage: int,
    evidence_split: dict,
    prior_output: Optional[dict] = None,
) -> str:
    """Build a prompt from QUESTION, CONTEXTS/LABELS, and (at Stage 2) prior output."""
    question = str(case["QUESTION"]).strip()
    if stage == 1:
        return "\n\n".join(
            [
                "STAGE 1 - PRELIMINARY COMMITMENT",
                f"QUESTION:\n{question}",
                "PRELIMINARY EVIDENCE:\n"
                + _format_evidence(evidence_split["stage1_evidence"]),
                'Return exactly {"action":"ANSWER","answer":"yes|no|maybe",'
                '"confidence":0.0-1.0,"reason_for_action":"..."} '
                "with one concrete answer value.",
            ]
        )

    if stage == 2:
        if prior_output is None:
            raise ValueError("Stage 2 prompt requires the valid Stage 1 output")
        return "\n\n".join(
            [
                "STAGE 2 - FINAL REVISION DECISION",
                f"QUESTION:\n{question}",
                "YOUR STAGE 1 OUTPUT:\n" + json.dumps(prior_output, ensure_ascii=False),
                "FULL PUBMEDQA CONTEXT:\n"
                + _format_evidence(evidence_split["full_context"]),
                "Choose exactly one action: KEEP_ANSWER with the same answer; "
                "REVISE_ANSWER with a different yes/no/maybe answer; or ABSTAIN "
                "with answer null. Include confidence (0.0-1.0) and "
                "reason_for_action in the JSON object.",
            ]
        )

    raise ValueError(f"Unsupported stage: {stage}")


REPAIR_TEMPLATE = """Your previous response was invalid: {ERROR}

INVALID RESPONSE:
{RESPONSE}

Return a corrected JSON object that obeys the stage rules in the original prompt.
Return JSON only.

ORIGINAL PROMPT:
{ORIGINAL}
"""
