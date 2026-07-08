"""
System prompts and user prompt builders for the fixed two-stage revision setup.
"""
import json
STAGE1_SYSTEM_PROMPT = """You are a careful biomedical reasoning agent in a fixed two-stage PubMedQA experiment.

Use only the QUESTION and the evidence shown to you. Do not invent hidden labels,
gold answers, or any unavailable metadata.

Return exactly one JSON object and nothing else.

Required keys:
  "action": must be "ANSWER"
  "answer": "yes" | "no" | "maybe"
  "confidence": float in [0.0, 1.0]
  "reason_for_action": short string (<= 240 chars)
"""


STAGE2_SYSTEM_PROMPT = """You are a careful biomedical reasoning agent in the revision stage of a fixed two-stage PubMedQA experiment.

Use only the QUESTION, the shown Stage 1 output, and the full context shown to you.
Do not use any hidden labels or gold answers.

Return exactly one JSON object and nothing else.

Required keys:
  "action": one of ["KEEP_ANSWER","REVISE_ANSWER","ABSTAIN"]
  "answer": "yes" | "no" | "maybe" | null
  "confidence": float in [0.0, 1.0]
  "reason_for_action": short string (<= 240 chars)

Rules:
- KEEP_ANSWER: final answer must exactly match the Stage 1 answer.
- REVISE_ANSWER: final answer must be different from the Stage 1 answer.
- ABSTAIN: final answer must be null.
"""


def _format_evidence_block(title: str, pairs) -> str:
    lines = [title]
    for label, context in pairs:
        lines.append(f"[{label}] {context}")
    return "\n".join(lines)


def build_stage1_user_prompt(case: dict) -> str:
    return "\n\n".join([
        "STAGE 1: Commit to a preliminary answer using only the preliminary evidence.",
        f"QUESTION:\n{case['QUESTION']}",
        _format_evidence_block("PRELIMINARY EVIDENCE:", case["stage1_evidence"]),
        'Respond with JSON only. "action" must be "ANSWER".',
    ])


def build_stage2_user_prompt(case: dict, stage1_output: dict) -> str:
    return "\n\n".join([
        "STAGE 2: Decide whether to keep, revise, or abstain.",
        f"QUESTION:\n{case['QUESTION']}",
        f"STAGE 1 OUTPUT:\n{json.dumps(stage1_output, ensure_ascii=False)}",
        _format_evidence_block("ADDED EVIDENCE:", case["stage2_added_evidence"]),
        _format_evidence_block("FULL CONTEXT:", case["stage2_full_context"]),
        "Respond with JSON only. Choose exactly one of KEEP_ANSWER, REVISE_ANSWER, or ABSTAIN.",
    ])


REPAIR_TEMPLATE = (
    "Your previous response failed schema validation with error: {ERROR}. "
    "Return a corrected JSON object with the required keys. No prose.\n\n"
    "Original prompt:\n{ORIGINAL}"
)
