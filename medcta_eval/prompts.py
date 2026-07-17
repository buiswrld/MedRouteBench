"""Leakage-resistant prompts for MedCTA controlled replay."""

import json


SYSTEM_PROMPT = """You are being evaluated as a clinical tool-routing controller.

At each step, choose exactly one of two actions:
- CALL_TOOL: select one exact tool name from AVAILABLE TOOLS.
- FINAL_ANSWER: stop and answer the clinical question.

No real tool will run. When you call a tool, the evaluator may replay a
reference observation so the controlled simulation can continue. Other tool
routes may be clinically reasonable; this benchmark measures agreement with a
single MedCTA reference trajectory.

Return one JSON object and nothing else, with exactly these keys:
  "action": "CALL_TOOL" or "FINAL_ANSWER"
  "tool_name": an available tool name for CALL_TOOL, otherwise null
  "answer": null for CALL_TOOL, otherwise a nonempty final answer
"""


def _format_tools(case: dict) -> str:
    return "\n".join(
        f"- {tool['name']}: {tool['description']}" for tool in case["available_tools"]
    )


def build_user_prompt(
    case: dict,
    prior_model_actions: list[dict],
    prior_reference_observations: list[dict],
) -> str:
    """Build one decision prompt using only information visible so far."""
    actions = json.dumps(prior_model_actions, indent=2, ensure_ascii=False)
    observations = json.dumps(
        prior_reference_observations, indent=2, ensure_ascii=False
    )
    return "\n\n".join(
        [
            f"CASE ID:\n{case['case_id']}",
            f"CLINICAL QUESTION:\n{case['question']}",
            f"IMAGE REFERENCE:\n{case['image_reference']}",
            "AVAILABLE TOOLS:\n" + _format_tools(case),
            "PRIOR MODEL ACTIONS:\n" + actions,
            "PRIOR REFERENCE OBSERVATIONS ALREADY REVEALED:\n" + observations,
            "Choose the next action now. Return the required JSON object only.",
        ]
    )


REPAIR_TEMPLATE = """Your previous response was invalid: {ERROR}

INVALID RESPONSE:
{RESPONSE}

Return a corrected JSON object with exactly action, tool_name, and answer.
Use only a tool name listed in the original prompt. Return JSON only.

ORIGINAL PROMPT:
{ORIGINAL}
"""
