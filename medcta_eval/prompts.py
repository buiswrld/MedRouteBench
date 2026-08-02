"""Leakage-resistant prompts for MedCTA controlled replay."""

import json


SYSTEM_PROMPT = """You are a clinical tool-use agent.

You have access to a set of tools. At each step, you must decide whether to
call a tool to gather more information, or to give your final answer.

Choose exactly one of two actions:
- CALL_TOOL: select one tool from AVAILABLE TOOLS to run. You will receive
  the tool's output as an observation before your next decision.
- FINAL_ANSWER: provide your answer to the clinical question. Only choose
  this when you have enough information.

Typically you should gather evidence through tools before answering.
Do not jump to FINAL_ANSWER without using tools when the question requires
observation or measurement from the image.

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
