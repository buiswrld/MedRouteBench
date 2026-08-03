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

At every step, regardless of which action you choose, you must also report
your current best clinical answer and a brief justification for it. State
your best guess even before you have gathered any evidence — this lets us
track how your thinking evolves as new evidence arrives.

Return one JSON object and nothing else, with exactly these keys:
  "action": "CALL_TOOL" or "FINAL_ANSWER"
  "tool_name": an available tool name for CALL_TOOL, otherwise null
  "answer": your current best clinical answer/hypothesis given the evidence
    revealed so far. Required and nonempty at every step, including
    CALL_TOOL steps. When action is FINAL_ANSWER, this is your final answer.
  "reasoning": a brief justification for answer. Required and nonempty at
    every step.
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
    """Build one decision prompt using only information visible so far.

    `answer`/`reasoning` are stripped from prior actions before display: they
    are the model's own past guesses, tracked for scoring but never shown
    back to the model, so its current answer isn't anchored on earlier
    pre-evidence guesses.
    """
    visible_actions = [
        {k: v for k, v in action.items() if k not in ("answer", "reasoning")}
        for action in prior_model_actions
    ]
    actions = json.dumps(visible_actions, indent=2, ensure_ascii=False)
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

Return a corrected JSON object with exactly action, tool_name, answer, and
reasoning. answer and reasoning are required and nonempty at every step,
for both CALL_TOOL and FINAL_ANSWER. Use only a tool name listed in the
original prompt. Return JSON only.

ORIGINAL PROMPT:
{ORIGINAL}
"""


FINAL_ACCURACY_SYSTEM_PROMPT = """You are a medical answer evaluator.

Compare the predicted FINAL answer against the gold FINAL clinical answer.
Assign a score from 0.0 to 1.0 based on semantic clinical correctness.

CRITICAL RULE (very important):
- If the predicted answer explicitly contains the correct gold answer, assign a score of 1.0.
- Presence of the correct diagnosis/finding overrides extra guesses unless contradictory.

General rules:
- Give partial credit if only partially correct.
- Do NOT give 0.0 unless completely wrong or unrelated.
- Judge by clinical meaning, not wording.
- Synonyms count as correct.

Scoring guide:
- 1.0 = gold answer clearly present OR fully correct
- 0.8–0.95 = correct but minor imprecision
- 0.5–0.75 = partially correct
- 0.2–0.45 = weak overlap
- 0.0–0.1 = wrong/unrelated

Return JSON only:
{
  "score": number
}
"""


ANSWER_EQUIVALENCE_SYSTEM_PROMPT = """You are a medical answer evaluator.

You will be shown two clinical answers, ANSWER A and ANSWER B, given by the
same agent at two different points in its reasoning. Decide whether they
express the same clinical conclusion — i.e. whether the agent's answer
actually changed between A and B.

This is a symmetric equivalence check, not a correctness check against a
gold answer. Neither answer is "the truth" to grade the other against.
Being more specific, more general, hedged, or reworded does NOT by itself
count as a change — only score low when the two answers point to a
different diagnosis/finding/conclusion.

Scoring guide:
- 1.0 = same conclusion (identical, reworded, or one is a more/less
  specific version of the other with no contradiction)
- 0.5–0.9 = overlapping but meaningfully different conclusions
- 0.0–0.4 = different or contradictory conclusions

Return JSON only:
{
  "score": number
}
"""
