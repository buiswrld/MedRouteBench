"""LLM vision client for MedCTA evaluation."""

import json

from shared.llm import (
    call_responses_json,
    resolve_image_url,
    vision_input,
)
from .config import (
    AZURE_DEPLOYMENT,
    IMAGE_URL_CACHE_TTL_SECONDS,
    JUDGE_DEPLOYMENT,
    MAX_TOKENS,
)


def call_json(
    system: str,
    user: str,
    image_url: str | None,
    *,
    model: str | None = None,
) -> str:
    """Call the configured vision model and return its raw JSON response text."""
    effective_model = model or AZURE_DEPLOYMENT
    if image_url:
        resolved_image_url = resolve_image_url(
            image_url, ttl_seconds=IMAGE_URL_CACHE_TTL_SECONDS
        )
        request_input = vision_input(user, resolved_image_url)
    else:
        request_input = user
    return call_responses_json(
        system,
        request_input,
        model=effective_model,
        max_output_tokens=MAX_TOKENS,
    )


def _run_judge(system_prompt: str, user: str) -> float | None:
    effective_model = JUDGE_DEPLOYMENT or AZURE_DEPLOYMENT
    try:
        raw = call_responses_json(
            system_prompt,
            user,
            model=effective_model,
            max_output_tokens=64,
        )
        parsed = json.loads(raw)
        score = parsed.get("score")
        if isinstance(score, (int, float)):
            return max(0.0, min(1.0, float(score)))
    except Exception:
        return None
    return None


def judge_answer(gold: str, pred: str) -> float | None:
    """Score a predicted answer against the gold using LLM-as-judge.

    Uses FINAL_ACCURACY_SYSTEM_PROMPT and the same backend client as inference.
    Returns a float in [0.0, 1.0], or None on failure.
    """
    from .prompts import FINAL_ACCURACY_SYSTEM_PROMPT

    if not gold or not pred:
        return None
    user = f"Gold final answer:\n{gold}\n\nPredicted final answer:\n{pred}"
    return _run_judge(FINAL_ACCURACY_SYSTEM_PROMPT, user)


def judge_equivalence(previous: str, current: str) -> float | None:
    """Score whether two of the agent's own answers express the same conclusion.

    Uses ANSWER_EQUIVALENCE_SYSTEM_PROMPT, a symmetric equivalence check —
    distinct from judge_answer, which grades correctness against a gold
    answer and is a poor fit here (its "contains the gold answer" rule
    biases toward scoring refinements as unchanged).
    """
    from .prompts import ANSWER_EQUIVALENCE_SYSTEM_PROMPT

    if not previous or not current:
        return None
    user = f"ANSWER A:\n{previous}\n\nANSWER B:\n{current}"
    return _run_judge(ANSWER_EQUIVALENCE_SYSTEM_PROMPT, user)
