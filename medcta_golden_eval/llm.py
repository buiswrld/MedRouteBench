"""LLM vision client for MedCTA evaluation."""

import json

from shared.llm import (
    call_responses_json,
    resolve_image_url,
    vision_input,
)
from .config import (
    IMAGE_URL_CACHE_TTL_SECONDS,
    JUDGE_API_KEY,
    JUDGE_BASE_URL,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    MAX_TOKENS,
    OPENROUTER_MODEL,
)


def call_json(
    system: str,
    user: str,
    image_url: str | None,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Call the configured vision model and return its raw JSON response text."""
    effective_model = model or OPENROUTER_MODEL
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
        api_key=api_key,
    )


def _run_judge(system_prompt: str, user: str) -> float | None:
    effective_model = JUDGE_MODEL
    if not effective_model:
        return None
    try:
        raw = call_responses_json(
            system_prompt,
            user,
            model=effective_model,
            max_output_tokens=64,
            api_key=JUDGE_API_KEY,
            provider=JUDGE_PROVIDER,
            base_url=JUDGE_BASE_URL,
            client_profile="judge",
        )
        parsed = json.loads(raw)
        score = parsed.get("score")
        if isinstance(score, (int, float)):
            return max(0.0, min(1.0, float(score)))
    except Exception:
        return None
    return None


def judge_answer(golds: list[str], pred: str) -> float | None:
    """Score an agent's answer against one or more accepted gold answers.

    Uses ANSWER_ACCURACY_SYSTEM_PROMPT for every call, whether `pred` is a
    genuine FINAL_ANSWER or an intermediate current-best hypothesis: using
    one prompt/function for both means identical answer text always
    receives the same score, so a stage-transition label can't flip purely
    because of which judge framing was used across the FINAL_ANSWER
    boundary — only because the answer actually changed. Returns a float
    in [0.0, 1.0], or None on failure.
    """
    from .prompts import ANSWER_ACCURACY_SYSTEM_PROMPT

    if not golds or not pred:
        return None
    gold_text = "\n".join(f"- {gold}" for gold in golds)
    user = (
        f"Accepted gold answers (matching any one is sufficient):\n{gold_text}"
        f"\n\nAgent's answer:\n{pred}"
    )
    return _run_judge(ANSWER_ACCURACY_SYSTEM_PROMPT, user)


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
