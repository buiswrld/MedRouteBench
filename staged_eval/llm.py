"""LLM client for staged PubMedQA evaluation."""

from shared.llm import (
    call_responses_json,
    get_client,
    make_retry_decorator,
)
from .config import (
    AZURE_DEPLOYMENT,
    LLM_API,
    MAX_TOKENS,
    SEED,
)


@make_retry_decorator()
def _call_chat_json(system: str, user: str, *, model: str | None) -> str:
    resp = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_completion_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        seed=SEED,
    )
    return resp.choices[0].message.content


def call_json(system: str, user: str, *, model: str | None = None) -> str:
    """Call the LLM API with JSON mode enabled.

    Uses chat.completions by default, or the Responses API when LLM_API is
    "responses". Returns the raw content string (a JSON object).
    """
    effective_model = model or AZURE_DEPLOYMENT
    if LLM_API == "responses":
        return call_responses_json(
            system,
            user,
            model=effective_model,
            max_output_tokens=MAX_TOKENS,
        )
    return _call_chat_json(system, user, model=effective_model)

