"""LLM client for staged PubMedQA evaluation."""

from shared.llm import (
    call_responses_json,
    get_client,
    make_retry_decorator,
)
from .config import (
    AZURE_DEPLOYMENT,
    LLM_PROVIDER,
    LLM_API,
    MAX_TOKENS,
    SEED,
    default_model,
)


@make_retry_decorator()
def _call_chat_json(
    system: str,
    user: str,
    *,
    model: str | None,
    provider: str | None,
) -> str:
    resp = get_client(provider=provider).chat.completions.create(
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


def call_json(
    system: str,
    user: str,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> str:
    """Call the LLM API with JSON mode enabled.

    Uses chat.completions by default, or the Responses API when LLM_API is
    "responses". Returns the raw content string (a JSON object).
    """
    effective_provider = provider or LLM_PROVIDER
    effective_model = model or default_model(effective_provider) or AZURE_DEPLOYMENT
    if LLM_API == "responses":
        return call_responses_json(
            system,
            user,
            model=effective_model,
            max_output_tokens=MAX_TOKENS,
            provider=effective_provider,
        )
    return _call_chat_json(
        system,
        user,
        model=effective_model,
        provider=effective_provider,
    )
