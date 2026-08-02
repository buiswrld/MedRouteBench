"""LLM client for staged PubMedQA evaluation."""

from shared.llm import (
    get_client,
    make_retry_decorator,
)
from .config import (
    AZURE_DEPLOYMENT,
    MAX_TOKENS,
)


@make_retry_decorator()
def call_json(system: str, user: str, *, model: str | None = None) -> str:
    """Call the LLM API with JSON mode enabled.

    Returns the raw content string (a JSON object).
    """
    effective_model = model or AZURE_DEPLOYMENT
    resp = get_client().chat.completions.create(
        model=effective_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_completion_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content

