"""LLM client for staged PubMedQA evaluation."""

from shared.llm import (
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
def call_json(system: str, user: str, *, model: str | None = None) -> str:
    """Call the LLM API with JSON mode enabled.

    Uses chat.completions by default, or the Responses API when LLM_API is
    "responses". Returns the raw content string (a JSON object).
    """
    effective_model = model or AZURE_DEPLOYMENT
    client = get_client()
    if LLM_API == "responses":
        resp = client.responses.create(
            model=effective_model,
            instructions=system,
            input=user,
            text={"format": {"type": "json_object"}},
            max_output_tokens=MAX_TOKENS,
        )
        return resp.output_text
    resp = client.chat.completions.create(
        model=effective_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_completion_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        seed=SEED,
    )
    return resp.choices[0].message.content

