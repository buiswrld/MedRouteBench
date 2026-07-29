"""LLM client for staged PubMedQA evaluation."""

from tenacity import retry, retry_if_exception_type, stop_after_attempt

from shared.llm import (
    _retryable_errors,
    _retry_wait,
    get_client,
)
from .config import (
    AZURE_DEPLOYMENT,
    MAX_TOKENS,
)


@retry(
    wait=_retry_wait,
    stop=stop_after_attempt(25),
    retry=retry_if_exception_type(_retryable_errors),
    reraise=True,
)
def call_json(system: str, user: str, *, model: str | None = None) -> str:
    """
    Call the LLM API with JSON mode enabled.
    Returns the raw content string (a JSON object).
    """
    effective_model = model or AZURE_DEPLOYMENT
    kwargs: dict = dict(
        model=effective_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_completion_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    resp = get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content

