"""Groq LLM client with quota-aware retry and JSON mode."""

import re

from groq import (
    APIConnectionError,
    APITimeoutError,
    Groq,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import GROQ_API_KEY, GROQ_MODEL, TEMPERATURE, MAX_TOKENS

_client: Groq | None = None
_fallback_wait = wait_exponential(multiplier=1, min=1, max=60)
_retryable_errors = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)


def get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Add it to MedRouteBench/.env or export it in your shell."
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _parse_retry_after_message(message: str) -> float | None:
    match = re.search(
        r"try again in\s+(?:(?P<minutes>\d+)m)?(?P<seconds>[\d.]+)s",
        message,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return 60 * int(match.group("minutes") or 0) + float(match.group("seconds"))


def _retry_wait(retry_state) -> float:
    """Honor Groq's Retry-After value, including rolling daily-token limits."""
    exception = retry_state.outcome.exception()
    seconds = None
    if isinstance(exception, RateLimitError):
        response = getattr(exception, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                seconds = float(retry_after)
            except (TypeError, ValueError):
                seconds = None
        if seconds is None:
            seconds = _parse_retry_after_message(str(exception))
    if seconds is None:
        seconds = float(_fallback_wait(retry_state))
    seconds = max(1.0, seconds) + 0.5
    print(
        f"[RETRY] {type(exception).__name__}; waiting {seconds:.1f}s "
        f"(attempt {retry_state.attempt_number})",
        flush=True,
    )
    return seconds


@retry(
    wait=_retry_wait,
    stop=stop_after_attempt(25),
    retry=retry_if_exception_type(_retryable_errors),
    reraise=True,
)
def call_json(system: str, user: str, *, model: str | None = None) -> str:
    """
    Call the Groq chat completion API with JSON mode enabled.
    Returns the raw content string (a JSON object).
    """
    resp = get_client().chat.completions.create(
        model=model or GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content
