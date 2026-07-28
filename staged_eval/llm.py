"""LLM client — routes to Azure OpenAI or Groq based on BACKEND configuration."""

import re

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    BACKEND,
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_TOKENS,
)

from openai import OpenAI, AzureOpenAI
from openai import (
    APIConnectionError as _OAIConnectionError,
    APITimeoutError as _OAITimeoutError,
    InternalServerError as _OAIInternalServerError,
    RateLimitError as _OAIRateLimitError,
)
from groq import Groq
from groq import (
    APIConnectionError as _GroqConnectionError,
    APITimeoutError as _GroqTimeoutError,
    InternalServerError as _GroqInternalServerError,
    RateLimitError as _GroqRateLimitError,
)

_client = None
_fallback_wait = wait_exponential(multiplier=1, min=1, max=60)
_retryable_errors = (
    _OAIRateLimitError,
    _OAIConnectionError,
    _OAITimeoutError,
    _OAIInternalServerError,
    _GroqRateLimitError,
    _GroqConnectionError,
    _GroqTimeoutError,
    _GroqInternalServerError,
)


def get_client():
    global _client
    if _client is None:
        if BACKEND == "azure":
            if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
                raise RuntimeError(
                    "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set. "
                    "Add them to MedRouteBench/.env or export in your shell."
                )
            if AZURE_OPENAI_API_VERSION:
                # Legacy versioned endpoint: AzureOpenAI client with explicit api-version.
                _client = AzureOpenAI(
                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    api_key=AZURE_OPENAI_API_KEY,
                    api_version=AZURE_OPENAI_API_VERSION,
                )
            else:
                # New Azure v1 API (recommended for Foundry resources).
                # Append /openai/v1/ to the base endpoint if not already present.
                base = AZURE_OPENAI_ENDPOINT.rstrip("/")
                if not base.endswith("/openai/v1"):
                    base = base + "/openai/v1"
                base = base + "/"  # OpenAI client requires trailing slash
                _client = OpenAI(
                    base_url=base,
                    api_key=AZURE_OPENAI_API_KEY,
                )
        else:
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
    if isinstance(exception, (_OAIRateLimitError, _GroqRateLimitError)):
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
    Call the LLM API with JSON mode enabled.
    Returns the raw content string (a JSON object).
    """
    effective_model = model or (AZURE_OPENAI_DEPLOYMENT if BACKEND == "azure" else GROQ_MODEL)
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
