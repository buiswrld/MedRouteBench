"""Shared LLM client factory and retry utilities for MedRouteBench."""

import re

from tenacity import wait_exponential
from openai import OpenAI, AzureOpenAI
from openai import (
    APIConnectionError as _OAIConnectionError,
    APITimeoutError as _OAITimeoutError,
    InternalServerError as _OAIInternalServerError,
    RateLimitError as _OAIRateLimitError,
)

from .config import (
    AZURE_API_KEY,
    AZURE_API_VERSION,
    AZURE_ENDPOINT,
)

_client = None
_fallback_wait = wait_exponential(multiplier=1, min=1, max=60)
_retryable_errors = (
    _OAIRateLimitError,
    _OAIConnectionError,
    _OAITimeoutError,
    _OAIInternalServerError,
)


def get_client():
    global _client
    if _client is None:
        if not AZURE_ENDPOINT or not AZURE_API_KEY:
            raise RuntimeError(
                "AZURE_ENDPOINT and AZURE_API_KEY must be set. "
                "Add them to MedRouteBench/.env or export in your environment."
            )
        if AZURE_API_VERSION:
            # Legacy versioned endpoint.
            _client = AzureOpenAI(
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY,
                api_version=AZURE_API_VERSION,
            )
        else:
            # Azure AI Foundry v1 API — append /openai/v1/ if not already present.
            base = AZURE_ENDPOINT.rstrip("/")
            if not base.endswith("/openai/v1"):
                base = base + "/openai/v1"
            base = base + "/"  # OpenAI client requires trailing slash
            _client = OpenAI(base_url=base, api_key=AZURE_API_KEY)
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


def _reported_retry_delay(exception: Exception) -> float | None:
    response = getattr(exception, "response", None)
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            pass
    return _parse_retry_after_message(str(exception))


def _retry_wait(retry_state) -> float:
    """Honor Retry-After headers, including rolling daily-token limits."""
    exception = retry_state.outcome.exception()
    seconds = None
    if isinstance(exception, _OAIRateLimitError):
        seconds = _reported_retry_delay(exception)
    if seconds is None:
        seconds = float(_fallback_wait(retry_state))
    seconds = max(1.0, seconds) + 0.5
    print(
        f"[RETRY] {type(exception).__name__}; waiting {seconds:.1f}s "
        f"(attempt {retry_state.attempt_number})",
        flush=True,
    )
    return seconds
