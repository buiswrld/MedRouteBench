"""Azure OpenAI-compatible text adapter for the PubMedQA evaluation."""

import re
from urllib.parse import urlparse, urlunparse

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import (
    AZURE_MAX_COMPLETION_TOKENS,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_REASONING_EFFORT,
    AZURE_SEED,
    MAX_RETRIES,
)


_client = None


class AzureJSONResponse(str):
    """JSON text with non-secret provider accounting metadata attached."""

    def __new__(cls, value: str, *, metadata: dict):
        instance = super().__new__(cls, value)
        instance.metadata = metadata
        return instance


def _dump_provider_value(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return {
            key: _dump_provider_value(item)
            for key, item in vars(value).items()
            if item is not None
        }
    return value


def normalize_base_url(endpoint: str | None) -> str:
    """Normalize an Azure resource endpoint to its OpenAI-compatible v1 root."""
    if not endpoint or not endpoint.strip():
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is not set. Add the Azure resource endpoint "
            "to MedRouteBench/.env; it may be either the resource root or end "
            "with /openai/v1/."
        )
    parsed = urlparse(endpoint.strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT must be an https URL")

    path = parsed.path.rstrip("/")
    path = re.sub(r"/(?:chat/completions|responses)$", "", path)
    if not path:
        path = "/openai/v1"
    elif not path.endswith("/openai/v1"):
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT must be the Azure resource root or end with "
            "/openai/v1/, /openai/v1/chat/completions, or /openai/v1/responses"
        )
    return urlunparse((parsed.scheme, parsed.netloc, f"{path}/", "", "", ""))


def get_client():
    """Create the OpenAI client lazily so offline tests need no Azure SDK."""
    global _client
    if _client is None:
        if not AZURE_OPENAI_API_KEY:
            raise RuntimeError(
                "AZURE_OPENAI_API_KEY is not set. Add it to MedRouteBench/.env."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for Azure runs. Install project "
                "requirements with: python -m pip install -r requirements.txt"
            ) from exc
        _client = OpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            base_url=normalize_base_url(AZURE_OPENAI_ENDPOINT),
            max_retries=0,
        )
    return _client


def preflight_config(model: str | None = None) -> dict:
    """Validate required Azure settings without spending inference tokens."""
    if not AZURE_OPENAI_API_KEY:
        raise RuntimeError(
            "AZURE_OPENAI_API_KEY is not set. Add it to MedRouteBench/.env."
        )
    selected = (model or AZURE_OPENAI_DEPLOYMENT).strip()
    if not selected:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT is not set")
    base_url = normalize_base_url(AZURE_OPENAI_ENDPOINT)
    return {
        "model": selected,
        "endpoint_host": urlparse(base_url).hostname,
        "check": "local_configuration_only",
    }


def _retryable_openai_errors():
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
    except ImportError:
        return (ConnectionError, TimeoutError)
    return (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


def _call_once(request_kwargs: dict):
    return get_client().chat.completions.create(**request_kwargs)


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(_retryable_openai_errors()),
    reraise=True,
)
def _call_with_retries(request_kwargs: dict):
    return _call_once(request_kwargs)


def call_json(
    system: str,
    user: str,
    *,
    model: str | None = None,
) -> str:
    """Call the Azure deployment and return its raw JSON response text."""
    request_kwargs = {
        "model": model or AZURE_OPENAI_DEPLOYMENT,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": AZURE_MAX_COMPLETION_TOKENS,
        "response_format": {"type": "json_object"},
    }
    if AZURE_REASONING_EFFORT is not None:
        request_kwargs["reasoning_effort"] = AZURE_REASONING_EFFORT
    if AZURE_SEED is not None:
        request_kwargs["seed"] = AZURE_SEED

    response = _call_with_retries(request_kwargs)
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Azure model returned an empty response")
    return AzureJSONResponse(
        content,
        metadata={
            "provider_model": getattr(response, "model", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "usage": _dump_provider_value(getattr(response, "usage", None)),
        },
    )
