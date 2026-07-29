"""Azure OpenAI-compatible vision adapter for MedCTA evaluation."""

import base64
import io
import re
import time
import urllib.request
from collections import OrderedDict
from pathlib import PurePosixPath
from urllib.parse import urlparse, urlunparse

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import (
    AZURE_IMAGE_DETAIL,
    AZURE_MAX_COMPLETION_TOKENS,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_REASONING_EFFORT,
    AZURE_SEED,
    IMAGE_URL_CACHE_TTL_SECONDS,
    MAX_RETRIES,
)


_client = None
_AZURE_SUPPORTED_IMAGE_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".webp"}
_IMAGE_URL_CACHE_MAX_SIZE = 128
_image_url_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()


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
                "AZURE_OPENAI_API_KEY is not set. Add the Azure resource key to "
                "MedRouteBench/.env; never commit or paste it into source files."
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


def clear_image_url_cache() -> None:
    """Clear resolved image redirects (primarily useful for tests)."""
    _image_url_cache.clear()


def resolve_image_url(image_url: str) -> str:
    """Resolve pinned Hugging Face image URLs to their current CDN target."""
    if urlparse(image_url).hostname != "huggingface.co":
        return image_url
    now = time.monotonic()
    cached = _image_url_cache.get(image_url)
    if cached is not None:
        expires_at, resolved = cached
        if expires_at > now:
            _image_url_cache.move_to_end(image_url)
            return resolved
        del _image_url_cache[image_url]

    request = urllib.request.Request(
        image_url,
        method="HEAD",
        headers={"User-Agent": "MedRouteBench-MedCTA-evaluator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        resolved = response.geturl()
        if response.status != 200:
            raise RuntimeError(
                f"MedCTA image resolution failed with HTTP {response.status}"
            )
        if not str(response.headers.get("Content-Type", "")).startswith("image/"):
            raise RuntimeError("MedCTA image URL did not resolve to image content")
    _image_url_cache[image_url] = (
        now + IMAGE_URL_CACHE_TTL_SECONDS,
        resolved,
    )
    _image_url_cache.move_to_end(image_url)
    while len(_image_url_cache) > _IMAGE_URL_CACHE_MAX_SIZE:
        _image_url_cache.popitem(last=False)
    return resolved


def prepare_image_url(image_url: str) -> str:
    """Resolve supported URLs or transcode unsupported MedCTA formats to PNG."""
    suffix = PurePosixPath(urlparse(image_url).path).suffix.casefold()
    resolved = resolve_image_url(image_url)
    if suffix in _AZURE_SUPPORTED_IMAGE_SUFFIXES:
        return resolved

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to transcode non-JPEG MedCTA images for Azure. "
            "Install project requirements with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    request = urllib.request.Request(
        resolved,
        headers={"User-Agent": "MedRouteBench-MedCTA-evaluator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        source = response.read()
    with Image.open(io.BytesIO(source)) as image:
        frame = image.copy()
    if frame.mode not in {"RGB", "RGBA"}:
        frame = frame.convert("RGB")
    output = io.BytesIO()
    frame.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def call_json(
    system: str,
    user: str,
    image_url: str | None,
    *,
    model: str | None = None,
) -> str:
    """Call an Azure-deployed multimodal model and return raw JSON text."""
    user_content: str | list[dict]
    if image_url:
        user_content = [
            {"type": "text", "text": user},
            {
                "type": "image_url",
                "image_url": {
                    "url": prepare_image_url(image_url),
                    "detail": AZURE_IMAGE_DETAIL,
                },
            },
        ]
    else:
        user_content = user

    request_kwargs = {
        "model": model or AZURE_OPENAI_DEPLOYMENT,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
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
