"""LLM vision client for MedCTA trajectory inference."""

import time
import urllib.request
from collections import OrderedDict
from urllib.parse import urlparse

from tenacity import retry, retry_if_exception, stop_after_attempt
from openai import RateLimitError as _OAIRateLimitError

from shared.llm import (
    _parse_retry_after_message,
    _reported_retry_delay,
    _retry_wait,
    _retryable_errors,
    get_client,
)
from .config import (
    AZURE_DEPLOYMENT,
    IMAGE_URL_CACHE_TTL_SECONDS,
    MAX_RETRIES,
    MAX_RETRY_WAIT_SECONDS,
    MAX_TOKENS,
    REASONING_EFFORT,
)

_IMAGE_URL_CACHE_MAX_SIZE = 128
_image_url_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()


def _should_retry(exception: Exception) -> bool:
    if not isinstance(exception, _retryable_errors):
        return False
    if isinstance(exception, _OAIRateLimitError):
        reported = _reported_retry_delay(exception)
        if reported is not None and reported > MAX_RETRY_WAIT_SECONDS:
            print(
                f"[NO RETRY] RateLimitError requested {reported:.1f}s, above "
                f"MEDCTA_MAX_RETRY_WAIT_SECONDS={MAX_RETRY_WAIT_SECONDS:.1f}s",
                flush=True,
            )
            return False
    return True


def clear_image_url_cache() -> None:
    """Clear resolved image redirects (primarily useful for tests)."""
    _image_url_cache.clear()


def resolve_image_url(image_url: str) -> str:
    """Resolve Hugging Face's redirect to its final CDN URL.

    Some model APIs reject the initial 302 returned by
    ``huggingface.co/.../resolve/...``. The final CDN URL is resolved in memory
    and cached for less than the signed URL lifetime; the pinned source URL
    remains the auditable reference stored in prompts, traces, and manifests.
    """
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


@retry(
    wait=_retry_wait,
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception(_should_retry),
    reraise=True,
)
def call_json(
    system: str,
    user: str,
    image_url: str | None,
    *,
    model: str | None = None,
) -> str:
    """Call the configured vision model and return its raw JSON response text."""
    user_content: str | list[dict]
    if image_url:
        resolved_image_url = resolve_image_url(image_url)
        user_content = [
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {"url": resolved_image_url}},
        ]
    else:
        user_content = user
    effective_model = model or AZURE_DEPLOYMENT
    request_kwargs = dict(
        model=effective_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    if REASONING_EFFORT:
        request_kwargs["reasoning_effort"] = REASONING_EFFORT
    try:
        response = get_client().chat.completions.create(**request_kwargs)
    except Exception as exc:
        if "json_validate_failed" in str(exc):
            # Some providers reject JSON-mode requests when generation is empty.
            # Retry without JSON mode.
            request_kwargs.pop("response_format")
            response = get_client().chat.completions.create(**request_kwargs)
        else:
            raise
    return response.choices[0].message.content
