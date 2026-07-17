"""Groq vision adapter with bounded transient retries and JSON mode."""

import re
import time
import urllib.request
from collections import OrderedDict
from urllib.parse import urlparse

from groq import (
    APIConnectionError,
    APITimeoutError,
    Groq,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    IMAGE_URL_CACHE_TTL_SECONDS,
    MAX_RETRIES,
    MAX_RETRY_WAIT_SECONDS,
    MAX_TOKENS,
    REASONING_EFFORT,
    TEMPERATURE,
)


_client: Groq | None = None
_IMAGE_URL_CACHE_MAX_SIZE = 128
_image_url_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
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
                "GROQ_API_KEY is not set. Add it to MedRouteBench/.env or "
                "export it in the calling environment."
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


def _should_retry(exception: Exception) -> bool:
    if not isinstance(exception, _retryable_errors):
        return False
    if isinstance(exception, RateLimitError):
        reported = _reported_retry_delay(exception)
        if reported is not None and reported > MAX_RETRY_WAIT_SECONDS:
            print(
                f"[NO RETRY] RateLimitError requested {reported:.1f}s, above "
                f"MEDCTA_MAX_RETRY_WAIT_SECONDS={MAX_RETRY_WAIT_SECONDS:.1f}s",
                flush=True,
            )
            return False
    return True


def _retry_wait(retry_state) -> float:
    exception = retry_state.outcome.exception()
    seconds = None
    if isinstance(exception, RateLimitError):
        seconds = _reported_retry_delay(exception)
    if seconds is None:
        seconds = float(_fallback_wait(retry_state))
    seconds = max(1.0, seconds) + 0.5
    print(
        f"[RETRY] {type(exception).__name__}; waiting {seconds:.1f}s "
        f"(attempt {retry_state.attempt_number}/{MAX_RETRIES})",
        flush=True,
    )
    return seconds


def clear_image_url_cache() -> None:
    """Clear resolved image redirects (primarily useful for tests)."""
    _image_url_cache.clear()


def resolve_image_url(image_url: str) -> str:
    """Resolve Hugging Face's redirect before giving the URL to Groq.

    Groq's media fetcher currently rejects the initial 302 returned by
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
    """Call a Groq vision model and return its raw JSON response text."""
    user_content: str | list[dict]
    if image_url:
        resolved_image_url = resolve_image_url(image_url)
        user_content = [
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {"url": resolved_image_url}},
        ]
    else:
        user_content = user
    request_kwargs = dict(
        model=model or GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        response_format={"type": "json_object"},
    )
    try:
        response = get_client().chat.completions.create(**request_kwargs)
    except Exception as exc:
        # Groq can reject an otherwise valid request when JSON-mode generation
        # is empty. Retry only that provider-side condition without JSON mode;
        # runner.py still validates and repairs the returned text contract.
        if "json_validate_failed" not in str(exc):
            raise
        request_kwargs.pop("response_format")
        response = get_client().chat.completions.create(**request_kwargs)
    return response.choices[0].message.content


def preflight_model(model: str | None = None) -> dict:
    """Confirm that the configured model is visible to the current Groq account."""
    selected = model or GROQ_MODEL
    available = {entry.id for entry in get_client().models.list().data}
    return {"model": selected, "available": selected in available}
