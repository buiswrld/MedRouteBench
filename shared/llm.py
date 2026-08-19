"""Shared LLM client factory and retry utilities for MedRouteBench."""

import base64
import io
import re
import threading
import time
import urllib.request
from collections import OrderedDict
from urllib.parse import urlparse

from PIL import Image
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from openai import OpenAI
from openai import (
    APIConnectionError as _OAIConnectionError,
    APITimeoutError as _OAITimeoutError,
    InternalServerError as _OAIInternalServerError,
    RateLimitError as _OAIRateLimitError,
)

from .config import (
    AZURE_API_KEY,
    AZURE_ENDPOINT,
    MAX_RETRIES,
    MAX_RETRY_WAIT_SECONDS,
)

_client = None
_client_lock = threading.Lock()
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
        # Double-checked locking so concurrent workers can't double-initialise.
        with _client_lock:
            if _client is None:
                if not AZURE_ENDPOINT or not AZURE_API_KEY:
                    raise RuntimeError(
                        "AZURE_ENDPOINT and AZURE_API_KEY must be set. "
                        "Add them to MedRouteBench/.env or export in your environment."
                    )
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


def _should_retry(exception: Exception) -> bool:
    """Return True iff the exception is retryable and within the wait budget."""
    if not isinstance(exception, _retryable_errors):
        return False
    if isinstance(exception, _OAIRateLimitError):
        reported = _reported_retry_delay(exception)
        if reported is not None and reported > MAX_RETRY_WAIT_SECONDS:
            print(
                f"[NO RETRY] RateLimitError requested {reported:.1f}s, above "
                f"MAX_RETRY_WAIT_SECONDS={MAX_RETRY_WAIT_SECONDS:.1f}s",
                flush=True,
            )
            return False
    return True


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


def make_retry_decorator():
    """Return the standard Tenacity retry decorator for LLM calls."""
    return retry(
        wait=_retry_wait,
        stop=stop_after_attempt(MAX_RETRIES),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )


def _final_answer_text(response) -> str:
    """Extract the model's committed response text from a Responses API result.

    ``response.output_text`` blindly concatenates every ``output_text``
    content block across every output item with no separator. Some
    deployments (e.g. gpt-5.4) emit two full ``message`` items per turn: one
    tagged ``phase="commentary"`` (a draft/preamble) and one
    ``phase="final_answer"`` (the model's actual committed output) -- and in
    JSON mode both get rendered as complete JSON objects, so blind
    concatenation glues two objects together (``{...}{...}``), breaking
    downstream JSON parsing. Prefer the final_answer-phase message text when
    phase is present; fall back to ``output_text``'s full-concatenation
    behavior for backends/models that don't emit phases.
    """
    final_answer_texts = [
        content.text
        for item in response.output
        if getattr(item, "type", None) == "message"
        and getattr(item, "phase", None) == "final_answer"
        for content in item.content
        if getattr(content, "type", None) == "output_text"
    ]
    if final_answer_texts:
        return "".join(final_answer_texts)
    return response.output_text


@make_retry_decorator()
def call_responses_json(
    instructions: str,
    input,
    *,
    model: str | None,
    max_output_tokens: int,
) -> str:
    """Call the Responses API in JSON mode and return the raw output text.

    ``input`` is either a plain string (text-only) or a pre-built list of
    Responses-API content items (e.g. from ``vision_input``); building that
    shape is the caller's job. Some providers reject JSON-mode requests when
    generation is empty, so a single retry without JSON mode is attempted
    before giving up.
    """
    client = get_client()
    request_kwargs = dict(
        model=model,
        instructions=instructions,
        input=input,
        text={"format": {"type": "json_object"}},
        max_output_tokens=max_output_tokens,
    )
    try:
        response = client.responses.create(**request_kwargs)
    except Exception as exc:
        message = str(exc)
        if "json_validate_failed" in message or "must contain the word 'json'" in message:
            request_kwargs.pop("text")
            response = client.responses.create(**request_kwargs)
        else:
            raise
    return _final_answer_text(response)


def vision_input(text: str, image_url: str) -> list[dict]:
    """Build one Responses-API user turn combining text and an image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": text},
                {"type": "input_image", "image_url": image_url},
            ],
        }
    ]


_IMAGE_URL_CACHE_MAX_SIZE = 128
_image_url_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
_image_url_cache_lock = threading.Lock()

# Content-types that the OpenAI vision API cannot render natively.
_CONVERT_TO_JPEG_TYPES = frozenset({
    "image/tiff",
    "image/bmp",
    "image/x-bmp",
    "image/x-ms-bmp",
})

# Re-encoding quality for converted images. Lossless PNG bloats large
# photographic/microscopy source images (e.g. one MedCTA histopathology TIFF
# encodes to a ~2.9MB base64 PNG, ~6x every other converted case) enough to
# trip connection resets against the vision endpoint; JPEG at this quality
# keeps full resolution while cutting that same image to ~550KB.
_CONVERTED_IMAGE_JPEG_QUALITY = 90


def clear_image_url_cache() -> None:
    """Clear resolved image redirects (primarily useful for tests)."""
    with _image_url_cache_lock:
        _image_url_cache.clear()


def _jpeg_data_url(url: str) -> str:
    """Download a non-web-compatible image and return it as a JPEG data URL."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MedRouteBench-evaluator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=_CONVERTED_IMAGE_JPEG_QUALITY)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def resolve_image_url(image_url: str, *, ttl_seconds: float) -> str:
    """Resolve Hugging Face's redirect to its final CDN URL.

    Some model APIs reject the initial 302 returned by
    ``huggingface.co/.../resolve/...``. The final CDN URL is resolved in
    memory and cached for less than the signed URL lifetime; the pinned
    source URL remains the auditable reference stored in prompts, traces,
    and manifests.

    For TIFF/BMP images (which the OpenAI vision API does not support), the
    image is downloaded, converted to JPEG, and returned as a data URL.
    """
    if urlparse(image_url).hostname != "huggingface.co":
        return image_url
    now = time.monotonic()
    with _image_url_cache_lock:
        cached = _image_url_cache.get(image_url)
        if cached is not None:
            expires_at, resolved = cached
            if expires_at > now:
                _image_url_cache.move_to_end(image_url)
                return resolved
            del _image_url_cache[image_url]

    # Network I/O (HEAD request, and for TIFF/BMP a full download+convert)
    # runs without the lock held, so concurrent workers resolving different
    # images don't serialize on each other. A same-URL race just means two
    # threads redundantly resolve it once each; the second write below wins,
    # which is harmless for a cache.
    request = urllib.request.Request(
        image_url,
        method="HEAD",
        headers={"User-Agent": "MedRouteBench-evaluator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        cdn_url = response.geturl()
        if response.status != 200:
            raise RuntimeError(
                f"Image resolution failed with HTTP {response.status}"
            )
        content_type = str(response.headers.get("Content-Type", ""))
        if not content_type.startswith("image/"):
            raise RuntimeError("Image URL did not resolve to image content")

    if content_type in _CONVERT_TO_JPEG_TYPES:
        resolved = _jpeg_data_url(cdn_url)
    else:
        resolved = cdn_url

    with _image_url_cache_lock:
        _image_url_cache[image_url] = (now + ttl_seconds, resolved)
        _image_url_cache.move_to_end(image_url)
        while len(_image_url_cache) > _IMAGE_URL_CACHE_MAX_SIZE:
            _image_url_cache.popitem(last=False)
    return resolved
