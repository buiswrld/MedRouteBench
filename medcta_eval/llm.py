"""LLM vision client for MedCTA evaluation."""

import base64
import io
import json
import time
import urllib.request
from collections import OrderedDict
from urllib.parse import urlparse

from PIL import Image

from shared.llm import (
    get_client,
    make_retry_decorator,
)
from .config import (
    AZURE_DEPLOYMENT,
    IMAGE_URL_CACHE_TTL_SECONDS,
    JUDGE_DEPLOYMENT,
    MAX_TOKENS,
)

_IMAGE_URL_CACHE_MAX_SIZE = 128
_image_url_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()

# Content-types that the OpenAI vision API cannot render natively.
_CONVERT_TO_PNG_TYPES = frozenset({
    "image/tiff",
    "image/bmp",
    "image/x-bmp",
    "image/x-ms-bmp",
})


def clear_image_url_cache() -> None:
    """Clear resolved image redirects (primarily useful for tests)."""
    _image_url_cache.clear()


def _png_data_url(url: str) -> str:
    """Download a non-web-compatible image and return it as a PNG data URL."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MedRouteBench-MedCTA-evaluator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def resolve_image_url(image_url: str) -> str:
    """Resolve Hugging Face's redirect to its final CDN URL.

    Some model APIs reject the initial 302 returned by
    ``huggingface.co/.../resolve/...``. The final CDN URL is resolved in memory
    and cached for less than the signed URL lifetime; the pinned source URL
    remains the auditable reference stored in prompts, traces, and manifests.

    For TIFF/BMP images (which the OpenAI vision API does not support), the
    image is downloaded, converted to PNG, and returned as a data URL.
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
        cdn_url = response.geturl()
        if response.status != 200:
            raise RuntimeError(
                f"MedCTA image resolution failed with HTTP {response.status}"
            )
        content_type = str(response.headers.get("Content-Type", ""))
        if not content_type.startswith("image/"):
            raise RuntimeError("MedCTA image URL did not resolve to image content")

    if content_type in _CONVERT_TO_PNG_TYPES:
        resolved = _png_data_url(cdn_url)
    else:
        resolved = cdn_url

    _image_url_cache[image_url] = (
        now + IMAGE_URL_CACHE_TTL_SECONDS,
        resolved,
    )
    _image_url_cache.move_to_end(image_url)
    while len(_image_url_cache) > _IMAGE_URL_CACHE_MAX_SIZE:
        _image_url_cache.popitem(last=False)
    return resolved


@make_retry_decorator()
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


def _run_judge(system_prompt: str, user: str) -> float | None:
    effective_model = JUDGE_DEPLOYMENT or AZURE_DEPLOYMENT
    try:
        response = get_client().chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=64,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        score = parsed.get("score")
        if isinstance(score, (int, float)):
            return max(0.0, min(1.0, float(score)))
    except Exception:
        return None
    return None


def judge_answer(gold: str, pred: str) -> float | None:
    """Score a predicted answer against the gold using LLM-as-judge.

    Uses FINAL_ACCURACY_SYSTEM_PROMPT and the same backend client as inference.
    Returns a float in [0.0, 1.0], or None on failure.
    """
    from .prompts import FINAL_ACCURACY_SYSTEM_PROMPT

    if not gold or not pred:
        return None
    user = f"Gold final answer:\n{gold}\n\nPredicted final answer:\n{pred}"
    return _run_judge(FINAL_ACCURACY_SYSTEM_PROMPT, user)


def judge_equivalence(previous: str, current: str) -> float | None:
    """Score whether two of the agent's own answers express the same conclusion.

    Uses ANSWER_EQUIVALENCE_SYSTEM_PROMPT, a symmetric equivalence check —
    distinct from judge_answer, which grades correctness against a gold
    answer and is a poor fit here (its "contains the gold answer" rule
    biases toward scoring refinements as unchanged).
    """
    from .prompts import ANSWER_EQUIVALENCE_SYSTEM_PROMPT

    if not previous or not current:
        return None
    user = f"ANSWER A:\n{previous}\n\nANSWER B:\n{current}"
    return _run_judge(ANSWER_EQUIVALENCE_SYSTEM_PROMPT, user)
