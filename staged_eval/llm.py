"""
Groq LLM client with exponential-backoff retry and JSON mode.
"""
from typing import Any

try:
    from tenacity import retry, wait_exponential, stop_after_attempt
except ImportError:  # pragma: no cover - exercised only in minimal envs
    def retry(*_args, **_kwargs):
        def decorator(fn):
            return fn
        return decorator

    def wait_exponential(*_args, **_kwargs):
        return None

    def stop_after_attempt(*_args, **_kwargs):
        return None

from .config import GROQ_API_KEY, GROQ_MODEL, TEMPERATURE, MAX_TOKENS

_client: Any = None


def get_client():
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Add it to MedRouteBench/.env or export it in your shell."
            )
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - exercised only in minimal envs
            raise RuntimeError(
                "The `groq` package is not installed. "
                "Install MedRouteBench requirements before running the online pipeline."
            ) from exc
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
def call_json(system: str, user: str) -> str:
    """
    Call the Groq chat completion API with JSON mode enabled.
    Returns the raw content string (a JSON object).
    """
    resp = get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content
