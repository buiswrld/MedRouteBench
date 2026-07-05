"""
Groq LLM client with exponential-backoff retry and JSON mode.
"""
from tenacity import retry, wait_exponential, stop_after_attempt
from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODEL, TEMPERATURE, MAX_TOKENS

_client: Groq | None = None


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
