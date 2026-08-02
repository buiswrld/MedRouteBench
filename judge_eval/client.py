"""Azure/OpenAI-compatible backend for judge calls."""

from __future__ import annotations

import os
import platform
import time
from importlib.metadata import version
from dataclasses import dataclass
from typing import Optional

from openai import (
    APIConnectionError,
    APITimeoutError,
    AzureOpenAI,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from shared.config import (
    AZURE_API_KEY,
    AZURE_API_VERSION,
    AZURE_ENDPOINT,
)
from shared.llm import _reported_retry_delay


RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)


@dataclass(frozen=True)
class JudgeClientConfig:
    endpoint: str
    api_key: str
    model: str
    model_family: str
    api_version: Optional[str] = None
    temperature: float = 0.0
    reasoning_effort: Optional[str] = None
    max_completion_tokens: int = 512
    max_retries: int = 5
    max_retry_wait_seconds: float = 60.0

    def public_dict(self) -> dict:
        return {
            "endpoint_host": self.endpoint.split("//", 1)[-1].split("/", 1)[0],
            "model": self.model,
            "model_family": self.model_family,
            "api_version": self.api_version,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_completion_tokens,
            "max_retries": self.max_retries,
            "max_retry_wait_seconds": self.max_retry_wait_seconds,
            "openai_sdk_version": version("openai"),
            "python_version": platform.python_version(),
        }


def config_from_env(
    *,
    model: Optional[str] = None,
    model_family: Optional[str] = None,
) -> JudgeClientConfig:
    endpoint = os.environ.get("JUDGE_AZURE_ENDPOINT") or AZURE_ENDPOINT
    api_key = os.environ.get("JUDGE_AZURE_API_KEY") or AZURE_API_KEY
    deployment = model or os.environ.get("JUDGE_AZURE_DEPLOYMENT")
    family = model_family or os.environ.get("JUDGE_MODEL_FAMILY")
    api_version = os.environ.get("JUDGE_AZURE_API_VERSION") or AZURE_API_VERSION
    missing = [
        name
        for name, value in (
            ("JUDGE_AZURE_ENDPOINT (or AZURE_ENDPOINT)", endpoint),
            ("JUDGE_AZURE_API_KEY (or AZURE_API_KEY)", api_key),
            ("JUDGE_AZURE_DEPLOYMENT", deployment),
            ("JUDGE_MODEL_FAMILY", family),
        )
        if not value
    ]
    if missing:
        raise RuntimeError("missing judge configuration: " + ", ".join(missing))
    return JudgeClientConfig(
        endpoint=str(endpoint),
        api_key=str(api_key),
        model=str(deployment),
        model_family=str(family).casefold(),
        api_version=api_version,
        temperature=float(os.environ.get("JUDGE_TEMPERATURE", "0")),
        reasoning_effort=os.environ.get("JUDGE_REASONING_EFFORT"),
        max_completion_tokens=int(os.environ.get("JUDGE_MAX_TOKENS", "512")),
        max_retries=int(os.environ.get("JUDGE_MAX_RETRIES", "5")),
        max_retry_wait_seconds=float(
            os.environ.get("JUDGE_MAX_RETRY_WAIT_SECONDS", "60")
        ),
    )


class OpenAIJudgeBackend:
    def __init__(self, config: JudgeClientConfig, *, client=None):
        self.config = config
        self.client = client or self._build_client(config)

    @staticmethod
    def _build_client(config: JudgeClientConfig):
        if config.api_version:
            return AzureOpenAI(
                azure_endpoint=config.endpoint,
                api_key=config.api_key,
                api_version=config.api_version,
            )
        base = config.endpoint.rstrip("/")
        if not base.endswith("/openai/v1"):
            base += "/openai/v1"
        return OpenAI(base_url=base + "/", api_key=config.api_key)

    def __call__(self, system: str, user: str) -> str:
        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                request = dict(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.config.temperature,
                    max_completion_tokens=self.config.max_completion_tokens,
                    response_format={"type": "json_object"},
                )
                if self.config.reasoning_effort:
                    request["reasoning_effort"] = self.config.reasoning_effort
                response = self.client.chat.completions.create(**request)
                return response.choices[0].message.content
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    raise
                reported = _reported_retry_delay(exc)
                wait = reported if reported is not None else 2 ** (attempt - 1)
                if wait > self.config.max_retry_wait_seconds:
                    raise
                time.sleep(max(1.0, wait))
        raise RuntimeError("judge call failed") from last_error
