"""Shared provider configuration for all MedRouteBench evaluators.

Reads .env once (override=False so shell / CI values take precedence) and
exposes the credentials used by every package in this repo.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env", override=False)

# ── Azure backend ────────────────────────────────────────────────────────────────────
AZURE_ENDPOINT    = os.environ.get("AZURE_ENDPOINT")
AZURE_API_KEY     = os.environ.get("AZURE_API_KEY")
AZURE_DEPLOYMENT  = os.environ.get("AZURE_DEPLOYMENT")

# ── Provider selection ──────────────────────────────────────────────────────
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "azure")
OPENROUTER_API_KEY = (
    os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
)
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL")
OPENROUTER_HTTP_REFERER = os.environ.get("OPENROUTER_HTTP_REFERER")
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME")

SUPPORTED_PROVIDERS = frozenset({"azure", "openrouter"})


def normalize_provider(provider: str | None = None) -> str:
    """Return a validated provider name."""
    selected = (provider or LLM_PROVIDER or "azure").strip().lower()
    if selected not in SUPPORTED_PROVIDERS:
        choices = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(f"unsupported LLM provider {selected!r}; choose {choices}")
    return selected


def default_model(provider: str | None = None) -> str | None:
    """Return the configured default model for a provider."""
    return (
        OPENROUTER_MODEL
        if normalize_provider(provider) == "openrouter"
        else AZURE_DEPLOYMENT
    )

# ── Shared retry settings ────────────────────────────────────────────────────
MAX_RETRIES            = int(os.environ.get("MAX_RETRIES", "10"))
MAX_RETRY_WAIT_SECONDS = float(os.environ.get("MAX_RETRY_WAIT_SECONDS", "60"))

# ── Shared concurrency default ───────────────────────────────────────────────
# Number of cases to run concurrently, shared by every pipeline's --workers
# flag. 1 reproduces sequential behaviour.
WORKERS = int(os.environ.get("WORKERS", "4"))
