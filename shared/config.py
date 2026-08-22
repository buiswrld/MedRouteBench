"""Shared OpenRouter environment configuration for all MedRouteBench evaluators.

Reads .env once (override=False so shell / CI values take precedence) and
exposes the OpenRouter credentials used by every package in this repo.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env", override=False)

# ── OpenRouter backend ───────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL    = os.environ.get("OPENROUTER_MODEL")

# ── Shared retry settings ────────────────────────────────────────────────────
MAX_RETRIES            = int(os.environ.get("MAX_RETRIES", "10"))
MAX_RETRY_WAIT_SECONDS = float(os.environ.get("MAX_RETRY_WAIT_SECONDS", "60"))

# ── Shared concurrency default ───────────────────────────────────────────────
# Number of cases to run concurrently, shared by every pipeline's --workers
# flag. 1 reproduces sequential behaviour.
WORKERS = int(os.environ.get("WORKERS", "4"))
