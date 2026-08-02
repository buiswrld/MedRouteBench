"""Shared Azure environment configuration for all MedRouteBench evaluators.

Reads .env once (override=False so shell / CI values take precedence) and
exposes the Azure credentials used by every package in this repo.
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

# ── Shared retry settings ────────────────────────────────────────────────────
MAX_RETRIES            = int(os.environ.get("MAX_RETRIES", "5"))
MAX_RETRY_WAIT_SECONDS = float(os.environ.get("MAX_RETRY_WAIT_SECONDS", "60"))
