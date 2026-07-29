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
# Set only for legacy versioned Azure endpoints; leave unset for the v1 API.
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION")
