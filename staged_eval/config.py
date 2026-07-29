"""
Centralised configuration for staged_eval.

Path resolution is deterministic (based on this file's location) so the
package works regardless of the caller's cwd.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ── directory anchors ────────────────────────────────────────────────────────
PACKAGE_DIR = Path(__file__).resolve().parent   # .../MedRouteBench/staged_eval/
PROJECT_DIR = PACKAGE_DIR.parent                 # .../MedRouteBench/

# Load local defaults without replacing values supplied by the calling process.
# This keeps shell, scheduler, and CI configuration authoritative.
load_dotenv(PROJECT_DIR / ".env", override=False)

# ── data paths ───────────────────────────────────────────────────────────────
_data_env = os.environ.get("PUBMEDQA_DATA_DIR")
DATA_DIR = Path(_data_env) if _data_env else PROJECT_DIR / "data" / "pubmedqa"

TEST_SET_PATH     = DATA_DIR / "test_set.json"
ORI_PQAL_PATH     = DATA_DIR / "ori_pqal.json"
GROUND_TRUTH_PATH = DATA_DIR / "test_ground_truth.json"

FIXTURE_DATA_DIR = PACKAGE_DIR / "fixtures" / "pubmedqa"
FIXTURE_TEST_SET_PATH = FIXTURE_DATA_DIR / "test_set.json"
FIXTURE_GROUND_TRUTH_PATH = FIXTURE_DATA_DIR / "test_ground_truth.json"

RUNS_DIR = PACKAGE_DIR / "runs"

# ── LLM settings ─────────────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.environ.get(
    "AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"
)
AZURE_REASONING_EFFORT = (
    os.environ.get("STAGED_AZURE_REASONING_EFFORT", "low").strip() or None
)
AZURE_MAX_COMPLETION_TOKENS = int(
    os.environ.get("STAGED_AZURE_MAX_COMPLETION_TOKENS", "1024")
)
_azure_seed = os.environ.get("STAGED_AZURE_SEED")
AZURE_SEED = (
    int(_azure_seed)
    if _azure_seed is not None and _azure_seed.strip()
    else None
)
MAX_RETRIES = int(os.environ.get("STAGED_MAX_RETRIES", "5"))
