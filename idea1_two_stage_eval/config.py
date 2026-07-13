"""
Centralised configuration for staged_eval.

Path resolution is deterministic (based on this file's location) so the
package works regardless of the caller's cwd.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only in minimal envs
    def load_dotenv(*_args, **_kwargs):
        return False

# ── directory anchors ────────────────────────────────────────────────────────
PACKAGE_DIR = Path(__file__).resolve().parent   # .../MedRouteBench/staged_eval/
PROJECT_DIR = PACKAGE_DIR.parent                 # .../MedRouteBench/
REPO_ROOT   = PROJECT_DIR.parent                 # .../Algoverse/

# Load .env from the project root (MedRouteBench/.env) before reading env vars.
load_dotenv(PROJECT_DIR / ".env", override=True)

# ── data paths ───────────────────────────────────────────────────────────────
_data_env = os.environ.get("PUBMEDQA_DATA_DIR")
DATA_DIR = Path(_data_env) if _data_env else REPO_ROOT / "pubmedqa" / "data"

TEST_SET_PATH     = DATA_DIR / "test_set.json"
GROUND_TRUTH_PATH = DATA_DIR / "test_ground_truth.json"

RUNS_DIR = PACKAGE_DIR / "runs"

# ── LLM settings ─────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
TEMPERATURE  = float(os.environ.get("TEMPERATURE", "0.0"))
MAX_TOKENS   = int(os.environ.get("MAX_TOKENS", "512"))
