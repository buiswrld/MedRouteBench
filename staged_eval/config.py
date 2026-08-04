"""Centralised configuration for staged_eval."""
import os
from pathlib import Path

from shared.config import (  # re-exported for package consumers
    AZURE_DEPLOYMENT,
    PROJECT_DIR,
)

# ── directory anchors ────────────────────────────────────────────────────────
PACKAGE_DIR = Path(__file__).resolve().parent   # .../MedRouteBench/staged_eval/

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
MAX_TOKENS  = int(os.environ.get("MAX_TOKENS", "2048"))
# Best-effort determinism. gpt-5-mini rejects temperature != 1, so we only pin a
# seed (which it accepts); leave temperature at the model default.
SEED = int(os.environ.get("SEED", "42"))

# ── ablation toggles ─────────────────────────────────────────────────────────
# LLM_API: "chat" (chat.completions) or "responses" (OpenAI Responses API).
LLM_API = os.environ.get("LLM_API", "responses")
# STAGE2_ADDED_EVIDENCE: when true, the Stage 2 prompt shows an explicit
# "ADDED EVIDENCE" block for the section revealed after Stage 1.
STAGE2_ADDED_EVIDENCE = os.environ.get("STAGE2_ADDED_EVIDENCE", "1") == "1"

# ── concurrency ──────────────────────────────────────────────────────────────
# Number of cases to run concurrently. 1 reproduces the sequential behaviour.
WORKERS = int(os.environ.get("WORKERS", "4"))

