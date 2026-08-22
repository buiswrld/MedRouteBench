"""Configuration for the MedCTA reference-trajectory evaluator."""

import os
from pathlib import Path

from shared.config import (  # re-exported for package consumers
    OPENROUTER_MODEL,
    PROJECT_DIR,
)

PACKAGE_DIR = Path(__file__).resolve().parent

_data_env = os.environ.get("MEDCTA_DATA_PATH")
DATA_PATH = (
    Path(_data_env)
    if _data_env
    else PROJECT_DIR / "data" / "medcta" / "subset_v1.json"
)
RUNS_DIR = PACKAGE_DIR / "runs"

MAX_TOKENS = int(os.environ.get("MEDCTA_MAX_TOKENS", "512"))
IMAGE_URL_CACHE_TTL_SECONDS = float(
    os.environ.get("MEDCTA_IMAGE_URL_CACHE_TTL_SECONDS", "2700")
)
JUDGE_MODEL = os.environ.get("MEDCTA_JUDGE_MODEL") or None
JUDGE_API_KEY = os.environ.get("MEDCTA_JUDGE_API_KEY") or None

FINAL_ACCURACY_CONFIDENCE_THRESHOLD = 0.8
ANSWER_EQUIVALENCE_CONFIDENCE_THRESHOLD = 0.8

