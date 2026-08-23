"""Configuration for the MedCTA reference-trajectory evaluator."""

import os
from pathlib import Path

from shared.config import (  # re-exported for package consumers
    JUDGE_AZURE_API_KEY,
    JUDGE_AZURE_DEPLOYMENT,
    JUDGE_AZURE_ENDPOINT,
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
JUDGE_PROVIDER = os.environ.get("MEDCTA_JUDGE_PROVIDER", "azure").strip().lower()
if JUDGE_PROVIDER not in {"azure", "openrouter"}:
    raise ValueError("MEDCTA_JUDGE_PROVIDER must be 'azure' or 'openrouter'")
JUDGE_MODEL = (
    os.environ.get("MEDCTA_JUDGE_MODEL")
    or os.environ.get("MEDCTA_JUDGE_DEPLOYMENT")
    or (JUDGE_AZURE_DEPLOYMENT if JUDGE_PROVIDER == "azure" else None)
)
JUDGE_API_KEY = (
    os.environ.get("MEDCTA_JUDGE_API_KEY")
    or (JUDGE_AZURE_API_KEY if JUDGE_PROVIDER == "azure" else None)
)
JUDGE_BASE_URL = (
    os.environ.get("MEDCTA_JUDGE_ENDPOINT")
    or (JUDGE_AZURE_ENDPOINT if JUDGE_PROVIDER == "azure" else None)
)
JUDGE_MODEL_FAMILY = (
    os.environ.get("JUDGE_MODEL_FAMILY")
    or ("openai" if JUDGE_PROVIDER == "azure" else None)
    or (JUDGE_MODEL.split("/", 1)[0] if JUDGE_MODEL and "/" in JUDGE_MODEL else None)
    or "unknown"
).strip().lower()

FINAL_ACCURACY_CONFIDENCE_THRESHOLD = 0.8
ANSWER_EQUIVALENCE_CONFIDENCE_THRESHOLD = 0.8
