"""Configuration for the MedCTA reference-trajectory evaluator."""

import os
from pathlib import Path

from shared.config import (  # re-exported for package consumers
    AZURE_DEPLOYMENT,
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

MAX_TOKENS             = int(os.environ.get("MEDCTA_MAX_TOKENS", "512"))
MAX_RETRIES            = int(os.environ.get("MEDCTA_MAX_RETRIES", "5"))
MAX_RETRY_WAIT_SECONDS = float(os.environ.get("MEDCTA_MAX_RETRY_WAIT_SECONDS", "60"))
IMAGE_URL_CACHE_TTL_SECONDS = float(
    os.environ.get("MEDCTA_IMAGE_URL_CACHE_TTL_SECONDS", "2700")
)

FINAL_ACCURACY_CONFIDENCE_THRESHOLD = 0.8

