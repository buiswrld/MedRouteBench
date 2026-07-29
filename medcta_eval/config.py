"""Configuration for the MedCTA reference-trajectory evaluator."""

import os
from pathlib import Path

from dotenv import load_dotenv


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent

load_dotenv(PROJECT_DIR / ".env", override=False)

SOURCE_DATASET = "IVUL-KAUST/MedCTA"
SOURCE_REVISION = "0be777092121a18ba2a85a0c4145a9d4fe8ed7db"
SOURCE_PAGE_URL = f"https://huggingface.co/datasets/{SOURCE_DATASET}"
SOURCE_RAW_URL = (
    f"{SOURCE_PAGE_URL}/resolve/{SOURCE_REVISION}/raw/dataset.json"
)
SOURCE_IMAGE_BASE_URL = f"{SOURCE_PAGE_URL}/resolve/{SOURCE_REVISION}"
STARTER_CASE_IDS = tuple(str(index) for index in range(11))
FULL_CASE_IDS = tuple(str(index) for index in range(107))

_data_env = os.environ.get("MEDCTA_DATA_PATH")
DATA_PATH = (
    Path(_data_env)
    if _data_env
    else PROJECT_DIR / "data" / "medcta" / "subset_v1.json"
)
RUNS_DIR = PACKAGE_DIR / "runs"

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.environ.get(
    "AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"
)
AZURE_REASONING_EFFORT = (
    os.environ.get("MEDCTA_AZURE_REASONING_EFFORT", "low").strip() or None
)
AZURE_MAX_COMPLETION_TOKENS = int(
    os.environ.get("MEDCTA_AZURE_MAX_COMPLETION_TOKENS", "1024")
)
AZURE_IMAGE_DETAIL = os.environ.get("MEDCTA_AZURE_IMAGE_DETAIL", "auto").strip()
_azure_seed = os.environ.get("MEDCTA_AZURE_SEED")
AZURE_SEED = (
    int(_azure_seed)
    if _azure_seed is not None and _azure_seed.strip()
    else None
)

MAX_RETRIES = int(os.environ.get("MEDCTA_MAX_RETRIES", "5"))
IMAGE_URL_CACHE_TTL_SECONDS = float(
    os.environ.get("MEDCTA_IMAGE_URL_CACHE_TTL_SECONDS", "2700")
)
