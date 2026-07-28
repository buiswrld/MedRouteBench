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

_data_env = os.environ.get("MEDCTA_DATA_PATH")
DATA_PATH = (
    Path(_data_env)
    if _data_env
    else PROJECT_DIR / "data" / "medcta" / "subset_v1.json"
)
RUNS_DIR = PACKAGE_DIR / "runs"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("MEDCTA_MODEL", "qwen/qwen3.6-27b")
TEMPERATURE = float(os.environ.get("MEDCTA_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.environ.get("MEDCTA_MAX_TOKENS", "512"))
MAX_RETRIES = int(os.environ.get("MEDCTA_MAX_RETRIES", "5"))
MAX_RETRY_WAIT_SECONDS = float(
    os.environ.get("MEDCTA_MAX_RETRY_WAIT_SECONDS", "60")
)
IMAGE_URL_CACHE_TTL_SECONDS = float(
    os.environ.get("MEDCTA_IMAGE_URL_CACHE_TTL_SECONDS", "2700")
)
REASONING_EFFORT = os.environ.get("MEDCTA_REASONING_EFFORT", "none")

# ── Azure OpenAI backend (takes precedence over Groq when endpoint is set) ───
AZURE_OPENAI_ENDPOINT   = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY    = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
BACKEND = "azure" if AZURE_OPENAI_ENDPOINT else "groq"
