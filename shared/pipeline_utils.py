"""Pipeline utilities shared across all MedRouteBench evaluators."""

import datetime
import hashlib
import json
import os
import re
import uuid
from pathlib import Path

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\s*```\s*$", re.DOTALL)


def strip_json_code_fence(value: str) -> str:
    """Strip a wrapping ```json ... ``` (or bare ``` ... ```) markdown fence.

    Some models (observed with Anthropic models via OpenRouter) wrap JSON
    output in a code fence even when JSON-only output was requested and
    the request otherwise succeeded, so a direct json.loads fails at char
    0. Returns ``value`` unchanged if it isn't fenced.
    """
    match = _CODE_FENCE_RE.match(value.strip())
    return match.group(1) if match else value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def callable_id(call_fn) -> str:
    module = getattr(call_fn, "__module__", type(call_fn).__module__)
    name = getattr(call_fn, "__qualname__", type(call_fn).__qualname__)
    return f"{module}.{name}"


def create_run_dir(runs_dir: Path):
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    run_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(exist_ok=False)
    return run_dir, run_id


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def ratio(numerator: int, denominator: int) -> dict:
    return {
        "rate": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }
