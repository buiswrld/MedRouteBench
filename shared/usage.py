"""Run-scoped recording of provider-reported LLM usage.

Every successful API response is appended to ``usage.jsonl`` immediately.
``usage_summary.json`` aggregates calls, tokens, and costs by provider, model,
and candidate/judge role. Calls without a provider-reported monetary cost stay
explicitly unpriced rather than receiving an estimated price.
"""

from __future__ import annotations

import datetime
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .pipeline_utils import write_json_atomic


USAGE_SCHEMA_VERSION = 1
_active_tracker = None
_active_tracker_lock = threading.Lock()


def _object_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return dict(getattr(value, "__dict__", {}) or {})


def _first_number(mapping: dict, *names: str) -> int | float | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def response_usage_record(
    response,
    *,
    provider: str,
    client_profile: str,
    model: str | None,
) -> dict:
    """Normalize usage fields exposed by OpenAI-compatible responses."""
    usage = _object_dict(getattr(response, "usage", None))
    usage_extra = usage.get("model_extra")
    if isinstance(usage_extra, dict):
        usage = {**usage_extra, **usage}
    response_extra = getattr(response, "model_extra", None)
    if isinstance(response_extra, dict):
        extra_usage = response_extra.get("usage")
        if isinstance(extra_usage, dict):
            usage = {**extra_usage, **usage}

    input_tokens = _first_number(usage, "input_tokens", "prompt_tokens")
    output_tokens = _first_number(usage, "output_tokens", "completion_tokens")
    total_tokens = _first_number(usage, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "provider": provider,
        "client_profile": client_profile,
        "model": model,
        "response_id": getattr(response, "id", None),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "provider_reported_cost_usd": _first_number(usage, "cost", "total_cost"),
    }


class RunUsageTracker:
    """Append-only usage ledger and deterministic per-run aggregation."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.ledger_path = self.run_dir / "usage.jsonl"
        self.summary_path = self.run_dir / "usage_summary.json"
        self._write_lock = threading.Lock()

    def record(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._write_lock:
            with open(self.ledger_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    def records(self) -> list[dict]:
        if not self.ledger_path.is_file():
            return []
        records = []
        with self._write_lock:
            with open(self.ledger_path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        return records

    def summary(self) -> dict:
        routes: dict[str, dict] = {}
        for record in self.records():
            route_key = "|".join(
                str(record.get(name) or "unknown")
                for name in ("provider", "client_profile", "model")
            )
            route = routes.setdefault(
                route_key,
                {
                    "provider": record.get("provider"),
                    "client_profile": record.get("client_profile"),
                    "model": record.get("model"),
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "priced_calls": 0,
                    "unpriced_calls": 0,
                    "provider_reported_cost_usd": 0.0,
                },
            )
            route["calls"] += 1
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = record.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    route[key] += value
            cost = record.get("provider_reported_cost_usd")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                route["priced_calls"] += 1
                route["provider_reported_cost_usd"] += float(cost)
            else:
                route["unpriced_calls"] += 1

        ordered_routes = [routes[key] for key in sorted(routes)]
        for route in ordered_routes:
            if route["priced_calls"] == 0:
                route["provider_reported_cost_usd"] = None
        unpriced_calls = sum(route["unpriced_calls"] for route in ordered_routes)
        known_cost = sum(
            route["provider_reported_cost_usd"] or 0.0 for route in ordered_routes
        )
        total_calls = sum(route["calls"] for route in ordered_routes)
        return {
            "schema_version": USAGE_SCHEMA_VERSION,
            "ledger_present": self.ledger_path.is_file(),
            "call_count": total_calls,
            "input_tokens": sum(route["input_tokens"] for route in ordered_routes),
            "output_tokens": sum(route["output_tokens"] for route in ordered_routes),
            "total_tokens": sum(route["total_tokens"] for route in ordered_routes),
            "provider_reported_cost_usd": round(known_cost, 10),
            "unpriced_calls": unpriced_calls,
            "cost_coverage_complete": total_calls > 0 and unpriced_calls == 0,
            "by_route": ordered_routes,
        }

    def write_summary(self) -> dict:
        summary = self.summary()
        write_json_atomic(self.summary_path, summary)
        return summary


@contextmanager
def activate_usage_tracker(tracker: RunUsageTracker) -> Iterator[RunUsageTracker]:
    """Make one run ledger visible to API calls, including worker threads."""
    global _active_tracker
    with _active_tracker_lock:
        if _active_tracker is not None and _active_tracker is not tracker:
            raise RuntimeError("another run is already collecting LLM usage")
        _active_tracker = tracker
    try:
        yield tracker
    finally:
        tracker.write_summary()
        with _active_tracker_lock:
            if _active_tracker is tracker:
                _active_tracker = None


def record_response_usage(
    response,
    *,
    provider: str,
    client_profile: str,
    model: str | None,
) -> None:
    """Record one successful response when a pipeline ledger is active."""
    with _active_tracker_lock:
        tracker = _active_tracker
    if tracker is None:
        return
    tracker.record(
        response_usage_record(
            response,
            provider=provider,
            client_profile=client_profile,
            model=model,
        )
    )
