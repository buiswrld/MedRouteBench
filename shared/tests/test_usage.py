import json
from types import SimpleNamespace

from shared.usage import RunUsageTracker, response_usage_record


def test_response_usage_record_normalizes_responses_api_fields():
    response = SimpleNamespace(
        id="gen-test",
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
            cost=0.0042,
        ),
    )

    record = response_usage_record(
        response,
        provider="openrouter",
        client_profile="candidate",
        model="provider/model",
    )

    assert record["response_id"] == "gen-test"
    assert record["input_tokens"] == 120
    assert record["output_tokens"] == 30
    assert record["total_tokens"] == 150
    assert record["provider_reported_cost_usd"] == 0.0042


def test_usage_tracker_persists_and_aggregates_priced_and_unpriced_calls(tmp_path):
    tracker = RunUsageTracker(tmp_path)
    tracker.record(
        {
            "provider": "openrouter",
            "client_profile": "candidate",
            "model": "provider/model",
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "provider_reported_cost_usd": 0.002,
        }
    )
    tracker.record(
        {
            "provider": "azure",
            "client_profile": "judge",
            "model": "gpt-5.4",
            "input_tokens": 40,
            "output_tokens": 5,
            "total_tokens": 45,
            "provider_reported_cost_usd": None,
        }
    )

    summary = tracker.write_summary()

    assert summary["call_count"] == 2
    assert summary["ledger_present"] is True
    assert summary["total_tokens"] == 165
    assert summary["provider_reported_cost_usd"] == 0.002
    assert summary["unpriced_calls"] == 1
    assert summary["cost_coverage_complete"] is False
    azure_route = next(
        route for route in summary["by_route"] if route["provider"] == "azure"
    )
    assert azure_route["provider_reported_cost_usd"] is None
    assert len(tmp_path.joinpath("usage.jsonl").read_text().splitlines()) == 2
    assert json.loads(tmp_path.joinpath("usage_summary.json").read_text()) == summary
