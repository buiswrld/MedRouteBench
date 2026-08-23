from types import SimpleNamespace

import shared.llm as shared_llm
from shared.llm import call_responses_json, resolve_image_url, vision_input
from shared.usage import RunUsageTracker, activate_usage_tracker


def test_azure_client_normalizes_the_fixed_judge_endpoint(monkeypatch):
    captured = {}

    def _fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(shared_llm, "OpenAI", _fake_openai)
    shared_llm._clients.clear()

    shared_llm.get_client(
        "judge-key",
        provider="azure",
        base_url="https://judge-resource.services.ai.azure.com",
    )

    assert captured == {
        "base_url": "https://judge-resource.services.ai.azure.com/openai/v1/",
        "api_key": "judge-key",
    }
    shared_llm._clients.clear()


def test_call_responses_json_sends_json_mode_request(monkeypatch):
    captured = {}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text='{"ok":true}')

    client = SimpleNamespace(responses=Responses())
    monkeypatch.setattr(
        shared_llm,
        "get_client",
        lambda api_key=None, provider="openrouter", base_url=None: client,
    )

    result = call_responses_json(
        "system prompt",
        "user text",
        model="my-model",
        max_output_tokens=123,
    )

    assert result == '{"ok":true}'
    assert captured["model"] == "my-model"
    assert captured["instructions"] == "system prompt"
    assert captured["input"] == "user text"
    assert captured["text"] == {"format": {"type": "json_object"}}
    assert captured["max_output_tokens"] == 123


def test_call_responses_json_falls_back_without_json_mode_on_failure(monkeypatch):
    captured = []

    class Responses:
        def create(self, **kwargs):
            captured.append(kwargs)
            if len(captured) == 1:
                raise RuntimeError("provider code=json_validate_failed")
            return SimpleNamespace(output_text='{"tool_name":"OCR"}')

    client = SimpleNamespace(responses=Responses())
    monkeypatch.setattr(
        shared_llm,
        "get_client",
        lambda api_key=None, provider="openrouter", base_url=None: client,
    )

    result = call_responses_json(
        "system prompt", "user text", model="my-model", max_output_tokens=64
    )

    assert result == '{"tool_name":"OCR"}'
    assert captured[0]["text"] == {"format": {"type": "json_object"}}
    assert "text" not in captured[1]


def test_vision_input_builds_content_items():
    result = vision_input("describe this", "https://example.test/image.jpg")
    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe this"},
                {"type": "input_image", "image_url": "https://example.test/image.jpg"},
            ],
        }
    ]


def test_huggingface_image_resolution_is_used_without_changing_other_hosts(monkeypatch):
    assert resolve_image_url("https://example.test/image.jpg", ttl_seconds=2700.0) == (
        "https://example.test/image.jpg"
    )

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "image/jpeg"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def geturl(self):
            return "https://cdn.example.test/resolved.jpg"

    shared_llm.clear_image_url_cache()
    monkeypatch.setattr(shared_llm.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    resolved = resolve_image_url(
        "https://huggingface.co/datasets/x/resolve/y/image.jpg", ttl_seconds=2700.0
    )
    assert resolved == "https://cdn.example.test/resolved.jpg"


def test_huggingface_image_resolution_cache_expires_before_signed_url(monkeypatch):
    clock = [100.0]
    requests = []

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "image/jpeg"}

        def __init__(self, resolved):
            self.resolved = resolved

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def geturl(self):
            return self.resolved

    def fake_urlopen(*_args, **_kwargs):
        requests.append(len(requests) + 1)
        return FakeResponse(f"https://cdn.example.test/resolved-{len(requests)}.jpg")

    shared_llm.clear_image_url_cache()
    monkeypatch.setattr(shared_llm.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(shared_llm.urllib.request, "urlopen", fake_urlopen)
    source = "https://huggingface.co/datasets/x/resolve/y/image.jpg"

    first = resolve_image_url(source, ttl_seconds=10.0)
    clock[0] = 109.9
    cached = resolve_image_url(source, ttl_seconds=10.0)
    clock[0] = 110.0
    refreshed = resolve_image_url(source, ttl_seconds=10.0)

    assert first == cached == "https://cdn.example.test/resolved-1.jpg"
    assert refreshed == "https://cdn.example.test/resolved-2.jpg"
    assert len(requests) == 2
