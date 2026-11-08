import json

import pytest

from mlac.chat_completions_backend import (
    ChatCompletionsProviderError,
    OpenAICompatibleChatBackend,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return self.payload


def sample_plan():
    return {
        "operation": "profile_model",
        "parameters": {
            "module_name": None,
            "heads_to_prune": None,
            "warmup_iterations": 3,
            "measurement_iterations": 20,
        },
        "hypothesis": "Profile the baseline before making structural changes.",
        "rationale": "No bottleneck should be assumed without measurements.",
        "expected_effect": {
            "latency": "measured",
            "quality": "unchanged",
            "parameters": "unchanged",
            "memory": "measured",
            "throughput": "measured",
        },
        "falsification_criteria": ["Repeated measurements are unstable"],
        "risk": "low",
        "estimated_value": 0.5,
        "feasibility": 0.95,
    }


def test_compatible_backend_calls_chat_completions_and_extracts_plan():
    captured = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse({
            "choices": [{
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps({"plans": [sample_plan()]})},
            }]
        })

    backend = OpenAICompatibleChatBackend(
        api_key="test-secret",
        model="test-model",
        base_url="https://llm.example/v1",
        opener=opener,
    )
    plans = backend.propose(
        objective={"statement": "reduce latency"},
        architecture={"name": "tiny"},
        diagnoses=[],
        constraints={"allowed_operations": ["profile_model"]},
    )
    assert plans == [sample_plan()]
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["temperature"] == 0
    assert captured["timeout"] == 60


def test_compatible_backend_rejects_refusal_and_truncated_output():
    for finish_reason, message in [
        ("stop", {"refusal": "cannot help", "content": None}),
        ("length", {"content": '{"plans": []}'}),
    ]:
        backend = OpenAICompatibleChatBackend(
            api_key="test-secret",
            model="test-model",
            opener=lambda request, timeout: FakeResponse({
                "choices": [{"finish_reason": finish_reason, "message": message}]
            }),
        )
        with pytest.raises(ChatCompletionsProviderError):
            backend.propose(objective={}, architecture={}, diagnoses=[], constraints={})


def test_compatible_backend_requires_https_and_key():
    with pytest.raises(ValueError, match="HTTPS"):
        OpenAICompatibleChatBackend(
            api_key="secret", model="test", base_url="http://localhost:8000/v1"
        )
    with pytest.raises(ValueError, match="API key"):
        OpenAICompatibleChatBackend(api_key="", model="test")
