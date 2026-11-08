import json
import pytest
from mlac.openai_backend import OpenAIProviderError, OpenAIResponsesBackend


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, limit): return self.payload


def test_openai_responses_backend_uses_schema_and_extracts_plans():
    plan = {
        "operation": "profile_model",
        "parameters": {"module_name": None, "heads_to_prune": None,
                       "warmup_iterations": 5, "measurement_iterations": 30},
        "hypothesis": "Profile the candidate",
        "rationale": "Establish baseline measurements first.",
        "expected_effect": {"latency": "measured", "quality": "preserve",
                           "parameters": "unchanged", "memory": "measured",
                           "throughput": "measured"},
        "falsification_criteria": ["Repeated measurements are unstable"],
        "risk": "low", "estimated_value": 0.4, "feasibility": 0.95,
    }
    response = {"status": "completed", "output": [{
        "type": "message", "content": [{"type": "output_text",
        "text": json.dumps({"plans": [plan]})}]}]}
    captured = {}
    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse(response)
    backend = OpenAIResponsesBackend(api_key="test-secret", model="test-model", opener=opener)
    plans = backend.propose(objective={"statement": "reduce latency"},
        architecture={"name": "tiny"}, diagnoses=[], constraints={})
    assert plans == [plan]
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["store"] is False
    assert captured["body"]["text"]["format"]["type"] == "json_schema"
    assert captured["body"]["text"]["format"]["strict"] is True
    assert captured["timeout"] == 60


def test_openai_backend_rejects_incomplete_response():
    def opener(request, timeout):
        return FakeResponse({"status": "incomplete", "output": []})
    backend = OpenAIResponsesBackend(api_key="test-secret", opener=opener)
    with pytest.raises(OpenAIProviderError, match="did not complete"):
        backend.propose(objective={}, architecture={}, diagnoses=[], constraints={})


def test_openai_backend_requires_key(monkeypatch):
    monkeypatch.delenv("MLAC_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert OpenAIResponsesBackend.from_env() is None
    with pytest.raises(ValueError, match="Set MLAC_OPENAI_API_KEY"):
        OpenAIResponsesBackend.from_env(required=True)
