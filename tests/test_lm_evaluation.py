import torch
import pytest

from mlac.lm_evaluation import evaluate_language_models


class TinyTokenizer:
    def encode(self, text, add_special_tokens=True):
        # Deterministic toy tokenizer: one token per character, IDs in [1, 4].
        ids = [1 + (ord(char) % 4) for char in text]
        return ([0] + ids) if add_special_tokens else ids


class TinyLM(torch.nn.Module):
    def __init__(self, preferred=1):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.preferred = preferred

    def forward(self, input_ids):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, 5)
        logits[..., self.preferred] = 2.0
        return type("Output", (), {"logits": logits})()


def test_lm_report_contains_perplexity_and_reasoning_metrics():
    tokenizer = TinyTokenizer()
    records = [
        {"text": "language model quality"},
        {"text": "held out evaluation"},
        {"question": "2 + 2 =", "choices": [" 4", " 5"], "answer_index": 0},
    ]
    baseline = TinyLM(preferred=1)
    candidate = TinyLM(preferred=1)
    report = evaluate_language_models(baseline, candidate, tokenizer, records,
                                      max_sequence_length=64, iterations=2, warmup=0)
    assert report["schema_version"] == 1
    assert report["quality_baseline"]["perplexity"] > 0
    assert report["quality_baseline"]["perplexity_score"] == -report["quality_baseline"]["perplexity"]
    assert report["quality_baseline"]["reasoning_accuracy"] == 1.0
    assert report["quality_candidate"] == report["quality_baseline"]
    assert report["baseline"]["parameter_count"] == 1


def test_lm_report_requires_valid_task_records():
    with pytest.raises(ValueError, match="invalid schema"):
        evaluate_language_models(TinyLM(), TinyLM(), TinyTokenizer(), [{"question": "bad"}])


def test_lm_report_requires_at_least_one_text_record():
    with pytest.raises(ValueError, match="at least one perplexity"):
        evaluate_language_models(TinyLM(), TinyLM(), TinyTokenizer(),
                                 [{"question": "q", "choices": ["a", "b"], "answer_index": 0}])


def test_lm_report_respects_sequence_and_iteration_budgets():
    with pytest.raises(ValueError, match="max_sequence_length"):
        evaluate_language_models(TinyLM(), TinyLM(), TinyTokenizer(), [{"text": "hello"}],
                                 max_sequence_length=99999)
