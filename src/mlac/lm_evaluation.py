"""Deterministic causal-LM perplexity and multiple-choice reasoning evaluation."""
from __future__ import annotations

import math
from time import perf_counter
from typing import Any, Iterable

import torch
import torch.nn.functional as F


def _encode(tokenizer: Any, text: str, *, special: bool = True) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=special)
    if not isinstance(ids, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in ids):
        raise ValueError("tokenizer must return a list of integer token IDs")
    return ids


def _model_logits(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    output = model(input_ids=input_ids)
    logits = output.logits if hasattr(output, "logits") else output[0]
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("causal language model must return [batch, sequence, vocabulary] logits")
    if not torch.isfinite(logits).all():
        raise ValueError("model produced non-finite logits")
    return logits


def _perplexity(model: Any, tokenizer: Any, texts: list[str], max_length: int) -> float:
    total_nll = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for text in texts:
            ids = _encode(tokenizer, text)
            ids = ids[-max_length:]
            if len(ids) < 2:
                raise ValueError("each perplexity text must encode to at least two tokens")
            tensor = torch.tensor([ids], dtype=torch.long)
            logits = _model_logits(model, tensor)
            shift_logits = logits[:, :-1, :].contiguous()
            labels = tensor[:, 1:].contiguous()
            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )
            total_nll += float(nll.item())
            total_tokens += int(labels.numel())
    if total_tokens == 0:
        raise ValueError("perplexity dataset contains no prediction tokens")
    value = math.exp(min(total_nll / total_tokens, 80.0))
    if not math.isfinite(value):
        raise ValueError("perplexity is not finite")
    return value


def _choice_score(model: Any, tokenizer: Any, prompt: str, choice: str, max_length: int) -> float:
    prefix = _encode(tokenizer, prompt)
    continuation = _encode(tokenizer, choice, special=False)
    if not prefix or not continuation:
        raise ValueError("reasoning prompts and choices must encode to non-empty token sequences")
    continuation = continuation[:max_length - 1]
    prefix = prefix[-(max_length - len(continuation)):]
    ids = prefix + continuation
    tensor = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        logits = _model_logits(model, tensor)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = tensor[:, 1:]
        # Continuation token j is predicted at sequence position len(prefix)-1+j.
        start = len(prefix) - 1
        selected = log_probs[0, start:start + len(continuation), :].gather(
            -1, targets[0, start:start + len(continuation)].unsqueeze(-1)
        )
    return float(selected.sum().item())


def _reasoning_accuracy(model: Any, tokenizer: Any, examples: list[dict], max_length: int) -> float:
    correct = 0
    with torch.inference_mode():
        for item in examples:
            scores = [
                _choice_score(model, tokenizer, item["question"], choice, max_length)
                for choice in item["choices"]
            ]
            correct += int(max(range(len(scores)), key=scores.__getitem__) == item["answer_index"])
    return correct / len(examples)


def _benchmark(model: Any, tokenizer: Any, sample_text: str, max_length: int,
               iterations: int, warmup: int) -> dict:
    ids = _encode(tokenizer, sample_text)[-max_length:]
    if len(ids) < 2:
        raise ValueError("benchmark sample must encode to at least two tokens")
    tensor = torch.tensor([ids], dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _model_logits(model, tensor)
        start = perf_counter()
        output = None
        for _ in range(iterations):
            output = _model_logits(model, tensor)
        latency = (perf_counter() - start) * 1000.0 / iterations
    return {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "latency_ms_per_iteration": latency,
        "iterations": iterations,
        "output_shape": list(output.shape),
        "finite_output": bool(torch.isfinite(output).all().item()),
    }


def evaluate_language_models(
    baseline_model: Any,
    candidate_model: Any,
    tokenizer: Any,
    records: Iterable[dict],
    *,
    max_sequence_length: int = 2048,
    iterations: int = 10,
    warmup: int = 2,
) -> dict:
    """Return the existing MLAC worker report schema with LM-specific quality metrics.

    Each record is either {"text": "..."} for perplexity or
    {"question": "...", "choices": [...], "answer_index": 0} for
    multiple-choice reasoning. Higher quality scores are better; perplexity
    is also reported directly and as negative perplexity for the pipeline gate.
    """
    if not 2 <= max_sequence_length <= 4096:
        raise ValueError("max_sequence_length must be in [2, 4096]")
    if not 1 <= iterations <= 100 or not 0 <= warmup <= 20:
        raise ValueError("iterations must be in [1, 100] and warmup in [0, 20]")
    texts, reasoning = [], []
    for index, item in enumerate(records):
        if index >= 10000 or not isinstance(item, dict):
            raise ValueError("dataset must contain at most 10000 JSON objects")
        if set(item) == {"text"} and isinstance(item["text"], str) and item["text"].strip():
            texts.append(item["text"])
        elif (set(item) == {"question", "choices", "answer_index"}
              and isinstance(item["question"], str) and item["question"].strip()
              and isinstance(item["choices"], list) and 2 <= len(item["choices"]) <= 8
              and all(isinstance(c, str) and c.strip() for c in item["choices"])
              and isinstance(item["answer_index"], int) and not isinstance(item["answer_index"], bool)
              and 0 <= item["answer_index"] < len(item["choices"])):
            reasoning.append(item)
        else:
            raise ValueError(f"dataset record {index} has an invalid schema")
    if not texts:
        raise ValueError("dataset must include at least one perplexity text record")
    sample = texts[0]
    base_ppl = _perplexity(baseline_model, tokenizer, texts, max_sequence_length)
    cand_ppl = _perplexity(candidate_model, tokenizer, texts, max_sequence_length)
    base_quality = {"score": -base_ppl, "perplexity": base_ppl, "perplexity_score": -base_ppl}
    cand_quality = {"score": -cand_ppl, "perplexity": cand_ppl, "perplexity_score": -cand_ppl}
    if reasoning:
        base_acc = _reasoning_accuracy(baseline_model, tokenizer, reasoning, max_sequence_length)
        cand_acc = _reasoning_accuracy(candidate_model, tokenizer, reasoning, max_sequence_length)
        base_quality["reasoning_accuracy"] = base_acc
        cand_quality["reasoning_accuracy"] = cand_acc
    return {
        "schema_version": 1,
        "baseline": _benchmark(baseline_model, tokenizer, sample, max_sequence_length, iterations, warmup),
        "candidate": _benchmark(candidate_model, tokenizer, sample, max_sequence_length, iterations, warmup),
        "quality_baseline": base_quality,
        "quality_candidate": cand_quality,
    }
