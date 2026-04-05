"""
Capability evaluation benchmarks.

Measures whether a model retains general capabilities after safety training
and fine-tuning:
- MMLU: general knowledge benchmark
- Perplexity: language modeling quality
"""

import torch
import numpy as np
from dataclasses import dataclass
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class CapabilityEvalResult:
    """Results from capability evaluation."""
    mmlu_accuracy: float | None
    perplexity: float | None


def evaluate_perplexity(
    model,
    tokenizer,
    texts: list[str] | None = None,
    max_length: int = 512,
    stride: int = 256,
) -> float:
    """Compute perplexity on a set of texts.

    Uses sliding window approach for texts longer than max_length.

    Args:
        model: HuggingFace CausalLM.
        tokenizer: Corresponding tokenizer.
        texts: Texts to evaluate on. Defaults to a sample of WikiText-2.
        max_length: Max context window per chunk.
        stride: Sliding window stride.

    Returns:
        Perplexity (lower is better).
    """
    model.eval()

    if texts is None:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in dataset["text"] if len(t.strip()) > 50][:200]

    total_nll = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc="Computing perplexity"):
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encodings["input_ids"].to(model.device)

        if input_ids.size(1) < 2:
            continue

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nll = outputs.loss.item()

        num_tokens = input_ids.size(1) - 1
        total_nll += nll * num_tokens
        total_tokens += num_tokens

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    return float(np.exp(avg_nll))


def evaluate_capability_full(
    model,
    tokenizer,
) -> CapabilityEvalResult:
    """Run full capability evaluation suite.

    Args:
        model: HuggingFace CausalLM.
        tokenizer: Corresponding tokenizer.

    Returns:
        CapabilityEvalResult with all metrics.
    """
    ppl = evaluate_perplexity(model, tokenizer)

    # TODO: Add MMLU evaluation
    # Requires loading MMLU dataset, formatting as multiple-choice,
    # and computing accuracy across subjects.

    return CapabilityEvalResult(
        mmlu_accuracy=None,
        perplexity=ppl,
    )
