"""
Wrapper to make HuggingFace causal LM models compatible with PyHessian.

PyHessian expects:
    - model(inputs) -> outputs
    - criterion(outputs, targets) -> scalar loss

HuggingFace models compute loss internally. This module bridges the gap.

This is the trickiest integration piece in the project. Expect to iterate
on this when moving from CPU debugging to GPU runs.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel


class HessianModelWrapper(nn.Module):
    """Wraps a HuggingFace CausalLM for use with PyHessian.

    PyHessian calls: loss = criterion(model(inputs), targets)
    We make model(inputs) return logits, and criterion compute the
    causal LM loss (or DPO loss, depending on configuration).
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):
        """Forward pass returning only logits."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits


class CausalLMCriterion(nn.Module):
    """Standard causal language modeling loss for PyHessian.

    Computes cross-entropy between shifted logits and labels,
    matching HuggingFace's internal loss computation.
    """

    def forward(self, logits, labels):
        # Shift: predict token t+1 from position t
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss


class DPOCriterion(nn.Module):
    """DPO loss for computing the safety-task Hessian.

    For Hessian computation, we need the DPO loss as a function of
    model outputs. This requires both the policy model logits and
    reference model log-probs.

    Note: The reference model log-probs are precomputed and fixed
    (they don't depend on the current model weights), so they're
    passed as part of the target data.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, policy_logits, targets):
        """Compute DPO loss.

        Args:
            policy_logits: Logits from the current policy model.
            targets: Dict containing:
                - chosen_labels: Token IDs for chosen responses
                - rejected_labels: Token IDs for rejected responses
                - ref_chosen_logps: Reference model log-probs for chosen
                - ref_rejected_logps: Reference model log-probs for rejected
        """
        # Extract chosen/rejected from packed input
        chosen_labels = targets["chosen_labels"]
        rejected_labels = targets["rejected_labels"]
        ref_chosen_logps = targets["ref_chosen_logps"]
        ref_rejected_logps = targets["ref_rejected_logps"]

        # Compute policy log-probs for chosen and rejected
        chosen_logps = self._compute_logps(policy_logits, chosen_labels)
        rejected_logps = self._compute_logps(policy_logits, rejected_labels)

        # DPO loss
        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios
        loss = -nn.functional.logsigmoid(self.beta * logits).mean()
        return loss

    def _compute_logps(self, logits, labels):
        """Compute per-sequence log probabilities from logits."""
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        # Gather log-probs at label positions
        per_token_logps = torch.gather(
            log_probs[:, :-1, :], 2, labels[:, 1:].unsqueeze(2)
        ).squeeze(2)
        # Mask padding and sum
        mask = labels[:, 1:] != -100
        return (per_token_logps * mask).sum(dim=-1)


def prepare_hessian_data(
    tokenizer,
    texts: list[str],
    max_length: int = 512,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize texts and prepare (inputs, targets) tuple for PyHessian.

    For causal LM, inputs and targets are the same token IDs —
    the CausalLMCriterion handles the shifting internally.

    Args:
        tokenizer: HuggingFace tokenizer.
        texts: List of text strings.
        max_length: Maximum sequence length.
        device: Device to place tensors on.

    Returns:
        Tuple of (input_ids, labels) tensors.
    """
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encodings["input_ids"].to(device)
    labels = input_ids.clone()
    # Mask padding tokens in labels
    labels[labels == tokenizer.pad_token_id] = -100
    return input_ids, labels
