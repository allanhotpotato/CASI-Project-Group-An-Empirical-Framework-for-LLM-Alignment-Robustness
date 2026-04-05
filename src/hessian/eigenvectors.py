"""
Hessian eigenvector computation via Lanczos iteration.

Wraps PyHessian to compute top-k eigenvalues and eigenvectors of the
loss Hessian. Supports both full-model and per-layer computation.

References:
    Peng et al. (2025) — adversarial alignment via Hessian eigenvectors
    PyHessian — https://github.com/amirgholami/PyHessian
"""

import torch
import numpy as np
from dataclasses import dataclass
from pyhessian import hessian as pyhessian_module

from src.models.pyhessian_wrapper import (
    HessianModelWrapper,
    CausalLMCriterion,
    DPOCriterion,
)


@dataclass
class HessianResult:
    """Result of Hessian eigenvector computation."""
    eigenvalues: list[float]
    eigenvectors: list[dict[str, torch.Tensor]]  # Each eigenvector is a dict of param_name -> tensor
    layer_index: int | None  # None if full-model
    loss_type: str           # "causal_lm" or "dpo"


def compute_top_eigenvectors(
    model,
    data: tuple[torch.Tensor, torch.Tensor],
    loss_type: str = "causal_lm",
    top_k: int = 20,
    dpo_beta: float = 0.1,
) -> HessianResult:
    """Compute top-k Hessian eigenvalues and eigenvectors for the full model.

    Args:
        model: HuggingFace CausalLM (unwrapped).
        data: Tuple of (input_ids, labels/targets).
        loss_type: "causal_lm" for standard LM loss, "dpo" for DPO loss.
        top_k: Number of top eigenvalues/eigenvectors to compute.
        dpo_beta: DPO beta parameter (only used if loss_type="dpo").

    Returns:
        HessianResult with eigenvalues and eigenvectors.
    """
    wrapped_model = HessianModelWrapper(model)

    if loss_type == "causal_lm":
        criterion = CausalLMCriterion()
    elif loss_type == "dpo":
        criterion = DPOCriterion(beta=dpo_beta)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    hessian_comp = pyhessian_module.hessian(
        wrapped_model,
        criterion,
        data=data,
        cuda=torch.cuda.is_available(),
    )

    eigenvalues, eigenvectors = hessian_comp.eigenvalues(top_n=top_k)

    # Convert eigenvectors from flat list to named param dicts
    named_eigvecs = _flatten_to_named(wrapped_model, eigenvectors)

    return HessianResult(
        eigenvalues=eigenvalues,
        eigenvectors=named_eigvecs,
        layer_index=None,
        loss_type=loss_type,
    )


def compute_top_eigenvectors_per_layer(
    model,
    data: tuple[torch.Tensor, torch.Tensor],
    layer_indices: list[int],
    loss_type: str = "causal_lm",
    top_k: int = 20,
    dpo_beta: float = 0.1,
) -> list[HessianResult]:
    """Compute Hessian eigenvectors for specific layers only.

    More memory-efficient than full-model computation. Only the
    parameters of the target layer are included in the Hessian.

    Args:
        model: HuggingFace CausalLM.
        data: Tuple of (input_ids, labels/targets).
        layer_indices: Which transformer layers to analyze.
        loss_type: Loss function type.
        top_k: Number of top eigenvalues/eigenvectors.
        dpo_beta: DPO beta parameter.

    Returns:
        List of HessianResult, one per layer.
    """
    results = []

    for layer_idx in layer_indices:
        print(f"Computing Hessian for layer {layer_idx}...")

        # Freeze all parameters except target layer
        for name, param in model.named_parameters():
            param.requires_grad = False

        layer_prefix = f"model.layers.{layer_idx}."
        for name, param in model.named_parameters():
            if name.startswith(layer_prefix):
                param.requires_grad = True

        wrapped_model = HessianModelWrapper(model)

        if loss_type == "causal_lm":
            criterion = CausalLMCriterion()
        elif loss_type == "dpo":
            criterion = DPOCriterion(beta=dpo_beta)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        hessian_comp = pyhessian_module.hessian(
            wrapped_model,
            criterion,
            data=data,
            cuda=torch.cuda.is_available(),
        )

        eigenvalues, eigenvectors = hessian_comp.eigenvalues(top_n=top_k)
        named_eigvecs = _flatten_to_named(wrapped_model, eigenvectors)

        results.append(HessianResult(
            eigenvalues=eigenvalues,
            eigenvectors=named_eigvecs,
            layer_index=layer_idx,
            loss_type=loss_type,
        ))

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    return results


def _flatten_to_named(
    model: HessianModelWrapper,
    eigenvectors: list,
) -> list[dict[str, torch.Tensor]]:
    """Convert PyHessian's flat eigenvector format to named parameter dicts.

    PyHessian returns eigenvectors as lists of tensors matching the order
    of model.parameters(). We convert to dicts keyed by parameter name
    for easier downstream analysis.
    """
    param_names = [name for name, _ in model.named_parameters() if _.requires_grad]
    named_eigvecs = []

    for eigvec in eigenvectors:
        named = {}
        for name, vec_part in zip(param_names, eigvec):
            named[name] = vec_part.cpu().detach()
        named_eigvecs.append(named)

    return named_eigvecs
