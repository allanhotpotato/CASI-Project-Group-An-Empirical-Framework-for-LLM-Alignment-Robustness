"""
SVD-based fragile subspace identification.

Approximation from Peng et al. (2025): the Hessian's top eigenvectors
are determined by the low-rank structure of weight matrices through the
network's Jacobian. The fragile subspace at layer i is the intersection
of forward signal subspace (layers below) and backward signal subspace
(layers above).

This is cheaper than full Hessian computation and used for:
1. Initial exploration across all layers
2. Validation against Lanczos results on a subset of layers
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class SVDResult:
    """SVD analysis result for a single weight matrix."""
    layer: int
    weight_name: str
    singular_values: np.ndarray
    left_singular_vectors: np.ndarray   # U: (out, k)
    right_singular_vectors: np.ndarray  # V^T: (k, in)
    effective_rank: int                 # Rank for 95% spectral energy
    spectral_energy_ratio: float        # Top-r / total spectral energy


@dataclass
class FragileSubspace:
    """Identified fragile subspace at a layer."""
    layer: int
    forward_basis: np.ndarray    # Basis vectors from forward signal subspace
    backward_basis: np.ndarray   # Basis vectors from backward signal subspace
    fragile_basis: np.ndarray    # Intersection basis (the fragile directions)
    fragile_dim: int             # Dimensionality of fragile subspace


def compute_weight_svd(
    model,
    layer_idx: int,
    weight_names: list[str] | None = None,
) -> list[SVDResult]:
    """Compute SVD of weight matrices at a given layer.

    Args:
        model: HuggingFace model.
        layer_idx: Transformer layer index.
        weight_names: Which weight matrices to analyze. Defaults to all
            attention and MLP projections.

    Returns:
        List of SVDResult for each weight matrix.
    """
    if weight_names is None:
        weight_names = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]

    layer = model.model.layers[layer_idx]
    results = []

    for wname in weight_names:
        # Navigate to weight tensor
        parts = wname.split(".")
        obj = layer
        for part in parts:
            obj = getattr(obj, part)

        W = obj.data.cpu().float().numpy()
        U, S, Vt = np.linalg.svd(W, full_matrices=False)

        # Effective rank: singular values for 95% of spectral energy
        energy = S ** 2
        cumulative = np.cumsum(energy) / energy.sum()
        eff_rank = int(np.searchsorted(cumulative, 0.95) + 1)

        results.append(SVDResult(
            layer=layer_idx,
            weight_name=wname,
            singular_values=S,
            left_singular_vectors=U[:, :eff_rank],
            right_singular_vectors=Vt[:eff_rank, :],
            effective_rank=eff_rank,
            spectral_energy_ratio=float(cumulative[eff_rank - 1]),
        ))

    return results


def identify_fragile_subspace(
    model,
    layer_idx: int,
    num_context_layers: int = 2,
) -> FragileSubspace:
    """Identify the fragile subspace at a layer using surrounding weight SVDs.

    Following Peng et al.: the fragile directions at layer i are those that
    align simultaneously with the forward signal subspace (from layers below)
    and the backward signal subspace (from layers above).

    Args:
        model: HuggingFace model.
        layer_idx: Target layer.
        num_context_layers: How many layers above/below to consider.

    Returns:
        FragileSubspace with the identified fragile directions.
    """
    num_layers = model.config.num_hidden_layers

    # Collect forward signal subspace from layers below
    forward_bases = []
    for i in range(max(0, layer_idx - num_context_layers), layer_idx):
        svd_results = compute_weight_svd(model, i)
        for svd in svd_results:
            forward_bases.append(svd.right_singular_vectors.T)  # (in, r)

    # Collect backward signal subspace from layers above
    backward_bases = []
    for i in range(layer_idx + 1, min(num_layers, layer_idx + num_context_layers + 1)):
        svd_results = compute_weight_svd(model, i)
        for svd in svd_results:
            backward_bases.append(svd.left_singular_vectors)  # (out, r)

    if not forward_bases or not backward_bases:
        # Edge case: first or last layers
        return FragileSubspace(
            layer=layer_idx,
            forward_basis=np.array([]),
            backward_basis=np.array([]),
            fragile_basis=np.array([]),
            fragile_dim=0,
        )

    # Concatenate and orthogonalize each subspace
    forward_combined = np.concatenate(forward_bases, axis=1)
    Q_fwd, _ = np.linalg.qr(forward_combined)

    backward_combined = np.concatenate(backward_bases, axis=1)
    Q_bwd, _ = np.linalg.qr(backward_combined)

    # Find intersection via SVD of the product Q_fwd^T @ Q_bwd
    # Singular values near 1.0 indicate shared directions
    cross = Q_fwd.T @ Q_bwd
    U_cross, S_cross, Vt_cross = np.linalg.svd(cross, full_matrices=False)

    # Directions with singular value > 0.5 are in the intersection
    threshold = 0.5
    mask = S_cross > threshold
    fragile_dirs = Q_fwd @ U_cross[:, mask]

    return FragileSubspace(
        layer=layer_idx,
        forward_basis=Q_fwd,
        backward_basis=Q_bwd,
        fragile_basis=fragile_dirs,
        fragile_dim=int(mask.sum()),
    )
