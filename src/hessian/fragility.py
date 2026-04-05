"""
Fragility metrics: measure overlap between RepE directions and Hessian eigenvectors.

Core analysis from Phase 1, Step 1.3:
- Cosine similarity between RepE reading vectors and top Hessian eigenvectors
- Effective rank of the Hessian
- Spectral concentration

The key prediction: safety RepE directions will have higher overlap with
top Hessian eigenvectors than capability RepE directions, meaning safety
knowledge lives in the most fragile (high-curvature) weight subspace.
"""

import torch
import numpy as np
from dataclasses import dataclass

from src.repe.extract import ReadingVector
from src.hessian.eigenvectors import HessianResult


@dataclass
class FragilityScore:
    """Fragility measurement for a concept at a specific layer."""
    concept: str
    layer: int
    # Cosine similarity between RepE vector and each top Hessian eigenvector
    cosine_similarities: list[float]
    # Max overlap with any single Hessian eigenvector
    max_overlap: float
    # Mean overlap across top-k Hessian eigenvectors
    mean_overlap: float
    # Subspace projection: fraction of RepE vector captured by top-k eigenvector span
    subspace_projection: float


def compute_fragility(
    reading_vector: ReadingVector,
    hessian_result: HessianResult,
    model,
) -> FragilityScore:
    """Compute fragility of a concept direction relative to Hessian eigenvectors.

    The reading vector lives in activation space (hidden_size,). To compare
    with Hessian eigenvectors (which live in weight space), we need to
    identify the weight-space directions that most influence the reading
    vector's activation-space direction.

    For now, we use the weight matrices of the target layer directly:
    the reading vector is projected through the layer's weight matrix to
    obtain the corresponding weight-space direction, which is then compared
    against Hessian eigenvectors.

    Args:
        reading_vector: RepE reading vector for a concept at a specific layer.
        hessian_result: Hessian eigenvectors for the same layer.
        model: The model (needed to access weight matrices).

    Returns:
        FragilityScore with overlap measurements.
    """
    layer_idx = reading_vector.layer

    # Get the weight-space representation of the reading vector
    # by projecting through the layer's value projection matrix
    weight_key = f"model.model.layers.{layer_idx}.self_attn.v_proj.weight"
    repe_weight_dir = _project_to_weight_space(
        reading_vector.vector, model, weight_key
    )

    # Compute cosine similarity with each Hessian eigenvector
    cosine_sims = []
    eigvec_components = []

    for eigvec_dict in hessian_result.eigenvectors:
        if weight_key in eigvec_dict:
            eigvec_flat = eigvec_dict[weight_key].flatten().numpy()
            cos_sim = _cosine_similarity(repe_weight_dir, eigvec_flat)
            cosine_sims.append(float(cos_sim))
            eigvec_components.append(eigvec_flat)

    # Subspace projection: how much of the RepE direction lies in the
    # span of the top-k Hessian eigenvectors
    if eigvec_components:
        subspace_proj = _subspace_projection(repe_weight_dir, eigvec_components)
    else:
        subspace_proj = 0.0

    return FragilityScore(
        concept=reading_vector.concept,
        layer=layer_idx,
        cosine_similarities=cosine_sims,
        max_overlap=max(cosine_sims) if cosine_sims else 0.0,
        mean_overlap=float(np.mean(cosine_sims)) if cosine_sims else 0.0,
        subspace_projection=subspace_proj,
    )


def compute_effective_rank(eigenvalues: list[float]) -> float:
    """Compute effective rank of the Hessian from its eigenvalues.

    Effective rank = exp(entropy of normalized eigenvalue distribution).
    Lower effective rank means the curvature is concentrated in fewer
    directions — the loss landscape is more "sharp" and fragile.
    """
    eigenvalues = np.array(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 0.0

    # Normalize to probability distribution
    p = eigenvalues / eigenvalues.sum()
    # Shannon entropy
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def compute_spectral_concentration(eigenvalues: list[float], top_k: int = 5) -> float:
    """Fraction of total spectral mass in top-k eigenvalues.

    High concentration = fragile (few directions dominate curvature).
    """
    eigenvalues = np.array(sorted(eigenvalues, reverse=True))
    total = eigenvalues.sum()
    if total == 0:
        return 0.0
    return float(eigenvalues[:top_k].sum() / total)


def _project_to_weight_space(
    activation_vector: np.ndarray,
    model,
    weight_key: str,
) -> np.ndarray:
    """Project an activation-space direction to weight-space.

    Uses the transpose of the weight matrix: if W maps weight-space to
    activation-space, then W^T maps activation-space back to weight-space.
    The result is flattened for comparison with Hessian eigenvectors.
    """
    # Navigate to the weight tensor
    parts = weight_key.split(".")
    obj = model
    for part in parts:
        obj = getattr(obj, part)

    weight = obj.data.cpu().float().numpy()  # (out_features, in_features)
    # Project: weight_dir = W^T @ activation_vector
    weight_dir = weight.T @ activation_vector
    return weight_dir.flatten()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.abs(np.dot(a, b)) / (norm_a * norm_b))


def _subspace_projection(vector: np.ndarray, basis_vectors: list[np.ndarray]) -> float:
    """Fraction of vector's norm captured by projection onto subspace.

    Constructs orthonormal basis from basis_vectors via QR decomposition,
    then computes ||proj||^2 / ||vector||^2.
    """
    basis = np.stack(basis_vectors, axis=1)  # (dim, k)
    Q, _ = np.linalg.qr(basis)

    projection = Q @ (Q.T @ vector)
    return float(np.linalg.norm(projection) ** 2 / (np.linalg.norm(vector) ** 2 + 1e-12))
