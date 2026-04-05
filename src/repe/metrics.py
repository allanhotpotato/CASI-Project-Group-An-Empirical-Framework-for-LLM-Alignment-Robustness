"""
Metrics for characterizing RepE reading vectors.

Measures from Phase 1, Step 1.2 of the methodology:
- L2 norm
- Explained variance ratio
- Effective dimensionality
- Layer-wise AUROC distribution
"""

import numpy as np
from dataclasses import dataclass

from src.repe.extract import ReadingVector


@dataclass
class VectorMetrics:
    """Computed metrics for a single reading vector."""
    concept: str
    layer: int
    l2_norm: float
    explained_variance_ratio: float  # How much the 1st PC captures
    effective_dimensionality: int    # PCs needed for 95% variance
    auroc: float | None


def compute_metrics(rv: ReadingVector) -> VectorMetrics:
    """Compute all metrics for a single reading vector.

    Args:
        rv: A ReadingVector from LAT extraction.

    Returns:
        VectorMetrics with all computed values.
    """
    return VectorMetrics(
        concept=rv.concept,
        layer=rv.layer,
        l2_norm=float(np.linalg.norm(rv.vector)),
        explained_variance_ratio=float(rv.explained_variance),
        effective_dimensionality=_effective_dim(rv.all_variances, threshold=0.95),
        auroc=rv.auroc,
    )


def _effective_dim(variance_ratios: np.ndarray, threshold: float = 0.95) -> int:
    """Number of PCs needed to explain `threshold` fraction of variance.

    Lower effective dimensionality = concept is stored in a more compressed,
    potentially more fragile subspace.
    """
    cumulative = np.cumsum(variance_ratios)
    return int(np.searchsorted(cumulative, threshold) + 1)


def compute_metrics_batch(
    reading_vectors: list[ReadingVector],
) -> list[VectorMetrics]:
    """Compute metrics for all reading vectors."""
    return [compute_metrics(rv) for rv in reading_vectors]


def compare_safety_vs_capability(
    safety_metrics: list[VectorMetrics],
    capability_metrics: list[VectorMetrics],
) -> dict:
    """Compare aggregate metrics between safety and capability vectors.

    The core prediction: safety concepts should have lower effective
    dimensionality than capability concepts, indicating more compressed
    (and thus more fragile) storage.

    Returns:
        Dict with comparison statistics.
    """
    safety_dims = [m.effective_dimensionality for m in safety_metrics]
    capability_dims = [m.effective_dimensionality for m in capability_metrics]

    safety_evr = [m.explained_variance_ratio for m in safety_metrics]
    capability_evr = [m.explained_variance_ratio for m in capability_metrics]

    return {
        "safety_mean_eff_dim": float(np.mean(safety_dims)),
        "capability_mean_eff_dim": float(np.mean(capability_dims)),
        "safety_mean_explained_var": float(np.mean(safety_evr)),
        "capability_mean_explained_var": float(np.mean(capability_evr)),
        "safety_dims_by_layer": safety_dims,
        "capability_dims_by_layer": capability_dims,
    }
