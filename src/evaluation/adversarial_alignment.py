"""
Adversarial alignment analysis.

Measures whether fine-tuning weight updates preferentially align with
the top Hessian eigenvectors of the safety loss vs. capability loss.

This is the core measurement of Stage 5: if fine-tuning delta_theta
projects more strongly onto safety Hessian eigenvectors than capability
Hessian eigenvectors, it confirms the adversarial alignment mechanism.

References:
    Peng et al. (2025) — adversarial alignment in continual learning
"""

import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from src.hessian.eigenvectors import HessianResult


@dataclass
class AdversarialAlignmentScore:
    """Adversarial alignment measurement at a single checkpoint."""
    checkpoint_step: int
    # Cosine similarity between delta_theta and safety Hessian eigenvectors
    safety_alignment_scores: list[float]
    safety_mean_alignment: float
    # Cosine similarity between delta_theta and capability Hessian eigenvectors
    capability_alignment_scores: list[float]
    capability_mean_alignment: float
    # Random baseline
    random_baseline_mean: float
    random_baseline_std: float


def compute_adversarial_alignment(
    delta_theta: dict[str, torch.Tensor],
    safety_hessian: HessianResult,
    capability_hessian: HessianResult,
    num_random_samples: int = 100,
) -> AdversarialAlignmentScore:
    """Compute adversarial alignment scores for a single checkpoint.

    Args:
        delta_theta: Weight update dict {param_name: delta_tensor}.
        safety_hessian: Top Hessian eigenvectors of the safety loss.
        capability_hessian: Top Hessian eigenvectors of the capability loss.
        num_random_samples: Number of random vectors for baseline.

    Returns:
        AdversarialAlignmentScore with all measurements.
    """
    # Flatten delta_theta to a single vector
    delta_flat = _flatten_params(delta_theta)

    # Compute alignment with safety Hessian eigenvectors
    safety_scores = []
    for eigvec_dict in safety_hessian.eigenvectors:
        eigvec_flat = _flatten_params(eigvec_dict)
        # Ensure same dimensionality (only use shared parameters)
        shared_flat_delta, shared_flat_eig = _align_param_vectors(delta_theta, eigvec_dict)
        if shared_flat_delta is not None:
            cos_sim = _cosine_similarity(shared_flat_delta, shared_flat_eig)
            safety_scores.append(cos_sim)

    # Compute alignment with capability Hessian eigenvectors
    capability_scores = []
    for eigvec_dict in capability_hessian.eigenvectors:
        shared_flat_delta, shared_flat_eig = _align_param_vectors(delta_theta, eigvec_dict)
        if shared_flat_delta is not None:
            cos_sim = _cosine_similarity(shared_flat_delta, shared_flat_eig)
            capability_scores.append(cos_sim)

    # Random baseline: cosine similarity with random vectors of same norm
    random_scores = []
    dim = len(delta_flat)
    delta_norm = np.linalg.norm(delta_flat)
    for _ in range(num_random_samples):
        random_vec = np.random.randn(dim)
        random_vec = random_vec / np.linalg.norm(random_vec) * delta_norm
        random_scores.append(_cosine_similarity(delta_flat, random_vec))

    return AdversarialAlignmentScore(
        checkpoint_step=0,  # Set by caller
        safety_alignment_scores=safety_scores,
        safety_mean_alignment=float(np.mean(safety_scores)) if safety_scores else 0.0,
        capability_alignment_scores=capability_scores,
        capability_mean_alignment=float(np.mean(capability_scores)) if capability_scores else 0.0,
        random_baseline_mean=float(np.mean(random_scores)),
        random_baseline_std=float(np.std(random_scores)),
    )


def analyze_checkpoints(
    checkpoint_dir: str | Path,
    initial_weights_path: str | Path,
    safety_hessian: HessianResult,
    capability_hessian: HessianResult,
    num_random_samples: int = 100,
) -> list[AdversarialAlignmentScore]:
    """Analyze adversarial alignment across all fine-tuning checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories.
        initial_weights_path: Path to initial_weights.pt (pre-finetune).
        safety_hessian: Safety-task Hessian eigenvectors.
        capability_hessian: Capability-task Hessian eigenvectors.
        num_random_samples: Random baseline samples per checkpoint.

    Returns:
        List of AdversarialAlignmentScore, one per checkpoint.
    """
    from src.models.loading import compute_delta_theta, load_model_and_tokenizer

    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=_extract_step)

    results = []
    for ckpt_path in checkpoints:
        step = _extract_step(ckpt_path)
        print(f"Analyzing checkpoint at step {step}...")

        # Load checkpoint model and compute delta_theta
        # Note: loading full model per checkpoint is expensive.
        # Alternative: save delta_theta at each checkpoint during training.
        delta_theta = compute_delta_theta_from_checkpoint(
            ckpt_path, initial_weights_path
        )

        score = compute_adversarial_alignment(
            delta_theta, safety_hessian, capability_hessian, num_random_samples
        )
        score.checkpoint_step = step
        results.append(score)

    return results


def compute_delta_theta_from_checkpoint(
    checkpoint_path: str | Path,
    initial_weights_path: str | Path,
) -> dict[str, torch.Tensor]:
    """Load a checkpoint and compute weight delta from initial weights."""
    initial = torch.load(
        Path(initial_weights_path) / "initial_weights.pt",
        map_location="cpu",
        weights_only=True,
    )

    # Load checkpoint weights
    import safetensors.torch
    ckpt_path = Path(checkpoint_path)
    safetensor_files = list(ckpt_path.glob("*.safetensors"))
    if safetensor_files:
        current = {}
        for f in safetensor_files:
            current.update(safetensors.torch.load_file(str(f)))
    else:
        current = torch.load(
            ckpt_path / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )

    delta = {}
    for name in initial:
        if name in current:
            delta[name] = current[name].float() - initial[name].float()

    return delta


def _flatten_params(param_dict: dict[str, torch.Tensor]) -> np.ndarray:
    """Flatten a parameter dict to a single numpy vector."""
    parts = []
    for name in sorted(param_dict.keys()):
        parts.append(param_dict[name].flatten().float().numpy())
    return np.concatenate(parts)


def _align_param_vectors(
    dict_a: dict[str, torch.Tensor],
    dict_b: dict[str, torch.Tensor],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract matching parameters from two dicts and flatten."""
    shared_keys = sorted(set(dict_a.keys()) & set(dict_b.keys()))
    if not shared_keys:
        return None, None

    parts_a, parts_b = [], []
    for key in shared_keys:
        parts_a.append(dict_a[key].flatten().float().numpy())
        parts_b.append(dict_b[key].flatten().float().numpy())

    return np.concatenate(parts_a), np.concatenate(parts_b)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute cosine similarity between two vectors."""
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.abs(np.dot(a, b)) / (norm_a * norm_b))


def _extract_step(path: Path) -> int:
    """Extract step number from checkpoint directory name."""
    return int(path.name.split("-")[-1])
