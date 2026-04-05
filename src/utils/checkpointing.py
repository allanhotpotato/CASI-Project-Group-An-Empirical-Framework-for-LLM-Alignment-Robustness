"""
Utilities for saving and loading intermediate results.

RepE vectors, Hessian eigenvectors, and delta_theta snapshots are
expensive to compute. This module handles serialization so they
don't need to be recomputed.
"""

import torch
import numpy as np
import json
from pathlib import Path
from dataclasses import asdict

from src.repe.extract import ReadingVector
from src.hessian.eigenvectors import HessianResult


def save_reading_vectors(
    vectors: list[ReadingVector],
    output_dir: str | Path,
) -> None:
    """Save extracted reading vectors to disk.

    Saves numpy arrays and metadata separately for easy loading.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for i, rv in enumerate(vectors):
        np.save(output_dir / f"vector_{rv.concept}_layer{rv.layer}.npy", rv.vector)
        np.save(output_dir / f"variances_{rv.concept}_layer{rv.layer}.npy", rv.all_variances)
        metadata.append({
            "concept": rv.concept,
            "layer": rv.layer,
            "explained_variance": rv.explained_variance,
            "auroc": rv.auroc,
        })

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_reading_vectors(input_dir: str | Path) -> list[ReadingVector]:
    """Load reading vectors from disk."""
    input_dir = Path(input_dir)

    with open(input_dir / "metadata.json") as f:
        metadata = json.load(f)

    vectors = []
    for m in metadata:
        vector = np.load(input_dir / f"vector_{m['concept']}_layer{m['layer']}.npy")
        variances = np.load(input_dir / f"variances_{m['concept']}_layer{m['layer']}.npy")
        vectors.append(ReadingVector(
            concept=m["concept"],
            layer=m["layer"],
            vector=vector,
            explained_variance=m["explained_variance"],
            all_variances=variances,
            auroc=m.get("auroc"),
        ))

    return vectors


def save_hessian_result(
    result: HessianResult,
    output_dir: str | Path,
) -> None:
    """Save Hessian computation results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save eigenvalues
    np.save(output_dir / "eigenvalues.npy", np.array(result.eigenvalues))

    # Save eigenvectors
    for i, eigvec_dict in enumerate(result.eigenvectors):
        torch.save(eigvec_dict, output_dir / f"eigenvector_{i}.pt")

    # Save metadata
    meta = {
        "layer_index": result.layer_index,
        "loss_type": result.loss_type,
        "num_eigenvectors": len(result.eigenvectors),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_hessian_result(input_dir: str | Path) -> HessianResult:
    """Load Hessian results from disk."""
    input_dir = Path(input_dir)

    with open(input_dir / "metadata.json") as f:
        meta = json.load(f)

    eigenvalues = np.load(input_dir / "eigenvalues.npy").tolist()
    eigenvectors = []
    for i in range(meta["num_eigenvectors"]):
        eigvec = torch.load(input_dir / f"eigenvector_{i}.pt", map_location="cpu", weights_only=True)
        eigenvectors.append(eigvec)

    return HessianResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        layer_index=meta["layer_index"],
        loss_type=meta["loss_type"],
    )


def save_delta_theta(
    delta: dict[str, torch.Tensor],
    output_path: str | Path,
) -> None:
    """Save a weight delta snapshot."""
    torch.save(delta, str(output_path))


def load_delta_theta(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a weight delta snapshot."""
    return torch.load(str(path), map_location="cpu", weights_only=True)
