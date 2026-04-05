"""
Stage 3: Measure RepE Vector Properties

Compute metrics for all extracted reading vectors and compare
safety concepts vs. capability concepts.
"""

import argparse
import json
from pathlib import Path

from src.repe.metrics import compute_metrics_batch, compare_safety_vs_capability
from src.utils.checkpointing import load_reading_vectors
from src.utils.logging import init_run, log_metrics, log_summary, finish


def main(
    repe_dir: str = "outputs/stage2",
    output_dir: str = "outputs/stage3",
):
    repe_dir = Path(repe_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_run(name="stage3-repe-metrics", tags=["stage3", "metrics"])

    # Load all reading vectors
    safety_concepts = ["harmlessness", "honesty", "morality"]
    # TODO: Add capability concepts (factual_recall, reasoning, coding)
    # once capability contrastive pairs are implemented

    all_safety_metrics = []

    for concept in safety_concepts:
        concept_dir = repe_dir / concept
        if not concept_dir.exists():
            print(f"Skipping {concept} — not found at {concept_dir}")
            continue

        vectors = load_reading_vectors(concept_dir)
        metrics = compute_metrics_batch(vectors)
        all_safety_metrics.extend(metrics)

        # Log per-concept summary
        for m in metrics:
            log_metrics({
                f"metrics/{m.concept}_layer{m.layer}_norm": m.l2_norm,
                f"metrics/{m.concept}_layer{m.layer}_evr": m.explained_variance_ratio,
                f"metrics/{m.concept}_layer{m.layer}_eff_dim": m.effective_dimensionality,
            })

        # Print summary
        print(f"\n{concept}:")
        for m in metrics:
            print(f"  Layer {m.layer}: norm={m.l2_norm:.3f}, "
                  f"EVR={m.explained_variance_ratio:.3f}, "
                  f"eff_dim={m.effective_dimensionality}, "
                  f"AUROC={m.auroc or 'N/A'}")

    # Save all metrics
    metrics_data = [{
        "concept": m.concept,
        "layer": m.layer,
        "l2_norm": m.l2_norm,
        "explained_variance_ratio": m.explained_variance_ratio,
        "effective_dimensionality": m.effective_dimensionality,
        "auroc": m.auroc,
    } for m in all_safety_metrics]

    with open(output_dir / "safety_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    # TODO: Compare against capability metrics once available
    # comparison = compare_safety_vs_capability(all_safety_metrics, capability_metrics)

    finish()
    print("\nStage 3 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: RepE Vector Metrics")
    parser.add_argument("--repe-dir", default="outputs/stage2")
    parser.add_argument("--output-dir", default="outputs/stage3")
    args = parser.parse_args()
    main(args.repe_dir, args.output_dir)
