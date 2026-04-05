"""
Stage 2: Extract LAT Vectors for Safety Concepts

Extract RepE reading vectors for harmlessness, honesty, and morality
from both the base model and safety-trained models.
"""

import argparse
import yaml
from pathlib import Path

from src.models.loading import load_model_and_tokenizer
from src.data.contrastive_pairs import generate_all_safety_pairs
from src.repe.extract import extract_reading_vectors, validate_reading_vector_auroc
from src.utils.logging import init_run, log_metrics, finish
from src.utils.checkpointing import save_reading_vectors


def main(
    model_config_path: str,
    model_checkpoint: str | None = None,
    output_dir: str = "outputs/stage2",
    cache_dir: str | None = None,
    run_name: str = "stage2-repe-extraction",
):
    output_dir = Path(output_dir)

    init_run(name=run_name, tags=["stage2", "repe"])

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        config_path=model_config_path,
        from_checkpoint=model_checkpoint,
        cache_dir=cache_dir,
    )

    # Generate contrastive pairs
    all_pairs = generate_all_safety_pairs()

    for concept, pairs in all_pairs.items():
        print(f"\nExtracting reading vectors for: {concept}")
        print(f"  Using {len(pairs)} contrastive pairs")

        # Split into train/held-out for AUROC validation
        split = int(0.8 * len(pairs))
        train_pairs = pairs[:split]
        held_out_pairs = pairs[split:]

        # Extract reading vectors
        reading_vectors = extract_reading_vectors(
            model, tokenizer, train_pairs,
            batch_size=4,
        )

        # Validate AUROC on held-out data
        for rv in reading_vectors:
            auroc = validate_reading_vector_auroc(
                model, tokenizer, rv, held_out_pairs,
            )
            rv.auroc = auroc

        # Log best layer AUROC
        best = max(reading_vectors, key=lambda v: v.auroc or 0)
        print(f"  Best AUROC: {best.auroc:.3f} at layer {best.layer}")
        log_metrics({
            f"repe/{concept}_best_auroc": best.auroc,
            f"repe/{concept}_best_layer": best.layer,
        })

        # Save
        save_reading_vectors(reading_vectors, output_dir / concept)

    finish()
    print("\nStage 2 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: LAT Vector Extraction")
    parser.add_argument("--model-config", default="configs/models/llama3_8b.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="outputs/stage2")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    main(args.model_config, args.checkpoint, args.output_dir, args.cache_dir)
