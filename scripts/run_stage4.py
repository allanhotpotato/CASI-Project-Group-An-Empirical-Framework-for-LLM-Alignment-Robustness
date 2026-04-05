"""
Stage 4: Hessian Fragility Analysis

Compare fragility of safety-relevant vs. capability-relevant weight directions
using Hessian eigenvectors computed via Lanczos iteration.
"""

import argparse
import json
import yaml
from pathlib import Path

from src.models.loading import load_model_and_tokenizer
from src.models.pyhessian_wrapper import prepare_hessian_data
from src.hessian.eigenvectors import compute_top_eigenvectors_per_layer
from src.hessian.fragility import (
    compute_fragility,
    compute_effective_rank,
    compute_spectral_concentration,
)
from src.utils.checkpointing import (
    load_reading_vectors,
    save_hessian_result,
    load_hessian_result,
)
from src.utils.logging import init_run, log_metrics, finish


def main(
    model_config_path: str = "configs/models/llama3_8b.yaml",
    model_checkpoint: str | None = None,
    repe_dir: str = "outputs/stage2",
    output_dir: str = "outputs/stage4",
    layer_indices: list[int] | None = None,
    top_k: int = 20,
    hessian_batch_size: int = 128,
    cache_dir: str | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_run(name="stage4-hessian-fragility", tags=["stage4", "hessian"])

    if layer_indices is None:
        layer_indices = [0, 4, 8, 12, 16, 20, 24, 28, 31]

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        config_path=model_config_path,
        from_checkpoint=model_checkpoint,
        cache_dir=cache_dir,
    )

    # Prepare Hessian data
    from src.data.datasets import load_preference_dataset
    data_raw = load_preference_dataset(max_samples=hessian_batch_size)
    texts = [ex["chosen"] for ex in data_raw]
    input_ids, labels = prepare_hessian_data(tokenizer, texts)

    # ---- Safety-task Hessian ----
    print("Computing safety-task Hessian eigenvectors...")
    safety_hessians = compute_top_eigenvectors_per_layer(
        model, data=(input_ids, labels),
        layer_indices=layer_indices,
        loss_type="causal_lm",
        top_k=top_k,
    )
    for h in safety_hessians:
        save_hessian_result(h, output_dir / f"safety_hessian_layer{h.layer_index}")

    # ---- Capability-task Hessian ----
    # Use general text (not preference data) for capability Hessian
    from src.evaluation.capability import evaluate_perplexity
    from datasets import load_dataset
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    cap_texts = [t for t in wiki["text"] if len(t.strip()) > 50][:hessian_batch_size]
    cap_input_ids, cap_labels = prepare_hessian_data(tokenizer, cap_texts)

    print("Computing capability-task Hessian eigenvectors...")
    capability_hessians = compute_top_eigenvectors_per_layer(
        model, data=(cap_input_ids, cap_labels),
        layer_indices=layer_indices,
        loss_type="causal_lm",
        top_k=top_k,
    )
    for h in capability_hessians:
        save_hessian_result(h, output_dir / f"capability_hessian_layer{h.layer_index}")

    # ---- Fragility analysis ----
    print("\nComputing fragility scores...")
    repe_dir = Path(repe_dir)
    results = []

    for concept in ["harmlessness", "honesty", "morality"]:
        concept_dir = repe_dir / concept
        if not concept_dir.exists():
            continue

        vectors = load_reading_vectors(concept_dir)

        for rv in vectors:
            if rv.layer not in layer_indices:
                continue

            # Find matching Hessian result
            matching_hessian = [h for h in safety_hessians if h.layer_index == rv.layer]
            if not matching_hessian:
                continue

            score = compute_fragility(rv, matching_hessian[0], model)
            results.append({
                "concept": score.concept,
                "layer": score.layer,
                "max_overlap": score.max_overlap,
                "mean_overlap": score.mean_overlap,
                "subspace_projection": score.subspace_projection,
            })

            log_metrics({
                f"fragility/{score.concept}_layer{score.layer}_max_overlap": score.max_overlap,
                f"fragility/{score.concept}_layer{score.layer}_subspace_proj": score.subspace_projection,
            })

    # Log Hessian properties
    for h in safety_hessians:
        eff_rank = compute_effective_rank(h.eigenvalues)
        spec_conc = compute_spectral_concentration(h.eigenvalues)
        log_metrics({
            f"hessian/safety_layer{h.layer_index}_eff_rank": eff_rank,
            f"hessian/safety_layer{h.layer_index}_spectral_conc": spec_conc,
        })
        print(f"Safety Hessian layer {h.layer_index}: "
              f"eff_rank={eff_rank:.1f}, spectral_conc={spec_conc:.3f}")

    for h in capability_hessians:
        eff_rank = compute_effective_rank(h.eigenvalues)
        spec_conc = compute_spectral_concentration(h.eigenvalues)
        log_metrics({
            f"hessian/capability_layer{h.layer_index}_eff_rank": eff_rank,
            f"hessian/capability_layer{h.layer_index}_spectral_conc": spec_conc,
        })

    # Save all results
    with open(output_dir / "fragility_results.json", "w") as f:
        json.dump(results, f, indent=2)

    finish()
    print("\nStage 4 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Hessian Fragility Analysis")
    parser.add_argument("--model-config", default="configs/models/llama3_8b.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--repe-dir", default="outputs/stage2")
    parser.add_argument("--output-dir", default="outputs/stage4")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--hessian-batch-size", type=int, default=128)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    main(args.model_config, args.checkpoint, args.repe_dir, args.output_dir,
         top_k=args.top_k, hessian_batch_size=args.hessian_batch_size,
         cache_dir=args.cache_dir)
