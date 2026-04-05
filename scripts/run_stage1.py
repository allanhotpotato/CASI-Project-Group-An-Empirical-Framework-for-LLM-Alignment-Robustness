"""
Stage 1: Reproduce Qi et al. + Validate Adversarial Alignment in LLMs

Goals:
1. Apply DPO safety training to base model
2. Evaluate safety pre/post fine-tuning to confirm degradation
3. Compute top-k Hessian eigenvectors of the DPO safety loss
4. Fine-tune on benign Alpaca data
5. Measure adversarial alignment: does delta_theta align with safety Hessian eigenvectors?

This is the proof-of-concept stage. If adversarial alignment is confirmed,
the rest of the pipeline is theoretically justified.
"""

import argparse
import yaml
from pathlib import Path

from src.models.loading import load_model_and_tokenizer, save_initial_weights
from src.models.pyhessian_wrapper import prepare_hessian_data
from src.training.dpo import run_dpo_training
from src.training.finetune import run_finetune
from src.hessian.eigenvectors import compute_top_eigenvectors_per_layer
from src.evaluation.safety import evaluate_safety_full
from src.evaluation.adversarial_alignment import analyze_checkpoints
from src.utils.logging import init_run, log_metrics, log_summary, finish
from src.utils.checkpointing import save_hessian_result


def main(config_path: str, cache_dir: str | None = None):
    # Load experiment config
    with open(config_path) as f:
        config = yaml.safe_load(f)["experiment"]

    model_config = config["model_config"]
    output_base = Path(config["steps"]["safety_training"]["output_dir"])

    init_run(
        name=config["name"],
        config=config,
        tags=["stage1", "adversarial-alignment"],
    )

    # ---- Step 1: DPO safety training ----
    print("=" * 60)
    print("Step 1: DPO safety training")
    print("=" * 60)
    dpo_model_path = run_dpo_training(
        model_config_path=model_config,
        training_config_path="configs/training/dpo.yaml",
        output_dir=str(output_base),
        cache_dir=cache_dir,
    )

    # ---- Step 2: Evaluate safety post-alignment ----
    print("=" * 60)
    print("Step 2: Safety evaluation (post-alignment)")
    print("=" * 60)
    model, tokenizer = load_model_and_tokenizer(
        config_path=model_config,
        from_checkpoint=dpo_model_path,
        cache_dir=cache_dir,
    )
    safety_pre = evaluate_safety_full(model, tokenizer)
    log_metrics({
        "safety/refusal_rate_post_dpo": safety_pre.refusal_rate,
    })
    print(f"Refusal rate after DPO: {safety_pre.refusal_rate:.2%}")

    # ---- Step 3: Compute Hessian eigenvectors ----
    print("=" * 60)
    print("Step 3: Computing Hessian eigenvectors of safety loss")
    print("=" * 60)
    hessian_cfg = config["steps"]["hessian"]
    layer_indices = hessian_cfg["layer_indices"]
    top_k = hessian_cfg["top_k"]

    # Prepare data for Hessian computation
    # Use a small subset of the DPO training data
    from src.data.datasets import load_preference_dataset
    hessian_data_raw = load_preference_dataset(max_samples=hessian_cfg["hessian_batch_size"])
    # For causal LM Hessian, use the chosen responses
    hessian_texts = [ex["chosen"] for ex in hessian_data_raw]
    input_ids, labels = prepare_hessian_data(tokenizer, hessian_texts)

    safety_hessian_results = compute_top_eigenvectors_per_layer(
        model,
        data=(input_ids, labels),
        layer_indices=layer_indices,
        loss_type="causal_lm",  # Using causal LM loss as proxy for now
        top_k=top_k,
    )

    # Save Hessian results
    hessian_dir = Path(hessian_cfg["output_dir"])
    for result in safety_hessian_results:
        save_hessian_result(
            result,
            hessian_dir / f"safety_layer{result.layer_index}",
        )
    print(f"Saved Hessian results for {len(safety_hessian_results)} layers")

    # ---- Step 4: Fine-tune on benign Alpaca data ----
    print("=" * 60)
    print("Step 4: Fine-tuning on Alpaca (safety erosion)")
    print("=" * 60)
    ft_cfg = config["steps"]["finetune"]
    ft_model_path = run_finetune(
        model_path=dpo_model_path,
        model_config_path=model_config,
        training_config_path="configs/training/finetune.yaml",
        output_dir=ft_cfg["output_dir"],
        cache_dir=cache_dir,
    )

    # ---- Step 5: Evaluate safety post-fine-tuning ----
    print("=" * 60)
    print("Step 5: Safety evaluation (post-fine-tuning)")
    print("=" * 60)
    ft_model, ft_tokenizer = load_model_and_tokenizer(
        config_path=model_config,
        from_checkpoint=ft_model_path,
        cache_dir=cache_dir,
    )
    safety_post = evaluate_safety_full(ft_model, ft_tokenizer)
    log_metrics({
        "safety/refusal_rate_post_finetune": safety_post.refusal_rate,
    })
    print(f"Refusal rate after fine-tuning: {safety_post.refusal_rate:.2%}")
    print(f"Safety degradation: {safety_pre.refusal_rate - safety_post.refusal_rate:.2%}")

    # ---- Step 6: Adversarial alignment analysis ----
    print("=" * 60)
    print("Step 6: Adversarial alignment analysis")
    print("=" * 60)
    aa_cfg = config["steps"]["adversarial_alignment"]

    # For now, use a single layer's Hessian result for the analysis
    # TODO: aggregate across layers
    if safety_hessian_results:
        # Compute capability Hessian for comparison
        print("Computing capability Hessian for comparison...")
        capability_hessian_results = compute_top_eigenvectors_per_layer(
            model,
            data=(input_ids, labels),
            layer_indices=[safety_hessian_results[0].layer_index],
            loss_type="causal_lm",
            top_k=top_k,
        )

        alignment_scores = analyze_checkpoints(
            checkpoint_dir=Path(ft_cfg["output_dir"]) / "checkpoints",
            initial_weights_path=ft_cfg["output_dir"],
            safety_hessian=safety_hessian_results[0],
            capability_hessian=capability_hessian_results[0],
            num_random_samples=aa_cfg["random_baseline_samples"],
        )

        for score in alignment_scores:
            log_metrics({
                "adversarial/safety_mean_alignment": score.safety_mean_alignment,
                "adversarial/capability_mean_alignment": score.capability_mean_alignment,
                "adversarial/random_baseline": score.random_baseline_mean,
            }, step=score.checkpoint_step)

        print("\nAdversarial Alignment Summary:")
        for score in alignment_scores:
            print(f"  Step {score.checkpoint_step}: "
                  f"safety={score.safety_mean_alignment:.4f}, "
                  f"capability={score.capability_mean_alignment:.4f}, "
                  f"random={score.random_baseline_mean:.4f}")

    log_summary({
        "refusal_rate_post_dpo": safety_pre.refusal_rate,
        "refusal_rate_post_finetune": safety_post.refusal_rate,
    })

    finish()
    print("\nStage 1 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Adversarial Alignment Validation")
    parser.add_argument("--config", default="configs/experiments/stage1.yaml")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    args = parser.parse_args()
    main(args.config, args.cache_dir)
