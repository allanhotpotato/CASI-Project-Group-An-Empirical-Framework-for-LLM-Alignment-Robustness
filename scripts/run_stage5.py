"""
Stage 5: Test Whether Fine-Tuning Preferentially Attacks Safety

Analyze fine-tuning checkpoints to show that weight updates
align more with safety Hessian eigenvectors than capability ones.
"""

import argparse
import json
from pathlib import Path

from src.evaluation.adversarial_alignment import analyze_checkpoints
from src.evaluation.safety import evaluate_refusal_rate
from src.models.loading import load_model_and_tokenizer
from src.utils.checkpointing import load_hessian_result
from src.utils.logging import init_run, log_metrics, finish


def main(
    model_config_path: str = "configs/models/llama3_8b.yaml",
    finetune_dir: str = "outputs/stage1/alpaca_finetuned",
    hessian_dir: str = "outputs/stage4",
    output_dir: str = "outputs/stage5",
    layer_index: int = 16,
    cache_dir: str | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_run(name="stage5-adversarial-attack-analysis", tags=["stage5"])

    # Load Hessian results
    safety_hessian = load_hessian_result(
        Path(hessian_dir) / f"safety_hessian_layer{layer_index}"
    )
    capability_hessian = load_hessian_result(
        Path(hessian_dir) / f"capability_hessian_layer{layer_index}"
    )

    # Analyze adversarial alignment across checkpoints
    print("Analyzing adversarial alignment across fine-tuning checkpoints...")
    scores = analyze_checkpoints(
        checkpoint_dir=Path(finetune_dir) / "checkpoints",
        initial_weights_path=finetune_dir,
        safety_hessian=safety_hessian,
        capability_hessian=capability_hessian,
    )

    # Also evaluate safety at each checkpoint
    checkpoint_dir = Path(finetune_dir) / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1]))

    print("\nEvaluating safety at each checkpoint...")
    safety_scores = []
    for ckpt_path in checkpoints:
        step = int(ckpt_path.name.split("-")[-1])
        model, tokenizer = load_model_and_tokenizer(
            config_path=model_config_path,
            from_checkpoint=str(ckpt_path),
            cache_dir=cache_dir,
        )
        refusal = evaluate_refusal_rate(model, tokenizer)
        safety_scores.append({"step": step, "refusal_rate": refusal})
        log_metrics({"safety/refusal_rate": refusal}, step=step)
        del model  # Free memory

    # Combine and save results
    results = {
        "adversarial_alignment": [
            {
                "step": s.checkpoint_step,
                "safety_mean_alignment": s.safety_mean_alignment,
                "capability_mean_alignment": s.capability_mean_alignment,
                "random_baseline_mean": s.random_baseline_mean,
                "random_baseline_std": s.random_baseline_std,
            }
            for s in scores
        ],
        "safety_degradation": safety_scores,
    }

    with open(output_dir / "stage5_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Stage 5 Results Summary")
    print("=" * 60)
    for s in scores:
        print(f"Step {s.checkpoint_step}: "
              f"safety_align={s.safety_mean_alignment:.4f}, "
              f"cap_align={s.capability_mean_alignment:.4f}, "
              f"random={s.random_baseline_mean:.4f}±{s.random_baseline_std:.4f}")

    finish()
    print("\nStage 5 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5: Fine-Tuning Attack Analysis")
    parser.add_argument("--model-config", default="configs/models/llama3_8b.yaml")
    parser.add_argument("--finetune-dir", default="outputs/stage1/alpaca_finetuned")
    parser.add_argument("--hessian-dir", default="outputs/stage4")
    parser.add_argument("--output-dir", default="outputs/stage5")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    main(args.model_config, args.finetune_dir, args.hessian_dir,
         args.output_dir, args.layer, args.cache_dir)
