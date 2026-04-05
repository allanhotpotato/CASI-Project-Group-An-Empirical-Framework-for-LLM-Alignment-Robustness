"""
Stage 6: Scale and Statistical Validation

Repeat Stages 2-5 across multiple models and training methods,
then run statistical significance tests.

Models: LLaMA-3-8B, Mistral-7B, Qwen2.5-7B
Methods: DPO, PPO-RLHF
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from itertools import product


MODELS = {
    "llama3_8b": "configs/models/llama3_8b.yaml",
    "mistral_7b": "configs/models/mistral_7b.yaml",
    "qwen2_7b": "configs/models/qwen2_7b.yaml",
}

METHODS = ["dpo", "ppo"]


def main(
    models: list[str] | None = None,
    methods: list[str] | None = None,
    cache_dir: str | None = None,
):
    models = models or list(MODELS.keys())
    methods = methods or METHODS

    output_base = Path("outputs/stage6")
    output_base.mkdir(parents=True, exist_ok=True)

    for model_name, method in product(models, methods):
        print(f"\n{'=' * 60}")
        print(f"Running: {model_name} + {method}")
        print(f"{'=' * 60}")

        model_config = MODELS[model_name]
        run_output = output_base / f"{model_name}_{method}"

        # Stage 2: RepE extraction
        print(f"\n--- Stage 2: RepE extraction for {model_name} ---")
        _run_script("scripts/run_stage2.py", [
            "--model-config", model_config,
            "--output-dir", str(run_output / "stage2"),
            *(["--cache-dir", cache_dir] if cache_dir else []),
        ])

        # Stage 3: RepE metrics
        print(f"\n--- Stage 3: RepE metrics for {model_name} ---")
        _run_script("scripts/run_stage3.py", [
            "--repe-dir", str(run_output / "stage2"),
            "--output-dir", str(run_output / "stage3"),
        ])

        # Stage 4: Hessian fragility
        print(f"\n--- Stage 4: Hessian fragility for {model_name} ---")
        _run_script("scripts/run_stage4.py", [
            "--model-config", model_config,
            "--repe-dir", str(run_output / "stage2"),
            "--output-dir", str(run_output / "stage4"),
            *(["--cache-dir", cache_dir] if cache_dir else []),
        ])

    # Statistical validation across all runs
    print(f"\n{'=' * 60}")
    print("Running statistical validation...")
    print(f"{'=' * 60}")
    _run_statistical_tests(output_base, models, methods)


def _run_script(script: str, args: list[str]):
    """Run a stage script as a subprocess."""
    cmd = [sys.executable, script] + args
    result = subprocess.run(cmd, check=True)
    return result


def _run_statistical_tests(output_base: Path, models: list[str], methods: list[str]):
    """Run statistical significance tests across all model/method combinations."""
    from scipy import stats
    import numpy as np

    all_results = {}

    for model_name, method in product(models, methods):
        key = f"{model_name}_{method}"
        metrics_path = output_base / key / "stage3" / "safety_metrics.json"
        fragility_path = output_base / key / "stage4" / "fragility_results.json"

        result = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                result["metrics"] = json.load(f)
        if fragility_path.exists():
            with open(fragility_path) as f:
                result["fragility"] = json.load(f)

        all_results[key] = result

    # Compare effective dimensionality across models
    print("\nEffective Dimensionality Comparison:")
    for key, data in all_results.items():
        if "metrics" in data:
            dims = [m["effective_dimensionality"] for m in data["metrics"]]
            print(f"  {key}: mean={np.mean(dims):.1f}, std={np.std(dims):.1f}")

    # Compare fragility overlap across models
    print("\nFragility Overlap Comparison:")
    for key, data in all_results.items():
        if "fragility" in data:
            overlaps = [f["max_overlap"] for f in data["fragility"]]
            print(f"  {key}: mean={np.mean(overlaps):.4f}, std={np.std(overlaps):.4f}")

    # TODO: Paired t-tests, effect sizes, cross-model/method significance
    # This will be filled in once we have actual data from runs

    summary = {
        "models": models,
        "methods": methods,
        "results": {k: {
            "num_metrics": len(v.get("metrics", [])),
            "num_fragility": len(v.get("fragility", [])),
        } for k, v in all_results.items()},
    }

    with open(output_base / "stage6_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nStage 6 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6: Scale and Statistical Validation")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODELS.keys()))
    parser.add_argument("--methods", nargs="+", default=None,
                        choices=METHODS)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    main(args.models, args.methods, args.cache_dir)
