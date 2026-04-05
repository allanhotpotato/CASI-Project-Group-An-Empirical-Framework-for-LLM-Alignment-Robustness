"""
Experiment logging via Weights & Biases.

Provides a thin wrapper for initializing W&B runs and logging metrics
from any stage of the pipeline.
"""

import json
from pathlib import Path
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def init_run(
    project: str = "safety-fragility",
    name: str | None = None,
    config: dict | None = None,
    tags: list[str] | None = None,
    mode: str = "online",
) -> None:
    """Initialize a W&B run.

    Args:
        project: W&B project name.
        name: Run name. Auto-generated if None.
        config: Hyperparameters and settings to log.
        tags: Tags for filtering runs.
        mode: "online", "offline", or "disabled".
    """
    if WANDB_AVAILABLE and mode != "disabled":
        wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            mode=mode,
        )
    else:
        print(f"W&B {'not installed' if not WANDB_AVAILABLE else 'disabled'}. "
              f"Logging to local files only.")


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log metrics to W&B and local file."""
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(metrics, step=step)

    # Always log locally as backup
    _log_local(metrics, step)


def log_summary(metrics: dict) -> None:
    """Log summary metrics (final results)."""
    if WANDB_AVAILABLE and wandb.run is not None:
        for key, value in metrics.items():
            wandb.run.summary[key] = value


def finish() -> None:
    """Finish the current W&B run."""
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


def _log_local(metrics: dict, step: int | None = None) -> None:
    """Append metrics to a local JSONL file."""
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    entry = {"timestamp": datetime.now().isoformat(), "step": step, **metrics}
    log_file = log_dir / "metrics.jsonl"

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
