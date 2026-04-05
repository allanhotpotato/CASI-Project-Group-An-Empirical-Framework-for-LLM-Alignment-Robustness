"""
DPO safety training pipeline.

Applies Direct Preference Optimization to teach the model safe behavior.
Saves initial weights and checkpoints with cumulative weight updates (delta_theta)
for downstream adversarial alignment analysis.

References:
    Rafailov et al. (2023) "Direct Preference Optimization"
"""

import torch
import yaml
from pathlib import Path
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig

from src.models.loading import load_model_and_tokenizer, save_initial_weights
from src.data.datasets import load_preference_dataset


def run_dpo_training(
    model_config_path: str,
    training_config_path: str,
    output_dir: str,
    cache_dir: str | None = None,
) -> str:
    """Run full DPO safety training pipeline.

    Args:
        model_config_path: Path to model YAML config.
        training_config_path: Path to DPO training YAML config.
        output_dir: Where to save checkpoints and results.
        cache_dir: HuggingFace cache directory.

    Returns:
        Path to the final trained model checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    with open(training_config_path) as f:
        train_cfg = yaml.safe_load(f)["training"]

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        config_path=model_config_path,
        cache_dir=cache_dir,
    )

    # Save initial weights for delta_theta computation
    save_initial_weights(model, output_dir)

    # Load reference model (frozen copy for DPO)
    ref_model, _ = load_model_and_tokenizer(
        config_path=model_config_path,
        cache_dir=cache_dir,
    )

    # Load preference dataset
    dataset = load_preference_dataset(
        name=train_cfg["dataset"],
        split=train_cfg["dataset_split"],
        max_samples=train_cfg.get("max_train_samples"),
    )

    # Configure DPO training
    training_args = DPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        weight_decay=train_cfg["weight_decay"],
        beta=train_cfg["beta"],
        max_length=train_cfg["max_length"],
        max_prompt_length=train_cfg["max_prompt_length"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        logging_steps=train_cfg["logging_steps"],
        report_to=train_cfg.get("report_to", "none"),
        bf16=True,
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    trainer.train()

    # Save final model
    final_path = output_dir / "final_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    return str(final_path)
