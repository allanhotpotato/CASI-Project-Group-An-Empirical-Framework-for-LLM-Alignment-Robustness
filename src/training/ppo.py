"""
PPO-RLHF safety training pipeline.

Alternative safety training method using Proximal Policy Optimization
with a reward model. Used in Stage 6 to compare against DPO.

References:
    Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
    Ouyang et al. (2022) "Training language models to follow instructions
    with human feedback"
"""

import torch
import yaml
from pathlib import Path
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.loading import load_model_and_tokenizer, save_initial_weights
from src.data.datasets import load_preference_dataset


def run_ppo_training(
    model_config_path: str,
    training_config_path: str,
    output_dir: str,
    cache_dir: str | None = None,
) -> str:
    """Run PPO-RLHF safety training pipeline.

    Args:
        model_config_path: Path to model YAML config.
        training_config_path: Path to PPO training YAML config.
        output_dir: Where to save checkpoints and results.
        cache_dir: HuggingFace cache directory.

    Returns:
        Path to the final trained model checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(training_config_path) as f:
        train_cfg = yaml.safe_load(f)["training"]

    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)["model"]

    model_id = model_cfg["hf_model_id"]

    # Load model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Save initial weights
    save_initial_weights(model.pretrained_model, output_dir)

    # Load reward model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        train_cfg["reward_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(
        train_cfg["reward_model"],
        cache_dir=cache_dir,
    )

    # Load dataset
    dataset = load_preference_dataset(
        name=train_cfg["dataset"],
        split=train_cfg["dataset_split"],
        max_samples=train_cfg.get("max_train_samples"),
    )

    # Configure PPO
    ppo_config = PPOConfig(
        model_name=model_id,
        learning_rate=train_cfg["learning_rate"],
        batch_size=train_cfg["per_device_batch_size"],
        mini_batch_size=train_cfg["per_device_batch_size"],
        ppo_epochs=train_cfg["ppo_epochs"],
        init_kl_coef=train_cfg["init_kl_coef"],
        log_with=train_cfg.get("report_to", "none"),
        output_dir=str(output_dir / "checkpoints"),
    )

    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    # TODO: Implement PPO training loop
    # This requires: generating responses, scoring with reward model,
    # computing PPO updates. The exact loop depends on trl version.
    # Placeholder for Stage 6 implementation.
    raise NotImplementedError(
        "PPO training loop not yet implemented. "
        "Priority is Stage 1 (DPO). Implement for Stage 6."
    )
