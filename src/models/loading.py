"""
Model loading utilities.

Provides a single entry point for loading HuggingFace models with the
correct dtype, device mapping, and tokenizer configuration.
"""

import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(
    model_id: str | None = None,
    config_path: str | Path | None = None,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    cache_dir: str | None = None,
    from_checkpoint: str | None = None,
) -> tuple:
    """Load a model and tokenizer from HuggingFace or a local checkpoint.

    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Meta-Llama-3-8B").
            Ignored if config_path is provided.
        config_path: Path to a model YAML config file.
        dtype: Weight dtype. "bfloat16" or "float16".
        device_map: How to distribute across GPUs. "auto" for automatic.
        cache_dir: HuggingFace cache directory. Useful for persistent
            storage on RunPod (/workspace/.cache/huggingface).
        from_checkpoint: Path to a local checkpoint directory to load
            instead of the base model.

    Returns:
        Tuple of (model, tokenizer).
    """
    if config_path is not None:
        config = load_config(config_path)
        model_id = config["model"]["hf_model_id"]
        dtype = config["model"].get("dtype", dtype)

    if model_id is None:
        raise ValueError("Must provide either model_id or config_path")

    torch_dtype = getattr(torch, dtype)

    load_path = from_checkpoint or model_id

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        cache_dir=cache_dir,
    )

    # Always load tokenizer from original model ID (checkpoints may not include it)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )

    # Ensure pad token is set (required for batched operations)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def save_initial_weights(model, output_dir: str | Path) -> None:
    """Save model weights before training starts.

    This is needed to compute delta_theta = theta_current - theta_initial
    at each checkpoint during training.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    torch.save(state_dict, output_dir / "initial_weights.pt")


def compute_delta_theta(
    model,
    initial_weights_path: str | Path,
) -> dict[str, torch.Tensor]:
    """Compute cumulative weight update from initial weights.

    Args:
        model: Current model state.
        initial_weights_path: Path to initial_weights.pt saved before training.

    Returns:
        Dict mapping parameter name to delta tensor (on CPU).
    """
    initial = torch.load(initial_weights_path, map_location="cpu", weights_only=True)
    delta = {}
    for name, param in model.named_parameters():
        if name in initial:
            delta[name] = param.cpu().detach() - initial[name]
    return delta
