"""
Load and preprocess training datasets.

- HH-RLHF / UltraFeedback: preference pairs for DPO training
- Alpaca: instruction-tuning data for benign fine-tuning (safety erosion)
"""

from datasets import load_dataset, Dataset


def load_preference_dataset(
    name: str = "Anthropic/hh-rlhf",
    split: str = "train",
    max_samples: int | None = None,
) -> Dataset:
    """Load a preference dataset for DPO training.

    Args:
        name: HuggingFace dataset ID.
        split: Dataset split to load.
        max_samples: Limit the number of samples. None = all.

    Returns:
        HuggingFace Dataset with 'chosen' and 'rejected' columns.
    """
    dataset = load_dataset(name, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset


def load_instruction_dataset(
    name: str = "tatsu-lab/alpaca",
    split: str = "train",
    max_samples: int | None = None,
) -> Dataset:
    """Load an instruction-tuning dataset for fine-tuning.

    Args:
        name: HuggingFace dataset ID.
        split: Dataset split to load.
        max_samples: Limit the number of samples. None = all.

    Returns:
        HuggingFace Dataset with instruction/input/output columns.
    """
    dataset = load_dataset(name, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset


def format_alpaca_prompt(example: dict) -> str:
    """Format an Alpaca example into a prompt string."""
    if example.get("input", "").strip():
        return (
            f"Below is an instruction that describes a task, paired with an input "
            f"that provides further context. Write a response that appropriately "
            f"completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        return (
            f"Below is an instruction that describes a task. Write a response that "
            f"appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
