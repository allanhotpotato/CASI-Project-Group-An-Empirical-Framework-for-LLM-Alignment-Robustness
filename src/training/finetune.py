"""
Standard instruction fine-tuning on benign data (Alpaca).

This simulates real-world safety erosion: a safety-trained model is
fine-tuned on benign instruction data, and safety degrades as a side effect.

References:
    Qi et al. (2023) "Fine-tuning Aligned Language Models Compromises Safety"
"""

import yaml
from pathlib import Path
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from src.models.loading import load_model_and_tokenizer, save_initial_weights
from src.data.datasets import load_instruction_dataset, format_alpaca_prompt


def run_finetune(
    model_path: str,
    model_config_path: str,
    training_config_path: str,
    output_dir: str,
    cache_dir: str | None = None,
) -> str:
    """Fine-tune a safety-trained model on benign instruction data.

    Args:
        model_path: Path to the safety-trained model checkpoint.
        model_config_path: Path to model YAML config (for tokenizer).
        training_config_path: Path to fine-tuning YAML config.
        output_dir: Where to save checkpoints.
        cache_dir: HuggingFace cache directory.

    Returns:
        Path to the final fine-tuned model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(training_config_path) as f:
        train_cfg = yaml.safe_load(f)["training"]

    # Load safety-trained model
    model, tokenizer = load_model_and_tokenizer(
        config_path=model_config_path,
        from_checkpoint=model_path,
        cache_dir=cache_dir,
    )

    # Save pre-finetune weights (these are the safety-trained weights)
    save_initial_weights(model, output_dir)

    # Load and tokenize Alpaca dataset
    dataset = load_instruction_dataset(
        name=train_cfg["dataset"],
        split=train_cfg["dataset_split"],
        max_samples=train_cfg.get("max_train_samples"),
    )

    def tokenize_fn(examples):
        texts = [format_alpaca_prompt(
            {"instruction": inst, "input": inp, "output": out}
        ) for inst, inp, out in zip(
            examples["instruction"],
            examples["input"],
            examples["output"],
        )]
        return tokenizer(
            texts,
            truncation=True,
            max_length=train_cfg["max_length"],
            padding=False,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    # Data collator handles dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        weight_decay=train_cfg["weight_decay"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        logging_steps=train_cfg["logging_steps"],
        report_to=train_cfg.get("report_to", "none"),
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()

    # Save final model
    final_path = output_dir / "final_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    return str(final_path)
