"""
Activation steering using RepE reading vectors.

Validates that extracted reading vectors actually influence model behavior
by adding/subtracting them during generation.

References:
    Zou et al. (2025) "Representation Engineering" — Section 4
"""

import torch
from functools import partial
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.repe.extract import ReadingVector


def _steering_hook(module, input, output, direction, alpha):
    """Hook that adds a steering vector to a layer's output.

    Args:
        direction: Steering vector (hidden_size,).
        alpha: Scaling factor. Positive = steer toward concept,
               negative = steer away.
    """
    # output is a tuple; hidden states are the first element
    hidden_states = output[0]
    steering = torch.tensor(direction, dtype=hidden_states.dtype, device=hidden_states.device)
    hidden_states = hidden_states + alpha * steering
    return (hidden_states,) + output[1:]


def generate_with_steering(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    reading_vector: ReadingVector,
    alpha: float = 1.0,
    max_new_tokens: int = 128,
) -> str:
    """Generate text with a reading vector added to activations.

    Args:
        model: HuggingFace CausalLM.
        tokenizer: Corresponding tokenizer.
        prompt: Input prompt string.
        reading_vector: The RepE reading vector to apply.
        alpha: Steering strength. Positive = more of the concept,
               negative = less.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated text string.
    """
    # Register hook on the target layer
    layer_module = model.model.layers[reading_vector.layer]
    hook = layer_module.register_forward_hook(
        partial(_steering_hook, direction=reading_vector.vector, alpha=alpha)
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        # Decode only the new tokens
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
    finally:
        hook.remove()

    return generated


def compare_steered_outputs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    reading_vector: ReadingVector,
    alphas: list[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
    max_new_tokens: int = 128,
) -> dict[float, str]:
    """Generate outputs at multiple steering strengths for comparison.

    Returns:
        Dict mapping alpha value to generated text.
    """
    results = {}
    for alpha in alphas:
        results[alpha] = generate_with_steering(
            model, tokenizer, prompt, reading_vector, alpha, max_new_tokens
        )
    return results
