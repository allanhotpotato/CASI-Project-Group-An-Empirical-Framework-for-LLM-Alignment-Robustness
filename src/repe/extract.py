"""
LAT (Linear Artificial Tomography) vector extraction.

Implements the core RepE procedure from Zou et al. (2025):
1. Run contrastive pairs through the model
2. Collect hidden-state activations at the last token position per layer
3. Compute difference vectors
4. Run PCA — first principal component is the reading vector

References:
    Zou et al. (2025) "Representation Engineering" — Section 3
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from dataclasses import dataclass
from tqdm import tqdm

from src.data.contrastive_pairs import ContrastivePair


@dataclass
class ReadingVector:
    """A concept reading vector extracted via LAT."""
    concept: str
    layer: int
    vector: np.ndarray           # The first PC direction (hidden_size,)
    explained_variance: float    # Fraction of variance explained by first PC
    all_variances: np.ndarray    # All singular values from PCA
    auroc: float | None = None   # Classification AUROC on held-out data


def collect_activations(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int] | None = None,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
) -> dict[int, torch.Tensor]:
    """Run texts through model and collect last-token hidden states per layer.

    Args:
        model: HuggingFace CausalLM.
        tokenizer: Corresponding tokenizer.
        texts: List of prompt strings.
        layers: Which layers to collect from. None = all layers.
        batch_size: Inference batch size.
        max_length: Max token length.
        device: Device for inference.

    Returns:
        Dict mapping layer index to tensor of shape (num_texts, hidden_size).
    """
    model.eval()
    if layers is None:
        layers = list(range(model.config.num_hidden_layers))

    # Storage for activations
    all_activations = {layer: [] for layer in layers}

    for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )

        # hidden_states is a tuple of (num_layers + 1) tensors,
        # each of shape (batch, seq_len, hidden_size).
        # Index 0 is the embedding layer, index i+1 is layer i.
        hidden_states = outputs.hidden_states

        # Get last non-padding token position for each sequence
        attention_mask = inputs["attention_mask"]
        # Shape: (batch,) — index of last real token
        last_token_idx = attention_mask.sum(dim=1) - 1

        for layer in layers:
            # hidden_states[layer + 1] because index 0 is embeddings
            layer_hidden = hidden_states[layer + 1]
            # Extract activation at last token position for each sequence
            batch_indices = torch.arange(layer_hidden.size(0), device=device)
            last_token_acts = layer_hidden[batch_indices, last_token_idx]
            all_activations[layer].append(last_token_acts.cpu())

    # Concatenate all batches
    return {
        layer: torch.cat(acts, dim=0)
        for layer, acts in all_activations.items()
    }


def extract_reading_vectors(
    model,
    tokenizer,
    pairs: list[ContrastivePair],
    layers: list[int] | None = None,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
) -> list[ReadingVector]:
    """Extract LAT reading vectors for a concept from contrastive pairs.

    Args:
        model: HuggingFace CausalLM.
        tokenizer: Corresponding tokenizer.
        pairs: Contrastive pairs for a single concept.
        layers: Which layers to extract from. None = all.
        batch_size: Inference batch size.
        max_length: Max token length.
        device: Device for inference.

    Returns:
        List of ReadingVector objects, one per layer.
    """
    concept = pairs[0].concept
    positive_texts = [p.positive for p in pairs]
    negative_texts = [p.negative for p in pairs]

    # Collect activations for both sides
    pos_activations = collect_activations(
        model, tokenizer, positive_texts, layers, batch_size, max_length, device
    )
    neg_activations = collect_activations(
        model, tokenizer, negative_texts, layers, batch_size, max_length, device
    )

    reading_vectors = []
    for layer in pos_activations:
        # Compute difference vectors: shape (num_pairs, hidden_size)
        diff = pos_activations[layer] - neg_activations[layer]
        diff_np = diff.numpy()

        # PCA on difference vectors
        pca = PCA()
        pca.fit(diff_np)

        # First PC is the reading vector
        reading_vec = pca.components_[0]
        explained_var = pca.explained_variance_ratio_[0]

        reading_vectors.append(ReadingVector(
            concept=concept,
            layer=layer,
            vector=reading_vec,
            explained_variance=explained_var,
            all_variances=pca.explained_variance_ratio_,
        ))

    return reading_vectors


def validate_reading_vector_auroc(
    model,
    tokenizer,
    reading_vector: ReadingVector,
    held_out_pairs: list[ContrastivePair],
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
) -> float:
    """Compute classification AUROC for a reading vector on held-out data.

    Projects held-out activations onto the reading vector direction and
    measures how well it separates positive from negative examples.

    Returns:
        AUROC score (should be > 0.85 for valid vectors).
    """
    from sklearn.metrics import roc_auc_score

    positive_texts = [p.positive for p in held_out_pairs]
    negative_texts = [p.negative for p in held_out_pairs]

    pos_acts = collect_activations(
        model, tokenizer, positive_texts,
        layers=[reading_vector.layer],
        batch_size=batch_size, max_length=max_length, device=device,
    )[reading_vector.layer].numpy()

    neg_acts = collect_activations(
        model, tokenizer, negative_texts,
        layers=[reading_vector.layer],
        batch_size=batch_size, max_length=max_length, device=device,
    )[reading_vector.layer].numpy()

    # Project onto reading vector direction
    direction = reading_vector.vector
    pos_scores = pos_acts @ direction
    neg_scores = neg_acts @ direction

    # Binary classification: positive=1, negative=0
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    auroc = roc_auc_score(labels, scores)
    # AUROC could be < 0.5 if direction is flipped; take max with complement
    return max(auroc, 1 - auroc)
