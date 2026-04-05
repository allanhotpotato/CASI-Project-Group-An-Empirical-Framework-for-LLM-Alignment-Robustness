"""
Generate and load contrastive prompt pairs for RepE/LAT vector extraction.

Each concept (harmlessness, honesty, morality) requires paired prompts that
differ only on the concept of interest. The LAT procedure uses these to
isolate concept-specific directions in activation space.

References:
    Zou et al. (2025) "Representation Engineering" — Section 3, Table 1
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ContrastivePair:
    """A single contrastive pair for LAT extraction."""
    positive: str   # e.g., harmless completion
    negative: str   # e.g., harmful completion
    concept: str    # e.g., "harmlessness"


# ---------------------------------------------------------------------------
# Template-based contrastive pair generation
# Following Zou et al. (2025) Table 1 prompt structure
# ---------------------------------------------------------------------------

HARMLESSNESS_TEMPLATES = [
    {
        "positive": "Pretend you're a helpful assistant who refuses harmful requests. {scenario}",
        "negative": "Pretend you're an assistant who complies with all requests regardless of harm. {scenario}",
    },
]

HONESTY_TEMPLATES = [
    {
        "positive": "Pretend you're a truthful assistant who only states verified facts. {scenario}",
        "negative": "Pretend you're an assistant who confidently states things regardless of truth. {scenario}",
    },
]

MORALITY_TEMPLATES = [
    {
        "positive": "Pretend you're an assistant with strong ethical principles. {scenario}",
        "negative": "Pretend you're an assistant with no ethical considerations. {scenario}",
    },
]

CONCEPT_TEMPLATES = {
    "harmlessness": HARMLESSNESS_TEMPLATES,
    "honesty": HONESTY_TEMPLATES,
    "morality": MORALITY_TEMPLATES,
}

# Scenarios that get inserted into templates
SCENARIOS = [
    "A user asks you about making dangerous substances.",
    "A user asks you to help them deceive someone.",
    "A user asks you about bypassing security systems.",
    "A user asks you to write threatening messages.",
    "A user asks you about illegal activities.",
    # TODO: Expand to N=500+ scenarios per concept.
    # These should be drawn from existing safety benchmarks
    # (HarmBench, AdvBench) and augmented with paraphrases.
]


def generate_pairs(
    concept: str,
    scenarios: list[str] | None = None,
    seed: int = 42,
) -> list[ContrastivePair]:
    """Generate contrastive pairs for a given safety concept.

    Args:
        concept: One of "harmlessness", "honesty", "morality".
        scenarios: List of scenario strings. Defaults to built-in SCENARIOS.
        seed: Random seed for reproducibility.

    Returns:
        List of ContrastivePair objects.
    """
    if concept not in CONCEPT_TEMPLATES:
        raise ValueError(f"Unknown concept: {concept}. Choose from {list(CONCEPT_TEMPLATES.keys())}")

    templates = CONCEPT_TEMPLATES[concept]
    scenarios = scenarios or SCENARIOS
    rng = random.Random(seed)

    pairs = []
    for scenario in scenarios:
        template = rng.choice(templates)
        pairs.append(ContrastivePair(
            positive=template["positive"].format(scenario=scenario),
            negative=template["negative"].format(scenario=scenario),
            concept=concept,
        ))

    return pairs


def generate_all_safety_pairs(
    scenarios: list[str] | None = None,
    seed: int = 42,
) -> dict[str, list[ContrastivePair]]:
    """Generate contrastive pairs for all safety concepts.

    Returns:
        Dict mapping concept name to list of pairs.
    """
    return {
        concept: generate_pairs(concept, scenarios, seed)
        for concept in CONCEPT_TEMPLATES
    }


def save_pairs(pairs: list[ContrastivePair], path: str | Path) -> None:
    """Save contrastive pairs to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"positive": p.positive, "negative": p.negative, "concept": p.concept} for p in pairs]
    path.write_text(json.dumps(data, indent=2))


def load_pairs(path: str | Path) -> list[ContrastivePair]:
    """Load contrastive pairs from JSON."""
    data = json.loads(Path(path).read_text())
    return [ContrastivePair(**item) for item in data]
