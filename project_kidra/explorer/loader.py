from pathlib import Path

import torch


def load_embeddings(path: Path) -> torch.Tensor:
    """Load the token embedding matrix from a training checkpoint."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "embed.weight" not in checkpoint.get(
        "model", {}
    ):
        raise ValueError("checkpoint does not contain token embeddings")
    weights: torch.Tensor = checkpoint["model"]["embed.weight"]
    return weights
