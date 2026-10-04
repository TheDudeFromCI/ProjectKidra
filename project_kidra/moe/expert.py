import torch
from torch import nn


class Expert(nn.Module):
    """A single feed-forward expert network."""

    def __init__(self, d_model: int, d_hidden: int) -> None:
        """Create an expert mapping d_model -> d_hidden -> d_model."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the expert to a (tokens, d_model) tensor."""

        out: torch.Tensor = self.net(x)
        return out
