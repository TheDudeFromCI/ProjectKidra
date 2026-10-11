from torch import Tensor
from torch.nn import GELU, Linear, Module, Sequential


class Expert(Module):
    """A single feed-forward expert network."""

    def __init__(self, d_model: int, d_hidden: int) -> None:
        """Create an expert mapping d_model -> d_hidden -> d_model."""

        super().__init__()
        self.net = Sequential(
            Linear(d_model, d_hidden),
            GELU(),
            Linear(d_hidden, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the expert to a (tokens, d_model) tensor."""

        out: Tensor = self.net(x)
        return out
