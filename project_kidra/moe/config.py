from dataclasses import dataclass


@dataclass(frozen=True)
class MoEConfig:
    """Hyperparameters for the Mixture of Experts model."""

    vocab_size: int = 256
    d_model: int = 64
    d_hidden: int = 128
    num_experts: int = 4
    top_k: int = 2
    num_layers: int = 2
    num_heads: int = 4
    context_window: int = 128
    aux_loss_weight: float = 0.01

    def __post_init__(self) -> None:
        """Validate that the configuration is self-consistent."""

        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if min(self.vocab_size, self.d_model, self.d_hidden, self.num_layers) < 1:
            raise ValueError("sizes must be positive")
        if self.num_heads < 1 or self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by a positive num_heads")
        if self.context_window < 1:
            raise ValueError("context_window must be positive")
