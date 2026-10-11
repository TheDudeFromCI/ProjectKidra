from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Configuration settings for the Agent."""

    vocab_size: int = 256
    d_model: int = 64
    d_hidden: int = 128
    n_experts: int = 4
    n_layers: int = 4
    n_heads: int = 4
    n_windows: int = 1
    n_local_ctx: int = 1024
    n_global_layers: int = 1
    expert_top_k: int = 2
    aux_loss_weight: float = 0.01
    trunk_lr: float = 0.0001
    expert_lr: float = 0.001

    def model_parameters(self) -> dict[str, int | float]:
        """Return only the serializable hyperparameters used to build the model."""

        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "n_experts": self.n_experts,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_windows": self.n_windows,
            "n_local_ctx": self.n_local_ctx,
            "n_global_layers": self.n_global_layers,
            "expert_top_k": self.expert_top_k,
            "aux_loss_weight": self.aux_loss_weight,
            "trunk_lr": self.trunk_lr,
            "expert_lr": self.expert_lr,
        }

    def __post_init__(self) -> None:
        """Validate that the configuration is self-consistent."""

        if not 1 <= self.expert_top_k <= self.n_experts:
            raise ValueError("expert_top_k must be between 1 and n_experts")

        if min(self.vocab_size, self.d_model, self.d_hidden, self.n_layers) < 1:
            raise ValueError("sizes must be positive")

        if self.n_heads < 1 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by a positive n_heads")

        if self.n_local_ctx < 1:
            raise ValueError("n_local_ctx must be positive")

        if self.n_windows < 1:
            raise ValueError("n_windows must be positive")

        if self.n_global_layers <= 0:
            raise ValueError("n_global_layers must be positive")

        if self.n_layers % self.n_global_layers != 0:
            raise ValueError("n_global_layers must divide n_layers evenly")

        if self.trunk_lr <= 0.0:
            raise ValueError("trunk_lr must be positive")

        if self.expert_lr <= 0.0:
            raise ValueError("expert_lr must be positive")
