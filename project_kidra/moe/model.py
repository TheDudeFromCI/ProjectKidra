import torch
from torch import nn

from project_kidra.moe.attention import CausalSelfAttention
from project_kidra.moe.config import MoEConfig
from project_kidra.moe.moe_layer import MoELayer


class MoEModel(nn.Module):
    """Token embedding -> stacked residual MoE layers -> vocabulary logits."""

    def __init__(self, config: MoEConfig) -> None:
        """Build the model from a config."""

        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.attention_norms = nn.ModuleList(
            nn.LayerNorm(config.d_model) for _ in range(config.num_layers)
        )
        self.attentions = nn.ModuleList(
            CausalSelfAttention(config.d_model, config.num_heads, config.context_window)
            for _ in range(config.num_layers)
        )
        self.moe_norms = nn.ModuleList(
            nn.LayerNorm(config.d_model) for _ in range(config.num_layers)
        )
        self.layers = nn.ModuleList(
            MoELayer(config.d_model, config.d_hidden, config.num_experts, config.top_k)
            for _ in range(config.num_layers)
        )
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map (batch, seq) token ids to (batch, seq, vocab) logits and the summed aux loss."""

        h = self.embed(tokens) + self._position_encoding(
            tokens.shape[1], self.config.d_model, self.embed.weight
        )
        aux = h.new_zeros(())
        for attention_norm, attention, moe_norm, layer in zip(
            self.attention_norms, self.attentions, self.moe_norms, self.layers
        ):
            h = h + attention(attention_norm(h))
            delta, layer_aux = layer(moe_norm(h))
            h = h + delta
            aux = aux + layer_aux
        logits: torch.Tensor = self.head(h)
        return logits, aux / len(self.layers)

    @staticmethod
    def _position_encoding(
        seq_len: int, d_model: int, reference: torch.Tensor
    ) -> torch.Tensor:
        """Create sinusoidal absolute positions for any sequence length."""

        positions = torch.arange(seq_len, device=reference.device, dtype=reference.dtype)
        dimensions = torch.arange(
            0, d_model, 2, device=reference.device, dtype=reference.dtype
        )
        scales = torch.exp(dimensions * (-torch.log(reference.new_tensor(10000.0)) / d_model))
        angles = positions[:, None] * scales[None, :]
        encoding = reference.new_zeros((seq_len, d_model))
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)
