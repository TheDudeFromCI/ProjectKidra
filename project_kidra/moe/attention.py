import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention restricted to a causal sliding context window."""

    def __init__(self, d_model: int, num_heads: int, context_window: int) -> None:
        """Build attention for tokens attending to at most `context_window` positions."""

        super().__init__()
        self.context_window = context_window
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local causal attention to a (batch, seq, d_model) input."""

        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        distance = positions[:, None] - positions[None, :]
        blocked = (distance < 0) | (distance >= self.context_window)
        output: torch.Tensor = self.attention(
            x, x, x, attn_mask=blocked, need_weights=False
        )[0]
        return output
