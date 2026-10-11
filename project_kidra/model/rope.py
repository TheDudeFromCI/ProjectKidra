"""Rotary Positional Embeddings with explicit positions."""

import torch
from torch import Tensor
from torch.nn import Module


class RopeEmbedding(Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()

        if head_dim % 2:
            raise ValueError("head_dim must be even")

        self.head_dim = head_dim

        self.inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def forward(
        self,
        x: Tensor,
        positions: Tensor | int,
    ) -> Tensor:
        """
        x: [..., head_dim]

        positions: integer or tensor broadcastable to
                   x.shape[:-1]

        Example:
            x:         [batch, heads, seq, head_dim]
            positions: [batch, 1, seq]
        """

        positions = torch.as_tensor(
            positions,
            device=x.device,
            dtype=torch.float32,
        )

        inv_freq = self.inv_freq.to(x.device)
        angles = positions.unsqueeze(-1) * inv_freq
        angles = torch.cat((angles, angles), dim=-1)

        cos = angles.cos().to(dtype=x.dtype)
        sin = angles.sin().to(dtype=x.dtype)

        return x * cos + self._rotate_half(x) * sin
