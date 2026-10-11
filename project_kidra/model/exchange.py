"""Global exchange module for multi-window attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, LayerNorm, Linear, Module

from project_kidra.model.window import ContextWindow


class GlobalExchange(Module):
    def __init__(self, d_model: int, n_heads: int):
        """
        Initialize the GlobalExchange (Full Attention) module.

        Args:
            d_model (int): The dimensionality of the model.
            n_heads (int): The number of attention heads.
        """

        super().__init__()
        self.n_heads = n_heads
        self.dim = d_model // n_heads

        self.norm = LayerNorm(d_model)
        self.query = Linear(d_model, d_model)
        self.out = Linear(d_model, d_model)

    def forward(
        self,
        x: list[Tensor],
        windows: list[ContextWindow],
        w_embed: Embedding,
        layer: int,
    ) -> list[Tensor]:
        ids: list[int] = list(range(len(x)))
        current: Tensor = torch.cat(x, dim=0)

        cache: list[tuple[Tensor, Tensor]] = [
            (w.cache[layer].k, w.cache[layer].v)
            if w.cache[layer] is not None
            else (
                torch.zeros(1, self.n_heads, 1, self.dim, device=current.device),
                torch.zeros(1, self.n_heads, 1, self.dim, device=current.device),
            )
            for w in windows
        ]

        embeds: list[Tensor] = [
            w_embed(torch.tensor([[w.id]], device=current.device, dtype=torch.long)).reshape(
                1, self.n_heads, 1, self.dim
            )
            for w in windows
        ]

        all_k = torch.cat([c[0] + e for c, e in zip(cache, embeds)], dim=-2)
        all_v = torch.cat([c[1] for c in cache], dim=-2)

        q = (
            self.query(self.norm(current[ids]))
            .reshape(1, -1, self.n_heads, self.dim)
            .transpose(1, 2)
        )

        response = F.scaled_dot_product_attention(q, all_k, all_v)
        response = self.out(response.transpose(1, 2).reshape(-1, self.n_heads * self.dim))

        for j, r in zip(ids, response.unbind(0)):
            x[j] = x[j] + r.reshape(1, 1, -1)

        return x
