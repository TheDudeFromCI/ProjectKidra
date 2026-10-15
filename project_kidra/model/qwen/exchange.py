"""Parameter-efficient cross-window attention over existing GQA caches."""

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class QwenExchange(nn.Module):
    """Shared low-rank exchange with a zero-initialized output projection."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        windows: int,
        rank: int,
        device: torch.device,
    ) -> None:
        """Allocate only new float32 adapter weights, independent of the backbone."""

        super().__init__()
        self.heads = heads
        self.groups = heads // kv_heads
        self.head_dim = head_dim
        self.norm = nn.RMSNorm(hidden, eps=1e-6, device=device)
        self.query = nn.Sequential(
            nn.Linear(hidden, rank, bias=False, device=device),
            nn.Linear(rank, heads * head_dim, bias=False, device=device),
        )
        self.output = nn.Sequential(
            nn.Linear(heads * head_dim, rank, bias=False, device=device),
            nn.Linear(rank, hidden, bias=False, device=device),
        )
        self.identities = nn.Embedding(windows, kv_heads * head_dim, device=device)
        nn.init.normal_(self.identities.weight, std=0.02)
        nn.init.zeros_(cast(nn.Linear, self.output[1]).weight)

    def forward(
        self,
        x: Tensor,
        keys: list[Tensor],
        values: list[Tensor],
        masks: list[Tensor],
        identities: list[int],
    ) -> Tensor:
        """Read only causally visible cached inputs from other windows."""

        keyed = [
            k + self.identities.weight[index].to(k.dtype).reshape(1, -1, 1, self.head_dim)
            for k, index in zip(keys, identities)
        ]
        q = self.query(self.norm(x.float())).to(x.dtype)
        q = q.reshape(1, -1, self.heads, self.head_dim).transpose(1, 2)
        k = torch.cat(keyed, dim=-2).repeat_interleave(self.groups, dim=1)
        v = torch.cat(values, dim=-2).repeat_interleave(self.groups, dim=1)
        mask = torch.cat(masks, dim=-1)
        response = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        response = response.transpose(1, 2).reshape(x.shape[0], -1)
        return cast(Tensor, x + self.output(response.float()).to(x.dtype))
