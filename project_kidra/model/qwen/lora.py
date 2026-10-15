"""Small trainable residuals around frozen pretrained projections."""

import math

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """Keep base weights frozen and optimize low-rank adapters in float32."""

    def __init__(self, base: nn.Linear, rank: int) -> None:
        """Start with an exactly zero residual and rank-scaled updates."""

        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = 1.0  # alpha = rank

    def forward(self, x: Tensor) -> Tensor:
        """Add the float32 low-rank branch, cast back to the base output dtype."""

        result: Tensor = self.base(x)
        update = nn.functional.linear(nn.functional.linear(x.float(), self.a), self.b)
        return result + (update * self.scale).to(result.dtype)
