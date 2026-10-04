from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RouterOutput:
    """Routing decision: chosen experts, their weights, and the load-balancing loss."""

    indices: torch.Tensor  # (tokens, top_k) expert ids
    weights: torch.Tensor  # (tokens, top_k) normalized gate weights
    aux_loss: torch.Tensor  # scalar load-balancing loss


class Router(nn.Module):
    """Top-k softmax gate with a Switch-style load-balancing auxiliary loss."""

    def __init__(self, d_model: int, num_experts: int, top_k: int) -> None:
        """Create a router over `num_experts` experts selecting `top_k` per token."""

        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """Route a (tokens, d_model) tensor."""

        probs = torch.softmax(self.gate(x), dim=-1)
        top_w, top_i = probs.topk(self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)

        # fraction of routed slots per expert * mean gate probability per expert
        assigned = torch.zeros_like(probs).scatter(1, top_i, 1.0)
        fraction = assigned.mean(dim=0) / self.top_k
        aux = self.num_experts * (fraction * probs.mean(dim=0)).sum()
        return RouterOutput(indices=top_i, weights=top_w, aux_loss=aux)
