from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Linear, Module


@dataclass(frozen=True)
class RouterOutput:
    """Routing decision: chosen experts, their weights, and the load-balancing loss."""

    indices: Tensor  # (tokens, top_k) expert ids
    weights: Tensor  # (tokens, top_k) normalized gate weights
    aux_loss: Tensor  # scalar load-balancing loss


class Router(Module):
    """Top-k softmax gate with a Switch-style load-balancing auxiliary loss."""

    def __init__(self, d_model: int, n_experts: int, top_k: int) -> None:
        """Create a router over `num_experts` experts selecting `top_k` per token."""

        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = Linear(d_model, n_experts, bias=False)

    def forward(self, x: Tensor) -> RouterOutput:
        """Route a (tokens, d_model) tensor."""

        probs = Tensor.softmax(self.gate(x), dim=-1)
        top_w, top_i = probs.topk(self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)

        assigned = torch.zeros_like(probs).scatter(1, top_i, 1.0)
        fraction = assigned.mean(dim=0) / self.top_k
        aux = self.n_experts * (fraction * probs.mean(dim=0)).sum()
        return RouterOutput(indices=top_i, weights=top_w, aux_loss=aux)
