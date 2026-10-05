import torch
from torch import nn

from project_kidra.moe.expert import Expert
from project_kidra.moe.router import Router


class MoELayer(nn.Module):
    """Sparse MoE feed-forward layer: each token is processed by its top-k experts."""

    def __init__(
        self, d_model: int, d_hidden: int, num_experts: int, top_k: int
    ) -> None:
        """Build the router and the experts."""

        super().__init__()
        self.router = Router(d_model, num_experts, top_k)
        self.experts = nn.ModuleList(
            Expert(d_model, d_hidden) for _ in range(num_experts)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (output, aux_loss) for a (batch, seq, d_model) input."""

        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        routed = self.router(flat)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            token_idx, slot = (routed.indices == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            weight = routed.weights[token_idx, slot].unsqueeze(-1)
            out.index_add_(0, token_idx, expert(flat[token_idx]) * weight)
        return out.reshape(shape), routed.aux_loss
