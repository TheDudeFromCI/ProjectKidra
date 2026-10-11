import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ModuleList
from torch.nn import functional as F

from project_kidra.model.cache import KVCache
from project_kidra.model.expert import Expert
from project_kidra.model.rope import RopeEmbedding
from project_kidra.model.router import Router


class LayerBlock(Module):
    """Sparse MoE feed-forward layer block."""

    def __init__(
        self, d_model: int, d_hidden: int, n_experts: int, top_k: int, n_heads: int
    ) -> None:
        """Build the router and the experts."""

        super().__init__()
        self.n_heads = n_heads
        self.dim = d_model // n_heads

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.router = Router(d_model, n_experts, top_k)
        self.experts = ModuleList(Expert(d_model, d_hidden) for _ in range(n_experts))
        self.qkv = Linear(d_model, 3 * d_model)
        self.out = Linear(d_model, d_model)
        self.rope = RopeEmbedding(d_model // n_heads)

    def forward(
        self, x: Tensor, caches: list[KVCache], window_positions: list[int]
    ) -> tuple[Tensor, Tensor]:
        """Process one token per window using their independent local caches."""

        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        batch_size = x.shape[0]
        q = q.reshape(batch_size, 1, self.n_heads, self.dim).transpose(1, 2)
        k = k.reshape(batch_size, 1, self.n_heads, self.dim).transpose(1, 2)
        v = v.reshape(batch_size, 1, self.n_heads, self.dim).transpose(1, 2)

        positions = torch.tensor(window_positions, device=x.device).reshape(batch_size, 1, 1)
        q = self.rope(q, positions=positions)
        k = self.rope(k, positions=positions)

        for index, cache in enumerate(caches):
            cache.append(k[index : index + 1], v[index : index + 1])

        max_length = max(cache.length for cache in caches)
        keys = torch.cat([F.pad(cache.k, (0, 0, 0, max_length - cache.length)) for cache in caches])
        values = torch.cat(
            [F.pad(cache.v, (0, 0, 0, max_length - cache.length)) for cache in caches]
        )
        attention_mask = torch.arange(max_length, device=x.device).unsqueeze(0) < torch.tensor(
            [cache.length for cache in caches], device=x.device
        ).unsqueeze(1)
        a = F.scaled_dot_product_attention(
            q, keys, values, attn_mask=attention_mask[:, None, None, :]
        )

        h = self.out(a.transpose(1, 2).reshape(batch_size, 1, -1))
        x2 = x + h

        tokens = self.norm2(x2).reshape(-1, x2.shape[-1])
        routing = self.router(tokens)
        routed = tokens.new_zeros(tokens.shape)

        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = (routing.indices == expert_id).nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue

            expert_output = expert(tokens[token_ids])
            weighted_output = expert_output * routing.weights[token_ids, slots].unsqueeze(-1)
            routed = routed.index_add(0, token_ids, weighted_output)

        return x2 + routed.reshape_as(x2), routing.aux_loss
