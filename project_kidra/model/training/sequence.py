"""Teacher-forced chunks with event-causal local and global attention."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from project_kidra.model.window import ContextWindow

if TYPE_CHECKING:
    from project_kidra.model.model import AgentModel


class SequenceBatch:
    """A validated schedule of target predictions, packed without padding tokens.

    Tick values are relative to this batch. Each tick ingests the preceding
    committed token in each scheduled window, then predicts targets together.
    Thus same-tick inputs are visible, but same-tick targets are not.
    """

    def __init__(
        self, model: AgentModel, updates: dict[ContextWindow, list[int]], cross_window: bool = True
    ) -> None:
        """Validate all updates before changing any window or optimizer state."""

        self.model = model
        self.cross_window = cross_window
        self.windows: list[ContextWindow] = []
        self.ticks: list[Tensor] = []
        self.targets: list[list[int]] = []
        device = model.token_embeddings.weight.device
        for window, ticks in updates.items():
            if window not in model.windows:
                raise ValueError("Scheduled window is not open on this model")
            if any(type(t) is not int or t < 0 for t in ticks):
                raise ValueError("Update ticks must be nonnegative integers")
            if any(a >= b for a, b in pairwise(ticks)):
                raise ValueError("Each window's update ticks must be strictly increasing")
            if len(ticks) > len(window.pending):
                raise ValueError("Each scheduled update requires a pending target")
            targets = window.pending[: len(ticks)]
            if model.free_token in targets:
                raise ValueError(
                    "Chunks require known targets; use model() for free-token generation"
                )
            if ticks:
                self.windows.append(window)
                self.ticks.append(torch.tensor(ticks, device=device, dtype=torch.long))
                self.targets.append(targets)
        self.lengths = [len(targets) for targets in self.targets]

    def forward(self) -> tuple[Tensor, Tensor]:
        """Return per-target logits and auxiliary loss, retaining final chunk caches.

        Weights stay fixed throughout the chunk. Historical caches are detached
        by the training caller between chunks; new entries remain differentiable.
        """

        model = self.model
        device = model.token_embeddings.weight.device
        states: list[Tensor] = []
        for window, targets in zip(self.windows, self.targets):
            inputs = [window.tokens[-1] if window.tokens else model.free_token] + targets[:-1]
            token_ids = torch.tensor(inputs, device=device, dtype=torch.long)
            window_id = torch.tensor(window.id, device=device, dtype=torch.long)
            states.append(
                model.proj(model.token_embeddings(token_ids) + model.window_embeddings(window_id))
            )
        x = torch.cat(states)
        query_ticks = torch.cat(self.ticks)
        aux_losses: list[Tensor] = []
        block = model.block

        for layer in range(model.config.n_layers):
            global_layer = (layer + 1) % model.config.n_global_layers == 0
            q, k, v = block.qkv(block.norm1(x)).chunk(3, dim=-1)
            queries = q.split(self.lengths)
            keys = k.split(self.lengths)
            values = v.split(self.lengths)
            local_outputs: list[Tensor] = []
            memories: dict[ContextWindow, tuple[Tensor, Tensor, Tensor]] = {}
            for window, ticks, q_part, k_part, v_part in zip(
                self.windows, self.ticks, queries, keys, values
            ):
                count = q_part.shape[0]
                positions = torch.arange(count, device=device) + window.position
                q_heads = block.rope(
                    q_part.reshape(1, count, block.n_heads, block.dim).transpose(1, 2),
                    positions,
                )
                k_heads = block.rope(
                    k_part.reshape(1, count, block.n_heads, block.dim).transpose(1, 2),
                    positions,
                )
                v_heads = v_part.reshape(1, count, block.n_heads, block.dim).transpose(1, 2)
                cache = window.cache[layer]
                history = cache.length
                all_k = torch.cat((cache.k, k_heads), dim=-2) if history else k_heads
                all_v = torch.cat((cache.v, v_heads), dim=-2) if history else v_heads
                key_index = torch.arange(history + count, device=device)
                end = history + torch.arange(count, device=device)[:, None]
                local_mask = (key_index <= end) & (key_index > end - cache.limit)
                response = F.scaled_dot_product_attention(
                    q_heads, all_k, all_v, attn_mask=local_mask
                )
                local_outputs.append(response.transpose(1, 2).reshape(count, -1))

                # Reconstruct this window's retained cache at every query tick,
                # before truncating the persistent cache at the end of the chunk.
                if global_layer:
                    seen = history + torch.searchsorted(ticks, query_ticks, right=True)
                    global_mask = (key_index < seen[:, None]) & (
                        key_index >= seen[:, None] - cache.limit
                    )
                    memories[window] = (all_k, all_v, global_mask)
                cache.append(k_heads, v_heads)

            x, aux = block.feed_forward(x + block.out(torch.cat(local_outputs)))
            aux_losses.append(aux)
            if global_layer:
                global_keys: list[Tensor] = []
                global_values: list[Tensor] = []
                global_masks: list[Tensor] = []
                for window in model.windows:
                    if window in memories:
                        memory_k, memory_v, mask = memories[window]
                    else:
                        cache = window.cache[layer]
                        if not cache.length:
                            continue
                        memory_k, memory_v = cache.k, cache.v
                        mask = torch.ones(
                            (x.shape[0], cache.length), device=device, dtype=torch.bool
                        )
                    window_id = torch.tensor(window.id, device=device, dtype=torch.long)
                    identity = model.window_embeddings(window_id).reshape(
                        1, block.n_heads, 1, block.dim
                    )
                    global_keys.append(memory_k + identity)
                    global_values.append(memory_v)
                    if not self.cross_window:
                        owners = torch.cat([
                            torch.full((n,), w.id, device=device)
                            for w, n in zip(self.windows, self.lengths)
                        ])
                        mask = mask & (owners == window.id)[:, None]
                    global_masks.append(mask)
                exchange = model.full_attn
                global_q = (
                    exchange.query(exchange.norm(x))
                    .reshape(1, -1, block.n_heads, block.dim)
                    .transpose(1, 2)
                )
                response = F.scaled_dot_product_attention(
                    global_q,
                    torch.cat(global_keys, dim=-2),
                    torch.cat(global_values, dim=-2),
                    attn_mask=torch.cat(global_masks, dim=-1),
                )
                x = x + exchange.out(response.transpose(1, 2).reshape_as(x))

        return model.classifier(model.out_norm(x)), torch.stack(aux_losses).mean()
