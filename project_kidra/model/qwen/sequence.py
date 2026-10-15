"""Packed Qwen3 chunks with independent positions and event-causal exchange."""

from __future__ import annotations

from functools import partial
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from project_kidra.model.rope import RopeEmbedding
from project_kidra.model.window import ContextWindow

if TYPE_CHECKING:
    from project_kidra.model.qwen.model import QwenAgent


class QwenSequence:
    """Validate schedules before mutation and keep checkpoint recomputation pure."""

    def __init__(
        self, model: QwenAgent, updates: dict[ContextWindow, list[int]], generating: bool = False
    ) -> None:
        """Pack only scheduled targets, rejecting future or invalid token accesses."""

        self.model = model
        self.windows: list[ContextWindow] = []
        self.targets: list[list[int]] = []
        self.ticks: list[Tensor] = []
        self.device = model.backbone.model.embed_tokens.weight.device
        for window, ticks in updates.items():
            if window not in model.windows:
                raise ValueError("scheduled window is not open on this model")
            if any(type(t) is not int or t < 0 for t in ticks) or any(
                a >= b for a, b in pairwise(ticks)
            ):
                raise ValueError("ticks must be nonnegative and strictly increasing")
            if not generating and len(ticks) > len(window.pending):
                raise ValueError("each scheduled tick requires a pending target")
            if window.position + len(ticks) > model.backbone.config.max_position_embeddings:
                raise ValueError("window exceeds pretrained position limit; open a new window")
            if ticks:
                self.windows.append(window)
                self.targets.append(
                    [model.start_token] if generating else window.pending[: len(ticks)]
                )
                self.ticks.append(torch.tensor(ticks, device=self.device))
        self.lengths = [len(ts) for ts in self.targets]
        self.all_windows = list(model.windows)
        cfg = model.backbone.config
        self.heads = int(cfg.num_attention_heads)
        self.kv_heads = int(cfg.num_key_value_heads)
        self.dim = int(cfg.head_dim)
        self.rope = RopeEmbedding(self.dim, cfg.rope_theta)

    def _layer(
        self, x: Tensor, *, layer: Any, exchange: bool, history: list[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, ...]:
        """Compute one layer without mutating caches, including on backward recomputation."""

        attn = layer.self_attn
        h = layer.input_layernorm(x)
        q = attn.q_norm(attn.q_proj(h).reshape(-1, self.heads, self.dim))
        k = attn.k_norm(attn.k_proj(h).reshape(-1, self.kv_heads, self.dim))
        v = attn.v_proj(h).reshape(-1, self.kv_heads, self.dim)
        query_ticks = torch.cat(self.ticks)
        outputs: list[Tensor] = []
        new_entries: list[Tensor] = []
        memories: dict[ContextWindow, tuple[Tensor, Tensor, Tensor]] = {}
        for window, ticks, qp, kp, vp in zip(
            self.windows,
            self.ticks,
            q.split(self.lengths),
            k.split(self.lengths),
            v.split(self.lengths),
        ):
            count = qp.shape[0]
            positions = torch.arange(count, device=x.device) + window.position
            qh = self.rope(qp.transpose(0, 1).unsqueeze(0), positions)
            kh = self.rope(kp.transpose(0, 1).unsqueeze(0), positions)
            vh = vp.transpose(0, 1).unsqueeze(0)
            old_k, old_v = history[self.all_windows.index(window)]
            size = old_k.shape[-2]
            all_k, all_v = torch.cat((old_k, kh), -2), torch.cat((old_v, vh), -2)
            indices = torch.arange(size + count, device=x.device)
            end = size + torch.arange(count, device=x.device)[:, None]
            mask = (indices <= end) & (indices > end - self.model.config.n_local_ctx)
            response = F.scaled_dot_product_attention(
                qh,
                all_k.repeat_interleave(self.heads // self.kv_heads, 1),
                all_v.repeat_interleave(self.heads // self.kv_heads, 1),
                attn_mask=mask,
            )
            outputs.append(response.squeeze(0).transpose(0, 1).reshape(count, -1))
            new_entries.extend((kh, vh))
            if exchange:
                seen = size + torch.searchsorted(ticks, query_ticks, right=True)
                mask = (indices < seen[:, None]) & (
                    indices >= seen[:, None] - self.model.config.n_local_ctx
                )
                memories[window] = (all_k, all_v, mask)
        x = x + attn.o_proj(torch.cat(outputs))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        if exchange and len(self.all_windows) > 1:
            keys: list[Tensor] = []
            values: list[Tensor] = []
            masks: list[Tensor] = []
            identities: list[int] = []
            owners = torch.cat(
                [
                    torch.full((n,), w.id, device=x.device)
                    for w, n in zip(self.windows, self.lengths)
                ]
            )
            for window, (old_k, old_v) in zip(self.all_windows, history):
                if window in memories:
                    mk, mv, mask = memories[window]
                else:
                    mk, mv = old_k, old_v
                    mask = torch.ones((x.shape[0], mk.shape[-2]), device=x.device, dtype=torch.bool)
                if not mk.shape[-2]:
                    continue
                keys.append(mk)
                values.append(mv)
                masks.append(mask & (owners != window.id)[:, None])
                identities.append(window.id)
            x = self.model.exchange(x, keys, values, masks, identities)
        return (x, *new_entries)

    def forward(self) -> Tensor:
        """Return one logit row per target and retain bounded per-window KV caches."""

        states: list[Tensor] = []
        for window, targets in zip(self.windows, self.targets):
            inputs = [window.tokens[-1] if window.tokens else self.model.start_token] + targets[:-1]
            states.append(
                self.model.backbone.model.embed_tokens(
                    torch.tensor(inputs, device=self.device, dtype=torch.long)
                )
            )
        x = torch.cat(states)
        for index, layer in enumerate(self.model.backbone.model.layers):
            history: list[tuple[Tensor, Tensor]] = []
            for window in self.all_windows:
                cache = window.cache[index]
                empty = x.new_empty(1, self.kv_heads, 0, self.dim)
                history.append((cache.k, cache.v) if cache.length else (empty, empty))
            run = partial(
                self._layer,
                layer=layer,
                history=history,
                exchange=(index + 1) % self.model.exchange_every == 0,
            )
            if self.model.gradient_checkpointing and torch.is_grad_enabled():
                result = checkpoint(run, x, use_reentrant=False)
            else:
                result = run(x)
            x = result[0]
            for i, window in enumerate(self.windows):
                window.cache[index].append(result[1 + i * 2], result[2 + i * 2])
        return cast(Tensor, self.model.backbone.lm_head(self.model.backbone.model.norm(x)))

    def commit(self) -> None:
        """Advance targets only after a successful optimizer step and detach all caches."""

        for window, tokens in zip(self.windows, self.targets):
            window.position += len(tokens)
            del window.pending[: len(tokens)]
            for token in tokens:
                window.append(token)
        for window in self.all_windows:
            window.detach_cache()
