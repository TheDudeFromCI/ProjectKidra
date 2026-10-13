import random
from typing import cast

import torch
from torch import Tensor
from torch.nn import (
    Embedding,
    LayerNorm,
    Linear,
    Module,
)
from torch.nn import functional as F
from torch.optim import AdamW

from project_kidra.model.config import Config
from project_kidra.model.exchange import GlobalExchange
from project_kidra.model.layer import LayerBlock
from project_kidra.model.training.sequence import SequenceBatch
from project_kidra.model.window import ContextWindow


class AgentModel(Module):
    """Token model with residual attention, sparse MoE layers, and context windows."""

    def __init__(self, config: Config, free_token: int):
        """Build shared layers and separate trunk/expert optimizers."""

        super().__init__()
        self.config = config
        self.free_token = free_token
        self.available_window_ids = [i for i in range(config.n_windows)]

        self.token_embeddings: Embedding = Embedding(config.vocab_size, config.d_model)
        self.window_embeddings: Embedding = Embedding(config.n_windows, config.d_model)

        self.proj: Linear = Linear(config.d_model, config.d_model)
        self.classifier: Linear = Linear(config.d_model, config.vocab_size)
        self.out_norm: LayerNorm = LayerNorm(config.d_model)
        self.windows: list[ContextWindow] = []
        self.full_attn: GlobalExchange = GlobalExchange(config.d_model, config.n_heads)
        self.trunk_optimizer = AdamW(self.parameters(), lr=config.trunk_lr)

        self.block = LayerBlock(
            config.d_model,
            config.d_hidden,
            config.n_experts,
            config.expert_top_k,
            config.n_heads,
        )
        self.expert_optimizer = AdamW(self.block.parameters(), lr=config.expert_lr)

    def _embedding(self, w: ContextWindow, token: int) -> Tensor:
        """Embed a single input token and its window identity."""

        device = self.token_embeddings.weight.device
        tok: Tensor = torch.tensor([[token]], device=device, dtype=torch.long)
        wid: Tensor = torch.tensor([[w.id]], device=device, dtype=torch.long)
        embedded: Tensor = self.token_embeddings(tok) + self.window_embeddings(wid)
        return cast(Tensor, self.proj(embedded))

    def forward(self) -> list[float]:
        """Train one token per open window, generating unspecified targets."""

        if len(self.windows) == 0:
            return []

        self.train()
        self.trunk_optimizer.zero_grad(set_to_none=True)
        self.expert_optimizer.zero_grad(set_to_none=True)

        positions = [w.position for w in self.windows]
        targets = [w.pop_pending() for w in self.windows]
        # Each step reads the previous token and predicts the target.
        inputs = [w.tokens[-1] if w.tokens else self.free_token for w in self.windows]
        x = torch.cat([self._embedding(w, t) for w, t in zip(self.windows, inputs)], dim=0)
        aux_losses: list[Tensor] = []

        for l_index in range(self.config.n_layers):
            x, aux_loss = self.block(
                x,
                [w.cache[l_index] for w in self.windows],
                positions,
            )
            aux_losses.append(aux_loss)

            if (l_index + 1) % self.config.n_global_layers == 0:
                window_states = [x[i : i + 1] for i in range(len(self.windows))]
                x = torch.cat(
                    self.full_attn(window_states, self.windows, self.window_embeddings, l_index),
                    dim=0,
                )

        y: list[Tensor] = []
        for i, token in enumerate(positions):
            w = self.windows[i]
            w.position += 1

            y.append(self.classifier(self.out_norm(x[i])).reshape(1, -1))
            # w.last_logits = y[-1]

        mean_aux_loss: Tensor = torch.stack(aux_losses).mean()
        logits: Tensor = torch.cat(y)
        expected: Tensor = torch.tensor(targets, device=logits.device, dtype=torch.long)
        predicted: Tensor = logits.argmax(dim=-1)

        for i, target in enumerate(targets):
            if target == self.free_token:
                expected[i] = predicted[i]

        losses: Tensor = F.cross_entropy(logits, expected, reduction="none")
        total: Tensor = losses.mean() + self.config.aux_loss_weight * mean_aux_loss
        total.backward()  # type: ignore

        self.trunk_optimizer.step()
        self.expert_optimizer.step()

        values: list[float] = losses.detach().tolist()

        for w, token in zip(self.windows, expected.tolist()):
            w.detach_cache()
            w.append(token)

        return values

    def train_chunk(
        self,
        max_tokens: int = 32,
        updates: dict[ContextWindow, list[int]] | None = None,
    ) -> list[float]:
        """Train all scheduled targets in one forward/backward/optimizer step.

        By default each window with pending targets advances up to max_tokens
        times, at ticks 0, 1, ... . An explicit updates mapping supplies relative
        ticks per window and overrides that default. Omitted windows stay idle
        but their committed caches remain readable. Losses are returned in
        mapping order, then target order, with one value per real token.
        """

        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if updates is None:
            updates = {
                w: list(range(min(max_tokens, len(w.pending)))) for w in self.windows if w.pending
            }
        batch = SequenceBatch(self, updates)
        if not batch.windows:
            return []
        self.train()
        self.trunk_optimizer.zero_grad(set_to_none=True)
        self.expert_optimizer.zero_grad(set_to_none=True)
        logits, aux_loss = batch.forward()
        targets = torch.tensor(
            [token for sequence in batch.targets for token in sequence],
            device=logits.device,
            dtype=torch.long,
        )
        losses = F.cross_entropy(logits, targets, reduction="none")
        total = losses.mean() + self.config.aux_loss_weight * aux_loss
        total.backward()  # type: ignore
        self.trunk_optimizer.step()
        self.expert_optimizer.step()
        for window, sequence in zip(batch.windows, batch.targets):
            window.position += len(sequence)
            del window.pending[: len(sequence)]
            for token in sequence:
                window.append(token)
        for window in self.windows:
            window.detach_cache()
        return cast(list[float], losses.detach().tolist())

    def open_window(self) -> ContextWindow:
        """Allocate an independent context window with an available identity."""

        if not self.available_window_ids:
            raise RuntimeError("No available window IDs to open a new window.")

        window: ContextWindow = ContextWindow(
            index=len(self.windows),
            id=random.choice(self.available_window_ids),
            n_layers=self.config.n_layers,
            n_local_ctx=self.config.n_local_ctx,
            vocab_size=self.config.vocab_size,
            free_token=self.free_token,
        )

        self.available_window_ids.remove(window.id)
        self.windows.append(window)

        return window

    def close_window(self, window: ContextWindow) -> None:
        """Close a window, freeing its id and reindexing the remaining windows."""

        self.windows.remove(window)
        self.available_window_ids.append(window.id)

        for i, w in enumerate(self.windows):
            w.index = i
