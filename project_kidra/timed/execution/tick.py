"""Independent native-layer inference, one token per active window per tick."""

import torch
from torch import Tensor

from project_kidra.model.model import AgentModel
from project_kidra.model.window import ContextWindow


class NativeTick:
    """Avoid SequenceBatch entirely, retaining asynchronous read-only window caches."""

    @staticmethod
    def forward(
        model: AgentModel, windows: list[ContextWindow], cross_window: bool = True
    ) -> tuple[Tensor, Tensor]:
        """Ingest preceding inputs simultaneously without changing tokens or optimizers."""

        if not windows or any(w not in model.windows for w in windows):
            raise ValueError("tick requires open active windows")
        device = model.token_embeddings.weight.device
        tokens = torch.tensor(
            [[w.tokens[-1] if w.tokens else model.free_token] for w in windows], device=device
        )
        identities = torch.tensor([[w.id] for w in windows], device=device)
        x = model.proj(model.token_embeddings(tokens) + model.window_embeddings(identities))
        losses: list[Tensor] = []
        for layer in range(model.config.n_layers):
            x, aux = model.block(x, [w.cache[layer] for w in windows], [w.position for w in windows])
            losses.append(aux)
            if (layer + 1) % model.config.n_global_layers == 0:
                states = [x[i:i + 1] for i in range(len(windows))]
                if cross_window:
                    readable = [w for w in model.windows if w.cache[layer].length]
                    states = model.full_attn(states, readable, model.window_embeddings, layer)
                else:
                    states = [model.full_attn([state], [window], model.window_embeddings, layer)[0]
                              for state, window in zip(states, windows)]
                x = torch.cat(states)
        return model.classifier(model.out_norm(x)).reshape(len(windows), -1), torch.stack(losses).mean()
