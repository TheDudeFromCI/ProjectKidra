"""Independent native-layer inference, one token per active window per tick."""

import torch
from torch import Tensor

from project_kidra.model.window import ContextWindow
from project_kidra.temporal.runtime.session import TemporalSession


class TemporalTick:
    """Avoid SequenceBatch entirely, retaining asynchronous read-only window caches."""

    @staticmethod
    def forward(
        session: TemporalSession, windows: list[ContextWindow]
    ) -> tuple[Tensor, Tensor]:
        """Ingest preceding inputs simultaneously without changing tokens or optimizers."""

        model, cross_window = session.model, session.cross_window
        if not windows or any(w not in model.windows for w in windows):
            raise ValueError("tick requires open active windows")
        device = model.token_embeddings.weight.device
        tokens = torch.tensor(
            [[w.tokens[-1] if w.tokens else model.free_token] for w in windows], device=device
        )
        identities = torch.tensor([[w.id] for w in windows], device=device)
        embedded = model.token_embeddings(tokens) + model.window_embeddings(identities)
        if session.variant == "global_time":
            births = embedded.new_tensor([session.input_tick(w) for w in windows])
            embedded = embedded + TemporalTick.encoding(births, embedded.shape[-1]).unsqueeze(1)
        x = model.proj(embedded)
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

    @staticmethod
    def encoding(ticks: Tensor, width: int) -> Tensor:
        """Encode absolute token birth ticks with fixed unit-scale sinusoids, base 10000."""

        frequencies = 10000.0 ** (-torch.arange(0, width, 2, device=ticks.device,
                                               dtype=torch.float32) / width)
        angles = ticks.float().unsqueeze(-1) * frequencies
        encoded = angles.new_empty((len(ticks), width))
        encoded[:, 0::2] = angles.sin()
        encoded[:, 1::2] = angles[:, :width // 2].cos()
        return encoded.to(ticks.dtype)
