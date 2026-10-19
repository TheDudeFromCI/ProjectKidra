"""Window lifecycle and cache operations shared by timed training and evaluation."""

import torch
from torch import Tensor

from project_kidra.model.model import AgentModel
from project_kidra.model.training.sequence import SequenceBatch
from project_kidra.model.window import ContextWindow
from project_kidra.timed.tokenizer import ToyTokenizer


class TimedSession:
    """Own a sample's windows; terminal NOOP is scored but never ingested as content."""

    def __init__(
        self, model: AgentModel, tokenizer: ToyTokenizer, cross_window: bool = True
    ) -> None:
        """Require a clean model so independent samples cannot share context."""

        if model.windows:
            raise ValueError("timed session requires no open windows")
        self.model = model
        self.tokenizer = tokenizer
        self.cross_window = cross_window
        self.windows: dict[int, ContextWindow] = {}

    def begin(self, window_id: int) -> ContextWindow:
        """Open an explicit slot or resume by recomputing the last content input."""

        if window_id not in self.windows:
            window = ContextWindow(
                len(self.model.windows), window_id, self.model.config.n_layers,
                self.model.config.n_local_ctx, self.model.config.vocab_size, self.model.free_token,
            )
            self.model.available_window_ids.remove(window_id)
            self.model.windows.append(window)
            self.windows[window_id] = window
            return window
        window = self.windows[window_id]
        if not window.tokens or window.tokens[-1] != self.tokenizer.noop:
            raise ValueError("only a NOOP-paused window can resume")
        window.tokens.pop()
        # The NOOP target was not ingested. Replay its preceding input so the
        # resumed query sees new context without duplicating a position in KV.
        window.position -= 1
        for cache in window.cache:
            cache.k = cache.k[..., :-1, :]
            cache.v = cache.v[..., :-1, :]
        return window

    def forward(self, updates: dict[ContextWindow, list[int]]) -> tuple[Tensor, Tensor]:
        """Compute logits using event-causal packed attention without an optimizer step."""

        return SequenceBatch(self.model, updates, self.cross_window).forward()

    def commit(self, window: ContextWindow, tokens: list[int]) -> None:
        """Record observed or generated targets after their corresponding forward pass."""

        window.position += len(tokens)
        window.pending.clear()
        for token in tokens:
            window.append(token)

    def detach(self) -> None:
        """Truncate backpropagation at a timing-group boundary, keeping weights fixed."""

        for window in self.model.windows:
            window.detach_cache()

    def close(self) -> None:
        """Release every sample window even when execution raises an exception."""

        for window in list(self.model.windows):
            self.model.close_window(window)

    def observe(self, streams: list[tuple[int, int, list[int]]]) -> None:
        """Prefill external observations without prediction loss."""

        updates = {}
        for window_id, tick, tokens in streams:
            window = self.begin(window_id).expect(tokens)
            updates[window] = list(range(tick, tick + len(tokens)))
        with torch.no_grad():
            self.forward(updates)
        for window in updates:
            self.commit(window, list(window.pending))
        self.detach()
