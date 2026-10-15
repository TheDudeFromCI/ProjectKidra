"""Train related windows together, without mixing independent corpus samples."""

from collections.abc import Iterable, Iterator

from project_kidra.model.model import AgentModel
from project_kidra.model.window import ContextWindow


class GroupedTrainer:
    """Keep one sample's windows alive until all of its targets are consumed."""

    def __init__(
        self, model: AgentModel, samples: Iterable[list[list[int]]], tokens_per_step: int = 32
    ) -> None:
        """Own the complete window group to prevent cross-sample context leakage."""

        if tokens_per_step < 1:
            raise ValueError("tokens_per_step must be positive")
        if model.windows:
            raise ValueError("grouped training requires a model with no open windows")
        self.model = model
        self.samples: Iterator[list[list[int]]] = iter(samples)
        self.tokens_per_step = tokens_per_step
        self.windows: list[ContextWindow] = []

    def step(self) -> list[float]:
        """Advance unfinished windows; retain completed context until the group ends."""

        if self.model.windows != self.windows:
            raise ValueError("windows were changed outside the grouped trainer")
        if not self.windows:
            sample = next(self.samples, None)
            if sample is None:
                return []
            if not sample or len(sample) > self.model.config.n_windows:
                raise ValueError("sample window count exceeds capacity or is empty")
            if any(
                not tokens
                or any(
                    type(t) is not int or t < 0 or t >= self.model.config.vocab_size for t in tokens
                )
                for tokens in sample
            ):
                raise ValueError("sample windows must contain valid token IDs")
            self.windows = [self.model.open_window().expect(tokens) for tokens in sample]
        losses = self.model.train_chunk(
            updates={
                w: list(range(min(self.tokens_per_step, len(w.pending))))
                for w in self.windows
                if w.pending
            }
        )
        if all(not window.pending for window in self.windows):
            for window in self.windows:
                self.model.close_window(window)
            self.windows.clear()
        return losses
