from collections.abc import Iterable, Iterator

from project_kidra.model.model import AgentModel
from project_kidra.model.window import ContextWindow


class Trainer:
    """Keeps an AgentModel fed with samples, one window per sample."""

    def __init__(self, model: AgentModel, samples: Iterable[list[int]], n_windows: int) -> None:
        """Create a trainer that keeps `n_windows` sample windows open on `model`."""

        if not 1 <= n_windows <= model.config.n_windows:
            raise ValueError("n_windows must be between 1 and the model's n_windows")

        self.model = model
        self.n_windows = n_windows
        self.samples: Iterator[list[int]] = iter(samples)
        self.windows: list[ContextWindow] = []

    def _fill(self) -> None:
        """Open windows for new samples until the target count is reached."""

        while len(self.windows) < self.n_windows:
            tokens: list[int] | None = next(self.samples, None)
            if tokens is None:
                return

            if not tokens:
                continue

            self.windows.append(self.model.open_window().expect(tokens))

    def step(self) -> list[float]:
        """Run one training step and return the losses of the trainer's windows.

        Returns an empty list once the samples are exhausted and all windows are closed.
        """

        self._fill()
        losses = self.model()
        owned = [losses[i] for i, w in enumerate(self.model.windows) if w in self.windows]

        for window in [w for w in self.windows if not w.pending]:
            self.model.close_window(window)
            self.windows.remove(window)

        return owned
