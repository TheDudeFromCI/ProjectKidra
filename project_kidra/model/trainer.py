from collections.abc import Iterable, Iterator

from project_kidra.model.model import AgentModel
from project_kidra.model.window import ContextWindow


class Trainer:
    """Keeps an AgentModel fed with samples, one window per sample."""

    def __init__(
        self,
        model: AgentModel,
        samples: Iterable[list[int]],
        n_windows: int,
        tokens_per_step: int = 1,
    ) -> None:
        """Create a trainer that keeps `n_windows` sample windows open on `model`."""

        if not 1 <= n_windows <= model.config.n_windows:
            raise ValueError("n_windows must be between 1 and the model's n_windows")

        if tokens_per_step < 1:
            raise ValueError("tokens_per_step must be positive")
        self.tokens_per_step = tokens_per_step
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
        """Train one chunk and return one loss per real target, in window order.

        Returns an empty list once the samples are exhausted and all windows are closed.
        """

        self._fill()
        owned = self.model.train_chunk(
            updates={
                w: list(range(min(self.tokens_per_step, len(w.pending)))) for w in self.windows
            }
        )

        for window in [w for w in self.windows if not w.pending]:
            self.model.close_window(window)
            self.windows.remove(window)

        return owned
