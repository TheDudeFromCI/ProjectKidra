"""Track token birth times independently from local cache positions."""

from project_kidra.model.model import AgentModel
from project_kidra.model.window import ContextWindow
from project_kidra.timed.session import TimedSession
from project_kidra.timed.tokenizer import ToyTokenizer


class TemporalSession(TimedSession):
    """Keep original NOOP replay except for the continuously advancing reasoning slot."""

    def __init__(self, model: AgentModel, tokenizer: ToyTokenizer, variant: str,
                 cross_window: bool = True) -> None:
        """Start a fresh sample clock; none is available only as a numerical control."""

        super().__init__(model, tokenizer, cross_window)
        if variant not in {"none", "global_time", "continuous_reasoning"}:
            raise ValueError("unknown temporal variant")
        self.variant = variant
        self.births: dict[int, list[int]] = {}
        self.opened: dict[int, int] = {}

    def continuous(self, window_id: int) -> bool:
        """Identify the reasoning slot whose control targets do not pause its clock."""

        return self.variant == "continuous_reasoning" and window_id == 3

    def begin(self, window_id: int, tick: int = 0) -> ContextWindow:
        """Retain timestamps on replay; do not roll back continuous reasoning positions."""

        if window_id not in self.windows:
            window = super().begin(window_id)
            self.births[window_id], self.opened[window_id] = [], tick
            return window
        if self.continuous(window_id):
            window = self.windows[window_id]
            if window.tokens[-1:] in ([self.tokenizer.noop], [self.tokenizer.end]):
                window.tokens.pop()
                self.births[window_id].pop()
            return window
        window = super().begin(window_id)
        self.births[window_id].pop()
        return window

    def input_tick(self, window: ContextWindow) -> int:
        """Return the insertion time of the preceding input, or the opening time of BOS."""

        return self.births[window.id][-1] if window.tokens else self.opened[window.id]

    def record(self, window: ContextWindow, token: int, tick: int) -> None:
        """Commit one target and its birth tick after simultaneous model inference."""

        self.commit(window, [token])
        self.births[window.id].append(tick)
        self.births[window.id] = self.births[window.id][-window.n_local_ctx:]
