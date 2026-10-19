"""Serializable experimental controls without changes to model capacity."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TrainingPolicy:
    """Keep optimizer timing constant while changing one training factor at a time."""

    gradients: Literal["full", "detached"] = "full"
    execution: Literal["packed", "tick"] = "packed"
    loss: Literal["all", "facts"] = "all"
    cross_window: bool = True
    noop_weight: float = 0.25

    def __post_init__(self) -> None:
        """Reject unsupported modes before touching model state."""

        if self.gradients not in {"full", "detached"}:
            raise ValueError("gradients must be full or detached")
        if self.execution not in {"packed", "tick"}:
            raise ValueError("execution must be packed or tick")
        if self.loss not in {"all", "facts"}:
            raise ValueError("loss must be all or facts")
        if not 0 < self.noop_weight <= 1:
            raise ValueError("noop_weight must be in (0, 1]")

    def weight(self, word: str, predict: bool) -> float:
        """Retain grammar/control supervision while emphasizing variable facts."""

        if not predict:
            return 0.0
        if word == "<|noop|>":
            return self.noop_weight
        if self.loss == "facts" and not word.startswith(("k", "v")):
            return 0.1 if word != "<|end|>" else 0.25
        return 1.0
