"""Two matched additions to the existing native-tick agent baseline."""

from dataclasses import dataclass
from typing import Any

from project_kidra.timed.execution.policy import TrainingPolicy
from project_kidra.timed.matrix.spec import RunSpec


@dataclass(frozen=True)
class TemporalSpec(RunSpec):
    """Include the temporal intervention in names, manifests, and checkpoint identities."""

    variant: str = "global_time"

    @property
    def name(self) -> str:
        """Separate temporal runs from the unchanged baseline, even in exported reports."""

        return super().name + "-" + self.variant

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemporalSpec":
        """Reject unsupported controls rather than silently changing an experiment."""

        result = cls(data["task"], TrainingPolicy(**data["policy"]),
                     data["learning_rate"], data["seed"], data["variant"])
        if (result.task != "agent" or result.learning_rate <= 0
                or result.variant not in {"global_time", "continuous_reasoning"}
                or result.policy != TrainingPolicy(execution="tick")):
            raise ValueError("invalid temporal run")
        return result
