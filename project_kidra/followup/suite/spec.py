"""Focused replication, checkpoint audit, and additional-budget run specifications."""

from dataclasses import dataclass
from typing import Any

from project_kidra.timed.execution.policy import TrainingPolicy
from project_kidra.timed.matrix.spec import RunSpec


@dataclass(frozen=True)
class FollowSpec(RunSpec):
    """Bind initialization source, intervention, and total training budget to run identity."""

    action: str = "replicate"
    variant: str = "none"
    source: str = ""
    budget: int = 50000

    @property
    def name(self) -> str:
        """Keep audits and continuations separate from completed original experiments."""

        return f"{self.action}-{super().name}-{self.variant}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FollowSpec":
        """Validate the deliberately narrow follow-up design."""

        result = cls(data["task"], TrainingPolicy(**data["policy"]), data["learning_rate"],
                     data["seed"], data["action"], data["variant"], data["source"], data["budget"])
        if (result.task != "agent" or result.action not in {"audit", "replicate", "extend"}
                or result.variant not in {"none", "global_time", "continuous_reasoning"}
                or result.learning_rate != 0.0001 or result.budget not in {50000, 100000}
                or (result.budget == 100000) != (result.action == "extend")
                or bool(result.source) != (result.action != "replicate")
                or result.policy.loss != "all" or not result.policy.cross_window
                or (result.action != "audit" and result.policy.gradients != "full")
                or (result.variant != "none" and result.policy.execution != "tick")):
            raise ValueError("invalid follow-up specification")
        return result
