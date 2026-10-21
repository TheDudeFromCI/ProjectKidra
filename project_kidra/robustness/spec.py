"""Separate initialization seed, sample order, and deterministic augmentation."""

from dataclasses import dataclass
from typing import Any

from project_kidra.followup.suite.spec import FollowSpec
from project_kidra.timed.execution.policy import TrainingPolicy


@dataclass(frozen=True)
class RobustSpec(FollowSpec):
    order_seed: int = 42
    augmentation: str = "none"

    @property
    def name(self) -> str:
        return f"{super().name}-order{self.order_seed}-{self.augmentation}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobustSpec":
        result = cls(**{**data, "policy": TrainingPolicy(**data["policy"])})
        if (result.task != "agent" or result.action not in {"audit", "cross", "extend", "repair"}
                or result.variant not in {"none", "global_time", "continuous_reasoning"}
                or result.learning_rate != 0.0001 or result.budget not in {50000, 100000}
                or result.augmentation not in {"none", "prefixmix"}
                or result.policy != TrainingPolicy(execution="tick")
                or type(result.order_seed) is not int or result.order_seed < 0
                or type(result.seed) is not int or result.seed < 0
                or bool(result.source) != (result.action != "cross")
                or (result.action in {"repair", "extend"} and result.budget != 100000)
                or (result.action == "cross" and (result.budget != 50000 or result.variant != "none"))
                or (result.augmentation == "prefixmix") != (result.action == "repair")
                or (result.action == "repair" and result.variant != "none")):
            raise ValueError("invalid robustness specification")
        return result
