"""Immutable run specifications and a full factorial diagnostic matrix."""

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Literal

from project_kidra.timed.execution.policy import TrainingPolicy
from project_kidra.timed.matrix.storage import AtomicFiles
from project_kidra.timed.matrix.tasks import TaskProjection


@dataclass(frozen=True)
class RunSpec:
    """All scientific controls are included in the checkpoint's run identity."""

    task: str
    policy: TrainingPolicy
    learning_rate: float = 0.0005
    seed: int = 42

    @property
    def name(self) -> str:
        """Produce a stable, readable directory name without device-dependent fields."""

        p = self.policy
        suffix = "" if p.cross_window else "-noexchange"
        return (f"{self.task}-{p.gradients}-{p.execution}-{p.loss}"
                f"-lr{self.learning_rate:g}-s{self.seed}{suffix}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize nested controls for a manifest or checkpoint."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunSpec":
        """Reconstruct and validate a stored run specification."""

        result = cls(data["task"], TrainingPolicy(**data["policy"]),
                     data["learning_rate"], data["seed"])
        if result.task not in TaskProjection.names or result.learning_rate <= 0:
            raise ValueError("invalid matrix run")
        return result

    @staticmethod
    def matrix(seeds: list[int]) -> list["RunSpec"]:
        """Build 64 factorial cells and four matched no-exchange controls per seed."""

        runs = []
        gradients_modes: tuple[Literal["full", "detached"], ...] = ("full", "detached")
        execution_modes: tuple[Literal["packed", "tick"], ...] = ("packed", "tick")
        loss_modes: tuple[Literal["all", "facts"], ...] = ("all", "facts")
        for task, gradients, execution, loss, lr, seed in product(
            TaskProjection.names, gradients_modes, execution_modes,
            loss_modes, (0.0005, 0.0001), seeds,
        ):
            policy = TrainingPolicy(gradients=gradients, execution=execution, loss=loss)
            runs.append(RunSpec(task, policy, lr, seed))
        for task, seed in product(TaskProjection.names, seeds):
            runs.append(RunSpec(task, TrainingPolicy(cross_window=False), seed=seed))
        return runs

    def identity(self, manifest_id: str) -> str:
        """Bind the run to its shared datasets, code, and training budget."""

        return AtomicFiles.identity({"manifest": manifest_id, "run": self.to_dict()})
