from dataclasses import dataclass

import torch
from torch import nn

from project_kidra.moe import MoEModel


@dataclass(frozen=True)
class TrainStepResult:
    """Losses from a single optimisation step."""

    loss: float
    task_loss: float
    aux_loss: float


class Trainer:
    """Runs training/evaluation steps; model and optimizer are injected."""

    def __init__(
        self,
        model: MoEModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module | None = None,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        """Create a trainer. `loss_fn` defaults to cross entropy."""

        self.model = model
        self.optimizer = optimizer
        self.loss_fn: nn.Module = (
            loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        )
        self.max_grad_norm = max_grad_norm

    def _losses(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (total, task, aux) losses for a batch."""

        logits, aux = self.model(inputs)
        task = self.loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return task + self.model.config.aux_loss_weight * aux, task, aux

    def train_step(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> TrainStepResult:
        """Perform one gradient update on a batch of token ids."""

        self.model.train()
        self.optimizer.zero_grad()
        total, task, aux = self._losses(inputs, targets)
        total.backward()  # type: ignore[no-untyped-call]
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return TrainStepResult(total.item(), task.item(), aux.item())

    @torch.no_grad()
    def evaluate(self, inputs: torch.Tensor, targets: torch.Tensor) -> TrainStepResult:
        """Compute losses on a batch without updating weights."""

        self.model.eval()
        total, task, aux = self._losses(inputs, targets)
        return TrainStepResult(total.item(), task.item(), aux.item())
