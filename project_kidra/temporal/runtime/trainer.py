"""Accumulate group gradients and update model weights once per complete trajectory."""

import torch
from torch.nn import functional as F

from project_kidra.model.model import AgentModel
from project_kidra.temporal.runtime.group import TemporalGroup
from project_kidra.temporal.runtime.session import TemporalSession
from project_kidra.timed.execution.policy import TrainingPolicy
from project_kidra.timed.schedule import Schedule
from project_kidra.timed.schema import TimedSample
from project_kidra.timed.tokenizer import ToyTokenizer
from project_kidra.timed.trainer import TrainResult


class TemporalTrainer:
    """Compare attached versus truncated history with identical sample-level updates."""

    def __init__(
        self, model: AgentModel, tokenizer: ToyTokenizer, noop_weight: float = 0.25,
        policy: TrainingPolicy | None = None, variant: str = "global_time",
    ) -> None:
        """Assign an explicit weight to waiting decisions rather than idle ticks."""

        if not 0 < noop_weight <= 1:
            raise ValueError("noop_weight must be in (0, 1]")
        self.model = model
        self.tokenizer = tokenizer
        self.noop_weight = noop_weight
        self.policy = policy or TrainingPolicy(execution="tick", noop_weight=noop_weight)
        if self.policy.execution != "tick" or self.policy.gradients != "full":
            raise ValueError("temporal controls require full-gradient single-tick execution")
        self.variant = variant

    def step(self, sample: TimedSample) -> TrainResult:
        """Update once per sample, preserving observation graphs in the full mode."""

        model = self.model
        schedule = Schedule(sample, self.tokenizer, model.config.n_windows, model.config.n_local_ctx)
        total_weight = sum(
            self.policy.weight(self.tokenizer.words[t], True)
            for group in schedule.groups for stream in group if stream.source.predict
            for t in stream.tokens
        )
        session = TemporalSession(model, self.tokenizer, self.variant, self.policy.cross_window)
        optimizers = (model.trunk_optimizer, model.expert_optimizer)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        model.train()
        loss_sum, labels, backwards = 0.0, 0, 0
        objectives: list[torch.Tensor] = []
        try:
            for group in schedule.groups:
                weights: list[float] = []
                targets: list[int] = []
                for stream in group:
                    targets.extend(stream.tokens)
                    weights.extend([
                        self.policy.weight(self.tokenizer.words[t], stream.source.predict)
                        for t in stream.tokens
                    ])
                supervised = any(weights)
                with torch.set_grad_enabled(supervised or self.policy.gradients == "full"):
                    logits, aux = TemporalGroup.forward(session, group)
                    if supervised:
                        weight = logits.new_tensor(weights)
                        target = torch.tensor(targets, device=logits.device)
                        loss = (F.cross_entropy(logits, target, reduction="none") * weight).sum()
                        total = loss / total_weight
                        total = total + model.config.aux_loss_weight * aux * weight.sum() / total_weight
                        if self.policy.gradients == "full":
                            objectives.append(total)
                        else:
                            total.backward()  # type: ignore[no-untyped-call]
                            backwards += 1
                        loss_sum += float(loss.detach())
                        labels += sum(w > 0 for w in weights)
                if self.policy.gradients == "detached":
                    session.detach()
            if objectives:
                torch.stack(objectives).sum().backward()  # type: ignore[no-untyped-call]
                backwards += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            for optimizer in optimizers:
                optimizer.step()
        finally:
            session.close()
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
        return TrainResult(loss_sum / total_weight, labels, backwards)
