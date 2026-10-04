from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch

from project_kidra.moe.token_corpus import TokenCorpus
from project_kidra.moe.trainer import TrainStepResult, Trainer


class ContinuousTrainer:
    """Repeatedly trains a model on a cycling token stream and saves resumable state."""

    def __init__(self, trainer: Trainer, corpus: TokenCorpus) -> None:
        """Create a continuous trainer using injected model, optimizer, and corpus."""

        self.trainer = trainer
        self.corpus = corpus
        self.step = 0

    def train(
        self,
        max_steps: int | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_interval: int = 1000,
        on_step: Callable[[int, TrainStepResult], None] | None = None,
    ) -> int:
        """Train indefinitely, or for max_steps, periodically saving checkpoints."""

        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be positive or None")
        if checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")

        completed = 0
        while max_steps is None or completed < max_steps:
            inputs, targets = self.corpus.next_batch()
            result = self.trainer.train_step(inputs, targets)
            self.step += 1
            completed += 1
            if on_step is not None:
                on_step(self.step, result)
            if checkpoint_path is not None and self.step % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_path)

        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path)
        return completed

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model weights, optimizer state, model configuration, and step."""

        torch.save(
            {
                "config": asdict(self.trainer.model.config),
                "model": self.trainer.model.state_dict(),
                "optimizer": self.trainer.optimizer.state_dict(),
                "step": self.step,
                "corpus_path": str(self.corpus.path.resolve()),
                "corpus_size": self.corpus.size,
                "corpus_offset": self.corpus.tell(),
            },
            Path(path),
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model and optimizer state after validating the model config."""

        checkpoint = torch.load(
            Path(path),
            map_location=next(self.trainer.model.parameters()).device,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must contain a mapping")
        if checkpoint.get("config") != asdict(self.trainer.model.config):
            raise ValueError("checkpoint model config does not match the current model")
        step = checkpoint.get("step")
        if not isinstance(step, int) or step < 0:
            raise ValueError("checkpoint has an invalid training step")
        if checkpoint.get("corpus_path") != str(self.corpus.path.resolve()):
            raise ValueError("checkpoint corpus path does not match the current corpus")
        if checkpoint.get("corpus_size") != self.corpus.size:
            raise ValueError("checkpoint corpus size does not match the current corpus")
        offset = checkpoint.get("corpus_offset")
        if not isinstance(offset, int) or not 0 <= offset < self.corpus.size:
            raise ValueError("checkpoint has an invalid corpus offset")
        self.trainer.model.load_state_dict(checkpoint["model"])
        self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        self.corpus.seek(offset)
        self.step = step
