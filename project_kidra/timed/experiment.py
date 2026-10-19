"""Reproducible CPU-sized experiment, checkpointing, and held-out evaluation."""

import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from project_kidra.model.config import Config
from project_kidra.model.model import AgentModel
from project_kidra.timed.evaluator import ToyEvaluator
from project_kidra.timed.schedule import Schedule
from project_kidra.timed.schema import TimedSample
from project_kidra.timed.tokenizer import ToyTokenizer
from project_kidra.timed.trainer import TimedTrainer


class ToyExperiment:
    """Train approximately one million parameters using the native shared-layer model."""

    def __init__(self, seed: int = 42, device: str = "cpu") -> None:
        """Initialize all RNGs and use a single dense expert to isolate scheduling."""

        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        self.tokenizer = ToyTokenizer()
        self.model = AgentModel(Config(
            vocab_size=len(self.tokenizer.words), d_model=192, d_hidden=1856,
            n_experts=1, expert_top_k=1, n_layers=2, n_heads=4, n_windows=5,
            n_local_ctx=128, n_global_layers=1, aux_loss_weight=0.0,
            trunk_lr=0.0005, expert_lr=0.0005,
        ), self.tokenizer.bos).to(device)
        self.completed_steps = 0
        self.sample_order: list[int] = []
        self.train_fingerprint: str | None = None

    @staticmethod
    def read(path: Path) -> list[TimedSample]:
        """Read JSONL episodes, retaining each complete trajectory as the split unit."""

        with path.open() as source:
            samples = [TimedSample.from_dict(json.loads(line)) for line in source if line.strip()]
        if not samples:
            raise ValueError("empty dataset")
        return samples

    def save(self, path: Path) -> None:
        """Persist architecture, vocabulary, optimizers, and RNG state without active caches."""

        if self.model.windows:
            raise ValueError("checkpoints require a completed sample with no open windows")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": 1, "seed": self.seed, "steps": self.completed_steps,
            "config": asdict(self.model.config), "vocabulary": self.tokenizer.words,
            "model": self.model.state_dict(),
            "trunk_optimizer": self.model.trunk_optimizer.state_dict(),
            "expert_optimizer": self.model.expert_optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "sample_order": self.sample_order, "train_fingerprint": self.train_fingerprint,
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "ToyExperiment":
        """Restore a trusted tensor-only checkpoint and reject vocabulary mismatches."""

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        instance = cls(checkpoint["seed"], device)
        if checkpoint["format"] != 1 or checkpoint["vocabulary"] != instance.tokenizer.words:
            raise ValueError("incompatible timed toy checkpoint")
        if checkpoint["config"] != asdict(instance.model.config):
            raise ValueError("checkpoint architecture differs from this experiment")
        instance.model.load_state_dict(checkpoint["model"])
        instance.model.trunk_optimizer.load_state_dict(checkpoint["trunk_optimizer"])
        instance.model.expert_optimizer.load_state_dict(checkpoint["expert_optimizer"])
        instance.completed_steps = checkpoint["steps"]
        instance.sample_order = checkpoint["sample_order"]
        instance.train_fingerprint = checkpoint["train_fingerprint"]
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if torch.cuda.is_available() and checkpoint["cuda_rng"]:
            if len(checkpoint["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("CUDA RNG restore requires the same number of visible GPUs")
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        return instance

    def train(
        self, samples: list[TimedSample], validation: list[TimedSample], steps: int,
        output: Path, eval_every: int = 250,
    ) -> dict[str, Any]:
        """Train shuffled complete samples and emit progress plus a final ablation report."""

        if steps < 1 or eval_every < 1:
            raise ValueError("steps and eval_every must be positive")
        if not samples or not validation:
            raise ValueError("training and evaluation require nonempty datasets")
        if {s.sample_id for s in samples} & {s.sample_id for s in validation}:
            raise ValueError("training and evaluation sample IDs must be disjoint")
        for sample in samples + validation:
            Schedule(sample, self.tokenizer, self.model.config.n_windows,
                     self.model.config.n_local_ctx)
        fingerprint = hashlib.sha256(json.dumps(
            [sample.to_dict() for sample in samples], sort_keys=True
        ).encode()).hexdigest()
        if self.train_fingerprint is not None and self.train_fingerprint != fingerprint:
            raise ValueError("resume requires the same training dataset and order")
        self.train_fingerprint = fingerprint
        output.mkdir(parents=True, exist_ok=True)
        trainer = TimedTrainer(self.model, self.tokenizer)
        evaluator = ToyEvaluator(self.model, self.tokenizer)
        report: dict[str, Any] = {
            "seed": self.seed, "parameters": sum(p.numel() for p in self.model.parameters()),
            "config": asdict(self.model.config), "train_samples": len(samples),
            "eval_samples": len(validation), "evaluation": "greedy, fixed external wakeups",
            "initial": evaluator.evaluate(validation), "history": [],
        }
        start = time.monotonic()
        losses: list[float] = []
        for i in range(steps):
            if not self.sample_order:
                self.sample_order = list(range(len(samples)))
                random.shuffle(self.sample_order)
            result = trainer.step(samples[self.sample_order.pop()])
            losses.append(result.loss)
            self.completed_steps += 1
            if (i + 1) % eval_every == 0 or i + 1 == steps:
                metrics = evaluator.evaluate(validation)
                entry = {
                    "step": self.completed_steps, "loss": sum(losses) / len(losses),
                    "elapsed_seconds": time.monotonic() - start, **metrics,
                }
                print(json.dumps(entry), flush=True)
                report["history"].append(entry)
                losses.clear()
                self.save(output / "checkpoint.pt")
                (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        report["final"] = evaluator.evaluate(validation)
        report["without_cross_window"] = evaluator.evaluate(validation, cross_window=False)
        report["steps"] = self.completed_steps
        report["elapsed_seconds"] = time.monotonic() - start
        report["examples"] = [
            {"sample_id": sample.sample_id, **asdict(evaluator.rollout(sample))}
            for sample in validation[:5]
        ]
        (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"final": report["final"], "ablation": report["without_cross_window"]}), flush=True)
        return report
