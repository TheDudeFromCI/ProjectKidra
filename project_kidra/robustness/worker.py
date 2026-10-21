"""Resume focused runs without reopening or modifying an original experiment directory."""

import json
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from project_kidra.robustness.evaluate import RobustEvaluation
from project_kidra.robustness.transforms import augment
from project_kidra.robustness.plan import RobustPlan
from project_kidra.robustness.spec import RobustSpec
from project_kidra.temporal.runtime.evaluate import TemporalDiagnostics
from project_kidra.temporal.runtime.trainer import TemporalTrainer
from project_kidra.timed.execution.checkpoint import CheckpointStore
from project_kidra.timed.experiment import ToyExperiment
from project_kidra.timed.matrix.storage import AtomicFiles
from project_kidra.timed.matrix.worker import MatrixWorker
from project_kidra.timed.trainer import TimedTrainer


class RobustWorker(MatrixWorker):
    """Reuse the established job lock, stop handlers, status publishing, and completed-run guard."""

    def __init__(self, root: Path, name: str, device: str, threads: int = 2) -> None:
        """Resolve only copied follow-up inputs; inherited run() dispatches to this execution method."""

        self.root, self.device, self.threads = root, device, threads
        self.manifest = RobustPlan.load(root)
        self.spec = next((s for s in map(RobustSpec.from_dict, self.manifest["runs"]) if s.name == name), None)
        if self.spec is None:
            raise ValueError(f"unknown follow-up run: {name}")
        self.directory = root / "experiments" / name
        self.identity = self.spec.identity(self.manifest["identity"])
        self.stop = False

    def _execute(self) -> int:
        """Audit fixed weights, train a new seed, or continue the exact final optimizer state."""

        assert isinstance(self.spec, RobustSpec)
        spec, options = self.spec, self.manifest["options"]
        experiment = ToyExperiment(spec.seed, self.device)
        model, tok = experiment.model, experiment.tokenizer
        for optimizer in (model.trunk_optimizer, model.expert_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = spec.learning_rate
        data = {name: experiment.read(self.root / record["path"])
                for name, record in self.manifest["datasets"].items()}
        if len(data["train"]) != 50000:
            raise ValueError("follow-up requires the same 50000 training records")
        store = CheckpointStore(self.directory)
        state = store.load(self.identity)
        self.steps = 0
        history: list[dict[str, Any]] = []
        loss_sum, loss_count, elapsed = 0.0, 0, 0.0
        if state is None and spec.source:
            record = self.manifest["sources"][spec.source]
            state = torch.load(self.root / record["path"], map_location="cpu", weights_only=True)
            if (state["identity"] != record["identity"] or state["steps"] != record["steps"]
                    or state["format"] != "kidra-matrix-1"):
                raise ValueError("copied checkpoint does not match its completed source run")
        if state is not None:
            if state["config"] != asdict(model.config):
                raise ValueError("checkpoint architecture mismatch")
            model.load_state_dict(state["model"])
            model.trunk_optimizer.load_state_dict(state["trunk_optimizer"])
            model.expert_optimizer.load_state_dict(state["expert_optimizer"])
            self.steps = state["steps"]
            torch.set_rng_state(state["torch_rng"])
            random.setstate(state["python_rng"])
            if self.device.startswith("cuda") and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state(state["cuda_rng"], self.device)
            if state["identity"] == self.identity:
                history, loss_sum, loss_count = state["history"], state["loss_sum"], state["loss_count"]
                elapsed = state["elapsed"]
        if any(group["lr"] != spec.learning_rate
               for optimizer in (model.trunk_optimizer, model.expert_optimizer)
               for group in optimizer.param_groups):
            raise ValueError("checkpoint optimizer learning rate does not match the experiment")
        if not 0 <= self.steps <= spec.budget:
            raise ValueError("checkpoint cursor outside follow-up budget")
        AtomicFiles.json(self.directory / "environment.json", {
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "device": self.device,
            "device_name": torch.cuda.get_device_name(self.device) if self.device.startswith("cuda") else "CPU",
            "threads": self.threads, "parameters": sum(p.numel() for p in model.parameters()),
            "config": asdict(model.config), "identity": self.identity,
            "optimizer_learning_rates": [[g["lr"] for g in o.param_groups] for o in
                                         (model.trunk_optimizer, model.expert_optimizer)],
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        })
        evaluator = TemporalDiagnostics(model, tok, spec.variant)
        trainer = (TimedTrainer(model, tok, policy=spec.policy) if spec.variant == "none" else
                   TemporalTrainer(model, tok, policy=spec.policy, variant=spec.variant))
        started = time.monotonic()

        def checkpoint() -> None:
            """Commit only complete samples, including optimizer moments and the global cursor."""

            if model.windows:
                raise RuntimeError("cannot checkpoint a partial trajectory")
            store.save({
                "format": "kidra-matrix-1", "identity": self.identity, "spec": spec.to_dict(),
                "config": asdict(model.config), "steps": self.steps, "history": history,
                "loss_sum": loss_sum, "loss_count": loss_count,
                "elapsed": elapsed + time.monotonic() - started,
                "model": model.state_dict(), "trunk_optimizer": model.trunk_optimizer.state_dict(),
                "expert_optimizer": model.expert_optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.startswith("cuda") else None,
            })
            AtomicFiles.json(self.directory / "history.json", history)
            self._status("training", self.steps)

        epoch, order = -1, []
        if spec.action != "audit":
            checkpoint()
            while self.steps < spec.budget and not self.stop:
                next_epoch = self.steps // len(data["train"])
                if epoch != next_epoch:
                    epoch = next_epoch
                    order = list(range(len(data["train"])))
                    random.Random(spec.order_seed + 1000003 + epoch).shuffle(order)
                sample = data["train"][order[self.steps % len(order)]]
                result = trainer.step(augment(sample, spec.augmentation, self.steps))
                self.steps += 1
                loss_sum += result.loss
                loss_count += 1
                if self.steps % options["checkpoint_every"] == 0:
                    checkpoint()
                if self.steps % options["eval_every"] == 0 or self.steps == spec.budget:
                    entry = {"step": self.steps, "loss": loss_sum / max(loss_count, 1),
                             "iid": evaluator.evaluate(data["iid"][:options["periodic_samples"]]),
                             "elapsed_seconds": elapsed + time.monotonic() - started}
                    history.append(entry)
                    print(json.dumps(entry), flush=True)
                    loss_sum, loss_count = 0.0, 0
                    checkpoint()
            checkpoint()
        if self.stop:
            self._status("interrupted", self.steps)
            return 130
        self._status("evaluating", self.steps)
        finished = RobustEvaluation(model, tok, spec.variant).run(
            self.directory, self.identity, spec.to_dict(), data, lambda: self.stop)
        if not finished or self.stop:
            self._status("interrupted", self.steps)
            return 130
        path = self.directory / "results.json"
        report = AtomicFiles.read(path)
        report.update(steps=self.steps, parameters=sum(p.numel() for p in model.parameters()),
                      elapsed_seconds=elapsed + time.monotonic() - started)
        AtomicFiles.json(path, report)
        AtomicFiles.json(self.directory / "complete.json", {
            "identity": self.identity, "steps": self.steps, "results_sha256": AtomicFiles.digest(path),
        })
        self._status("completed", self.steps)
        return 0
