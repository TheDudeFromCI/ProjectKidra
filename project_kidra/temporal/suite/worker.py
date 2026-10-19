"""One locked experiment, resumable at sample boundaries and final-evaluation boundaries."""

import json
import os
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path
from types import FrameType
from typing import Any

import torch

from project_kidra.temporal.runtime.evaluate import TemporalDiagnostics
from project_kidra.temporal.runtime.trainer import TemporalTrainer
from project_kidra.temporal.suite.base import BaseSuite
from project_kidra.temporal.suite.plan import TemporalPlan
from project_kidra.temporal.suite.spec import TemporalSpec
from project_kidra.timed.execution.checkpoint import CheckpointStore
from project_kidra.timed.experiment import ToyExperiment
from project_kidra.timed.matrix.storage import AtomicFiles, FileLock
from project_kidra.timed.matrix.tasks import TaskProjection
from project_kidra.timed.schema import TimedSample


class TemporalWorker:
    """Train the same model and record order under a single scientific condition."""

    def __init__(self, root: Path, name: str, device: str, threads: int = 2) -> None:
        """Load immutable metadata; model construction happens after acquiring the job lock."""

        self.root, self.device, self.threads = root, device, threads
        self.manifest = TemporalPlan.load(root)
        self.spec = next((r for r in map(TemporalSpec.from_dict, self.manifest["runs"])
                          if r.name == name), None)
        if self.spec is None:
            raise ValueError(f"unknown run: {name}")
        self.directory = root / "experiments" / name
        self.identity = self.spec.identity(self.manifest["identity"])
        self.stop = False

    def request_stop(self, signum: int, frame: FrameType | None) -> None:
        """Finish the current sample, commit a checkpoint, and stop without partial updates."""

        self.stop = True

    def _status(self, status: str, steps: int, **extra: object) -> None:
        """Publish progress atomically for the scheduler and status command."""

        AtomicFiles.json(self.directory / "status.json", {
            "identity": self.identity, "status": status, "steps": steps,
            "device": self.device, "pid": os.getpid(), "updated": time.time(), **extra,
        })

    def run(self) -> int:
        """Avoid duplicate workers and leave durable state on completion or graceful stop."""

        lock = FileLock(self.directory / "worker.lock")
        if not lock.acquire():
            return 75
        previous_handlers: dict[int, Any] = {}
        try:
            done = self.directory / "complete.json"
            if done.exists():
                marker = AtomicFiles.read(done)
                if marker["identity"] != self.identity:
                    raise ValueError("completion marker identity mismatch")
                report = self.directory / "results.json"
                if not report.exists() or AtomicFiles.digest(report) != marker["results_sha256"]:
                    raise ValueError("completed report is missing or altered")
                return 0
            if not BaseSuite(self.manifest["base"]).ready():
                raise RuntimeError("original suite is still running; use the temporal scheduler to wait")
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, self.request_stop)
            torch.set_num_threads(self.threads)
            self._status("starting", 0)
            return self._execute()
        except Exception as exc:
            self._status("failed", getattr(self, "steps", 0), error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            for previous_signum, handler in previous_handlers.items():
                signal.signal(previous_signum, handler)
            lock.release()

    def _execute(self) -> int:
        """Restore the next record exactly, train to a total budget, then finish diagnostics."""

        assert self.spec is not None
        spec, options = self.spec, self.manifest["options"]
        experiment = ToyExperiment(spec.seed, self.device)
        model, tokenizer = experiment.model, experiment.tokenizer
        for optimizer in (model.trunk_optimizer, model.expert_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = spec.learning_rate
        data: dict[str, list[TimedSample]] = {}
        for name, record in self.manifest["datasets"].items():
            samples = experiment.read(self.root / record["path"])
            data[name] = [TaskProjection.apply(sample, spec.task) for sample in samples]
        if len(data["train"]) != options["samples"]:
            raise ValueError("training count differs from manifest")
        # A fixed permutation gives every run the same records at every step.
        order = list(range(len(data["train"])))
        random.Random(spec.seed + 1000003).shuffle(order)
        store = CheckpointStore(self.directory)
        state = store.load(self.identity)
        self.steps = 0
        history: list[dict[str, Any]] = []
        loss_sum, loss_count, elapsed = 0.0, 0, 0.0
        if state:
            if state["config"] != asdict(model.config):
                raise ValueError("checkpoint model configuration mismatch")
            model.load_state_dict(state["model"])
            model.trunk_optimizer.load_state_dict(state["trunk_optimizer"])
            model.expert_optimizer.load_state_dict(state["expert_optimizer"])
            self.steps, history = state["steps"], state["history"]
            if not 0 <= self.steps <= options["samples"]:
                raise ValueError("checkpoint cursor is outside this suite's budget")
            loss_sum, loss_count, elapsed = state["loss_sum"], state["loss_count"], state["elapsed"]
            torch.set_rng_state(state["torch_rng"])
            random.setstate(state["python_rng"])
            if self.device.startswith("cuda") and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state(state["cuda_rng"], self.device)
        if self.steps < options["samples"] and (self.directory / "results.json").exists():
            os.replace(self.directory / "results.json", self.directory / "results.stale.json")
        trainer = TemporalTrainer(model, tokenizer, policy=spec.policy, variant=spec.variant)
        evaluator = TemporalDiagnostics(model, tokenizer, spec.variant)
        started = time.monotonic()

        def checkpoint() -> None:
            """Persist sample cursor, optimizers, RNGs, and unreported loss together."""

            if model.windows:
                raise RuntimeError("cannot save a partial trajectory")
            store.save({
                "format": "kidra-matrix-1", "identity": self.identity,
                "config": asdict(model.config), "spec": spec.to_dict(),
                "steps": self.steps, "history": history, "loss_sum": loss_sum,
                "loss_count": loss_count, "elapsed": elapsed + time.monotonic() - started,
                "model": model.state_dict(), "trunk_optimizer": model.trunk_optimizer.state_dict(),
                "expert_optimizer": model.expert_optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state(self.device)
                    if self.device.startswith("cuda") else None,
            })
            self._status("training", self.steps)

        # Persist initialization too, so early crashes also have a recoverable state.
        if state is None:
            checkpoint()
        while self.steps < options["samples"] and not self.stop:
            result = trainer.step(data["train"][order[self.steps]])
            self.steps += 1
            loss_sum += result.loss
            loss_count += 1
            if self.steps % options["checkpoint_every"] == 0:
                checkpoint()
            if self.steps % options["eval_every"] == 0 or self.steps == options["samples"]:
                size = options["periodic_samples"]
                metrics = {
                    "step": self.steps, "loss": loss_sum / max(loss_count, 1),
                    "elapsed_seconds": elapsed + time.monotonic() - started,
                    "train_probe": evaluator.evaluate(
                        [data["train"][i] for i in order[:min(self.steps, size)]],
                        cross_window=spec.policy.cross_window),
                    "iid": evaluator.evaluate(data["iid"][:size], cross_window=spec.policy.cross_window),
                }
                history.append(metrics)
                print(json.dumps(metrics), flush=True)
                loss_sum, loss_count = 0.0, 0
                checkpoint()
                AtomicFiles.json(self.directory / "history.json", history)
        checkpoint()
        if self.stop:
            self._status("interrupted", self.steps)
            return 130
        self._status("evaluating", self.steps)
        report_path = self.directory / "results.json"
        report: dict[str, Any] = (AtomicFiles.read(report_path) if report_path.exists() else {
            "identity": self.identity, "spec": spec.to_dict(), "steps": self.steps,
            "parameters": sum(p.numel() for p in model.parameters()), "metrics": {},
        })
        if report["identity"] != self.identity:
            raise ValueError("evaluation report identity mismatch")
        groups = {"train_probe": [data["train"][i] for i in order[:options["eval_samples"]]],
                  **{k: v for k, v in data.items() if k != "train"}}
        for name, samples in groups.items():
            for mode in ("free", "conditional", "noexchange"):
                key = name + "/" + mode
                if key not in report["metrics"]:
                    report["metrics"][key] = evaluator.evaluate(
                        samples, cross_window=(spec.policy.cross_window and mode != "noexchange"),
                        oracle_prefix=mode == "conditional",
                    )
                    AtomicFiles.json(report_path, report)
                if self.stop:
                    self._status("interrupted", self.steps)
                    return 130
        if "counterfactual" not in report["metrics"]:
            report["metrics"]["counterfactual"] = evaluator.counterfactual(
                data["iid"], spec.task, spec.policy.cross_window)
        report["elapsed_seconds"] = elapsed + time.monotonic() - started
        AtomicFiles.json(report_path, report)
        if self.stop:
            self._status("interrupted", self.steps)
            return 130
        AtomicFiles.json(self.directory / "complete.json", {
            "identity": self.identity, "steps": self.steps, "results_sha256": AtomicFiles.digest(report_path),
        })
        self._status("completed", self.steps)
        return 0
