"""Isolation, launch gating, and recovery checks with no experiment training."""

import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from test_timed_matrix import FixtureExperiment

from project_kidra.temporal.suite.base import BaseSuite
from project_kidra.temporal.suite.plan import TemporalPlan
from project_kidra.temporal.suite.scheduler import TemporalScheduler
from project_kidra.temporal.suite.spec import TemporalSpec
from project_kidra.temporal.suite.worker import TemporalWorker
from project_kidra.timed.execution.checkpoint import CheckpointStore
from project_kidra.timed.matrix.plan import MatrixPlan
from project_kidra.timed.matrix.spec import RunSpec
from project_kidra.timed.matrix.storage import AtomicFiles, FileLock
from project_kidra.timed.trainer import TrainResult


def finish_base(root: Path, manifest: dict[str, Any]) -> None:
    """Create fixture completion reports only; no original workers are run."""

    for raw in manifest["runs"]:
        spec = RunSpec.from_dict(raw)
        directory = root / "experiments" / spec.name
        AtomicFiles.json(directory / "results.json", {"metrics": {}})
        AtomicFiles.json(directory / "complete.json", {
            "identity": spec.identity(manifest["identity"]),
            "results_sha256": AtomicFiles.digest(directory / "results.json"),
        })


def snapshot(root: Path) -> dict[str, str]:
    """Include names and bytes so even newly created lock files violate isolation."""

    return {str(p.relative_to(root)): AtomicFiles.digest(p) for p in root.rglob("*") if p.is_file()}


def test_extension_does_not_touch_original_and_copies_identical_records(tmp_path: Path) -> None:
    """Preparation and waiting checks preserve both the base tree and code fingerprint."""

    base, root = tmp_path / "base", tmp_path / "temporal"
    parent = MatrixPlan.prepare(base, samples=3, eval_samples=2)
    original, digest = snapshot(base), MatrixPlan.source_digest()
    manifest = TemporalPlan.prepare(base, root)
    assert TemporalPlan.prepare(base, root) == manifest
    assert len(manifest["runs"]) == 2
    assert manifest["base"]["baseline"] == "agent-full-tick-all-lr0.0001-s42"
    assert manifest["options"]["samples"] == 3
    for name, record in manifest["datasets"].items():
        assert (root / record["path"]).read_bytes() == (base / parent["datasets"][name]["path"]).read_bytes()
    assert not BaseSuite(manifest["base"]).ready()
    assert snapshot(base) == original
    assert MatrixPlan.source_digest() == digest
    assert MatrixPlan.load(base) == parent
    with pytest.raises(ValueError, match="separate"):
        TemporalPlan.prepare(base, base / "extension")
    with pytest.raises(ValueError, match="baseline is absent"):
        TemporalPlan.prepare(base, tmp_path / "other", learning_rate=0.002)
    (root / "data" / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="dataset changed"):
        TemporalPlan.load(root)


def test_base_readiness_requires_reports_and_released_workers(tmp_path: Path) -> None:
    """An early complete marker cannot allow overlap with a still-live parent worker."""

    base, root = tmp_path / "base", tmp_path / "temporal"
    parent = MatrixPlan.prepare(base, samples=1, eval_samples=1)
    manifest = TemporalPlan.prepare(base, root)
    finish_base(base, parent)
    spec = RunSpec.from_dict(parent["runs"][0])
    lock = FileLock(base / "experiments" / spec.name / "worker.lock")
    assert lock.acquire()
    try:
        before = snapshot(base)
        assert not BaseSuite(manifest["base"]).ready()
        assert snapshot(base) == before
    finally:
        lock.release()
    assert BaseSuite(manifest["base"]).ready()


def test_scheduler_waits_then_runs_two_slots_and_skips_completed(tmp_path: Path) -> None:
    """Harmless report-writing children test slot dispatch and restarting, never training."""

    base, root = tmp_path / "base", tmp_path / "temporal"
    parent = MatrixPlan.prepare(base, samples=1, eval_samples=1)
    manifest = TemporalPlan.prepare(base, root)
    scheduler = TemporalScheduler(root, ["cpu", "cpu"], poll_seconds=0.01)
    assert not BaseSuite(manifest["base"]).ready()
    with (
        patch.object(scheduler, "_launch", side_effect=AssertionError("early GPU launch")),
        patch("project_kidra.temporal.suite.scheduler.time.sleep",
              side_effect=lambda _: scheduler.request_stop(15, None)) as waited,
    ):
        assert scheduler.run() == 130
    assert waited.call_count == 1
    finish_base(base, parent)
    scheduler.stop = False
    peak, launches = 0, 0
    script = '''
import sys, time
from pathlib import Path
from project_kidra.timed.matrix.storage import AtomicFiles
directory, identity = Path(sys.argv[1]), sys.argv[2]
time.sleep(0.04)
AtomicFiles.json(directory / "results.json", {"metrics": {}})
AtomicFiles.json(directory / "complete.json", {"identity": identity,
    "results_sha256": AtomicFiles.digest(directory / "results.json")})
'''

    def launch(spec: TemporalSpec, device: str) -> subprocess.Popen[bytes]:
        """Replace the model subprocess with a short completion-marker fixture."""

        nonlocal peak, launches
        launches += 1
        peak = max(peak, len(scheduler.children) + 1)
        return subprocess.Popen([sys.executable, "-c", script,
                                 str(root / "experiments" / spec.name),
                                 spec.identity(manifest["identity"])])

    with patch.object(scheduler, "_launch", launch):
        assert scheduler.run() == 0
    assert peak == launches == 2
    with patch.object(scheduler, "_launch", side_effect=AssertionError("completed job rerun")):
        assert scheduler.run() == 0


def test_worker_checkpoint_resume_and_completed_skip_without_training(tmp_path: Path) -> None:
    """Mock training and evaluation; exercise actual worker cursor/checkpoint lifecycle."""

    base, root = tmp_path / "base", tmp_path / "temporal"
    parent = MatrixPlan.prepare(base, samples=3, eval_samples=1, checkpoint_every=1)
    manifest = TemporalPlan.prepare(base, root)
    finish_base(base, parent)
    spec = TemporalSpec.from_dict(manifest["runs"][0])
    worker = TemporalWorker(root, spec.name, "cpu", 1)
    with (
        patch("project_kidra.temporal.suite.worker.ToyExperiment", FixtureExperiment),
        patch("project_kidra.temporal.suite.worker.TemporalTrainer", autospec=True) as trainer,
        patch("project_kidra.temporal.suite.worker.TemporalDiagnostics", autospec=True) as evaluator,
    ):
        trainer.return_value.step.side_effect = [TrainResult(1.0, 1, 1), RuntimeError("fixture crash")]
        evaluator.return_value.evaluate.return_value = {"exact": 0.0}
        evaluator.return_value.counterfactual.return_value = {"both_correct": 0.0}
        with pytest.raises(RuntimeError, match="fixture crash"):
            worker.run()
        state = CheckpointStore(worker.directory).load(worker.identity)
        assert state is not None and state["steps"] == 1
        trainer.return_value.step.reset_mock()
        trainer.return_value.step.side_effect = None
        trainer.return_value.step.return_value = TrainResult(1.0, 1, 1)
        assert TemporalWorker(root, spec.name, "cpu", 1).run() == 0
        assert trainer.return_value.step.call_count == 2
        trainer.return_value.step.side_effect = AssertionError("completed run retrained")
        assert TemporalWorker(root, spec.name, "cpu", 1).run() == 0
