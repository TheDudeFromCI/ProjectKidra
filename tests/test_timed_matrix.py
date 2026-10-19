"""Validate matrix coverage, matched data, metrics, and resumable state on tiny fixtures."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from test_timed_execution import tiny_model

from project_kidra.timed.execution.checkpoint import CheckpointStore
from project_kidra.timed.experiment import ToyExperiment
from project_kidra.timed.generator import ToyGenerator
from project_kidra.timed.matrix.plan import MatrixPlan
from project_kidra.timed.matrix.spec import RunSpec
from project_kidra.timed.matrix.storage import AtomicFiles
from project_kidra.timed.matrix.tasks import TaskProjection
from project_kidra.timed.matrix.worker import MatrixWorker
from project_kidra.timed.schedule import Schedule
from project_kidra.timed.schema import TimedSample
from project_kidra.timed.tokenizer import ToyTokenizer
from project_kidra.timed.trainer import TimedTrainer, TrainResult


def test_matrix_coverage_and_matched_task_facts() -> None:
    """The 64 factorial cells plus four controls retain identical answers and IDs."""

    runs = RunSpec.matrix([42])
    assert len(runs) == len({run.name for run in runs}) == 68
    assert sum(not r.policy.cross_window for r in runs) == 4
    generator, tok = ToyGenerator(3), ToyTokenizer()
    positions = []
    for i in range(64):
        sample = generator.sample(i)
        for task in TaskProjection.names:
            projected = TaskProjection.apply(sample, task)
            Schedule(projected, tok, 5, 128)
            assert projected.sample_id == sample.sample_id
            assert projected.streams[-1].text == sample.streams[-1].text
            changed = TaskProjection.counterfactual(projected, task)
            assert changed.group_ends == projected.group_ends
            assert changed.streams[-1].text != projected.streams[-1].text
            if task == "lookup2":
                words = projected.streams[2].text.split()
                key = projected.streams[1].text.split()[-1]
                index = words.index(key)
                positions.append(index)
                assert words[index + 1] == projected.streams[-1].text.split()[-1]
                assert changed.streams[2].text.split()[index + 1] == changed.streams[-1].text.split()[-1]
    assert 0.3 < positions.count(1) / len(positions) < 0.7


def test_preparation_is_idempotent_and_detects_modified_data(tmp_path: Path) -> None:
    """Preparing again cannot silently overwrite datasets or change scientific controls."""

    first = MatrixPlan.prepare(tmp_path, samples=4, eval_samples=2)
    assert MatrixPlan.prepare(tmp_path, samples=4, eval_samples=2) == first
    with pytest.raises(ValueError, match="different options"):
        MatrixPlan.prepare(tmp_path, samples=5, eval_samples=2)
    path = tmp_path / first["datasets"]["train"]["path"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="dataset changed"):
        MatrixPlan.load(tmp_path)


def test_imported_master_records_are_preserved_exactly(tmp_path: Path) -> None:
    """Existing datasets retain their bytes, including blank lines and final-newline choice."""

    import json

    source = tmp_path / "source.jsonl"
    generator = ToyGenerator(1)
    source.write_text("\n".join(json.dumps(generator.sample(i).to_dict()) for i in range(2)) + "\n\n")
    root = tmp_path / "suite"
    manifest = MatrixPlan.prepare(root, samples=2, eval_samples=1, train_path=source)
    assert (root / manifest["datasets"]["train"]["path"]).read_bytes() == source.read_bytes()
    assert MatrixPlan.prepare(root, samples=2, eval_samples=1, train_path=source) == manifest


def test_atomic_checkpoint_falls_back_without_overwriting_good_state(tmp_path: Path) -> None:
    """A partial or corrupt newest checkpoint never destroys the last readable generation."""

    store = CheckpointStore(tmp_path)
    one = {"identity": "fixture", "format": "kidra-matrix-1", "steps": 1}
    store.save(one)
    store.save({**one, "steps": 2})
    store.current.write_bytes(b"incomplete checkpoint")
    assert store.load("fixture") == one
    store.save({**one, "steps": 3})
    assert store.load("fixture")["steps"] == 3  # type: ignore[index]
    assert torch.load(store.previous, weights_only=True) == one
    with pytest.raises(ValueError, match="different experiment"):
        store.load("other")


class FixtureExperiment:
    """Substitute a tiny model to test worker recovery without running any 1M experiment."""

    def __init__(self, seed: int, device: str) -> None:
        """Match model initialization exactly across uninterrupted and resumed fixtures."""

        self.model = tiny_model().to(device)
        self.tokenizer = ToyTokenizer()

    read = staticmethod(ToyExperiment.read)


def test_crash_resume_equals_uninterrupted_updates(tmp_path: Path) -> None:
    """Resume replays no committed samples and restores optimizer moments exactly."""

    interrupted, straight = tmp_path / "interrupted", tmp_path / "straight"
    manifests = [MatrixPlan.prepare(p, samples=4, eval_samples=1, checkpoint_every=1,
                                   eval_every=10, periodic_samples=1) for p in (interrupted, straight)]
    spec = RunSpec.from_dict(manifests[0]["runs"][0])
    original = TimedTrainer.step
    calls = 0

    def crash(trainer: TimedTrainer, sample: TimedSample) -> TrainResult:
        """Simulate failure immediately before the third sample's forward pass."""

        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected crash")
        return original(trainer, sample)

    with (
        patch("project_kidra.timed.matrix.worker.ToyExperiment", FixtureExperiment),
        patch("project_kidra.timed.matrix.worker.Diagnostics", autospec=True) as evaluator,
    ):
        evaluator.return_value.evaluate.return_value = {"exact": 0.0}
        evaluator.return_value.counterfactual.return_value = {"both_correct": 0.0}
        with patch.object(TimedTrainer, "step", crash), pytest.raises(RuntimeError, match="injected crash"):
            MatrixWorker(interrupted, spec.name, "cpu", 1).run()
        failed = AtomicFiles.read(interrupted / "experiments" / spec.name / "status.json")
        assert failed["steps"] == 2
        assert MatrixWorker(interrupted, spec.name, "cpu", 1).run() == 0
        assert MatrixWorker(straight, spec.name, "cpu", 1).run() == 0
        with patch.object(TimedTrainer, "step", side_effect=AssertionError("completed run retrained")):
            assert MatrixWorker(interrupted, spec.name, "cpu", 1).run() == 0
    a = torch.load(interrupted / "experiments" / spec.name / "checkpoint.pt", weights_only=True)
    b = torch.load(straight / "experiments" / spec.name / "checkpoint.pt", weights_only=True)
    assert a["steps"] == b["steps"] == 4
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], atol=0, rtol=0)
    for optimizer in ("trunk_optimizer", "expert_optimizer"):
        for parameter, state in a[optimizer]["state"].items():
            for field, value in state.items():
                torch.testing.assert_close(value, b[optimizer]["state"][parameter][field], atol=0, rtol=0)


def test_stop_commits_the_completed_sample_and_resumes(tmp_path: Path) -> None:
    """SIGTERM-style stopping checkpoints the current completed sample before returning."""

    manifest = MatrixPlan.prepare(tmp_path, samples=3, eval_samples=1, checkpoint_every=3)
    spec = RunSpec.from_dict(manifest["runs"][0])
    worker = MatrixWorker(tmp_path, spec.name, "cpu", 1)
    original = TimedTrainer.step

    def stop_after_step(trainer: TimedTrainer, sample: TimedSample) -> TrainResult:
        """Request stop after one committed sample, before the periodic checkpoint interval."""

        result = original(trainer, sample)
        worker.request_stop(15, None)
        return result

    with (patch("project_kidra.timed.matrix.worker.ToyExperiment", FixtureExperiment),
          patch.object(TimedTrainer, "step", stop_after_step)):
        assert worker.run() == 130
    checkpoint = torch.load(tmp_path / "experiments" / spec.name / "checkpoint.pt", weights_only=True)
    assert checkpoint["steps"] == 1
    assert not (tmp_path / "experiments" / spec.name / "complete.json").exists()
