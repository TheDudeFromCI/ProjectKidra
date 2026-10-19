"""Use harmless subprocesses to verify concurrency and restart semantics without training."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from project_kidra.timed.matrix.plan import MatrixPlan
from project_kidra.timed.matrix.scheduler import MatrixScheduler
from project_kidra.timed.matrix.spec import RunSpec
from project_kidra.timed.matrix.storage import AtomicFiles, FileLock


def test_device_slots_retry_refill_and_skip_completed(tmp_path: Path) -> None:
    """Two slots stay bounded, a failed job retries, and restarting does no completed work."""

    manifest = MatrixPlan.prepare(tmp_path, samples=2, eval_samples=1)
    scheduler = MatrixScheduler(tmp_path, ["cpu", "cpu"], only="echo-full-packed-*", poll_seconds=0.01)
    attempts: dict[str, int] = {}
    peak = 0
    script = """
import json, sys, time
from pathlib import Path
from project_kidra.timed.matrix.storage import AtomicFiles
directory, identity, fail = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'yes'
directory.mkdir(parents=True, exist_ok=True)
time.sleep(0.08)
if fail:
    raise SystemExit(2)
AtomicFiles.json(directory / 'results.json', {'identity': identity, 'metrics': {}})
AtomicFiles.json(directory / 'complete.json', {'identity': identity, 'steps': 2,
    'results_sha256': AtomicFiles.digest(directory / 'results.json')})
"""

    def launch(spec: RunSpec, device: str) -> subprocess.Popen[bytes]:
        """Launch only a filesystem-writing stub, never a model worker."""

        nonlocal peak
        attempts[spec.name] = attempts.get(spec.name, 0) + 1
        peak = max(peak, 1 + sum(p.poll() is None for p, _ in scheduler.children.values()))
        assert device == "cpu"
        fail = len(attempts) == 1 and attempts[spec.name] == 1
        return subprocess.Popen([sys.executable, "-c", script,
                                 str(tmp_path / "experiments" / spec.name),
                                 spec.identity(manifest["identity"]), "yes" if fail else "no"])

    with patch.object(scheduler, "_launch", launch):
        assert scheduler.run() == 0
    assert len(attempts) == 5 and sum(attempts.values()) == 6
    assert peak == 2
    with patch.object(scheduler, "_launch", side_effect=AssertionError("completed run launched")):
        assert scheduler.run() == 0


def test_surviving_worker_reserves_a_slot_and_is_not_duplicated(tmp_path: Path) -> None:
    """A live lock survives scheduler restart even if its owner is not a tracked child."""

    manifest = MatrixPlan.prepare(tmp_path, samples=2, eval_samples=1)
    spec = RunSpec.from_dict(manifest["runs"][0])
    directory = tmp_path / "experiments" / spec.name
    external = FileLock(directory / "worker.lock")
    assert external.acquire()
    AtomicFiles.json(directory / "status.json", {"status": "training", "device": "cpu"})
    scheduler = MatrixScheduler(tmp_path, ["cpu"], only=spec.name, poll_seconds=0.01)

    def finish(_seconds: float) -> None:
        """Simulate an existing worker completing during the first scheduler poll."""

        AtomicFiles.json(directory / "results.json", {"metrics": {}})
        AtomicFiles.json(directory / "complete.json", {
            "identity": spec.identity(manifest["identity"]),
            "results_sha256": AtomicFiles.digest(directory / "results.json"),
        })
        external.release()

    try:
        with (patch.object(scheduler, "_launch", side_effect=AssertionError("duplicate worker")),
              patch("project_kidra.timed.matrix.scheduler.time.sleep", finish)):
            assert scheduler.run() == 0
    finally:
        external.release()


def test_scheduler_lock_and_corrupt_completed_report(tmp_path: Path) -> None:
    """A second scheduler is refused, and completion cannot hide missing results."""

    manifest = MatrixPlan.prepare(tmp_path, samples=2, eval_samples=1)
    lock = FileLock(tmp_path / "scheduler.lock")
    assert lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="another scheduler"):
            MatrixScheduler(tmp_path, ["cpu"]).run()
    finally:
        lock.release()
    spec = RunSpec.from_dict(manifest["runs"][0])
    AtomicFiles.json(tmp_path / "experiments" / spec.name / "complete.json", {
        "identity": spec.identity(manifest["identity"]), "results_sha256": "missing",
    })
    with pytest.raises(ValueError, match="missing or altered"):
        MatrixScheduler(tmp_path, ["cpu"], only=spec.name).run()


def test_failures_have_bounded_retries(tmp_path: Path) -> None:
    """A permanently failing job cannot hold up the queue forever."""

    manifest = MatrixPlan.prepare(tmp_path, samples=2, eval_samples=1)
    spec = RunSpec.from_dict(manifest["runs"][0])
    scheduler = MatrixScheduler(tmp_path, ["cpu"], only=spec.name, retries=1, poll_seconds=0.01)

    def fail(_spec: RunSpec, _device: str) -> subprocess.Popen[bytes]:
        """Return an unsuccessful harmless child for each allowed attempt."""

        return subprocess.Popen([sys.executable, "-c", "raise SystemExit(2)"])

    with patch.object(scheduler, "_launch", side_effect=fail) as launch:
        assert scheduler.run() == 1
    assert launch.call_count == 2
