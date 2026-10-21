"""Bounded concurrent subprocess scheduling with durable per-run ownership."""

import fnmatch
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Any

from project_kidra.robustness.plan import RobustPlan
from project_kidra.robustness.spec import RobustSpec
from project_kidra.timed.matrix.storage import AtomicFiles, FileLock


class RobustScheduler:
    """Refill explicit device slots; checkpoints and completion markers belong to workers."""

    def __init__(
        self, root: Path, slots: list[str], threads: int = 2, retries: int = 1,
        only: str = "*", poll_seconds: float = 2.0,
    ) -> None:
        """A repeated device name allocates another concurrent process on that device."""

        if not slots or threads < 1 or retries < 0 or poll_seconds <= 0:
            raise ValueError("invalid scheduler limits")
        self.root, self.slots, self.threads = root.resolve(), slots, threads
        self.retries, self.only, self.poll_seconds = retries, only, poll_seconds
        self.stop = False
        self.children: dict[str, tuple[subprocess.Popen[bytes], str]] = {}

    def request_stop(self, signum: int, frame: FrameType | None) -> None:
        """Stop dispatching and ask owned children to checkpoint at a safe boundary."""

        self.stop = True
        for process, _ in self.children.values():
            if process.poll() is None:
                process.terminate()

    def completed(self, spec: RobustSpec, manifest_id: str) -> bool:
        """Skip only a matching completion marker backed by an intact report."""

        directory = self.root / "experiments" / spec.name
        path = directory / "complete.json"
        if not path.exists():
            return False
        marker = AtomicFiles.read(path)
        if marker["identity"] != spec.identity(manifest_id):
            raise ValueError("completed experiment identity mismatch")
        report = directory / "results.json"
        if not report.exists() or marker["results_sha256"] != AtomicFiles.digest(report):
            raise ValueError(f"completed report is missing or altered: {spec.name}")
        return True

    def _launch(self, spec: RobustSpec, device: str) -> subprocess.Popen[bytes]:
        """Isolate each run's CUDA allocator, optimizer, stdout, and error log."""

        directory = self.root / "experiments" / spec.name
        directory.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS=str(self.threads), MKL_NUM_THREADS=str(self.threads))
        command = [sys.executable, "-m", "project_kidra.robustness", "worker",
                   "--root", str(self.root), "--name", spec.name, "--device", device,
                   "--threads", str(self.threads)]
        with (directory / "worker.log").open("ab") as log:
            return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)

    def run(self) -> int:
        """Resume interrupted work, detect surviving workers, and never spin on failed jobs."""

        manifest = RobustPlan.load(self.root)
        all_specs = [RobustSpec.from_dict(r) for r in manifest["runs"]]
        specs = [s for s in all_specs if fnmatch.fnmatchcase(s.name, self.only)]
        if not specs:
            raise ValueError("run filter matched no experiments")
        lock = FileLock(self.root / "scheduler.lock")
        if not lock.acquire():
            raise RuntimeError("another scheduler owns this suite")
        handlers: dict[int, Any] = {}
        attempts: dict[str, int] = {}
        failed: set[str] = set()
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, self.request_stop)
            while True:
                for name, (process, _) in list(self.children.items()):
                    code = process.poll()
                    if code is not None:
                        del self.children[name]
                        completed_path = self.root / "experiments" / name / "complete.json"
                        if code != 75 and not completed_path.exists() and attempts[name] > self.retries:
                            failed.add(name)
                        print(f"exit {name}: {code}", flush=True)
                free = list(self.slots)
                busy = set(self.children)
                for _, device in self.children.values():
                    if device in free:
                        free.remove(device)
                # A killed scheduler can leave healthy children behind. Their
                # kernel locks, not stale status text or PID reuse, prove liveness.
                for spec in all_specs:
                    if spec.name in busy:
                        continue
                    probe = FileLock(self.root / "experiments" / spec.name / "worker.lock")
                    if probe.acquire():
                        probe.release()
                    else:
                        busy.add(spec.name)
                        state_path = self.root / "experiments" / spec.name / "status.json"
                        external_device = AtomicFiles.read(state_path).get("device") if state_path.exists() else None
                        if external_device in free:
                            free.remove(external_device)
                        elif free:
                            free.pop()
                completed = {s.name for s in specs if self.completed(s, manifest["identity"])}
                pending = [s for s in specs if s.name not in completed | busy | failed]
                if self.stop:
                    if not self.children:
                        AtomicFiles.json(self.root / "scheduler.json", {
                            "completed": len(completed), "total": len(specs), "running": [],
                            "failed": sorted(failed), "pending": len(pending), "stopped": True,
                            "updated": time.time(),
                        })
                        return 130
                elif not pending and not any(s.name in busy for s in specs):
                    AtomicFiles.json(self.root / "scheduler.json", {
                        "completed": len(completed), "total": len(specs), "running": [],
                        "failed": sorted(failed), "pending": 0, "updated": time.time(),
                    })
                    return 1 if failed else 0
                else:
                    for spec, device in zip(pending, free):
                        attempts[spec.name] = attempts.get(spec.name, 0) + 1
                        try:
                            self.children[spec.name] = (self._launch(spec, device), device)
                            print(f"start {spec.name} on {device}", flush=True)
                        except OSError as exc:
                            failed.add(spec.name)
                            print(f"launch failed {spec.name}: {exc}", flush=True)
                AtomicFiles.json(self.root / "scheduler.json", {
                    "completed": len(completed), "total": len(specs),
                    "running": sorted(busy), "failed": sorted(failed),
                    "pending": len(pending), "updated": time.time(),
                })
                time.sleep(self.poll_seconds)
        finally:
            # A scheduler exception leaves healthy workers running and discoverable.
            for previous_sig, handler in handlers.items():
                signal.signal(previous_sig, handler)
            lock.release()
