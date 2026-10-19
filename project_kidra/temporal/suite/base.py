"""Read-only readiness checks for the still-running original suite."""

import fcntl
from pathlib import Path
from typing import Any

from project_kidra.timed.matrix.plan import MatrixPlan
from project_kidra.timed.matrix.scheduler import MatrixScheduler
from project_kidra.timed.matrix.spec import RunSpec


class BaseSuite:
    """Wait for original reports and live workers without creating parent lock files."""

    def __init__(self, provenance: dict[str, Any]) -> None:
        """Verify the referenced parent manifest before considering it complete."""

        self.root = Path(provenance["root"])
        self.manifest = MatrixPlan.load(self.root, verify_data=False)
        if self.manifest["identity"] != provenance["identity"]:
            raise ValueError("original suite identity differs from temporal provenance")
        self.specs = [RunSpec.from_dict(r) for r in self.manifest["runs"]]
        self.reader = MatrixScheduler(self.root, ["cpu"])

    @staticmethod
    def locked(path: Path) -> bool:
        """Probe an existing lock read-only; never create or rewrite a parent file."""

        if not path.exists():
            return False
        with path.open("rb") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(handle, fcntl.LOCK_UN)
        return False

    def progress(self) -> tuple[int, int, bool]:
        """Report completion and whether any original worker still holds its lock."""

        complete = sum(self.reader.completed(s, self.manifest["identity"]) for s in self.specs)
        busy = any(self.locked(self.root / "experiments" / s.name / "worker.lock")
                   for s in self.specs)
        return complete, len(self.specs), busy

    def ready(self) -> bool:
        """Require every original run to finish, including final diagnostics."""

        complete, total, busy = self.progress()
        return complete == total and not busy
