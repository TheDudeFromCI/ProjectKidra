"""Crash-safe files and kernel-owned locks; no stale PID deletion is needed."""

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO


class AtomicFiles:
    """Commit JSON and completed byte streams with atomic replacement and fsync."""

    @staticmethod
    def json(path: Path, value: object) -> None:
        """Replace a JSON file only after all new bytes have reached the filesystem."""

        AtomicFiles.bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())

    @staticmethod
    def bytes(path: Path, data: bytes) -> None:
        """Keep the old destination valid if writing the replacement fails."""

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as target:
                target.write(data)
                target.flush()
                os.fsync(target.fileno())
            os.replace(name, path)
            AtomicFiles.sync_directory(path.parent)
        finally:
            Path(name).unlink(missing_ok=True)

    @staticmethod
    def sync_directory(path: Path) -> None:
        """Persist rename metadata on POSIX filesystems."""

        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def read(path: Path) -> dict[str, Any]:
        """Read a previously committed JSON object."""

        with path.open() as source:
            result: dict[str, Any] = json.load(source)
        return result

    @staticmethod
    def digest(path: Path) -> str:
        """Hash large datasets without retaining their encoded bytes in memory."""

        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()

    @staticmethod
    def identity(value: object) -> str:
        """Create a stable identity for an experiment specification."""

        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class FileLock:
    """Hold an advisory lock released automatically on exit, crash, or machine restart."""

    def __init__(self, path: Path) -> None:
        """Retain the path; acquisition is explicit and nonblocking."""

        self.path = path
        self.handle: BinaryIO | None = None

    def acquire(self) -> bool:
        """Return false while another live process owns this exact work item."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        self.handle = handle
        return True

    def release(self) -> None:
        """Release only this instance's lock, without deleting the shared inode."""

        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
