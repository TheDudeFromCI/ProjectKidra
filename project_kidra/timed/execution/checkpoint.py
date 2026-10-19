"""Atomic tensor checkpoints with one verified fallback generation."""

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import torch

from project_kidra.timed.matrix.storage import AtomicFiles


class CheckpointStore:
    """Retain the last complete checkpoint if a write or a process is interrupted."""

    def __init__(self, directory: Path) -> None:
        """Assign fixed checkpoint paths inside a locked experiment directory."""

        self.directory = directory
        self.current = directory / "checkpoint.pt"
        self.previous = directory / "checkpoint.previous.pt"

    def save(self, payload: dict[str, Any]) -> None:
        """Flush the new checkpoint before rotating either known-good generation."""

        self.directory.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.directory, suffix=".pt.tmp")
        try:
            with os.fdopen(fd, "wb") as target:
                torch.save(payload, target)
                target.flush()
                os.fsync(target.fileno())
            if self.current.exists():
                os.replace(self.current, self.previous)
            os.replace(name, self.current)
            AtomicFiles.sync_directory(self.directory)
        finally:
            Path(name).unlink(missing_ok=True)

    def load(self, identity: str) -> dict[str, Any] | None:
        """Recover a readable generation, but never accept a different experiment."""

        errors = []
        for path in (self.current, self.previous):
            if not path.exists():
                continue
            try:
                data: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
            except (OSError, EOFError, RuntimeError, pickle.UnpicklingError) as exc:
                errors.append(f"{path.name}: {type(exc).__name__}")
                continue
            if data.get("identity") != identity or data.get("format") != "kidra-matrix-1":
                raise ValueError("checkpoint belongs to a different experiment")
            if path == self.previous:
                print("Recovering previous complete checkpoint", flush=True)
                # Do not rotate a corrupt current file over the valid fallback.
                self.current.unlink(missing_ok=True)
            return data
        if errors:
            raise RuntimeError("no readable checkpoint: " + "; ".join(errors))
        return None
