"""Prepare shared data once and bind every run to the exact same master records."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from project_kidra.timed.generator import ToyGenerator
from project_kidra.timed.matrix.spec import RunSpec
from project_kidra.timed.matrix.storage import AtomicFiles, FileLock


class MatrixPlan:
    """Own the immutable manifest, data fingerprints, and reproducible run order."""

    @staticmethod
    def source_digest() -> str:
        """Reject a silent mixture of different model/training code within one matrix."""

        package = Path(__file__).resolve().parents[1]
        entries = [(str(p.relative_to(package)), AtomicFiles.digest(p))
                   for p in sorted(package.rglob("*.py"))]
        model = package.parent / "model"
        entries.extend((str(p.relative_to(package.parent)), AtomicFiles.digest(p))
                       for p in sorted(model.rglob("*.py")))
        return AtomicFiles.identity(entries)

    @classmethod
    def prepare(
        cls, root: Path, samples: int = 50000, eval_samples: int = 2000,
        seeds: list[int] | None = None, train_path: Path | None = None,
        checkpoint_every: int = 250, eval_every: int = 2000,
        periodic_samples: int = 256,
    ) -> dict[str, Any]:
        """Create or verify one immutable suite without constructing or training a model."""

        if min(samples, eval_samples, checkpoint_every, eval_every, periodic_samples) < 1:
            raise ValueError("sample counts and intervals must be positive")
        seeds = [42] if seeds is None else seeds
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be nonempty and unique")
        root.mkdir(parents=True, exist_ok=True)
        lock = FileLock(root / "prepare.lock")
        if not lock.acquire():
            raise RuntimeError("another process is preparing this suite")
        try:
            options = {"samples": samples, "eval_samples": eval_samples, "seeds": seeds,
                       "checkpoint_every": checkpoint_every, "eval_every": eval_every,
                       "periodic_samples": periodic_samples,
                       "source": cls.source_digest()}
            if (root / "manifest.json").exists():
                old = cls.load(root)
                if old["options"] != options:
                    raise ValueError("suite already exists with different options; use a new root")
                if train_path and AtomicFiles.digest(train_path) != old["datasets"]["train"]["sha256"]:
                    raise ValueError("provided training data differs from the prepared suite")
                return old
            definitions = [("train", 1, "train", False, samples),
                           ("iid", 2, "train", False, eval_samples),
                           ("ood", 3, "test", False, eval_samples),
                           ("delayed", 4, "train", True, eval_samples)]
            datasets = {}
            for name, seed, split, delayed, count in definitions:
                path = root / "data" / (name + ".jsonl")
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(dir=path.parent)
                try:
                    with os.fdopen(fd, "w", newline="") as target:
                        if name == "train" and train_path:
                            with train_path.open(newline="") as source:
                                actual = 0
                                for line in source:
                                    target.write(line)
                                    if line.strip():
                                        actual += 1
                            if actual != count:
                                raise ValueError(f"expected {count} training records, found {actual}")
                        else:
                            generator = ToyGenerator(seed, split, delayed)
                            for i in range(count):
                                record = generator.sample(i).to_dict()
                                record["sample_id"] = f"matrix-{name}-{i}"
                                target.write(json.dumps(record) + "\n")
                        target.flush()
                        os.fsync(target.fileno())
                    os.replace(temporary, path)
                    AtomicFiles.sync_directory(path.parent)
                finally:
                    Path(temporary).unlink(missing_ok=True)
                datasets[name] = {"path": str(path.relative_to(root)), "count": count,
                                  "sha256": AtomicFiles.digest(path)}
            payload = {"version": 1, "options": options, "datasets": datasets,
                       "runs": [run.to_dict() for run in RunSpec.matrix(seeds)]}
            payload["identity"] = AtomicFiles.identity(payload)
            AtomicFiles.json(root / "manifest.json", payload)
            return payload
        finally:
            lock.release()

    @classmethod
    def load(cls, root: Path, verify_data: bool = True) -> dict[str, Any]:
        """Validate the manifest and optionally its data before launching any workers."""

        data = AtomicFiles.read(root / "manifest.json")
        payload = {k: v for k, v in data.items() if k != "identity"}
        if data["version"] != 1 or AtomicFiles.identity(payload) != data["identity"]:
            raise ValueError("matrix manifest identity is invalid")
        if data["options"]["source"] != cls.source_digest():
            raise ValueError("code changed since preparation; use a new suite root")
        if verify_data:
            for record in data["datasets"].values():
                if AtomicFiles.digest(root / record["path"]) != record["sha256"]:
                    raise ValueError("prepared dataset changed")
        return data
