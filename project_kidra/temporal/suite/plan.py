"""Copy immutable master data into a separate, independently fingerprinted extension."""

import json
from pathlib import Path
from typing import Any

from project_kidra.temporal.suite.spec import TemporalSpec
from project_kidra.timed.execution.policy import TrainingPolicy
from project_kidra.timed.matrix.plan import MatrixPlan
from project_kidra.timed.matrix.spec import RunSpec
from project_kidra.timed.matrix.storage import AtomicFiles, FileLock
from project_kidra.timed.schedule import Schedule
from project_kidra.timed.schema import TimedSample
from project_kidra.timed.tokenizer import ToyTokenizer


class TemporalPlan:
    """Never write into the original suite or include extension code in its fingerprint."""

    @staticmethod
    def source_digest() -> str:
        """Fingerprint this extension separately from the unchanged base implementation."""

        package = Path(__file__).resolve().parents[1]
        entries = [(str(p.relative_to(package)), AtomicFiles.digest(p))
                   for p in sorted(package.rglob("*.py"))]
        return AtomicFiles.identity(entries)

    @staticmethod
    def separate(base: Path, root: Path) -> None:
        """Require disjoint roots, including after resolving symbolic links."""

        if root == base or root in base.parents or base in root.parents:
            raise ValueError("temporal root must be separate from the original suite")

    @staticmethod
    def validate_records(raw: bytes) -> None:
        """Reject unsupported agents or heartbeat context overflow before launching jobs."""

        tok = ToyTokenizer()
        for line in raw.splitlines():
            if not line.strip():
                continue
            sample = TimedSample.from_dict(json.loads(line))
            schedule = Schedule(sample, tok, 5, 128)
            events = {tick for group in schedule.groups for stream in group
                      for tick in range(stream.source.offset_tick,
                                        stream.source.offset_tick + len(stream.tokens))}
            if len(events) > 128:
                raise ValueError("continuous reasoning exceeds the 128-position context")
            reasoning = [s for s in sample.streams if s.window_id == 3]
            if (len(reasoning) != 2 or not reasoning[0].text.startswith("TOOL CALL ")
                    or not reasoning[1].text.startswith("FOUND ")):
                raise ValueError("temporal suite requires the original five-window agent records")
            if any(any(s.source.predict for s in group)
                   and any(not s.source.predict for s in group) for group in schedule.groups):
                raise ValueError("diagnostics require separate observation groups")

    @classmethod
    def prepare(cls, base: Path, root: Path, seed: int | None = None,
                learning_rate: float = 0.0001) -> dict[str, Any]:
        """Reuse exact bytes, counts, intervals, seed, and order; only add two conditions."""

        base, root = base.resolve(), root.resolve()
        cls.separate(base, root)
        parent = MatrixPlan.load(base)
        seed = parent["options"]["seeds"][0] if seed is None else seed
        policy = TrainingPolicy(execution="tick")
        baseline = RunSpec("agent", policy, learning_rate, seed)
        if baseline.to_dict() not in parent["runs"]:
            raise ValueError("requested matched baseline is absent from the original matrix")
        options = {**parent["options"], "seeds": [seed], "source": cls.source_digest()}
        provenance = {"root": str(base), "identity": parent["identity"],
                      "source": parent["options"]["source"], "baseline": baseline.name}
        runs = [TemporalSpec("agent", policy, learning_rate, seed, variant).to_dict()
                for variant in ("global_time", "continuous_reasoning")]
        root.mkdir(parents=True, exist_ok=True)
        lock = FileLock(root / "prepare.lock")
        if not lock.acquire():
            raise RuntimeError("another process is preparing this temporal suite")
        try:
            if (root / "manifest.json").exists():
                previous = cls.load(root)
                if (previous["options"] != options or previous["base"] != provenance
                        or previous["runs"] != runs):
                    raise ValueError("temporal suite already exists with different options")
                return previous
            datasets = {}
            for name, record in parent["datasets"].items():
                raw = (base / record["path"]).read_bytes()
                cls.validate_records(raw)
                if sum(bool(line.strip()) for line in raw.splitlines()) != record["count"]:
                    raise ValueError("base dataset record count differs from its manifest")
                destination = root / "data" / (name + ".jsonl")
                AtomicFiles.bytes(destination, raw)
                if AtomicFiles.digest(destination) != record["sha256"]:
                    raise ValueError("base dataset changed during preparation")
                datasets[name] = {**record, "path": str(destination.relative_to(root))}
            payload: dict[str, Any] = {"version": "kidra-temporal-1", "base": provenance,
                                       "options": options, "datasets": datasets, "runs": runs}
            payload["identity"] = AtomicFiles.identity(payload)
            AtomicFiles.json(root / "manifest.json", payload)
            return payload
        finally:
            lock.release()

    @classmethod
    def load(cls, root: Path, verify_data: bool = True) -> dict[str, Any]:
        """Verify the extension and base source code without modifying either suite."""

        data = AtomicFiles.read(root / "manifest.json")
        payload = {k: v for k, v in data.items() if k != "identity"}
        if (data["version"] != "kidra-temporal-1"
                or AtomicFiles.identity(payload) != data["identity"]):
            raise ValueError("temporal manifest identity is invalid")
        cls.separate(Path(data["base"]["root"]).resolve(), root.resolve())
        if (data["options"]["source"] != cls.source_digest()
                or data["base"]["source"] != MatrixPlan.source_digest()):
            raise ValueError("code changed since preparation; use a new temporal root")
        for raw in data["runs"]:
            TemporalSpec.from_dict(raw)
        if verify_data:
            for record in data["datasets"].values():
                if AtomicFiles.digest(root / record["path"]) != record["sha256"]:
                    raise ValueError("prepared temporal dataset changed")
        return data
