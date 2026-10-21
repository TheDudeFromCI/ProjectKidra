"""Freeze the original data and final checkpoints into an independent follow-up suite."""

from pathlib import Path
from typing import Any, Literal

from project_kidra.followup.suite.spec import FollowSpec
from project_kidra.temporal.suite.plan import TemporalPlan
from project_kidra.temporal.suite.spec import TemporalSpec
from project_kidra.timed.execution.policy import TrainingPolicy
from project_kidra.timed.matrix.plan import MatrixPlan
from project_kidra.timed.matrix.scheduler import MatrixScheduler
from project_kidra.timed.matrix.spec import RunSpec
from project_kidra.timed.matrix.storage import AtomicFiles, FileLock


class FollowPlan:
    """Copy only from completed runs, preserving all original files and code fingerprints."""

    @staticmethod
    def source_digest() -> str:
        """Fingerprint this package separately from both original suites."""

        package = Path(__file__).resolve().parents[1]
        return AtomicFiles.identity([(str(p.relative_to(package)), AtomicFiles.digest(p))
                                     for p in sorted(package.rglob("*.py"))])

    @classmethod
    def prepare(cls, base: Path, temporal: Path, root: Path) -> dict[str, Any]:
        """Prepare five checkpoint audits, eight seed replications, and three continuations."""

        base, temporal, root = base.resolve(), temporal.resolve(), root.resolve()
        for parent_root in (base, temporal):
            TemporalPlan.separate(parent_root, root)
        parent = MatrixPlan.load(base)
        extra = TemporalPlan.load(temporal)
        if (extra["base"]["identity"] != parent["identity"]
                or extra["datasets"] != parent["datasets"]
                or parent["options"]["samples"] != 50000):
            raise ValueError("follow-up requires matched completed 50000-record suites")
        versions = {"base": parent["options"]["source"], "temporal": extra["options"]["source"],
                    "followup": cls.source_digest()}
        provenance = {"base": parent["identity"], "temporal": extra["identity"]}
        root.mkdir(parents=True, exist_ok=True)
        lock = FileLock(root / "prepare.lock")
        if not lock.acquire():
            raise RuntimeError("another process is preparing this follow-up suite")
        try:
            if (root / "manifest.json").exists():
                previous = cls.load(root)
                if previous["provenance"] != provenance:
                    raise ValueError("existing suite has different parent experiments")
                return previous
            definitions: list[tuple[str, Path, dict[str, Any], RunSpec, str]] = [
                ("packed", base, parent, RunSpec("agent", TrainingPolicy(execution="packed"), 0.0001, 42), "none"),
                ("tick", base, parent, RunSpec("agent", TrainingPolicy(execution="tick"), 0.0001, 42), "none"),
                ("detached", base, parent, RunSpec("agent", TrainingPolicy(gradients="detached", execution="tick"), 0.0001, 42), "none"),
                ("global_time", temporal, extra, TemporalSpec("agent", TrainingPolicy(execution="tick"), 0.0001, 42, "global_time"), "global_time"),
                ("continuous_reasoning", temporal, extra, TemporalSpec("agent", TrainingPolicy(execution="tick"), 0.0001, 42, "continuous_reasoning"), "continuous_reasoning"),
            ]
            sources: dict[str, Any] = {}
            runs = []
            for key, directory, manifest, spec, variant in definitions:
                if spec.to_dict() not in manifest["runs"]:
                    raise ValueError(f"missing required original run: {spec.name}")
                if not MatrixScheduler(directory, ["cpu"]).completed(spec, manifest["identity"]):
                    raise ValueError(f"original run is not complete: {spec.name}")
                source = directory / "experiments" / spec.name / "checkpoint.pt"
                if not source.is_file():
                    raise ValueError(f"final checkpoint required locally: {source}")
                copy = root / "sources" / (key + ".pt")
                AtomicFiles.bytes(copy, source.read_bytes())
                sources[key] = {"path": str(copy.relative_to(root)), "sha256": AtomicFiles.digest(copy),
                                "identity": spec.identity(manifest["identity"]), "steps": 50000,
                                "original_run": spec.name}
                runs.append(FollowSpec("agent", spec.policy, 0.0001, 42, "audit", variant, key).to_dict())
            modes: tuple[tuple[Literal["packed", "tick"], str], ...] = (
                ("packed", "none"), ("tick", "none"),
                ("tick", "global_time"), ("tick", "continuous_reasoning"))
            for seed in (43, 44):
                for execution, variant in modes:
                    runs.append(FollowSpec("agent", TrainingPolicy(execution=execution), 0.0001, seed,
                                           "replicate", variant).to_dict())
            for key, variant in (("tick", "none"), ("global_time", "global_time"),
                                 ("continuous_reasoning", "continuous_reasoning")):
                runs.append(FollowSpec("agent", TrainingPolicy(execution="tick"), 0.0001, 42,
                                       "extend", variant, key, 100000).to_dict())
            for record in parent["datasets"].values():
                AtomicFiles.bytes(root / record["path"], (base / record["path"]).read_bytes())
                if AtomicFiles.digest(root / record["path"]) != record["sha256"]:
                    raise ValueError("original data changed while copying")
            payload: dict[str, Any] = {"version": "kidra-followup-1", "versions": versions,
                                       "provenance": provenance, "sources": sources,
                                       "options": parent["options"], "datasets": parent["datasets"],
                                       "runs": runs}
            payload["identity"] = AtomicFiles.identity(payload)
            AtomicFiles.json(root / "manifest.json", payload)
            return payload
        finally:
            lock.release()

    @classmethod
    def load(cls, root: Path, verify_data: bool = True) -> dict[str, Any]:
        """Validate copied sources and data; execution no longer writes or depends on parent roots."""

        data = AtomicFiles.read(root / "manifest.json")
        if (data["version"] != "kidra-followup-1" or data["identity"] != AtomicFiles.identity(
                {k: v for k, v in data.items() if k != "identity"})):
            raise ValueError("invalid follow-up manifest")
        if data["versions"] != {"base": MatrixPlan.source_digest(),
                                "temporal": TemporalPlan.source_digest(), "followup": cls.source_digest()}:
            raise ValueError("source code changed; prepare a new follow-up root")
        for raw in data["runs"]:
            FollowSpec.from_dict(raw)
        if verify_data:
            for record in (*data["datasets"].values(), *data["sources"].values()):
                if AtomicFiles.digest(root / record["path"]) != record["sha256"]:
                    raise ValueError("copied data or source checkpoint changed")
        return data
