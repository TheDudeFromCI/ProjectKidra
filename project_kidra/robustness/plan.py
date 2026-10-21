"""Freeze a small second follow-up without changing any completed suite."""

from pathlib import Path
from typing import Any

from project_kidra.followup.suite.plan import FollowPlan
from project_kidra.followup.suite.scheduler import FollowScheduler
from project_kidra.followup.suite.spec import FollowSpec
from project_kidra.robustness.spec import RobustSpec
from project_kidra.temporal.suite.plan import TemporalPlan
from project_kidra.timed.execution.policy import TrainingPolicy
from project_kidra.timed.matrix.plan import MatrixPlan
from project_kidra.timed.matrix.storage import AtomicFiles, FileLock


class RobustPlan:
    @staticmethod
    def source_digest() -> str:
        package = Path(__file__).resolve().parent
        return AtomicFiles.identity([(str(p.relative_to(package)), AtomicFiles.digest(p))
                                     for p in sorted(package.rglob("*.py"))])

    @classmethod
    def versions(cls) -> dict[str, str]:
        return {"base": MatrixPlan.source_digest(), "temporal": TemporalPlan.source_digest(),
                "followup": FollowPlan.source_digest(), "robustness": cls.source_digest()}

    @staticmethod
    def definitions() -> tuple[list[RobustSpec], dict[str, str]]:
        """Nine training jobs (450k new updates) and three checkpoint audits."""
        runs = []
        sources = {"tick42": "@tick"}
        policy = TrainingPolicy(execution="tick")
        for variant in ("none", "global_time", "continuous_reasoning"):
            key = "seed44_" + variant
            sources[key] = f"replicate-agent-full-tick-all-lr0.0001-s44-{variant}"
            runs.append(RobustSpec("agent", policy, 0.0001, 44, "extend", variant,
                                   key, 100000, 44))
        for init_seed, order_seed in ((42, 42), (42, 44), (44, 42), (44, 44)):
            runs.append(RobustSpec("agent", policy, 0.0001, init_seed, "cross", "none",
                                   "", 50000, order_seed))
        for seed, key in ((42, "tick42"), (44, "seed44_none")):
            runs.append(RobustSpec("agent", policy, 0.0001, seed, "repair", "none",
                                   key, 100000, seed, "prefixmix"))
        for key, name, variant, seed, budget in (
            ("tick42_100k", "extend-agent-full-tick-all-lr0.0001-s42-none", "none", 42, 100000),
            ("global42_100k", "extend-agent-full-tick-all-lr0.0001-s42-global_time", "global_time", 42, 100000),
            ("continuous43", "replicate-agent-full-tick-all-lr0.0001-s43-continuous_reasoning", "continuous_reasoning", 43, 50000),
        ):
            sources[key] = name
            runs.append(RobustSpec("agent", policy, 0.0001, seed, "audit", variant,
                                   key, budget, seed))
        for spec in runs:
            RobustSpec.from_dict(spec.to_dict())
        return runs, sources

    @classmethod
    def prepare(cls, parent_root: Path, root: Path) -> dict[str, Any]:
        parent_root, root = parent_root.resolve(), root.resolve()
        TemporalPlan.separate(parent_root, root)
        parent = FollowPlan.load(parent_root)
        if parent["datasets"]["train"]["count"] != 50000:
            raise ValueError("requires the original 50000-record training set")
        root.mkdir(parents=True, exist_ok=True)
        lock = FileLock(root / "prepare.lock")
        if not lock.acquire():
            raise RuntimeError("another process is preparing this suite")
        try:
            if (root / "manifest.json").exists():
                previous = cls.load(root)
                if previous["parent"] != parent["identity"]:
                    raise ValueError("different parent suite")
                return previous
            runs, definitions = cls.definitions()
            parent_specs = {s.name: s for s in map(FollowSpec.from_dict, parent["runs"])}
            sources = {}
            for key, name in definitions.items():
                if name.startswith("@"):
                    record = parent["sources"][name[1:]]
                    source = parent_root / record["path"]
                    identity, steps = record["identity"], record["steps"]
                else:
                    spec = parent_specs[name]
                    if not FollowScheduler(parent_root, ["cpu"]).completed(spec, parent["identity"]):
                        raise ValueError(f"source run is not complete: {name}")
                    record = AtomicFiles.read(parent_root / "experiments" / name / "complete.json")
                    identity, steps = record["identity"], record["steps"]
                    source = parent_root / "experiments" / name / "checkpoint.pt"
                if not source.is_file():
                    raise ValueError(f"final checkpoint required locally: {source}")
                expected = 100000 if key.endswith("100k") else 50000
                if steps != expected:
                    raise ValueError(f"unexpected source cursor: {name}")
                before = AtomicFiles.digest(source)
                target = root / "sources" / (key + ".pt")
                AtomicFiles.bytes(target, source.read_bytes())
                if AtomicFiles.digest(target) != before or AtomicFiles.digest(source) != before:
                    raise ValueError("source checkpoint changed during preparation")
                sources[key] = {"path": str(target.relative_to(root)), "sha256": before,
                                "identity": identity, "steps": steps, "original_run": name}
            for record in parent["datasets"].values():
                AtomicFiles.bytes(root / record["path"], (parent_root / record["path"]).read_bytes())
                if AtomicFiles.digest(root / record["path"]) != record["sha256"]:
                    raise ValueError("dataset changed during preparation")
            manifest = {"version": "kidra-robustness-1", "parent": parent["identity"],
                        "versions": cls.versions(), "sources": sources,
                        "datasets": parent["datasets"], "options": parent["options"],
                        "runs": [s.to_dict() for s in runs]}
            manifest["identity"] = AtomicFiles.identity(manifest)
            AtomicFiles.json(root / "manifest.json", manifest)
            return manifest
        finally:
            lock.release()

    @classmethod
    def load(cls, root: Path, verify_data: bool = True) -> dict[str, Any]:
        data = AtomicFiles.read(root / "manifest.json")
        if (data["version"] != "kidra-robustness-1" or data["identity"] != AtomicFiles.identity(
                {k: v for k, v in data.items() if k != "identity"})):
            raise ValueError("invalid robustness manifest")
        if data["versions"] != cls.versions():
            raise ValueError("source changed; prepare a separate suite")
        specs = [RobustSpec.from_dict(raw) for raw in data["runs"]]
        if len({s.name for s in specs}) != len(specs):
            raise ValueError("duplicate run names")
        if verify_data:
            for record in (*data["datasets"].values(), *data["sources"].values()):
                if AtomicFiles.digest(root / record["path"]) != record["sha256"]:
                    raise ValueError("copied data or source checkpoint changed")
        return data
