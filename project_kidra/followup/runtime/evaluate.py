"""Paired per-record measurements with bounded traces and resumable diagnostic sections."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from project_kidra.followup.runtime.bounded import BoundedProtocol
from project_kidra.followup.runtime.probes import Probes
from project_kidra.model.model import AgentModel
from project_kidra.temporal.runtime.evaluate import TemporalDiagnostics
from project_kidra.timed.matrix.storage import AtomicFiles
from project_kidra.timed.schema import TimedSample
from project_kidra.timed.tokenizer import ToyTokenizer


class FollowEvaluation:
    """Record sample-level outcomes so paired errors can be compared across interventions and seeds."""

    def __init__(self, model: AgentModel, tok: ToyTokenizer, variant: str) -> None:
        """Share the native temporal evaluator used in the completed runs."""

        self.standard = TemporalDiagnostics(model, tok, variant)
        self.bounded = BoundedProtocol(model, tok, variant)

    def run(self, directory: Path, identity: str, spec: dict[str, Any],
            data: dict[str, list[TimedSample]], stop: Callable[[], bool]) -> bool:
        """Commit one diagnostic section at a time; only finished sections are skipped on restart."""

        path = directory / "results.json"
        report: dict[str, Any] = AtomicFiles.read(path) if path.exists() else {
            "identity": identity, "spec": spec, "metrics": {}, "sections": {},
        }
        if report["identity"] != identity:
            raise ValueError("evaluation report identity mismatch")
        conditions = [(split, mode) for split in ("iid", "ood")
                      for mode in (*Probes.names, "bounded16")]
        conditions += [("delayed", "original"), ("iid", "conditional"), ("ood", "conditional"),
                       ("iid", "noexchange")]
        for split, mode in conditions:
            key = split + "/" + mode
            file = directory / "records" / (split + "-" + mode + ".jsonl")
            if key in report["sections"]:
                if not file.exists() or AtomicFiles.digest(file) != report["sections"][key]["sha256"]:
                    raise ValueError("completed per-record diagnostic file is missing or altered")
                continue
            rows = []
            totals: dict[str, float] = {}
            traces = 0
            for sample in data[split]:
                if stop():
                    return False
                if mode == "bounded16":
                    detail = self.bounded.rollout(sample)
                    scores = detail["scores"]
                    row = {"sample_id": sample.sample_id, "scores": scores}
                    if not scores["exact"] and traces < 16:
                        row["trace"] = detail["trace"]
                        traces += 1
                else:
                    transformed = Probes.apply(sample, mode) if mode in Probes.names else sample
                    scores = self.standard.rollout(transformed, cross_window=mode != "noexchange",
                                                   oracle_prefix=mode == "conditional")
                    row = {"sample_id": sample.sample_id, "scores": scores}
                rows.append(row)
                for metric, value in scores.items():
                    totals[metric] = totals.get(metric, 0.0) + value
            AtomicFiles.bytes(file, ("\n".join(json.dumps(row) for row in rows) + "\n").encode())
            metrics = {k: v / len(rows) for k, v in totals.items()}
            original_path = directory / "records" / (split + "-original.jsonl")
            if mode not in {"original", "conditional", "noexchange"}:
                baseline = [json.loads(line) for line in original_path.read_text().splitlines()]
                if [r["sample_id"] for r in baseline] != [r["sample_id"] for r in rows]:
                    raise ValueError("paired record order mismatch")
                pairs = [(a["scores"]["exact"], b["scores"]["exact"]) for a, b in zip(baseline, rows)]
                metrics["both_exact"] = sum(a * b for a, b in pairs) / len(pairs)
                metrics["lost_exact"] = sum(a * (1 - b) for a, b in pairs) / len(pairs)
                metrics["gained_exact"] = sum((1 - a) * b for a, b in pairs) / len(pairs)
            report["metrics"][key] = metrics
            report["sections"][key] = {"count": len(rows), "sha256": AtomicFiles.digest(file)}
            AtomicFiles.json(path, report)
        return True
