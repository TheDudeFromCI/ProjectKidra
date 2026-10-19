"""Flatten scientific results and progress into inspectable JSON and CSV."""

import csv
import io
from pathlib import Path
from typing import Any

from project_kidra.timed.matrix.plan import MatrixPlan
from project_kidra.timed.matrix.spec import RunSpec
from project_kidra.timed.matrix.storage import AtomicFiles, FileLock


class MatrixSummary:
    """Aggregate completed reports without loading models or reopening checkpoints."""

    @staticmethod
    def write(root: Path) -> dict[str, Any]:
        """Include incomplete runs so crashes and failures remain visible in comparisons."""

        manifest = MatrixPlan.load(root, verify_data=False)
        rows: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for raw in manifest["runs"]:
            spec = RunSpec.from_dict(raw)
            directory = root / "experiments" / spec.name
            status = (AtomicFiles.read(directory / "status.json")
                      if (directory / "status.json").exists() else {"status": "pending", "steps": 0})
            if status["status"] in {"starting", "training", "evaluating"}:
                probe = FileLock(directory / "worker.lock")
                if probe.acquire():
                    probe.release()
                    status["status"] = "interrupted"
            row = {"name": spec.name, "task": spec.task, "seed": spec.seed,
                   "learning_rate": spec.learning_rate, **raw["policy"],
                   "status": status["status"], "steps": status["steps"]}
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if (directory / "results.json").exists():
                report = AtomicFiles.read(directory / "results.json")
                for condition, metrics in report["metrics"].items():
                    for metric, value in metrics.items():
                        row[condition + "/" + metric] = value
            rows.append(row)
        AtomicFiles.json(root / "summary.json", {"counts": counts, "runs": rows})
        output = io.StringIO()
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        AtomicFiles.bytes(root / "summary.csv", output.getvalue().encode())
        return counts
