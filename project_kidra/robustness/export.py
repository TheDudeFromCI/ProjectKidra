"""Export results only after validating every declared per-example record file."""

import json
import os
import tarfile
import tempfile
from pathlib import Path

from project_kidra.timed.matrix.storage import AtomicFiles


def export_results(root: Path, output: Path) -> int:
    """Works for old followup and robustness suites; never trains or loads weights."""
    root, output = root.resolve(), output.resolve()
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise ValueError("suite manifest missing")
    data = AtomicFiles.read(manifest)
    if data["identity"] != AtomicFiles.identity({k: v for k, v in data.items() if k != "identity"}):
        raise ValueError("manifest hash mismatch")
    reports = sorted((root / "experiments").glob("*/results.json"))
    if len(reports) != len(data["runs"]):
        raise ValueError("suite is incomplete; finish the pending jobs before export")
    expected = {AtomicFiles.identity({"manifest": data["identity"], "run": r})
                for r in data["runs"]}
    seen = set()
    for report in reports:
        info = AtomicFiles.read(report)
        marker = AtomicFiles.read(report.parent / "complete.json")
        if (marker["results_sha256"] != AtomicFiles.digest(report)
                or marker["identity"] != info["identity"]
                or info["identity"] not in expected or info["identity"] in seen):
            raise ValueError(f"invalid completed report: {report.parent.name}")
        seen.add(info["identity"])
        if not info.get("sections"):
            raise ValueError("report has no completed sections")
        for key, section in info["sections"].items():
            split, mode = key.split("/")
            path = report.parent / "records" / f"{split}-{mode}.jsonl"
            if not path.is_file():
                raise ValueError(f"missing per-example records: {path}; do not rerun training")
            if AtomicFiles.digest(path) != section["sha256"]:
                raise ValueError(f"record hash mismatch: {path}")
            with path.open() as stream:
                count = sum(1 for line in stream if line.strip())
            if count != section["count"]:
                raise ValueError(f"record count mismatch: {path}")
    files = [p for p in sorted(root.rglob("*")) if p.is_file()
             and p.suffix in {".json", ".jsonl", ".csv", ".log"}
             and p.relative_to(root).parts[0] not in {"data", "sources"}]
    if output == manifest or output in files:
        raise ValueError("output would overwrite a suite file")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    os.close(fd)
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for path in files:
                archive.add(path, arcname=str(Path(root.name) / path.relative_to(root)), recursive=False)
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return len(files)
