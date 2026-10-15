"""Build versioned, indexed multi-window corpora from JSON or streaming JSONL."""

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from project_kidra.corpus.metadata import CorpusMetadata
from project_kidra.corpus.tokenizer import Tokenizer

FORMAT_VERSION = 2


def extract_samples(source: Path) -> Iterator[list[str]]:
    """Read {windows: [text, ...]} objects, preserving text and window order.

    JSON accepts one object or an array of objects. JSONL reads one object per
    nonblank line, keeping memory bounded by the largest sample.
    """

    def validate(value: Any, location: str) -> list[str]:
        """Reject malformed samples with a useful source location."""

        if not isinstance(value, dict) or set(value) != {"windows"}:
            raise ValueError(f"{location}: expected an object containing only 'windows'")
        windows = value["windows"]
        if not isinstance(windows, list) or not windows:
            raise ValueError(f"{location}: windows must be a nonempty list of strings")
        if any(not isinstance(text, str) or not text for text in windows):
            raise ValueError(f"{location}: each window must contain nonempty text")
        return windows

    with source.open(encoding="utf-8") as stream:
        if source.suffix.lower() == ".jsonl":
            for line, text in enumerate(stream, 1):
                if text.strip():
                    yield validate(json.loads(text), f"{source}:{line}")
        elif source.suffix.lower() == ".json":
            data = json.load(stream)
            for index, sample in enumerate(data if isinstance(data, list) else [data]):
                yield validate(sample, f"{source}:sample {index + 1}")
        else:
            raise ValueError(f"{source}: expected .json or .jsonl; convert legacy text samples")


class CorpusBuilder:
    """Tokenize grouped samples into an atomically replaced SQLite corpus."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        """Use the same injected tokenizer as the intended model."""

        self.tokenizer = tokenizer

    @staticmethod
    def find_sources(
        inputs: Iterable[str | Path], pattern: str | None = None, log: bool = True
    ) -> list[Path]:
        """Discover JSON and JSONL recursively, or use an explicit glob."""

        files: set[Path] = set()
        for item in inputs:
            path = Path(item)
            if path.is_dir():
                for glob in [pattern] if pattern else ["*.json", "*.jsonl"]:
                    files.update(p for p in path.rglob(glob) if p.is_file())
            elif path.is_file():
                files.add(path)
            else:
                raise FileNotFoundError(f"corpus source not found: {path}")
        if log:
            print(f"Found {len(files)} source files.")
        return sorted(files)

    def build(self, sources: list[Path], output: str | Path) -> CorpusMetadata:
        """Write each sample as one record; reject empty corpora and partial builds."""

        if not sources:
            raise ValueError("no source files to build a corpus from")
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() in {source.resolve() for source in sources}:
            raise ValueError("output must not overwrite a source file")
        descriptor, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        os.close(descriptor)
        connection = sqlite3.connect(temporary)
        try:
            connection.execute("CREATE TABLE metadata (value TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE samples (id INTEGER PRIMARY KEY, windows TEXT NOT NULL)"
            )
            count = total = max_windows = 0
            for source in sources:
                for texts in extract_samples(source):
                    windows = [self.tokenizer.encode_window(text) for text in texts]
                    count += 1
                    total += sum(map(len, windows))
                    max_windows = max(max_windows, len(windows))
                    connection.execute(
                        "INSERT INTO samples VALUES (?, ?)",
                        (count, json.dumps(windows, separators=(",", ":"))),
                    )
            if not count:
                raise ValueError("no samples found in corpus sources")
            metadata = CorpusMetadata(
                vocab_size=self.tokenizer.vocab_size,
                tokenizer_fingerprint=self.tokenizer.fingerprint,
                num_tokens=total,
                sources=[str(s) for s in sources],
                samples=count,
                max_windows=max_windows,
                format_version=FORMAT_VERSION,
            )
            connection.execute("INSERT INTO metadata VALUES (?)", (json.dumps(asdict(metadata)),))
            connection.execute(f"PRAGMA user_version = {FORMAT_VERSION}")
            connection.commit()
            connection.close()
            os.replace(temporary, target)
            return metadata
        finally:
            connection.close()
            Path(temporary).unlink(missing_ok=True)
