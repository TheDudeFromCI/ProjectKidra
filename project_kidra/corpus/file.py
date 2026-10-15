"""Lazy random access to grouped, tokenized corpus samples."""

import json
import random
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import cast

from project_kidra.corpus.builder import FORMAT_VERSION
from project_kidra.corpus.metadata import CorpusMetadata
from project_kidra.corpus.tokenizer import Tokenizer


class CorpusFile:
    """Load bounded batches of random samples without splitting their windows."""

    def __init__(self, path: str | Path, preload: int = 75) -> None:
        """Validate the version and persisted metadata before sampling."""

        if preload < 1:
            raise ValueError("preload must be positive")
        self.path = Path(path).resolve()
        self.preload = preload
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        try:
            with closing(self._connect()) as connection:
                if connection.execute("PRAGMA user_version").fetchone()[0] != FORMAT_VERSION:
                    raise ValueError("unsupported corpus version; rebuild from JSON samples")
                row = connection.execute("SELECT value FROM metadata").fetchone()
                if row is None:
                    raise ValueError("missing corpus metadata")
                self.metadata = CorpusMetadata(**json.loads(row[0]))
                count = connection.execute("SELECT count(*) FROM samples").fetchone()[0]
                if count != self.metadata.samples or count < 1:
                    raise ValueError("invalid or empty corpus")
        except sqlite3.DatabaseError as error:
            raise ValueError("invalid or legacy corpus; rebuild from JSON samples") from error
        self.samples = self.metadata.samples

    def _connect(self) -> sqlite3.Connection:
        """Open read-only so a typo never creates an empty corpus."""

        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)

    def validate_tokenizer(self, tokenizer: Tokenizer) -> None:
        """Reject a corpus built with different token IDs or boundary tokens."""

        if (
            self.metadata.tokenizer_fingerprint != tokenizer.fingerprint
            or self.metadata.vocab_size != tokenizer.vocab_size
        ):
            raise ValueError("corpus tokenizer mismatch; rebuild with the selected model tokenizer")

    def __iter__(self) -> Iterator[list[list[int]]]:
        """Sample indefinitely, with at most preload complete samples in memory."""

        connection = self._connect()
        try:
            while True:
                batch: list[list[list[int]]] = []
                for _ in range(self.preload):
                    index = random.randrange(1, self.samples + 1)
                    row = connection.execute(
                        "SELECT windows FROM samples WHERE id = ?", (index,)
                    ).fetchone()
                    if row is None:
                        raise ValueError(f"missing corpus sample {index}")
                    batch.append(cast(list[list[int]], json.loads(row[0])))
                yield from batch
        finally:
            connection.close()
