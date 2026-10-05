import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from project_kidra.corpus.tokenizer import Tokenizer


@dataclass(frozen=True)
class CorpusMetadata:
    """Describes how a prebuilt token corpus was produced."""

    vocab_size: int
    tokenizer_fingerprint: str
    num_tokens: int
    num_unknown: int
    sources: list[str]


class CorpusFile:
    """Reads and writes prebuilt token corpora: a `.npy` array plus a `.json` sidecar."""

    @staticmethod
    def metadata_path(path: str | Path) -> Path:
        """Return the sidecar metadata path for a corpus file."""

        return Path(path).with_suffix(".json")

    @staticmethod
    def save(path: str | Path, ids: list[int], metadata: CorpusMetadata) -> None:
        """Write token ids (as uint16/uint32) and metadata."""

        dtype = (
            np.uint16
            if metadata.vocab_size <= np.iinfo(np.uint16).max + 1
            else np.uint32
        )
        np.save(Path(path), np.asarray(ids, dtype=dtype))
        CorpusFile.metadata_path(path).write_text(
            json.dumps(asdict(metadata), indent=2)
        )

    @staticmethod
    def load(
        path: str | Path, tokenizer: Tokenizer
    ) -> tuple[npt.NDArray[np.integer[npt.NBitBase]], CorpusMetadata]:
        """Memory-map a corpus and verify it matches the given tokenizer."""

        meta_path = CorpusFile.metadata_path(path)
        if not meta_path.exists():
            raise ValueError(f"missing corpus metadata file: {meta_path}")
        metadata = CorpusMetadata(**json.loads(meta_path.read_text()))
        if metadata.tokenizer_fingerprint != tokenizer.fingerprint():
            raise ValueError("corpus was built with a different tokenizer vocabulary")
        ids = np.load(Path(path), mmap_mode="r")
        if ids.shape != (metadata.num_tokens,):
            raise ValueError("corpus size does not match its metadata")
        return ids, metadata
