from collections.abc import Iterable
from pathlib import Path

from project_kidra.corpus.file import CorpusFile, CorpusMetadata
from project_kidra.corpus.tokenizer import Tokenizer, UNK_TOKEN


class CorpusBuilder:
    """Tokenizes text files into a single prebuilt corpus."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        """Create a builder using an injected tokenizer."""

        self.tokenizer = tokenizer

    @staticmethod
    def find_sources(
        inputs: Iterable[str | Path], pattern: str = "*.txt"
    ) -> list[Path]:
        """Expand files and directories (recursively, by glob pattern) into a sorted file list."""

        files: list[Path] = []
        for item in inputs:
            path = Path(item)
            if path.is_dir():
                files.extend(p for p in path.rglob(pattern) if p.is_file())
            elif path.is_file():
                files.append(path)
            else:
                raise FileNotFoundError(f"corpus source not found: {path}")
        return sorted(set(files))

    def build(self, sources: list[Path], output: str | Path) -> CorpusMetadata:
        """Tokenize all sources (unknown words become `<|unk|>`) and write the corpus."""

        if not sources:
            raise ValueError("no source files to build a corpus from")
        ids: list[int] = []
        for source in sources:
            ids.extend(
                self.tokenizer.tokenize(
                    source.read_text(encoding="utf-8"), keep_unknown=True
                )
            )
        unk = self.tokenizer.token_id(UNK_TOKEN)
        metadata = CorpusMetadata(
            vocab_size=self.tokenizer.vocab_size(),
            tokenizer_fingerprint=self.tokenizer.fingerprint(),
            num_tokens=len(ids),
            num_unknown=sum(1 for i in ids if i == unk),
            sources=[str(s) for s in sources],
        )
        CorpusFile.save(output, ids, metadata)
        return metadata
