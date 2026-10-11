from collections.abc import Generator, Iterable
from pathlib import Path

from tqdm import tqdm

from project_kidra.corpus.metadata import CorpusMetadata
from project_kidra.corpus.tokenizer import BOS_TOKEN_STR, EOS_TOKEN_STR, Tokenizer

MAGIC_NUMBER = b"KIDRA"
"""Magic number for the corpus file header."""

MAX_SAMPLES = 1024 * 1024 * 10
"""Maximum number of samples in the corpus."""

HEADER_SIZE = (8 + 4) * MAX_SAMPLES
"""Size of the corpus file header in bytes, not yet including the magic number."""

TOKEN_SIZE = 3
"""Size of each token in bytes."""

CHUNK_SIZE = 1024 * 1024
"""Size of each read chunk in bytes."""

TRANSLATION = {
    **{i: None for i in range(32) if i not in (9, 10)},
    **{i: None for i in range(127, 0x110000)},
    9: "  ",
}
"""Translation table for cleaning up text before tokenization."""


class CorpusBuilder:
    """Tokenizes text files into a single prebuilt corpus."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        """Create a builder using an injected tokenizer."""

        self.tokenizer = tokenizer

    @staticmethod
    def find_sources(
        inputs: Iterable[str | Path], pattern: str = "*.txt", log: bool = True
    ) -> list[Path]:
        """Expand files and directories (recursively, by glob pattern) into a sorted file list."""

        files: list[Path] = []
        for item in inputs:
            path = Path(item)
            if path.is_dir():
                found = [p for p in path.rglob(pattern) if p.is_file()]
                files.extend(found)

            elif path.is_file():
                files.append(path)

            else:
                raise FileNotFoundError(f"corpus source not found: {path}")

        if log:
            num_bytes: int = sum(p.stat().st_size for p in files)
            print(f"Found {len(files)} files, containing {bytes_to_text(num_bytes)} of data.")

        return sorted(set(files))

    def build(self, sources: list[Path], output: str | Path) -> CorpusMetadata:
        """Tokenize all sources and writes the corpus."""

        if not sources:
            raise ValueError("no source files to build a corpus from")

        print(f"Building corpus from {len(sources)} sources...")

        count_pos = len(MAGIC_NUMBER)
        header_pos = count_pos + 4
        content_pos = header_pos + HEADER_SIZE
        num_tokens = 0
        samples = 0

        # Open first in write-only mode to create and initialize the corpus file.
        with open(output, "wb") as output_file:
            output_file.write(MAGIC_NUMBER)
            output_file.write((0).to_bytes(4, "little"))
            output_file.write(bytes((0).to_bytes(1, "little") * HEADER_SIZE))

        # Reopen the new corpus file in read-write mode to start writing content.
        with open(output, "r+b") as output_file:
            pbar = tqdm(sources, desc="Tokenizing sources")
            for source in pbar:
                pbar.set_postfix(file=f"...{str(source)[-15:]:}")

                for sample in extract_samples(source):
                    samples += 1

                    if samples % 10000 == 0:
                        pbar.write(f"Processed {samples} samples...")

                    tokens = self.tokenizer.encode(sample)
                    num_tokens += len(tokens)

                    byte_tokens = [bytes(t.to_bytes(3, "little")) for t in tokens]

                    output_file.seek(header_pos)
                    output_file.write(content_pos.to_bytes(8, "little"))
                    output_file.write(len(byte_tokens).to_bytes(4, "little"))
                    header_pos += 8 + 4

                    output_file.seek(content_pos)
                    output_file.write(b"".join(byte_tokens))
                    content_pos += len(byte_tokens) * TOKEN_SIZE

            output_file.seek(count_pos)
            output_file.write(samples.to_bytes(4, "little"))

        print(f"Finished building corpus with {samples:,} samples and {num_tokens:,} tokens.")

        return CorpusMetadata(
            vocab_size=self.tokenizer.vocab_size,
            tokenizer_fingerprint=self.tokenizer.fingerprint,
            num_tokens=num_tokens,
            sources=[str(s) for s in sources],
            samples=samples,
        )


def extract_samples(src: Path) -> Generator[str]:
    buffer = ""
    inside = False
    with open(src, "r", encoding="utf-8", errors="replace", buffering=CHUNK_SIZE) as f:
        while chunk := f.read(CHUNK_SIZE):
            chunk = chunk.translate(TRANSLATION)
            buffer += chunk

            while True:
                if not inside:
                    start = buffer.find(BOS_TOKEN_STR)

                    if start == -1:
                        buffer = buffer[-(len(BOS_TOKEN_STR) - 1) :]
                        break

                    buffer = buffer[start:]
                    inside = True

                end = buffer.find(EOS_TOKEN_STR, len(BOS_TOKEN_STR))

                if end == -1:
                    break

                end += len(EOS_TOKEN_STR)
                yield buffer[:end]

                buffer = buffer[end:]
                inside = False


def bytes_to_text(num_bytes: int) -> str:
    """Convert a byte count into a human-readable string with appropriate units."""

    kb = num_bytes / 1024
    mb = kb / 1024
    gb = mb / 1024

    if gb >= 1:
        return f"{gb:.1f} GiB"
    elif mb >= 1:
        return f"{mb:.1f} MiB"
    elif kb >= 1:
        return f"{kb:.1f} KiB"
    else:
        return f"{num_bytes} B"
