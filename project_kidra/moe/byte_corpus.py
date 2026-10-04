from pathlib import Path

import torch


class ByteCorpus:
    """Streams overlapping next-byte training windows from a file indefinitely."""

    def __init__(
        self,
        path: str | Path,
        context_window: int,
        device: torch.device | str = "cpu",
    ) -> None:
        """Open a corpus file and validate it can supply one training window."""

        self.path = Path(path)
        if context_window < 1:
            raise ValueError("context_window must be positive")
        self.context_window = context_window
        self.device = torch.device(device)
        self.size_bytes = self.path.stat().st_size
        if self.size_bytes < context_window + 1:
            raise ValueError("corpus must contain at least context_window + 1 bytes")
        self._file = self.path.open("rb")

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a one-sequence batch, wrapping at EOF and preserving byte overlap."""

        tokens = bytearray()
        required = self.context_window + 1
        while len(tokens) < required:
            chunk = self._file.read(required - len(tokens))
            if chunk:
                tokens.extend(chunk)
            else:
                self._file.seek(0)
        self._file.seek(-1, 1)
        values = torch.tensor(list(tokens), dtype=torch.long, device=self.device)
        return values[:-1].unsqueeze(0), values[1:].unsqueeze(0)

    def tell(self) -> int:
        """Return the byte offset of the next training window."""

        return self._file.tell()

    def seek(self, offset: int) -> None:
        """Set the byte offset of the next training window."""

        if not 0 <= offset < self.size_bytes:
            raise ValueError("corpus offset is outside the file")
        self._file.seek(offset)

    def close(self) -> None:
        """Close the underlying corpus file."""

        self._file.close()

    def __enter__(self) -> "ByteCorpus":
        """Return this stream for use as a context manager."""

        return self

    def __exit__(self, *_: object) -> None:
        """Close the corpus file when leaving a context manager."""

        self.close()
