from pathlib import Path

import numpy as np
import torch

from project_kidra.corpus.file import CorpusFile
from project_kidra.corpus.tokenizer import Tokenizer


class TokenCorpus:
    """Streams overlapping next-token training windows from token ids indefinitely."""

    def __init__(
        self,
        tokens: torch.Tensor,
        context_window: int,
        path: str | Path,
        device: torch.device | str = "cpu",
    ) -> None:
        """Wrap a 1-D tensor of token ids; `path` identifies the source for checkpoints."""

        self.path = Path(path)
        if context_window < 1:
            raise ValueError("context_window must be positive")
        if tokens.dim() != 1:
            raise ValueError("tokens must be one-dimensional")
        self.context_window = context_window
        self.device = torch.device(device)
        self.size = int(tokens.shape[0])
        if self.size < context_window + 1:
            raise ValueError("corpus must contain at least context_window + 1 tokens")
        self._tokens = tokens.to(device=self.device, dtype=torch.long)
        self._offset = 0

    @classmethod
    def from_text(
        cls,
        path: str | Path,
        context_window: int,
        tokenizer: Tokenizer,
        device: torch.device | str = "cpu",
    ) -> "TokenCorpus":
        """Tokenize a raw text file at load time (unknown words become `<|unk|>`)."""

        text = Path(path).read_text(encoding="utf-8")
        ids = torch.tensor(
            tokenizer.tokenize(text, keep_unknown=True), dtype=torch.long
        )
        return cls(ids, context_window, path, device)

    @classmethod
    def from_prebuilt(
        cls,
        path: str | Path,
        context_window: int,
        tokenizer: Tokenizer,
        device: torch.device | str = "cpu",
    ) -> "TokenCorpus":
        """Load a corpus written by `CorpusBuilder`, validating it against the tokenizer."""

        ids, _ = CorpusFile.load(path, tokenizer)
        return cls(
            torch.from_numpy(np.asarray(ids, dtype=np.int64)),
            context_window,
            path,
            device,
        )

    @classmethod
    def open(
        cls,
        path: str | Path,
        context_window: int,
        tokenizer: Tokenizer,
        device: torch.device | str = "cpu",
    ) -> "TokenCorpus":
        """Load a prebuilt `.npy` corpus, or tokenize any other file as raw text."""

        if Path(path).suffix == ".npy":
            return cls.from_prebuilt(path, context_window, tokenizer, device)
        return cls.from_text(path, context_window, tokenizer, device)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a one-sequence batch, wrapping at the end and overlapping by one token."""

        offsets = torch.arange(self.context_window + 1, device=self.device)
        values = self._tokens[(self._offset + offsets) % self.size]
        self._offset = (self._offset + self.context_window) % self.size
        return values[:-1].unsqueeze(0), values[1:].unsqueeze(0)

    def tell(self) -> int:
        """Return the token offset of the next training window."""

        return self._offset

    def seek(self, offset: int) -> None:
        """Set the token offset of the next training window."""

        if not 0 <= offset < self.size:
            raise ValueError("corpus offset is outside the corpus")
        self._offset = offset
