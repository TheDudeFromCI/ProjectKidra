import random
import threading
from collections.abc import Generator
from pathlib import Path

from project_kidra.corpus.builder import MAGIC_NUMBER


class CorpusFile:
    """
    A dynamic loader for prebuilt token corpora.

    Loads samples as needed from the disk, keeping a portion preloaded into memory for efficiency.
    Additional samples will be loaded asynchronously as needed.
    """

    def __init__(self, path: str | Path, preload: int = 75) -> None:
        """
        Load a prebuilt token corpus from the specified file.

        Args:
            path: Path to the corpus file.
            preload: The number of samples to keep preloaded into memory (default is 75).
        """

        self.path: Path = Path(path)
        self.preload: int = preload
        self.sample_locs: list[tuple[int, int]] = []
        self.loaded_samples: list[list[int]] = []
        self._condition: threading.Condition = threading.Condition()
        self._loading: bool = False

        with open(self.path, "rb") as f:
            if f.read(len(MAGIC_NUMBER)) != MAGIC_NUMBER:
                raise ValueError("invalid corpus file")

            self.samples: int = int.from_bytes(f.read(4), "little")

            for _ in range(self.samples):
                offset = int.from_bytes(f.read(8), "little")
                length = int.from_bytes(f.read(4), "little")
                self.sample_locs.append((offset, length))

            for _ in range(self.preload):
                i = random.randint(0, self.samples - 1)
                offset, length = self.sample_locs[i]

                f.seek(offset)
                token_bytes = f.read(length * 3)
                tokens = [
                    int.from_bytes(token_bytes[i : i + 3], "little")
                    for i in range(0, len(token_bytes), 3)
                ]

                self.loaded_samples.append(tokens)

    def __iter__(self) -> Generator[list[int]]:
        while True:
            self.__queue_async_loading()
            self.__block_until_loaded()
            with self._condition:
                sample = self.loaded_samples.pop()
            yield sample

    def __block_until_loaded(self) -> None:
        """Block until at least one sample is loaded into memory."""

        with self._condition:
            self._condition.wait_for(lambda: len(self.loaded_samples) > 0)

    def __queue_async_loading(self) -> None:
        """Queue asynchronous loading of additional samples into memory."""

        with self._condition:
            if self._loading or len(self.loaded_samples) > self.preload:
                return

            self._loading = True

        threading.Thread(target=self.__load_worker, daemon=True).start()

    def __load_worker(self) -> None:
        """Load random samples from disk until the preload target is met."""

        try:
            with open(self.path, "rb") as f:
                while True:
                    with self._condition:
                        if len(self.loaded_samples) >= self.preload:
                            # Clear atomically with the check so a new worker can start immediately.
                            self._loading = False
                            return

                    offset, length = self.sample_locs[random.randint(0, self.samples - 1)]
                    f.seek(offset)
                    token_bytes = f.read(length * 3)
                    tokens = [
                        int.from_bytes(token_bytes[i : i + 3], "little")
                        for i in range(0, len(token_bytes), 3)
                    ]

                    with self._condition:
                        self.loaded_samples.append(tokens)
                        self._condition.notify_all()
        finally:
            with self._condition:
                self._loading = False
