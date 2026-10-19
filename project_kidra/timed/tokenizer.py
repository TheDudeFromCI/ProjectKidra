"""Fixed, transparent toy vocabulary; offsets count tokens rather than characters."""


class ToyTokenizer:
    """Encode whitespace-separated task symbols with explicit control tokens."""

    def __init__(self) -> None:
        """Keep token IDs stable across generation, training, and evaluation."""

        self.words = [
            "<|bos|>", "<|noop|>", "<|end|>", "SYSTEM", "LOOKUP", "USER", "ASK",
            "MAP", "TOOL", "CALL", "RESULT", "FOUND", "ANSWER", "FILL",
            *[f"k{i}" for i in range(8)],
            *[f"v{i}" for i in range(16)],
        ]
        self.ids = {word: i for i, word in enumerate(self.words)}
        self.bos = self.ids["<|bos|>"]
        self.noop = self.ids["<|noop|>"]
        self.end = self.ids["<|end|>"]

    def encode(self, text: str) -> list[int]:
        """Reject unknown symbols instead of silently changing the task."""

        return [self.ids[word] for word in text.split()]

    def decode(self, tokens: list[int]) -> str:
        """Render model output, including its control tokens."""

        return " ".join(self.words[token] for token in tokens)
