import re
from pathlib import Path


class Tokenizer:
    """A simple tokenizer class for processing text."""

    tokens: list[str]


    def __init__(self) -> None:
        """Initialize the tokenizer and load the vocabulary from the tokens file."""

        self.tokens = []

        path = Path(__file__).parent
        tokens_path = path / "assets/tokens.txt"
        if tokens_path.exists():
            with open(tokens_path, "r") as f:
                self.tokens = f.read().splitlines()
        else:
            print("Error: Tokens file not found.")


    def tokenize(self, text: str) -> list[int]:
        """Tokenize the input text into a list of token IDs.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list[int]: A list of token IDs corresponding to the input text.

        Note:
            Tokens not found in the tokenizer's vocabulary will be ignored.
        """

        text = text.lower()
        tokens = re.findall(r"\d|\w+|[^\w\s]", text)
        token_ids = [self.tokens.index(token) for token in tokens if token in self.tokens]

        return token_ids

    
    def vocab_size(self) -> int:
        """Return the size of the tokenizer's vocabulary."""
        return len(self.tokens)