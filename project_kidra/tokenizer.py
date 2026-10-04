import hashlib
import re
from pathlib import Path

PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
TOOL_BEGIN_TOKEN = "<|tool_begin|>"
TOOL_END_TOKEN = "<|tool_end|>"
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, TOOL_BEGIN_TOKEN, TOOL_END_TOKEN)


class Tokenizer:
    """Word-level tokenizer with special control tokens appended after the vocabulary."""

    def __init__(self) -> None:
        """Initialize the tokenizer and load the vocabulary from the tokens file."""

        self._tokens: list[str] = []
        self._index: dict[str, int] = {}

        tokens_path = Path(__file__).parent / "assets/tokens.txt"
        if tokens_path.exists():
            with open(tokens_path, "r") as f:
                self.tokens = f.read().splitlines()
        else:
            print("Error: Tokens file not found.")

    @property
    def tokens(self) -> list[str]:
        """The regular (non-special) vocabulary, indexed by token ID."""

        return self._tokens

    @tokens.setter
    def tokens(self, tokens: list[str]) -> None:
        """Replace the vocabulary and rebuild the lookup index (first occurrence wins)."""

        self._tokens = list(tokens)
        self._index = {}
        for i, token in enumerate(self._tokens):
            self._index.setdefault(token, i)

    def special_id(self, token: str) -> int:
        """Return the ID of a special token; special IDs follow the regular vocabulary."""

        return len(self._tokens) + SPECIAL_TOKENS.index(token)

    def tokenize(self, text: str, keep_unknown: bool = False) -> list[int]:
        """Tokenize the input text into a list of token IDs.

        Args:
            text (str): The input text to tokenize.
            keep_unknown (bool): Emit the `<|unk|>` ID for out-of-vocabulary
                tokens instead of dropping them.

        Returns:
            list[int]: A list of token IDs corresponding to the input text.

        Note:
            Special tokens such as `<|tool_begin|>` are recognised verbatim.
        """

        specials = "|".join(re.escape(token) for token in SPECIAL_TOKENS)
        pieces = re.findall(rf"{specials}|\d|\w+|[^\w\s]", text.lower())
        token_ids: list[int] = []
        for piece in pieces:
            if piece in SPECIAL_TOKENS:
                token_ids.append(self.special_id(piece))
            elif piece in self._index:
                token_ids.append(self._index[piece])
            elif keep_unknown:
                token_ids.append(self.special_id(UNK_TOKEN))
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back to space-separated text, rendering special tokens verbatim."""

        words: list[str] = []
        for token_id in token_ids:
            if 0 <= token_id < len(self._tokens):
                words.append(self._tokens[token_id])
            elif len(self._tokens) <= token_id < self.vocab_size():
                words.append(SPECIAL_TOKENS[token_id - len(self._tokens)])
            else:
                raise ValueError(f"token id {token_id} is outside the vocabulary")
        return " ".join(words)

    def fingerprint(self) -> str:
        """Return a hash of the vocabulary so prebuilt corpora can detect a changed tokenizer."""

        digest = hashlib.sha256("\n".join(self._tokens + list(SPECIAL_TOKENS)).encode("utf-8"))
        return digest.hexdigest()

    def vocab_size(self) -> int:
        """Return the total vocabulary size, including special tokens."""

        return len(self._tokens) + len(SPECIAL_TOKENS)
