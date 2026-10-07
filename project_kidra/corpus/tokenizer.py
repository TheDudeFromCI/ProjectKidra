import hashlib
import re
from pathlib import Path

PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
TOOL_BEGIN_TOKEN = "<|tool_begin|>"
TOOL_END_TOKEN = "<|tool_end|>"
NEW_LINE = "\n"
NAME_BEGIN = "<|name|>"
SPECIAL_TOKENS: tuple[str, ...] = (
    PAD_TOKEN,
    UNK_TOKEN,
    TOOL_BEGIN_TOKEN,
    TOOL_END_TOKEN,
    NEW_LINE,
    NAME_BEGIN,
)
LETTER_TOKENS: tuple[str, ...] = tuple(chr(i) for i in range(ord("a"), ord("z") + 1))


class Tokenizer:
    """Word-level tokenizer with special control tokens appended after the vocabulary."""

    @staticmethod
    def default_vocab() -> Tokenizer:
        vocab_path = Path(__file__).parent / "../assets/vocab.txt"
        return Tokenizer(vocab_path)

    def __init__(self, vocab: list[str] | Path) -> None:
        """
        Create a new tokenizer and generate the necessary token indices.

        Tokens are converted to lowercase automatically.

        Args:
            vocab (list[str] | Path): A list of vocabulary tokens or a path to a vocabulary file.
        """

        self._tokens: list[str] = []
        self._index: dict[str, int] = {}

        if isinstance(vocab, list):
            self._tokens = [token.lower() for token in vocab]
        elif isinstance(vocab, Path):
            if vocab.exists():
                with open(vocab, "r") as f:
                    self._tokens = [token.lower() for token in f.read().splitlines()]
            elif not self._tokens:
                raise ValueError("Vocabulary file does not exist.")
        else:
            raise ValueError("vocab must be a list or a Path")

        for i, token in enumerate(self._tokens):
            self._index[token] = i
        for i, token in enumerate(SPECIAL_TOKENS):
            self._index[token] = len(self._tokens) + i
        for i, token in enumerate(LETTER_TOKENS):
            self._index[token] = len(self._tokens) + len(SPECIAL_TOKENS) + i

        self._no_space: set[int] = set(
            [
                *[
                    self._index[token]
                    for token in SPECIAL_TOKENS
                    if token != NAME_BEGIN
                ],
                *[self._index[letter] for letter in LETTER_TOKENS],
                *[
                    self._index[token]
                    for token in self._tokens
                    if not re.match(r"^[a-z]+$", token)
                ],
            ]
        )

        self._no_space_after: set[int] = set(
            [self._index[token] for token in SPECIAL_TOKENS],
        )

    def token_id(self, token: str) -> int | None:
        """
        Return the ID of a token. Token is case insensitive.

        Args:
            token (str): The token for which to retrieve the ID.

        Returns:
            int | None: The ID of the token, or None if the token is not found.
        """

        token = token.lower()

        if token in self._index:
            return self._index[token]

        return None

    def has_space(self, token_id: int) -> bool:
        """Return True if the token ID corresponds to a token that should be preceded by a space."""

        return token_id not in self._no_space

    def has_space_after(self, token_id: int) -> bool:
        """Return True if the token ID corresponds to a token that should be followed by a space."""

        return token_id not in self._no_space_after

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
        unknown_tokens: set[str] = set()

        for piece in pieces:
            token_id = self.token_id(piece)

            if token_id is not None:
                token_ids.append(token_id)
                continue

            if re.match(r"^[a-z]+$", piece):
                token_ids.append(self._index[NAME_BEGIN])
                for char in piece:
                    token_ids.append(self._index[char])
            elif keep_unknown:
                token_ids.append(self._index[UNK_TOKEN])
                unknown_tokens.add(piece)

        for token in unknown_tokens:
            print(f"Unknown token encountered during tokenization: {token}")

        return token_ids

    def get_token(self, token_id: int) -> str:
        """Return the string representation of a token ID."""

        # vocab tokens
        if 0 <= token_id < len(self._tokens):
            return self._tokens[token_id]

        # special tokens
        elif len(self._tokens) <= token_id < len(self._tokens) + len(SPECIAL_TOKENS):
            return SPECIAL_TOKENS[token_id - len(self._tokens)]

        # letter tokens
        elif len(self._tokens) + len(SPECIAL_TOKENS) <= token_id < self.vocab_size():
            return LETTER_TOKENS[token_id - len(self._tokens) - len(SPECIAL_TOKENS)]

        # invalid token ID
        else:
            raise ValueError(f"token id {token_id} is outside the vocabulary")

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back to space-separated text, rendering special tokens verbatim."""

        output = ""
        previous_token_id: int | None = None

        for token_id in token_ids:
            token = self.get_token(token_id)
            if (
                previous_token_id is not None
                and self.has_space(token_id)
                and self.has_space_after(previous_token_id)
            ):
                output += " "
            output += token
            previous_token_id = token_id

        return output

    def fingerprint(self) -> str:
        """Return a hash of the vocabulary so prebuilt corpora can detect a changed tokenizer."""

        digest = hashlib.sha256(
            "\n".join(self._tokens + list(SPECIAL_TOKENS)).encode("utf-8")
        )
        return digest.hexdigest()

    def vocab_size(self) -> int:
        """Return the total vocabulary size, including special tokens."""

        return len(self._tokens) + len(SPECIAL_TOKENS) + len(LETTER_TOKENS)
