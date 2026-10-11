"""Word-level tokenizer for ASCII text."""

import hashlib
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer

BOS_TOKEN_STR = "<|bos|>"
EOS_TOKEN_STR = "<|eos|>"
FREE_TOKEN_STR = "<|free|>"


class Tokenizer:
    """Word-level tokenizer for ASCII text."""

    def __init__(self) -> None:
        path = Path(__file__).parent / "../assets/tokenizer.json"

        with open(path, "rb") as f:
            self.fingerprint = hashlib.sha256(f.read()).hexdigest()

        self.hf_tokenizer = HFTokenizer.from_file(str(path))

        def try_get_token_id(token: str) -> int:
            token_id = self.hf_tokenizer.token_to_id(token)
            if token_id is None:
                raise ValueError(f"Tokenizer is missing required special token: {token}")
            return token_id

        self.bos_token: int = try_get_token_id(BOS_TOKEN_STR)
        self.eos_token: int = try_get_token_id(EOS_TOKEN_STR)
        self.free_token: int = try_get_token_id(FREE_TOKEN_STR)
        self.vocab_size: int = self.hf_tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs using the Hugging Face tokenizer."""
        return self.hf_tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back into a string using the Hugging Face tokenizer."""
        return self.hf_tokenizer.decode(token_ids)
