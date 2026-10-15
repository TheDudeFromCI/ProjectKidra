"""Tokenization with either Kidra's vocabulary or an unmodified local Qwen vocabulary."""

import hashlib
import json
from pathlib import Path
from typing import cast

from tokenizers import Tokenizer as HFTokenizer

BOS_TOKEN_STR = "<|bos|>"
EOS_TOKEN_STR = "<|eos|>"
FREE_TOKEN_STR = "<|free|>"


class Tokenizer:
    """Keep checkpoint token IDs intact, including the padded Qwen vocabulary size."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        """Read local tokenizer.json and, for Qwen, config.json; never download files."""

        directory = Path(model_path) if model_path is not None else None
        path = (
            directory / "tokenizer.json"
            if directory
            else Path(__file__).parent.parent / "assets/tokenizer.json"
        )
        self.hf_tokenizer = HFTokenizer.from_file(str(path))
        if directory:
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            if config.get("model_type") != "qwen3":
                raise ValueError("expected a Qwen3 model directory")
            self.bos_token = int(config["bos_token_id"])
            self.eos_token = int(config["eos_token_id"])
            self.free_token = -1  # Runtime sentinel, never embedded or used as a target.
            self.vocab_size = int(config["vocab_size"])
            if max(self.hf_tokenizer.get_vocab().values()) >= self.vocab_size:
                raise ValueError("tokenizer IDs exceed the model vocabulary")
        else:

            def required(token: str) -> int:
                """Resolve a required special token without altering vocabulary IDs."""

                value = self.hf_tokenizer.token_to_id(token)
                if value is None:
                    raise ValueError(f"Tokenizer is missing required special token: {token}")
                return cast(int, value)

            self.bos_token = required(BOS_TOKEN_STR)
            self.eos_token = required(EOS_TOKEN_STR)
            self.free_token = required(FREE_TOKEN_STR)
            self.vocab_size = self.hf_tokenizer.get_vocab_size()
        if not all(0 <= token < self.vocab_size for token in (self.bos_token, self.eos_token)):
            raise ValueError("boundary token IDs exceed the model vocabulary")
        digest = hashlib.sha256(path.read_bytes())
        digest.update(f":{self.vocab_size}:{self.bos_token}:{self.eos_token}".encode())
        self.fingerprint = digest.hexdigest()

    def encode(self, text: str) -> list[int]:
        """Encode text exactly, without implicit boundary markers or ASCII filtering."""

        return cast(list[int], self.hf_tokenizer.encode(text, add_special_tokens=False).ids)

    def encode_window(self, text: str) -> list[int]:
        """Append EOS to raw window text; the model supplies BOS as the first input."""

        return self.encode(text) + [self.eos_token]

    def decode(self, token_ids: list[int]) -> str:
        """Decode model token IDs using the original tokenizer."""

        return cast(str, self.hf_tokenizer.decode(token_ids))
