from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusMetadata:
    """Describes how a prebuilt token corpus was produced."""

    vocab_size: int
    tokenizer_fingerprint: str
    num_tokens: int
    sources: list[str]
    samples: int

    max_windows: int = 1
    format_version: int = 2
