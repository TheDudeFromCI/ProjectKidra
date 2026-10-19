"""Serializable content streams and explicit backward-pass boundaries."""

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ContentStream:
    """Append text to a window, then predict or observe its terminal control token."""

    window_id: int
    offset_tick: int
    text: str
    predict: bool
    terminal: Literal["<|noop|>", "<|end|>"] = "<|noop|>"


@dataclass(frozen=True)
class TimedSample:
    """One trajectory with exclusive global group ends and zero-based stream offsets."""

    sample_id: str
    streams: tuple[ContentStream, ...]
    group_ends: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Produce a JSON-compatible record without runtime state."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimedSample":
        """Parse structural fields; token-aware validation happens before execution."""

        if set(data) != {"sample_id", "streams", "group_ends"}:
            raise ValueError("expected sample_id, streams and group_ends")
        if not isinstance(data["sample_id"], str):
            raise TypeError("sample_id must be a string")
        if not isinstance(data["streams"], list) or not isinstance(data["group_ends"], list):
            raise TypeError("streams and group_ends must be lists")
        return cls(
            data["sample_id"],
            tuple(ContentStream(**stream) for stream in data["streams"]),
            tuple(data["group_ends"]),
        )
