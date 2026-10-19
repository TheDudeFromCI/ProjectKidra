"""Validate complete schedules before allocating windows or changing parameters."""

from dataclasses import dataclass
from itertools import pairwise

from project_kidra.timed.schema import ContentStream, TimedSample
from project_kidra.timed.tokenizer import ToyTokenizer


@dataclass(frozen=True)
class EncodedStream:
    """A validated stream and its content plus terminal token."""

    source: ContentStream
    tokens: list[int]


class Schedule:
    """Partition streams into nonoverlapping backward groups with absolute ticks."""

    def __init__(
        self, sample: TimedSample, tokenizer: ToyTokenizer, windows: int, context: int
    ) -> None:
        """Reject malformed schedules, overlap, control text, and context overflow."""

        ends = sample.group_ends
        if not ends or any(type(t) is not int or t <= 0 for t in ends):
            raise ValueError("group_ends must contain positive integer exclusive bounds")
        if any(a >= b for a, b in pairwise(ends)):
            raise ValueError("group_ends must increase strictly")
        self.groups: list[list[EncodedStream]] = [[] for _ in ends]
        previous: dict[int, tuple[int, str]] = {}
        lengths: dict[int, int] = {}
        for stream in sample.streams:
            if type(stream.window_id) is not int or not 0 <= stream.window_id < windows:
                raise ValueError("window_id outside model capacity")
            if type(stream.offset_tick) is not int or stream.offset_tick < 0:
                raise ValueError("offset_tick must be a nonnegative integer")
            if type(stream.predict) is not bool or not isinstance(stream.text, str):
                raise ValueError("invalid stream text or predict flag")
            if stream.terminal not in {"<|noop|>", "<|end|>"}:
                raise ValueError("invalid stream terminal")
        for stream in sorted(sample.streams, key=lambda s: s.offset_tick):
            content = tokenizer.encode(stream.text)
            if any(t in {tokenizer.bos, tokenizer.noop, tokenizer.end} for t in content):
                raise ValueError("control tokens must not appear inside stream text")
            tokens = content + tokenizer.encode(stream.terminal)
            start = stream.offset_tick
            end = start + len(tokens)
            group = next((i for i, bound in enumerate(ends) if start < bound), -1)
            if group < 0 or end > ends[group]:
                raise ValueError("stream must fit inside one backward group")
            if any(s.source.window_id == stream.window_id for s in self.groups[group]):
                raise ValueError("a window may have only one stream per backward group")
            if stream.window_id in previous:
                prior_end, terminal = previous[stream.window_id]
                if start < prior_end or terminal == "<|end|>":
                    raise ValueError("overlapping stream or resuming a completed window")
            previous[stream.window_id] = end, stream.terminal
            lengths[stream.window_id] = lengths.get(stream.window_id, 0) + len(content)
            if lengths[stream.window_id] + 1 > context:
                raise ValueError("timed replay requires content plus BOS to fit local context")
            self.groups[group].append(EncodedStream(stream, tokens))
        if any(not group for group in self.groups):
            raise ValueError("every backward group must contain at least one stream")
        if not any(s.predict for s in sample.streams):
            raise ValueError("sample must contain supervised predictions")
