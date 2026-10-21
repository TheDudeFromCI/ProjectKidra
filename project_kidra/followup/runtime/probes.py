"""Deterministic, paired changes to timing, layout, context length, and routing facts."""

from dataclasses import replace

from project_kidra.timed.schedule import Schedule
from project_kidra.timed.schema import ContentStream, TimedSample
from project_kidra.timed.tokenizer import ToyTokenizer


class Probes:
    """Retain record identity and target semantics except in the explicit routing intervention."""

    names = ("original", "shift4096", "gaps64", "stagger", "reverse_tables",
             "prefix24", "route_change")

    @staticmethod
    def route(sample: TimedSample) -> TimedSample:
        """Change the queried map edge and recompute both downstream facts from the tool table."""

        query = sample.streams[1].text.split()[-1]
        old = sample.streams[4].text.split()[-1]
        destination = f"k{(int(old[1:]) + 1) % 8}"
        result = sample.streams[5].text.split()[1:]
        values = dict(zip(result[::2], result[1::2]))
        answer = values[destination]
        streams = []
        for stream in sample.streams:
            words = stream.text.split()
            if words[:1] == ["MAP"]:
                for i in range(1, len(words) - 1, 2):
                    if words[i] == query:
                        words[i + 1] = destination
            if stream.predict and words[:2] == ["TOOL", "CALL"]:
                words[-1] = destination
            if stream.predict and words[:1] in (["FOUND"], ["ANSWER"]):
                words[-1] = answer
            streams.append(replace(stream, text=" ".join(words)))
        return replace(sample, streams=tuple(streams))

    @staticmethod
    def apply(sample: TimedSample, name: str) -> TimedSample:
        """Transform complete groups without moving a dependent stage before its observations."""

        if name not in Probes.names:
            raise ValueError("unknown probe")
        if name == "original":
            return sample
        if name == "route_change":
            return Probes.route(sample)
        tok = ToyTokenizer()
        schedule = Schedule(sample, tok, 5, 128)
        streams: list[ContentStream] = []
        ends: list[int] = []
        previous_old = 0
        previous_new = 0
        for index, group in enumerate(schedule.groups):
            first = min(s.source.offset_tick for s in group)
            start = previous_new + first - previous_old
            if name == "shift4096" and index == 0:
                start += 4096
            if name == "gaps64" and index > 0:
                start += 64
            current = []
            for encoded in group:
                source = encoded.source
                words = source.text.split()
                offset = start + source.offset_tick - first
                if name == "stagger" and index == 0:
                    offset += {0: 5, 1: 11, 2: 0}[source.window_id]
                if words[:1] in (["MAP"], ["RESULT"]):
                    if name == "reverse_tables":
                        content = [w for w in words[1:] if w != "FILL"]
                        pairs = list(zip(content[::2], content[1::2]))
                        words = [words[0]] + [v for pair in reversed(pairs) for v in pair] + (
                            ["FILL"] * words.count("FILL"))
                    elif name == "prefix24":
                        words = ["FILL"] * 24 + words
                current.append(replace(source, offset_tick=offset, text=" ".join(words)))
            end = max(s.offset_tick + len(tok.encode(s.text)) + 1 for s in current)
            streams.extend(current)
            ends.append(end)
            previous_old, previous_new = sample.group_ends[index], end
        result = replace(sample, streams=tuple(streams), group_ends=tuple(ends))
        Schedule(result, tok, 5, 128)
        return result
