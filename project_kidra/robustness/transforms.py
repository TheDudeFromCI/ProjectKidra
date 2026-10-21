"""Label-preserving layout probes; replay-stable training augmentation."""

import hashlib
import random
from dataclasses import replace

from project_kidra.followup.runtime.probes import Probes
from project_kidra.timed.schedule import Schedule
from project_kidra.timed.schema import TimedSample
from project_kidra.timed.tokenizer import ToyTokenizer


def layout(sample: TimedSample, count: int, placement: str = "prefix") -> TimedSample:
    """Move only padding, preserving original inter-group gaps and every target."""
    if count < 0 or placement not in {"prefix", "suffix", "aftermarker"}:
        raise ValueError("invalid layout transform")
    tok = ToyTokenizer()
    schedule = Schedule(sample, tok, 5, 128)
    streams, ends = [], []
    old_end = new_end = 0
    for index, group in enumerate(schedule.groups):
        start = min(s.source.offset_tick for s in group)
        first = new_end + start - old_end
        current = []
        for encoded in group:
            source = encoded.source
            words = source.text.split()
            if words[:1] in (["MAP"], ["RESULT"]):
                padding = ["FILL"] * count
                if placement == "prefix":
                    words = padding + words
                elif placement == "suffix":
                    words += padding
                else:
                    words = words[:1] + padding + words[1:]
            current.append(replace(source, text=" ".join(words),
                                   offset_tick=first + source.offset_tick - start))
        new_end = max(s.offset_tick + len(tok.encode(s.text)) + 1 for s in current)
        old_end = sample.group_ends[index]
        streams.extend(current)
        ends.append(new_end)
    result = replace(sample, streams=tuple(streams), group_ends=tuple(ends))
    Schedule(result, tok, 5, 128)
    return result


def augment(sample: TimedSample, mode: str, step: int) -> TimedSample:
    """Use no global RNG; restarting the same sample cursor gives identical padding."""
    if mode == "none":
        return sample
    if mode != "prefixmix":
        raise ValueError("unknown augmentation")
    digest = hashlib.sha256(f"prefixmix-v1:{step}:{sample.sample_id}".encode()).digest()
    count = random.Random(int.from_bytes(digest, "big")).choice((0, 8, 16, 24))
    return layout(sample, count)


class RobustProbes:
    names = (*Probes.names, "suffix24", "aftermarker24", "prefix40",
             "prefix24_conditional", "stagger_conditional", "rebase_shift4096")

    @staticmethod
    def apply(sample: TimedSample, name: str) -> TimedSample:
        if name in Probes.names:
            return Probes.apply(sample, name)
        if name in {"suffix24", "aftermarker24", "prefix40"}:
            count, placement = {"suffix24": (24, "suffix"),
                                "aftermarker24": (24, "aftermarker"),
                                "prefix40": (40, "prefix")}[name]
            return layout(sample, count, placement)
        if name in {"prefix24_conditional", "stagger_conditional"}:
            return Probes.apply(sample, name.removesuffix("_conditional"))
        if name == "rebase_shift4096":
            shifted = Probes.apply(sample, "shift4096")
            origin = min(s.offset_tick for s in shifted.streams)
            return replace(shifted,
                           streams=tuple(replace(s, offset_tick=s.offset_tick-origin)
                                         for s in shifted.streams),
                           group_ends=tuple(e-origin for e in shifted.group_ends))
        raise ValueError("unknown robustness probe")
