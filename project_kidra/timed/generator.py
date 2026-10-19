"""Deterministic multi-window lookup episodes with delayed tool observations."""

import random

from project_kidra.timed.schema import ContentStream, TimedSample


class ToyGenerator:
    """Generate lookup chains whose random facts cannot be memorized from the prompt."""

    def __init__(self, seed: int, split: str = "train", delayed: bool = False) -> None:
        """Partition query/value combinations and vary timing independently of answers."""

        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        self.random = random.Random(seed)
        self.seed = seed
        self.split = split
        self.delayed = delayed

    def sample(self, index: int) -> TimedSample:
        """Create input, call, observation, resumed reasoning, and answer timing groups."""

        rng = self.random
        query = rng.randrange(8)
        values = [v for v in range(16) if ((query + v) % 4 == 0) == (self.split == "test")]
        answer = rng.choice(values)
        destination = rng.choice([k for k in range(8) if k != query])
        keys = rng.sample(range(8), 8)
        mapping = {key: rng.randrange(8) for key in keys}
        mapping[query] = destination
        result = {key: rng.randrange(16) for key in keys}
        result[destination] = answer
        pad = " FILL" * rng.randint(0, 6 if self.delayed else 2)
        document = "MAP " + " ".join(f"k{k} k{mapping[k]}" for k in keys) + pad
        response = "RESULT " + " ".join(f"k{k} v{result[k]}" for k in rng.sample(keys, 8))
        streams: list[ContentStream] = []
        ends: list[int] = []
        tick = 0

        def group(items: list[tuple[int, str, bool, bool]]) -> None:
            """Pack concurrent streams and move beyond all terminal tokens."""

            nonlocal tick
            for window, text, predict, final in items:
                streams.append(ContentStream(
                    window, tick, text, predict, "<|end|>" if final else "<|noop|>"
                ))
            tick += max(len(text.split()) + 1 for _, text, _, _ in items)
            ends.append(tick)
            if self.delayed:
                tick += rng.randint(1, 9)

        group([(0, "SYSTEM LOOKUP", False, False), (1, f"USER ASK k{query}", False, False),
               (2, document, False, False)])
        group([(1, "", True, False), (3, f"TOOL CALL k{destination}", True, False)])
        group([(4, response, False, False)])
        group([(3, f"FOUND v{answer}", True, True)])
        group([(1, f"ANSWER v{answer}", True, True)])
        return TimedSample(f"{self.split}-{self.seed}-{index}", tuple(streams), tuple(ends))
