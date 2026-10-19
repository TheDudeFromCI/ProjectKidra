"""Greedy held-out rollout with no teacher forcing of model-owned streams."""

from dataclasses import asdict, dataclass

import torch

from project_kidra.model.model import AgentModel
from project_kidra.timed.schedule import Schedule
from project_kidra.timed.schema import TimedSample
from project_kidra.timed.session import TimedSession
from project_kidra.timed.tokenizer import ToyTokenizer


@dataclass(frozen=True)
class Rollout:
    """Separate answer, call, control, and complete-trajectory accuracy."""

    answer_correct: bool
    call_correct: bool
    reasoning_correct: bool
    controls_correct: bool
    exact: bool
    predictions: dict[str, str]


class ToyEvaluator:
    """Evaluate scheduled wakeups, not an autonomous learned scheduler."""

    def __init__(self, model: AgentModel, tokenizer: ToyTokenizer) -> None:
        """Use the same tokenization and causal execution as the trainer."""

        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def rollout(self, sample: TimedSample, cross_window: bool = True) -> Rollout:
        """Generate each predicted stream; a wrong call receives no tool observation.

        Ground-truth lengths define deadlines only. Predictions stop at their own
        control token and are never replaced with reference text. The evaluator
        supplies external observations and wakeups at the recorded offsets.
        """

        model, tokenizer = self.model, self.tokenizer
        schedule = Schedule(sample, tokenizer, model.config.n_windows, model.config.n_local_ctx)
        session = TimedSession(model, tokenizer, cross_window)
        model.eval()
        actual: dict[tuple[int, int], list[int]] = {}
        call_correct = False
        try:
            for group in schedule.groups:
                if all(not s.source.predict for s in group):
                    observations = [
                        (s.source.window_id, s.source.offset_tick, s.tokens) for s in group
                        if s.source.window_id != 4 or call_correct
                    ]
                    if observations:
                        session.observe(observations)
                    continue
                active = {}
                first = min(s.source.offset_tick for s in group)
                last = max(s.source.offset_tick + len(s.tokens) for s in group)
                for tick in range(first, last):
                    for stream in group:
                        source = stream.source
                        if source.offset_tick != tick:
                            continue
                        key = (source.window_id, source.offset_tick)
                        actual[key] = []
                        old = session.windows.get(source.window_id)
                        if old is not None and old.tokens[-1] != tokenizer.noop:
                            continue
                        active[key] = (session.begin(source.window_id), stream)
                    updates = {}
                    for key, (window, stream) in active.items():
                        if tick >= stream.source.offset_tick + len(stream.tokens):
                            continue
                        if actual[key] and actual[key][-1] in {tokenizer.noop, tokenizer.end}:
                            continue
                        target = stream.tokens[tick - stream.source.offset_tick]
                        window.expect([tokenizer.noop if stream.source.predict else target])
                        updates[window] = [tick]
                    if not updates:
                        continue
                    logits, _ = session.forward(updates)
                    for window, prediction in zip(updates, logits.argmax(-1).tolist()):
                        key = next(k for k, (w, _) in active.items() if w is window)
                        source = active[key][1].source
                        token = int(prediction) if source.predict else window.pending[0]
                        session.commit(window, [token])
                        actual[key].append(token)
                    session.detach()
                for stream in group:
                    if stream.source.text.startswith("TOOL CALL"):
                        key = (stream.source.window_id, stream.source.offset_tick)
                        call_correct = actual[key] == stream.tokens
        finally:
            session.close()
        predicted = [s for g in schedule.groups for s in g if s.source.predict]
        exact = all(actual.get((s.source.window_id, s.source.offset_tick)) == s.tokens for s in predicted)
        controls = all(
            len(actual.get((s.source.window_id, s.source.offset_tick), [])) == len(s.tokens)
            and actual[(s.source.window_id, s.source.offset_tick)][-1] == s.tokens[-1]
            for s in predicted
        )
        answer = next(s for s in predicted if s.source.text.startswith("ANSWER"))
        reasoning = next(s for s in predicted if s.source.text.startswith("FOUND"))
        return Rollout(
            actual.get((answer.source.window_id, answer.source.offset_tick)) == answer.tokens,
            call_correct,
            actual.get((reasoning.source.window_id, reasoning.source.offset_tick)) == reasoning.tokens,
            controls, exact,
            {f"{wid}@{tick}": tokenizer.decode(tokens) for (wid, tick), tokens in actual.items()},
        )

    def evaluate(self, samples: list[TimedSample], cross_window: bool = True) -> dict[str, float]:
        """Aggregate exact-match rates over a fixed held-out set."""

        if not samples:
            raise ValueError("evaluation requires at least one sample")
        totals = dict.fromkeys(
            ["answer_correct", "call_correct", "reasoning_correct", "controls_correct", "exact"], 0.0
        )
        for sample in samples:
            values = asdict(self.rollout(sample, cross_window))
            for key in totals:
                totals[key] += float(values[key])
        return {key: value / len(samples) for key, value in totals.items()}
