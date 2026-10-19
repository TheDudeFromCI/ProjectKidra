"""Independent free-running and correct-prefix diagnostics for every task."""

import torch

from project_kidra.model.model import AgentModel
from project_kidra.model.window import ContextWindow
from project_kidra.temporal.runtime.group import TemporalGroup
from project_kidra.temporal.runtime.session import TemporalSession
from project_kidra.timed.matrix.tasks import TaskProjection
from project_kidra.timed.schedule import EncodedStream, Schedule
from project_kidra.timed.schema import TimedSample
from project_kidra.timed.tokenizer import ToyTokenizer


class TemporalDiagnostics:
    """Score variable facts separately from formatting, controls, and upstream mistakes."""

    def __init__(self, model: AgentModel, tokenizer: ToyTokenizer, variant: str) -> None:
        """Share vocabulary and model weights with training, but use native tick inference."""

        self.model, self.tokenizer, self.variant = model, tokenizer, variant

    @torch.no_grad()
    def rollout(
        self, sample: TimedSample, cross_window: bool = True, oracle_prefix: bool = False
    ) -> dict[str, float]:
        """Generate without reference tokens; optionally reset to gold only between groups."""

        model, tok = self.model, self.tokenizer
        model.eval()
        schedule = Schedule(sample, tok, model.config.n_windows, model.config.n_local_ctx)
        session = TemporalSession(model, tok, self.variant, cross_window)
        correct_call = False
        scores: dict[str, float] = {}
        controls: list[bool] = []
        exact: list[bool] = []
        try:
            for group in schedule.groups:
                if all(not s.source.predict for s in group):
                    visible = [s for s in group if s.source.window_id != 4
                               or correct_call or oracle_prefix]
                    if visible:
                        TemporalGroup.forward(session, visible)
                    continue
                if any(not s.source.predict for s in group):
                    raise ValueError("matrix diagnostics require separate observation groups")
                before = {w: (list(w.tokens), w.position,
                              [(c.k, c.v, c._placeholder) for c in w.cache])
                          for w in model.windows}
                births = {wid: list(times) for wid, times in session.births.items()}
                opened = dict(session.opened)
                generated = self._generate(session, group)
                for index, stream in enumerate(group):
                    tokens = generated[index]
                    target = stream.tokens
                    match = tokens == target
                    exact.append(match)
                    controls.append(len(tokens) == len(target) and tokens[-1:] == target[-1:])
                    text = stream.source.text
                    kind = ("call" if text.startswith("TOOL CALL") else
                            "reasoning" if text.startswith("FOUND") else
                            "answer" if text.startswith("ANSWER") else "wait")
                    fact_positions = [i for i, token in enumerate(target)
                                      if tok.words[token].startswith(("k", "v"))]
                    if fact_positions:
                        scores[kind + "_fact"] = float(all(
                            i < len(tokens) and tokens[i] == target[i] for i in fact_positions
                        ))
                    scores[kind + "_exact"] = float(match)
                    if kind == "call":
                        correct_call = match
                if oracle_prefix:
                    for window in list(model.windows):
                        if window not in before:
                            del session.windows[window.id]
                            model.close_window(window)
                    for window, (tokens, position, caches) in before.items():
                        window.tokens, window.position, window.pending = tokens, position, []
                        for cache, (k, v, placeholder) in zip(window.cache, caches):
                            cache.k, cache.v, cache._placeholder = k, v, placeholder
                    session.births, session.opened = births, opened
                    TemporalGroup.forward(session, group)
            scores["controls"] = sum(controls) / len(controls)
            scores["exact"] = float(all(exact))
            return scores
        finally:
            session.close()

    def _generate(self, session: TemporalSession, group: list[EncodedStream]) -> list[list[int]]:
        """Generate one token per active window with supplied offsets and deadlines."""

        generated: list[list[int]] = [[] for _ in group]
        active: dict[int, ContextWindow] = {}
        first = min(s.source.offset_tick for s in group)
        last = max(s.source.offset_tick + len(s.tokens) for s in group)
        for tick in range(first, last):
            updates: list[tuple[int, ContextWindow]] = []
            for i, stream in enumerate(group):
                offset = tick - stream.source.offset_tick
                if offset == 0:
                    old = session.windows.get(stream.source.window_id)
                    if (old is None or session.continuous(stream.source.window_id)
                            or old.tokens[-1:] == [self.tokenizer.noop]):
                        active[i] = session.begin(stream.source.window_id, tick)
                if (i in active and 0 <= offset < len(stream.tokens)
                        and generated[i][-1:] not in ([self.tokenizer.noop], [self.tokenizer.end])):
                    updates.append((i, active[i]))
            if updates:
                logits, _ = TemporalGroup.advance(session, [w for _, w in updates], tick)
                for (i, window), token in zip(updates, logits.argmax(-1).tolist()):
                    generated[i].append(token)
                    session.record(window, token, tick)
        return generated

    def evaluate(
        self, samples: list[TimedSample], cross_window: bool = True, oracle_prefix: bool = False
    ) -> dict[str, float]:
        """Aggregate same-denominator metrics over complete held-out records."""

        if not samples:
            raise ValueError("diagnostics require samples")
        totals: dict[str, float] = {}
        for sample in samples:
            for key, value in self.rollout(sample, cross_window, oracle_prefix).items():
                totals[key] = totals.get(key, 0.0) + value
        return {key: value / len(samples) for key, value in totals.items()}

    def counterfactual(
        self, samples: list[TimedSample], task: str, cross_window: bool = True
    ) -> dict[str, float]:
        """Require both the original and changed fact to be answered correctly."""

        original = changed = paired = 0.0
        for sample in samples:
            a = self.rollout(sample, cross_window)["answer_fact"]
            b = self.rollout(TaskProjection.counterfactual(sample, task), cross_window)["answer_fact"]
            original += a
            changed += b
            paired += a * b
        return {"original": original / len(samples), "changed": changed / len(samples),
                "both_correct": paired / len(samples)}
