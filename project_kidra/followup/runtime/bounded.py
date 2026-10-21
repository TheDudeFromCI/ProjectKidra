"""Generate to a model-emitted control under a fixed, reference-independent token cap."""

from dataclasses import replace
from typing import Any

import torch

from project_kidra.model.model import AgentModel
from project_kidra.model.window import ContextWindow
from project_kidra.temporal.runtime.group import TemporalGroup
from project_kidra.temporal.runtime.session import TemporalSession
from project_kidra.timed.schedule import EncodedStream, Schedule
from project_kidra.timed.schema import TimedSample
from project_kidra.timed.tokenizer import ToyTokenizer


class BoundedProtocol:
    """Keep scripted stage readiness, but remove gold response lengths and correctness-gated tools."""

    def __init__(self, model: AgentModel, tokenizer: ToyTokenizer, variant: str,
                 max_tokens: int = 16) -> None:
        """Use a common cap for all predicted streams, including an otherwise one-token wait."""

        if max_tokens < 1:
            raise ValueError("generation cap must be positive")
        self.model, self.tokenizer, self.variant = model, tokenizer, variant
        self.max_tokens = max_tokens

    def generate(self, session: TemporalSession, group: list[EncodedStream]
                 ) -> tuple[list[list[int]], int, bool]:
        """Return generated tokens and the actual exclusive completion tick, without reading targets."""

        generated: list[list[int]] = [[] for _ in group]
        active: dict[int, ContextWindow] = {}
        first = min(s.source.offset_tick for s in group)
        deadline = max(s.source.offset_tick for s in group) + self.max_tokens
        last = first
        overflow = False
        for tick in range(first, deadline):
            updates = []
            for i, stream in enumerate(group):
                if tick == stream.source.offset_tick:
                    old = session.windows.get(stream.source.window_id)
                    if (old is None or session.continuous(stream.source.window_id)
                            or old.tokens[-1:] == [self.tokenizer.noop]):
                        active[i] = session.begin(stream.source.window_id, tick)
                if (i in active and len(generated[i]) < self.max_tokens
                        and generated[i][-1:] not in ([self.tokenizer.noop], [self.tokenizer.end])):
                    updates.append((i, active[i]))
            if not updates:
                continue
            check = [w for _, w in updates]
            if session.continuous(3) and 3 in session.windows:
                check.append(session.windows[3])
            if any(w.position >= w.n_local_ctx for w in check):
                overflow = True
                break
            logits, _ = TemporalGroup.advance(session, [w for _, w in updates], tick)
            for (index, window), token in zip(updates, logits.argmax(-1).tolist()):
                generated[index].append(token)
                session.record(window, token, tick)
            last = tick + 1
        return generated, last, overflow

    @torch.no_grad()
    def rollout(self, sample: TimedSample) -> dict[str, Any]:
        """Score actual streams; tool delivery accepts a well-formed key without a gold-key oracle."""

        tok, model = self.tokenizer, self.model
        model.eval()
        schedule = Schedule(sample, tok, model.config.n_windows, model.config.n_local_ctx)
        session = TemporalSession(model, tok, self.variant)
        previous_old = previous_new = 0
        deliver_tool = False
        scores: dict[str, float] = {}
        traces: list[dict[str, Any]] = []
        matches: list[bool] = []
        terminals: list[bool] = []
        capped = overflow = False
        try:
            for index, original in enumerate(schedule.groups):
                first = min(s.source.offset_tick for s in original)
                start = previous_new + first - previous_old
                group = [EncodedStream(replace(s.source, offset_tick=start + s.source.offset_tick - first),
                                       s.tokens) for s in original]
                previous_old = sample.group_ends[index]
                if all(not s.source.predict for s in group):
                    visible = [s for s in group if s.source.window_id != 4 or deliver_tool]
                    if visible:
                        TemporalGroup.forward(session, visible)
                        previous_new = max(s.source.offset_tick + len(s.tokens) for s in visible)
                    else:
                        previous_new = start
                    continue
                if any(not s.source.predict for s in group):
                    raise ValueError("bounded protocol requires separate observation groups")
                generated, previous_new, exhausted = self.generate(session, group)
                overflow |= exhausted
                for stream, tokens in zip(group, generated):
                    words = stream.source.text.split()
                    kind = ("call" if words[:2] == ["TOOL", "CALL"] else
                            "reasoning" if words[:1] == ["FOUND"] else
                            "answer" if words[:1] == ["ANSWER"] else "wait")
                    match = tokens == stream.tokens
                    matches.append(match)
                    terminal = tokens[-1:] in ([tok.noop], [tok.end])
                    terminals.append(terminal)
                    capped |= len(tokens) == self.max_tokens and not terminal
                    facts = [i for i, t in enumerate(stream.tokens) if tok.words[t].startswith(("k", "v"))]
                    scores[kind + "_exact"] = float(match)
                    if facts:
                        scores[kind + "_fact"] = float(all(i < len(tokens) and tokens[i] == stream.tokens[i]
                                                         for i in facts))
                    if kind == "call":
                        deliver_tool = (len(tokens) == 4 and tokens[:2] == tok.encode("TOOL CALL")
                                        and tokens[2] in tok.encode("k0 k1 k2 k3 k4 k5 k6 k7")
                                        and tokens[-1] == tok.noop)
                        scores["valid_call"] = float(deliver_tool)
                    traces.append({"kind": kind, "start_tick": stream.source.offset_tick,
                                   "generated": tok.decode(tokens), "expected": tok.decode(stream.tokens)})
                if overflow:
                    break
            scores.setdefault("answer_fact", 0.0)
            scores["exact"] = float(all(matches) and len(traces) == 4 and not overflow)
            scores["terminated_fraction"] = sum(terminals) / max(len(terminals), 1)
            scores["hit_cap"] = float(capped)
            scores["context_overflow"] = float(overflow)
            return {"scores": scores, "trace": traces}
        finally:
            session.close()
