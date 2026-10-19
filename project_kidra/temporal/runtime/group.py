"""Teacher-forced group execution with the same event clock used by inference."""

import torch
from torch import Tensor

from project_kidra.model.window import ContextWindow
from project_kidra.temporal.runtime.session import TemporalSession
from project_kidra.temporal.runtime.tick import TemporalTick
from project_kidra.timed.schedule import EncodedStream


class TemporalGroup:
    """Add heartbeat computations without adding labels or changing original event times."""

    @staticmethod
    def advance(session: TemporalSession, windows: list[ContextWindow], tick: int
                ) -> tuple[Tensor, Tensor]:
        """Compute simultaneous updates and commit an unlabelled reasoning heartbeat if needed."""

        active = list(windows)
        heartbeat = None
        if active and session.continuous(3) and all(w.id != 3 for w in active):
            heartbeat = session.begin(3, tick)
            active.append(heartbeat)
        if any(w.position >= w.n_local_ctx for w in active):
            raise ValueError("temporal trajectory exceeds local context; use a larger-context suite")
        logits, aux = TemporalTick.forward(session, active)
        if heartbeat is not None:
            session.record(heartbeat, session.tokenizer.ids["FILL"], tick)
        return logits[:len(windows)], aux

    @staticmethod
    def forward(session: TemporalSession, group: list[EncodedStream]) -> tuple[Tensor, Tensor]:
        """Return original targets' logits in stream order; heartbeat targets have zero loss."""

        rows: list[list[Tensor]] = [[] for _ in group]
        auxiliaries: list[Tensor] = []
        first = min(s.source.offset_tick for s in group)
        last = max(s.source.offset_tick + len(s.tokens) for s in group)
        for tick in range(first, last):
            active = []
            for index, stream in enumerate(group):
                offset = tick - stream.source.offset_tick
                if 0 <= offset < len(stream.tokens):
                    window = (session.begin(stream.source.window_id, tick) if offset == 0
                              else session.windows[stream.source.window_id])
                    active.append((index, window, stream.tokens[offset]))
            if not active:
                continue
            logits, aux = TemporalGroup.advance(session, [w for _, w, _ in active], tick)
            auxiliaries.append(aux * len(active))
            for row, (index, window, target) in enumerate(active):
                rows[index].append(logits[row:row + 1])
                session.record(window, target, tick)
        count = sum(len(s.tokens) for s in group)
        return torch.cat([torch.cat(r) for r in rows]), torch.stack(auxiliaries).sum() / count
