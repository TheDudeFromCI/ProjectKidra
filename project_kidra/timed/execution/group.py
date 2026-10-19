"""Execute one timing group with a packed or an independent single-tick path."""

import torch
from torch import Tensor

from project_kidra.timed.execution.tick import NativeTick
from project_kidra.timed.schedule import EncodedStream
from project_kidra.timed.session import TimedSession


class GroupExecution:
    """Return logits in stream order so both paths optimize exactly the same labels."""

    @staticmethod
    def forward(session: TimedSession, group: list[EncodedStream], mode: str) -> tuple[Tensor, Tensor]:
        """Commit targets only after their matching forward, never advance idle windows."""

        if mode == "packed":
            updates = {}
            for stream in group:
                window = session.begin(stream.source.window_id).expect(stream.tokens)
                tick = stream.source.offset_tick
                updates[window] = list(range(tick, tick + len(stream.tokens)))
            logits, aux = session.forward(updates)
            for window in updates:
                session.commit(window, list(window.pending))
            return logits, aux
        rows: list[list[Tensor]] = [[] for _ in group]
        auxiliaries: list[Tensor] = []
        first = min(s.source.offset_tick for s in group)
        last = max(s.source.offset_tick + len(s.tokens) for s in group)
        for tick in range(first, last):
            active = []
            for index, stream in enumerate(group):
                offset = tick - stream.source.offset_tick
                if 0 <= offset < len(stream.tokens):
                    window = (session.begin(stream.source.window_id) if offset == 0
                              else session.windows[stream.source.window_id])
                    window.expect([stream.tokens[offset]])
                    active.append((index, window, stream.tokens[offset]))
            if not active:
                continue
            logits, aux = NativeTick.forward(
                session.model, [w for _, w, _ in active], session.cross_window
            )
            auxiliaries.append(aux * len(active))
            for row, (index, window, target) in enumerate(active):
                rows[index].append(logits[row:row + 1])
                session.commit(window, [target])
        count = sum(len(s.tokens) for s in group)
        return torch.cat([torch.cat(r) for r in rows]), torch.stack(auxiliaries).sum() / count
