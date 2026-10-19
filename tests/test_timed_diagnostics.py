"""Distinguish control, fact, and upstream failures with controlled generation fixtures."""

from unittest.mock import patch

import torch
from test_timed_execution import tiny_model

from project_kidra.timed.execution.tick import NativeTick
from project_kidra.timed.generator import ToyGenerator
from project_kidra.timed.matrix.evaluate import Diagnostics
from project_kidra.timed.matrix.tasks import TaskProjection
from project_kidra.timed.schedule import EncodedStream
from project_kidra.timed.session import TimedSession
from project_kidra.timed.tokenizer import ToyTokenizer


def test_conditional_metrics_do_not_confuse_fact_errors_with_controls() -> None:
    """A wrong call fails its fact metric while correct pauses still score correctly."""

    tok = ToyTokenizer()
    sample = ToyGenerator(5).sample(0)
    evaluator = Diagnostics(tiny_model(), tok)

    def generated(session: TimedSession, group: list[EncodedStream]) -> list[list[int]]:
        """Script content failures while preserving token counts and control types."""

        result = []
        for stream in group:
            tokens = list(stream.tokens)
            if stream.source.text.startswith("TOOL CALL"):
                tokens[2] = tok.ids["k0"] if tokens[2] != tok.ids["k0"] else tok.ids["k1"]
            elif stream.source.text.startswith(("FOUND", "ANSWER")) and 4 not in session.windows:
                tokens[1] = tok.ids["v0"] if tokens[1] != tok.ids["v0"] else tok.ids["v1"]
            result.append(tokens)
        return result

    with patch.object(evaluator, "_generate", generated):
        free = evaluator.rollout(sample)
        conditional = evaluator.rollout(sample, oracle_prefix=True)
    assert free["controls"] == conditional["controls"] == 1.0
    assert free["call_fact"] == conditional["call_fact"] == 0.0
    assert free["reasoning_fact"] == free["answer_fact"] == 0.0
    assert conditional["reasoning_fact"] == conditional["answer_fact"] == 1.0
    assert free["exact"] == conditional["exact"] == 0.0


def test_early_generated_end_is_not_replaced_by_reference() -> None:
    """Native greedy inference stops on its own control token and fails missing facts."""

    tok, model = ToyTokenizer(), tiny_model()
    sample = TaskProjection.apply(ToyGenerator(5).sample(0), "lookup2")
    original = NativeTick.forward
    calls = 0

    def end_early(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep real observation caches but end the answer at its first generation step."""

        nonlocal calls
        logits, aux = original(*args, **kwargs)  # type: ignore[arg-type]
        calls += 1
        logits = torch.full_like(logits, -100)
        logits[:, tok.end] = 100
        return logits, aux

    with patch.object(NativeTick, "forward", end_early):
        result = Diagnostics(model, tok).rollout(sample)
    assert calls > 1
    assert result["answer_fact"] == result["controls"] == result["exact"] == 0
    assert not model.windows
