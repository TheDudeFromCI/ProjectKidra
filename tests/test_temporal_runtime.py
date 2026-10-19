"""Small forward checks and exactly two unsaved single-sample training smoke passes."""

import copy
from unittest.mock import patch

import pytest
import torch
from test_timed_execution import tiny_model

from project_kidra.model.training.sequence import SequenceBatch
from project_kidra.temporal.runtime.evaluate import TemporalDiagnostics
from project_kidra.temporal.runtime.group import TemporalGroup
from project_kidra.temporal.runtime.session import TemporalSession
from project_kidra.temporal.runtime.tick import TemporalTick
from project_kidra.temporal.runtime.trainer import TemporalTrainer
from project_kidra.timed.execution.group import GroupExecution
from project_kidra.timed.experiment import ToyExperiment
from project_kidra.timed.matrix.evaluate import Diagnostics
from project_kidra.timed.schedule import Schedule
from project_kidra.timed.schema import ContentStream, TimedSample
from project_kidra.timed.session import TimedSession
from project_kidra.timed.tokenizer import ToyTokenizer


def sample() -> TimedSample:
    """Exercise five windows, NOOP replay, terminal reasoning, and a real idle gap."""

    return TimedSample("temporal-fixture", (
        ContentStream(0, 0, "SYSTEM LOOKUP", False),
        ContentStream(1, 0, "USER ASK k0", False),
        ContentStream(2, 0, "MAP k0 k1", False),
        ContentStream(1, 7, "", True),
        ContentStream(3, 7, "TOOL CALL k1", True),
        ContentStream(4, 11, "RESULT k1 v1", False),
        ContentStream(3, 15, "FOUND v1", True, "<|end|>"),
        ContentStream(1, 18, "ANSWER v1", True, "<|end|>"),
    ), (4, 11, 15, 18, 21))


@torch.no_grad()
def test_none_matches_original_native_execution_and_diagnostics() -> None:
    """The extension introduces no unrelated numerical or evaluation changes."""

    model, tok = tiny_model(), ToyTokenizer()
    reference = copy.deepcopy(model)
    session = TemporalSession(model, tok, "none")
    original = TimedSession(reference, tok)
    try:
        for group in Schedule(sample(), tok, 5, 128).groups:
            actual, _ = TemporalGroup.forward(session, group)
            expected, _ = GroupExecution.forward(original, group, "tick")
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        session.close()
        original.close()
    for oracle in (False, True):
        assert TemporalDiagnostics(model, tok, "none").rollout(sample(), oracle_prefix=oracle) == (
            Diagnostics(reference, tok).rollout(sample(), oracle_prefix=oracle))


@torch.no_grad()
def test_birth_tick_survives_noop_replay_and_changes_encoding() -> None:
    """Waking a window at tick seven must not relabel its tick-two input as new content."""

    model, tok = tiny_model(), ToyTokenizer()
    before = set(model.state_dict())
    session = TemporalSession(model, tok, "global_time")
    try:
        groups = Schedule(sample(), tok, 5, 128).groups
        TemporalGroup.forward(session, groups[0])
        window = session.begin(1, 7)
        assert session.input_tick(window) == 2
        assert window.position == 3
        assert all(cache.length == 3 for cache in window.cache)
        timestamps = torch.tensor([0.0, 2.0, 7.0])
        encoded = TemporalTick.encoding(timestamps, 24)
        assert encoded.shape == (3, 24)
        assert not torch.equal(encoded[1], encoded[2])
        logits, _ = TemporalTick.forward(session, [window])
        assert torch.isfinite(logits).all()
    finally:
        session.close()
    assert set(model.state_dict()) == before


@torch.no_grad()
def test_continuous_reasoning_advances_exactly_once_per_event_tick() -> None:
    """Heartbeats fill active gaps, survive END, and neither double-count nor fill idle time."""

    model, tok = tiny_model(), ToyTokenizer()
    session = TemporalSession(model, tok, "continuous_reasoning")
    schedule = Schedule(sample(), tok, 5, 128)
    expected = {tick for group in schedule.groups for stream in group
                for tick in range(stream.source.offset_tick,
                                  stream.source.offset_tick + len(stream.tokens))}
    try:
        for group in schedule.groups:
            TemporalGroup.forward(session, group)
        window = session.windows[3]
        assert window.position == len(expected) == 18
        assert all(c.length == 18 for c in window.cache)
        assert set(session.births[3]) <= expected
        assert session.births[3][-1] == 20
        assert window.tokens[-1] == tok.ids["FILL"]
        assert tok.noop not in window.tokens and tok.end not in window.tokens
        assert tok.encode("TOOL CALL k1") == window.tokens[4:7]
    finally:
        session.close()


@pytest.mark.parametrize("variant", ["global_time", "continuous_reasoning"])
def test_one_unsaved_smoke_step_per_variant(variant: str) -> None:
    """Backpropagate through observations and heartbeats without invoking token batching."""

    model, tok = tiny_model(), ToyTokenizer()
    before = model.token_embeddings.weight.detach().clone()
    gradients: list[torch.Tensor] = []
    hook = model.token_embeddings.weight.register_hook(  # type: ignore[no-untyped-call]
        lambda grad: gradients.append(grad.clone()))
    try:
        with patch.object(SequenceBatch, "forward", side_effect=AssertionError("batched path")):
            result = TemporalTrainer(model, tok, variant=variant).step(sample())
    finally:
        hook.remove()
    assert result.backward_passes == 1 and result.labels == 11
    assert torch.isfinite(torch.tensor(result.loss)) and result.loss > 0
    assert not torch.equal(model.token_embeddings.weight, before)
    assert gradients[0][tok.ids["MAP"]].abs().sum() > 0
    if variant == "continuous_reasoning":
        assert gradients[0][tok.ids["FILL"]].abs().sum() > 0
    assert not model.windows
    for oracle in (False, True):
        metrics = TemporalDiagnostics(model, tok, variant).rollout(sample(), oracle_prefix=oracle)
        assert all(0 <= value <= 1 for value in metrics.values())
        assert not model.windows


def test_production_architecture_is_unchanged() -> None:
    """Construct once, without inference or training, to check the approximately 1M budget."""

    experiment = ToyExperiment()
    assert sum(p.numel() for p in experiment.model.parameters()) == 991462
