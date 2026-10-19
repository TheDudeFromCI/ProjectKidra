"""Protocol tests on tiny fixtures; these do not run the toy training experiment."""

import copy
import json
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from project_kidra.model.config import Config
from project_kidra.model.model import AgentModel
from project_kidra.model.window import ContextWindow
from project_kidra.timed.evaluator import ToyEvaluator
from project_kidra.timed.generator import ToyGenerator
from project_kidra.timed.schedule import Schedule
from project_kidra.timed.schema import ContentStream, TimedSample
from project_kidra.timed.session import TimedSession
from project_kidra.timed.tokenizer import ToyTokenizer
from project_kidra.timed.trainer import TimedTrainer


@pytest.fixture
def model() -> AgentModel:
    """Create a small two-layer model with explicit toy window slots."""

    torch.manual_seed(8)
    tokenizer = ToyTokenizer()
    return AgentModel(Config(
        vocab_size=len(tokenizer.words), d_model=16, d_hidden=24, n_layers=2,
        n_experts=1, expert_top_k=1, n_heads=2, n_windows=5, n_local_ctx=128,
        aux_loss_weight=0.0,
    ), tokenizer.bos)


def test_generated_splits_and_json_roundtrip() -> None:
    """Generation is reproducible; held-out query/answer pairs never enter training."""

    tokenizer = ToyTokenizer()
    for split in ("train", "test"):
        left, right = ToyGenerator(15, split, True), ToyGenerator(15, split, True)
        for index in range(20):
            sample = left.sample(index)
            assert sample == right.sample(index)
            assert TimedSample.from_dict(json.loads(json.dumps(sample.to_dict()))) == sample
            Schedule(sample, tokenizer, 5, 128)
            query = int(sample.streams[1].text.split()[-1][1:])
            answer = int(sample.streams[-1].text.split()[-1][1:])
            assert ((query + answer) % 4 == 0) == (split == "test")
            assert sample.streams[3].offset_tick > sample.group_ends[0]


def test_ticks_include_one_terminal_and_allow_concurrent_windows() -> None:
    """Twenty-five content tokens and NOOP consume ticks 0..25, then tick 26 opens."""

    sample = TimedSample("ticks", (
        ContentStream(0, 0, " ".join(["FILL"] * 25), False),
        ContentStream(1, 0, "USER", False),
        ContentStream(2, 26, "ANSWER", True, "<|end|>"),
    ), (26, 28))
    schedule = Schedule(sample, ToyTokenizer(), 5, 128)
    assert len(schedule.groups[0][0].tokens) == 26
    assert schedule.groups[1][0].source.offset_tick == 26


@pytest.mark.parametrize("change", ["overlap", "completed", "boundary", "control", "capacity"])
def test_invalid_schedule_rejected_before_any_model_change(model: AgentModel, change: str) -> None:
    """Malformed timelines cannot partially update weights or allocate windows."""

    sample = ToyGenerator(1).sample(0)
    streams = list(sample.streams)
    if change == "overlap":
        streams[3] = replace(streams[3], offset_tick=0)
    elif change == "completed":
        streams[1] = replace(streams[1], terminal="<|end|>")
    elif change == "boundary":
        streams[-1] = replace(streams[-1], offset_tick=sample.group_ends[-1] - 1)
    elif change == "control":
        streams[0] = replace(streams[0], text="<|noop|>")
    else:
        streams[0] = replace(streams[0], window_id=5)
    with patch.object(model.trunk_optimizer, "step") as step:
        with pytest.raises(ValueError):
            TimedTrainer(model, ToyTokenizer()).step(replace(sample, streams=tuple(streams)))
        step.assert_not_called()
    assert not model.windows
    with pytest.raises(ValueError, match="context"):
        Schedule(sample, ToyTokenizer(), 5, 2)


def test_resume_replays_input_without_ingesting_noop(model: AgentModel) -> None:
    """Resuming removes the unconsumed target and one cached input, retaining positions."""

    tokenizer = ToyTokenizer()
    session = TimedSession(model, tokenizer)
    session.observe([(1, 0, tokenizer.encode("USER ASK k1 <|noop|>"))])
    window = session.windows[1]
    assert window.position == window.cache[0].length == 4
    window = session.begin(1)
    assert window.tokens == tokenizer.encode("USER ASK k1")
    assert window.position == window.cache[0].length == 3
    window.expect(tokenizer.encode("ANSWER v2 <|end|>"))
    with torch.no_grad():
        session.forward({window: [20, 21, 22]})
    session.commit(window, list(window.pending))
    assert window.position == window.cache[0].length == 6
    assert tokenizer.noop not in window.tokens
    with pytest.raises(ValueError, match="NOOP"):
        session.begin(1)
    session.close()
    assert not model.windows and sorted(model.available_window_ids) == list(range(5))


def test_loss_masks_observations_and_accumulates_per_sample(model: AgentModel) -> None:
    """Three backwards precede one optimizer step; only predicted tokens affect CE."""

    tokenizer = ToyTokenizer()
    sample = ToyGenerator(3).sample(0)
    schedule = Schedule(sample, tokenizer, 5, 128)
    original = TimedSession.forward
    outputs: list[Tensor] = []
    baseline = model.classifier.weight.detach().clone()

    def capture(session: TimedSession, updates: dict[ContextWindow, list[int]]) -> tuple[Tensor, Tensor]:
        """Record fixed-weight logits and confirm no early optimizer updates."""

        torch.testing.assert_close(session.model.classifier.weight, baseline)
        result = original(session, updates)
        outputs.append(result[0].detach())
        return result

    with (
        patch.object(TimedSession, "forward", capture),
        patch.object(model.trunk_optimizer, "step") as trunk,
        patch.object(model.expert_optimizer, "step") as expert,
    ):
        result = TimedTrainer(model, tokenizer).step(sample)
    trunk.assert_called_once()
    expert.assert_called_once()
    numerator = denominator = 0.0
    labels = 0
    for logits, group in zip(outputs, schedule.groups):
        offset = 0
        for stream in group:
            if stream.source.predict:
                loss = torch.nn.functional.cross_entropy(
                    logits[offset:offset + len(stream.tokens)], torch.tensor(stream.tokens),
                    reduction="none",
                )
                weight = torch.tensor([0.25 if t == tokenizer.noop else 1.0 for t in stream.tokens])
                numerator += float((loss * weight).sum())
                denominator += float(weight.sum())
                labels += len(stream.tokens)
            offset += len(stream.tokens)
    assert result.loss == pytest.approx(numerator / denominator)
    assert result.labels == labels == 11
    assert result.backward_passes == 3
    assert not model.windows and all(p.grad is None for p in model.parameters())


def test_exception_closes_windows_and_skips_optimizer(model: AgentModel) -> None:
    """Failed forward passes cannot leak context or partial accumulated gradients."""

    with (
        patch.object(TimedSession, "forward", side_effect=RuntimeError("fixture")),
        patch.object(model.trunk_optimizer, "step") as step,
    ):
        with pytest.raises(RuntimeError, match="fixture"):
            TimedTrainer(model, ToyTokenizer()).step(ToyGenerator(3).sample(0))
        step.assert_not_called()
    assert not model.windows and all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("cross_window", [False, True])
def test_ablation_removes_cross_window_content_paths(model: AgentModel, cross_window: bool) -> None:
    """Changing an observed document affects outputs only when exchange is enabled."""

    other = copy.deepcopy(model)
    tokenizer = ToyTokenizer()
    outputs = []
    for candidate, value in ((model, "v1"), (other, "v2")):
        session = TimedSession(candidate, tokenizer, cross_window)
        session.observe([(2, 0, tokenizer.encode(f"RESULT {value} <|noop|>"))])
        window = session.begin(1).expect(tokenizer.encode("ANSWER <|end|>"))
        with torch.no_grad():
            outputs.append(session.forward({window: [10, 11]})[0])
        session.close()
    if cross_window:
        assert not torch.allclose(outputs[0], outputs[1])
    else:
        torch.testing.assert_close(outputs[0], outputs[1])


@pytest.mark.parametrize("wrong_call", [False, True])
def test_evaluator_obeys_predictions_and_gates_tool_result(model: AgentModel, wrong_call: bool) -> None:
    """Scripted logits verify scoring and deny the reference tool result after a wrong call."""

    tokenizer = ToyTokenizer()
    sample = ToyGenerator(4).sample(0)
    schedule = Schedule(sample, tokenizer, 5, 128)
    expected = {(s.source.window_id, s.source.offset_tick + i): token
                for group in schedule.groups for s in group for i, token in enumerate(s.tokens)}
    call = sample.streams[4]
    incorrect = tokenizer.ids["k0"] if not call.text.endswith("k0") else tokenizer.ids["k1"]
    seen_tool = False
    saw_generated_input = False
    original = TimedSession.forward

    def scripted(session: TimedSession, updates: dict[ContextWindow, list[int]]) -> tuple[Tensor, Tensor]:
        """Run real cache operations, replacing only output logits with controlled choices."""

        nonlocal seen_tool, saw_generated_input
        seen_tool |= any(w.id == 4 for w in updates)
        if wrong_call and any(w.id == 3 and w.tokens and w.tokens[-1] == incorrect for w in updates):
            saw_generated_input = True
        logits, aux = original(session, updates)
        result = torch.full_like(logits, -100.0)
        row = 0
        for window, ticks in updates.items():
            for tick in ticks:
                token = expected[(window.id, tick)]
                if wrong_call and window.id == 3 and tick == call.offset_tick + 2:
                    token = incorrect
                result[row, token] = 100.0
                row += 1
        return result, aux

    with patch.object(TimedSession, "forward", scripted):
        result = ToyEvaluator(model, tokenizer).rollout(sample)
    assert result.call_correct == result.exact == (not wrong_call)
    assert seen_tool == (not wrong_call)
    assert saw_generated_input == wrong_call
    assert result.controls_correct
    assert not model.windows
