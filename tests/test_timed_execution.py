"""Numerical controls for attached graphs and the independent no-batching path."""

import copy
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from project_kidra.model.config import Config
from project_kidra.model.model import AgentModel
from project_kidra.model.training.sequence import SequenceBatch
from project_kidra.timed.execution.group import GroupExecution
from project_kidra.timed.execution.policy import TrainingPolicy
from project_kidra.timed.generator import ToyGenerator
from project_kidra.timed.matrix.evaluate import Diagnostics
from project_kidra.timed.matrix.tasks import TaskProjection
from project_kidra.timed.schedule import Schedule
from project_kidra.timed.schema import ContentStream, TimedSample
from project_kidra.timed.session import TimedSession
from project_kidra.timed.tokenizer import ToyTokenizer
from project_kidra.timed.trainer import TimedTrainer


def tiny_model() -> AgentModel:
    """Use roughly ten thousand parameters for bounded implementation tests."""

    torch.manual_seed(42)
    tok = ToyTokenizer()
    return AgentModel(Config(vocab_size=len(tok.words), d_model=24, d_hidden=64, n_layers=2,
                             n_heads=2, n_experts=1, expert_top_k=1, n_windows=5,
                             n_local_ctx=128, aux_loss_weight=0.0,
                             trunk_lr=0.003, expert_lr=0.003), tok.bos)


@pytest.mark.parametrize("gradients", ["full", "detached"])
@pytest.mark.parametrize("cross_window", [True, False])
def test_packed_and_native_tick_match_loss_and_gradients(gradients: str, cross_window: bool) -> None:
    """Temporal batching changes neither targets, sample updates, nor gradients."""

    a = tiny_model()
    b = copy.deepcopy(a)
    tok = ToyTokenizer()
    sample = ToyGenerator(9).sample(0)
    saved: list[dict[str, Tensor]] = []
    results = []
    for model, execution in ((a, "packed"), (b, "tick")):
        def capture(candidate: AgentModel = model) -> None:
            """Inspect the exact gradients reaching each optimizer."""

            saved.append({n: p.grad.clone() for n, p in candidate.named_parameters() if p.grad is not None})

        policy = TrainingPolicy(gradients=gradients, execution=execution, cross_window=cross_window)  # type: ignore[arg-type]
        with patch.object(model.trunk_optimizer, "step", capture), patch.object(model.expert_optimizer, "step"):
            results.append(TimedTrainer(model, tok, policy=policy).step(sample))
    assert results[0].loss == pytest.approx(results[1].loss, abs=2e-6)
    assert saved[0].keys() == saved[1].keys()
    for name in saved[0]:
        torch.testing.assert_close(saved[0][name], saved[1][name], atol=3e-6, rtol=3e-4)


@pytest.mark.parametrize("mode", ["packed", "tick"])
def test_full_graph_reaches_observation_only_embeddings(mode: str) -> None:
    """The lookup objective trains document encoding only when cache graphs survive."""

    tok = ToyTokenizer()
    gradients = []
    for policy in (TrainingPolicy(gradients="full", execution=mode),  # type: ignore[arg-type]
                   TrainingPolicy(gradients="detached", execution=mode)):  # type: ignore[arg-type]
        model = tiny_model()

        def capture(candidate: AgentModel = model) -> None:
            """MAP occurs only in an external document, never as a supervised input."""

            grad = candidate.token_embeddings.weight.grad
            assert grad is not None
            gradients.append(float(grad[tok.ids["MAP"]].abs().sum()))

        with patch.object(model.trunk_optimizer, "step", capture), patch.object(model.expert_optimizer, "step"):
            result = TimedTrainer(model, tok, policy=policy).step(ToyGenerator(1).sample(0))
        assert result.backward_passes == (1 if policy.gradients == "full" else 3)
    assert gradients[0] > 0
    assert gradients[1] == 0


def test_tick_execution_never_calls_sequence_batch() -> None:
    """The no-batching control uses a genuinely independent layer execution path."""

    model, tok = tiny_model(), ToyTokenizer()
    with patch.object(SequenceBatch, "forward", side_effect=AssertionError("batched path used")):
        TimedTrainer(model, tok, policy=TrainingPolicy(execution="tick")).step(ToyGenerator(3).sample(0))


def test_tick_echo_learning_on_small_fixture() -> None:
    """Single-token native execution learns cross-window echo on a tiny bounded test."""

    model, tok = tiny_model(), ToyTokenizer()
    trainer = TimedTrainer(model, tok, policy=TrainingPolicy(execution="tick"))
    samples = [TimedSample(f"echo-{i}", (
        ContentStream(0, 0, f"v{i}", False),
        ContentStream(1, 2, f"ANSWER v{i}", True, "<|end|>"),
    ), (2, 5)) for i in range(4)]
    with patch.object(SequenceBatch, "forward", side_effect=AssertionError("batched path used")):
        for i in range(256):
            trainer.step(samples[i % len(samples)])
        evaluator = Diagnostics(model, tok)
        normal = evaluator.evaluate(samples)["answer_fact"]
        ablated = evaluator.evaluate(samples, cross_window=False)["answer_fact"]
    assert normal > 0.8
    assert ablated < 0.4


def test_future_group_observations_cannot_change_earlier_outputs() -> None:
    """Full autograd links losses backward without making future facts visible forward."""

    tok = ToyTokenizer()
    base = ToyGenerator(2).sample(0)
    changed = TaskProjection.counterfactual(base, "agent")
    outputs = []
    for sample in (base, changed):
        model = tiny_model()
        session = TimedSession(model, tok)
        schedule = Schedule(sample, tok, 5, 128)
        for group in schedule.groups[:2]:
            logits, _ = GroupExecution.forward(session, group, "packed")
        outputs.append(logits.detach())
        session.close()
    torch.testing.assert_close(outputs[0], outputs[1])
