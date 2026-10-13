"""Regression tests for event-causal, teacher-forced chunk training."""

import copy
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from project_kidra.model.config import Config
from project_kidra.model.model import AgentModel
from project_kidra.model.trainer import Trainer
from project_kidra.model.training.sequence import SequenceBatch


def make_model(context: int = 3) -> AgentModel:
    """Build deterministic small multi-layer models with stable window identities."""

    torch.manual_seed(21)
    model = AgentModel(
        Config(
            vocab_size=16,
            d_model=16,
            d_hidden=24,
            n_layers=2,
            n_heads=2,
            n_windows=3,
            n_local_ctx=context,
            n_experts=2,
        ),
        free_token=15,
    )
    for identity in range(3):
        model.open_window().id = identity
    return model


def commit(batch: SequenceBatch) -> None:
    """Commit supplied targets without changing weights for an execution reference."""

    for window, targets in zip(batch.windows, batch.targets):
        window.position += len(targets)
        del window.pending[: len(targets)]
        for token in targets:
            window.append(token)
        window.detach_cache()


@pytest.mark.parametrize("context", [1, 3, 16])
def test_chunk_matches_scheduled_single_ticks(context: int) -> None:
    """Ragged chunks reproduce each tick's logits and eviction, including idle history."""

    model = make_model(context)
    for window in model.windows:
        window.expect([2, 4])
    seed = SequenceBatch(model, {w: [0, 1] for w in model.windows})
    with torch.no_grad():
        seed.forward()
    commit(seed)
    model.windows[0].expect([1, 2, 3, 4, 5, 6])
    model.windows[1].expect([7, 8, 9])
    reference = copy.deepcopy(model)
    schedule = [[0, 1, 2, 4, 5, 8], [1, 5, 7]]
    idle_before = model.windows[2].cache[0].k.clone()
    with torch.no_grad():
        batch = SequenceBatch(model, dict(zip(model.windows[:2], schedule)))
        actual, _ = batch.forward()
        commit(batch)
        expected: list[list[Tensor]] = [[], []]
        for tick in range(9):
            active = [i for i, ticks in enumerate(schedule) if tick in ticks]
            if not active:
                continue
            step = SequenceBatch(reference, {reference.windows[i]: [0] for i in active})
            logits, _ = step.forward()
            for row, index in enumerate(active):
                expected[index].append(logits[row : row + 1])
            commit(step)
    torch.testing.assert_close(
        actual, torch.cat([torch.cat(rows) for rows in expected]), atol=2e-6, rtol=2e-5
    )
    for a, b in zip(model.windows, reference.windows):
        assert a.position == b.position
        assert a.tokens == b.tokens
        for ac, bc in zip(a.cache, b.cache):
            torch.testing.assert_close(ac.k, bc.k, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(ac.v, bc.v, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(model.windows[2].cache[0].k, idle_before)


def test_one_token_matches_legacy_loss_and_gradients() -> None:
    """The chunk size-one path preserves existing supervised forward semantics."""

    model = make_model()
    for window in model.windows:
        window.expect([window.id + 1])
    reference = copy.deepcopy(model)
    with (
        patch.object(reference.trunk_optimizer, "step"),
        patch.object(reference.expert_optimizer, "step"),
    ):
        expected = reference()
    with patch.object(model.trunk_optimizer, "step"), patch.object(model.expert_optimizer, "step"):
        actual = model.train_chunk(1)
    assert actual == pytest.approx(expected, abs=2e-6)
    for a, b in zip(model.parameters(), reference.parameters()):
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)


def test_future_targets_cannot_change_earlier_logits() -> None:
    """Cross-window paths cannot expose a future or same-tick target in any layer."""

    model = make_model()
    model.windows[0].expect([1, 2, 3, 4])
    model.windows[1].expect([5, 6, 7])
    changed = copy.deepcopy(model)
    changed.windows[0].pending[2:] = [9, 10]
    changed.windows[1].pending[1:] = [11, 12]
    outputs = []
    for candidate in (model, changed):
        with torch.no_grad():
            logits, _ = SequenceBatch(
                candidate,
                {
                    candidate.windows[0]: [0, 1, 2, 3],
                    candidate.windows[1]: [0, 2, 4],
                },
            ).forward()
        outputs.append(logits)
    torch.testing.assert_close(outputs[0][[0, 1, 2, 4, 5]], outputs[1][[0, 1, 2, 4, 5]])
    assert not torch.allclose(outputs[0][3], outputs[1][3])


def test_chunk_updates_only_scheduled_windows_and_detaches() -> None:
    """One optimizer step consumes all requested targets and leaves other windows untouched."""

    model = make_model()
    a, b, idle = model.windows
    a.expect([1, 2, 3, 4])
    b.expect([5, 6])
    idle.expect([7])
    before = model.classifier.weight.detach().clone()
    with patch.object(model.trunk_optimizer, "step", wraps=model.trunk_optimizer.step) as step:
        losses = model.train_chunk(updates={a: [0, 1, 2], b: [0, 2]})
        assert step.call_count == 1
    assert len(losses) == 5
    assert (a.position, b.position, idle.position) == (3, 2, 0)
    assert (a.pending, b.pending, idle.pending) == ([4], [], [7])
    assert idle.cache[0].length == 0
    assert not torch.equal(before, model.classifier.weight)
    assert all(
        not c.k.requires_grad and not c.v.requires_grad for w in model.windows for c in w.cache
    )
    assert len(model.train_chunk(1)) == 2


@pytest.mark.parametrize("ticks", [[1, 0], [0, 0], [-1], [0, 1, 2]])
def test_invalid_schedule_does_not_consume_targets(ticks: list[int]) -> None:
    """Schedule errors are rejected before caches or queues are modified."""

    model = make_model()
    window = model.windows[0].expect([1, 2])
    with pytest.raises(ValueError):
        model.train_chunk(updates={window: ticks})
    assert window.pending == [1, 2]
    assert window.position == 0 and window.cache[0].length == 0


def test_empty_and_free_token_chunks() -> None:
    """Empty schedules do no work; unknown future generated inputs cannot be batched."""

    model = make_model()
    assert model.train_chunk() == []
    with pytest.raises(ValueError):
        model.train_chunk(0)
    window = model.windows[0].expect([1, 15])
    with pytest.raises(ValueError, match="known targets"):
        model.train_chunk()
    assert window.pending == [1, 15]
    foreign = make_model().windows[0].expect([1])
    with pytest.raises(ValueError, match="not open"):
        model.train_chunk(updates={foreign: [0]})


def test_trainer_chunks_unequal_samples_without_advancing_foreign_window() -> None:
    """Trainer returns actual token losses, closes completed samples, and refills next step."""

    model = make_model()
    for window in list(model.windows[1:]):
        model.close_window(window)
    foreign = model.windows[0].expect([8, 9])
    trainer = Trainer(model, [[1, 2, 3], [4], [5, 6]], 2, tokens_per_step=2)
    assert len(trainer.step()) == 3
    assert len(trainer.step()) == 3
    assert trainer.step() == []
    assert model.windows == [foreign]
    assert foreign.pending == [8, 9] and foreign.position == 0
    with pytest.raises(ValueError):
        Trainer(model, [], 1, tokens_per_step=0)


def test_chunk_training_learns_repeating_sequence() -> None:
    """Multi-token loss decreases through the complete trainer/optimizer path."""

    model = make_model()
    for window in list(model.windows):
        model.close_window(window)
    for optimizer in (model.trunk_optimizer, model.expert_optimizer):
        for group in optimizer.param_groups:
            group["lr"] = 0.01
    trainer = Trainer(model, [[1, 2, 1, 2, 1, 2]] * 35, 1, tokens_per_step=6)
    losses = [sum(trainer.step()) / 6 for _ in range(35)]
    assert sum(losses[-5:]) / 5 < losses[0] * 0.35


def test_multi_token_chunk_matches_legacy_fixed_weight_execution() -> None:
    """With optimizer updates disabled, chunk losses match the original streaming path."""

    model = make_model(context=2)
    for window in model.windows:
        window.expect([1 + window.id, 4, 5, 6, 7])
    reference = copy.deepcopy(model)
    with (
        patch.object(reference.trunk_optimizer, "step"),
        patch.object(reference.expert_optimizer, "step"),
    ):
        expected = torch.tensor([reference() for _ in range(5)]).T.reshape(-1)
    with patch.object(model.trunk_optimizer, "step"), patch.object(model.expert_optimizer, "step"):
        actual = torch.tensor(model.train_chunk(5))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_later_target_backpropagates_through_earlier_chunk_inputs() -> None:
    """A last-position-only objective reaches a unique earlier input via attention."""

    model = make_model(context=4)
    window = model.windows[0].expect([1, 2, 3, 4])
    logits, _ = SequenceBatch(model, {window: [0, 1, 2, 3]}).forward()
    logits[-1, 4].backward()  # type: ignore
    gradient = model.token_embeddings.weight.grad
    assert gradient is not None
    assert gradient[1].abs().sum() > 0
