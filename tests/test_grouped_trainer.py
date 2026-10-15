"""Lifecycle checks for multi-window sample isolation."""

import pytest

from project_kidra.model.config import Config
from project_kidra.model.grouped_trainer import GroupedTrainer
from project_kidra.model.model import AgentModel


def model() -> AgentModel:
    """Build a tiny scratch backend to exercise the shared trainer."""

    return AgentModel(
        Config(vocab_size=16, d_model=16, n_heads=2, n_layers=1, n_windows=3, n_local_ctx=8),
        free_token=15,
    )


def test_completed_context_survives_until_whole_sample_finishes() -> None:
    """Do not replace a short window with an unrelated sample during a long window."""

    agent = model()
    trainer = GroupedTrainer(agent, [[[1], [2, 3, 4]], [[5], [6]]], tokens_per_step=1)
    assert len(trainer.step()) == 2
    first, second = trainer.windows
    assert not first.pending and first in agent.windows
    assert len(trainer.step()) == 1
    assert first.position == 1 and second.position == 2
    assert len(trainer.step()) == 1
    assert agent.windows == []
    assert len(trainer.step()) == 2
    assert agent.windows == []
    assert trainer.step() == []


def test_invalid_groups_rejected_before_window_mutation() -> None:
    """Invalid token IDs and excessive window counts cannot partially open a group."""

    for sample in ([], [[1], []], [[1], [99]], [[1]] * 4):
        agent = model()
        trainer = GroupedTrainer(agent, [sample])
        with pytest.raises(ValueError):
            trainer.step()
        assert not agent.windows


def test_foreign_windows_rejected() -> None:
    """Group isolation requires exclusive ownership of the model's windows."""

    agent = model()
    agent.open_window()
    with pytest.raises(ValueError):
        GroupedTrainer(agent, [])
