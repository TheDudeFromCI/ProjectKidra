"""Tests for the Trainer, which streams corpus samples through short-lived windows."""

import pytest

from project_kidra.model.config import Config
from project_kidra.model.model import AgentModel
from project_kidra.model.trainer import Trainer

CONFIG = Config(
    vocab_size=16, d_model=16, d_hidden=16, n_layers=2, n_heads=2, n_windows=4, n_local_ctx=8
)


def make_model() -> AgentModel:
    """Create a small model."""

    return AgentModel(CONFIG, free_token=15)


def samples(*lengths: int) -> list[list[int]]:
    """Create samples of the given lengths."""

    return [[int(i % 10) for i in range(n)] for n in lengths]


def test_rejects_invalid_window_count() -> None:
    """The trainer cannot hold more windows than the model supports."""

    with pytest.raises(ValueError):
        Trainer(make_model(), [], 0)
    with pytest.raises(ValueError):
        Trainer(make_model(), [], CONFIG.n_windows + 1)


def test_keeps_target_windows_open() -> None:
    """With plenty of samples, exactly n_windows stay open after each step."""

    model = make_model()
    trainer = Trainer(model, samples(*[5] * 10), 3)

    for _ in range(8):
        trainer.step()
        assert len(model.windows) <= 3
    assert len(trainer.windows) <= 3


def test_window_closes_when_sample_completes() -> None:
    """A window is closed once its sample is consumed and replaced from the corpus."""

    model = make_model()
    trainer = Trainer(model, samples(2, 4), 1)

    assert len(trainer.step()) == 1
    assert len(model.windows) == 1
    assert len(trainer.step()) == 1
    assert len(model.windows) == 0
    assert len(model.available_window_ids) == CONFIG.n_windows

    # Next step pulls the second sample.
    assert len(trainer.step()) == 1
    assert len(model.windows) == 1


def test_returns_empty_when_exhausted() -> None:
    """Once samples run out and windows close, step returns nothing."""

    trainer = Trainer(make_model(), samples(1), 2)
    assert len(trainer.step()) == 1
    assert trainer.step() == []


def test_ignores_foreign_windows() -> None:
    """Windows opened outside the trainer are neither closed nor reported."""

    model = make_model()
    foreign = model.open_window()
    foreign.append(1)
    trainer = Trainer(model, samples(1, 1), 1)

    losses = trainer.step()

    assert len(losses) == 1
    assert foreign in model.windows
    assert foreign not in trainer.windows
    assert len(model.windows) == 1


def test_skips_empty_samples() -> None:
    """Empty samples don't open windows."""

    trainer = Trainer(make_model(), samples(0, 2), 1)
    assert len(trainer.step()) == 1
    assert len(trainer.windows) == 1
