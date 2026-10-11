"""Tests for saving and loading AgentModel checkpoints."""

from pathlib import Path

import pytest
import torch

from project_kidra.model.checkpoint import Checkpoint
from project_kidra.model.config import Config
from project_kidra.model.model import AgentModel

SMALL = Config(vocab_size=16, d_model=16, d_hidden=16, n_layers=2, n_heads=2, n_local_ctx=8)


def make_model() -> AgentModel:
    """Create a small model."""

    return AgentModel(SMALL, free_token=15)


class TestCheckpoint:
    """Round-trip behavior of Checkpoint."""

    def test_round_trip_restores_config_and_weights(self, tmp_path: Path) -> None:
        """A loaded model has the same config, free token, and parameters."""

        model = make_model()
        path = tmp_path / "model.pt"
        Checkpoint.save(model, path)
        loaded = Checkpoint.load(path)

        assert loaded.config == model.config
        assert loaded.free_token == model.free_token
        original = model.state_dict()
        restored = loaded.state_dict()
        assert original.keys() == restored.keys()
        for key in original:
            assert torch.equal(original[key], restored[key])

    def test_save_creates_missing_directories(self, tmp_path: Path) -> None:
        """Saving into a nonexistent directory creates it."""

        path = tmp_path / "nested" / "dir" / "model.pt"
        Checkpoint.save(make_model(), path)
        assert path.is_file()

    def test_optimizer_state_is_restored(self, tmp_path: Path) -> None:
        """Training state survives a round trip, so the optimizer has moment buffers."""

        model = make_model()
        window = model.open_window()
        window.append(1)
        window.append(2)
        model.forward()

        path = tmp_path / "model.pt"
        Checkpoint.save(model, path)
        loaded = Checkpoint.load(path)

        assert len(loaded.trunk_optimizer.state_dict()["state"]) > 0
        assert len(loaded.expert_optimizer.state_dict()["state"]) > 0

    def test_loaded_model_can_train(self, tmp_path: Path) -> None:
        """A loaded model can open a window and run a training step."""

        path = tmp_path / "model.pt"
        Checkpoint.save(make_model(), path)
        loaded = Checkpoint.load(path)

        window = loaded.open_window()
        window.append(3)
        window.append(4)
        assert len(loaded.forward()) == 1

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """Loading a nonexistent path raises FileNotFoundError."""

        with pytest.raises(FileNotFoundError):
            Checkpoint.load(tmp_path / "missing.pt")
