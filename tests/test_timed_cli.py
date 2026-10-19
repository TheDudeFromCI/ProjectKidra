"""Check command wiring and checkpoint metadata without running the experiment."""

import json
import random
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from project_kidra.timed.__main__ import ToyCLI
from project_kidra.timed.experiment import ToyExperiment
from project_kidra.timed.generator import ToyGenerator
from project_kidra.timed.schedule import Schedule
from project_kidra.timed.tokenizer import ToyTokenizer


def test_generate_cli_writes_valid_jsonl(tmp_path: Path) -> None:
    """The generator emits complete timed samples without constructing a model."""

    output = tmp_path / "data" / "test.jsonl"
    args = ["timed", "generate", "--count", "3", "--split", "test", "--seed", "7",
            "--delayed", "--output", str(output)]
    with patch("sys.argv", args), patch.object(ToyExperiment, "__init__") as init:
        ToyCLI.run()
        init.assert_not_called()
    samples = ToyExperiment.read(output)
    assert len(samples) == 3
    assert samples[0] == ToyGenerator(7, "test", True).sample(0)
    for sample in samples:
        Schedule(sample, ToyTokenizer(), 5, 128)


def test_checkpoint_restores_weights_rng_and_sample_order(tmp_path: Path) -> None:
    """An untrained model round-trips without any forward or optimizer execution."""

    experiment = ToyExperiment(seed=19)
    assert sum(p.numel() for p in experiment.model.parameters()) == 991462
    experiment.completed_steps = 17
    experiment.sample_order = [5, 1, 8]
    experiment.train_fingerprint = "fixture"
    checkpoint = tmp_path / "checkpoint.pt"
    experiment.save(checkpoint)
    expected_python = random.random()
    expected_torch = torch.rand(4)
    restored = ToyExperiment.load(checkpoint)
    assert restored.completed_steps == 17
    assert restored.sample_order == [5, 1, 8]
    assert restored.train_fingerprint == "fixture"
    assert random.random() == expected_python
    torch.testing.assert_close(torch.rand(4), expected_torch)
    for name, value in experiment.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[name], value)
    assert not restored.model.windows
    experiment.model.open_window()
    with pytest.raises(ValueError, match="completed sample"):
        experiment.save(checkpoint)


def test_train_cli_routes_resume_without_executing_training(tmp_path: Path) -> None:
    """Command wiring passes additional steps to a mocked orchestration boundary."""

    data = tmp_path / "train.jsonl"
    validation = tmp_path / "eval.jsonl"
    for path, split in ((data, "train"), (validation, "test")):
        path.write_text(json.dumps(ToyGenerator(1, split).sample(0).to_dict()) + "\n")
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "run"
    args = ["timed", "train", "--data", str(data), "--eval-data", str(validation),
            "--output", str(output), "--steps", "9", "--eval-every", "3",
            "--resume", str(checkpoint), "--device", "cpu"]
    experiment = ToyExperiment()
    with (
        patch("sys.argv", args),
        patch.object(ToyExperiment, "load", return_value=experiment) as load,
        patch.object(ToyExperiment, "train", return_value={}) as train,
    ):
        ToyCLI.run()
    load.assert_called_once_with(checkpoint, "cpu")
    train.assert_called_once_with(
        experiment.read(data), experiment.read(validation), 9, output, 3
    )


def test_evaluate_cli_routes_both_conditions_without_running_model(tmp_path: Path) -> None:
    """Saved-checkpoint evaluation requests the normal and cross-window-ablated modes."""

    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps(ToyGenerator(1, "test").sample(0).to_dict()) + "\n")
    output = tmp_path / "metrics.json"
    experiment = ToyExperiment()
    args = ["timed", "evaluate", "--checkpoint", "unused.pt", "--data", str(data),
            "--output", str(output)]
    with (
        patch("sys.argv", args),
        patch.object(ToyExperiment, "load", return_value=experiment),
        patch("project_kidra.timed.__main__.ToyEvaluator", autospec=True) as evaluator,
    ):
        evaluator.return_value.evaluate.return_value = {"exact": 0.5}
        ToyCLI.run()
    assert evaluator.return_value.evaluate.call_count == 2
    assert evaluator.return_value.evaluate.call_args.kwargs == {"cross_window": False}
    assert json.loads(output.read_text())["without_cross_window"] == {"exact": 0.5}
