import torch

from project_kidra.app import main


def test_main_returns_zero(capsys: object) -> None:
    assert main() == 0


def test_torch_available() -> None:
    assert torch.ones(3).sum().item() == 3.0
