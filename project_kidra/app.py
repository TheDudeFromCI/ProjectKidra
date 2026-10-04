import torch


def main() -> int:
    x: torch.Tensor = torch.zeros(2, 2)
    print(f"Project Kidra ready. torch {torch.__version__}, sample tensor shape {tuple(x.shape)}")
    return 0
