import torch
from project_kidra.config import Config
from project_kidra.model import AgentModel


def validate_torch() -> bool:
    """Check if PyTorch CUDA is available and functional."""

    try:
        x = torch.zeros(1).to("cuda")
        return x.device.type == "cuda"
    except Exception as e:
        print(f"PyTorch validation failed: {e}")
        return False


def main() -> int:
    """Main entry point for the application."""

    if not validate_torch():
        print("PyTorch CUDA validation failed.")
        return 1
    print("PyTorch CUDA validation succeeded.")

    config = Config()
    config.debug = True
    config.database_url = "sqlite:///project_kidra.db"

    model = AgentModel(config)

    return 0
