from pathlib import Path
from typing import Any

import torch

from project_kidra.model.config import Config
from project_kidra.model.model import AgentModel


class Checkpoint:
    """Saves and loads AgentModel instances, including optimizer state."""

    @staticmethod
    def save(model: AgentModel, path: str | Path) -> None:
        """Write the model's config, weights, and optimizer state to `path`."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        state: dict[str, Any] = {
            "config": model.config.model_parameters(),
            "free_token": model.free_token,
            "model": model.state_dict(),
            "trunk_optimizer": model.trunk_optimizer.state_dict(),
            "expert_optimizer": model.expert_optimizer.state_dict(),
        }
        torch.save(state, target)

    @staticmethod
    def load(path: str | Path, device: str | torch.device = "cpu") -> AgentModel:
        """Rebuild an AgentModel from a checkpoint written by `save`.

        Context windows are runtime state and are not restored.
        """

        state: dict[str, Any] = torch.load(Path(path), map_location=device, weights_only=True)

        model = AgentModel(Config(**state["config"]), state["free_token"])
        model.load_state_dict(state["model"])
        model.trunk_optimizer.load_state_dict(state["trunk_optimizer"])
        model.expert_optimizer.load_state_dict(state["expert_optimizer"])
        model.to(device)

        return model
