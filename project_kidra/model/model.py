from project_kidra.model.config import Config
from project_kidra.moe import MoEModel


class AgentModel:
    """The primary model representing an agent."""

    def __init__(self, config: Config):
        self.config = config
        self.moe = MoEModel(config.moe_config)
