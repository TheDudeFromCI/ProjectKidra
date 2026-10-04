"""Pytorch model classes for the ProjectKidra agent."""

from project_kidra.config import Config

class AgentModel:
    """The primary model representing an agent."""


    def __init__(self, config: Config):
        self.config = config