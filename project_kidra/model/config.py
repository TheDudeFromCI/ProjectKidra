from dataclasses import dataclass
from project_kidra.moe import MoEConfig
from project_kidra.corpus import Tokenizer
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Configuration settings for the Agent."""

    debug: bool = False
    database_url: str = "sqlite:///:memory:"
    model_path: str = "models/agent_model.pt"
    moe_config: MoEConfig = MoEConfig()
    tokenizer: Tokenizer = Tokenizer.default_vocab()
