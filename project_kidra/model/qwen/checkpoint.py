"""Portable adapter checkpoints referencing an unchanged local Qwen base model."""

import os
from pathlib import Path
from typing import Any

import torch

from project_kidra.model.qwen.model import QwenAgent


class QwenCheckpoint:
    """Save adapters and optimizer state without duplicating billions of frozen weights."""

    @staticmethod
    def save(model: QwenAgent, path: str | Path, tokenizer_fingerprint: str) -> None:
        """Atomically save trainable tensors; windows and dataset cursor are runtime state."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        state: dict[str, Any] = {
            "format": "kidra-qwen3-adapters-v1",
            "qwen_config": model.backbone.config.to_dict(),
            "options": model.options,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "adapters": {
                name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad
            },
            "optimizer": model.optimizer.state_dict(),
        }
        try:
            torch.save(state, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def load(
        path: str | Path,
        pretrained: str | Path,
        tokenizer_fingerprint: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> QwenAgent:
        """Restore adapters and optimizer against a compatible local base checkpoint."""

        state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != "kidra-qwen3-adapters-v1":
            raise ValueError("not a Qwen3 adapter checkpoint")
        if state["tokenizer_fingerprint"] != tokenizer_fingerprint:
            raise ValueError("checkpoint tokenizer mismatch")
        model = QwenAgent.from_pretrained(
            pretrained, device=device, dtype=dtype, **state["options"]
        )
        actual = model.backbone.config.to_dict()
        for key in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "rope_theta",
            "rms_norm_eps",
            "tie_word_embeddings",
            "bos_token_id",
            "eos_token_id",
        ):
            if actual.get(key) != state["qwen_config"].get(key):
                raise ValueError(f"checkpoint base model mismatch: {key}")
        parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if parameters.keys() != state["adapters"].keys():
            raise ValueError("checkpoint adapter keys mismatch")
        with torch.no_grad():
            for name, value in state["adapters"].items():
                if parameters[name].shape != value.shape:
                    raise ValueError(f"checkpoint adapter shape mismatch: {name}")
                parameters[name].copy_(value)
        model.optimizer.load_state_dict(state["optimizer"])
        return model
