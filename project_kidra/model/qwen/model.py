"""Load dense Qwen3 safetensors without changing the pretrained computation."""

from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from project_kidra.model.config import Config
from project_kidra.model.model import AgentModel
from project_kidra.model.qwen.exchange import QwenExchange
from project_kidra.model.qwen.lora import LoRALinear
from project_kidra.model.window import ContextWindow


class QwenAgent(AgentModel):
    """Frozen Qwen3 layers plus LoRA and causal multi-window exchange adapters."""

    def __init__(
        self,
        backbone: Any,
        n_windows: int = 4,
        n_local_ctx: int = 256,
        exchange_every: int = 4,
        lora_rank: int = 8,
        exchange_rank: int = 32,
        learning_rate: float = 2e-4,
        gradient_checkpointing: bool = True,
    ) -> None:
        """Wrap an injected Transformers backbone, preserving all original weights."""

        nn.Module.__init__(self)
        cfg = backbone.config
        if (
            cfg.model_type != "qwen3"
            or cfg.rope_scaling is not None
            or getattr(cfg, "use_sliding_window", False)
            or cfg.hidden_act != "silu"
            or cfg.attention_dropout != 0
        ):
            raise ValueError("only dense Qwen3 with standard RoPE and full attention is supported")
        if (
            lora_rank < 1
            or exchange_rank < 1
            or not 1 <= exchange_every <= cfg.num_hidden_layers
            or learning_rate <= 0
        ):
            raise ValueError("invalid adapter settings")
        if n_local_ctx > cfg.max_position_embeddings:
            raise ValueError("local context exceeds the pretrained context length")
        self.config = Config(
            vocab_size=cfg.vocab_size,
            d_model=cfg.hidden_size,
            d_hidden=cfg.intermediate_size,
            n_layers=cfg.num_hidden_layers,
            n_heads=cfg.num_attention_heads,
            n_windows=n_windows,
            n_local_ctx=n_local_ctx,
        )
        self.start_token = int(cfg.bos_token_id)
        self.free_token = -1
        self.windows: list[ContextWindow] = []
        self.available_window_ids = list(range(n_windows))
        self.backbone = backbone
        backbone.requires_grad_(False)
        self.options: dict[str, Any] = {
            "n_windows": n_windows,
            "n_local_ctx": n_local_ctx,
            "exchange_every": exchange_every,
            "lora_rank": lora_rank,
            "exchange_rank": exchange_rank,
            "learning_rate": learning_rate,
            "gradient_checkpointing": gradient_checkpointing,
        }
        self.exchange_every = exchange_every
        self.gradient_checkpointing = gradient_checkpointing
        device = backbone.model.embed_tokens.weight.device
        self.exchange = QwenExchange(
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            n_windows,
            exchange_rank,
            device,
        )
        if lora_rank:
            for layer in backbone.model.layers:
                for module, names in (
                    (layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                    (layer.mlp, ("gate_proj", "up_proj", "down_proj")),
                ):
                    for name in names:
                        setattr(module, name, LoRALinear(getattr(module, name), lora_rank))
        self.optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad], lr=learning_rate
        )

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        **options: Any,
    ) -> "QwenAgent":
        """Load local single-file or indexed safetensors; never fetch remote files."""

        path = Path(directory)
        if not path.is_dir():
            raise ValueError("pretrained path must be a local model directory")
        config = Qwen3Config.from_pretrained(path, local_files_only=True)
        if config.model_type != "qwen3":
            raise ValueError("expected a dense Qwen3 checkpoint")
        backbone, info = Qwen3ForCausalLM.from_pretrained(
            path,
            config=config,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=dtype,
            device_map={"": device},
            low_cpu_mem_usage=True,
            output_loading_info=True,
            attn_implementation="sdpa",
        )
        if info["missing_keys"] or info["unexpected_keys"] or info["mismatched_keys"]:
            raise ValueError(f"incompatible Qwen3 weights: {info}")
        return cls(backbone, **options)

    def open_window(self) -> ContextWindow:
        """Assign stable slot IDs in JSON window order."""

        if not self.available_window_ids:
            raise RuntimeError("No available window IDs to open a new window.")
        window = ContextWindow(
            len(self.windows),
            min(self.available_window_ids),
            self.config.n_layers,
            self.config.n_local_ctx,
            self.config.vocab_size,
            self.free_token,
        )
        self.available_window_ids.remove(window.id)
        self.windows.append(window)
        return window

    def train_chunk(
        self, max_tokens: int = 32, updates: dict[ContextWindow, list[int]] | None = None
    ) -> list[float]:
        """Optimize adapters on all real targets, with detached caches between steps."""

        from project_kidra.model.qwen.sequence import QwenSequence

        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if updates is None:
            updates = {
                w: list(range(min(max_tokens, len(w.pending)))) for w in self.windows if w.pending
            }
        batch = QwenSequence(self, updates)
        if not batch.windows:
            return []
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = batch.forward()
        targets = torch.tensor([t for ts in batch.targets for t in ts], device=logits.device)
        losses = F.cross_entropy(logits.float(), targets, reduction="none")
        losses.mean().backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_([p for p in self.parameters() if p.requires_grad], 1.0)
        self.optimizer.step()
        batch.commit()
        return cast(list[float], losses.detach().tolist())

    def forward(self) -> list[float]:
        """Train pending targets only; generation uses generate_step without optimization."""

        return self.train_chunk(max_tokens=1)

    @torch.no_grad()
    def generate_step(self, temperature: float = 0.0) -> list[int]:
        """Generate one token per open window without training or replacing pending targets."""

        from project_kidra.model.qwen.sequence import QwenSequence

        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        if any(w.pending for w in self.windows):
            raise ValueError("consume pending targets before generating")
        if not self.windows:
            return []
        self.eval()
        batch = QwenSequence(self, {w: [0] for w in self.windows}, generating=True)
        logits = batch.forward().float()
        tokens = (
            logits.argmax(-1)
            if temperature == 0
            else torch.multinomial((logits / temperature).softmax(-1), 1).squeeze(-1)
        )
        for window, token in zip(self.windows, tokens.tolist()):
            window.position += 1
            window.append(token)
            window.detach_cache()
        return cast(list[int], tokens.tolist())
