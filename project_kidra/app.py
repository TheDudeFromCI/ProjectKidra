"""Train scratch Kidra models or local Qwen3 adapters on grouped JSON corpora."""

import argparse
import time
from pathlib import Path

import torch

from project_kidra.corpus.file import CorpusFile
from project_kidra.corpus.tokenizer import Tokenizer
from project_kidra.model.checkpoint import Checkpoint
from project_kidra.model.config import Config
from project_kidra.model.grouped_trainer import GroupedTrainer
from project_kidra.model.model import AgentModel


def main() -> int:
    """Run grouped corpus training on one explicitly selected device."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("corpus", type=Path, help="prebuilt JSON/JSONL corpus (.kidra)")
    parser.add_argument("--pretrained", type=Path, help="local Qwen3 model directory")
    parser.add_argument("--checkpoint", type=Path, default=Path("kidra.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=0, help="optimizer steps; 0 runs indefinitely")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--tokens-per-step", type=int, default=32)
    parser.add_argument("--windows", type=int, default=4, help="maximum windows in one sample")
    parser.add_argument("--local-context", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--exchange-rank", type=int, default=32)
    parser.add_argument("--exchange-every", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    if args.steps < 0 or args.checkpoint_interval < 1 or args.tokens_per_step < 1:
        parser.error(
            "steps must be nonnegative; checkpoint interval and chunk size must be positive"
        )
    tokenizer = Tokenizer(args.pretrained)
    corpus = CorpusFile(args.corpus, preload=4)
    corpus.validate_tokenizer(tokenizer)
    model: AgentModel
    if args.pretrained:
        from project_kidra.model.qwen.checkpoint import QwenCheckpoint
        from project_kidra.model.qwen.model import QwenAgent

        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
        if args.resume:
            model = QwenCheckpoint.load(
                args.checkpoint,
                args.pretrained,
                tokenizer.fingerprint,
                device=args.device,
                dtype=dtype,
            )
        else:
            model = QwenAgent.from_pretrained(
                args.pretrained,
                device=args.device,
                dtype=dtype,
                n_windows=args.windows,
                n_local_ctx=args.local_context,
                lora_rank=args.lora_rank,
                exchange_rank=args.exchange_rank,
                exchange_every=args.exchange_every,
                learning_rate=args.learning_rate,
                gradient_checkpointing=not args.no_gradient_checkpointing,
            )
    elif args.resume:
        model = Checkpoint.load(args.checkpoint, device=args.device)
        if (
            model.config.vocab_size != tokenizer.vocab_size
            or model.free_token != tokenizer.free_token
        ):
            raise ValueError("scratch checkpoint tokenizer mismatch")
    else:
        config = Config(
            vocab_size=tokenizer.vocab_size, n_windows=args.windows, n_local_ctx=args.local_context
        )
        model = AgentModel(config, tokenizer.free_token).to(args.device)
    if corpus.metadata.max_windows > model.config.n_windows:
        raise ValueError(f"corpus needs {corpus.metadata.max_windows} windows; increase --windows")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total:,} parameters, {trainable:,} trainable, device={args.device}")
    print(f"Corpus: {corpus.samples:,} grouped samples")
    trainer = GroupedTrainer(model, corpus, tokens_per_step=args.tokens_per_step)

    def save() -> None:
        """Persist the selected backend's weights and optimizer state."""

        if args.pretrained:
            assert isinstance(model, QwenAgent)
            QwenCheckpoint.save(model, args.checkpoint, tokenizer.fingerprint)
        else:
            Checkpoint.save(model, args.checkpoint)

    step = 0
    try:
        while not args.steps or step < args.steps:
            started = time.monotonic()
            losses = trainer.step()
            if not losses:
                break
            step += 1
            elapsed = time.monotonic() - started
            print(
                f"Step: {step}, Tokens/sec: {len(losses) / elapsed:.2f}, "
                f"Loss: {sum(losses) / len(losses):.4f}",
                flush=True,
            )
            if step % args.checkpoint_interval == 0:
                save()
    except KeyboardInterrupt:
        print("Training interrupted; saving adapters/checkpoint.")
    if step:
        save()
    return 0
