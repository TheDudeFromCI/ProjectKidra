import argparse
import time
from pathlib import Path

import torch

from project_kidra.corpus.file import CorpusFile
from project_kidra.corpus.tokenizer import Tokenizer
from project_kidra.model.checkpoint import Checkpoint
from project_kidra.model.config import Config
from project_kidra.model.model import AgentModel
from project_kidra.model.trainer import Trainer


def validate_torch() -> bool:
    """Check if PyTorch CUDA is available and functional."""

    try:
        x = torch.zeros(1).to("cuda")
        return x.device.type == "cuda"
    except (AssertionError, RuntimeError) as error:
        print(f"PyTorch validation failed: {error}")
        return False


def main() -> int:
    """Main entry point for the application."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("corpus", type=Path, help="raw text file or prebuilt .npy corpus")
    parser.add_argument("--checkpoint", type=Path, default=Path("kidra.pt"))
    parser.add_argument("--resume", action="store_true", help="resume from --checkpoint")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument(
        "--preview-tokens",
        type=int,
        default=50,
        help="tokens to sample at each checkpoint (0 disables)",
    )
    parser.add_argument("--preview-temperature", type=float, default=1.0)
    args = parser.parse_args()

    if not validate_torch():
        print("PyTorch CUDA validation failed.")
        return 1
    print("PyTorch CUDA validation succeeded.")
    device = torch.device("cuda:1")

    tokenizer = Tokenizer()
    print(f"Tokenizer loaded: {tokenizer.vocab_size:,} tokens")

    config = Config(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        d_hidden=1536,
        n_experts=96,
        n_layers=12,
        n_heads=8,
        n_windows=100,
        n_local_ctx=1024,
        n_global_layers=4,
        expert_top_k=8,
        aux_loss_weight=0.01,
        trunk_lr=0.0001,
        expert_lr=0.001,
    )

    print(f"Configuration loaded: {config}")

    corpus = CorpusFile(args.corpus)
    print(f"Corpus loaded: {corpus.samples:,} samples available")

    if args.resume:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        model = Checkpoint.load(args.checkpoint, device=device)
        print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    else:
        print("Initializing new model.")
        model = AgentModel(config, tokenizer.free_token).to(device)
        print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")

    start_time = time.time()

    print("Starting training loop...")
    trainer = Trainer(model, corpus, 100)

    loops = 0
    while True:
        losses = trainer.step()
        loops += 1
        avg_loss = sum(losses) / len(losses)

        elapsed_time = time.time() - start_time
        start_time = time.time()
        toks = trainer.n_windows / elapsed_time

        print(
            f"Loops: {loops}, Elapsed Time: {elapsed_time:.2f}s, Tokens/sec: {toks:.2f}, Avg Loss: {avg_loss / loops:.4f}"
        )

        if loops % args.checkpoint_interval == 0:
            Checkpoint.save(model, args.checkpoint)

    return 0
