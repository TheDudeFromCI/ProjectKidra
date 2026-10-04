import argparse
from pathlib import Path

import torch

from project_kidra.moe import MoEConfig, MoEModel, Trainer
from project_kidra.moe.token_corpus import TokenCorpus
from project_kidra.moe.continuous_trainer import ContinuousTrainer
from project_kidra.moe.trainer import TrainStepResult
from project_kidra.tokenizer import Tokenizer


def main() -> int:
    """Run continuous token-level training from a corpus file."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("corpus", type=Path, help="raw text file or prebuilt .npy corpus")
    parser.add_argument("--checkpoint", type=Path, default=Path("kidra-continuous.pt"))
    parser.add_argument("--resume", action="store_true", help="resume from --checkpoint")
    parser.add_argument("--steps", type=int, help="stop after this many steps (default: forever)")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--context-window", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = Tokenizer()
    config = MoEConfig(vocab_size=tokenizer.vocab_size(), context_window=args.context_window)
    model = MoEModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    trainer = Trainer(model, optimizer)
    corpus = TokenCorpus.open(args.corpus, config.context_window, tokenizer, device)
    continuous = ContinuousTrainer(trainer, corpus)
    if args.resume:
        continuous.load_checkpoint(args.checkpoint)

    def report(step: int, result: TrainStepResult) -> None:
        """Print periodic progress without retaining step results."""

        if step == 1 or step % 100 == 0:
            print(f"step {step}: loss={result.loss:.4f}", flush=True)

    continuous.train(
        max_steps=args.steps,
        checkpoint_path=args.checkpoint,
        checkpoint_interval=args.checkpoint_interval,
        on_step=report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
