import argparse
from pathlib import Path

import torch

from project_kidra.moe import MoEConfig, MoEModel, Trainer
from project_kidra.moe.byte_corpus import ByteCorpus
from project_kidra.moe.continuous_trainer import ContinuousTrainer
from project_kidra.moe.trainer import TrainStepResult


def main() -> int:
    """Run continuous byte-level training from a corpus file."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("corpus", type=Path, help="UTF-8 text corpus to read continuously")
    parser.add_argument("--checkpoint", type=Path, default=Path("kidra-continuous.pt"))
    parser.add_argument("--resume", action="store_true", help="resume from --checkpoint")
    parser.add_argument("--steps", type=int, help="stop after this many steps (default: forever)")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--context-window", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = MoEConfig(vocab_size=256, context_window=args.context_window)
    model = MoEModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    trainer = Trainer(model, optimizer)
    with ByteCorpus(args.corpus, config.context_window, device) as corpus:
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
