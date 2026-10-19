"""Generate JSONL, train the toy model, or evaluate a saved checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from project_kidra.timed.evaluator import ToyEvaluator
from project_kidra.timed.experiment import ToyExperiment
from project_kidra.timed.generator import ToyGenerator


class ToyCLI:
    """Expose reproducible commands without downloading models or datasets."""

    @staticmethod
    def run() -> None:
        """Parse one operation and write its datasets or experiment artifacts."""

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--threads", type=int, default=1)
        commands = parser.add_subparsers(dest="command", required=True)
        generate = commands.add_parser("generate")
        generate.add_argument("--output", type=Path, required=True)
        generate.add_argument("--count", type=int, default=2000)
        generate.add_argument("--seed", type=int, default=1)
        generate.add_argument("--split", choices=["train", "test"], default="train")
        generate.add_argument("--delayed", action="store_true")
        train = commands.add_parser("train")
        train.add_argument("--data", type=Path, required=True)
        train.add_argument("--eval-data", type=Path, required=True)
        train.add_argument("--output", type=Path, required=True)
        train.add_argument("--steps", type=int, default=2000)
        train.add_argument("--eval-every", type=int, default=250)
        train.add_argument("--seed", type=int, default=42)
        train.add_argument("--device", default="cpu")
        train.add_argument("--resume", type=Path)
        evaluate = commands.add_parser("evaluate")
        evaluate.add_argument("--checkpoint", type=Path, required=True)
        evaluate.add_argument("--data", type=Path, required=True)
        evaluate.add_argument("--output", type=Path, required=True)
        evaluate.add_argument("--device", default="cpu")
        args = parser.parse_args()
        if args.threads < 1:
            parser.error("threads must be positive")
        torch.set_num_threads(args.threads)
        if args.command == "generate":
            if args.count < 1:
                parser.error("count must be positive")
            generator = ToyGenerator(args.seed, args.split, args.delayed)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w") as target:
                for i in range(args.count):
                    target.write(json.dumps(generator.sample(i).to_dict()) + "\n")
        elif args.command == "train":
            experiment = (
                ToyExperiment.load(args.resume, args.device)
                if args.resume else ToyExperiment(args.seed, args.device)
            )
            experiment.train(
                experiment.read(args.data), experiment.read(args.eval_data),
                args.steps, args.output, args.eval_every,
            )
        else:
            experiment = ToyExperiment.load(args.checkpoint, args.device)
            evaluator = ToyEvaluator(experiment.model, experiment.tokenizer)
            samples = experiment.read(args.data)
            metrics = {
                "samples": len(samples), "normal": evaluator.evaluate(samples),
                "without_cross_window": evaluator.evaluate(samples, cross_window=False),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    ToyCLI.run()
