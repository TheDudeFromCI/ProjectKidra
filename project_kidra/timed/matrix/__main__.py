"""Prepare a diagnostic matrix, run it concurrently, or summarize saved results."""

import argparse
import json
from pathlib import Path

from project_kidra.timed.matrix.plan import MatrixPlan
from project_kidra.timed.matrix.scheduler import MatrixScheduler
from project_kidra.timed.matrix.summary import MatrixSummary


class MatrixCLI:
    """Keep the scheduler CPU-only; only worker subprocesses import PyTorch."""

    @staticmethod
    def run() -> int:
        """Dispatch explicit preparation, scheduling, reporting, or one worker."""

        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        prepare = commands.add_parser("prepare")
        prepare.add_argument("--root", type=Path, required=True)
        prepare.add_argument("--samples", type=int, default=50000)
        prepare.add_argument("--eval-samples", type=int, default=2000)
        prepare.add_argument("--seeds", type=int, nargs="+", default=[42])
        prepare.add_argument("--train-data", type=Path)
        prepare.add_argument("--checkpoint-every", type=int, default=250)
        prepare.add_argument("--eval-every", type=int, default=2000)
        prepare.add_argument("--periodic-samples", type=int, default=256)
        run = commands.add_parser("run")
        run.add_argument("--root", type=Path, required=True)
        run.add_argument("--slots", nargs="+", default=["cuda:0", "cuda:0"])
        run.add_argument("--threads-per-job", type=int, default=2)
        run.add_argument("--retries", type=int, default=1)
        run.add_argument("--only", default="*")
        report = commands.add_parser("status")
        report.add_argument("--root", type=Path, required=True)
        worker = commands.add_parser("worker", help="execute one run (used by the scheduler)")
        worker.add_argument("--root", type=Path, required=True)
        worker.add_argument("--name", required=True)
        worker.add_argument("--device", default="cpu")
        worker.add_argument("--threads", type=int, default=2)
        args = parser.parse_args()
        if args.command == "prepare":
            manifest = MatrixPlan.prepare(
                args.root, args.samples, args.eval_samples, args.seeds, args.train_data,
                args.checkpoint_every, args.eval_every, args.periodic_samples,
            )
            print(json.dumps({"experiments": len(manifest["runs"]), "samples_per_run": args.samples,
                              "root": str(args.root.resolve())}))
            return 0
        if args.command == "status":
            print(json.dumps(MatrixSummary.write(args.root)))
            return 0
        if args.command == "run":
            result = MatrixScheduler(args.root, args.slots, args.threads_per_job,
                                     args.retries, args.only).run()
            print(json.dumps(MatrixSummary.write(args.root)))
            return result
        from project_kidra.timed.matrix.worker import MatrixWorker

        return MatrixWorker(args.root, args.name, args.device, args.threads).run()


if __name__ == "__main__":
    raise SystemExit(MatrixCLI.run())
