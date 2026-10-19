"""Add two temporal controls in a separate suite; wait for the original matrix to finish."""

import argparse
import json
from pathlib import Path

from project_kidra.temporal.suite.plan import TemporalPlan
from project_kidra.temporal.suite.scheduler import TemporalScheduler
from project_kidra.temporal.suite.summary import TemporalSummary


class TemporalCLI:
    """Keep the scheduler CPU-only; only worker subprocesses import PyTorch."""

    @staticmethod
    def run() -> int:
        """Dispatch explicit preparation, scheduling, reporting, or one worker."""

        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        prepare = commands.add_parser("prepare")
        prepare.add_argument("--root", type=Path, required=True)
        prepare.add_argument("--base", type=Path, required=True)
        prepare.add_argument("--seed", type=int)
        prepare.add_argument("--learning-rate", type=float, default=0.0001)
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
            manifest = TemporalPlan.prepare(args.base, args.root, args.seed, args.learning_rate)
            print(json.dumps({"experiments": len(manifest["runs"]),
                              "samples_per_run": manifest["options"]["samples"],
                              "baseline": manifest["base"]["baseline"],
                              "root": str(args.root.resolve())}))
            return 0
        if args.command == "status":
            print(json.dumps(TemporalSummary.write(args.root)))
            return 0
        if args.command == "run":
            result = TemporalScheduler(args.root, args.slots, args.threads_per_job,
                                     args.retries, args.only).run()
            print(json.dumps(TemporalSummary.write(args.root)))
            return result
        from project_kidra.temporal.suite.worker import TemporalWorker

        return TemporalWorker(args.root, args.name, args.device, args.threads).run()


if __name__ == "__main__":
    raise SystemExit(TemporalCLI.run())
