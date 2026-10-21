"""Audit saved checkpoints, replicate decisive conditions, and test additional training budget."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from project_kidra.followup.suite.plan import FollowPlan
from project_kidra.followup.suite.scheduler import FollowScheduler
from project_kidra.followup.suite.summary import FollowSummary


class FollowCLI:
    """Keep the scheduler CPU-only; only worker subprocesses import PyTorch."""

    @staticmethod
    def run() -> int:
        """Dispatch explicit preparation, scheduling, reporting, or one worker."""

        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        prepare = commands.add_parser("prepare")
        prepare.add_argument("--root", type=Path, required=True)
        prepare.add_argument("--base", type=Path, required=True)
        prepare.add_argument("--temporal", type=Path, required=True)
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
            manifest = FollowPlan.prepare(args.base, args.temporal, args.root)
            print(json.dumps({"experiments": len(manifest["runs"]), "audits": 5,
                              "replications": 8, "continuations": 3, "root": str(args.root.resolve())}))
            return 0
        if args.command == "status":
            print(json.dumps(FollowSummary.write(args.root)))
            return 0
        if args.command == "run":
            if importlib.util.find_spec("torch") is None:
                raise RuntimeError(f"PyTorch is not installed for {sys.executable}; activate the training environment")
            result = FollowScheduler(args.root, args.slots, args.threads_per_job,
                                     args.retries, args.only).run()
            print(json.dumps(FollowSummary.write(args.root)))
            return result
        from project_kidra.followup.suite.worker import FollowWorker

        return FollowWorker(args.root, args.name, args.device, args.threads).run()


if __name__ == "__main__":
    raise SystemExit(FollowCLI.run())
