"""Audit saved checkpoints, replicate decisive conditions, and test additional training budget."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from project_kidra.robustness.plan import RobustPlan
from project_kidra.robustness.scheduler import RobustScheduler
from project_kidra.robustness.summary import RobustSummary


class RobustCLI:
    """Keep the scheduler CPU-only; only worker subprocesses import PyTorch."""

    @staticmethod
    def run() -> int:
        """Dispatch explicit preparation, scheduling, reporting, or one worker."""

        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        prepare = commands.add_parser("prepare")
        prepare.add_argument("--root", type=Path, required=True)
        prepare.add_argument("--followup", type=Path, required=True)
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
        export = commands.add_parser("export")
        export.add_argument("--root", type=Path, required=True)
        export.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        if args.command == "export":
            from project_kidra.robustness.export import export_results
            print(json.dumps({"files": export_results(args.root, args.output), "output": str(args.output)}))
            return 0
        if args.command == "prepare":
            manifest = RobustPlan.prepare(args.followup, args.root)
            print(json.dumps({"experiments": len(manifest["runs"]), "audits": 3,
                              "training_jobs": 9, "additional_sample_updates": 450000, "root": str(args.root.resolve())}))
            return 0
        if args.command == "status":
            print(json.dumps(RobustSummary.write(args.root)))
            return 0
        if args.command == "run":
            if importlib.util.find_spec("torch") is None:
                raise RuntimeError(f"PyTorch is not installed for {sys.executable}; activate the training environment")
            result = RobustScheduler(args.root, args.slots, args.threads_per_job,
                                     args.retries, args.only).run()
            print(json.dumps(RobustSummary.write(args.root)))
            return result
        from project_kidra.robustness.worker import RobustWorker

        return RobustWorker(args.root, args.name, args.device, args.threads).run()


if __name__ == "__main__":
    raise SystemExit(RobustCLI.run())
