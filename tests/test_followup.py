"""Bounded read-only probes and mocked lifecycle tests; no training experiment is run."""

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from test_temporal_runtime import sample
from test_timed_execution import tiny_model

from project_kidra.followup.runtime.bounded import BoundedProtocol
from project_kidra.followup.runtime.evaluate import FollowEvaluation
from project_kidra.followup.runtime.probes import Probes
from project_kidra.followup.suite.plan import FollowPlan
from project_kidra.followup.suite.spec import FollowSpec
from project_kidra.temporal.runtime.group import TemporalGroup
from project_kidra.temporal.runtime.session import TemporalSession
from project_kidra.temporal.suite.plan import TemporalPlan
from project_kidra.timed.execution.policy import TrainingPolicy
from project_kidra.timed.generator import ToyGenerator
from project_kidra.timed.matrix.plan import MatrixPlan
from project_kidra.timed.matrix.spec import RunSpec
from project_kidra.timed.matrix.storage import AtomicFiles
from project_kidra.timed.schedule import Schedule
from project_kidra.timed.tokenizer import ToyTokenizer


def test_probes_preserve_content_or_recompute_dependencies() -> None:
    """Layout/timing changes keep labels, while routing changes propagate through the tool lookup."""

    tok = ToyTokenizer()
    generator = ToyGenerator(2)
    for index in range(12):
        original = generator.sample(index)
        for name in Probes.names:
            transformed = Probes.apply(original, name)
            schedule = Schedule(transformed, tok, 5, 128)
            if name != "route_change":
                assert [(s.window_id, s.text, s.terminal) for s in original.streams if s.predict] == [
                    (s.window_id, s.text, s.terminal) for s in transformed.streams if s.predict]
            else:
                old_key = original.streams[4].text.split()[-1]
                new_key = transformed.streams[4].text.split()[-1]
                assert old_key != new_key
                result = original.streams[5].text.split()[1:]
                expected = dict(zip(result[::2], result[1::2]))[new_key]
                assert transformed.streams[-1].text == "ANSWER " + expected
                assert transformed.streams[-2].text == "FOUND " + expected
            events = {t for group in schedule.groups for stream in group
                      for t in range(stream.source.offset_tick, stream.source.offset_tick + len(stream.tokens))}
            assert len(events) <= 128
            if name == "prefix24":
                assert transformed.streams[2].text.startswith("FILL " * 24)
            if name == "shift4096":
                assert transformed.group_ends == tuple(t + 4096 for t in original.group_ends)


@torch.no_grad()
def test_plain_model_is_invariant_to_absolute_shift_and_empty_gaps() -> None:
    """Those changes carry no new information in the baseline, so they must not change logits."""

    tok = ToyTokenizer()
    initial = tiny_model()
    outputs = []
    for name in ("original", "shift4096", "gaps64"):
        model = copy.deepcopy(initial)
        session = TemporalSession(model, tok, "none")
        try:
            rows = [TemporalGroup.forward(session, group)[0] for group in
                    Schedule(Probes.apply(sample(), name), tok, 5, 128).groups]
            outputs.append(torch.cat(rows))
        finally:
            session.close()
    for output in outputs[1:]:
        torch.testing.assert_close(outputs[0], output, atol=0, rtol=0)


def test_bounded_generation_ignores_gold_lengths_and_waits_for_control() -> None:
    """A one-token gold wait cannot cut off a longer generated response or force termination."""

    model, tok = tiny_model(), ToyTokenizer()
    evaluator = BoundedProtocol(model, tok, "none", max_tokens=6)
    session = TemporalSession(model, tok, "none")
    group = Schedule(sample(), tok, 5, 128).groups[1][:1]
    assert len(group[0].tokens) == 1
    sequence = iter(tok.encode("ANSWER v1 <|end|>"))

    def predict(_session: TemporalSession, windows: list[Any], tick: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Feed a controlled continuation, without looking at reference tokens."""

        logits = torch.full((len(windows), len(tok.words)), -100.0)
        logits[:, next(sequence)] = 1.0
        return logits, torch.tensor(0.0)

    try:
        with patch.object(TemporalGroup, "advance", predict):
            generated, end, overflow = evaluator.generate(session, group)
        assert generated == [tok.encode("ANSWER v1 <|end|>")]
        assert end == group[0].source.offset_tick + 3 and not overflow
    finally:
        session.close()


def test_bounded_cap_handles_no_control_and_releases_windows() -> None:
    """A pathological model reaches the uniform cap without looping indefinitely."""

    model, tok = tiny_model(), ToyTokenizer()
    evaluator = BoundedProtocol(model, tok, "none", max_tokens=6)
    session = TemporalSession(model, tok, "none")
    group = Schedule(sample(), tok, 5, 128).groups[1][:1]

    def predict(_session: TemporalSession, windows: list[Any], tick: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Always emit a content token."""

        logits = torch.full((len(windows), len(tok.words)), -100.0)
        logits[:, tok.ids["FILL"]] = 1.0
        return logits, torch.tensor(0.0)

    try:
        with patch.object(TemporalGroup, "advance", predict):
            generated, end, _ = evaluator.generate(session, group)
        assert len(generated[0]) == 6 and end == group[0].source.offset_tick + 6
    finally:
        session.close()
    assert not model.windows


def test_followup_matrix_freezes_original_inputs_without_retraining(tmp_path: Path) -> None:
    """Create a tiny manifest fixture claiming the production budget; never construct a worker."""

    base, temporal, root = (tmp_path / s for s in ("base", "temporal", "followup"))
    parent = MatrixPlan.prepare(base, samples=1, eval_samples=1)
    extra = TemporalPlan.prepare(base, temporal)
    parent["options"]["samples"] = 50000
    parent["identity"] = AtomicFiles.identity({k: v for k, v in parent.items() if k != "identity"})
    AtomicFiles.json(base / "manifest.json", parent)
    extra["base"]["identity"] = parent["identity"]
    extra["identity"] = AtomicFiles.identity({k: v for k, v in extra.items() if k != "identity"})
    AtomicFiles.json(temporal / "manifest.json", extra)
    from project_kidra.temporal.suite.spec import TemporalSpec

    for path, manifest, cls in ((base, parent, RunSpec), (temporal, extra, TemporalSpec)):
        for raw in manifest["runs"]:
            spec = cls.from_dict(raw)
            directory = path / "experiments" / spec.name
            AtomicFiles.json(directory / "results.json", {"metrics": {}})
            AtomicFiles.json(directory / "complete.json", {
                "identity": spec.identity(manifest["identity"]),
                "results_sha256": AtomicFiles.digest(directory / "results.json"),
            })
            AtomicFiles.bytes(directory / "checkpoint.pt", b"fixture; never deserialized")
    before = {str(p): AtomicFiles.digest(p) for path in (base, temporal) for p in path.rglob("*") if p.is_file()}
    manifest = FollowPlan.prepare(base, temporal, root)
    runs = [FollowSpec.from_dict(raw) for raw in manifest["runs"]]
    assert len(runs) == 16
    assert sum(s.action == "audit" for s in runs) == 5
    assert sum(s.action == "replicate" for s in runs) == 8
    assert sum(s.action == "extend" for s in runs) == 3
    assert {s.seed for s in runs if s.action == "replicate"} == {43, 44}
    assert FollowPlan.prepare(base, temporal, root) == manifest
    assert FollowPlan.load(root) == manifest
    assert before == {str(p): AtomicFiles.digest(p) for path in (base, temporal) for p in path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="separate"):
        FollowPlan.prepare(base, temporal, base / "bad")
    (root / "sources" / "tick.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint changed"):
        FollowPlan.load(root)


def test_per_record_evaluation_is_paired_and_resumes_completed_sections(tmp_path: Path) -> None:
    """Mock inference; verify section commits and interrupted-report reuse without a training step."""

    evaluation = FollowEvaluation(tiny_model(), ToyTokenizer(), "none")
    data = {name: [ToyGenerator(2).sample(0)] for name in ("iid", "ood", "delayed")}
    spec = FollowSpec("agent", TrainingPolicy(execution="tick"), 0.0001, 43).to_dict()
    with (patch.object(evaluation.standard, "rollout", return_value={"exact": 1.0, "answer_fact": 1.0}) as standard,
          patch.object(evaluation.bounded, "rollout", return_value={"scores": {"exact": 1.0}, "trace": []}) as bounded):
        assert evaluation.run(tmp_path, "fixture", spec, data, lambda: False)
        assert standard.call_count == 18 and bounded.call_count == 2
        report = AtomicFiles.read(tmp_path / "results.json")
        assert report["metrics"]["iid/route_change"]["both_exact"] == 1.0
        assert report["sections"]["iid/bounded16"]["count"] == 1
        assert evaluation.run(tmp_path, "fixture", spec, data, lambda: False)
        assert standard.call_count == 18 and bounded.call_count == 2
    rows = [json.loads(line) for line in (tmp_path / "records" / "iid-original.jsonl").read_text().splitlines()]
    assert rows[0]["sample_id"] == data["iid"][0].sample_id


def test_tool_delivery_uses_generated_syntax_not_correct_reference_key() -> None:
    """A syntactically valid wrong call receives the tool table; later correctness stays separately scored."""

    model, tok = tiny_model(), ToyTokenizer()
    evaluator = BoundedProtocol(model, tok, "none")
    item = ToyGenerator(3).sample(0)
    delivered: list[int] = []

    def observe(session: TemporalSession, group: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Record which external observations the environment actually releases."""

        delivered.extend(s.source.window_id for s in group)
        return torch.zeros((1, len(tok.words))), torch.tensor(0.0)

    def generate(session: TemporalSession, group: list[Any]) -> tuple[list[list[int]], int, bool]:
        """Emit a different but valid key, with other streams unchanged for metric isolation."""

        rows = []
        for stream in group:
            tokens = list(stream.tokens)
            if stream.source.text.startswith("TOOL CALL"):
                old = int(tok.words[tokens[2]][1:])
                tokens[2] = tok.ids[f"k{(old + 1) % 8}"]
            rows.append(tokens)
        return rows, max(s.source.offset_tick + len(t) for s, t in zip(group, rows)), False

    with patch.object(TemporalGroup, "forward", observe), patch.object(evaluator, "generate", generate):
        report = evaluator.rollout(item)
    assert 4 in delivered
    assert report["scores"]["valid_call"] == 1.0
    assert report["scores"]["call_exact"] == 0.0
    assert report["scores"]["exact"] == 0.0


@pytest.mark.parametrize("action,cursor", [("replicate", 49999), ("extend", 99999)])
def test_worker_resumes_cursor_and_epoch_without_training(tmp_path: Path, action: str, cursor: int) -> None:
    """Mock the update and checkpoint store; exercise the new worker's epoch and completion logic."""

    import random
    from dataclasses import asdict

    from project_kidra.followup.suite.worker import FollowWorker
    from project_kidra.timed.trainer import TrainResult

    model, tok = tiny_model(), ToyTokenizer()
    spec = FollowSpec("agent", TrainingPolicy(execution="tick"), 0.0001, 43, action,
                      "none", "tick" if action == "extend" else "", 100000 if action == "extend" else 50000)
    manifest: dict[str, Any] = {"identity": "fixture", "options": {"checkpoint_every": 250, "eval_every": 2000,
                "periodic_samples": 1}, "runs": [spec.to_dict()],
                "datasets": {name: {"path": name} for name in ("train", "iid", "ood", "delayed")}}
    identity = spec.identity(manifest["identity"])
    state = {"identity": identity, "config": asdict(model.config), "steps": cursor, "history": [],
             "model": model.state_dict(), "trunk_optimizer": model.trunk_optimizer.state_dict(),
             "expert_optimizer": model.expert_optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
             "python_rng": random.getstate(), "cuda_rng": None, "loss_sum": 0.0, "loss_count": 0,
             "elapsed": 0.0}
    records = [sample()] * 50000
    with (
        patch.object(FollowPlan, "load", return_value=manifest),
        patch("project_kidra.followup.suite.worker.ToyExperiment", autospec=True) as experiment,
        patch("project_kidra.followup.suite.worker.CheckpointStore", autospec=True) as store,
        patch("project_kidra.followup.suite.worker.TimedTrainer", autospec=True) as trainer,
        patch("project_kidra.followup.suite.worker.TemporalDiagnostics", autospec=True) as diagnostics,
        patch("project_kidra.followup.suite.worker.FollowEvaluation", autospec=True) as evaluation,
    ):
        experiment.return_value.model, experiment.return_value.tokenizer = model, tok
        experiment.return_value.read.side_effect = lambda p: records if p.name == "train" else [sample()]
        store.return_value.load.return_value = state
        trainer.return_value.step.return_value = TrainResult(1.0, 1, 1)
        diagnostics.return_value.evaluate.return_value = {"exact": 0.0}

        def finish(directory: Path, run_id: str, raw: dict[str, Any], data: dict[str, Any], stop: Any) -> bool:
            """Emit an evaluation fixture rather than running model inference."""

            AtomicFiles.json(directory / "results.json", {"identity": run_id, "metrics": {}})
            return True

        evaluation.return_value.run.side_effect = finish
        worker = FollowWorker(tmp_path, spec.name, "cpu", 1)
        assert worker.run() == 0
        assert trainer.return_value.step.call_count == 1
        assert store.return_value.save.call_args.args[0]["steps"] == cursor + 1
        assert worker.run() == 0
        assert trainer.return_value.step.call_count == 1


def test_followup_scheduler_dispatch_and_final_status(tmp_path: Path) -> None:
    """Two harmless subprocesses verify concurrency, final status, and completed-run skipping."""

    import subprocess
    import sys

    from project_kidra.followup.suite.scheduler import FollowScheduler

    specs = [FollowSpec("agent", TrainingPolicy(execution="tick"), 0.0001, seed) for seed in (43, 44)]
    manifest: dict[str, Any] = {"identity": "fixture", "runs": [s.to_dict() for s in specs]}
    scheduler = FollowScheduler(tmp_path, ["cpu", "cpu"], poll_seconds=0.01)
    peak = 0
    script = '''
import sys,time
from pathlib import Path
from project_kidra.timed.matrix.storage import AtomicFiles
p=Path(sys.argv[1]);time.sleep(0.03)
AtomicFiles.json(p/'results.json', {'metrics': {}})
AtomicFiles.json(p/'complete.json', {'identity': sys.argv[2],
    'results_sha256': AtomicFiles.digest(p/'results.json')})
'''

    def launch(spec: FollowSpec, device: str) -> subprocess.Popen[bytes]:
        """Write only a temporary completion fixture, never invoke a training worker."""

        nonlocal peak
        peak = max(peak, len(scheduler.children) + 1)
        return subprocess.Popen([sys.executable, "-c", script, str(tmp_path / "experiments" / spec.name),
                                 spec.identity(manifest["identity"])])

    with patch.object(FollowPlan, "load", return_value=manifest), patch.object(scheduler, "_launch", launch):
        assert scheduler.run() == 0
    assert peak == 2
    status = AtomicFiles.read(tmp_path / "scheduler.json")
    assert status["completed"] == 2 and status["running"] == [] and status["pending"] == 0
    with (patch.object(FollowPlan, "load", return_value=manifest),
          patch.object(scheduler, "_launch", side_effect=AssertionError("completed run launched"))):
        assert scheduler.run() == 0
