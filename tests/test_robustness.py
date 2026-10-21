"""Setup and recovery checks: no PyTorch, model training, or model checkpoint writes."""

import json
import random
import tarfile
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from project_kidra.followup.runtime.probes import Probes
from project_kidra.followup.suite.spec import FollowSpec
from project_kidra.robustness.export import export_results
from project_kidra.robustness.plan import RobustPlan
from project_kidra.robustness.scheduler import RobustScheduler
from project_kidra.robustness.spec import RobustSpec
from project_kidra.robustness.transforms import RobustProbes, augment, layout
from project_kidra.temporal.suite.plan import TemporalPlan
from project_kidra.timed.generator import ToyGenerator
from project_kidra.timed.matrix.storage import AtomicFiles
from project_kidra.timed.schedule import Schedule
from project_kidra.timed.tokenizer import ToyTokenizer


class RobustnessSetupTests(unittest.TestCase):
    def test_design_has_matched_controls(self):
        specs, sources = RobustPlan.definitions()
        self.assertEqual(len(specs), 12)
        self.assertEqual(len({s.name for s in specs}), 12)
        self.assertEqual(sum(s.budget - (50000 if s.source else 0)
                             for s in specs if s.action != "audit"), 450000)
        self.assertEqual({(s.seed, s.order_seed) for s in specs if s.action == "cross"},
                         {(42, 42), (42, 44), (44, 42), (44, 44)})
        for spec in specs:
            self.assertEqual(RobustSpec.from_dict(spec.to_dict()), spec)
            if spec.source:
                self.assertIn(spec.source, sources)
        repair = next(s for s in specs if s.action == "repair" and s.seed == 44)
        control = next(s for s in specs if s.action == "extend" and s.variant == "none")
        self.assertEqual(repair.source, control.source)
        self.assertEqual(repair.order_seed, control.order_seed)

    def test_invalid_design_rejected_and_identity_changes(self):
        spec = RobustPlan.definitions()[0][0]
        for bad in [replace(spec, action="cross"), replace(spec, augmentation="prefixmix"),
                    replace(spec, order_seed=-1), replace(spec, budget=200000)]:
            with self.assertRaises(ValueError):
                RobustSpec.from_dict(bad.to_dict())
        self.assertNotEqual(spec.identity("parent"), replace(spec, order_seed=42).identity("parent"))

    def test_layout_retains_targets_facts_and_gaps(self):
        tok = ToyTokenizer()
        gen = ToyGenerator(5, "test", delayed=True)
        for i in range(4):
            sample = gen.sample(i)
            original = Schedule(sample, tok, 5, 128)
            for count, placement in [(0, "prefix"), (24, "prefix"), (24, "suffix"),
                                     (24, "aftermarker"), (40, "prefix")]:
                changed = layout(sample, count, placement)
                self.assertEqual(changed.sample_id, sample.sample_id)
                current = Schedule(changed, tok, 5, 128)
                old_end = new_end = 0
                for index, (a, b) in enumerate(zip(original.groups, current.groups)):
                    self.assertEqual(min(s.source.offset_tick for s in a)-old_end,
                                     min(s.source.offset_tick for s in b)-new_end)
                    for left, right in zip(a, b):
                        l, r = left.source, right.source
                        self.assertEqual((l.window_id, l.predict, l.terminal),
                                         (r.window_id, r.predict, r.terminal))
                        self.assertEqual([w for w in l.text.split() if w != "FILL"],
                                         [w for w in r.text.split() if w != "FILL"])
                        if l.predict:
                            self.assertEqual(l.text, r.text)
                    old_end, new_end = sample.group_ends[index], changed.group_ends[index]
            self.assertEqual(layout(sample, 24), Probes.apply(sample, "prefix24"))

    def test_prefix_augmentation_replays_without_global_rng(self):
        sample = ToyGenerator(1).sample(0)
        before = random.getstate()
        seen = set()
        for step in range(40):
            a = augment(sample, "prefixmix", step)
            self.assertEqual(a, augment(sample, "prefixmix", step))
            seen.add(len(a.streams[2].text.split())-len(sample.streams[2].text.split()))
        self.assertEqual(seen, {0, 8, 16, 24})
        self.assertEqual(before, random.getstate())
        self.assertIs(augment(sample, "none", 0), sample)

    def test_all_probes_fit_heartbeat_context(self):
        gen = ToyGenerator(4, "test")
        for i in range(4):
            sample = gen.sample(i)
            for name in RobustProbes.names:
                result = RobustProbes.apply(sample, name)
                TemporalPlan.validate_records(json.dumps(result.to_dict()).encode())
            self.assertEqual(RobustProbes.apply(sample, "rebase_shift4096"), sample)

    def test_prepare_copies_parent_and_rejects_changed_inputs(self):
        specs, definitions = RobustPlan.definitions()
        with tempfile.TemporaryDirectory() as temporary:
            parent, root = Path(temporary)/"parent", Path(temporary)/"new"
            parent.mkdir()
            AtomicFiles.bytes(parent/"data/train.jsonl", b"fixture only\n")
            raw_specs = []
            for key, name in definitions.items():
                if name.startswith("@"):
                    path = parent/"sources/tick.pt"
                    AtomicFiles.bytes(path, b"immutable source fixture, not model weights")
                    continue
                source_spec = next(s for s in specs if s.source == key)
                action = "extend" if key.endswith("100k") else "replicate"
                raw_specs.append(FollowSpec("agent", source_spec.policy, 0.0001, source_spec.seed,
                                            action, source_spec.variant,
                                            "tick" if action == "extend" else "",
                                            100000 if action == "extend" else 50000).to_dict())
                directory = parent/"experiments"/name
                AtomicFiles.bytes(directory/"checkpoint.pt", b"immutable fixture, not model weights")
                AtomicFiles.json(directory/"complete.json", {"identity": name,
                                 "steps": 100000 if key.endswith("100k") else 50000})
            manifest = {"identity": "fixture", "runs": raw_specs, "options": {},
                        "sources": {"tick": {"path": "sources/tick.pt", "identity": "tick", "steps": 50000}},
                        "datasets": {"train": {"path": "data/train.jsonl", "count": 50000,
                                               "sha256": AtomicFiles.digest(parent/"data/train.jsonl")}}}
            before = {str(p.relative_to(parent)):p.read_bytes() for p in parent.rglob("*") if p.is_file()}
            with patch("project_kidra.robustness.plan.FollowPlan.load", return_value=manifest), \
                 patch("project_kidra.robustness.plan.FollowScheduler.completed", return_value=True):
                prepared = RobustPlan.prepare(parent, root)
                self.assertEqual(RobustPlan.prepare(parent, root), prepared)
                with self.assertRaises(ValueError):
                    RobustPlan.prepare(parent, parent/"nested")
            after = {str(p.relative_to(parent)):p.read_bytes() for p in parent.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            copied = root/prepared["sources"]["tick42"]["path"]
            copied.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                RobustPlan.load(root)

    def test_export_rejects_missing_records_and_excludes_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"suite"
            spec = RobustPlan.definitions()[0][0]
            manifest = {"runs": [spec.to_dict()]}
            manifest["identity"] = AtomicFiles.identity(manifest)
            AtomicFiles.json(root/"manifest.json", manifest)
            directory = root/"experiments"/spec.name
            row = b'{"sample_id":"fixture","scores":{"exact":1}}\n'
            path = directory/"records/iid-original.jsonl"
            AtomicFiles.bytes(path, row)
            report = {"identity": spec.identity(manifest["identity"]),
                      "sections": {"iid/original": {"count": 1, "sha256": AtomicFiles.digest(path)}}}
            AtomicFiles.json(directory/"results.json", report)
            AtomicFiles.json(directory/"complete.json", {"identity": report["identity"],
                             "results_sha256": AtomicFiles.digest(directory/"results.json")})
            AtomicFiles.bytes(root/"data/train.jsonl", b"excluded")
            output = Path(temporary)/"export.tar.gz"
            export_results(root, output)
            with tarfile.open(output) as archive:
                self.assertTrue(any(n.endswith("iid-original.jsonl") for n in archive.getnames()))
                self.assertFalse(any("data/" in n or n.endswith(".pt") for n in archive.getnames()))
            path.unlink()
            with self.assertRaisesRegex(ValueError, "missing per-example"):
                export_results(root, output)

    def test_scheduler_refills_and_skips_completed(self):
        specs = RobustPlan.definitions()[0][:3]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scheduler = RobustScheduler(root, ["cpu", "cpu"], poll_seconds=0.001)
            launched = []
            completed = {specs[0].name}

            class Process:
                def __init__(self, name):
                    self.name, self.polls = name, 0
                def poll(self):
                    self.polls += 1
                    if self.polls < 2:
                        return None
                    completed.add(self.name)
                    return 0

            def launch(spec, device):
                launched.append(spec.name)
                self.assertLess(len(scheduler.children), 2)
                return Process(spec.name)

            with patch("project_kidra.robustness.scheduler.RobustPlan.load",
                       return_value={"runs": [s.to_dict() for s in specs], "identity": "fixture"}), \
                 patch.object(scheduler, "completed", side_effect=lambda s, m: s.name in completed), \
                 patch.object(scheduler, "_launch", side_effect=launch):
                self.assertEqual(scheduler.run(), 0)
            self.assertEqual(set(launched), {s.name for s in specs[1:]})
            self.assertEqual(AtomicFiles.read(root/"scheduler.json")["running"], [])


if __name__ == "__main__":
    unittest.main()
