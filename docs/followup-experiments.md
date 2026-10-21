# Follow-up after the completed 68 + 2 experiments

This patch adds `project_kidra/followup/`, `tests/test_followup.py`, and this guide.
It changes no original model, timed-matrix, or temporal source files. Original
manifests, checkpoints, and results are read-only inputs. Both older source
fingerprints remain valid. It requires both previously installed experiment patches.

## Prepare and run

Activate the same Python environment used for training. The returned result archives
do not contain checkpoints or data; these commands use their existing local copies.
From the patched repository:

```bash
git apply --check /path/to/followup-update.patch
git apply /path/to/followup-update.patch

python -m project_kidra.followup prepare \
  --base runs/matrix --temporal runs/temporal --root runs/followup

# First evaluate saved checkpoints; no training:
python -m project_kidra.followup run \
  --root runs/followup --slots cuda:0 cuda:1 --only 'audit-*'

# Then run the remaining experiments; completed audits are skipped:
python -m project_kidra.followup run \
  --root runs/followup --slots cuda:0 cuda:0 cuda:1 cuda:1

python -m project_kidra.followup status --root runs/followup
```

Use the actual original suite paths. The new root must be outside both original
roots and must not contain either one. Preparation validates completion markers
and result hashes, then freezes five final checkpoints and all four datasets in
the new root. It does not deserialize models or train. Workers validate checkpoint
identity, architecture, and final step before using them. A missing final checkpoint
causes an explicit error, not a silent retrain or a substitution from an earlier step.

The scheduler defaults to two workers on `cuda:0`. Repeated device names allocate
multiple independent processes. Each worker uses two CPU threads by default.
The same interpreter launches all children; a missing PyTorch installation is
reported before launching the queue. Device model, Python/PyTorch/CUDA versions,
TF32 flag, model configuration, and parameter count are recorded for every job.

## Sixteen jobs, with separate questions

| Action | Jobs | Input and budget | Question |
| --- | ---: | --- | --- |
| `audit` | 5 | Existing final weights only | Do stricter generation and paired changes break the learned algorithm? |
| `replicate` | 8 | Four conditions × seeds 43 and 44; 50,000 updates each | Are seed-42 outcomes reproducible? |
| `extend` | 3 | Existing seed-42 weights **and optimizer state**, 50,000 → 100,000 total updates | Did the timing conditions simply need more training? |

Audits cover the low-LR full-gradient packed baseline, full-gradient tick baseline,
detached tick baseline, global-time run, and continuous-reasoning run. Replications
cover full packed, full tick, global time, and continuous reasoning. Continuations
cover full tick, global time, and continuous reasoning. Every condition uses original
loss weights, learning rate 0.0001, and 991,462 parameters.

Replications use exactly the copied 50,000 training records. Seed controls both
initialization and sample permutation, matching the original suite's convention.
Continuations use a **second shuffled pass over those same records**; they do not
introduce 50,000 new unique examples. Epoch zero uses `seed + 1000003` for its
permutation, exactly as before. Epoch one uses `seed + 1000004`. A resumed cursor
selects the same epoch and record. Continued optimizers are not reinitialized, and
learning rate remains constant. A continuation's history file contains its new
50k–100k segment; its original segment stays in the original suite.

There are **550,000 additional sample updates**, compared with 3.5 million in the
completed studies. Evaluation can still be substantial: each job runs 20 diagnostic
sections with up to 2,000 records each. Audits are evaluation-only, not zero-cost.
Concurrency and hardware affected the previous timings, so no runtime guarantee is
made from those totals. Filters also allow `--only 'replicate-*'` and `--only 'extend-*'`.

## Diagnostic sections

IID and OOD records receive the following paired tests, preserving their sample IDs:

| Probe | Change | Interpretation |
| --- | --- | --- |
| `original` | Original evaluator and data | Reproduce the saved accuracy before interpreting other probes. |
| `shift4096` | Add 4096 to every global tick | Arbitrary clock-origin invariance; baseline should be exactly unchanged. |
| `gaps64` | Add 64 empty ticks between stages | Idle-time sensitivity; baseline should be exactly unchanged. |
| `stagger` | Start initial document at 0, system at 5, and query at 11 relative ticks | Unequal observation arrivals; later stages still wait for observations. |
| `reverse_tables` | Reverse complete map and result key/value pairs | Layout invariance with unchanged facts, padding, lengths, and labels. |
| `prefix24` | Add 24 `FILL` tokens before each map/result table | Longer context and shifted local positions within the unchanged 128-position limit. |
| `route_change` | Change the queried map destination, recompute call/FOUND/ANSWER from the result table | First-hop causal sensitivity, complementing the old value-only counterfactual. |
| `bounded16` | Generate each output until NOOP/END, capped at 16 tokens independently of reference length | Detect dependence on gold-length truncation; retain failed generation traces. |

The delayed split also receives the original evaluator. IID and OOD receive
correct-prefix diagnostics, and IID receives the no-exchange ablation. All runs use
native tick inference, including those trained with packed execution.

For `bounded16`, an actual generated control ends a stream. The next stage starts
after the actual generated group ends, retaining the supplied external delay. A
well-formed `TOOL CALL kN NOOP` releases the existing result table even if `kN` differs
from the reference key. The toy tool returns its full table, so every valid key can
receive the same table; this does not simulate a general external tool API. No gold
prefix is inserted. Syntax, requested key, reasoning value, answer, and complete
trajectory are scored separately. Exceeding the common cap or context is reported.

**This is still a scripted protocol, not autonomous planning.** The environment
supplies which stage/window is eligible to act. The new probe removes gold-length
cutoffs and the old correctness-gated tool response, not every external scheduling
assumption. Likewise, the timing probes test robustness to irrelevant timing,
not whether the model can solve a problem whose answer depends on elapsed time.

`prefix24` is an out-of-training-distribution layout/length challenge. Failure would
not alone prove an architectural defect; it could reflect missing training coverage.
`route_change` can cross the original query/value split and can preserve the final
value when two table entries share it; it always changes the required call. Its paired
score uses complete trajectory correctness, so that case still tests routing.

## Evidence and recovery

Each job writes:

- `results.json`: aggregate measurements, paired both-correct/lost/gained rates,
  and hash/count metadata for completed sections.
- `records/<split>-<probe>.jsonl`: metrics for every sample in a stable order;
  bounded generation additionally stores the first 16 failed traces per split.
- `environment.json`, `status.json`, `worker.log`, and `complete.json`.
- `history.json` and resumable `checkpoint.pt` / `checkpoint.previous.pt` for
  jobs that train. Audit jobs use immutable copied source weights.

Ctrl-C/SIGTERM stops dispatch and lets a training worker finish its current sample
and save. Diagnostic evaluation checks for stopping between samples; an unfinished
section is recomputed on resume, while completed sections are hash-checked and skipped.
Hard crashes resume from the latest intact checkpoint (previous-generation fallback),
repeating at most the uncheckpointed work. Completed jobs are skipped. Surviving
workers retain their slots via locks. Retry behavior matches the original scheduler.

Rerun the same command after a crash. Original source checkpoints are never rotated,
rewritten, or repaired; only the follow-up copies and new outputs are used.

Export after completion:

```bash
python -m project_kidra.followup status --root runs/followup
find runs/followup -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.log' \) \
  ! -path 'runs/followup/data/*' -print0 | \
  tar --null -T - -czf kidra-followup-results.tar.gz
```

Keep all `.pt` files and copied datasets locally; per-record measurement JSONL files
should be returned. The export includes the manifest, summary, environment metadata,
paired records, histories, status, and logs, but excludes the training/evaluation
input datasets and model weights.

## Decision rules

1. Reproduce existing final scores on `original` before interpreting a changed probe.
2. Compare seeds as independent training replicates; 2,000 examples from one model
   do not substitute for additional training seeds.
3. Retain packed execution if its accuracy and seed variability match tick execution.
4. Attribute global-time failures under clock shifts to absolute-origin sensitivity,
   not to failure of cross-window learning in general.
5. If continuous reasoning improves by 100k, the 50k result was budget-limited in
   this setting. If it fails across seeds and at 100k, reject this exact forced-FILL
   intervention for this workload. Neither result establishes the value of learned
   reasoning or of every possible continuous-state design.
6. If reference-length and bounded scores differ, inspect traces before changing
   architecture; protocol training and termination need to match deployment.

Remaining later-stage scope includes a matched serialized-context baseline, tasks
with information genuinely encoded in event order or elapsed time, autonomous
scheduling/tool recovery, dynamic window eviction/reuse, larger vocabularies and
long contexts, sparse routing, and pretrained adapters. This patch deliberately
resolves the immediate diagnostic ambiguities without claiming that a toy benchmark
can determine an ideal general-purpose agent architecture.
