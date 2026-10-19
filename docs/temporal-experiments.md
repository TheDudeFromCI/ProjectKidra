# Two additive temporal experiments

This patch adds only `project_kidra/temporal/`, two new test files, and this guide.
It does not modify `project_kidra/timed/`, `project_kidra/model/`, the original
matrix manifest, or any original experiment output. The original suite fingerprints
the `timed/` and `model/` source trees; those fingerprints remain unchanged.
Applying this patch does not require stopping the existing scheduler or workers.
Do not regenerate the original matrix or apply an older full replacement patch.

## Apply and queue

From the repository containing the already-installed 68-experiment matrix:

```bash
git apply --check /path/to/temporal-update.patch
git apply /path/to/temporal-update.patch

python -m project_kidra.temporal prepare \
  --base runs/matrix --root runs/temporal

python -m project_kidra.temporal run \
  --root runs/temporal --slots cuda:0 cuda:0 --threads-per-job 2
```

Replace `runs/matrix` with the **existing suite directory containing its
manifest.json**. Choose a new, separate `runs/temporal` directory; it cannot be
inside the original directory, contain it, or resolve to it through a symlink.
Use the same Python environment that runs the existing matrix. If the project is
installed non-editably, run these commands from the patched repository checkout.
The scheduler uses POSIX file locks, like the original scheduler.

`prepare` constructs no model and trains nothing. It validates the old manifest,
copies the exact existing data bytes, and writes a separate temporal manifest.
Counts, train/evaluation records, checkpoint and evaluation intervals are inherited.
With the original defaults, each addition trains once on the same **50,000 records**.
No new data is generated. The seed defaults to the first original seed (normally 42),
and the learning rate defaults to **0.0001**. Optional `--seed` and
`--learning-rate` must select a baseline already present in the original manifest.

`run` can be started now: it waits until **every original run has an intact completion
report and no original worker remains live**, then starts the two additions. At
62/68 it will wait. A failed or abandoned original run therefore needs to be
completed with the original scheduler before the additions start. The wait prevents
new GPU contention from affecting the unfinished runs. It creates no files in the
original suite. Direct invocation of a temporal `worker` also refuses an unfinished
original suite. Keep the original directory available at its recorded absolute path.

Repeated device names run multiple jobs on one GPU. Use `--slots cuda:0` to run
sequentially, or `--slots cuda:0 cuda:1` for separate GPUs. No baseline is retrained.
The comparison baseline is recorded in the new manifest, normally:

`agent-full-tick-all-lr0.0001-s42`

## Conditions

Both runs use the existing **991,462-parameter** architecture, initialization, sample
order, full-trajectory gradients, single-token native execution, original labels,
loss weights (including NOOP 0.25), clipping, and optimizer settings. They do not
use `SequenceBatch`. Neither adds trainable parameters. These are two individual
interventions; there is no combined intervention or learning-rate sweep.

| Suffix | Intervention | What stays fixed |
| --- | --- | --- |
| `global_time` | Add a fixed sinusoidal encoding of each input token's global birth tick to its token and window embeddings, before projection. Scale 1, frequency base 10000. | Local RoPE, scheduling, window lifecycle and supervised targets. |
| `continuous_reasoning` | Advance reasoning window 3 once whenever an original window updates. If reasoning already has an authored update that tick, use it without an extra heartbeat. Otherwise force the existing `FILL` token with zero direct loss. | Original token times, original supervised targets, architecture; no explicit global-time vector. |

Global time resets for every sample and includes empty delayed gaps. A token's birth
tick is its insertion/generation tick, not the later tick at which it becomes the
preceding model input. BOS uses the window's opening tick. On NOOP wakeup, the
replayed input keeps its original birth time; it is not relabelled with wakeup time.
This means the global-time condition exposes token timestamps, not an additional
wall-clock query vector on every wakeup.

Continuous reasoning opens on the first active event tick. It also advances during
external observations and after its authored `END`, through the answer phase. Its
NOOP/END outputs are still scored normally, but do not stop the clock stream. Before
the next update, its unconsumed control token is removed without rolling back its
local position or KV cache. The preceding content may be replayed at the next
position, then a heartbeat or authored token is committed. Other windows retain the
original NOOP rollback behavior.

There is no heartbeat on a completely empty global tick. Thus this condition
encodes **active event count**, not absolute time across idle gaps. Multiple windows
updating simultaneously cause one reasoning tick. Heartbeats are forced during both
training and inference, including after an incorrect reasoning control; they are
not model-selected tokens and are not additional supervised labels. Their computation
remains in the gradient graph so later losses can train it. A tool result is still
hidden when the generated call is wrong in free-running evaluation.

This second intervention changes available computation and reasoning-history length
as well as timing information. An improvement would not isolate positional encoding
alone. The extension rejects records whose active-tick count would exceed the existing
128-position reasoning context, rather than silently truncating the clock history.
The original five-window generator's standard and delayed records fit this limit.

## Stop, resume, and inspect

```bash
python -m project_kidra.temporal status --root runs/temporal

# Resume with exactly the same command after a stop or crash:
python -m project_kidra.temporal run \
  --root runs/temporal --slots cuda:0 cuda:0 --threads-per-job 2
```

Ctrl-C or SIGTERM requests a checkpoint at a sample boundary; during final evaluation,
the current diagnostic section finishes first. Wait for the processes to exit.
Hard crashes resume from the last durable checkpoint, so at most the uncheckpointed
samples are repeated (default interval inherited from the base: 250). Model weights,
both optimizers, sample cursor, RNG states, partial loss totals, and history are saved.
There is a previous-checkpoint fallback. No temporal cache is needed at a sample
boundary because every sample has an independent clock and window session.

Matching completed runs are skipped. Final diagnostic sections already saved are
also skipped on resume. Surviving workers retain their slots after a scheduler crash;
new schedulers detect their locks. The scheduler retries a failed launch/run once by
default and then exits with a failure status instead of retrying indefinitely. Inspect
`experiments/<name>/worker.log` and `status.json` if a job fails.

The new manifest binds both base and extension source fingerprints. Subsequent edits
to either implementation require a new temporal root. Keep this additive patch fixed
while the two new runs are underway.

Evaluation matches the original suite: train probe, IID, OOD, delayed data, free rollout,
correct-prefix rollout, no-exchange ablation, and paired counterfactuals. Conditional
rollout restores timestamps and heartbeat caches as well as ordinary token state.

## Files to return

Run `status` for both suites when finished, then send:

- Both `manifest.json` files and both `summary.json` / `summary.csv` files.
- New `experiments/*/results.json`, `history.json`, `status.json`, and `worker.log`.
- The same files for the original baseline named in the temporal manifest.

Checkpoint `.pt` files and dataset JSONL files are unnecessary for initial analysis;
keep them locally in case a follow-up needs replay. Include logs for failed runs too.

## Verification performed

Ten targeted checks passed, including original-path numerical parity, birth-time
retention on wakeup, one reasoning update per active tick, parent-suite isolation,
read-only completion gating, two-slot dispatch, completed-run skipping, and worker
checkpoint recovery. There were exactly two actual training smoke passes: one
single-sample update per condition on a roughly 10k-parameter model, with no saved
experiment checkpoint. Worker lifecycle tests used mocked training and evaluation
and temporary test fixtures. The 991,462-parameter model was constructed only to
verify its parameter count, with no training or inference. No experimental result
or claim of improved learning has been produced.
