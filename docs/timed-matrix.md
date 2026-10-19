# Timed-window diagnostic matrix

This suite isolates training failures while keeping the native model fixed at
**991,462 parameters**. It is prepared for local execution; the full experiments
have not been run. The existing `project_kidra.timed` CLI retains its original
behavior. Use the new `project_kidra.timed.matrix` CLI for these experiments.

## Run the suite

Use the same Python environment as the previous timed experiment. No additional
dependencies or pretrained weights are needed. The scheduler targets Linux/POSIX
and uses kernel file locks to prevent duplicate workers.

From the repository root, prepare the suite using your existing 50,000 records:

```bash
python -m project_kidra.timed.matrix prepare \
  --root runs/matrix --train-data data/train.jsonl
```

Omit `--train-data` to generate the common 50,000-record training set once.
Preparation writes a manifest and datasets but does not build or train models.
The default suite contains **68 experiments**, each processing all 50,000
training records once. Total training work is 3.4 million sample updates.

Start four workers, two on each of the two visible GPUs:

```bash
python -m project_kidra.timed.matrix run \
  --root runs/matrix \
  --slots cuda:0 cuda:0 cuda:1 cuda:1 \
  --threads-per-job 2
```

Each repeated device name grants one independent process slot. With
`--slots cuda:0 cuda:0`, two experiments run on the first GPU. With `--slots cpu`,
one runs on CPU. Defaults are two workers on `cuda:0`, two CPU threads per worker.
The scheduler does not pool VRAM, distribute one model across cards, or infer
available memory. GPU memory and full-suite runtime have not been benchmarked.

The next queued experiment starts when a slot becomes free. Training progress
goes to each experiment's `worker.log`; the scheduler prints starts and exits.
To continue after a crash or interruption, run **the same `run` command** again.
Concurrency and device assignments can change between invocations.

```bash
python -m project_kidra.timed.matrix status --root runs/matrix
```

`status` prints counts and refreshes `summary.json` and `summary.csv`, including
partial results. The summaries are also refreshed when the scheduler finishes.
An unowned stale `training` status is displayed as interrupted.

## Experimental factors

| Factor | Values | What it probes |
| --- | --- | --- |
| Task | Echo; two-entry lookup; eight-entry lookup; full agent | Where content retrieval or the protocol first fails |
| Gradient history | Full trajectory; detached between groups | Whether losses can train earlier document encoders |
| Execution | Packed group; native single tick | Whether temporal batching changes the learning computation |
| Loss weights | Original; fact-focused | Whether easy syntax dominates the objective |
| Learning rate | 0.0005; 0.0001 | Sensitivity to the original step size |

These factors form **4 × 2 × 2 × 2 × 2 = 64** experiments. Four additional
matched controls train without cross-window attention, one per task, using full
gradients, packed execution, original weighting, and learning rate 0.0005.
This is a diagnostic factorial design, not an exhaustive search over model designs.

All runs use the same fixed vocabulary and architecture. The default seed is
42 for both model initialization and a separate deterministic sample permutation.
Different task projections retain the same record IDs, answers, underlying
facts, and training order. Within a task, every condition receives byte-for-byte
equivalent sample content. Across tasks, the visible problem necessarily changes:

- **Echo:** a document supplies one value, which another window must emit.
- **Lookup2:** retrieve a value by key from two shuffled entries.
- **Lookup8:** retrieve it from eight shuffled entries.
- **Agent:** the original map lookup, generated tool call, delayed observation,
  resumed reasoning, and chat answer.

Lookup task projections relabel the original result-table keys so the same
original query identifies the same answer without the agent task's first hop.
Distractor selection is independent of table order. Simpler tasks remove waiting
and tool protocol decisions; they are diagnostic controls, not new agent datasets.

The full mode encodes observations with autograd, masks their direct prediction
loss, retains cache graphs, and runs one backward at sample completion. The
detached mode reproduces the old boundaries: observation-only groups have no
autograd, each supervised group runs backward, and caches detach between groups.
Both update each of the model's two disjoint optimizers **once per sample**.

In `tick` mode, each forward processes exactly one preceding input per active
window. Concurrent windows may still share a batch dimension; no forward batches
multiple time positions. This path calls the native `LayerBlock` and
`GlobalExchange` directly and never invokes `SequenceBatch`. Gradients still
accumulate to the same sample update, so execution batching is isolated from
optimizer frequency. It is not an optimizer update after every token.

The original loss weights ordinary targets by 1 and NOOP by 0.25. Fact-focused
loss weights key/value targets by 1, fixed syntax by 0.1, and both terminal types
by 0.25. Each objective is normalized by its own total weight over the complete
sample. Compare accuracy across weighting modes; their scalar losses are not
directly comparable.

## Evaluation

The shared evaluation sets each default to 2,000 records:

| Set | Distribution |
| --- | --- |
| `iid` | Fresh trajectories from the training generator's allowed query/value combinations |
| `ood` | Query/value combinations excluded from training |
| `delayed` | Training-distribution combinations with longer padding and idle gaps |
| `train_probe` | A fixed subset of the actual training order |

Every 2,000 training steps, 256 training-probe and IID examples are evaluated.
Final evaluation uses the full configured evaluation count. All inference uses
the native single-tick path, including models trained with packed groups.

For each set, the report contains:

- `free`: generated requests and outputs, with no teacher forcing. A wrong call
  receives no tool result.
- `conditional`: each group is generated freely and scored, then the state is
  reset to the correct reference prefix before the next group. This isolates
  later stages from earlier errors. It is explicitly not end-to-end accuracy.
- `noexchange`: free generation with attention to other windows removed.

`call_fact`, `reasoning_fact`, and `answer_fact` score the variable key/value at
its required output position. Corresponding `_exact` metrics include syntax and
the terminal token. `controls` is the fraction of predicted streams with the
correct terminal type at the expected position, independently of key/value
correctness. `exact` requires every generated stream to match.

A final counterfactual check changes only the requested value and its reference
labels, retaining the other facts and schedule. `both_correct` requires success
on the original and changed problem. Attention ablations remain disabled for
workers deliberately trained without exchange.

Wakeups, generation deadlines, and window identities remain externally supplied.
These experiments do not test autonomous scheduling or RL. Idle gaps alone do
not change model state; ticks control visibility, without an elapsed-time embedding.

## Checkpoint and restart behavior

Each experiment has its own directory under `runs/matrix/experiments/`:

| File | Purpose |
| --- | --- |
| `checkpoint.pt` | Latest committed sample cursor, weights, optimizer states, RNGs, loss totals, and metric history |
| `checkpoint.previous.pt` | Previous complete checkpoint for corruption recovery |
| `worker.log` | Appended progress records and error tracebacks across attempts |
| `status.json` | Current progress, device, and failure information |
| `history.json` | Periodic metric history restored across resumed execution |
| `results.json` | Incrementally committed final diagnostic sections |
| `complete.json` | Completion marker tied to the experiment identity and result hash |

Checkpoints default to every **250 samples**, separately from evaluation. A
temporary checkpoint is fully written and flushed before atomic replacement.
The previous generation is retained. If the newest cannot be read, the worker
uses the previous valid generation; if neither is readable it fails explicitly.
It never silently restarts a damaged experiment from random initialization.

A hard crash loses work after the latest valid checkpoint. Restart restores the
next sample index and optimizer moments; the shuffle order is deterministic and
does not restart at sample zero. CPU tests verify identical weights and optimizer
states after interruption. CUDA or cross-device bitwise equivalence is not promised.

Ctrl+C or SIGTERM to the scheduler stops dispatch and signals its own workers.
They finish the current sample and save a checkpoint; an evaluation already in
progress finishes its current section. Abrupt process termination or power loss
uses the last durable checkpoint. Healthy orphan workers keep their job locks;
a restarted scheduler counts their slots instead of launching duplicates.

A completed experiment is skipped only when its marker matches the manifest and
its results file is intact. An interrupted final evaluation resumes missing
sections without retraining. Failed jobs retry once per scheduler invocation,
then the queue moves on. A later invocation tries unfinished jobs again. Reduce
the repeated GPU slots if errors report insufficient VRAM.

The manifest fingerprints the datasets and relevant source code. Changed files,
training budgets, or scientific settings are rejected for an existing suite;
prepare a new root for a different study. The previous standalone experiment's
checkpoints are not imported: new matrix runs start from matched initialization.

## Optional scope controls

```bash
# Run only the independent single-tick variants; completed jobs still skip.
python -m project_kidra.timed.matrix run \
  --root runs/matrix --slots cuda:0 cuda:0 --only '*-tick-*'

# Replicate the full matrix across three initialization/order seeds: 204 runs.
python -m project_kidra.timed.matrix prepare \
  --root runs/matrix-three-seeds --seeds 42 43 44 --train-data data/train.jsonl
```

Preparation also accepts `--samples`, `--eval-samples`, `--checkpoint-every`,
`--eval-every`, and `--periodic-samples`. A supplied `--train-data` file must
contain exactly the configured number of original agent-format records.

## Verification

```bash
python -m mypy
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_timed_execution.py tests/test_timed_matrix.py \
  tests/test_timed_diagnostics.py tests/test_timed_scheduler.py -q
```

Tests compare packed/native losses and gradients, check that observation-only
embeddings receive gradients in full mode, learn a minimal four-value echo task
with a roughly 10k model using the native single-tick path, verify metric
separation, and exercise checkpoint fallback and scheduler recovery. Worker
recovery tests substitute tiny models; scheduler tests launch harmless file-writing
subprocesses. No 50,000-sample matrix experiment is launched by these tests.
