# Timed multi-window toy experiment

This experiment trains the native Kidra model from random initialization on a
synthetic two-hop lookup task. It has **991,462 trainable parameters**, a fixed
38-token vocabulary, two applications of the shared native layer, one dense
expert, and five window slots. It does not load Qwen or use LoRA.

The experiment is prepared but has not been run. No convergence, accuracy,
throughput, or GPU-memory result is claimed.

## Install and generate

Run from the repository root using Python 3.11 or later. Install a PyTorch build
appropriate for your GPU if it is not already installed; the existing project
environment can be reused.

```bash
python -m pip install -e '.[dev]'

python -m project_kidra.timed generate --output data/timed/train.jsonl --count 2000 --seed 1
python -m project_kidra.timed generate --output data/timed/eval.jsonl --count 128 --seed 2 --split test
python -m project_kidra.timed generate --output data/timed/delayed.jsonl --count 128 --seed 3 --split test --delayed
```

Generation uses only the Python standard library and the fixed toy tokenizer;
it does not download a corpus or a model. The command entry point imports
PyTorch but never constructs a model for `generate`.

## Train, resume, evaluate

`cuda:0` selects the first GPU visible to PyTorch. Use `cuda:1` for the second,
or `cpu`. Training uses one selected device. `--steps` counts **complete
samples**, with one update of each of the model's two disjoint optimizers per
sample. The two optimizers divide parameter ownership; they do not update the
same weights twice.

```bash
python -m project_kidra.timed --threads 4 train \
  --data data/timed/train.jsonl --eval-data data/timed/eval.jsonl \
  --output runs/timed --steps 2000 --eval-every 250 --seed 42 --device cuda:0

python -m project_kidra.timed --threads 4 train \
  --data data/timed/train.jsonl --eval-data data/timed/eval.jsonl \
  --output runs/timed-resumed --steps 2000 --eval-every 250 \
  --resume runs/timed/checkpoint.pt --device cuda:0

python -m project_kidra.timed --threads 4 evaluate \
  --checkpoint runs/timed/checkpoint.pt --data data/timed/delayed.jsonl \
  --output runs/timed/delayed-metrics.json --device cuda:0
```

The step count is an initial budget, not an established sufficient training
duration. Resume adds the requested number of steps. It restores weights,
optimizer states, completed steps, RNG state, and the unused shuffled sample
order; it rejects a changed training dataset or row order. Reproducing CUDA
execution requires the same environment and visible GPU layout. Cross-device
loads do not promise bitwise reproducibility.

`checkpoint.pt` is saved at each evaluation interval and at the final step.
`metrics.json` contains initial accuracy, periodic loss and accuracy, final
accuracy, a cross-window ablation, and five generated example trajectories.
Progress is also printed as JSON. A resumed run writes a fresh metrics report;
using a separate output directory retains the earlier report.

## Content-stream format

Each JSONL row is one complete sample:

```json
{
  "sample_id": "example",
  "streams": [
    {"window_id": 0, "offset_tick": 0, "text": "SYSTEM LOOKUP", "predict": false, "terminal": "<|noop|>"},
    {"window_id": 1, "offset_tick": 0, "text": "USER ASK k2", "predict": false, "terminal": "<|noop|>"},
    {"window_id": 1, "offset_tick": 4, "text": "", "predict": true, "terminal": "<|noop|>"}
  ],
  "group_ends": [4, 5]
}
```

This minimal example illustrates the format; the task-specific evaluator
expects the generator's complete call, result, reasoning, and answer sequence.

- `window_id` selects the persistent slot. A later stream with the same ID
  resumes that window and appends content; it does not replace previous text.
- `offset_tick` is a zero-based **global decoding tick**. Concurrent windows
  each consume at most one token per tick. Two windows advancing together for
  25 tokens consume 25 ticks, not 50. A packed forward call may process many
  logical ticks using causal masks.
- `text` contains whitespace-separated toy symbols, not arbitrary natural
  language. `ToyTokenizer.words` defines the stable vocabulary.
- `predict: false` supplies an observation with zero direct prediction loss;
  `true` supervises all content tokens and the terminal control token.
- `terminal` defaults to `<|noop|>`, which pauses the window. `<|end|>` completes
  it permanently for that sample. Terminal tokens occupy a tick even for
  empty content. Control tokens must not occur inside `text`.
- `group_ends` gives increasing, exclusive tick bounds for backward groups.
  Streams must fit within a group, and a window can have only one stream per
  group. Streams can overlap across different windows and start at different
  offsets. Idle gaps need no repeated waiting targets.

For 25 content tokens starting at offset 0, content occupies ticks 0–24,
NOOP occupies tick 25, and the next stream can start at tick 26.

## Task and timing groups

| Window | Role | Content |
| --- | --- | --- |
| 0 | System | Fixed lookup instruction |
| 1 | Chat | User query, supervised wait, final answer |
| 2 | Document | Shuffled random map from eight keys to destination keys |
| 3 | Reasoning / request | Predicted `TOOL CALL kN`, wait, then `FOUND vN` |
| 4 | Tool observation | Shuffled table of random values by key |

1. Supply system, user query, and map concurrently.
2. Predict chat NOOP and the destination-key tool request concurrently.
3. Supply the delayed tool observation.
4. Resume reasoning and predict the lookup value followed by END.
5. Resume chat and predict the answer followed by END.

The correct call requires the query in window 1 and the map in window 2. The
answer requires the tool table in window 4. The request is a supervised output,
not a supplied observation. Random map/table order and distractor entries
prevent solving by reading a fixed table position.

Training and evaluation use disjoint query/value combinations, selected by
`(query + answer) % 4`. All individual keys and values occur in both splits.
This is a compositional holdout within one task family, not a test of broad
agent intelligence or unseen tools. The delayed set adds longer padding and
idle gaps. Idle gaps alone do not change model state: ticks control visibility,
and elapsed time is not separately embedded.

## Training and cache semantics

Weights stay fixed throughout a sample. Each supervised group runs a packed
forward and backward pass. Observation-only groups run without autograd.
Gradients accumulate across groups and are clipped before the sample's final
optimizer update. Caches are detached between groups, so later losses do not
backpropagate through earlier groups' cached computations.

Loss is normalized over all supervised weighted tokens in the sample, not over
group means. NOOP has weight 0.25; other targets have weight 1.0. Waiting does
not get an entire group's worth of weight merely because it is a one-token
stream. The constructor exposes this weight for experiments.

A target token is predicted from the preceding content input. A terminal
NOOP is scored and recorded as the paused state, but is never ingested into KV.
Resuming removes that target, rolls back one cached input and its position,
then recomputes the last content input against the new cross-window context.
It never reuses stale waiting logits. END is also not ingested, and a completed
window's content remains readable until sample cleanup.

Within a packed group, same-tick preceding inputs are visible; same-tick
targets and future inputs are hidden. All windows close at the end of the
sample, including on exceptions. This prototype rejects context overflow;
it does not implement long-document eviction with replay.

## Reading evaluation results

Evaluation generates model-owned streams greedily, token by token. It never
teacher-forces their reference content. The first generated NOOP or END stops
that stream. A wrong tool call receives no tool observation; the environment
does not substitute the correct call. There is no loss or optimizer update
during evaluation.

| Metric | Meaning |
| --- | --- |
| `call_correct` | Exact request, destination key, and NOOP |
| `reasoning_correct` | Exact resumed `FOUND` stream and END |
| `answer_correct` | Exact answer stream and END |
| `controls_correct` | Correct terminal type at the expected position for every predicted stream |
| `exact` | Every predicted stream correct, including the initial chat wait |

The normal and `without_cross_window` conditions use the same checkpoint.
The ablation removes attention to other windows while retaining local global
attention. A meaningful drop supports reliance on cross-window information.

**Scheduling is supplied externally.** Reference offsets and lengths define
wakeups and generation deadlines. Early completion stops generation; missing
the deadline fails that stream. The evaluator does not establish autonomous
window scheduling, efficient reasoning length, or recovery from arbitrary
tool errors. Its tool handling and metrics are specific to this lookup task.

## Verification without training the experiment

```bash
python -m mypy
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_timed.py tests/test_timed_cli.py -q
```

These tests exercise tiny protocol fixtures, controlled evaluator logits,
mocked CLI training dispatch, and an untrained checkpoint round-trip. They
do not train or evaluate the 991,462-parameter experiment for learning quality.
