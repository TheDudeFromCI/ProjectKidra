# Project Kidra

An experimental toy LLM that mixes together several different niche ideas for possible LLM architectures.

## Usage

```sh
./scripts/build.sh  # install deps, type-check, build package
./scripts/test.sh   # mypy + pytest
./scripts/run.sh    # run the app
```

## Layout

- `project_kidra/` – source package (`app.py` entry point)
- `tests/` – unit tests

## Sequence training

`AgentModel` combines RoPE local attention, top-k routed experts, and global
attention over the open windows' local KV caches. Windows retain separate token
positions and bounded caches.

Standard next-token training predicts **every next token**, not just the last
word of a sample. Given targets `[a, b, c]`, this model uses inputs
`[previous_token, a, b]` (or `[free_token, a, b]` for a new window). Causal masks
prevent any prediction from reading its target or future inputs, including
through another window and across multiple layers.

```python
from project_kidra.model.model import AgentModel
from project_kidra.model.config import Config
from project_kidra.model.trainer import Trainer

model = AgentModel(Config(vocab_size=16, n_windows=2), free_token=15)
trainer = Trainer(model, [[1, 2, 3, 4], [5, 6]], n_windows=2, tokens_per_step=32)
while losses := trainer.step():
    print(sum(losses) / len(losses))
```

Each `step()` performs one batched forward pass, one backward pass, and one step
of each optimizer. The objective averages cross-entropy over **all real target
tokens**, plus the router auxiliary loss. Returned losses are per token, ordered
by window then position; short samples need no padding. Completed windows close
after the chunk and replacements open on the next step. The trainer only
advances its own windows. Its API defaults to one token per window for existing
callers; the CLI defaults to 32.

### Independent update schedules

```python
model = AgentModel(Config(vocab_size=16, n_windows=2), free_token=15)
fast = model.open_window().expect([1, 2, 3, 4])
slow = model.open_window().expect([5, 6])
losses = model.train_chunk(updates={fast: [0, 1, 2, 3], slow: [0, 3]})
```

Tick lists must be nonnegative and strictly increasing within each window, with one pending
target per tick. Ticks are relative to the current chunk; they are **not** RoPE
positions. RoPE advances only when that particular window updates. Without an
explicit schedule, `model.train_chunk(max_tokens=32)` consumes up to 32 pending
targets per open window, scheduled at ticks 0, 1, ... . An explicit `updates`
mapping defines the complete schedule and overrides `max_tokens`.

At each tick, scheduled windows ingest their previous committed token and then
predict their new targets together. Same-tick **inputs** are visible across
windows, matching the original model; same-tick **targets** are not. Idle
windows remain readable but do not advance, and empty windows contribute no
cache entries. Both local and global attention see only the newest
`n_local_ctx` entries available **at the query's tick**, even when a chunk
exceeds that limit. Open/close operations occur between chunks; split a schedule
at a lifecycle boundary when a window must disappear earlier.

Gradients flow through earlier inputs within a chunk. Caches are detached after
the optimizer step, retaining truncated history for later chunks. This is not
numerically equivalent to updating weights after every token: weights stay
fixed within a chunk, gradients reach earlier chunk entries, and router balance
is measured across the chunk. Retained caches are not recomputed after a weight
update, preserving the existing streaming approximation.

Chunks require known teacher-forced targets. A pending `free_token` is rejected
before mutation; use the existing `model()` one-token path for generation,
where future generated inputs are not yet known. `model()` retains its original
behavior of advancing every open window and self-training unspecified targets.

Larger chunks reduce optimizer and Python overhead but retain more activations
and construct explicit global attention masks. Global attention still scales
with all open windows' memory. Reduce `--tokens-per-step` to fit available VRAM;
this change does not remove the global attention bottleneck.

### Building a corpus

```sh
# Simply point the corpus builder script to your .txt directories.
# Any number of input directories can be supplied.
./scripts/corpus.sh <input_dir1> <input_dir2> --output <output.npy>
```
## Run and resume training

```sh
python -m project_kidra corpus.npy --checkpoint kidra.pt --tokens-per-step 32
python -m project_kidra corpus.npy --checkpoint kidra.pt --resume --tokens-per-step 16
```

Checkpoint intervals count optimizer steps, each now potentially training many
tokens. Tokens/sec counts actual target tokens, and Avg Loss reports the current
chunk's mean token loss. Model parameter shapes and checkpoint format are unchanged.

## License

MIT – see [LICENSE](LICENSE).
