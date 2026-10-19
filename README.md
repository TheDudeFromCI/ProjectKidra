# Project Kidra

An experimental multi-window language model. Train the original scratch model,
or adapt a local **Qwen3-1.7B-Base / Qwen3-0.6B-Base** checkpoint with LoRA.

## Setup

```sh
./scripts/build.sh  # installs development + Qwen dependencies, type-checks, builds
./scripts/test.sh   # mypy + pytest
```

For training without development tools: `pip install -e '.[qwen]'`.
Qwen support uses Transformers 4.51–4.x, Accelerate, and safetensors. Install a
PyTorch CUDA build that supports your GPU; the 3060 and 5060 may require different
minimum CUDA/PyTorch versions. No model or tokenizer files are downloaded at runtime.

## Local pretrained files

Point `--pretrained` at a directory containing:

- `config.json` from the selected Qwen3 Base model.
- Its original `tokenizer.json` (do not use Kidra's bundled vocabulary or renumber IDs).
- `model.safetensors`, **or** `model.safetensors.index.json` and all referenced shards.

Dimensions come from `config.json`; there is no hardcoded 1.7B/0.6B selection.
The loader rejects missing, extra, or incompatible weights. It supports dense,
unquantized Qwen3 with standard RoPE and full attention, including both requested
Base checkpoints. It does not load GGUF or quantized safetensors checkpoints.

The Qwen backend preserves separate decoder layers, GQA, explicit head dimensions,
Q/K RMSNorm, SwiGLU, RoPE theta, and tied embeddings. Kidra's shared MoE block is
not substituted into the pretrained network. All pretrained weights stay frozen;
rank-8 LoRA adapters train the attention and MLP projections. A shared low-rank
exchange module reads other windows' local KV caches every four layers. It has no
separate KV cache. LoRA and exchange output residuals start at zero, so initial
logits match unmodified Qwen for each independent window within the retained
context. Learned slot embeddings are used only in cross-window keys.

## JSON corpus format

Each sample is an object containing an ordered list of raw text blocks:

```json
{
  "windows": [
    "The workshop inventory contains six copper gears and two iron gears.",
    "Question: How many gears are there? Answer: Eight."
  ]
}
```

A `.json` file can contain one sample or an array of samples. A `.jsonl` file
contains one complete sample object per nonblank line. JSONL is streamed one
sample at a time for large corpora; JSON arrays are read into memory. Directory
inputs recursively find both extensions. Unicode and whitespace are preserved.
Empty windows, empty sample lists, unknown fields, and malformed records are
rejected. Write raw text without BOS/EOS wrappers; the builder appends the
model's EOS token to each block, and Qwen supplies BOS as its first input.

```sh
python -m project_kidra.build_corpus ./datasets \
  --pretrained /path/to/Qwen3-1.7B-Base \
  --output corpus.kidra
```

The compiled `.kidra` corpus is an indexed SQLite file with complete window groups,
tokenizer fingerprint, vocabulary size, and sample/token counts. It samples groups
randomly with replacement, loading only a bounded number into RAM. Legacy `.txt`
inputs and old `.npy` corpus files must be rebuilt from JSON. Failed builds do not
overwrite an existing corpus.

`GroupedTrainer` opens one sample at a time. It never mixes unrelated samples in
cross-window attention. Window list order assigns stable Qwen slot IDs. Short
windows stop advancing but remain readable until the entire sample finishes;
then all windows close and their caches are discarded. Every text token and EOS
is a training target; there are no prompt-only or loss-masked fields in this schema.

Windows advance at ticks 0, 1, ... within each chunk. They read previous inputs
from the same tick, never same-tick targets or future inputs. Consequently, a
window cannot immediately read another block's entire text: its available
context grows as that block advances. Custom asynchronous schedules are available
through `train_chunk(updates={window: ticks})`; JSON does not encode schedules.

## Fine-tune 1.7B and fall back to 0.6B

Start with the 12 GB card, selecting its actual CUDA index:

```sh
nvidia-smi --query-gpu=index,name,memory.total --format=csv

python -m project_kidra corpus.kidra \
  --pretrained /path/to/Qwen3-1.7B-Base \
  --device cuda:0 --dtype bfloat16 \
  --windows 4 --local-context 256 --tokens-per-step 32 \
  --checkpoint kidra-1.7b.pt --steps 1000
```

`--windows` must cover the largest sample. LoRA and exchange weights/Adam state
stay float32; the frozen backbone is bfloat16 by default. Non-reentrant gradient
checkpointing is enabled, and only real target rows are projected into logits.
`--tokens-per-step` is the maximum targets **per window**, per optimizer step.
Reduce chunk size and local context to reduce activation and cache memory.
`--lora-rank`, `--exchange-rank`, `--exchange-every`, and `--learning-rate` configure
the adapters. `--no-gradient-checkpointing` trades memory for less recomputation.
Use `--device cpu --dtype float32` for CPU smoke tests.

This implementation uses **one selected GPU**. It does not pool the 12 GB and
8 GB cards, use inference-only `device_map="auto"`, or implement distributed
training/CPU offload. 1.7B's BF16 weights alone occupy roughly 3.4 GB in decimal
units; adapters, optimizer state, activations, logits, and caches add memory.
These defaults are a starting configuration, not a measured VRAM or speed claim.

For the fallback, build a corpus with the 0.6B tokenizer and start a new run:

```sh
python -m project_kidra.build_corpus ./datasets \
  --pretrained /path/to/Qwen3-0.6B-Base --output corpus-0.6b.kidra
python -m project_kidra corpus-0.6b.kidra \
  --pretrained /path/to/Qwen3-0.6B-Base --device cuda:0 \
  --windows 4 --local-context 256 --tokens-per-step 32 \
  --checkpoint kidra-0.6b.pt --steps 1000
```

The loader permits corpus reuse only if tokenizer fingerprints and vocabulary
sizes match exactly. 1.7B adapter weights cannot be resumed on 0.6B. A few million
fine-tuning tokens can test this adaptation, but do not guarantee that useful
cross-window behavior will emerge; dataset design and evaluation still matter.

## Resume and inference

```sh
python -m project_kidra corpus.kidra \
  --pretrained /path/to/Qwen3-1.7B-Base --device cuda:0 \
  --checkpoint kidra-1.7b.pt --resume --steps 1000
```

Qwen checkpoints store trainable adapters, optimizer state, architecture/options,
and tokenizer fingerprint. They require the **same original base weights** on
reload; configuration/tokenizer checks do not hash every frozen weight. They are
Kidra adapter checkpoints, not standalone Hugging Face models. Resume restores
saved architecture, adapter settings, and learning rate. Chunk size/device/dtype
can change. Context caches, dataset cursor, RNG state, and step counter are not
restored: resume starts fresh randomly sampled groups, not an exact stream replay.
Checkpoints are written periodically, on normal completion, and after Ctrl-C
once at least one step has completed. `--steps 0` trains indefinitely.

The Qwen API provides `generate_step(temperature=0.0)` for one inference token per
open window without optimization. `forward()` trains pending targets only;
it never self-trains sampled predictions. The older scratch API retains its
original behavior described below.

```python
import torch
from project_kidra.model.qwen.model import QwenAgent
from project_kidra.model.grouped_trainer import GroupedTrainer

model = QwenAgent.from_pretrained(
    "/path/to/Qwen3-1.7B-Base", device="cuda:0", dtype=torch.bfloat16,
    n_windows=4, n_local_ctx=256,
)
# samples is an iterable of list[list[int]], one token list per window.
trainer = GroupedTrainer(model, samples, tokens_per_step=32)
while losses := trainer.step():
    print(sum(losses) / len(losses))
```

All caches are detached after each optimizer step. Retained history is not
recomputed after adapter updates, preserving Kidra's truncated streaming training
approximation. Windows keep independent RoPE positions; a Qwen window reaching
the pretrained position limit must be closed and reopened.

## Validation

Tests create tiny local Qwen3 safetensors fixtures; no production weights are
needed. They compare reference logits, test GQA with explicit head dimensions,
single-file/sharded loading, frozen-weight training, BF16 adapters, checkpointed
backpropagation, asynchronous schedules, future-token leakage, grouped corpus
boundaries, and the build/train/resume CLI. Production model quality, full-size
GPU throughput, and VRAM use require testing on the target hardware.

## Scratch-model API

The original `AgentModel` remains available and combines RoPE local attention, top-k routed experts, and global
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

## Timed multi-window toy experiment

The [diagnostic matrix](docs/timed-matrix.md) adds 68 matched experiments, an
independent single-tick execution path, full-trajectory gradients, stage-level
metrics, and a concurrent scheduler with checkpoint recovery. Prepare once,
then rerun the scheduler command to continue interrupted work and skip completed
experiments.

The `project_kidra.timed` commands generate event-ordered JSONL content streams,
train a 991,462-parameter native model with sample-level gradient accumulation,
and evaluate greedy rollouts with a cross-window attention ablation. See
[the setup, protocol, and run commands](docs/timed-toy.md). This experiment is
prepared but has not been run; learning quality is unmeasured.

## License

MIT – see [LICENSE](LICENSE). Qwen3 model weights retain their upstream Apache-2.0
license. Transformers provides the pretrained Qwen3 modules.
