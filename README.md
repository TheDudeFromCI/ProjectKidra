# Project Kidra

An experimental PyTorch project. Strictly typed (mypy `--strict`), tested with pytest,
with SQLite (stdlib) for storage and an optional vector database (ChromaDB).

## Usage

```sh
python -m venv .venv   # first time only
./build.sh             # install deps, type-check, build package
./run.sh               # run the app
./test.sh              # mypy + pytest
pip install -e .[vector]   # optional: vector database support
```

## Layout

- `project_kidra/` – source package (`app.py` entry point, `db.py` SQLite helpers)
- `tests/` – unit tests

## Mixture of Experts model

`project_kidra.moe.MoEModel` combines causal multi-head self-attention with
top-k routed feed-forward experts. Each token can attend to itself and up to
`context_window - 1` preceding tokens. Inputs may be longer than the window;
sinusoidal positions encode token order, and attention remains local. Configure
the number of heads and context window with `MoEConfig`, then use `Trainer`
with token inputs and same-shaped next-token targets to train the model.

## Continuous token-level training

The stream trainer tokenizes a text corpus with the word-level `Tokenizer`
(10k words plus `<|pad|>`, `<|unk|>`, `<|tool_begin|>`, `<|tool_end|>`), predicts the
next token, and wraps to the beginning at the end. Out-of-vocabulary words become `<|unk|>`.

### Building a corpus

```sh
python -m project_kidra.build_corpus texts/ more.txt --output corpus.npy   # dirs are searched for *.txt
python -m project_kidra.continuous_train corpus.npy --checkpoint kidra.pt
```

`build_corpus` writes `corpus.npy` plus a `corpus.json` sidecar (token counts, unknown
rate, sources, tokenizer fingerprint). `continuous_train` loads a `.npy` corpus directly
and refuses it if the tokenizer vocabulary has changed; any other file is tokenized
as raw text at startup. Model and optimizer state are periodically
saved so training can resume:

```sh
python -m project_kidra.continuous_train corpus.txt --checkpoint kidra.pt
python -m project_kidra.continuous_train corpus.txt --checkpoint kidra.pt --resume --steps 1000
```

Training runs indefinitely unless `--steps` is set. This prototype uses the
existing fixed-size MoE architecture; it does not implement mini-AGI's dynamic
capacity growth or disk/VRAM expert paging.

## License

MIT – see [LICENSE](LICENSE).
