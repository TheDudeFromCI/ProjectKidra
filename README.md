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

## License

MIT – see [LICENSE](LICENSE).
