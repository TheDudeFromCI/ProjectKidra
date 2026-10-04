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

## License

MIT – see [LICENSE](LICENSE).
