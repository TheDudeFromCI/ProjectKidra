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

## Mixture of Experts model

`project_kidra.model.AgentModel`, configured with `project_kidra.model.Config`,
combines causal multi-head self-attention with top-k routed feed-forward experts.
Each token can attend to itself and up to
`context_window - 1` preceding tokens. Inputs may be longer than the window;
sinusoidal positions encode token order, and attention remains local. Configure
the number of heads and context window with `Config`, then use `Trainer`
with token inputs and same-shaped next-token targets to train the model.

## Continuous token-level training

The stream trainer tokenizes a text corpus with the word-level `Tokenizer`
(10k words plus `<|pad|>`, `<|unk|>`, `<|tool_begin|>`, `<|tool_end|>`), predicts the
next token, and wraps to the beginning at the end. Out-of-vocabulary words become `<|unk|>`.

### Building a corpus

```sh
# Simply point the corpus builder script to your .txt directories.
# Any number of input directories can be supplied.
./scripts/corpus.sh <input_dir1> <input_dir2> --output <output.npy>
```
## Resume training

```sh
python -m project_kidra.continuous_train corpus.txt --checkpoint kidra.pt
python -m project_kidra.continuous_train corpus.txt --checkpoint kidra.pt --resume --steps 1000
```

Training runs indefinitely unless `--steps` is set. This prototype uses the
existing fixed-size MoE architecture; it does not implement mini-AGI's dynamic
capacity growth or disk/VRAM expert paging.

## License

MIT – see [LICENSE](LICENSE).
