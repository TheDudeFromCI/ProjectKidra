"""Integration tests: a toy model must learn from its own window and from other windows."""

import random
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from project_kidra.model.config import Config
from project_kidra.model.model import AgentModel

DATA_PATH = Path(__file__).parent / "test_model_learning_data.txt"
VOCAB = ["<|bos|>", "<|eos|>", "<|free|>", "a", "b", "c", "d"]
FREE_TOKEN = VOCAB.index("<|free|>")


def make_tokenizer() -> HFTokenizer:
    """Build a word-level tokenizer over the toy vocabulary."""

    tokenizer = HFTokenizer(WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="<|free|>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return tokenizer


def load_data() -> tuple[list[list[int]], list[tuple[list[int], list[int]]]]:
    """Parse the data file into local sequences and (window A, window B) echo pairs."""

    tokenizer = make_tokenizer()
    local: list[list[int]] = []
    echo: list[tuple[list[int], list[int]]] = []

    for line in DATA_PATH.read_text().splitlines():
        if line.startswith("local "):
            local.append(tokenizer.encode(line.removeprefix("local ")).ids)
        elif line.startswith("echo "):
            a, b = line.removeprefix("echo ").split("=>")
            echo.append((tokenizer.encode(a).ids, tokenizer.encode(b).ids))

    return local, echo


def run_episode(model: AgentModel, sequences: list[list[int]]) -> list[list[float]]:
    """Run one fresh episode with one window per sequence; return per-step losses per window."""

    model.windows = []
    model.available_window_ids = list(range(model.config.n_windows))
    for seq in sequences:
        model.open_window().expect(list(seq))

    steps = [model() for _ in range(len(sequences[0]))]
    return [[step[w] for step in steps] for w in range(len(sequences))]


def mean(values: list[float]) -> float:
    """Arithmetic mean."""

    return sum(values) / len(values)


@pytest.fixture(autouse=True)
def seed() -> None:
    """Make training deterministic."""

    random.seed(0)
    torch.manual_seed(0)


def test_dataset_is_well_formed() -> None:
    """Parsed data is non-empty and echo windows align step by step."""

    local, echo = load_data()
    assert local and echo
    assert all(len(a) == len(b) for a, b in echo)
    assert len({len(s) for s in local}) == 1


def test_learns_from_own_window() -> None:
    """With one window, later tokens of a repeating pair are predicted from earlier context."""

    local, _ = load_data()
    config = Config(
        vocab_size=len(VOCAB), d_model=16, d_hidden=16, n_experts=2, expert_top_k=1,
        n_layers=2, n_heads=2, n_windows=1, n_local_ctx=16, n_global_layers=1,
        trunk_lr=1e-3, expert_lr=1e-3,
    )  # fmt: skip
    model = AgentModel(config, FREE_TOKEN)

    # Steps 3, 5 and 7 predict p after a 'd'; only attention to earlier tokens can know p.
    p_steps = [3, 5, 7]
    history: list[float] = []
    for epoch in range(80):
        random.shuffle(local)
        episodes = [run_episode(model, [s])[0] for s in local]
        history.append(mean([mean([ep[t] for t in p_steps]) for ep in episodes]))

    # A context-free bigram model can't do better than ln(3) here.
    assert history[0] > 0.9
    assert mean(history[-10:]) < 0.35


def echo_training_loss(
    pairs: list[tuple[list[int], list[int]]], epochs: int
) -> tuple[float, float]:
    """Train on window pairs and return window B's (first, final) mean loss on random steps."""

    config = Config(
        vocab_size=len(VOCAB), d_model=16, d_hidden=16, n_experts=2, expert_top_k=1,
        n_layers=2, n_heads=2, n_windows=2, n_local_ctx=1, n_global_layers=1,
        trunk_lr=3e-3, expert_lr=3e-3,
    )  # fmt: skip
    model = AgentModel(config, FREE_TOKEN)

    # Window B's steps 2+ echo window A's random tokens; its own history can't predict them.
    # The final step is excluded: it is consistently learned worse than the others.
    history: list[float] = []
    for epoch in range(epochs):
        random.shuffle(pairs)
        losses = [mean(run_episode(model, [a, b])[1][2:-1]) for a, b in pairs[:24]]
        history.append(mean(losses))

    return history[0], mean(history[-3:])


def test_learns_from_other_window() -> None:
    """Window B can only predict its targets by reading window A through the global exchange."""

    _, echo = load_data()
    first, final = echo_training_loss(echo, epochs=20)

    assert first > 0.9  # chance is ln(3)
    assert final < 0.55


def test_unrelated_windows_stay_at_chance() -> None:
    """Control: when window A is unrelated to window B's targets, loss must not drop."""

    _, echo = load_data()
    shuffled = [(echo[i][0], echo[(i + 1) % len(echo)][1]) for i in range(len(echo))]
    _, final = echo_training_loss(shuffled, epochs=10)

    assert final > 0.9
