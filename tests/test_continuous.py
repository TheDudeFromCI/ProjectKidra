from pathlib import Path

import pytest
import torch

from project_kidra.corpus import Tokenizer, TokenCorpus
from project_kidra.moe import MoEConfig, MoEModel
from project_kidra.training import ContinuousTrainer, Trainer, Previewer


def _tokenizer() -> Tokenizer:
    """Build a tiny tokenizer so tests don't depend on the full vocabulary."""

    return Tokenizer(["apple", "banana", "orange", "red", "blue", "green"])


def test_token_corpus_produces_overlapping_windows_and_wraps(tmp_path: Path) -> None:
    """The stream predicts each next token and wraps at the end of the corpus."""

    path = tmp_path / "corpus.txt"
    path.write_text("apple banana orange red blue")
    corpus = TokenCorpus.from_text(path, context_window=3, tokenizer=_tokenizer())
    inputs, targets = corpus.next_batch()
    assert inputs.tolist() == [[0, 1, 2]]
    assert targets.tolist() == [[1, 2, 3]]

    inputs, targets = corpus.next_batch()
    assert inputs.tolist() == [[3, 4, 0]]
    assert targets.tolist() == [[4, 0, 1]]


def test_token_corpus_rejects_short_corpus(tmp_path: Path) -> None:
    """A corpus shorter than one window plus a target is invalid."""

    path = tmp_path / "corpus.txt"
    path.write_text("a b")
    with pytest.raises(ValueError):
        TokenCorpus.from_text(path, context_window=3, tokenizer=_tokenizer())


def test_continuous_trainer_saves_and_resumes(tmp_path: Path) -> None:
    """A checkpoint preserves the model, optimizer, and global step."""

    torch.manual_seed(0)
    config = MoEConfig(
        vocab_size=_tokenizer().vocab_size(),
        d_model=8,
        d_hidden=16,
        num_experts=2,
        top_k=1,
        num_layers=1,
        num_heads=2,
        context_window=3,
    )
    path = tmp_path / "corpus.txt"
    path.write_text("apple banana orange red blue apple banana orange red blue")
    checkpoint = tmp_path / "checkpoint.pt"
    model = MoEModel(config)
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=1e-3))
    corpus = TokenCorpus.from_text(path, config.context_window, _tokenizer())
    continuous = ContinuousTrainer(trainer, corpus)
    assert continuous.train(max_steps=1, checkpoint_path=checkpoint) == 1
    saved_weights = model.head.weight.detach().clone()

    resumed_model = MoEModel(config)
    resumed_trainer = Trainer(
        resumed_model, torch.optim.Adam(resumed_model.parameters(), lr=1e-3)
    )
    corpus = TokenCorpus.from_text(path, config.context_window, _tokenizer())
    resumed = ContinuousTrainer(resumed_trainer, corpus)
    resumed.load_checkpoint(checkpoint)
    assert resumed.step == 1
    assert corpus.tell() == 3
    assert torch.equal(resumed_model.head.weight, saved_weights)
    assert resumed_trainer.optimizer.state_dict()["state"]


def test_checkpoint_callback_and_previewer(tmp_path: Path) -> None:
    """on_checkpoint fires at each save and the previewer yields the requested token count."""

    torch.manual_seed(0)
    tokenizer = _tokenizer()
    config = MoEConfig(
        vocab_size=tokenizer.vocab_size(),
        d_model=8,
        d_hidden=16,
        num_experts=2,
        top_k=1,
        num_layers=1,
        num_heads=2,
        context_window=3,
    )
    path = tmp_path / "corpus.txt"
    path.write_text("apple banana orange red blue apple banana orange red blue")
    model = MoEModel(config)
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=1e-3))
    continuous = ContinuousTrainer(trainer, TokenCorpus.from_text(path, 3, tokenizer))
    previewer = Previewer(model, tokenizer, num_tokens=4)
    previews: list[tuple[int, int]] = []

    def on_checkpoint(step: int, inputs: torch.Tensor) -> None:
        """Record the step and generated length."""

        previews.append((step, len(previewer.generate(inputs[:1]))))

    continuous.train(
        max_steps=4,
        checkpoint_path=tmp_path / "c.pt",
        checkpoint_interval=2,
        on_checkpoint=on_checkpoint,
    )
    assert previews == [(2, 4), (4, 4)]
    assert model.training
