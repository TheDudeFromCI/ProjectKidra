from pathlib import Path

import pytest
import torch

from project_kidra.tokenizer import Tokenizer
from project_kidra.moe import TokenCorpus, ContinuousTrainer, MoEConfig, MoEModel, Trainer


def _tokenizer() -> Tokenizer:
    """Build a tiny tokenizer so tests don't depend on the full vocabulary."""

    tokenizer = Tokenizer()
    tokenizer.tokens = ["a", "b", "c", "d", "e"]
    return tokenizer


def test_token_corpus_produces_overlapping_windows_and_wraps(tmp_path: Path) -> None:
    """The stream predicts each next token and wraps at the end of the corpus."""

    path = tmp_path / "corpus.txt"
    path.write_text("a b c d e")
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
    path.write_text("a b c d e a b c d e")
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
