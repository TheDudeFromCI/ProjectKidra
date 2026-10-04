from pathlib import Path

import torch

from project_kidra.moe import ByteCorpus, ContinuousTrainer, MoEConfig, MoEModel, Trainer


def test_byte_corpus_produces_overlapping_windows_and_wraps(tmp_path: Path) -> None:
    """The byte stream predicts each next byte and wraps at the end of the file."""

    path = tmp_path / "corpus.txt"
    path.write_bytes(bytes(range(5)))
    with ByteCorpus(path, context_window=3) as corpus:
        inputs, targets = corpus.next_batch()
        assert inputs.tolist() == [[0, 1, 2]]
        assert targets.tolist() == [[1, 2, 3]]

        inputs, targets = corpus.next_batch()
        assert inputs.tolist() == [[3, 4, 0]]
        assert targets.tolist() == [[4, 0, 1]]


def test_continuous_trainer_saves_and_resumes(tmp_path: Path) -> None:
    """A checkpoint preserves the model, optimizer, and global step."""

    torch.manual_seed(0)
    config = MoEConfig(
        vocab_size=256,
        d_model=8,
        d_hidden=16,
        num_experts=2,
        top_k=1,
        num_layers=1,
        num_heads=2,
        context_window=3,
    )
    path = tmp_path / "corpus.txt"
    path.write_bytes(b"continuous training data")
    checkpoint = tmp_path / "checkpoint.pt"
    model = MoEModel(config)
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=1e-3))
    with ByteCorpus(path, config.context_window) as corpus:
        continuous = ContinuousTrainer(trainer, corpus)
        assert continuous.train(max_steps=1, checkpoint_path=checkpoint) == 1
        saved_weights = model.head.weight.detach().clone()

    resumed_model = MoEModel(config)
    resumed_trainer = Trainer(
        resumed_model, torch.optim.Adam(resumed_model.parameters(), lr=1e-3)
    )
    with ByteCorpus(path, config.context_window) as corpus:
        resumed = ContinuousTrainer(resumed_trainer, corpus)
        resumed.load_checkpoint(checkpoint)
        assert resumed.step == 1
        assert corpus.tell() == 3
        assert torch.equal(resumed_model.head.weight, saved_weights)
        assert resumed_trainer.optimizer.state_dict()["state"]
