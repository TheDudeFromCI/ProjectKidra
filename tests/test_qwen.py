"""Local tiny Qwen checkpoints verify compatibility without downloading model weights."""

import copy
from pathlib import Path
from typing import Any, cast

import pytest
import torch

pytest.importorskip("transformers")
from transformers import Qwen3Config, Qwen3ForCausalLM

from project_kidra.model.qwen.checkpoint import QwenCheckpoint
from project_kidra.model.qwen.model import QwenAgent
from project_kidra.model.qwen.sequence import QwenSequence


def backbone() -> Any:
    """Use GQA and a head width unequal to hidden_size / heads, as in Qwen3-0.6B."""

    torch.manual_seed(3)
    cfg = Qwen3Config(  # type: ignore[no-untyped-call]
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        attention_dropout=0.0,
        rope_theta=1000000.0,
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(cfg).eval()  # type: ignore[no-untyped-call]


def agent(checkpointing: bool = False) -> QwenAgent:
    """Wrap a reproducible model with small adapters and multiple slots."""

    return QwenAgent(
        backbone(),
        n_windows=3,
        n_local_ctx=4,
        lora_rank=2,
        exchange_rank=4,
        exchange_every=1,
        gradient_checkpointing=checkpointing,
    )


def activate_exchange(model: QwenAgent) -> None:
    """Make exchange nonzero so causality tests cannot pass due to its initialization."""

    with torch.no_grad():
        cast(torch.nn.Linear, model.exchange.output[1]).weight.normal_(std=0.1)


def test_initial_logits_match_unmodified_qwen() -> None:
    """Zero adapters preserve reference logits in every independent window."""

    reference = backbone()
    model = QwenAgent(
        copy.deepcopy(reference),
        n_windows=2,
        n_local_ctx=16,
        lora_rank=2,
        exchange_every=1,
        exchange_rank=4,
    )
    windows = [model.open_window().expect(tokens) for tokens in ([3, 4, 5], [6, 7])]
    with torch.no_grad():
        actual = QwenSequence(model, {w: list(range(len(w.pending))) for w in windows}).forward()
        expected = torch.cat(
            [reference(torch.tensor([[1] + w.pending[:-1]])).logits[0] for w in windows]
        )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert model.backbone.lm_head.weight is model.backbone.model.embed_tokens.weight
    assert model.backbone.model.layers[0] is not model.backbone.model.layers[1]


@pytest.mark.parametrize("sharded", [False, True])
def test_load_local_safetensors(tmp_path: Path, sharded: bool) -> None:
    """Both a single safetensors file and an index of shards load with exact weights."""

    reference = backbone()
    reference.save_pretrained(
        tmp_path, safe_serialization=True, max_shard_size="2KB" if sharded else "1GB"
    )
    loaded = QwenAgent.from_pretrained(
        tmp_path,
        dtype=torch.float32,
        n_windows=2,
        n_local_ctx=16,
        lora_rank=2,
        exchange_rank=4,
        exchange_every=1,
    )
    torch.testing.assert_close(
        reference.model.embed_tokens.weight, loaded.backbone.model.embed_tokens.weight
    )
    window = loaded.open_window().expect([3, 4])
    with torch.no_grad():
        logits = QwenSequence(loaded, {window: [0, 1]}).forward()
        expected = reference(torch.tensor([[1, 3]])).logits[0]
    torch.testing.assert_close(logits, expected, atol=2e-6, rtol=2e-5)


def test_chunk_matches_tickwise_with_idle_history() -> None:
    """Packed chunks respect asynchronous schedules, context eviction, and idle windows."""

    model = agent()
    activate_exchange(model)
    for ts in ([3, 4], [5, 6], [7, 8]):
        model.open_window().expect(ts)
    with torch.no_grad():
        warm = QwenSequence(model, {w: [0, 1] for w in model.windows})
        warm.forward()
        warm.commit()
    model.windows[0].expect([9, 10, 11, 12, 13, 14])
    model.windows[1].expect([15, 16, 17])
    reference = copy.deepcopy(model)
    schedules = [[0, 1, 2, 3, 4, 5], [0, 2, 5]]
    with torch.no_grad():
        actual = QwenSequence(model, dict(zip(model.windows[:2], schedules))).forward()
        by_window: list[list[torch.Tensor]] = [[], []]
        for tick in range(6):
            active = [i for i, ticks in enumerate(schedules) if tick in ticks]
            step = QwenSequence(reference, {reference.windows[i]: [0] for i in active})
            logits = step.forward()
            for i, row in zip(active, logits):
                by_window[i].append(row)
            step.commit()
        expected = torch.stack([row for rows in by_window for row in rows])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert all(c.length <= 4 for w in model.windows for c in w.cache)


def test_future_targets_cannot_leak_across_windows() -> None:
    """Changing same-tick targets alters only later predictions, through every layer."""

    model = agent()
    activate_exchange(model)
    a = model.open_window().expect([3, 4, 5])
    b = model.open_window().expect([6, 7, 8])
    other = copy.deepcopy(model)
    other.windows[1].pending[1] = 20
    with torch.no_grad():
        first = QwenSequence(model, {a: [0, 1, 2], b: [0, 1, 2]}).forward()
        second = QwenSequence(other, {w: [0, 1, 2] for w in other.windows}).forward()
    torch.testing.assert_close(first[[0, 1, 3, 4]], second[[0, 1, 3, 4]], atol=0, rtol=0)
    assert not torch.allclose(first[2], second[2])


def test_checkpointed_backward_matches_regular_and_freezes_base() -> None:
    """Checkpoint recomputation produces the same updates without appending caches twice."""

    model = agent(False)
    activate_exchange(model)
    reference = copy.deepcopy(model)
    reference.gradient_checkpointing = True
    frozen = model.backbone.model.embed_tokens.weight.detach().clone()
    before = cast(torch.nn.Linear, model.exchange.output[1]).weight.detach().clone()
    for m in (model, reference):
        m.open_window().expect([3, 4, 5])
        m.open_window().expect([6, 7])
    losses = model.train_chunk()
    other = reference.train_chunk()
    assert losses == pytest.approx(other, abs=1e-6)
    for (name, value), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(value, expected, atol=2e-6, rtol=2e-5, msg=name)
    assert torch.equal(model.backbone.model.embed_tokens.weight, frozen)
    assert not torch.equal(cast(torch.nn.Linear, model.exchange.output[1]).weight, before)
    assert all(
        c.length == w.position and not c.k.requires_grad for w in reference.windows for c in w.cache
    )


def test_adapter_checkpoint_roundtrip_and_resume(tmp_path: Path) -> None:
    """Adapters, FP32 optimizer state, and options survive reload against local weights."""

    base = tmp_path / "base"
    backbone().save_pretrained(base, safe_serialization=True)
    model = QwenAgent.from_pretrained(
        base,
        dtype=torch.float32,
        n_windows=2,
        n_local_ctx=16,
        lora_rank=2,
        exchange_rank=4,
        exchange_every=1,
    )
    model.open_window().expect([3, 4])
    model.open_window().expect([5, 6])
    model.train_chunk()
    path = tmp_path / "adapters.pt"
    QwenCheckpoint.save(model, path, "fingerprint")
    loaded = QwenCheckpoint.load(path, base, "fingerprint", dtype=torch.float32)
    assert not loaded.windows
    for a, b in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert loaded.optimizer.state
    loaded.open_window().expect([7, 8])
    loaded.open_window().expect([9, 10])
    assert len(loaded.train_chunk()) == 4
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        QwenCheckpoint.load(path, base, "wrong", dtype=torch.float32)


def test_generation_does_not_optimize() -> None:
    """Inference never performs the old self-training-on-argmax behavior."""

    model = agent()
    model.open_window()
    weights = {name: p.clone() for name, p in model.named_parameters()}
    assert len(model.generate_step()) == 1
    assert model.windows[0].position == 1
    assert not model.optimizer.state
    assert all(torch.equal(weights[name], p) for name, p in model.named_parameters())


def test_invalid_schedule_does_not_mutate() -> None:
    """Schedule validation runs before any cache updates."""

    model = agent()
    window = model.open_window().expect([3, 4])
    for ticks in ([0, 0], [-1], [0, 1, 2]):
        with pytest.raises(ValueError):
            model.train_chunk(updates={window: ticks})
        assert window.position == 0
        assert all(c.length == 0 for c in window.cache)


def test_missing_weight_rejected(tmp_path: Path) -> None:
    """Never silently initialize a missing pretrained projection."""

    from safetensors.torch import load_file, save_file

    backbone().save_pretrained(tmp_path, safe_serialization=True)
    path = tmp_path / "model.safetensors"
    weights = load_file(path)
    del weights["model.layers.0.self_attn.q_proj.weight"]
    save_file(weights, path, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="incompatible Qwen3 weights"):
        QwenAgent.from_pretrained(tmp_path, dtype=torch.float32, n_local_ctx=16, exchange_every=1)


def test_bfloat16_training_and_optimizer_resume(tmp_path: Path) -> None:
    """The intended low-memory dtype keeps trainable adapters and Adam state in float32."""

    base = tmp_path / "base"
    backbone().save_pretrained(base, safe_serialization=True)
    model = QwenAgent.from_pretrained(
        base, n_windows=2, n_local_ctx=16, lora_rank=2, exchange_every=1, exchange_rank=4
    )
    assert model.backbone.model.embed_tokens.weight.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    model.open_window().expect([3, 4])
    model.open_window().expect([5, 6])
    assert all(torch.isfinite(torch.tensor(model.train_chunk())))
    QwenCheckpoint.save(model, tmp_path / "state.pt", "test")
    restored = QwenCheckpoint.load(tmp_path / "state.pt", base, "test")
    assert all(s["exp_avg"].dtype == torch.float32 for s in restored.optimizer.state.values())
    restored.open_window().expect([3, 4])
    restored.open_window().expect([5, 6])
    assert all(torch.isfinite(torch.tensor(restored.train_chunk())))


def test_cli_corpus_train_resume(tmp_path: Path) -> None:
    """Exercise the documented command path using only local synthetic model files."""

    import json
    import subprocess
    import sys

    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    from project_kidra.corpus.tokenizer import Tokenizer

    base = tmp_path / "base"
    backbone().save_pretrained(base, safe_serialization=True)
    tokenizer = HFTokenizer(
        WordLevel(
            {"<unk>": 0, "<start>": 1, "<end>": 2, "hello": 3, "world": 4, "café": 5},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.save(str(base / "tokenizer.json"))
    wrapped = Tokenizer(base)
    assert wrapped.vocab_size == 32  # The model has padded rows absent from tokenizer.json.
    assert wrapped.encode("café") == [5]
    assert wrapped.free_token == -1 and wrapped.eos_token == 2
    source = tmp_path / "samples.json"
    source.write_text(json.dumps({"windows": ["hello world café", "world"]}), encoding="utf-8")
    corpus = tmp_path / "corpus.kidra"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "project_kidra.build_corpus",
            str(source),
            "--output",
            str(corpus),
            "--pretrained",
            str(base),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    command = [
        sys.executable,
        "-m",
        "project_kidra",
        str(corpus),
        "--pretrained",
        str(base),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--windows",
        "2",
        "--local-context",
        "16",
        "--exchange-every",
        "1",
        "--tokens-per-step",
        "2",
        "--steps",
        "1",
        "--checkpoint",
        str(tmp_path / "adapters.pt"),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "Step: 1" in result.stdout
    result = subprocess.run([*command, "--resume"], check=True, capture_output=True, text=True)
    assert "Step: 1" in result.stdout
