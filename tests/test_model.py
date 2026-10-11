"""Tests for the model module: config, cache, window, rope, router, expert, layer."""

import dataclasses

import pytest
import torch

from project_kidra.model.cache import KVCache
from project_kidra.model.config import Config
from project_kidra.model.expert import Expert
from project_kidra.model.layer import LayerBlock
from project_kidra.model.model import AgentModel
from project_kidra.model.rope import RopeEmbedding
from project_kidra.model.router import Router
from project_kidra.model.window import ContextWindow


def make_model() -> AgentModel:
    """Create a small model."""

    config = Config(
        vocab_size=16, d_model=16, d_hidden=16, n_layers=2, n_heads=2, n_windows=4, n_local_ctx=8
    )
    return AgentModel(config, free_token=15)


def make_window(n_local_ctx: int = 4, vocab_size: int = 10) -> ContextWindow:
    """Create a small context window."""

    return ContextWindow(
        index=0, id=0, n_layers=2, n_local_ctx=n_local_ctx, vocab_size=vocab_size, free_token=9
    )


# --- Config ---


class TestConfig:
    """Validation and serialization of Config."""

    def test_defaults_are_valid(self) -> None:
        """The default configuration constructs without error."""

        Config()

    def test_model_parameters_matches_fields(self) -> None:
        """model_parameters exposes every dataclass field."""

        config = Config()
        fields = {f.name for f in dataclasses.fields(config)}
        assert set(config.model_parameters()) == fields

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"expert_top_k": 0},
            {"expert_top_k": 5},
            {"vocab_size": 0},
            {"d_model": 0},
            {"d_hidden": 0},
            {"n_layers": 0},
            {"n_heads": 0},
            {"d_model": 64, "n_heads": 5},
            {"n_local_ctx": 0},
            {"n_windows": 0},
            {"n_global_layers": 0},
            {"n_layers": 4, "n_global_layers": 3},
            {"trunk_lr": 0.0},
            {"expert_lr": -1.0},
        ],
    )
    def test_invalid_values_rejected(self, kwargs: dict[str, int | float]) -> None:
        """Inconsistent or non-positive settings raise ValueError."""

        with pytest.raises(ValueError):
            Config(**kwargs)  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        """Config instances are immutable."""

        with pytest.raises(dataclasses.FrozenInstanceError):
            Config().d_model = 8  # type: ignore[misc]


# --- KVCache ---


class TestKVCache:
    """Sliding-window key/value cache."""

    def test_append_grows_and_truncates_to_limit(self) -> None:
        """Only the newest `limit` entries are kept."""

        cache = KVCache(limit=3)
        for i in range(6):
            cache.append(torch.full((1, 1, 1, 1), float(i)), torch.full((1, 1, 1, 1), float(i)))

        assert cache.length == 3
        assert cache.k.flatten().tolist() == [3.0, 4.0, 5.0]
        assert cache.v.flatten().tolist() == [3.0, 4.0, 5.0]

    def test_detach_removes_grad_history(self) -> None:
        """After detach, cached tensors no longer require grad."""

        cache = KVCache(limit=3)
        k = torch.ones(1, 1, 1, 1, requires_grad=True)
        cache.append(k * 2, k * 3)
        assert cache.k.requires_grad

        cache.detach()
        assert not cache.k.requires_grad
        assert not cache.v.requires_grad

    def test_first_append_replaces_placeholder(self) -> None:
        """The first append replaces the placeholder instead of concatenating with it."""

        cache = KVCache(limit=8)
        cache.append(torch.ones(1, 4, 2, 16), torch.ones(1, 4, 2, 16))
        assert cache.k.shape == (1, 4, 2, 16)
        assert cache.k.min() == 1.0

    def test_incompatible_shape_replaces_contents(self) -> None:
        """Keys with a different head/dim layout replace the cache rather than erroring."""

        cache = KVCache(limit=8)
        cache.append(torch.zeros(1, 4, 1, 16), torch.zeros(1, 4, 1, 16))
        cache.append(torch.zeros(1, 2, 3, 8), torch.zeros(1, 2, 3, 8))
        assert cache.k.shape == (1, 2, 3, 8)

    def test_oversized_append_keeps_newest(self) -> None:
        """Appending more than `limit` entries keeps only the newest ones."""

        cache = KVCache(limit=2)
        k = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1)
        cache.append(k, k)
        assert cache.k.flatten().tolist() == [3.0, 4.0]


# --- ContextWindow ---


class TestContextWindow:
    """Token bookkeeping for a context window."""

    def test_append_keeps_only_local_context(self) -> None:
        """Tokens beyond n_local_ctx are dropped from the front."""

        window = make_window(n_local_ctx=3)
        for t in range(6):
            window.append(t)
        assert window.tokens == [3, 4, 5]

    @pytest.mark.parametrize("token", [-1, 10])
    def test_append_rejects_out_of_vocab(self, token: int) -> None:
        """Out-of-range token IDs raise ValueError."""

        with pytest.raises(ValueError):
            make_window().append(token)

    def test_expect_and_pop_pending_in_order(self) -> None:
        """Pending tokens come out FIFO, then the free token."""

        window = make_window().expect([1, 2])
        assert [window.pop_pending() for _ in range(3)] == [1, 2, 9]

    def test_expect_rejects_out_of_vocab(self) -> None:
        """Pending tokens are validated against the vocabulary."""

        with pytest.raises(ValueError):
            make_window().expect([1, 10])

    def test_one_cache_per_layer(self) -> None:
        """A cache is created for each layer with the local context limit."""

        window = make_window(n_local_ctx=4)
        assert len(window.cache) == 2
        assert all(c.limit == 4 for c in window.cache)

    def test_detach_cache_detaches_all(self) -> None:
        """detach_cache clears gradient history on every layer cache."""

        window = make_window()
        for c in window.cache:
            c.append(torch.ones(1, 1, 1, 1, requires_grad=True) * 2, torch.ones(1, 1, 1, 1))
        window.detach_cache()
        assert not any(c.k.requires_grad for c in window.cache)

    def test_close_window_frees_id_and_reindexes(self) -> None:
        """Closing a window frees its ID and reindexes remaining windows."""

        model = make_model()
        assert len(model.available_window_ids) == 4

        window1 = model.open_window()
        window2 = model.open_window()
        assert len(model.available_window_ids) == 2

        model.close_window(window1)
        assert len(model.available_window_ids) == 3
        assert window2.index == 0
        assert window1.id in model.available_window_ids

    def test_open_window_raises_when_no_ids_available(self) -> None:
        """Opening a window with no available IDs raises RuntimeError."""

        model = make_model()
        for _ in range(4):
            model.open_window()

        assert len(model.available_window_ids) == 0

        with pytest.raises(RuntimeError):
            model.open_window()


# --- RopeEmbedding ---


class TestRope:
    """Rotary positional embeddings."""

    def test_odd_head_dim_rejected(self) -> None:
        """head_dim must be even."""

        with pytest.raises(ValueError):
            RopeEmbedding(3)

    def test_position_zero_is_identity(self) -> None:
        """No rotation is applied at position 0."""

        rope = RopeEmbedding(8)
        x = torch.randn(1, 2, 3, 8)
        assert torch.allclose(rope(x, 0), x)

    def test_preserves_shape_and_norm(self) -> None:
        """Rotation keeps shape and per-vector norm."""

        rope = RopeEmbedding(8)
        x = torch.randn(2, 4, 5, 8)
        positions = torch.arange(5).reshape(1, 1, 5)
        y = rope(x, positions)
        assert y.shape == x.shape
        assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)

    def test_relative_position_property(self) -> None:
        """Dot products depend only on the relative offset of positions."""

        rope = RopeEmbedding(8)
        q, k = torch.randn(8), torch.randn(8)
        a = rope(q, 3) @ rope(k, 1)
        b = rope(q, 10) @ rope(k, 8)
        assert torch.allclose(a, b, atol=1e-4)


# --- Router ---


class TestRouter:
    """Top-k gating."""

    def test_output_shapes_and_normalization(self) -> None:
        """Indices/weights have top_k columns and weights sum to one."""

        router = Router(d_model=8, n_experts=4, top_k=2)
        out = router(torch.randn(5, 8))

        assert out.indices.shape == (5, 2)
        assert out.weights.shape == (5, 2)
        assert torch.allclose(out.weights.sum(dim=-1), torch.ones(5))
        assert out.aux_loss.dim() == 0

    def test_indices_in_range_and_distinct(self) -> None:
        """Chosen experts are valid IDs and unique per token."""

        router = Router(d_model=8, n_experts=4, top_k=3)
        idx = router(torch.randn(10, 8)).indices
        assert idx.min() >= 0 and idx.max() < 4
        assert all(len(set(row.tolist())) == 3 for row in idx)

    def test_aux_loss_is_differentiable(self) -> None:
        """The load-balancing loss backpropagates into the gate."""

        router = Router(d_model=8, n_experts=4, top_k=2)
        router(torch.randn(6, 8)).aux_loss.backward()
        assert router.gate.weight.grad is not None


# --- Expert ---


class TestExpert:
    """Feed-forward expert."""

    def test_preserves_shape(self) -> None:
        """Output has the same (tokens, d_model) shape as the input."""

        assert Expert(8, 16)(torch.randn(5, 8)).shape == (5, 8)


# --- LayerBlock ---


class TestLayerBlock:
    """Sparse MoE block."""

    def test_forward_shape_aux_loss_and_cache(self) -> None:
        """forward returns a residual output of the input shape, a scalar loss, and fills the cache."""

        block = LayerBlock(d_model=8, d_hidden=16, n_experts=4, top_k=2, n_heads=2)
        cache = KVCache(limit=4)
        x = torch.randn(1, 1, 8)
        y, aux = block(x, [cache], [0])
        assert y.shape == x.shape
        assert aux.dim() == 0
        assert cache.k.shape == (1, 2, 1, 4)

    def test_forward_backpropagates_to_attention_and_experts(self) -> None:
        """Gradients reach the attention projections, router, and experts."""

        block = LayerBlock(d_model=8, d_hidden=16, n_experts=2, top_k=2, n_heads=2)
        y, aux = block(torch.randn(1, 1, 8), [KVCache(limit=4)], [0])
        (y.sum() + aux).backward()
        assert block.qkv.weight.grad is not None
        assert block.router.gate.weight.grad is not None
        assert any(p.grad is not None for p in block.experts.parameters())

    def test_attention_output_reaches_result(self) -> None:
        """The attention residual is not dropped before the expert stage."""

        block = LayerBlock(d_model=8, d_hidden=16, n_experts=2, top_k=2, n_heads=2)
        x = torch.randn(1, 1, 8)
        y1, _ = block(x, [KVCache(limit=4)], [0])
        block.out.weight.data.mul_(5.0)
        y2, _ = block(x, [KVCache(limit=4)], [0])
        assert not torch.allclose(y1, y2)

    def test_batches_windows_with_different_cache_lengths(self) -> None:
        """Batched attention keeps each window's cache length independent."""

        block = LayerBlock(d_model=8, d_hidden=16, n_experts=2, top_k=2, n_heads=2)
        caches = [KVCache(limit=4), KVCache(limit=4)]
        caches[0].append(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))
        caches[1].append(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))

        output, aux = block(torch.randn(2, 1, 8), caches, [0, 2])

        assert output.shape == (2, 1, 8)
        assert aux.dim() == 0
        assert [cache.length for cache in caches] == [2, 4]


# --- AgentModel ---


class TestAgentModel:
    """Window management on the full model."""

    def test_open_window_registers_window(self) -> None:
        """open_window returns a window and consumes its ID."""

        model = AgentModel(Config(n_windows=2, d_model=8, n_heads=2, d_hidden=8), free_token=0)
        window = model.open_window()

        assert model.windows == [window]
        assert window.id not in model.available_window_ids
        assert len(model.available_window_ids) == 1

    def test_forward_without_windows_returns_empty(self) -> None:
        """With no open windows there is nothing to train on."""

        model = AgentModel(Config(d_model=8, n_heads=2, d_hidden=8), free_token=0)
        assert model() == []

    def test_forward_batches_active_windows(self) -> None:
        """All active windows are processed together through local and global layers."""

        model = AgentModel(
            Config(
                vocab_size=8,
                d_model=8,
                d_hidden=8,
                n_layers=2,
                n_heads=2,
                n_windows=2,
                n_local_ctx=4,
            ),
            free_token=0,
        )
        windows = [model.open_window(), model.open_window()]
        windows[0].expect([1])
        windows[1].expect([2])

        losses = model()

        assert len(losses) == 2
        assert all(window.position == 1 for window in windows)
        assert all(cache.length == 1 for window in windows for cache in window.cache)
