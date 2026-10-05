import pytest
import torch

from project_kidra.moe import (
    CausalSelfAttention,
    MoEConfig,
    MoELayer,
    MoEModel,
    Router,
)
from project_kidra.training import Trainer

CFG = MoEConfig(
    vocab_size=16, d_model=16, d_hidden=32, num_experts=4, top_k=2, num_layers=2
)


def test_config_rejects_bad_top_k() -> None:
    """top_k larger than num_experts is invalid."""
    with pytest.raises(ValueError):
        MoEConfig(num_experts=2, top_k=3)


def test_config_rejects_invalid_attention_dimensions() -> None:
    """Attention dimensions and context window must be valid."""
    with pytest.raises(ValueError):
        MoEConfig(d_model=15, num_heads=4)
    with pytest.raises(ValueError):
        MoEConfig(context_window=0)


def test_attention_masks_future_and_out_of_window_tokens() -> None:
    """Changing future or too-distant tokens cannot affect a token's output."""
    torch.manual_seed(0)
    attention = CausalSelfAttention(d_model=8, num_heads=2, context_window=2).eval()
    original = torch.randn(1, 5, 8)
    changed = original.clone()
    changed[0, 0] += 100
    changed[0, 4] += 100
    baseline = attention(original)
    revised = attention(changed)
    assert torch.allclose(baseline[0, 2], revised[0, 2])
    assert torch.allclose(baseline[0, 3], revised[0, 3])


def test_router_weights_normalized_and_aux_positive() -> None:
    """Gate weights sum to 1 per token and the aux loss is positive."""
    out = Router(16, 4, 2)(torch.randn(10, 16))
    assert out.indices.shape == (10, 2)
    assert torch.allclose(out.weights.sum(-1), torch.ones(10))
    assert out.aux_loss.item() > 0


def test_layer_preserves_shape() -> None:
    """The MoE layer maps (B, T, D) to (B, T, D)."""
    out, aux = MoELayer(16, 32, 4, 2)(torch.randn(3, 5, 16))
    assert out.shape == (3, 5, 16)
    assert aux.dim() == 0


def test_model_output_shape() -> None:
    """The model returns per-token vocabulary logits."""
    logits, _ = MoEModel(CFG)(torch.randint(0, 16, (2, 7)))
    assert logits.shape == (2, 7, 16)


def test_model_accepts_sequences_longer_than_context_window() -> None:
    """Sliding attention works on sequences longer than the local context."""
    model = MoEModel(
        MoEConfig(
            vocab_size=16,
            d_model=16,
            d_hidden=32,
            num_experts=4,
            top_k=2,
            num_layers=1,
            context_window=3,
        )
    )
    logits, _ = model(torch.randint(0, 16, (2, 9)))
    assert logits.shape == (2, 9, 16)


def test_training_reduces_loss_and_updates_router() -> None:
    """Training on a fixed batch lowers loss and sends gradients to the router."""
    torch.manual_seed(0)
    model = MoEModel(CFG)
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=1e-2))
    x = torch.randint(0, 16, (4, 8))
    y = (x + 1) % 16
    before = trainer.evaluate(x, y).loss
    for _ in range(50):
        trainer.train_step(x, y)
    assert trainer.evaluate(x, y).loss < before
    assert model.layers[0].router.gate.weight.grad is not None  # type: ignore[union-attr]
