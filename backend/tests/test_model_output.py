from __future__ import annotations

import torch
import pytest

from src.models.eeg_encoder import EEGEncoderModel
from src.training.callbacks import EarlyStopping


_SMALL_CFG = {
    "model": {
        "embedding_dim": 256,
        "dropout": 0.1,
        "f1": 4,
        "d": 2,
        "f2": 8,
        "temporal_kernel": 16,
    }
}


def test_model_output_shapes_and_l2_norm():
    model = EEGEncoderModel(n_channels=19, n_samples=256, config=_SMALL_CFG)
    out = model(torch.randn(3, 1, 19, 256))
    assert out["logits"].shape == (3, 2)
    assert out["probabilities"].shape == (3, 2)
    assert out["ad_probability"].shape == (3,)
    assert out["z_eeg"].shape == (3, 256)
    norms = torch.linalg.norm(out["z_eeg"], dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"z_eeg L2 norm not 1.0: {norms.tolist()}"


def test_model_probabilities_sum_to_one():
    model = EEGEncoderModel(n_channels=19, n_samples=256, config=_SMALL_CFG)
    out = model(torch.randn(4, 1, 19, 256))
    prob_sums = out["probabilities"].sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(4), atol=1e-5), \
        f"Probabilities do not sum to 1: {prob_sums.tolist()}"


def test_ad_probability_matches_probabilities():
    model = EEGEncoderModel(n_channels=19, n_samples=256, config=_SMALL_CFG)
    out = model(torch.randn(2, 1, 19, 256))
    assert torch.allclose(out["ad_probability"], out["probabilities"][:, 1], atol=1e-6)


def test_model_rejects_mismatched_epoch_shape():
    model = EEGEncoderModel(n_channels=19, n_samples=256, config=_SMALL_CFG)
    with pytest.raises(ValueError, match="does not match the trained encoder"):
        model(torch.randn(2, 1, 19, 255))


def test_projection_head_receives_gradient():
    model = EEGEncoderModel(n_channels=19, n_samples=256, config=_SMALL_CFG)
    x = torch.randn(4, 1, 19, 256)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    out = model(x)
    loss = torch.nn.CrossEntropyLoss()(out["logits"], y)
    loss.backward()

    grads = [p.grad for p in model.projection.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads), (
        "Projection head is not receiving gradient; z_eeg is not being trained."
    )


def test_early_stopping_returns_tuple():
    """EarlyStopping.step() must return (should_stop, is_best) — not a single bool."""
    stopper = EarlyStopping(patience=2)
    result = stopper.step(0.5)
    assert isinstance(result, tuple) and len(result) == 2, \
        "EarlyStopping.step() must return (should_stop, is_best)"
    should_stop, is_best = result
    assert is_best is True  # first call is always best
    assert should_stop is False


def test_early_stopping_saves_best_and_stops():
    stopper = EarlyStopping(patience=2)
    _, is_best = stopper.step(0.5)
    assert is_best
    _, is_best = stopper.step(0.4)  # improved
    assert is_best
    _, is_best = stopper.step(0.6)  # worse
    assert not is_best
    should_stop, is_best = stopper.step(0.7)  # worse again — hits patience=2
    assert should_stop
    assert not is_best
