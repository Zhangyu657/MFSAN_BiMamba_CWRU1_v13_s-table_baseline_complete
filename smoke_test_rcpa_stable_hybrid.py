# -*- coding: utf-8 -*-
"""CPU-only smoke tests for stable-hybrid RCPA components."""

import sys

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "./models")

from models.rcpa_components import (
    StableSourcePruningController,
    three_source_consensus_scores,
)
from models.MFSAN_BIMAMBA_RCPA_STABLE_HYBRID import DomainDiscriminator


def test_delayed_source_gate():
    gate = StableSourcePruningController(
        num_sources=3,
        enabled=True,
        warmup_epochs=3,
        start_epoch=4,
        confirm_epochs=3,
        release_epochs=3,
        confirm_gap=0.08,
        preconfirm_floor=0.10,
        bottom_floor=0.01,
        max_source_weight=0.65,
    )

    raw = torch.tensor([0.49, 0.48, 0.03])
    warmup = gate.apply(raw, epoch=2)
    assert torch.allclose(warmup, torch.full_like(warmup, 1 / 3), atol=1e-6)

    preconfirm = gate.apply(raw, epoch=4)
    assert float(preconfirm.min()) >= 0.099

    for epoch in (4, 5, 6):
        status = gate.update_epoch(raw, epoch)
    assert status["confirmed_index"] == 2

    confirmed = gate.apply(raw, epoch=7)
    assert abs(float(confirmed[2]) - 0.01) < 1e-5
    assert abs(float(confirmed.sum()) - 1.0) < 1e-5
    assert float(confirmed.max()) <= 0.65001


def test_consensus_scores():
    # Source 0 and source 1 agree; source 2 predicts the opposite class.
    p0 = torch.tensor([[0.95, 0.05], [0.90, 0.10], [0.92, 0.08]])
    p1 = torch.tensor([[0.93, 0.07], [0.91, 0.09], [0.94, 0.06]])
    p2 = torch.tensor([[0.05, 0.95], [0.10, 0.90], [0.08, 0.92]])
    score = three_source_consensus_scores([p0, p1, p2], confidence_threshold=0.55)
    assert score.shape == (3,)
    assert float(score[2]) < float(score[0])
    assert float(score[2]) < float(score[1])


def test_discriminator_backward():
    discriminator = DomainDiscriminator(input_dim=18, hidden_dim=16, dropout=0.0)
    x = torch.randn(12, 18, requires_grad=True)
    logits = discriminator(x)
    labels = torch.randint(0, 2, (12,))
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None


if __name__ == "__main__":
    test_delayed_source_gate()
    test_consensus_scores()
    test_discriminator_backward()
    print("All stable-hybrid RCPA smoke tests passed.")
