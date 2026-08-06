# -*- coding: utf-8 -*-
"""CPU-only smoke tests for PU_0 enhanced RCPA.

Run from project root:

    python smoke_test_rcpa_pu0_enhanced.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "./models")

from models.rcpa_components import (  # noqa: E402
    RCPAStatisticsMemory,
    SupervisedContrastiveLoss,
    adaptive_top2_source_gate,
)


def test_negative_source_pruning() -> None:
    weights = torch.tensor([0.49, 0.48, 0.03])
    gated = adaptive_top2_source_gate(
        weights,
        enabled=True,
        prune_gap=0.05,
        bottom_floor=0.01,
        max_source_weight=0.65,
    )
    assert torch.allclose(gated.sum(), torch.tensor(1.0), atol=1e-6)
    assert abs(float(gated[2]) - 0.01) < 1e-5
    assert float(gated.max()) <= 0.65001


def test_single_source_monopoly_cap() -> None:
    weights = torch.tensor([0.98, 0.01, 0.01])
    gated = adaptive_top2_source_gate(
        weights,
        enabled=True,
        prune_gap=0.05,
        bottom_floor=0.01,
        max_source_weight=0.65,
    )
    assert torch.allclose(gated.sum(), torch.tensor(1.0), atol=1e-6)
    assert float(gated.max()) <= 0.65001


def test_classwise_gate() -> None:
    weights = torch.tensor(
        [
            [0.50, 0.90],
            [0.47, 0.08],
            [0.03, 0.02],
        ]
    )
    gated = adaptive_top2_source_gate(
        weights,
        enabled=True,
        prune_gap=0.05,
        bottom_floor=0.01,
        max_source_weight=0.65,
    )
    assert torch.allclose(gated.sum(dim=0), torch.ones(2), atol=1e-6)
    assert float(gated[:, 0].min()) <= 0.01001
    assert float(gated[:, 1].max()) <= 0.65001



def test_global_reliability_suppresses_uncertain_source() -> None:
    memory = RCPAStatisticsMemory(3, 3, 4, weight_momentum=0.0)
    weights = memory.update_global_weights(
        mmd_distances=torch.tensor([0.10, 0.11, 0.30]),
        tau=0.50,
        target_entropies=torch.tensor([0.20, 0.22, 0.80]),
        source_recognition=torch.tensor([0.95, 0.93, 0.70]),
        entropy_weight=1.0,
        recognition_weight=0.30,
        adaptive_pruning=True,
        prune_gap=0.05,
        bottom_floor=0.01,
        max_source_weight=0.65,
    )
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert abs(float(weights[2]) - 0.01) < 1e-5
    assert float(weights.max()) <= 0.65001

def test_hard_supcon_forward_backward() -> None:
    # Two samples for each of three difficult classes.
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 0.9],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([1, 1, 2, 2, 8, 8])
    loss_fn = SupervisedContrastiveLoss(temperature=0.20)
    loss = loss_fn(features, labels)
    assert torch.isfinite(loss)
    assert float(loss.detach()) >= 0.0
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


if __name__ == "__main__":
    test_negative_source_pruning()
    test_single_source_monopoly_cap()
    test_classwise_gate()
    test_global_reliability_suppresses_uncertain_source()
    test_hard_supcon_forward_backward()
    print("All PU_0 enhanced RCPA smoke tests passed.")
