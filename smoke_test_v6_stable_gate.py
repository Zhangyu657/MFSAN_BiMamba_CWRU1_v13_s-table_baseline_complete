# -*- coding: utf-8 -*-
"""CPU-only unit checks for V6 gate utilities."""

import sys
sys.path.extend(['./models', './data_loader'])

import torch

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V6_STABLE_GATE import Trainer
from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V5_MCA import SupConLoss


def build_stub():
    t = Trainer.__new__(Trainer)
    t.device = torch.device('cpu')
    t.num_source = 3
    t.num_classes = 9
    t.entropy_eps = 1e-8
    t.rw_detach_weights = True
    t.v6_source_names = ['PU_1', 'PU_2', 'PU_3']
    t.v6_gate_enabled = True
    t.v6_gate_start_epoch = 4
    t.v6_gate_confirm_epochs = 3
    t.v6_gate_release_epochs = 3
    t.v6_gate_confirm_gap = 0.08
    t.v6_gate_release_gap = 0.03
    t.v6_gate_preconfirm_floor = 0.05
    t.v6_gate_bottom_floor = 0.01
    t.v6_gate_max_source_weight = 0.75
    t.v6_gate_apply_to_supcon = True
    t.v6_supcon_source_min_weight = 0.05
    t.v6_class_weight_power = 1.20
    t._v6_candidate_source = -1
    t._v6_candidate_streak = 0
    t._v6_confirmed_negative_source = -1
    t._v6_release_streak = 0
    t._v6_last_raw_global_weights = torch.tensor([1/3, 1/3, 1/3])
    t._v6_last_effective_global_weights = t._v6_last_raw_global_weights.clone()
    t._v6_active_supcon_source_weights = t._v6_last_raw_global_weights.clone()
    t.lambda_supcon = 0.01
    t.supcon_start_epoch = 3
    t.supcon_feature_mode = 'G'
    t.supcon_focus_classes = [1, 3, 8]
    t.supcon_loss = SupConLoss(temperature=0.20)
    return t


def test_stable_confirmation():
    t = build_stub()
    raw = torch.tensor([0.49, 0.48, 0.03])
    for epoch in (4, 5, 6):
        t._cur_epoch = epoch
        t._update_stable_gate_from_epoch(raw)
    assert t._v6_confirmed_negative_source == 2
    effective = t._apply_stable_gate_vector(raw)
    assert abs(float(effective[2]) - 0.01) < 1e-5
    assert abs(float(effective.sum()) - 1.0) < 1e-5


def test_class_difference_is_preserved():
    t = build_stub()
    t._v6_confirmed_negative_source = 2
    t._cur_epoch = 8
    t.num_classes = 3
    cw = torch.tensor([
        [0.70, 0.35, 0.60],
        [0.25, 0.60, 0.35],
        [0.05, 0.05, 0.05],
    ])
    out = t._apply_stable_gate_matrix(cw)
    assert torch.allclose(out[2], torch.full((3,), 0.01), atol=1e-5)
    assert out[0, 0] > out[1, 0]
    assert out[1, 1] > out[0, 1]
    assert torch.allclose(out.sum(dim=0), torch.ones(3), atol=1e-5)


def test_supcon_excludes_confirmed_source():
    t = build_stub()
    t._v6_confirmed_negative_source = 2
    t._cur_epoch = 5
    # Only source 2 contains class 8. If it is correctly excluded, the remaining
    # batch still has valid positive pairs for classes 1 and 3 and returns finite loss.
    g = [
        torch.randn(6, 12, requires_grad=True),
        torch.randn(6, 12, requires_grad=True),
        torch.randn(6, 12, requires_grad=True),
    ]
    labels = [
        torch.tensor([1, 1, 3, 3, 0, 0]),
        torch.tensor([1, 1, 3, 3, 4, 4]),
        torch.tensor([8, 8, 8, 8, 8, 8]),
    ]
    loss = t._compute_source_supcon_loss(g, g, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert g[0].grad is not None and g[1].grad is not None
    assert g[2].grad is None


if __name__ == '__main__':
    test_stable_confirmation()
    test_class_difference_is_preserved()
    test_supcon_excludes_confirmed_source()
    print('V6 StableGate smoke tests passed.')
