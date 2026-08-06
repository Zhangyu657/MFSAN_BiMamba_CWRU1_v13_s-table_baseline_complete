# -*- coding: utf-8 -*-
"""CPU-only tests for V7 class gate and conflict-aware fusion."""

import sys
sys.path.extend(['./models', './data_loader'])

import torch

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V7_CLASS_GATE_CONFLICT_FUSION import Trainer


def build_stub(num_classes=4):
    t = Trainer.__new__(Trainer)
    t.device = torch.device('cpu')
    t.num_source = 3
    t.num_classes = num_classes
    t.entropy_eps = 1e-8
    t.rw_detach_weights = True
    t.rw_eval_use_entropy = True
    t.rw_eval_tau = 0.5
    t.v6_source_names = ['PU_1', 'PU_2', 'PU_3']
    t.v6_class_weight_power = 1.0
    t.v6_gate_max_source_weight = 0.75

    t.v7_class_gate_enabled = True
    t.v7_class_gate_start_epoch = 4
    t.v7_class_gate_confirm_epochs = 2
    t.v7_class_gate_release_epochs = 3
    t.v7_class_gate_confirm_gap = 0.10
    t.v7_class_gate_release_gap = 0.04
    t.v7_class_gate_max_bad_weight = 0.20
    t.v7_class_gate_preconfirm_floor = 0.0
    t.v7_class_gate_bottom_floor = 0.01
    t._v7_class_candidate_source = torch.full((num_classes,), -1, dtype=torch.long)
    t._v7_class_candidate_streak = torch.zeros(num_classes, dtype=torch.long)
    t._v7_class_confirmed_source = torch.full((num_classes,), -1, dtype=torch.long)
    t._v7_class_release_streak = torch.zeros(num_classes, dtype=torch.long)
    t._v7_last_pre_gate_class_weights = torch.full((3, num_classes), 1 / 3)
    t._v7_pre_gate_class_weight_sum = None
    t._v7_pre_gate_class_weight_count = 0
    t._v3_collect_epoch_weights = False
    t._cw_log_classes = list(range(num_classes))

    t.v7_conflict_fusion_enabled = True
    t.v7_agree_prior_power = 1.0
    t.v7_conflict_prior_power = 0.30
    t.v7_conflict_top1_margin_bonus = 1.0
    t.v7_conflict_weight_temperature = 1.0
    t.class_source_weight_last_epoch = torch.full((3, num_classes), 1 / 3)
    t.class_source_weight_ema = t.class_source_weight_last_epoch.clone()
    t._v7_eval_total_samples = 0
    t._v7_eval_conflict_samples = 0
    t._v7_eval_changed_predictions = 0
    return t


def test_class_gate_preserves_specialist():
    t = build_stub(num_classes=4)
    # For class 0, src2 is weak. For class 3, src2 is the best specialist.
    cw = torch.tensor([
        [0.55, 0.45, 0.45, 0.10],
        [0.40, 0.45, 0.45, 0.15],
        [0.05, 0.10, 0.10, 0.75],
    ])
    for epoch in (4, 5):
        t._cur_epoch = epoch
        t._update_class_gate_from_epoch(cw)
    assert int(t._v7_class_confirmed_source[0]) == 2
    assert int(t._v7_class_confirmed_source[3]) != 2

    out = t._apply_stable_gate_matrix(cw)
    assert abs(float(out[2, 0]) - 0.01) < 1e-5
    assert float(out[2, 3]) > float(out[0, 3])
    assert torch.allclose(out.sum(dim=0), torch.ones(4), atol=1e-5)


def test_conflict_fusion_allows_confident_minority_expert():
    t = build_stub(num_classes=3)
    # src2 is a strong specialist for class 1, despite lower priors elsewhere.
    t.class_source_weight_last_epoch = torch.tensor([
        [0.48, 0.15, 0.48],
        [0.48, 0.15, 0.48],
        [0.04, 0.70, 0.04],
    ])
    probs = [
        torch.tensor([[0.51, 0.45, 0.04]]),
        torch.tensor([[0.52, 0.44, 0.04]]),
        torch.tensor([[0.03, 0.95, 0.02]]),
    ]
    fused, weights = t._eval_class_weighted_fusion(probs)
    assert fused.argmax(dim=1).item() == 1
    assert float(weights[2, 0, 1]) > float(weights[0, 0, 1])
    assert t._v7_eval_conflict_samples == 1


if __name__ == '__main__':
    test_class_gate_preserves_specialist()
    test_conflict_fusion_allows_confident_minority_expert()
    print('V7 class gate + conflict fusion smoke tests passed.')
