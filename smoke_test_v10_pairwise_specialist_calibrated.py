# -*- coding: utf-8 -*-
"""CPU-only smoke tests for V10 targeted mechanisms."""

import sys
sys.path.extend(['./models', './data_loader'])

import torch

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V10_PAIRWISE_SPECIALIST_CALIBRATED import Trainer
from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V9_SPECIALIST_RADIUS_ALIGNMENT import Trainer as V9Trainer


def build_stub(num_source=3, num_classes=5):
    t = Trainer.__new__(Trainer)
    t.device = torch.device('cpu')
    t.num_source = num_source
    t.num_classes = num_classes
    t.entropy_eps = 1e-8
    t._cur_epoch = 8
    t.v6_source_names = [f'src{i}' for i in range(num_source)]

    t.v9_specialist_protection_enabled = True
    t.v9_specialist_start_epoch = 4
    t.v9_specialist_confirm_epochs = 2
    t.v9_specialist_min_weight = 0.45
    t.v9_specialist_min_gap = 0.10
    t.v9_specialist_floor = 0.05
    t.v9_specialist_release_weight = 0.15
    t.v9_specialist_release_epochs = 4
    t.v10_specialist_switch_enabled = True
    t.v10_specialist_switch_epochs = 2
    t.v10_specialist_switch_min_weight = 0.55
    t.v10_specialist_switch_min_gap = 0.15
    t._v9_specialist_candidate_source = torch.full((num_classes,), -1, dtype=torch.long)
    t._v9_specialist_candidate_streak = torch.zeros(num_classes, dtype=torch.long)
    t._v9_protected_specialist_source = torch.full((num_classes,), -1, dtype=torch.long)
    t._v9_specialist_release_streak = torch.zeros(num_classes, dtype=torch.long)
    t._v10_last_specialist_events = []

    t.v8_hard_supcon_enabled = True
    t.v9_hard_supcon_start_epoch = 8
    t.v9_hard_supcon_ramp_epochs = 4
    t.v10_hard_pair_weights = {(0, 3): 1.10, (3, 4): 1.30}

    t.v9_radius_std_scale = 2.0
    t.v9_radius_min = 0.03
    t.v9_radius_max = 0.30
    t._v9_radius_mean = torch.zeros(num_source, num_classes)
    t._v9_radius_var = torch.zeros(num_source, num_classes)
    t.v10_radius_class_min = torch.tensor([0.025, 0.03, 0.03, 0.025, 0.025])
    t.v10_radius_class_max = torch.tensor([0.04, 0.30, 0.30, 0.05, 0.05])
    t.v10_radius_weak_source_threshold = 0.05
    t.v10_radius_weak_source_cap_scale = 0.80
    t._v7_active_supcon_class_weights = torch.full(
        (num_source, num_classes), 1.0 / num_source
    )
    return t


def test_specialist_can_switch():
    t = build_stub()
    # First protect source 0 for class 3.
    w0 = torch.tensor([
        [0.40, 0.40, 0.40, 0.70, 0.40],
        [0.35, 0.35, 0.35, 0.20, 0.35],
        [0.25, 0.25, 0.25, 0.10, 0.25],
    ])
    t._update_v9_specialist_memory(w0)
    t._update_v9_specialist_memory(w0)
    assert int(t._v9_protected_specialist_source[3].item()) == 0

    # Then source 2 becomes clearly dominant and must replace stale source 0.
    w2 = torch.tensor([
        [0.35, 0.40, 0.40, 0.15, 0.40],
        [0.35, 0.35, 0.35, 0.15, 0.35],
        [0.30, 0.25, 0.25, 0.70, 0.25],
    ])
    t._update_v9_specialist_memory(w2)
    t._update_v9_specialist_memory(w2)
    assert int(t._v9_protected_specialist_source[3].item()) == 2


def test_pair_specific_ramp():
    t = build_stub()
    t._cur_epoch = 7
    m = t._build_hard_pair_matrix(torch.tensor([0, 3, 4]))
    assert abs(float(m[0, 1]) - 1.0) < 1e-8
    t._cur_epoch = 20
    m = t._build_hard_pair_matrix(torch.tensor([0, 3, 4]))
    assert abs(float(m[0, 1]) - 1.10) < 1e-6
    assert abs(float(m[1, 2]) - 1.30) < 1e-6


def test_class_radius_cap_and_weak_source_tightening():
    t = build_stub()
    # Make source 2/class 3 diffuse enough to exceed the configured cap.
    t._v9_radius_mean[2, 3] = 0.04
    t._v9_radius_var[2, 3] = 0.01 ** 2
    t._v7_active_supcon_class_weights[2, 3] = 0.01
    threshold = float(t._v9_radius_threshold(2, 3).item())
    # class cap 0.05 * weak-source scale 0.8 = 0.04
    assert abs(threshold - 0.04) < 1e-6


def test_normal_guard_changes_low_confidence_normal_only():
    t = build_stub(num_source=3, num_classes=5)
    t.v10_normal_guard_enabled = True
    t.v10_normal_class = 0
    t.v10_normal_min_prob = 0.80
    t.v10_normal_guard_min_fault_prob = 0.05
    t._v10_guard_total = 0
    t._v10_guard_changed = 0

    original = V9Trainer._eval_class_weighted_fusion
    try:
        def fake_parent(self, probs_list):
            fused = torch.tensor([
                [0.70, 0.20, 0.05, 0.03, 0.02],
                [0.90, 0.05, 0.02, 0.02, 0.01],
            ])
            weights = torch.ones(3, 2, 5) / 3.0
            return fused, weights
        V9Trainer._eval_class_weighted_fusion = fake_parent
        guarded, _ = t._eval_class_weighted_fusion([])
    finally:
        V9Trainer._eval_class_weighted_fusion = original

    assert int(guarded[0].argmax().item()) == 1
    assert int(guarded[1].argmax().item()) == 0
    assert t._v10_guard_changed == 1


if __name__ == '__main__':
    test_specialist_can_switch()
    test_pair_specific_ramp()
    test_class_radius_cap_and_weak_source_tightening()
    test_normal_guard_changes_low_confidence_normal_only()
    print('V10 pairwise specialist calibrated smoke tests passed.')
