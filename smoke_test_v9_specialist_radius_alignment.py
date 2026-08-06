# -*- coding: utf-8 -*-
"""CPU-only smoke tests for V9 specialist and radius mechanisms."""

import sys
sys.path.extend(['./models', './data_loader'])

import torch

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V9_SPECIALIST_RADIUS_ALIGNMENT import Trainer


def build_stub(num_source=3, num_classes=2):
    t = Trainer.__new__(Trainer)
    t.device = torch.device('cpu')
    t.num_source = num_source
    t.num_classes = num_classes
    t.entropy_eps = 1e-8
    t._cur_epoch = 8
    t.v6_source_names = [f'src{i}' for i in range(num_source)]

    # Generic reliability/gate helpers.
    t.rw_detach_weights = True
    t.v6_class_weight_power = 1.0
    t.v6_gate_max_source_weight = 0.75
    t.v7_class_gate_enabled = True
    t.v7_class_gate_start_epoch = 4
    t.v7_class_gate_confirm_epochs = 1
    t.v7_class_gate_release_epochs = 3
    t.v7_class_gate_confirm_gap = 0.10
    t.v7_class_gate_release_gap = 0.04
    t.v7_class_gate_max_bad_weight = 0.20
    t.v7_class_gate_preconfirm_floor = 0.005
    t.v7_class_gate_bottom_floor = 0.01
    t._cw_log_classes = list(range(num_classes))
    t._v3_collect_epoch_weights = False
    t._v7_pre_gate_class_weight_sum = None
    t._v7_pre_gate_class_weight_count = 0
    t._v7_last_pre_gate_class_weights = torch.full(
        (num_source, num_classes), 1.0 / num_source
    )
    t._v7_active_supcon_class_weights = t._v7_last_pre_gate_class_weights.clone()
    t._v7_class_candidate_source = torch.full((num_classes,), -1, dtype=torch.long)
    t._v7_class_candidate_streak = torch.zeros(num_classes, dtype=torch.long)
    t._v7_class_confirmed_source = torch.full((num_classes,), -1, dtype=torch.long)
    t._v7_class_release_streak = torch.zeros(num_classes, dtype=torch.long)

    # V9 specialist state.
    t.v9_specialist_protection_enabled = True
    t.v9_specialist_start_epoch = 4
    t.v9_specialist_confirm_epochs = 2
    t.v9_specialist_min_weight = 0.45
    t.v9_specialist_min_gap = 0.10
    t.v9_specialist_floor = 0.05
    t.v9_specialist_release_weight = 0.15
    t.v9_specialist_release_epochs = 4
    t._v9_specialist_candidate_source = torch.full((num_classes,), -1, dtype=torch.long)
    t._v9_specialist_candidate_streak = torch.zeros(num_classes, dtype=torch.long)
    t._v9_protected_specialist_source = torch.full((num_classes,), -1, dtype=torch.long)
    t._v9_specialist_release_streak = torch.zeros(num_classes, dtype=torch.long)

    # SupCon.
    t.lambda_supcon = 0.01
    t.supcon_start_epoch = 3
    t.supcon_feature_mode = 'G'
    t.supcon_temperature = 0.2
    t.v6_gate_apply_to_supcon = False
    t.v7_supcon_class_min_weight = 0.02
    t.v8_hard_supcon_enabled = True
    t.v8_hard_negative_pairs = {(0, 1)}
    t.v8_hard_negative_weight = 1.4
    t.v8_supcon_anchor_classes = None
    t.v9_hard_supcon_start_epoch = 8
    t.v9_hard_supcon_ramp_epochs = 4

    # CLMMD/prototype.
    t.use_pl_conf_gate = True
    t.pl_conf_thresh = 0.8
    t.pl_min_target = 1
    t.clmmd_min_source = 1
    t.clmmd_min_target_weight = 1e-3
    t.clmmd_kernel_num = 3
    t.clmmd_kernel_mul = 2.0
    t.v8_prototype_filter_enabled = True
    t.v8_prototype_start_epoch = 8
    t.v8_prototype_ema_momentum = 0.0
    t.v8_prototype_margin = 0.05
    t.v8_prototype_min_updates = 1
    t.v8_prototype_conf_thresholds = torch.tensor([0.8] * num_classes)
    t.v8_clmmd_class_boost = torch.ones(num_classes)
    t.v9_prototype_filter_classes = list(range(num_classes))
    t.v9_prototype_filter_class_set = set(range(num_classes))
    t.v9_radius_ema_momentum = 0.0
    t.v9_radius_std_scale = 1.0
    t.v9_radius_min = 0.01
    t.v9_radius_max = 0.08
    t.v9_prototype_min_similarity = 0.0
    t.v9_prototype_soft_tau = 0.10
    t._v8_source_class_prototypes = None
    t._v8_source_class_prototype_updates = torch.zeros(
        num_source, num_classes, dtype=torch.long
    )
    t._v9_radius_mean = torch.zeros(num_source, num_classes)
    t._v9_radius_var = torch.zeros(num_source, num_classes)
    t._v9_radius_updates = torch.zeros(
        num_source, num_classes, dtype=torch.long
    )
    t._reset_v8_prototype_epoch_stats()
    t._active_source_idx_for_clmmd = 0
    return t


def test_specialist_is_protected():
    t = build_stub()
    # Source 2 is a clear specialist for class 1 for two consecutive epochs.
    weights = torch.tensor([
        [0.45, 0.15],
        [0.45, 0.15],
        [0.10, 0.70],
    ])
    t._update_class_gate_from_epoch(weights)
    t._update_class_gate_from_epoch(weights)
    assert int(t._v9_protected_specialist_source[1].item()) == 2

    # Even if stale V7 state marks source 2 as bad, protection clears it and
    # the applied class weight cannot fall below the specialist floor.
    t._v7_class_confirmed_source[1] = 2
    out = t._apply_stable_gate_matrix(weights)
    assert float(out[2, 1]) >= t.v9_specialist_floor - 1e-6


def test_hard_weight_is_delayed_and_ramped():
    t = build_stub()
    t._cur_epoch = 7
    assert abs(t._current_v9_hard_weight() - 1.0) < 1e-8
    t._cur_epoch = 8
    assert 1.0 < t._current_v9_hard_weight() < t.v8_hard_negative_weight
    t._cur_epoch = 20
    assert abs(t._current_v9_hard_weight() - t.v8_hard_negative_weight) < 1e-8


def test_radius_filter_rejects_far_same_prototype_sample():
    t = build_stub(num_source=3, num_classes=2)
    # Tight source clusters create a small source-derived class radius.
    f_s = torch.tensor([
        [1.00, 0.00],
        [0.99, 0.01],
        [0.00, 1.00],
        [0.01, 0.99],
    ])
    labels_s = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    # Both are predicted as class 0. Sample 0 is close to class 0. Sample 1 is
    # still nearest to class 0 but lies outside the compact source radius.
    f_t = torch.tensor([
        [0.999, 0.001],
        [0.72, 0.28],
    ])
    probs_t = torch.tensor([
        [0.98, 0.02],
        [0.95, 0.05],
    ])

    loss_vec, valid_vec = t._classwise_lmmd_per_class(
        f_s, f_t, labels_s, probs_t
    )
    assert torch.isfinite(loss_vec).all()
    assert int(valid_vec[0].item()) == 1
    assert int(t._v8_proto_conf_candidates[0, 0].item()) == 2
    assert int(t._v8_proto_accepted[0, 0].item()) == 1
    assert int(t._v9_proto_reject_radius[0, 0].item()) == 1


if __name__ == '__main__':
    test_specialist_is_protected()
    test_hard_weight_is_delayed_and_ramped()
    test_radius_filter_rejects_far_same_prototype_sample()
    print('V9 specialist + radius-alignment smoke tests passed.')
