# -*- coding: utf-8 -*-
"""CPU-only smoke tests for the V11 CWRU_1 class-rescue mechanisms."""

import sys
sys.path.extend(['./models', './data_loader'])

import torch

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V11_CWRU1_CLASS_RESCUE import Trainer
from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V10_PAIRWISE_SPECIALIST_CALIBRATED import (
    Trainer as V10Trainer,
)


def build_stub():
    t = Trainer.__new__(Trainer)
    t.device = torch.device('cpu')
    t.num_source = 3
    t.num_classes = 10
    t.entropy_eps = 1e-8
    t.v11_cwru1_rescue_enabled = True
    t.v11_rescue_class = 2
    t.v11_confusion_classes = [0, 1]
    t.v11_rescue_topk = 2
    t.v11_eval_rescue_enabled = True
    t.v11_eval_min_class_prob = 0.08
    t.v11_eval_competitor_ratio = 0.35
    t.v11_eval_min_source_votes = 2
    t.v11_eval_boost = 2.0
    t._v11_eval_total = 0
    t._v11_eval_candidates = 0
    t._v11_eval_changed = 0
    return t


def test_eval_rescue_changes_only_supported_ball_confusion():
    t = build_stub()
    parent = V10Trainer._eval_class_weighted_fusion
    try:
        def fake_parent(self, probs_list):
            fused = torch.tensor([
                [0.44, 0.20, 0.23, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                [0.65, 0.10, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02],
            ])
            weights = torch.ones(3, 2, 10) / 3.0
            return fused, weights
        V10Trainer._eval_class_weighted_fusion = fake_parent

        probs = [
            torch.tensor([
                [0.40, 0.10, 0.35, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                [0.70, 0.08, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02],
            ]),
            torch.tensor([
                [0.35, 0.20, 0.32, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01],
                [0.68, 0.10, 0.05, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01],
            ]),
            torch.tensor([
                [0.48, 0.18, 0.20, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01],
                [0.66, 0.12, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02],
            ]),
        ]
        calibrated, _ = t._eval_class_weighted_fusion(probs)
    finally:
        V10Trainer._eval_class_weighted_fusion = parent

    assert int(calibrated[0].argmax().item()) == 2
    assert int(calibrated[1].argmax().item()) == 0
    assert t._v11_eval_candidates == 1
    assert t._v11_eval_changed == 1


if __name__ == '__main__':
    test_eval_rescue_changes_only_supported_ball_confusion()
    print('V11 CWRU1 class rescue smoke tests passed.')
