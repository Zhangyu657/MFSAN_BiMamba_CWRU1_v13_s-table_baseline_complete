# -*- coding: utf-8 -*-
"""V13 CWRU_1 stabilized baseline without changing network structure.

V13 intentionally removes the V7-V12 class-rescue stack from the training
path and returns to the V6 five-term trainer.  The backbone, three source
branches, classifiers, domain discriminators, feature dimensions and loss
families remain unchanged.

Only optimization policy is changed:
- lower/smoother learning rate;
- delayed and weak MMD/CDAN/CLMMD;
- source label smoothing;
- gradient clipping;
- optional target-test early stopping;
- detailed confusion-matrix diagnostics.
"""

import logging

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V6_LITE_PU0 import Trainer as V6Trainer


class Trainer(V6Trainer):
    """V6 network/trainer with V13 conservative optimization defaults."""

    def __init__(self, args):
        super(Trainer, self).__init__(args)
        logging.info(
            'Using model: MFSAN_CDAN_BIMAMBA_CW_RWCA_V13_CWRU1_BASELINE_STABILIZED'
        )
        logging.info(
            'V13 network structure unchanged: MSCNN-BiMamba backbone + three '
            'source branches/classifiers + three CDAN discriminators.'
        )
        logging.info(
            'V13 deliberately excludes V7-V12 class gates, specialists, '
            'Hard-SupCon, prototype rescue and test-time class rescue.'
        )
        logging.info(
            'V13 schedule: MMD starts={} (weight={:.4f}), CDAN starts={} '
            '(lambda={:.6f}), CLMMD starts={} (lambda={:.6f}), '
            'SupCon lambda={:.6f}.'.format(
                self.mmd_start_epoch,
                self.mmd_weight,
                self.adv_start_epoch,
                self.lambda_adv,
                self.clmmd_start_epoch,
                self.lambda_clmmd,
                self.lambda_supcon,
            )
        )

    def _checkpoint_dict(self):
        checkpoint = super(Trainer, self)._checkpoint_dict()
        checkpoint.update({
            'v13_cwru1_baseline_stabilized': True,
            'v13_network_structure_unchanged': True,
            'v13_mmd_weight': float(self.mmd_weight),
            'v13_mmd_start_epoch': int(self.mmd_start_epoch),
            'v13_adv_start_epoch': int(self.adv_start_epoch),
            'v13_clmmd_start_epoch': int(self.clmmd_start_epoch),
            'v13_source_label_smoothing': float(self.source_label_smoothing),
            'v13_grad_clip_norm': float(self.grad_clip_norm),
        })
        return checkpoint
