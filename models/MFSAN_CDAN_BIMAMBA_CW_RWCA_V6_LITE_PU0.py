# -*- coding: utf-8 -*-
"""
MFSAN-CDAN-BiMamba-CW-RWCA-V6-Lite-PU0

PU0-oriented simplified version of V6 StableGate.

The trainable architecture and target-test evaluation behavior are unchanged.
The model still evaluates the held-out target test set after every epoch and
updates the best checkpoint according to target-test accuracy.

The default optimization objective is simplified to five effective terms:

    reliability-weighted source CE
    + global MK-MMD
    + reliability-weighted CDAN
    + class-wise reliability-weighted CLMMD
    + reliability-weighted focused SupCon

Removed from the default objective and skipped during computation:
    conditional MMD/CDA, target entropy minimization, CDD/L1 consistency,
    and MCA.

PU0 defaults:
    - focused SupCon classes: 3,4 (KA30 and KB23 in the current 9-class map)
    - earlier stable-gate confirmation
    - much lower pre-confirmation and confirmed-source floors

Recommended model name:
    MFSAN_CDAN_BIMAMBA_CW_RWCA_V6_LITE_PU0
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V5_MCA import Trainer as V5Trainer


class Trainer(V5Trainer):
    """Simplified V6 trainer for PU0 while retaining the original test loop."""

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # Explicit switch retained for ablation; PU0 Lite default is 0.
        self.lambda_l1 = float(getattr(args, 'lambda_l1', 0.0))


        # V13-style stabilization controls. These only change optimization
        # timing/regularization; the network and the five existing losses are unchanged.
        self.mmd_weight = max(0.0, float(getattr(args, 'mmd_weight', 1.0)))
        self.mmd_start_epoch = max(1, int(getattr(args, 'mmd_start_epoch', 1)))
        self.adv_start_epoch = max(1, int(getattr(args, 'adv_start_epoch', 1)))
        self.clmmd_start_epoch = max(1, int(getattr(args, 'clmmd_start_epoch', 1)))
        self.source_label_smoothing = min(
            max(float(getattr(args, 'source_label_smoothing', 0.0)), 0.0), 0.5
        )
        self.grad_clip_norm = max(0.0, float(getattr(args, 'grad_clip_norm', 0.0)))

        # ------------------------------------------------------------------
        # Stable source gate
        # ------------------------------------------------------------------
        self.v6_gate_enabled = bool(getattr(args, 'v6_gate_enabled', True))
        self.v6_gate_start_epoch = int(getattr(args, 'v6_gate_start_epoch', 4))
        self.v6_gate_confirm_epochs = max(1, int(getattr(args, 'v6_gate_confirm_epochs', 3)))
        self.v6_gate_release_epochs = max(1, int(getattr(args, 'v6_gate_release_epochs', 3)))
        self.v6_gate_confirm_gap = float(getattr(args, 'v6_gate_confirm_gap', 0.08))
        self.v6_gate_release_gap = float(getattr(args, 'v6_gate_release_gap', 0.03))
        self.v6_gate_preconfirm_floor = float(getattr(args, 'v6_gate_preconfirm_floor', 0.05))
        self.v6_gate_bottom_floor = float(getattr(args, 'v6_gate_bottom_floor', 0.01))
        self.v6_gate_max_source_weight = float(getattr(args, 'v6_gate_max_source_weight', 0.75))
        self.v6_gate_apply_to_supcon = bool(getattr(args, 'v6_gate_apply_to_supcon', True))
        self.v6_supcon_source_min_weight = float(getattr(args, 'v6_supcon_source_min_weight', 0.05))

        # Mild sharpening keeps true class-wise PU_1/PU_2 differences visible.
        self.v6_class_weight_power = float(getattr(args, 'v6_class_weight_power', 1.20))

        # Independent class-alignment multiplier. 1.0 exactly reproduces V5.
        self.v6_class_alignment_boost = float(getattr(args, 'v6_class_alignment_boost', 1.0))
        self.v6_mca_pairwise_weight = float(getattr(args, 'v6_mca_pairwise_weight', 0.25))

        self._v6_candidate_source = -1
        self._v6_candidate_streak = 0
        self._v6_confirmed_negative_source = -1
        self._v6_release_streak = 0
        self._v6_last_raw_global_weights = torch.full(
            (self.num_source,), 1.0 / float(self.num_source), device=self.device
        )
        self._v6_last_effective_global_weights = self._v6_last_raw_global_weights.clone()
        self._v6_active_supcon_source_weights = self._v6_last_effective_global_weights.clone()

        self.v6_source_names = list(getattr(args, 'source_name', []))
        if len(self.v6_source_names) != self.num_source:
            self.v6_source_names = [f'src{i}' for i in range(self.num_source)]

        logging.info('Using model: MFSAN_CDAN_BIMAMBA_CW_RWCA_V6_LITE_PU0')
        logging.info(
            'V6 stable gate: enabled={} sources={} start={} confirm={} release={} '
            'confirm_gap={:.4f} release_gap={:.4f} pre_floor={:.4f} bottom_floor={:.4f} cap={:.4f}'.format(
                self.v6_gate_enabled,
                self.v6_source_names,
                self.v6_gate_start_epoch,
                self.v6_gate_confirm_epochs,
                self.v6_gate_release_epochs,
                self.v6_gate_confirm_gap,
                self.v6_gate_release_gap,
                self.v6_gate_preconfirm_floor,
                self.v6_gate_bottom_floor,
                self.v6_gate_max_source_weight,
            )
        )
        logging.info(
            'V6 source-supervision gate: source_CE=class-wise weighted, SupCon_gate={}, '
            'SupCon_min_source_weight={:.4f}'.format(
                self.v6_gate_apply_to_supcon,
                self.v6_supcon_source_min_weight,
            )
        )
        logging.info(
            'V6 class reliability: cw_alpha_max={:.4f}, class_weight_power={:.4f}, '
            'class_alignment_boost={:.4f}, MCA_pairwise_weight={:.4f}'.format(
                self.cw_alpha,
                self.v6_class_weight_power,
                self.v6_class_alignment_boost,
                self.v6_mca_pairwise_weight,
            )
        )
        logging.info(
            'V6-Lite effective objective: CE + MMD + CDAN + CLMMD + SupCon | '
            'lambda_l1={:.6f}, lambda_cda={:.6f}, lambda_ent={:.6f}, lambda_mca={:.6f}'.format(
                self.lambda_l1, self.lambda_cda, self.lambda_ent, self.lambda_mca
            )
        )

        logging.info(
            'Stable optimization: mmd_weight={:.6f} starts={} | CDAN starts={} | '
            'CLMMD starts={} | source_label_smoothing={:.4f} | grad_clip_norm={:.4f}'.format(
                self.mmd_weight,
                self.mmd_start_epoch,
                self.adv_start_epoch,
                self.clmmd_start_epoch,
                self.source_label_smoothing,
                self.grad_clip_norm,
            )
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def _checkpoint_dict(self):
        ckpt = super(Trainer, self)._checkpoint_dict()
        ckpt.update({
            'v6_stable_gate': True,
            'v6_candidate_source': int(self._v6_candidate_source),
            'v6_candidate_streak': int(self._v6_candidate_streak),
            'v6_confirmed_negative_source': int(self._v6_confirmed_negative_source),
            'v6_release_streak': int(self._v6_release_streak),
            'v6_last_raw_global_weights': self._v6_last_raw_global_weights.detach().cpu(),
            'v6_last_effective_global_weights': self._v6_last_effective_global_weights.detach().cpu(),
            'v6_gate_enabled': self.v6_gate_enabled,
            'v6_gate_start_epoch': self.v6_gate_start_epoch,
            'v6_gate_confirm_epochs': self.v6_gate_confirm_epochs,
            'v6_gate_release_epochs': self.v6_gate_release_epochs,
            'v6_gate_confirm_gap': self.v6_gate_confirm_gap,
            'v6_gate_release_gap': self.v6_gate_release_gap,
            'v6_gate_preconfirm_floor': self.v6_gate_preconfirm_floor,
            'v6_gate_bottom_floor': self.v6_gate_bottom_floor,
            'v6_gate_max_source_weight': self.v6_gate_max_source_weight,
            'v6_class_weight_power': self.v6_class_weight_power,
            'v6_class_alignment_boost': self.v6_class_alignment_boost,
            'v6_mca_pairwise_weight': self.v6_mca_pairwise_weight,
            'v6_lite_pu0': True,
            'lambda_l1': self.lambda_l1,
        })
        return ckpt

    def load_model(self):
        super(Trainer, self).load_model()
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        self._v6_candidate_source = int(ckpt.get('v6_candidate_source', -1))
        self._v6_candidate_streak = int(ckpt.get('v6_candidate_streak', 0))
        self._v6_confirmed_negative_source = int(
            ckpt.get('v6_confirmed_negative_source', -1)
        )
        self._v6_release_streak = int(ckpt.get('v6_release_streak', 0))
        if 'v6_last_raw_global_weights' in ckpt:
            self._v6_last_raw_global_weights = ckpt['v6_last_raw_global_weights'].to(
                self.device
            ).float()
        if 'v6_last_effective_global_weights' in ckpt:
            self._v6_last_effective_global_weights = ckpt[
                'v6_last_effective_global_weights'
            ].to(self.device).float()
        logging.info(
            'Loaded V6 gate state: candidate={} streak={} confirmed={} release_streak={}'.format(
                self._source_name(self._v6_candidate_source),
                self._v6_candidate_streak,
                self._source_name(self._v6_confirmed_negative_source),
                self._v6_release_streak,
            )
        )

    # ------------------------------------------------------------------
    # Stable gate utilities
    # ------------------------------------------------------------------
    def _source_name(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.v6_source_names):
            return 'none'
        return '{}(src{})'.format(self.v6_source_names[idx], idx)

    def _normalize_source_vector(self, weights: torch.Tensor) -> torch.Tensor:
        weights = torch.clamp(weights.float(), min=self.entropy_eps)
        return weights / (weights.sum() + self.entropy_eps)

    def _cap_source_vector(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply a soft maximum without changing source ordering."""
        weights = self._normalize_source_vector(weights)
        cap = min(max(self.v6_gate_max_source_weight, 1.0 / self.num_source), 0.999)

        # A few iterations are sufficient for K=3 and avoid a hard-coded source count.
        for _ in range(max(self.num_source, 1) + 2):
            over = weights > cap
            if not bool(over.any()):
                break
            excess = (weights[over] - cap).sum()
            weights = weights.clone()
            weights[over] = cap
            under = ~over
            if bool(under.any()):
                base = weights[under]
                weights[under] = base + excess * base / (base.sum() + self.entropy_eps)
            weights = self._normalize_source_vector(weights)
        return weights

    def _apply_stable_gate_vector(self, raw_weights: torch.Tensor) -> torch.Tensor:
        raw = self._normalize_source_vector(raw_weights)
        epoch = int(getattr(self, '_cur_epoch', 1))

        if not self.v6_gate_enabled:
            return raw

        effective = raw.clone()

        if self._v6_confirmed_negative_source >= 0:
            bad = self._v6_confirmed_negative_source
            keep = torch.ones(self.num_source, dtype=torch.bool, device=raw.device)
            keep[bad] = False
            effective[bad] = max(self.v6_gate_bottom_floor, self.entropy_eps)
            remain = max(1.0 - float(effective[bad].item()), self.entropy_eps)
            effective[keep] = remain * raw[keep] / (raw[keep].sum() + self.entropy_eps)
        elif epoch >= self.v6_gate_start_epoch:
            # Before confirmation, do not allow an early noisy estimate to eliminate a source.
            floor = max(self.v6_gate_preconfirm_floor, self.entropy_eps)
            effective = torch.clamp(effective, min=floor)
            effective = self._normalize_source_vector(effective)

        effective = self._cap_source_vector(effective)
        # Re-enforce the confirmed floor after cap redistribution.
        if self._v6_confirmed_negative_source >= 0:
            bad = self._v6_confirmed_negative_source
            keep = torch.ones(self.num_source, dtype=torch.bool, device=raw.device)
            keep[bad] = False
            floor = max(self.v6_gate_bottom_floor, self.entropy_eps)
            effective = effective.clone()
            effective[bad] = floor
            effective[keep] = (1.0 - floor) * effective[keep] / (
                effective[keep].sum() + self.entropy_eps
            )
        return effective.detach() if self.rw_detach_weights else effective

    def _global_loss_source_weights(
        self,
        effective_global_weights: torch.Tensor,
        class_weights: torch.Tensor,
        class_average_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Source weights used by scalar global alignment losses (MMD/CDAN).

        V6 keeps its historical behavior and uses the class-averaged final
        source weights.  Subclasses can override this hook without copying the
        full training loop.
        """
        return class_average_weights

    def _apply_stable_gate_matrix(self, class_weights: torch.Tensor) -> torch.Tensor:
        """
        Gate only the globally unreliable source and preserve class-wise differences
        between the remaining sources.
        """
        cw = self._normalize_class_source_weights(class_weights.float())
        epoch = int(getattr(self, '_cur_epoch', 1))

        # Mild class-wise sharpening. Power=1 means no change.
        power = max(self.v6_class_weight_power, self.entropy_eps)
        if abs(power - 1.0) > 1e-8:
            cw = torch.pow(cw.clamp_min(self.entropy_eps), power)
            cw = self._normalize_class_source_weights(cw)

        if not self.v6_gate_enabled:
            return cw.detach() if self.rw_detach_weights else cw

        if self._v6_confirmed_negative_source >= 0:
            bad = self._v6_confirmed_negative_source
            keep = torch.ones(self.num_source, dtype=torch.bool, device=cw.device)
            keep[bad] = False
            floor = max(self.v6_gate_bottom_floor, self.entropy_eps)
            cw = cw.clone()
            cw[bad, :] = floor
            remain = 1.0 - floor
            cw[keep, :] = remain * cw[keep, :] / (
                cw[keep, :].sum(dim=0, keepdim=True) + self.entropy_eps
            )
        elif epoch >= self.v6_gate_start_epoch:
            floor = max(self.v6_gate_preconfirm_floor, self.entropy_eps)
            cw = torch.clamp(cw, min=floor)
            cw = self._normalize_class_source_weights(cw)

        # Apply the global source cap per class while preserving within-class ratios.
        cap = min(max(self.v6_gate_max_source_weight, 1.0 / self.num_source), 0.999)
        for c in range(self.num_classes):
            vec = cw[:, c]
            for _ in range(max(self.num_source, 1) + 2):
                over = vec > cap
                if not bool(over.any()):
                    break
                excess = (vec[over] - cap).sum()
                vec = vec.clone()
                vec[over] = cap
                under = ~over
                if bool(under.any()):
                    base = vec[under]
                    vec[under] = base + excess * base / (base.sum() + self.entropy_eps)
                vec = self._normalize_source_vector(vec)
            cw[:, c] = vec

        cw = self._normalize_class_source_weights(cw)
        if self._v6_confirmed_negative_source >= 0:
            bad = self._v6_confirmed_negative_source
            keep = torch.ones(self.num_source, dtype=torch.bool, device=cw.device)
            keep[bad] = False
            floor = max(self.v6_gate_bottom_floor, self.entropy_eps)
            cw = cw.clone()
            cw[bad, :] = floor
            cw[keep, :] = (1.0 - floor) * cw[keep, :] / (
                cw[keep, :].sum(dim=0, keepdim=True) + self.entropy_eps
            )
            cw = self._normalize_class_source_weights(cw)
        return cw.detach() if self.rw_detach_weights else cw

    def _update_stable_gate_from_epoch(self, raw_epoch_weights: torch.Tensor) -> None:
        """Update gate state once per epoch using epoch-averaged raw weights."""
        raw = self._normalize_source_vector(raw_epoch_weights.detach())
        self._v6_last_raw_global_weights = raw.clone()

        epoch = int(getattr(self, '_cur_epoch', 1))
        if not self.v6_gate_enabled or epoch < self.v6_gate_start_epoch:
            logging.info(
                'V6 stable gate: event=warmup epoch={} raw={}'.format(
                    epoch,
                    ', '.join(
                        '{}={:.4f}'.format(self._source_name(i), raw[i].item())
                        for i in range(self.num_source)
                    ),
                )
            )
            return

        sorted_weights, sorted_indices = torch.sort(raw, descending=False)
        worst = int(sorted_indices[0].item())
        second = float(sorted_weights[1].item()) if self.num_source > 1 else 1.0
        worst_value = float(sorted_weights[0].item())
        gap = second - worst_value

        event = 'observe'
        if self._v6_confirmed_negative_source < 0:
            if gap >= self.v6_gate_confirm_gap:
                if self._v6_candidate_source == worst:
                    self._v6_candidate_streak += 1
                else:
                    self._v6_candidate_source = worst
                    self._v6_candidate_streak = 1

                if self._v6_candidate_streak >= self.v6_gate_confirm_epochs:
                    self._v6_confirmed_negative_source = worst
                    self._v6_release_streak = 0
                    event = 'confirmed'
            else:
                self._v6_candidate_source = -1
                self._v6_candidate_streak = 0
        else:
            confirmed = self._v6_confirmed_negative_source
            still_worst = worst == confirmed
            still_separated = gap >= self.v6_gate_release_gap
            if still_worst and still_separated:
                self._v6_release_streak = 0
            else:
                self._v6_release_streak += 1
                if self._v6_release_streak >= self.v6_gate_release_epochs:
                    event = 'released'
                    self._v6_confirmed_negative_source = -1
                    self._v6_candidate_source = -1
                    self._v6_candidate_streak = 0
                    self._v6_release_streak = 0

        logging.info(
            'V6 stable gate: event={} candidate={} streak={} confirmed={} release_streak={} '
            'worst={} gap={:.5f} raw={}'.format(
                event,
                self._source_name(self._v6_candidate_source),
                self._v6_candidate_streak,
                self._source_name(self._v6_confirmed_negative_source),
                self._v6_release_streak,
                self._source_name(worst),
                gap,
                ', '.join(
                    '{}={:.4f}'.format(self._source_name(i), raw[i].item())
                    for i in range(self.num_source)
                ),
            )
        )

    # ------------------------------------------------------------------
    # SupCon: remove a confirmed negative source from shared-feature contrast
    # ------------------------------------------------------------------
    def _compute_source_supcon_loss(self, g_s_list, f_s_all, source_label_list):
        """
        Reliability-weighted focused SupCon.

        This remains the same supervised contrastive objective. The only change
        is that valid anchors are averaged with current source reliability, so a
        weak source cannot contribute the same shared-backbone gradient as a
        highly reliable source. A confirmed negative source is excluded.
        """
        cur_epoch = int(getattr(self, '_cur_epoch', 1))
        if self.lambda_supcon <= 0.0 or cur_epoch < self.supcon_start_epoch:
            return torch.tensor(0.0, device=self.device)

        mode = self.supcon_feature_mode.upper()
        feat_list = f_s_all if mode == 'F' else g_s_list

        source_w = self._normalize_source_vector(
            self._v6_active_supcon_source_weights.to(self.device)
        ).detach()
        active_indices: List[int] = list(range(self.num_source))

        if self.v6_gate_apply_to_supcon:
            if self._v6_confirmed_negative_source >= 0:
                active_indices = [
                    k for k in active_indices
                    if k != self._v6_confirmed_negative_source
                ]
            else:
                active_indices = [
                    k for k in active_indices
                    if float(source_w[k].item()) >= self.v6_supcon_source_min_weight
                ]

        if len(active_indices) == 0:
            return torch.tensor(0.0, device=self.device)

        features = torch.cat([feat_list[k] for k in active_indices], dim=0)
        labels = torch.cat([source_label_list[k] for k in active_indices], dim=0)
        anchor_weights = torch.cat([
            torch.full(
                (feat_list[k].size(0),),
                float(source_w[k].item()),
                device=self.device,
                dtype=features.dtype,
            )
            for k in active_indices
        ], dim=0)

        if self.supcon_focus_classes is not None:
            mask = torch.zeros_like(labels, dtype=torch.bool, device=labels.device)
            for c in self.supcon_focus_classes:
                mask = mask | (labels == int(c))
            if int(mask.sum().item()) <= 1:
                return torch.tensor(0.0, device=self.device)
            features = features[mask]
            labels = labels[mask]
            anchor_weights = anchor_weights[mask]

        if features.size(0) <= 1:
            return torch.tensor(0.0, device=self.device)

        eps = max(float(getattr(self, 'entropy_eps', 1e-8)), 1e-12)
        temperature = max(float(self.supcon_temperature), eps)
        features = F.normalize(features, p=2, dim=1)
        logits = torch.matmul(features, features.t()) / temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        same_class = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
        self_mask = torch.ones_like(same_class)
        self_mask.fill_diagonal_(0.0)
        pos_mask = same_class * self_mask

        exp_logits = torch.exp(logits) * self_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)
        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if int(valid.sum().item()) == 0:
            return torch.tensor(0.0, device=self.device)

        anchor_loss = -(pos_mask * log_prob).sum(dim=1) / (pos_count + eps)
        valid_weights = anchor_weights[valid].clamp_min(eps)
        return (anchor_loss[valid] * valid_weights).sum() / (
            valid_weights.sum() + eps
        )

    def _multi_classifier_alignment_loss(self, probs_t_all, probs_t_fused, class_src_weights):
        """MCA is intentionally removed from V6-Lite-PU0."""
        return torch.tensor(0.0, device=probs_t_fused.device)

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        """
        Simplified PU0 training loop.

        Effective objective:
            L = L_cls + t0*L_mmd + t2*lambda_adv*L_cdan
                + t2*lambda_clmmd*L_clmmd + lambda_supcon*L_supcon

        CDA, target-entropy minimization, CDD/L1 and MCA are not computed when
        their coefficients are zero. The inherited train() and test() methods
        are untouched, so target-test metrics are still printed every epoch and
        still determine the best checkpoint.
        """
        self._cur_epoch = int(getattr(self, '_cur_epoch', 1))

        self._v3_collect_epoch_weights = True
        self._v3_class_weight_sum = torch.zeros(
            self.num_source, self.num_classes, device=self.device
        )
        self._v3_class_weight_count = 0

        weight_sum = torch.zeros(self.num_source, device=self.device)
        global_loss_weight_sum = torch.zeros(self.num_source, device=self.device)
        raw_global_weight_sum = torch.zeros(self.num_source, device=self.device)
        effective_global_weight_sum = torch.zeros(self.num_source, device=self.device)
        raw_class_weight_sum = torch.zeros(
            self.num_source, self.num_classes, device=self.device
        )
        class_weight_sum = torch.zeros(
            self.num_source, self.num_classes, device=self.device
        )
        rec_score_sum = torch.zeros(
            self.num_source, self.num_classes, device=self.device
        )
        rec_guided_weight_sum = torch.zeros(
            self.num_source, self.num_classes, device=self.device
        )
        alpha_sum = 0.0
        supcon_sum = torch.tensor(0.0, device=self.device)

        zero = lambda: torch.tensor(0.0, device=self.device)

        for _ in tqdm(range(self.num_iter), ascii=True):
            target_data, _ = self._get_next_batch('train')

            source_data_list = []
            source_label_list = []
            for k in range(self.num_source):
                source_data_k, source_labels_k = self._get_next_batch(
                    self.src[k], return_actual=True
                )
                source_labels_k = self._get_train_label(
                    source_labels_k, label_set=self.src_labels_flat
                )
                source_data_list.append(source_data_k)
                source_label_list.append(source_labels_k)

            self.optimizer.zero_grad()

            data = torch.cat(source_data_list + [target_data], dim=0)
            g_all = self.G(data)
            split_sizes = [x.size(0) for x in source_data_list] + [target_data.size(0)]
            g_split = torch.split(g_all, split_sizes, dim=0)
            g_s_list = list(g_split[:-1])
            g_t = g_split[-1]

            loss_cls_vec_list = []
            loss_mmd_list = []
            loss_adv_list = []
            loss_clmmd_vec_list = []
            clmmd_valid_list = []
            ent_list = []
            domain_acc_list = []
            probs_t_all = []
            probs_s_all = []
            f_s_all = []
            f_t_all = []

            base_adapt_tradeoff = self.tradeoff[2] if len(self.tradeoff) > 2 else 1.0
            cur_epoch = int(getattr(self, '_cur_epoch', 1))
            mmd_tradeoff = (
                self.tradeoff[0] if cur_epoch >= self.mmd_start_epoch else 0.0
            )
            adv_tradeoff = (
                base_adapt_tradeoff if cur_epoch >= self.adv_start_epoch else 0.0
            )
            clmmd_tradeoff = (
                base_adapt_tradeoff if cur_epoch >= self.clmmd_start_epoch else 0.0
            )
            grl_coeff = self.lambda_grl * adv_tradeoff

            for k in range(self.num_source):
                f_s = self.Fs[k](g_s_list[k])
                f_t = self.Fs[k](g_t)
                f_s_all.append(f_s)
                f_t_all.append(f_t)

                y_s = self.Cs[k](f_s)
                y_t = self.Cs[k](f_t)
                p_s = F.softmax(y_s, dim=1)
                p_t = F.softmax(y_t, dim=1)
                probs_t_all.append(p_t)
                probs_s_all.append(p_s)

                labels_s = source_label_list[k]
                loss_cls_vec_k = F.cross_entropy(
                    y_s,
                    labels_s,
                    reduction='none',
                    label_smoothing=self.source_label_smoothing,
                )
                loss_mmd_k = self.mkmmd(f_s, f_t)

                if self.lambda_clmmd > 0.0 and clmmd_tradeoff > 0.0:
                    # Expose the active source branch to subclasses such as V8,
                    # which maintain source-specific class prototypes for CLMMD.
                    self._active_source_idx_for_clmmd = k
                    loss_clmmd_vec_k, clmmd_valid_k = self._classwise_lmmd_per_class(
                        f_s, f_t, labels_s, p_t
                    )
                else:
                    loss_clmmd_vec_k = torch.zeros(
                        self.num_classes, device=self.device
                    )
                    clmmd_valid_k = torch.zeros(
                        self.num_classes, device=self.device
                    )

                if self.lambda_adv > 0.0 and adv_tradeoff > 0.0:
                    loss_adv_k, domain_acc_k = self._domain_adversarial_loss(
                        cur_src_idx=k,
                        f_s=f_s,
                        f_t=f_t,
                        source_labels=labels_s,
                        prob_t=p_t,
                        grl_coeff=grl_coeff,
                    )
                else:
                    loss_adv_k = zero()
                    domain_acc_k = zero()

                # Entropy remains a reliability signal, not an independent loss.
                ent_k = self._entropy_scalar(p_t)

                loss_cls_vec_list.append(loss_cls_vec_k)
                loss_mmd_list.append(loss_mmd_k)
                loss_adv_list.append(loss_adv_k)
                loss_clmmd_vec_list.append(loss_clmmd_vec_k)
                clmmd_valid_list.append(clmmd_valid_k)
                ent_list.append(ent_k)
                domain_acc_list.append(
                    domain_acc_k if torch.is_tensor(domain_acc_k)
                    else torch.tensor(domain_acc_k, device=self.device)
                )
                epoch_acc['Source Data'] += self._get_accuracy(
                    y_s, labels_s
                ) / float(self.num_source)

            raw_global_src_weights = self._source_reliability_weights(
                loss_mmd_list, ent_list
            )
            effective_global_src_weights = self._apply_stable_gate_vector(
                raw_global_src_weights
            )

            raw_class_src_weights = self._class_source_reliability_weights(
                f_s_all,
                f_t_all,
                source_label_list,
                probs_t_all,
                loss_mmd_list,
                ent_list,
            )
            rec_scores = self._source_per_class_recognition_scores(
                probs_s_all, source_label_list
            )
            rec_guided_class_src_weights = self._recognition_guided_class_weights(
                raw_class_src_weights, rec_scores
            )

            cw_alpha_now = self._get_cw_alpha()
            class_src_weights = self._global_guided_class_weights(
                effective_global_src_weights,
                rec_guided_class_src_weights,
                alpha=cw_alpha_now,
            )
            class_src_weights = self._apply_stable_gate_matrix(class_src_weights)
            src_weights = self._source_weights_from_class_weights(class_src_weights)
            global_loss_src_weights = self._global_loss_source_weights(
                effective_global_src_weights, class_src_weights, src_weights
            )
            global_loss_src_weights = self._normalize_source_vector(
                global_loss_src_weights
            )
            self._v6_active_supcon_source_weights = src_weights.detach()

            self._update_class_source_weight_ema(class_src_weights)
            raw_global_weight_sum += raw_global_src_weights.detach()
            effective_global_weight_sum += effective_global_src_weights.detach()
            raw_class_weight_sum += raw_class_src_weights.detach()
            rec_score_sum += rec_scores.detach()
            rec_guided_weight_sum += rec_guided_class_src_weights.detach()
            weight_sum += src_weights.detach()
            global_loss_weight_sum += global_loss_src_weights.detach()
            class_weight_sum += class_src_weights.detach()
            alpha_sum += float(cw_alpha_now)

            probs_t_fused = self._class_weighted_fusion(
                probs_t_all, class_src_weights
            )

            # Reliability-weighted source classification.
            cls_num = zero()
            cls_den = zero()
            for k in range(self.num_source):
                sample_w = class_src_weights[k, source_label_list[k]]
                cls_num = cls_num + (loss_cls_vec_list[k] * sample_w).sum()
                cls_den = cls_den + sample_w.sum()
            loss_cls = cls_num / (cls_den + self.entropy_eps)

            loss_mmd = sum(
                global_loss_src_weights[k] * loss_mmd_list[k]
                for k in range(self.num_source)
            )

            if self.lambda_adv > 0.0 and adv_tradeoff > 0.0:
                loss_adv = sum(
                    global_loss_src_weights[k] * loss_adv_list[k]
                    for k in range(self.num_source)
                )
                domain_acc = sum(
                    global_loss_src_weights[k] * domain_acc_list[k]
                    for k in range(self.num_source)
                )
            else:
                loss_adv = zero()
                domain_acc = zero()

            if self.lambda_clmmd > 0.0 and clmmd_tradeoff > 0.0:
                clmmd_num = zero()
                clmmd_den = zero()
                for k in range(self.num_source):
                    valid = clmmd_valid_list[k]
                    clmmd_num = clmmd_num + (
                        class_src_weights[k] * valid * loss_clmmd_vec_list[k]
                    ).sum()
                    clmmd_den = clmmd_den + (
                        class_src_weights[k] * valid
                    ).sum()
                if float(clmmd_den.detach().item()) > 0.0:
                    loss_clmmd = clmmd_num / (clmmd_den + self.entropy_eps)
                else:
                    loss_clmmd = zero()
            else:
                loss_clmmd = zero()

            loss_supcon = self._compute_source_supcon_loss(
                g_s_list, f_s_all, source_label_list
            )
            supcon_sum += loss_supcon.detach()

            class_boost = self.v6_class_alignment_boost
            loss = (
                loss_cls
                + self.mmd_weight * mmd_tradeoff * loss_mmd
                + adv_tradeoff * self.lambda_adv * loss_adv
                + class_boost * clmmd_tradeoff * self.lambda_clmmd * loss_clmmd
                + self.lambda_supcon * loss_supcon
            )

            epoch_acc['Domain Data'] += domain_acc.detach().item()
            epoch_loss['Source Classifier'] += loss_cls.detach()
            epoch_loss['MMD'] += loss_mmd.detach()
            epoch_loss['CLMMD'] += loss_clmmd.detach()
            epoch_loss['CDAN Domain'] += loss_adv.detach()
            epoch_loss['MMD Weighted'] += (
                self.mmd_weight * mmd_tradeoff * loss_mmd
            ).detach()
            epoch_loss['CLMMD Weighted'] += (
                class_boost * clmmd_tradeoff * self.lambda_clmmd * loss_clmmd
            ).detach()
            epoch_loss['CDAN Weighted'] += (
                adv_tradeoff * self.lambda_adv * loss_adv
            ).detach()
            epoch_loss['SupCon'] += loss_supcon.detach()
            epoch_loss['SupCon Weighted'] += (
                self.lambda_supcon * loss_supcon
            ).detach()
            epoch_loss['Total'] += loss.detach()
            epoch_loss['CW Alpha'] += torch.tensor(
                float(cw_alpha_now), device=self.device
            )

            for k in range(self.num_source):
                epoch_loss[f'Raw Global Prior src{k}'] += raw_global_src_weights[k].detach()
                epoch_loss[f'Effective Global Prior src{k}'] += effective_global_src_weights[k].detach()
                epoch_loss[f'RW Weight src{k}'] += src_weights[k].detach()
                epoch_loss[f'Global Loss Weight src{k}'] += global_loss_src_weights[k].detach()
            for c in self._cw_log_classes:
                for k in range(self.num_source):
                    epoch_loss[f'Raw CW Weight c{c} src{k}'] += raw_class_src_weights[k, c].detach()
                    epoch_loss[f'Rec Score c{c} src{k}'] += rec_scores[k, c].detach()
                    epoch_loss[f'Rec-Guided CW Weight c{c} src{k}'] += rec_guided_class_src_weights[k, c].detach()
                    epoch_loss[f'V6 Final CW Weight c{c} src{k}'] += class_src_weights[k, c].detach()

            loss.backward()
            if self.grad_clip_norm > 0.0:
                trainable_params = [
                    param
                    for group in self.optimizer.param_groups
                    for param in group['params']
                    if param.grad is not None
                ]
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, self.grad_clip_norm
                )
                epoch_loss['Gradient Norm'] += torch.as_tensor(
                    float(grad_norm), device=self.device
                )
            self.optimizer.step()

        denom_iter = max(float(self.num_iter), 1.0)
        avg_weights = (weight_sum / denom_iter).detach()
        avg_global_loss_weights = (global_loss_weight_sum / denom_iter).detach()
        avg_raw_global = (raw_global_weight_sum / denom_iter).detach()
        avg_effective_global = (effective_global_weight_sum / denom_iter).detach()
        avg_raw_class = (raw_class_weight_sum / denom_iter).detach()
        avg_class = (class_weight_sum / denom_iter).detach()
        avg_rec_scores = (rec_score_sum / denom_iter).detach()
        avg_rec_guided = (rec_guided_weight_sum / denom_iter).detach()
        avg_alpha = alpha_sum / denom_iter
        avg_supcon = (supcon_sum / denom_iter).detach().item()

        self._v6_last_effective_global_weights = avg_effective_global.clone()
        self._update_stable_gate_from_epoch(avg_raw_global)

        logging.info(
            'V6-Lite raw global source weights: {}'.format(
                ', '.join(
                    '{}={:.4f}'.format(self._source_name(i), avg_raw_global[i].item())
                    for i in range(self.num_source)
                )
            )
        )
        logging.info(
            'V6-Lite effective global source weights: {}'.format(
                ', '.join(
                    '{}={:.4f}'.format(self._source_name(i), avg_effective_global[i].item())
                    for i in range(self.num_source)
                )
            )
        )
        logging.info(
            'V6-Lite final class-averaged source weights: {}'.format(
                ', '.join(
                    '{}={:.4f}'.format(self._source_name(i), avg_weights[i].item())
                    for i in range(self.num_source)
                )
            )
        )
        logging.info(
            'V6-Lite global alignment loss source weights: {}'.format(
                ', '.join(
                    '{}={:.4f}'.format(
                        self._source_name(i), avg_global_loss_weights[i].item()
                    )
                    for i in range(self.num_source)
                )
            )
        )
        logging.info(
            'V6-Lite CW alpha average: {:.4f} | warmup={} | alpha_max={:.4f} | ramp={}'.format(
                avg_alpha,
                self.cw_warmup_epochs,
                self.cw_alpha,
                self.cw_alpha_ramp_epochs,
            )
        )
        for c in self._cw_log_classes:
            logging.info(
                'V6-Lite class-{} raw transfer weights: {}'.format(
                    c,
                    ', '.join(
                        '{}={:.4f}'.format(self._source_name(i), avg_raw_class[i, c].item())
                        for i in range(self.num_source)
                    ),
                )
            )
            logging.info(
                'V6-Lite class-{} recognition scores: {}'.format(
                    c,
                    ', '.join(
                        '{}={:.4f}'.format(self._source_name(i), avg_rec_scores[i, c].item())
                        for i in range(self.num_source)
                    ),
                )
            )
            logging.info(
                'V6-Lite class-{} rec-guided weights: {}'.format(
                    c,
                    ', '.join(
                        '{}={:.4f}'.format(self._source_name(i), avg_rec_guided[i, c].item())
                        for i in range(self.num_source)
                    ),
                )
            )
            logging.info(
                'V6-Lite class-{} final gated weights: {}'.format(
                    c,
                    ', '.join(
                        '{}={:.4f}'.format(self._source_name(i), avg_class[i, c].item())
                        for i in range(self.num_source)
                    ),
                )
            )

        logging.info(
            'V6-Lite SupCon average: {:.6f} | weighted={:.6f} | focus_classes={} | confirmed_negative={}'.format(
                avg_supcon,
                self.lambda_supcon * avg_supcon,
                'all' if self.supcon_focus_classes is None else self.supcon_focus_classes,
                self._source_name(self._v6_confirmed_negative_source),
            )
        )

        if hasattr(self.G, 'get_gate'):
            logging.info(
                'BiMamba-Att residual gate: {:.6f}'.format(
                    self.G.get_gate().detach().item()
                )
            )

        self._finalize_v3_epoch_weights()
        return epoch_acc, epoch_loss

