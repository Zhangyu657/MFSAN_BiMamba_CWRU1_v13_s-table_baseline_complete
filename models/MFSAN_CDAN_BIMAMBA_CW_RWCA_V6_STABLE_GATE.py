# -*- coding: utf-8 -*-
"""
MFSAN-CDAN-BiMamba-CW-RWCA-V6-StableGate

This model deliberately returns to the original high-performing V5-MCA training
path and adds only targeted stability improvements:

1. Stable negative-source gate
   - gate decisions are made from epoch-averaged global reliability;
   - a source must remain the least reliable source for several consecutive
     epochs before it is suppressed;
   - the gate is automatic and is not hard-coded to a specific source name.

2. Reliability weights participate in source supervision
   - source CE is class-wise reliability weighted (inherited from V5);
   - after a negative source is confirmed, it is also removed from focused
     SupCon, so it can no longer dominate the shared backbone through the
     contrastive objective.

3. Preserve class-specific source selection
   - the stable gate only controls the globally unreliable source;
   - the relative PU_1/PU_2 class-wise weights are preserved and can be mildly
     sharpened instead of being forced to a uniform 50/50 split.

4. Keep the original V5 independent multi-loss formulation
   - MMD, CDD/L1, CDAN, target entropy, CLMMD, MCA and SupCon are independent
     terms; prototype mixing does not double-attenuate class alignment.

Recommended model name:
    MFSAN_CDAN_BIMAMBA_CW_RWCA_V6_STABLE_GATE
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V5_MCA import Trainer as V5Trainer


class Trainer(V5Trainer):
    """V5-MCA plus a stable automatic negative-source gate."""

    def __init__(self, args):
        super(Trainer, self).__init__(args)

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

        logging.info('Using model: MFSAN_CDAN_BIMAMBA_CW_RWCA_V6_STABLE_GATE')
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
        cur_epoch = int(getattr(self, '_cur_epoch', 1))
        if self.lambda_supcon <= 0.0 or cur_epoch < self.supcon_start_epoch:
            return torch.tensor(0.0, device=self.device)

        mode = self.supcon_feature_mode.upper()
        feat_list = f_s_all if mode == 'F' else g_s_list

        active_indices: List[int] = list(range(self.num_source))
        if self.v6_gate_apply_to_supcon:
            if self._v6_confirmed_negative_source >= 0:
                active_indices = [
                    k for k in active_indices
                    if k != self._v6_confirmed_negative_source
                ]
            else:
                source_w = self._v6_active_supcon_source_weights
                active_indices = [
                    k for k in active_indices
                    if float(source_w[k].item()) >= self.v6_supcon_source_min_weight
                ]

        if len(active_indices) == 0:
            return torch.tensor(0.0, device=self.device)

        features = torch.cat([feat_list[k] for k in active_indices], dim=0)
        labels = torch.cat([source_label_list[k] for k in active_indices], dim=0)
        features, labels = self._filter_supcon_features(features, labels)
        if features is None:
            return torch.tensor(0.0, device=self.device)
        return self.supcon_loss(features, labels)

    # ------------------------------------------------------------------
    # MCA: retain original fused-reference term and add active-source pairwise term
    # ------------------------------------------------------------------
    def _multi_classifier_alignment_loss(self, probs_t_all, probs_t_fused, class_src_weights):
        base_loss = super(Trainer, self)._multi_classifier_alignment_loss(
            probs_t_all=probs_t_all,
            probs_t_fused=probs_t_fused,
            class_src_weights=class_src_weights,
        )

        cur_epoch = int(getattr(self, '_cur_epoch', 1))
        if (
            self.lambda_mca <= 0.0
            or cur_epoch < self.mca_start_epoch
            or self.v6_mca_pairwise_weight <= 0.0
            or len(probs_t_all) <= 1
        ):
            return base_loss

        cw = self._normalize_class_source_weights(class_src_weights.float())
        src_w = self._source_weights_from_class_weights(cw).detach()
        active = [
            k for k in range(self.num_source)
            if float(src_w[k].item()) >= self.v6_supcon_source_min_weight
        ]
        if len(active) <= 1:
            return base_loss

        pair_num = torch.tensor(0.0, device=self.device)
        pair_den = torch.tensor(0.0, device=self.device)
        for pos_i in range(len(active)):
            for pos_j in range(pos_i + 1, len(active)):
                i, j = active[pos_i], active[pos_j]
                corr_i = self._normalize_class_correlation(probs_t_all[i])
                corr_j = self._normalize_class_correlation(probs_t_all[j])
                pair_weight = torch.sqrt(src_w[i] * src_w[j] + self.mca_eps)
                pair_num = pair_num + pair_weight * torch.mean((corr_i - corr_j) ** 2)
                pair_den = pair_den + pair_weight

        pair_loss = pair_num / (pair_den + self.mca_eps)
        return base_loss + self.v6_mca_pairwise_weight * pair_loss

    # ------------------------------------------------------------------
    # Main V5 training loop with the V6 stable gate inserted at the exact
    # points where source/class weights affect CE, fusion and adaptation.
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch_acc, epoch_loss):
        self._cur_epoch = int(getattr(self, '_cur_epoch', 1))

        self._v3_collect_epoch_weights = True
        self._v3_class_weight_sum = torch.zeros(
            self.num_source, self.num_classes, device=self.device
        )
        self._v3_class_weight_count = 0

        weight_sum = torch.zeros(self.num_source, device=self.device)
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
        mca_sum = torch.tensor(0.0, device=self.device)

        for batch_index in tqdm(range(self.num_iter), ascii=True):
            target_data, target_meta = self._get_next_batch('train')

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
            loss_cda_list = []
            loss_adv_list = []
            loss_clmmd_vec_list = []
            clmmd_valid_list = []
            ent_list = []
            domain_acc_list = []
            probs_t_all = []
            probs_s_all = []
            f_s_all = []
            f_t_all = []

            adv_tradeoff = self.tradeoff[2] if len(self.tradeoff) > 2 else 1.0
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
                loss_cls_vec_k = F.cross_entropy(y_s, labels_s, reduction='none')
                loss_mmd_k = self.mkmmd(f_s, f_t)
                loss_cda_k = self._conditional_mmd(f_s, f_t, p_s, p_t)
                loss_clmmd_vec_k, clmmd_valid_k = self._classwise_lmmd_per_class(
                    f_s, f_t, labels_s, p_t
                )
                loss_adv_k, domain_acc_k = self._domain_adversarial_loss(
                    cur_src_idx=k,
                    f_s=f_s,
                    f_t=f_t,
                    source_labels=labels_s,
                    prob_t=p_t,
                    grl_coeff=grl_coeff,
                )
                ent_k = self._entropy_scalar(p_t)

                loss_cls_vec_list.append(loss_cls_vec_k)
                loss_mmd_list.append(loss_mmd_k)
                loss_cda_list.append(loss_cda_k)
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

            # Raw V5 reliability.
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
            self._v6_active_supcon_source_weights = src_weights.detach()

            self._update_class_source_weight_ema(class_src_weights)
            raw_global_weight_sum += raw_global_src_weights.detach()
            effective_global_weight_sum += effective_global_src_weights.detach()
            raw_class_weight_sum += raw_class_src_weights.detach()
            rec_score_sum += rec_scores.detach()
            rec_guided_weight_sum += rec_guided_class_src_weights.detach()
            weight_sum += src_weights.detach()
            class_weight_sum += class_src_weights.detach()
            alpha_sum += float(cw_alpha_now)

            probs_t_fused = self._class_weighted_fusion(
                probs_t_all, class_src_weights
            )

            # Reliability-weighted source CE: PU_3 cannot keep one-third of the
            # shared-backbone supervision after it is confirmed negative.
            cls_num = torch.tensor(0.0, device=self.device)
            cls_den = torch.tensor(0.0, device=self.device)
            for k in range(self.num_source):
                sample_w = class_src_weights[k, source_label_list[k]]
                cls_num = cls_num + (loss_cls_vec_list[k] * sample_w).sum()
                cls_den = cls_den + sample_w.sum()
            loss_cls = cls_num / (cls_den + self.entropy_eps)

            loss_mmd = sum(
                src_weights[k] * loss_mmd_list[k]
                for k in range(self.num_source)
            )
            loss_cda = sum(
                src_weights[k] * loss_cda_list[k]
                for k in range(self.num_source)
            )
            loss_adv = sum(
                src_weights[k] * loss_adv_list[k]
                for k in range(self.num_source)
            )

            clmmd_num = torch.tensor(0.0, device=self.device)
            clmmd_den = torch.tensor(0.0, device=self.device)
            for k in range(self.num_source):
                valid = clmmd_valid_list[k]
                clmmd_num = clmmd_num + (
                    class_src_weights[k] * valid * loss_clmmd_vec_list[k]
                ).sum()
                clmmd_den = clmmd_den + (
                    class_src_weights[k] * valid
                ).sum()
            if clmmd_den.detach().item() > 0:
                loss_clmmd = clmmd_num / (clmmd_den + self.entropy_eps)
            else:
                loss_clmmd = torch.tensor(0.0, device=self.device)

            loss_l1 = torch.tensor(0.0, device=self.device)
            for k in range(self.num_source):
                abs_diff = torch.abs(
                    probs_t_all[k] - probs_t_fused.detach()
                )
                loss_l1 = loss_l1 + (
                    abs_diff * class_src_weights[k].view(1, -1)
                ).sum(dim=1).mean() / float(self.num_classes)

            loss_mca = self._multi_classifier_alignment_loss(
                probs_t_all=probs_t_all,
                probs_t_fused=probs_t_fused,
                class_src_weights=class_src_weights,
            )
            mca_sum += loss_mca.detach()

            loss_ent = self._target_entropy(probs_t_fused)
            domain_acc = sum(
                src_weights[k] * domain_acc_list[k]
                for k in range(self.num_source)
            )

            loss_supcon = self._compute_source_supcon_loss(
                g_s_list, f_s_all, source_label_list
            )
            supcon_sum += loss_supcon.detach()

            class_boost = self.v6_class_alignment_boost
            new_tradeoff = adv_tradeoff
            loss = (
                loss_cls
                + self.tradeoff[0] * loss_mmd
                + self.tradeoff[1] * loss_l1
                + new_tradeoff * self.lambda_cda * loss_cda
                + new_tradeoff * self.lambda_ent * loss_ent
                + new_tradeoff * self.lambda_adv * loss_adv
                + class_boost * new_tradeoff * self.lambda_clmmd * loss_clmmd
                + class_boost * new_tradeoff * self.lambda_mca * loss_mca
                + self.lambda_supcon * loss_supcon
            )

            epoch_acc['Domain Data'] += domain_acc.detach().item()
            epoch_loss['Source Classifier'] += loss_cls.detach()
            epoch_loss['MMD'] += loss_mmd.detach()
            epoch_loss['CDD/L1'] += loss_l1.detach()
            epoch_loss['CDA MMD'] += loss_cda.detach()
            epoch_loss['CLMMD'] += loss_clmmd.detach()
            epoch_loss['Target Entropy'] += loss_ent.detach()
            epoch_loss['CDAN Domain'] += loss_adv.detach()
            epoch_loss['CDA Weighted'] += (
                new_tradeoff * self.lambda_cda * loss_cda
            ).detach()
            epoch_loss['CLMMD Weighted'] += (
                class_boost * new_tradeoff * self.lambda_clmmd * loss_clmmd
            ).detach()
            epoch_loss['Entropy Weighted'] += (
                new_tradeoff * self.lambda_ent * loss_ent
            ).detach()
            epoch_loss['CDAN Weighted'] += (
                new_tradeoff * self.lambda_adv * loss_adv
            ).detach()
            epoch_loss['MCA'] += loss_mca.detach()
            epoch_loss['MCA Weighted'] += (
                class_boost * new_tradeoff * self.lambda_mca * loss_mca
            ).detach()
            epoch_loss['SupCon'] += loss_supcon.detach()
            epoch_loss['SupCon Weighted'] += (
                self.lambda_supcon * loss_supcon
            ).detach()
            epoch_loss['CW Alpha'] += torch.tensor(
                float(cw_alpha_now), device=self.device
            )

            for k in range(self.num_source):
                epoch_loss[f'Raw Global Prior src{k}'] += raw_global_src_weights[k].detach()
                epoch_loss[f'Effective Global Prior src{k}'] += effective_global_src_weights[k].detach()
                epoch_loss[f'RW Weight src{k}'] += src_weights[k].detach()
            for c in self._cw_log_classes:
                for k in range(self.num_source):
                    epoch_loss[f'Raw CW Weight c{c} src{k}'] += raw_class_src_weights[k, c].detach()
                    epoch_loss[f'Rec Score c{c} src{k}'] += rec_scores[k, c].detach()
                    epoch_loss[f'Rec-Guided CW Weight c{c} src{k}'] += rec_guided_class_src_weights[k, c].detach()
                    epoch_loss[f'V6 Final CW Weight c{c} src{k}'] += class_src_weights[k, c].detach()

            loss.backward()
            self.optimizer.step()

        denom_iter = max(float(self.num_iter), 1.0)
        avg_weights = (weight_sum / denom_iter).detach()
        avg_raw_global = (raw_global_weight_sum / denom_iter).detach()
        avg_effective_global = (effective_global_weight_sum / denom_iter).detach()
        avg_raw_class = (raw_class_weight_sum / denom_iter).detach()
        avg_class = (class_weight_sum / denom_iter).detach()
        avg_rec_scores = (rec_score_sum / denom_iter).detach()
        avg_rec_guided = (rec_guided_weight_sum / denom_iter).detach()
        avg_alpha = alpha_sum / denom_iter
        avg_supcon = (supcon_sum / denom_iter).detach().item()
        avg_mca = (mca_sum / denom_iter).detach().item()

        self._v6_last_effective_global_weights = avg_effective_global.clone()
        self._update_stable_gate_from_epoch(avg_raw_global)

        logging.info(
            'V6 raw global source weights: {}'.format(
                ', '.join(
                    '{}={:.4f}'.format(self._source_name(i), avg_raw_global[i].item())
                    for i in range(self.num_source)
                )
            )
        )
        logging.info(
            'V6 effective global source weights: {}'.format(
                ', '.join(
                    '{}={:.4f}'.format(self._source_name(i), avg_effective_global[i].item())
                    for i in range(self.num_source)
                )
            )
        )
        logging.info(
            'V6 final class-averaged source weights: {}'.format(
                ', '.join(
                    '{}={:.4f}'.format(self._source_name(i), avg_weights[i].item())
                    for i in range(self.num_source)
                )
            )
        )
        logging.info(
            'V6 CW alpha average: {:.4f} | warmup={} | alpha_max={:.4f} | ramp={}'.format(
                avg_alpha,
                self.cw_warmup_epochs,
                self.cw_alpha,
                self.cw_alpha_ramp_epochs,
            )
        )
        for c in self._cw_log_classes:
            logging.info(
                'V6 class-{} raw transfer weights: {}'.format(
                    c,
                    ', '.join(
                        '{}={:.4f}'.format(self._source_name(i), avg_raw_class[i, c].item())
                        for i in range(self.num_source)
                    ),
                )
            )
            logging.info(
                'V6 class-{} recognition scores: {}'.format(
                    c,
                    ', '.join(
                        '{}={:.4f}'.format(self._source_name(i), avg_rec_scores[i, c].item())
                        for i in range(self.num_source)
                    ),
                )
            )
            logging.info(
                'V6 class-{} rec-guided weights: {}'.format(
                    c,
                    ', '.join(
                        '{}={:.4f}'.format(self._source_name(i), avg_rec_guided[i, c].item())
                        for i in range(self.num_source)
                    ),
                )
            )
            logging.info(
                'V6 class-{} final gated weights: {}'.format(
                    c,
                    ', '.join(
                        '{}={:.4f}'.format(self._source_name(i), avg_class[i, c].item())
                        for i in range(self.num_source)
                    ),
                )
            )

        logging.info(
            'V6 MCA average: {:.6f} | weighted={:.6f} | class_alignment_boost={:.4f}'.format(
                avg_mca,
                self.v6_class_alignment_boost * self.lambda_mca * avg_mca,
                self.v6_class_alignment_boost,
            )
        )
        logging.info(
            'V6 SupCon average: {:.6f} | weighted={:.6f} | focus_classes={} | confirmed_negative={}'.format(
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
