# -*- coding: utf-8 -*-
"""
MFSAN-CDAN-BiMamba-CW-RWCA-V9-SpecialistRadiusAlignment

V9 is a conservative correction of V8 built from the V7 result analysis.
It keeps V7 as the main structure and retains the same five-term objective:

    weighted CE + MMD + CDAN + CLMMD + SupCon

The target-test evaluation and best-checkpoint protocol are unchanged.

Compared with V8, V9 makes three targeted corrections:

1. Historical class-specialist protection
   A source that is repeatedly the strongest source for one class is memorized
   as a class specialist. The class gate is prevented from hard-suppressing
   that source for the protected class. This is designed to preserve useful
   specialists such as the PU3 expert behavior observed for KI04 in V7.

2. Radius-aware prototype filtering
   V8 only checked nearest-prototype agreement and a top1-top2 margin. In the
   observed run all candidates passed. V9 additionally estimates the source
   class compactness (cosine-distance mean/variance) and rejects target samples
   that lie outside a source-derived class radius. Accepted samples receive a
   distance-based soft weight in CLMMD.

3. Delayed, softly-ramped confusion-pair SupCon
   Hard-negative weighting starts after the V7 class gate has had time to find
   class specialists, and grows gradually from 1.0 to a mild final weight.
   The default run only emphasizes K001-KA30 and KA30-KB23, avoiding an early
   forced separation of KA30-KI04 that harmed KI04 in V8.
"""

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V7_CLASS_GATE_CONFLICT_FUSION import (
    Trainer as V7Trainer,
)
from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V8_CONFUSION_PROTO_ALIGNMENT import (
    Trainer as V8Trainer,
)


class Trainer(V8Trainer):
    """V9 trainer: V8 corrected with specialist memory and radius filtering."""

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # ------------------------------------------------------------------
        # V9 specialist memory/protection
        # ------------------------------------------------------------------
        self.v9_specialist_protection_enabled = bool(
            getattr(args, 'v9_specialist_protection_enabled', True)
        )
        self.v9_specialist_start_epoch = max(
            1, int(getattr(args, 'v9_specialist_start_epoch', 4))
        )
        self.v9_specialist_confirm_epochs = max(
            1, int(getattr(args, 'v9_specialist_confirm_epochs', 2))
        )
        self.v9_specialist_min_weight = min(
            max(float(getattr(args, 'v9_specialist_min_weight', 0.45)), 0.0),
            1.0,
        )
        self.v9_specialist_min_gap = max(
            0.0, float(getattr(args, 'v9_specialist_min_gap', 0.10))
        )
        self.v9_specialist_floor = min(
            max(float(getattr(args, 'v9_specialist_floor', 0.05)), 0.0),
            0.49,
        )
        self.v9_specialist_release_weight = min(
            max(float(getattr(args, 'v9_specialist_release_weight', 0.15)), 0.0),
            1.0,
        )
        self.v9_specialist_release_epochs = max(
            1, int(getattr(args, 'v9_specialist_release_epochs', 4))
        )

        self._v9_specialist_candidate_source = torch.full(
            (self.num_classes,), -1, dtype=torch.long, device=self.device
        )
        self._v9_specialist_candidate_streak = torch.zeros(
            self.num_classes, dtype=torch.long, device=self.device
        )
        self._v9_protected_specialist_source = torch.full(
            (self.num_classes,), -1, dtype=torch.long, device=self.device
        )
        self._v9_specialist_release_streak = torch.zeros(
            self.num_classes, dtype=torch.long, device=self.device
        )

        # ------------------------------------------------------------------
        # V9 delayed/ramped hard-negative SupCon
        # ------------------------------------------------------------------
        self.v9_hard_supcon_start_epoch = max(
            1, int(getattr(args, 'v9_hard_supcon_start_epoch', 8))
        )
        self.v9_hard_supcon_ramp_epochs = max(
            1, int(getattr(args, 'v9_hard_supcon_ramp_epochs', 4))
        )

        # ------------------------------------------------------------------
        # V9 radius-aware prototype filtering
        # ------------------------------------------------------------------
        self.v9_prototype_filter_classes = self._parse_class_list(
            getattr(args, 'v9_prototype_filter_classes', '0,3,4'),
            allow_all=True,
        )
        if self.v9_prototype_filter_classes is None:
            self.v9_prototype_filter_classes = list(range(self.num_classes))
        self.v9_prototype_filter_class_set = set(
            int(c) for c in self.v9_prototype_filter_classes
        )

        self.v9_radius_ema_momentum = min(
            max(float(getattr(args, 'v9_radius_ema_momentum', 0.90)), 0.0),
            0.9999,
        )
        self.v9_radius_std_scale = max(
            0.0, float(getattr(args, 'v9_radius_std_scale', 2.0))
        )
        self.v9_radius_min = max(
            0.0, float(getattr(args, 'v9_radius_min', 0.03))
        )
        self.v9_radius_max = max(
            self.v9_radius_min,
            float(getattr(args, 'v9_radius_max', 0.30)),
        )
        self.v9_prototype_min_similarity = min(
            max(float(getattr(args, 'v9_prototype_min_similarity', 0.30)), -1.0),
            1.0,
        )
        self.v9_prototype_soft_tau = max(
            float(getattr(args, 'v9_prototype_soft_tau', 0.10)), 1e-6
        )

        shape = (self.num_source, self.num_classes)
        self._v9_radius_mean = torch.zeros(*shape, device=self.device)
        self._v9_radius_var = torch.zeros(*shape, device=self.device)
        self._v9_radius_updates = torch.zeros(
            *shape, dtype=torch.long, device=self.device
        )

        logging.info(
            'Using model: '
            'MFSAN_CDAN_BIMAMBA_CW_RWCA_V9_SPECIALIST_RADIUS_ALIGNMENT'
        )
        logging.info(
            'V9 specialist protection: enabled={} start={} confirm={} '
            'min_weight={:.3f} min_gap={:.3f} floor={:.3f} '
            'release_weight={:.3f} release_epochs={}'.format(
                self.v9_specialist_protection_enabled,
                self.v9_specialist_start_epoch,
                self.v9_specialist_confirm_epochs,
                self.v9_specialist_min_weight,
                self.v9_specialist_min_gap,
                self.v9_specialist_floor,
                self.v9_specialist_release_weight,
                self.v9_specialist_release_epochs,
            )
        )
        logging.info(
            'V9 radius prototype filter: classes={} start={} margin={:.3f} '
            'radius_scale={:.3f} radius_range=[{:.3f},{:.3f}] '
            'min_similarity={:.3f} soft_tau={:.3f}'.format(
                self.v9_prototype_filter_classes,
                self.v8_prototype_start_epoch,
                self.v8_prototype_margin,
                self.v9_radius_std_scale,
                self.v9_radius_min,
                self.v9_radius_max,
                self.v9_prototype_min_similarity,
                self.v9_prototype_soft_tau,
            )
        )
        logging.info(
            'V9 delayed Hard-SupCon: start={} ramp={} final_weight={:.3f} '
            'pairs={}'.format(
                self.v9_hard_supcon_start_epoch,
                self.v9_hard_supcon_ramp_epochs,
                self.v8_hard_negative_weight,
                self._format_pair_set(self.v8_hard_negative_pairs),
            )
        )
        logging.info(
            'V9 effective objective remains five terms: '
            'CE + MMD + CDAN + radius-filtered CLMMD + soft Hard-SupCon.'
        )
        logging.info(
            'Evaluation policy: eval_each_epoch={} select_best_on_target={}'.format(
                bool(getattr(args, 'eval_each_epoch', True)),
                bool(getattr(args, 'select_best_on_target', True)),
            )
        )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def _checkpoint_dict(self):
        ckpt = super(Trainer, self)._checkpoint_dict()
        ckpt.update({
            'v9_specialist_radius_alignment': True,
            'v9_specialist_candidate_source': (
                self._v9_specialist_candidate_source.detach().cpu()
            ),
            'v9_specialist_candidate_streak': (
                self._v9_specialist_candidate_streak.detach().cpu()
            ),
            'v9_protected_specialist_source': (
                self._v9_protected_specialist_source.detach().cpu()
            ),
            'v9_specialist_release_streak': (
                self._v9_specialist_release_streak.detach().cpu()
            ),
            'v9_radius_mean': self._v9_radius_mean.detach().cpu(),
            'v9_radius_var': self._v9_radius_var.detach().cpu(),
            'v9_radius_updates': self._v9_radius_updates.detach().cpu(),
        })
        return ckpt

    def load_model(self):
        super(Trainer, self).load_model()
        ckpt = torch.load(self.args.load_path, map_location=self.device)

        def _load_vec(name, current):
            value = ckpt.get(name, None)
            if value is None:
                return current
            value = value.to(self.device, dtype=current.dtype)
            return value if value.shape == current.shape else current

        self._v9_specialist_candidate_source = _load_vec(
            'v9_specialist_candidate_source',
            self._v9_specialist_candidate_source,
        )
        self._v9_specialist_candidate_streak = _load_vec(
            'v9_specialist_candidate_streak',
            self._v9_specialist_candidate_streak,
        )
        self._v9_protected_specialist_source = _load_vec(
            'v9_protected_specialist_source',
            self._v9_protected_specialist_source,
        )
        self._v9_specialist_release_streak = _load_vec(
            'v9_specialist_release_streak',
            self._v9_specialist_release_streak,
        )
        self._v9_radius_mean = _load_vec('v9_radius_mean', self._v9_radius_mean)
        self._v9_radius_var = _load_vec('v9_radius_var', self._v9_radius_var)
        self._v9_radius_updates = _load_vec(
            'v9_radius_updates', self._v9_radius_updates
        )
        logging.info(
            'Loaded V9 protected specialists: %s',
            self._format_v9_protected_specialists(),
        )

    # ------------------------------------------------------------------
    # Specialist memory/protection
    # ------------------------------------------------------------------
    def _format_v9_protected_specialists(self) -> str:
        items: List[str] = []
        for c in range(self.num_classes):
            source_idx = int(self._v9_protected_specialist_source[c].item())
            if source_idx >= 0:
                items.append('c{}={}'.format(c, self._source_name(source_idx)))
        return ', '.join(items) if items else 'none'

    @torch.no_grad()
    def _update_v9_specialist_memory(self, epoch_weights: torch.Tensor) -> None:
        if not self.v9_specialist_protection_enabled:
            return
        epoch = int(getattr(self, '_cur_epoch', 1))
        if epoch < self.v9_specialist_start_epoch:
            return

        cw = self._normalize_class_source_weights(epoch_weights.detach())
        for c in range(self.num_classes):
            vec = cw[:, c]
            sorted_w, sorted_idx = torch.sort(vec, descending=True)
            best = int(sorted_idx[0].item())
            best_value = float(sorted_w[0].item())
            second_value = (
                float(sorted_w[1].item()) if self.num_source > 1 else 0.0
            )
            gap = best_value - second_value
            protected = int(self._v9_protected_specialist_source[c].item())

            if protected < 0:
                eligible = (
                    best_value >= self.v9_specialist_min_weight
                    and gap >= self.v9_specialist_min_gap
                )
                if eligible:
                    if int(self._v9_specialist_candidate_source[c].item()) == best:
                        self._v9_specialist_candidate_streak[c] += 1
                    else:
                        self._v9_specialist_candidate_source[c] = best
                        self._v9_specialist_candidate_streak[c] = 1
                    if int(self._v9_specialist_candidate_streak[c].item()) >= (
                        self.v9_specialist_confirm_epochs
                    ):
                        self._v9_protected_specialist_source[c] = best
                        self._v9_specialist_release_streak[c] = 0
                else:
                    self._v9_specialist_candidate_source[c] = -1
                    self._v9_specialist_candidate_streak[c] = 0
            else:
                protected_weight = float(vec[protected].item())
                if protected_weight < self.v9_specialist_release_weight:
                    self._v9_specialist_release_streak[c] += 1
                    if int(self._v9_specialist_release_streak[c].item()) >= (
                        self.v9_specialist_release_epochs
                    ):
                        self._v9_protected_specialist_source[c] = -1
                        self._v9_specialist_candidate_source[c] = -1
                        self._v9_specialist_candidate_streak[c] = 0
                        self._v9_specialist_release_streak[c] = 0
                else:
                    self._v9_specialist_release_streak[c] = 0

    def _update_class_gate_from_epoch(self, epoch_weights: torch.Tensor) -> None:
        # Memorize specialists before the V7 gate changes any state.
        self._update_v9_specialist_memory(epoch_weights)
        super(Trainer, self)._update_class_gate_from_epoch(epoch_weights)

        # A protected specialist cannot simultaneously be the class's hard-
        # suppressed source. Clear both candidate and confirmed suppression.
        if self.v9_specialist_protection_enabled:
            for c in range(self.num_classes):
                protected = int(self._v9_protected_specialist_source[c].item())
                if protected < 0:
                    continue
                if int(self._v7_class_confirmed_source[c].item()) == protected:
                    self._v7_class_confirmed_source[c] = -1
                    self._v7_class_release_streak[c] = 0
                if int(self._v7_class_candidate_source[c].item()) == protected:
                    self._v7_class_candidate_source[c] = -1
                    self._v7_class_candidate_streak[c] = 0

        logging.info(
            'V9 protected specialist map: %s',
            self._format_v9_protected_specialists(),
        )

    def _apply_stable_gate_matrix(self, class_weights: torch.Tensor) -> torch.Tensor:
        out = super(Trainer, self)._apply_stable_gate_matrix(class_weights)
        if not self.v9_specialist_protection_enabled:
            return out

        out = out.clone()
        for c in range(self.num_classes):
            protected = int(self._v9_protected_specialist_source[c].item())
            if protected < 0:
                continue
            vec = out[:, c]
            if float(vec[protected].item()) < self.v9_specialist_floor:
                keep = torch.ones(
                    self.num_source, dtype=torch.bool, device=vec.device
                )
                keep[protected] = False
                vec[protected] = self.v9_specialist_floor
                if bool(keep.any()):
                    vec[keep] = (
                        (1.0 - self.v9_specialist_floor)
                        * vec[keep]
                        / (vec[keep].sum() + self.entropy_eps)
                    )
                out[:, c] = self._normalize_source_vector(vec)

        out = self._normalize_class_source_weights(out)
        self._v7_active_supcon_class_weights = out.detach().clone()
        return out.detach() if self.rw_detach_weights else out

    # ------------------------------------------------------------------
    # Source class radius estimation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_v8_source_prototypes(
        self,
        source_idx: int,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        super(Trainer, self)._update_v8_source_prototypes(
            source_idx, features, labels
        )
        if not (0 <= source_idx < self.num_source):
            return
        if self._v8_source_class_prototypes is None:
            return

        momentum = self.v9_radius_ema_momentum
        feature_norm = F.normalize(features.detach(), p=2, dim=1)
        for c in range(self.num_classes):
            mask = labels == c
            if int(mask.sum().item()) == 0:
                continue
            proto = F.normalize(
                self._v8_source_class_prototypes[source_idx, c].view(1, -1),
                p=2,
                dim=1,
            ).view(-1)
            cosine = torch.matmul(feature_norm[mask], proto)
            distance = (1.0 - cosine).clamp_min(0.0)
            batch_mean = distance.mean()
            batch_var = distance.var(unbiased=False)
            updates = int(self._v9_radius_updates[source_idx, c].item())
            if updates == 0:
                self._v9_radius_mean[source_idx, c] = batch_mean
                self._v9_radius_var[source_idx, c] = batch_var
            else:
                self._v9_radius_mean[source_idx, c] = (
                    momentum * self._v9_radius_mean[source_idx, c]
                    + (1.0 - momentum) * batch_mean
                )
                self._v9_radius_var[source_idx, c] = (
                    momentum * self._v9_radius_var[source_idx, c]
                    + (1.0 - momentum) * batch_var
                )
            self._v9_radius_updates[source_idx, c] += 1

    def _v9_radius_threshold(self, source_idx: int, class_idx: int) -> torch.Tensor:
        mean = self._v9_radius_mean[source_idx, class_idx]
        std = torch.sqrt(self._v9_radius_var[source_idx, class_idx].clamp_min(0.0))
        threshold = mean + self.v9_radius_std_scale * std
        return threshold.clamp(self.v9_radius_min, self.v9_radius_max)

    def _reset_v8_prototype_epoch_stats(self) -> None:
        super(Trainer, self)._reset_v8_prototype_epoch_stats()
        shape = (self.num_source, self.num_classes)
        self._v9_proto_reject_radius = torch.zeros(
            *shape, dtype=torch.long, device=self.device
        )
        self._v9_proto_reject_similarity = torch.zeros(
            *shape, dtype=torch.long, device=self.device
        )
        self._v9_proto_radius_threshold_sum = torch.zeros(
            *shape, device=self.device
        )
        self._v9_proto_radius_threshold_count = torch.zeros(
            *shape, dtype=torch.long, device=self.device
        )

    # ------------------------------------------------------------------
    # Radius-aware prototype-filtered CLMMD
    # ------------------------------------------------------------------
    def _classwise_lmmd_per_class(self, f_s, f_t, labels_s, probs_t):
        source_idx = int(getattr(self, '_active_source_idx_for_clmmd', -1))
        self._update_v8_source_prototypes(source_idx, f_s, labels_s)

        # Start from V7 confidence-only CLMMD. Non-targeted classes remain
        # unchanged except for the configured V8/V9 class multiplier.
        base_loss, base_valid = V7Trainer._classwise_lmmd_per_class(
            self, f_s, f_t, labels_s, probs_t
        )
        loss_vec = self._boost_clmmd_vector(base_loss)
        valid_vec = base_valid.clone()

        epoch = int(getattr(self, '_cur_epoch', 1))
        valid_source = 0 <= source_idx < self.num_source
        if (
            not self.v8_prototype_filter_enabled
            or epoch < self.v8_prototype_start_epoch
            or not valid_source
            or self._v8_source_class_prototypes is None
            or not self.v9_prototype_filter_class_set
        ):
            return loss_vec, valid_vec

        proto_valid = (
            self._v8_source_class_prototype_updates[source_idx]
            >= self.v8_prototype_min_updates
        ) & (
            self._v9_radius_updates[source_idx]
            >= self.v8_prototype_min_updates
        )
        if int(proto_valid.sum().item()) < 2:
            return loss_vec, valid_vec

        device = f_s.device
        probs_t = torch.clamp(probs_t, min=self.entropy_eps, max=1.0)
        probs_t = probs_t / (
            probs_t.sum(dim=1, keepdim=True) + self.entropy_eps
        )
        max_prob, pseudo_label = probs_t.max(dim=1)

        target_norm = F.normalize(f_t, p=2, dim=1)
        prototypes = F.normalize(
            self._v8_source_class_prototypes[source_idx], p=2, dim=1
        )
        similarity = torch.matmul(target_norm, prototypes.t())
        similarity[:, ~proto_valid] = -1e9
        top2_sim, top2_cls = similarity.topk(k=2, dim=1)
        proto_label = top2_cls[:, 0]
        proto_margin = top2_sim[:, 0] - top2_sim[:, 1]
        proto_distance = (1.0 - top2_sim[:, 0]).clamp_min(0.0)

        for c in self.v9_prototype_filter_classes:
            if not (0 <= c < self.num_classes):
                continue
            # Replace the confidence-only value for targeted classes.
            loss_vec[c] = 0.0
            valid_vec[c] = 0.0

            mask_s = labels_s == c
            n_s = int(mask_s.sum().item())
            if n_s < self.clmmd_min_source:
                continue

            conf_threshold = float(
                self.v8_prototype_conf_thresholds[c].item()
            )
            mask_conf = (pseudo_label == c) & (max_prob >= conf_threshold)
            candidate_count = int(mask_conf.sum().item())
            self._v8_proto_conf_candidates[source_idx, c] += candidate_count
            if candidate_count == 0:
                continue

            if not bool(proto_valid[c].item()):
                self._v8_proto_reject_disagree[source_idx, c] += candidate_count
                continue

            agree = proto_label == c
            margin_ok = proto_margin >= self.v8_prototype_margin
            similarity_ok = top2_sim[:, 0] >= self.v9_prototype_min_similarity
            radius_threshold = self._v9_radius_threshold(source_idx, c)
            radius_ok = proto_distance <= radius_threshold

            mask_t = mask_conf & agree & margin_ok & similarity_ok & radius_ok

            disagree_count = int((mask_conf & ~agree).sum().item())
            margin_reject_count = int(
                (mask_conf & agree & ~margin_ok).sum().item()
            )
            similarity_reject_count = int(
                (mask_conf & agree & margin_ok & ~similarity_ok).sum().item()
            )
            radius_reject_count = int(
                (
                    mask_conf
                    & agree
                    & margin_ok
                    & similarity_ok
                    & ~radius_ok
                ).sum().item()
            )
            accept_count = int(mask_t.sum().item())

            self._v8_proto_reject_disagree[source_idx, c] += disagree_count
            self._v8_proto_reject_margin[source_idx, c] += margin_reject_count
            self._v9_proto_reject_similarity[
                source_idx, c
            ] += similarity_reject_count
            self._v9_proto_reject_radius[source_idx, c] += radius_reject_count
            self._v8_proto_accepted[source_idx, c] += accept_count
            self._v9_proto_radius_threshold_sum[source_idx, c] += (
                radius_threshold.detach()
            )
            self._v9_proto_radius_threshold_count[source_idx, c] += 1

            if accept_count < self.pl_min_target:
                continue

            distance_soft = torch.exp(
                -proto_distance[mask_t] / self.v9_prototype_soft_tau
            ).clamp_min(1e-4)
            wt_raw = probs_t[mask_t, c] * distance_soft
            wt_sum = wt_raw.sum()
            if wt_sum.detach().item() < self.clmmd_min_target_weight:
                continue

            xs = f_s[mask_s]
            xt = f_t[mask_t]
            k_ss, k_st, k_tt = self._gaussian_kernel_matrix(xs, xt)
            ws = torch.ones(n_s, device=device) / float(n_s)
            wt = wt_raw / (wt_sum + self.entropy_eps)
            loss_c = (
                torch.sum(ws.view(-1, 1) * ws.view(1, -1) * k_ss)
                + torch.sum(wt.view(-1, 1) * wt.view(1, -1) * k_tt)
                - 2.0
                * torch.sum(ws.view(-1, 1) * wt.view(1, -1) * k_st)
            )
            loss_vec[c] = loss_c * self.v8_clmmd_class_boost[c]
            valid_vec[c] = 1.0

        return loss_vec, valid_vec

    # ------------------------------------------------------------------
    # Delayed/ramped Hard-negative SupCon
    # ------------------------------------------------------------------
    def _current_v9_hard_weight(self) -> float:
        epoch = int(getattr(self, '_cur_epoch', 1))
        if epoch < self.v9_hard_supcon_start_epoch:
            return 1.0
        progress = (
            epoch - self.v9_hard_supcon_start_epoch + 1
        ) / float(self.v9_hard_supcon_ramp_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 1.0 + progress * (self.v8_hard_negative_weight - 1.0)

    def _build_hard_pair_matrix(self, labels: torch.Tensor) -> torch.Tensor:
        n = labels.numel()
        pair_weights = torch.ones(
            n, n, device=labels.device, dtype=torch.float32
        )
        hard_weight = self._current_v9_hard_weight()
        if (
            not self.v8_hard_supcon_enabled
            or not self.v8_hard_negative_pairs
            or hard_weight <= 1.0 + 1e-12
        ):
            return pair_weights
        for a, b in self.v8_hard_negative_pairs:
            hard = (
                ((labels.view(-1, 1) == a) & (labels.view(1, -1) == b))
                | ((labels.view(-1, 1) == b) & (labels.view(1, -1) == a))
            )
            pair_weights[hard] = hard_weight
        return pair_weights

    # ------------------------------------------------------------------
    # Epoch diagnostics
    # ------------------------------------------------------------------
    def _finalize_v3_epoch_weights(self):
        # V8 resets its statistics inside super(), so preserve a copy first.
        radius_reject = self._v9_proto_reject_radius.detach().clone()
        similarity_reject = self._v9_proto_reject_similarity.detach().clone()
        threshold_sum = self._v9_proto_radius_threshold_sum.detach().clone()
        threshold_count = self._v9_proto_radius_threshold_count.detach().clone()

        super(Trainer, self)._finalize_v3_epoch_weights()

        logging.info(
            'V9 current Hard-SupCon pair weight: %.4f',
            self._current_v9_hard_weight(),
        )
        for c in self.v9_prototype_filter_classes:
            for k in range(self.num_source):
                count = int(threshold_count[k, c].item())
                avg_threshold = (
                    float(threshold_sum[k, c].item()) / float(max(count, 1))
                )
                radius_mean = float(self._v9_radius_mean[k, c].item())
                radius_std = float(
                    torch.sqrt(self._v9_radius_var[k, c].clamp_min(0.0)).item()
                )
                logging.info(
                    'V9 radius filter src{} class-{}: reject_radius={} '
                    'reject_similarity={} source_radius_mean={:.5f} '
                    'source_radius_std={:.5f} accept_radius={:.5f}'.format(
                        k,
                        c,
                        int(radius_reject[k, c].item()),
                        int(similarity_reject[k, c].item()),
                        radius_mean,
                        radius_std,
                        avg_threshold,
                    )
                )


__all__ = ['Trainer']
