# -*- coding: utf-8 -*-
"""V12: stable CWRU_1 class-rescue strategy without changing network structure.

This trainer keeps the exact V10 network modules and the same five loss terms.
Only the training policy for the existing class-wise MMD term is changed.

The CWRU_1 logs showed that the former V11 rescue accepted hundreds of target
samples in later epochs, while class-2 precision/recall remained poor and the
normal class collapsed.  V12 therefore changes rescue from "many permissive
pseudo labels" to "few high-quality pseudo labels":

* delayed, time-limited rescue;
* rescue from the configured class specialist only;
* probability-ratio and normal-probability guards;
* explicit separation from the normal-class prototype;
* stricter prototype similarity/margin/radius requirements;
* top-score cap per mini-batch;
* conservative blending with the original CLMMD value instead of replacement;
* no evaluation-time probability rewriting by default.

No convolution, BiMamba, feature extractor, classifier, discriminator, or
fusion architecture is added, removed, or resized.
"""

import logging
from typing import List

import torch
import torch.nn.functional as F

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V10_PAIRWISE_SPECIALIST_CALIBRATED import (
    Trainer as V10Trainer,
)


class Trainer(V10Trainer):
    """V10 architecture with a stricter class-2 training policy."""

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        self.v12_rescue_enabled = bool(
            getattr(args, 'v12_rescue_enabled', False)
        )
        self.v12_rescue_class = int(getattr(args, 'v12_rescue_class', 2))
        self.v12_normal_class = int(getattr(args, 'v12_normal_class', 6))
        self.v12_confusion_classes = self._parse_v12_class_list(
            getattr(args, 'v12_confusion_classes', '0,1'),
            excluded={self.v12_normal_class},
        )
        self.v12_rescue_source_indices = self._parse_source_list(
            getattr(args, 'v12_rescue_source_indices', '2')
        )

        self.v12_rescue_start_epoch = max(
            1, int(getattr(args, 'v12_rescue_start_epoch', 6))
        )
        self.v12_rescue_end_epoch = max(
            self.v12_rescue_start_epoch,
            int(getattr(args, 'v12_rescue_end_epoch', 10)),
        )
        self.v12_rescue_topk = max(
            1, int(getattr(args, 'v12_rescue_topk', 2))
        )
        self.v12_rescue_min_class_prob = self._clip01(
            getattr(args, 'v12_rescue_min_class_prob', 0.20)
        )
        self.v12_rescue_min_competitor_ratio = max(
            0.0,
            float(getattr(args, 'v12_rescue_min_competitor_ratio', 0.50)),
        )
        self.v12_rescue_max_normal_prob = self._clip01(
            getattr(args, 'v12_rescue_max_normal_prob', 0.20)
        )
        self.v12_rescue_proto_margin = max(
            0.0, float(getattr(args, 'v12_rescue_proto_margin', 0.08))
        )
        self.v12_rescue_normal_proto_margin = max(
            0.0,
            float(getattr(args, 'v12_rescue_normal_proto_margin', 0.10)),
        )
        self.v12_rescue_min_similarity = min(
            max(float(getattr(args, 'v12_rescue_min_similarity', 0.45)), -1.0),
            1.0,
        )
        self.v12_rescue_radius_cap = max(
            0.0, float(getattr(args, 'v12_rescue_radius_cap', 0.05))
        )
        self.v12_rescue_max_per_batch = max(
            1, int(getattr(args, 'v12_rescue_max_per_batch', 4))
        )
        self.v12_rescue_min_target = max(
            1, int(getattr(args, 'v12_rescue_min_target', 2))
        )
        self.v12_rescue_mix_alpha = min(
            max(float(getattr(args, 'v12_rescue_mix_alpha', 0.25)), 0.0),
            1.0,
        )
        self.v12_rescue_clmmd_boost = max(
            0.0, float(getattr(args, 'v12_rescue_clmmd_boost', 1.10))
        )
        self.v12_rescue_score_tau = max(
            1e-4, float(getattr(args, 'v12_rescue_score_tau', 0.10))
        )

        # Evaluation-time rewriting was ineffective in V11 and made training
        # and inference inconsistent.  It remains optional, but is off by
        # default and the dedicated runner keeps it disabled.
        self.v12_eval_rescue_enabled = bool(
            getattr(args, 'v12_eval_rescue_enabled', False)
        )
        self.v12_eval_min_class_prob = self._clip01(
            getattr(args, 'v12_eval_min_class_prob', 0.20)
        )
        self.v12_eval_competitor_ratio = max(
            0.0, float(getattr(args, 'v12_eval_competitor_ratio', 0.60))
        )
        self.v12_eval_min_source_votes = max(
            1, int(getattr(args, 'v12_eval_min_source_votes', 2))
        )
        self.v12_eval_boost = max(
            1.0, float(getattr(args, 'v12_eval_boost', 1.25))
        )

        self._reset_v12_epoch_stats()
        self._v12_eval_total = 0
        self._v12_eval_candidates = 0
        self._v12_eval_changed = 0

        logging.info(
            'Using model: MFSAN_CDAN_BIMAMBA_CW_RWCA_V12_CWRU1_STABLE_RESCUE'
        )
        logging.info(
            'V12 stable rescue: enabled={} class={} normal={} confusion={} '
            'sources={} epoch=[{},{}] topk={} min_prob={:.3f} ratio={:.3f} '
            'max_normal_prob={:.3f} proto_margin={:.3f} '
            'normal_proto_margin={:.3f} min_similarity={:.3f} '
            'radius_cap={:.3f} max_per_batch={} min_target={} mix_alpha={:.3f} '
            'clmmd_boost={:.3f}'.format(
                self.v12_rescue_enabled,
                self.v12_rescue_class,
                self.v12_normal_class,
                self.v12_confusion_classes,
                self.v12_rescue_source_indices,
                self.v12_rescue_start_epoch,
                self.v12_rescue_end_epoch,
                self.v12_rescue_topk,
                self.v12_rescue_min_class_prob,
                self.v12_rescue_min_competitor_ratio,
                self.v12_rescue_max_normal_prob,
                self.v12_rescue_proto_margin,
                self.v12_rescue_normal_proto_margin,
                self.v12_rescue_min_similarity,
                self.v12_rescue_radius_cap,
                self.v12_rescue_max_per_batch,
                self.v12_rescue_min_target,
                self.v12_rescue_mix_alpha,
                self.v12_rescue_clmmd_boost,
            )
        )
        logging.info(
            'V12 eval rescue: enabled={} min_prob={:.3f} ratio={:.3f} '
            'min_votes={} boost={:.3f}'.format(
                self.v12_eval_rescue_enabled,
                self.v12_eval_min_class_prob,
                self.v12_eval_competitor_ratio,
                self.v12_eval_min_source_votes,
                self.v12_eval_boost,
            )
        )
        logging.info(
            'V12 network structure unchanged; objective remains CE + MMD + CDAN + CLMMD + SupCon.'
        )

    @staticmethod
    def _clip01(value) -> float:
        return min(max(float(value), 0.0), 1.0)

    def _parse_v12_class_list(self, value, excluded=None) -> List[int]:
        excluded = set(excluded or set())
        result: List[int] = []
        for item in str(value or '').split(','):
            item = item.strip()
            if not item:
                continue
            try:
                class_idx = int(item)
            except ValueError:
                logging.warning('Ignore invalid V12 class id: %s', item)
                continue
            if (
                0 <= class_idx < self.num_classes
                and class_idx != self.v12_rescue_class
                and class_idx not in excluded
            ):
                result.append(class_idx)
        return sorted(set(result))

    def _parse_source_list(self, value) -> List[int]:
        result: List[int] = []
        for item in str(value or '').split(','):
            item = item.strip()
            if not item:
                continue
            try:
                source_idx = int(item)
            except ValueError:
                logging.warning('Ignore invalid V12 source id: %s', item)
                continue
            if 0 <= source_idx < self.num_source:
                result.append(source_idx)
        return sorted(set(result))

    def _reset_v12_epoch_stats(self):
        shape = int(getattr(self, 'num_source', 1))
        device = getattr(self, 'device', torch.device('cpu'))
        self._v12_candidates = torch.zeros(shape, dtype=torch.long, device=device)
        self._v12_reject_normal_prob = torch.zeros(
            shape, dtype=torch.long, device=device
        )
        self._v12_reject_proto = torch.zeros(shape, dtype=torch.long, device=device)
        self._v12_pre_cap_accepted = torch.zeros(
            shape, dtype=torch.long, device=device
        )
        self._v12_final_accepted = torch.zeros(
            shape, dtype=torch.long, device=device
        )
        self._v12_valid_batches = torch.zeros(
            shape, dtype=torch.long, device=device
        )

    def _checkpoint_dict(self):
        checkpoint = super(Trainer, self)._checkpoint_dict()
        checkpoint.update({
            'v12_cwru1_stable_rescue': True,
            'v12_rescue_class': int(self.v12_rescue_class),
            'v12_normal_class': int(self.v12_normal_class),
            'v12_rescue_source_indices': list(self.v12_rescue_source_indices),
        })
        return checkpoint

    def _classwise_lmmd_per_class(self, f_s, f_t, labels_s, probs_t):
        # Keep all original V10 CLMMD behavior as the baseline.  Stable rescue
        # is only a small, filtered addition to the existing class-2 term.
        loss_vec, valid_vec = super(Trainer, self)._classwise_lmmd_per_class(
            f_s, f_t, labels_s, probs_t
        )

        epoch = int(getattr(self, '_cur_epoch', 1))
        class_idx = int(self.v12_rescue_class)
        normal_idx = int(self.v12_normal_class)
        source_idx = int(getattr(self, '_active_source_idx_for_clmmd', -1))

        if (
            not self.v12_rescue_enabled
            or epoch < self.v12_rescue_start_epoch
            or epoch > self.v12_rescue_end_epoch
            or source_idx not in self.v12_rescue_source_indices
            or not (0 <= class_idx < self.num_classes)
            or not (0 <= normal_idx < self.num_classes)
            or self._v8_source_class_prototypes is None
        ):
            return loss_vec, valid_vec

        proto_updates = self._v8_source_class_prototype_updates[source_idx]
        radius_updates = self._v9_radius_updates[source_idx]
        proto_valid = (
            proto_updates >= self.v8_prototype_min_updates
        ) & (
            radius_updates >= self.v8_prototype_min_updates
        )
        if not bool(proto_valid[class_idx].item()):
            return loss_vec, valid_vec
        if not bool(proto_valid[normal_idx].item()):
            return loss_vec, valid_vec
        if int(proto_valid.sum().item()) < 3:
            return loss_vec, valid_vec

        source_mask = labels_s == class_idx
        num_source_class = int(source_mask.sum().item())
        if num_source_class < self.clmmd_min_source:
            return loss_vec, valid_vec

        probs_t = torch.clamp(probs_t, min=self.entropy_eps, max=1.0)
        probs_t = probs_t / (
            probs_t.sum(dim=1, keepdim=True) + self.entropy_eps
        )
        pseudo_label = probs_t.argmax(dim=1)
        topk = min(self.v12_rescue_topk, self.num_classes)
        topk_label = probs_t.topk(k=topk, dim=1).indices
        class_in_topk = (topk_label == class_idx).any(dim=1)

        confusion_pred = pseudo_label == class_idx
        competitor_prob = torch.zeros_like(probs_t[:, class_idx])
        for competitor in self.v12_confusion_classes:
            confusion_pred = confusion_pred | (pseudo_label == competitor)
            competitor_prob = torch.maximum(
                competitor_prob, probs_t[:, competitor]
            )

        probability_candidate = (
            class_in_topk
            & confusion_pred
            & (probs_t[:, class_idx] >= self.v12_rescue_min_class_prob)
            & (
                probs_t[:, class_idx]
                >= self.v12_rescue_min_competitor_ratio * competitor_prob
            )
        )
        candidate_count = int(probability_candidate.sum().item())
        self._v12_candidates[source_idx] += candidate_count
        if candidate_count == 0:
            return loss_vec, valid_vec

        normal_prob_ok = probs_t[:, normal_idx] <= self.v12_rescue_max_normal_prob
        self._v12_reject_normal_prob[source_idx] += int(
            (probability_candidate & ~normal_prob_ok).sum().item()
        )

        target_norm = F.normalize(f_t, p=2, dim=1)
        prototypes = F.normalize(
            self._v8_source_class_prototypes[source_idx], p=2, dim=1
        )
        similarity = torch.matmul(target_norm, prototypes.t())
        similarity[:, ~proto_valid] = -1e9

        top2_sim, top2_cls = similarity.topk(k=2, dim=1)
        proto_label = top2_cls[:, 0]
        proto_margin = top2_sim[:, 0] - top2_sim[:, 1]
        class_similarity = similarity[:, class_idx]
        normal_similarity = similarity[:, normal_idx]
        normal_proto_gap = class_similarity - normal_similarity
        proto_distance = (1.0 - class_similarity).clamp_min(0.0)

        source_radius = self._v9_radius_threshold(source_idx, class_idx)
        radius_cap = torch.as_tensor(
            self.v12_rescue_radius_cap,
            device=source_radius.device,
            dtype=source_radius.dtype,
        )
        accepted_radius = torch.minimum(source_radius, radius_cap)

        prototype_ok = (
            (proto_label == class_idx)
            & (proto_margin >= self.v12_rescue_proto_margin)
            & (class_similarity >= self.v12_rescue_min_similarity)
            & (normal_proto_gap >= self.v12_rescue_normal_proto_margin)
            & (proto_distance <= accepted_radius)
        )
        preliminary = probability_candidate & normal_prob_ok & prototype_ok
        self._v12_reject_proto[source_idx] += int(
            (probability_candidate & normal_prob_ok & ~prototype_ok).sum().item()
        )
        pre_cap_count = int(preliminary.sum().item())
        self._v12_pre_cap_accepted[source_idx] += pre_cap_count
        if pre_cap_count < self.v12_rescue_min_target:
            return loss_vec, valid_vec

        # Quality score.  Only the strongest few samples from each batch are
        # permitted to alter the existing class-wise alignment.
        quality_score = (
            probs_t[:, class_idx]
            * torch.sigmoid(normal_proto_gap / self.v12_rescue_score_tau)
            * torch.exp(-proto_distance / self.v12_rescue_score_tau)
        )
        preliminary_idx = preliminary.nonzero(as_tuple=False).flatten()
        keep_count = min(
            int(preliminary_idx.numel()), self.v12_rescue_max_per_batch
        )
        if keep_count < self.v12_rescue_min_target:
            return loss_vec, valid_vec

        selected_local = quality_score[preliminary_idx].topk(
            k=keep_count, largest=True
        ).indices
        selected_idx = preliminary_idx[selected_local]
        accepted = torch.zeros_like(preliminary)
        accepted[selected_idx] = True
        accepted_count = int(accepted.sum().item())
        self._v12_final_accepted[source_idx] += accepted_count

        distance_soft = torch.exp(
            -proto_distance[accepted] / self.v9_prototype_soft_tau
        ).clamp_min(1e-4)
        target_weight_raw = probs_t[accepted, class_idx] * distance_soft
        target_weight_sum = target_weight_raw.sum()
        if target_weight_sum.detach().item() < self.clmmd_min_target_weight:
            return loss_vec, valid_vec

        source_features = f_s[source_mask]
        target_features = f_t[accepted]
        kernel_ss, kernel_st, kernel_tt = self._gaussian_kernel_matrix(
            source_features, target_features
        )
        source_weight = torch.ones(
            num_source_class, device=f_s.device
        ) / float(num_source_class)
        target_weight = target_weight_raw / (
            target_weight_sum + self.entropy_eps
        )
        rescue_loss = (
            torch.sum(
                source_weight.view(-1, 1)
                * source_weight.view(1, -1)
                * kernel_ss
            )
            + torch.sum(
                target_weight.view(-1, 1)
                * target_weight.view(1, -1)
                * kernel_tt
            )
            - 2.0
            * torch.sum(
                source_weight.view(-1, 1)
                * target_weight.view(1, -1)
                * kernel_st
            )
        )
        rescue_loss = rescue_loss * self.v12_rescue_clmmd_boost

        if bool(valid_vec[class_idx].detach().item() > 0):
            loss_vec[class_idx] = (
                (1.0 - self.v12_rescue_mix_alpha) * loss_vec[class_idx]
                + self.v12_rescue_mix_alpha * rescue_loss
            )
        else:
            # When no ordinary high-confidence class-2 pseudo labels exist,
            # still keep the rescue contribution conservative.
            loss_vec[class_idx] = (
                self.v12_rescue_mix_alpha * rescue_loss
            )
        valid_vec[class_idx] = 1.0
        self._v12_valid_batches[source_idx] += 1
        return loss_vec, valid_vec

    def _eval_class_weighted_fusion(self, probs_list):
        fused_prob, weights = super(Trainer, self)._eval_class_weighted_fusion(
            probs_list
        )
        if not self.v12_eval_rescue_enabled:
            return fused_prob, weights

        class_idx = int(self.v12_rescue_class)
        if not (0 <= class_idx < self.num_classes):
            return fused_prob, weights

        probs_stack = torch.stack(probs_list, dim=0)
        topk = min(self.v12_rescue_topk, self.num_classes)
        branch_topk = probs_stack.topk(k=topk, dim=2).indices
        source_votes = (branch_topk == class_idx).any(dim=2).sum(dim=0)

        original_pred = fused_prob.argmax(dim=1)
        confusion_pred = original_pred == class_idx
        competitor_prob = torch.zeros_like(fused_prob[:, class_idx])
        for competitor in self.v12_confusion_classes:
            confusion_pred = confusion_pred | (original_pred == competitor)
            competitor_prob = torch.maximum(
                competitor_prob, fused_prob[:, competitor]
            )

        candidate = (
            confusion_pred
            & (source_votes >= self.v12_eval_min_source_votes)
            & (fused_prob[:, class_idx] >= self.v12_eval_min_class_prob)
            & (
                fused_prob[:, class_idx]
                >= self.v12_eval_competitor_ratio * competitor_prob
            )
        )
        self._v12_eval_total += int(fused_prob.size(0))
        self._v12_eval_candidates += int(candidate.sum().item())
        if not bool(candidate.any().item()):
            return fused_prob, weights

        calibrated = fused_prob.clone()
        calibrated[candidate, class_idx] *= self.v12_eval_boost
        calibrated[candidate] = calibrated[candidate] / (
            calibrated[candidate].sum(dim=1, keepdim=True) + self.entropy_eps
        )
        new_pred = calibrated.argmax(dim=1)
        self._v12_eval_changed += int(
            (candidate & (new_pred != original_pred)).sum().item()
        )
        return calibrated, weights

    def test(self):
        self._v12_eval_total = 0
        self._v12_eval_candidates = 0
        self._v12_eval_changed = 0
        accuracy = super(Trainer, self).test()
        logging.info(
            'V12 eval rescue diagnostics: enabled={} candidates={}/{} '
            '({:.2f}%) changed={}/{} ({:.2f}%) class={}'.format(
                self.v12_eval_rescue_enabled,
                self._v12_eval_candidates,
                self._v12_eval_total,
                100.0 * self._v12_eval_candidates / max(self._v12_eval_total, 1),
                self._v12_eval_changed,
                self._v12_eval_total,
                100.0 * self._v12_eval_changed / max(self._v12_eval_total, 1),
                self.v12_rescue_class,
            )
        )
        return accuracy

    def _finalize_v3_epoch_weights(self):
        candidates = self._v12_candidates.detach().cpu().tolist()
        reject_normal = self._v12_reject_normal_prob.detach().cpu().tolist()
        reject_proto = self._v12_reject_proto.detach().cpu().tolist()
        pre_cap = self._v12_pre_cap_accepted.detach().cpu().tolist()
        final_accepted = self._v12_final_accepted.detach().cpu().tolist()
        valid_batches = self._v12_valid_batches.detach().cpu().tolist()

        super(Trainer, self)._finalize_v3_epoch_weights()
        logging.info(
            'V12 stable rescue epoch stats class-{}: {}'.format(
                self.v12_rescue_class,
                ' | '.join(
                    (
                        'src{} candidates={} reject_normal={} reject_proto={} '
                        'pre_cap={} accepted={} valid_batches={}'
                    ).format(
                        source_idx,
                        candidates[source_idx],
                        reject_normal[source_idx],
                        reject_proto[source_idx],
                        pre_cap[source_idx],
                        final_accepted[source_idx],
                        valid_batches[source_idx],
                    )
                    for source_idx in range(self.num_source)
                ),
            )
        )
        self._reset_v12_epoch_stats()


__all__ = ['Trainer']
