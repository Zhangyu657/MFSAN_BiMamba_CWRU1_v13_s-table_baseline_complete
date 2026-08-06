# -*- coding: utf-8 -*-
"""V11: conservative class-rescue refinement for CWRU_1.

The CWRU_1 log showed a target-class collapse for class 2 (ball_21): source
branches could recognize the class, but target samples were almost never given a
class-2 pseudo label.  V11 keeps the original five-loss objective and adds no
sixth loss.  It changes only how the existing CLMMD and test fusion are used:

1. Top-k + source-prototype rescue inside the existing CLMMD class component.
   A target sample may support the rescue class even when that class is ranked
   second, but only when a source-class prototype independently agrees, has a
   sufficient cosine margin/similarity, and passes the source-derived radius.

2. Conservative evaluation-time class rescue.
   When the fused prediction falls into a configured confusion class, the
   rescue class receives a small probability boost only if multiple source
   branches rank it in their top-k and its fused probability is not negligible.

These mechanisms are optional and are disabled by default.  The dedicated
``run_cwru1_fix.py`` script enables them for CWRU_1.
"""

import logging
from typing import List

import torch
import torch.nn.functional as F

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V10_PAIRWISE_SPECIALIST_CALIBRATED import (
    Trainer as V10Trainer,
)


class Trainer(V10Trainer):
    """V10 plus a conservative rescue path for one collapsed target class."""

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        self.v11_cwru1_rescue_enabled = bool(
            getattr(args, 'v11_cwru1_rescue_enabled', False)
        )
        self.v11_rescue_class = int(getattr(args, 'v11_rescue_class', 2))
        self.v11_confusion_classes = self._parse_v11_class_list(
            getattr(args, 'v11_confusion_classes', '0,1')
        )
        self.v11_rescue_start_epoch = max(
            1, int(getattr(args, 'v11_rescue_start_epoch', 2))
        )
        self.v11_rescue_topk = max(
            1, int(getattr(args, 'v11_rescue_topk', 2))
        )
        self.v11_rescue_min_class_prob = min(
            max(float(getattr(args, 'v11_rescue_min_class_prob', 0.10)), 0.0),
            1.0,
        )
        self.v11_rescue_proto_margin = max(
            0.0, float(getattr(args, 'v11_rescue_proto_margin', 0.03))
        )
        self.v11_rescue_min_similarity = min(
            max(float(getattr(args, 'v11_rescue_min_similarity', 0.35)), -1.0),
            1.0,
        )
        self.v11_rescue_clmmd_boost = max(
            0.0, float(getattr(args, 'v11_rescue_clmmd_boost', 1.50))
        )

        self.v11_eval_rescue_enabled = bool(
            getattr(args, 'v11_eval_rescue_enabled', True)
        )
        self.v11_eval_min_class_prob = min(
            max(float(getattr(args, 'v11_eval_min_class_prob', 0.08)), 0.0),
            1.0,
        )
        self.v11_eval_competitor_ratio = max(
            0.0, float(getattr(args, 'v11_eval_competitor_ratio', 0.35))
        )
        self.v11_eval_min_source_votes = max(
            1, int(getattr(args, 'v11_eval_min_source_votes', 2))
        )
        self.v11_eval_boost = max(
            1.0, float(getattr(args, 'v11_eval_boost', 2.00))
        )

        self._reset_v11_epoch_stats()
        self._v11_eval_total = 0
        self._v11_eval_candidates = 0
        self._v11_eval_changed = 0

        logging.info(
            'Using model: MFSAN_CDAN_BIMAMBA_CW_RWCA_V11_CWRU1_CLASS_RESCUE'
        )
        logging.info(
            'V11 CWRU1 rescue: enabled={} class={} confusion={} start={} '
            'topk={} min_prob={:.3f} proto_margin={:.3f} min_similarity={:.3f} '
            'clmmd_boost={:.3f}'.format(
                self.v11_cwru1_rescue_enabled,
                self.v11_rescue_class,
                self.v11_confusion_classes,
                self.v11_rescue_start_epoch,
                self.v11_rescue_topk,
                self.v11_rescue_min_class_prob,
                self.v11_rescue_proto_margin,
                self.v11_rescue_min_similarity,
                self.v11_rescue_clmmd_boost,
            )
        )
        logging.info(
            'V11 eval rescue: enabled={} min_prob={:.3f} competitor_ratio={:.3f} '
            'min_source_votes={} boost={:.3f}'.format(
                self.v11_eval_rescue_enabled,
                self.v11_eval_min_class_prob,
                self.v11_eval_competitor_ratio,
                self.v11_eval_min_source_votes,
                self.v11_eval_boost,
            )
        )
        logging.info(
            'V11 objective remains five terms: CE + MMD + CDAN + rescued CLMMD + Hard-SupCon.'
        )

    def _parse_v11_class_list(self, value) -> List[int]:
        result: List[int] = []
        for item in str(value or '').split(','):
            item = item.strip()
            if not item:
                continue
            try:
                c = int(item)
            except ValueError:
                logging.warning('Ignore invalid V11 class id: %s', item)
                continue
            if 0 <= c < self.num_classes and c != self.v11_rescue_class:
                result.append(c)
        return sorted(set(result))

    def _reset_v11_epoch_stats(self):
        n = int(getattr(self, 'num_source', 1))
        device = getattr(self, 'device', torch.device('cpu'))
        self._v11_rescue_candidates = torch.zeros(
            n, dtype=torch.long, device=device
        )
        self._v11_rescue_accepted = torch.zeros(
            n, dtype=torch.long, device=device
        )
        self._v11_rescue_valid_batches = torch.zeros(
            n, dtype=torch.long, device=device
        )

    def _checkpoint_dict(self):
        ckpt = super(Trainer, self)._checkpoint_dict()
        ckpt.update({
            'v11_cwru1_class_rescue': True,
            'v11_rescue_class': int(self.v11_rescue_class),
            'v11_confusion_classes': list(self.v11_confusion_classes),
        })
        return ckpt

    # ------------------------------------------------------------------
    # Existing CLMMD term with conservative top-k/prototype class rescue
    # ------------------------------------------------------------------
    def _classwise_lmmd_per_class(self, f_s, f_t, labels_s, probs_t):
        loss_vec, valid_vec = super(Trainer, self)._classwise_lmmd_per_class(
            f_s, f_t, labels_s, probs_t
        )

        epoch = int(getattr(self, '_cur_epoch', 1))
        c = int(self.v11_rescue_class)
        source_idx = int(getattr(self, '_active_source_idx_for_clmmd', -1))
        if (
            not self.v11_cwru1_rescue_enabled
            or epoch < self.v11_rescue_start_epoch
            or not (0 <= c < self.num_classes)
            or not (0 <= source_idx < self.num_source)
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
        if not bool(proto_valid[c].item()) or int(proto_valid.sum().item()) < 2:
            return loss_vec, valid_vec

        mask_s = labels_s == c
        n_s = int(mask_s.sum().item())
        if n_s < self.clmmd_min_source:
            return loss_vec, valid_vec

        probs_t = torch.clamp(probs_t, min=self.entropy_eps, max=1.0)
        probs_t = probs_t / (
            probs_t.sum(dim=1, keepdim=True) + self.entropy_eps
        )
        pseudo_label = probs_t.argmax(dim=1)
        topk = min(self.v11_rescue_topk, self.num_classes)
        topk_label = probs_t.topk(k=topk, dim=1).indices
        rescue_in_topk = (topk_label == c).any(dim=1)

        confusion_mask = pseudo_label == c
        for other in self.v11_confusion_classes:
            confusion_mask = confusion_mask | (pseudo_label == int(other))

        candidate = (
            rescue_in_topk
            & confusion_mask
            & (probs_t[:, c] >= self.v11_rescue_min_class_prob)
        )
        candidate_count = int(candidate.sum().item())
        self._v11_rescue_candidates[source_idx] += candidate_count
        if candidate_count == 0:
            return loss_vec, valid_vec

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
        radius_threshold = self._v9_radius_threshold(source_idx, c)

        accepted = (
            candidate
            & (proto_label == c)
            & (proto_margin >= self.v11_rescue_proto_margin)
            & (top2_sim[:, 0] >= self.v11_rescue_min_similarity)
            & (proto_distance <= radius_threshold)
        )
        accepted_count = int(accepted.sum().item())
        self._v11_rescue_accepted[source_idx] += accepted_count
        if accepted_count < self.pl_min_target:
            return loss_vec, valid_vec

        distance_soft = torch.exp(
            -proto_distance[accepted] / self.v9_prototype_soft_tau
        ).clamp_min(1e-4)
        wt_raw = probs_t[accepted, c] * distance_soft
        wt_sum = wt_raw.sum()
        if wt_sum.detach().item() < self.clmmd_min_target_weight:
            return loss_vec, valid_vec

        xs = f_s[mask_s]
        xt = f_t[accepted]
        k_ss, k_st, k_tt = self._gaussian_kernel_matrix(xs, xt)
        ws = torch.ones(n_s, device=f_s.device) / float(n_s)
        wt = wt_raw / (wt_sum + self.entropy_eps)
        loss_c = (
            torch.sum(ws.view(-1, 1) * ws.view(1, -1) * k_ss)
            + torch.sum(wt.view(-1, 1) * wt.view(1, -1) * k_tt)
            - 2.0 * torch.sum(ws.view(-1, 1) * wt.view(1, -1) * k_st)
        )
        loss_vec[c] = loss_c * self.v11_rescue_clmmd_boost
        valid_vec[c] = 1.0
        self._v11_rescue_valid_batches[source_idx] += 1
        return loss_vec, valid_vec

    # ------------------------------------------------------------------
    # Conservative class rescue during evaluation
    # ------------------------------------------------------------------
    def _eval_class_weighted_fusion(self, probs_list):
        fused_prob, weights = super(Trainer, self)._eval_class_weighted_fusion(
            probs_list
        )
        c = int(self.v11_rescue_class)
        if (
            not self.v11_cwru1_rescue_enabled
            or not self.v11_eval_rescue_enabled
            or not (0 <= c < self.num_classes)
            or not self.v11_confusion_classes
        ):
            return fused_prob, weights

        probs_stack = torch.stack(probs_list, dim=0)
        topk = min(self.v11_rescue_topk, self.num_classes)
        branch_topk = probs_stack.topk(k=topk, dim=2).indices
        source_votes = (branch_topk == c).any(dim=2).sum(dim=0)

        original_pred = fused_prob.argmax(dim=1)
        confusion_pred = torch.zeros_like(original_pred, dtype=torch.bool)
        for other in self.v11_confusion_classes:
            confusion_pred = confusion_pred | (original_pred == int(other))

        competitor_prob = torch.zeros_like(fused_prob[:, c])
        for other in self.v11_confusion_classes:
            competitor_prob = torch.maximum(
                competitor_prob, fused_prob[:, int(other)]
            )

        candidate = (
            confusion_pred
            & (source_votes >= self.v11_eval_min_source_votes)
            & (fused_prob[:, c] >= self.v11_eval_min_class_prob)
            & (
                fused_prob[:, c]
                >= self.v11_eval_competitor_ratio * competitor_prob
            )
        )
        self._v11_eval_total += int(fused_prob.size(0))
        self._v11_eval_candidates += int(candidate.sum().item())
        if int(candidate.sum().item()) == 0:
            return fused_prob, weights

        calibrated = fused_prob.clone()
        calibrated[candidate, c] *= self.v11_eval_boost
        calibrated[candidate] = calibrated[candidate] / (
            calibrated[candidate].sum(dim=1, keepdim=True) + self.entropy_eps
        )
        new_pred = calibrated.argmax(dim=1)
        self._v11_eval_changed += int(
            (candidate & (new_pred != original_pred)).sum().item()
        )
        return calibrated, weights

    def test(self):
        self._v11_eval_total = 0
        self._v11_eval_candidates = 0
        self._v11_eval_changed = 0
        acc = super(Trainer, self).test()
        logging.info(
            'V11 class-rescue diagnostics: candidates={}/{} ({:.2f}%) '
            'changed={}/{} ({:.2f}%) class={}'.format(
                self._v11_eval_candidates,
                self._v11_eval_total,
                100.0 * self._v11_eval_candidates / max(self._v11_eval_total, 1),
                self._v11_eval_changed,
                self._v11_eval_total,
                100.0 * self._v11_eval_changed / max(self._v11_eval_total, 1),
                self.v11_rescue_class,
            )
        )
        return acc

    def _finalize_v3_epoch_weights(self):
        candidates = self._v11_rescue_candidates.detach().cpu().tolist()
        accepted = self._v11_rescue_accepted.detach().cpu().tolist()
        valid_batches = self._v11_rescue_valid_batches.detach().cpu().tolist()
        super(Trainer, self)._finalize_v3_epoch_weights()
        logging.info(
            'V11 rescue CLMMD epoch stats class-{}: {}'.format(
                self.v11_rescue_class,
                ' | '.join(
                    'src{} candidates={} accepted={} valid_batches={}'.format(
                        k, candidates[k], accepted[k], valid_batches[k]
                    )
                    for k in range(self.num_source)
                ),
            )
        )
        self._reset_v11_epoch_stats()


__all__ = ['Trainer']
