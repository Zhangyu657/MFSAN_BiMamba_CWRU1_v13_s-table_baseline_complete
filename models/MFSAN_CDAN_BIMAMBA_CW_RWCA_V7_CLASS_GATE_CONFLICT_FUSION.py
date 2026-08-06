# -*- coding: utf-8 -*-
"""
MFSAN-CDAN-BiMamba-CW-RWCA-V7-ClassGate-ConflictFusion

This version inherits the V6-Lite-PU0 training objective and keeps the original
training/evaluation protocol unchanged:

    - the held-out target test set is evaluated after every epoch;
    - the best checkpoint is still selected by target-test accuracy.

Compared with V6-Lite-PU0, V7 adds two targeted mechanisms:

1. Class-level stable source gate
   The global gate is retained for global scalar losses (MMD/CDAN), but it no
   longer forces one globally weak source to have the same tiny weight for every
   class. Each class independently tracks its least reliable source over epochs.
   This preserves a source that is globally weak but is a useful specialist for
   a particular fault class.

2. Conflict-aware dynamic fusion
   When all source classifiers agree, evaluation behaves close to the original
   entropy-guided class-wise fusion. When source classifiers disagree, the
   global/class prior is softened and a confident branch receives a class-local
   top-1 margin bonus. This gives a reliable minority expert a chance to correct
   the fused decision.

The effective loss remains five terms:

    class/source weighted CE + MMD + CDAN + CLMMD + focused SupCon

No new loss term is introduced.
"""

import logging
from typing import List

import torch
import torch.nn.functional as F

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V6_LITE_PU0 import (
    Trainer as V6LiteTrainer,
)


class Trainer(V6LiteTrainer):
    """V7 trainer with class-level stable gate and conflict-aware fusion."""

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # ------------------------------------------------------------------
        # V7 class-level stable gate
        # ------------------------------------------------------------------
        self.v7_class_gate_enabled = bool(
            getattr(args, 'v7_class_gate_enabled', True)
        )
        self.v7_class_gate_start_epoch = int(
            getattr(args, 'v7_class_gate_start_epoch', 4)
        )
        self.v7_class_gate_confirm_epochs = max(
            1, int(getattr(args, 'v7_class_gate_confirm_epochs', 3))
        )
        self.v7_class_gate_release_epochs = max(
            1, int(getattr(args, 'v7_class_gate_release_epochs', 3))
        )
        self.v7_class_gate_confirm_gap = float(
            getattr(args, 'v7_class_gate_confirm_gap', 0.10)
        )
        self.v7_class_gate_release_gap = float(
            getattr(args, 'v7_class_gate_release_gap', 0.04)
        )
        self.v7_class_gate_max_bad_weight = float(
            getattr(args, 'v7_class_gate_max_bad_weight', 0.20)
        )
        self.v7_class_gate_preconfirm_floor = float(
            getattr(args, 'v7_class_gate_preconfirm_floor', 0.005)
        )
        self.v7_class_gate_bottom_floor = float(
            getattr(args, 'v7_class_gate_bottom_floor', 0.01)
        )

        # Adaptive class-specialist rescue.  When class-level evidence has a
        # clear winner, use more class evidence and less global prior.
        self.v7_class_rescue_enabled = bool(
            getattr(args, 'v7_class_rescue_enabled', True)
        )
        self.v7_class_rescue_max_alpha = float(
            getattr(args, 'v7_class_rescue_max_alpha', 0.70)
        )
        self.v7_class_rescue_gap_start = float(
            getattr(args, 'v7_class_rescue_gap_start', 0.10)
        )
        self.v7_class_rescue_gap_full = float(
            getattr(args, 'v7_class_rescue_gap_full', 0.40)
        )

        # Per-class gate state.  One source can be confirmed as weak for each
        # class.  This is deliberately different from one global bad source.
        self._v7_class_candidate_source = torch.full(
            (self.num_classes,), -1, dtype=torch.long, device=self.device
        )
        self._v7_class_candidate_streak = torch.zeros(
            self.num_classes, dtype=torch.long, device=self.device
        )
        self._v7_class_confirmed_source = torch.full(
            (self.num_classes,), -1, dtype=torch.long, device=self.device
        )
        self._v7_class_release_streak = torch.zeros(
            self.num_classes, dtype=torch.long, device=self.device
        )
        self._v7_last_pre_gate_class_weights = torch.full(
            (self.num_source, self.num_classes),
            1.0 / float(self.num_source),
            device=self.device,
        )
        self._v7_pre_gate_class_weight_sum = None
        self._v7_pre_gate_class_weight_count = 0

        # Class-aware SupCon weights.  A globally weak source is no longer
        # removed wholesale; samples are weighted by source-class reliability.
        self.v7_supcon_class_min_weight = float(
            getattr(args, 'v7_supcon_class_min_weight', 0.02)
        )
        self._v7_active_supcon_class_weights = torch.full(
            (self.num_source, self.num_classes),
            1.0 / float(self.num_source),
            device=self.device,
        )

        # ------------------------------------------------------------------
        # V7 conflict-aware evaluation fusion
        # ------------------------------------------------------------------
        self.v7_conflict_fusion_enabled = bool(
            getattr(args, 'v7_conflict_fusion_enabled', True)
        )
        self.v7_agree_prior_power = float(
            getattr(args, 'v7_agree_prior_power', 1.0)
        )
        self.v7_conflict_prior_power = float(
            getattr(args, 'v7_conflict_prior_power', 0.30)
        )
        self.v7_conflict_top1_margin_bonus = float(
            getattr(args, 'v7_conflict_top1_margin_bonus', 1.00)
        )
        self.v7_conflict_weight_temperature = float(
            getattr(args, 'v7_conflict_weight_temperature', 1.0)
        )

        # Classes printed in detailed logs.  Defaults cover the observed PU0
        # confusion set: K001, KA30, KB23, KI04.
        class_log_text = str(
            getattr(args, 'v7_class_gate_log_classes', '0,3,4,6')
        ).strip()
        extra_log_classes = []
        if class_log_text and class_log_text.lower() not in ('all', 'none'):
            for item in class_log_text.split(','):
                item = item.strip()
                if item:
                    try:
                        c = int(item)
                    except ValueError:
                        continue
                    if 0 <= c < self.num_classes:
                        extra_log_classes.append(c)
        elif class_log_text.lower() == 'all':
            extra_log_classes = list(range(self.num_classes))
        self._cw_log_classes = sorted(
            set(list(getattr(self, '_cw_log_classes', [])) + extra_log_classes)
        )

        logging.info(
            'Using model: MFSAN_CDAN_BIMAMBA_CW_RWCA_V7_CLASS_GATE_CONFLICT_FUSION'
        )
        logging.info(
            'V7 class gate: enabled={} start={} confirm={} release={} '
            'confirm_gap={:.4f} release_gap={:.4f} max_bad={:.4f} '
            'pre_floor={:.4f} bottom_floor={:.4f}'.format(
                self.v7_class_gate_enabled,
                self.v7_class_gate_start_epoch,
                self.v7_class_gate_confirm_epochs,
                self.v7_class_gate_release_epochs,
                self.v7_class_gate_confirm_gap,
                self.v7_class_gate_release_gap,
                self.v7_class_gate_max_bad_weight,
                self.v7_class_gate_preconfirm_floor,
                self.v7_class_gate_bottom_floor,
            )
        )
        logging.info(
            'V7 class specialist rescue: enabled={} max_alpha={:.4f} '
            'gap_start={:.4f} gap_full={:.4f}'.format(
                self.v7_class_rescue_enabled,
                self.v7_class_rescue_max_alpha,
                self.v7_class_rescue_gap_start,
                self.v7_class_rescue_gap_full,
            )
        )
        logging.info(
            'V7 conflict fusion: enabled={} agree_prior_power={:.4f} '
            'conflict_prior_power={:.4f} top1_margin_bonus={:.4f} '
            'weight_temperature={:.4f}'.format(
                self.v7_conflict_fusion_enabled,
                self.v7_agree_prior_power,
                self.v7_conflict_prior_power,
                self.v7_conflict_top1_margin_bonus,
                self.v7_conflict_weight_temperature,
            )
        )
        logging.info(
            'V7 protocol unchanged: target-test accuracy is still evaluated every '
            'epoch and still selects the best checkpoint.'
        )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def _checkpoint_dict(self):
        ckpt = super(Trainer, self)._checkpoint_dict()
        ckpt.update({
            'v7_class_gate_conflict_fusion': True,
            'v7_class_candidate_source': self._v7_class_candidate_source.detach().cpu(),
            'v7_class_candidate_streak': self._v7_class_candidate_streak.detach().cpu(),
            'v7_class_confirmed_source': self._v7_class_confirmed_source.detach().cpu(),
            'v7_class_release_streak': self._v7_class_release_streak.detach().cpu(),
            'v7_last_pre_gate_class_weights': self._v7_last_pre_gate_class_weights.detach().cpu(),
            'v7_class_gate_enabled': self.v7_class_gate_enabled,
            'v7_class_gate_start_epoch': self.v7_class_gate_start_epoch,
            'v7_class_gate_confirm_epochs': self.v7_class_gate_confirm_epochs,
            'v7_class_gate_release_epochs': self.v7_class_gate_release_epochs,
            'v7_class_gate_confirm_gap': self.v7_class_gate_confirm_gap,
            'v7_class_gate_release_gap': self.v7_class_gate_release_gap,
            'v7_class_gate_max_bad_weight': self.v7_class_gate_max_bad_weight,
            'v7_class_gate_preconfirm_floor': self.v7_class_gate_preconfirm_floor,
            'v7_class_gate_bottom_floor': self.v7_class_gate_bottom_floor,
            'v7_class_rescue_enabled': self.v7_class_rescue_enabled,
            'v7_class_rescue_max_alpha': self.v7_class_rescue_max_alpha,
            'v7_class_rescue_gap_start': self.v7_class_rescue_gap_start,
            'v7_class_rescue_gap_full': self.v7_class_rescue_gap_full,
            'v7_supcon_class_min_weight': self.v7_supcon_class_min_weight,
            'v7_conflict_fusion_enabled': self.v7_conflict_fusion_enabled,
            'v7_agree_prior_power': self.v7_agree_prior_power,
            'v7_conflict_prior_power': self.v7_conflict_prior_power,
            'v7_conflict_top1_margin_bonus': self.v7_conflict_top1_margin_bonus,
            'v7_conflict_weight_temperature': self.v7_conflict_weight_temperature,
        })
        return ckpt

    def load_model(self):
        super(Trainer, self).load_model()
        ckpt = torch.load(self.args.load_path, map_location=self.device)

        def load_vector(key, default, dtype):
            value = ckpt.get(key, None)
            if value is None:
                return default
            value = value.to(self.device, dtype=dtype)
            if value.numel() != self.num_classes:
                logging.warning(
                    'Ignore {} due to incompatible shape {}'.format(
                        key, tuple(value.shape)
                    )
                )
                return default
            return value.view(self.num_classes)

        self._v7_class_candidate_source = load_vector(
            'v7_class_candidate_source',
            self._v7_class_candidate_source,
            torch.long,
        )
        self._v7_class_candidate_streak = load_vector(
            'v7_class_candidate_streak',
            self._v7_class_candidate_streak,
            torch.long,
        )
        self._v7_class_confirmed_source = load_vector(
            'v7_class_confirmed_source',
            self._v7_class_confirmed_source,
            torch.long,
        )
        self._v7_class_release_streak = load_vector(
            'v7_class_release_streak',
            self._v7_class_release_streak,
            torch.long,
        )

        if 'v7_last_pre_gate_class_weights' in ckpt:
            value = ckpt['v7_last_pre_gate_class_weights'].to(self.device).float()
            if value.shape == (self.num_source, self.num_classes):
                self._v7_last_pre_gate_class_weights = (
                    self._normalize_class_source_weights(value)
                )

        logging.info(
            'Loaded V7 class gate state: {}'.format(
                self._format_confirmed_class_sources()
            )
        )

    # ------------------------------------------------------------------
    # Class specialist rescue before the class gate
    # ------------------------------------------------------------------
    def _global_guided_class_weights(self, global_weights, class_weights, alpha):
        """
        Adaptively combine global and class-specific reliability.

        V6 used one scalar alpha for every class.  V7 increases alpha only when
        the class-specific reliability has a clear winner.  This allows a source
        that is globally weak to remain a specialist for one class.
        """
        global_weights = global_weights.to(class_weights.device).float()
        global_weights = self._normalize_source_vector(global_weights)
        class_weights = self._normalize_class_source_weights(class_weights.float())
        global_mat = global_weights.view(-1, 1).repeat(1, self.num_classes)

        base_alpha = float(alpha)
        epoch = int(getattr(self, '_cur_epoch', 1))
        if (
            not self.v7_class_rescue_enabled
            or epoch <= int(getattr(self, 'cw_warmup_epochs', 0))
            or self.num_source <= 1
        ):
            alpha_vec = torch.full(
                (self.num_classes,),
                base_alpha,
                device=class_weights.device,
                dtype=class_weights.dtype,
            )
        else:
            sorted_w, _ = torch.sort(class_weights, dim=0, descending=True)
            gap = sorted_w[0] - sorted_w[1]
            start = self.v7_class_rescue_gap_start
            full = max(self.v7_class_rescue_gap_full, start + self.entropy_eps)
            rescue = ((gap - start) / (full - start)).clamp(0.0, 1.0)
            max_alpha = max(base_alpha, min(self.v7_class_rescue_max_alpha, 1.0))
            alpha_vec = base_alpha + (max_alpha - base_alpha) * rescue

        final_weights = (
            (1.0 - alpha_vec.view(1, -1)) * global_mat
            + alpha_vec.view(1, -1) * class_weights
        )
        final_weights = self._normalize_class_source_weights(final_weights)
        return final_weights.detach() if self.rw_detach_weights else final_weights

    def _global_loss_source_weights(
        self,
        effective_global_weights: torch.Tensor,
        class_weights: torch.Tensor,
        class_average_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Keep the global V6 gate for global MMD/CDAN losses."""
        return self._normalize_source_vector(effective_global_weights)

    # ------------------------------------------------------------------
    # Class-level stable gate
    # ------------------------------------------------------------------
    def _apply_floor_to_one_source(
        self, vec: torch.Tensor, source_idx: int, floor: float
    ) -> torch.Tensor:
        vec = self._normalize_source_vector(vec)
        if source_idx < 0 or source_idx >= self.num_source:
            return vec
        floor = min(max(float(floor), self.entropy_eps), 0.99)
        keep = torch.ones(self.num_source, dtype=torch.bool, device=vec.device)
        keep[source_idx] = False
        out = vec.clone()
        out[source_idx] = floor
        if bool(keep.any()):
            out[keep] = (1.0 - floor) * vec[keep] / (
                vec[keep].sum() + self.entropy_eps
            )
        return self._normalize_source_vector(out)

    def _apply_stable_gate_matrix(self, class_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply an independent stable gate for every class.

        Important: this method intentionally does NOT copy the global confirmed
        source to every class.  The global gate continues to control MMD/CDAN,
        while CE, CLMMD, SupCon and evaluation use class-specific evidence.
        """
        cw = self._normalize_class_source_weights(class_weights.float())

        power = max(self.v6_class_weight_power, self.entropy_eps)
        if abs(power - 1.0) > 1e-8:
            cw = torch.pow(cw.clamp_min(self.entropy_eps), power)
            cw = self._normalize_class_source_weights(cw)

        # Store the evidence before the class gate.  The state is updated once
        # per epoch in _finalize_v3_epoch_weights().
        with torch.no_grad():
            self._v7_last_pre_gate_class_weights = cw.detach().clone()
            if getattr(self, '_v3_collect_epoch_weights', False):
                if self._v7_pre_gate_class_weight_sum is None:
                    self._v7_pre_gate_class_weight_sum = torch.zeros_like(cw)
                    self._v7_pre_gate_class_weight_count = 0
                self._v7_pre_gate_class_weight_sum += cw.detach()
                self._v7_pre_gate_class_weight_count += 1

        if not self.v7_class_gate_enabled:
            self._v7_active_supcon_class_weights = cw.detach().clone()
            return cw.detach() if self.rw_detach_weights else cw

        epoch = int(getattr(self, '_cur_epoch', 1))
        out = cw.clone()

        for c in range(self.num_classes):
            vec = out[:, c]
            confirmed = int(self._v7_class_confirmed_source[c].item())

            if confirmed >= 0:
                vec = self._apply_floor_to_one_source(
                    vec, confirmed, self.v7_class_gate_bottom_floor
                )
            elif epoch >= self.v7_class_gate_start_epoch:
                floor = max(
                    self.v7_class_gate_preconfirm_floor, self.entropy_eps
                )
                if floor > 0.0:
                    vec = torch.clamp(vec, min=floor)
                    vec = self._normalize_source_vector(vec)

            vec = self._cap_source_vector(vec)

            # Re-enforce the exact confirmed floor after cap redistribution.
            if confirmed >= 0:
                vec = self._apply_floor_to_one_source(
                    vec, confirmed, self.v7_class_gate_bottom_floor
                )
            out[:, c] = vec

        out = self._normalize_class_source_weights(out)
        self._v7_active_supcon_class_weights = out.detach().clone()
        return out.detach() if self.rw_detach_weights else out

    def _update_class_gate_from_epoch(self, epoch_weights: torch.Tensor) -> None:
        """Update all class gate states from epoch-averaged pre-gate weights."""
        cw = self._normalize_class_source_weights(epoch_weights.detach())
        self._v7_last_pre_gate_class_weights = cw.clone()
        epoch = int(getattr(self, '_cur_epoch', 1))

        if not self.v7_class_gate_enabled or epoch < self.v7_class_gate_start_epoch:
            logging.info(
                'V7 class gate: event=warmup epoch={} confirmed={}'.format(
                    epoch, self._format_confirmed_class_sources()
                )
            )
            return

        event_parts: List[str] = []
        for c in range(self.num_classes):
            vec = cw[:, c]
            sorted_w, sorted_idx = torch.sort(vec, descending=False)
            worst = int(sorted_idx[0].item())
            worst_value = float(sorted_w[0].item())
            second_value = (
                float(sorted_w[1].item()) if self.num_source > 1 else 1.0
            )
            gap = second_value - worst_value
            confirmed = int(self._v7_class_confirmed_source[c].item())
            event = 'observe'

            if confirmed < 0:
                eligible = (
                    gap >= self.v7_class_gate_confirm_gap
                    and worst_value <= self.v7_class_gate_max_bad_weight
                )
                if eligible:
                    if int(self._v7_class_candidate_source[c].item()) == worst:
                        self._v7_class_candidate_streak[c] += 1
                    else:
                        self._v7_class_candidate_source[c] = worst
                        self._v7_class_candidate_streak[c] = 1

                    if int(self._v7_class_candidate_streak[c].item()) >= (
                        self.v7_class_gate_confirm_epochs
                    ):
                        self._v7_class_confirmed_source[c] = worst
                        self._v7_class_release_streak[c] = 0
                        event = 'confirmed'
                else:
                    self._v7_class_candidate_source[c] = -1
                    self._v7_class_candidate_streak[c] = 0
            else:
                still_worst = worst == confirmed
                still_separated = gap >= self.v7_class_gate_release_gap
                still_low = worst_value <= (
                    self.v7_class_gate_max_bad_weight
                    + self.v7_class_gate_release_gap
                )
                if still_worst and still_separated and still_low:
                    self._v7_class_release_streak[c] = 0
                else:
                    self._v7_class_release_streak[c] += 1
                    if int(self._v7_class_release_streak[c].item()) >= (
                        self.v7_class_gate_release_epochs
                    ):
                        self._v7_class_confirmed_source[c] = -1
                        self._v7_class_candidate_source[c] = -1
                        self._v7_class_candidate_streak[c] = 0
                        self._v7_class_release_streak[c] = 0
                        event = 'released'

            if c in self._cw_log_classes or event in ('confirmed', 'released'):
                event_parts.append(
                    'c{}:{} worst={} value={:.4f} gap={:.4f} cand={} streak={} confirmed={}'.format(
                        c,
                        event,
                        self._source_name(worst),
                        worst_value,
                        gap,
                        self._source_name(
                            int(self._v7_class_candidate_source[c].item())
                        ),
                        int(self._v7_class_candidate_streak[c].item()),
                        self._source_name(
                            int(self._v7_class_confirmed_source[c].item())
                        ),
                    )
                )

        logging.info(
            'V7 class gate update: {}'.format(
                ' | '.join(event_parts) if event_parts else 'no logged classes'
            )
        )

    def _format_confirmed_class_sources(self) -> str:
        items = []
        for c in range(self.num_classes):
            source_idx = int(self._v7_class_confirmed_source[c].item())
            if source_idx >= 0:
                items.append('c{}={}'.format(c, self._source_name(source_idx)))
        return ', '.join(items) if items else 'none'

    def _finalize_v3_epoch_weights(self):
        """Update class gate once per epoch, then keep V3 no-EMA evaluation."""
        with torch.no_grad():
            if (
                self._v7_pre_gate_class_weight_sum is not None
                and self._v7_pre_gate_class_weight_count > 0
            ):
                avg_pre_gate = self._v7_pre_gate_class_weight_sum / float(
                    self._v7_pre_gate_class_weight_count
                )
                self._update_class_gate_from_epoch(avg_pre_gate)
            else:
                logging.warning(
                    'V7 class gate did not collect pre-gate weights this epoch.'
                )

        # This stores the current epoch's final class-wise weights for test-time
        # fusion.  It deliberately preserves the original test-every-epoch flow.
        super(Trainer, self)._finalize_v3_epoch_weights()

        self._v7_pre_gate_class_weight_sum = None
        self._v7_pre_gate_class_weight_count = 0
        logging.info(
            'V7 class gate confirmed map: {}'.format(
                self._format_confirmed_class_sources()
            )
        )

    # ------------------------------------------------------------------
    # Class-aware focused SupCon
    # ------------------------------------------------------------------
    def _compute_source_supcon_loss(self, g_s_list, f_s_all, source_label_list):
        """
        Focused SupCon with source-class anchor weights.

        V6 excluded a globally confirmed source completely.  V7 keeps samples
        from classes for which that source remains reliable.
        """
        cur_epoch = int(getattr(self, '_cur_epoch', 1))
        if self.lambda_supcon <= 0.0 or cur_epoch < self.supcon_start_epoch:
            return torch.tensor(0.0, device=self.device)

        mode = self.supcon_feature_mode.upper()
        feat_list = f_s_all if mode == 'F' else g_s_list
        class_w = self._normalize_class_source_weights(
            self._v7_active_supcon_class_weights.to(self.device).float()
        ).detach()

        feature_parts = []
        label_parts = []
        anchor_weight_parts = []

        for k in range(self.num_source):
            labels_k = source_label_list[k]
            weights_k = class_w[k, labels_k]

            if self.v6_gate_apply_to_supcon:
                keep = weights_k >= self.v7_supcon_class_min_weight
            else:
                keep = torch.ones_like(labels_k, dtype=torch.bool)

            if int(keep.sum().item()) == 0:
                continue

            feature_parts.append(feat_list[k][keep])
            label_parts.append(labels_k[keep])
            anchor_weight_parts.append(weights_k[keep])

        if len(feature_parts) == 0:
            return torch.tensor(0.0, device=self.device)

        features = torch.cat(feature_parts, dim=0)
        labels = torch.cat(label_parts, dim=0)
        anchor_weights = torch.cat(anchor_weight_parts, dim=0).to(features.dtype)

        if self.supcon_focus_classes is not None:
            focus_mask = torch.zeros_like(labels, dtype=torch.bool)
            for c in self.supcon_focus_classes:
                focus_mask = focus_mask | (labels == int(c))
            if int(focus_mask.sum().item()) <= 1:
                return torch.tensor(0.0, device=self.device)
            features = features[focus_mask]
            labels = labels[focus_mask]
            anchor_weights = anchor_weights[focus_mask]

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
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True) + eps
        )
        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if int(valid.sum().item()) == 0:
            return torch.tensor(0.0, device=self.device)

        anchor_loss = -(pos_mask * log_prob).sum(dim=1) / (pos_count + eps)
        valid_weights = anchor_weights[valid].clamp_min(eps)
        return (anchor_loss[valid] * valid_weights).sum() / (
            valid_weights.sum() + eps
        )

    # ------------------------------------------------------------------
    # Conflict-aware dynamic fusion
    # ------------------------------------------------------------------
    def _eval_class_weighted_fusion(self, probs_list):
        """
        Class-wise entropy fusion with conflict-aware prior softening.

        - agreement: keep the original class prior strength;
        - disagreement: reduce prior strength and reward a branch's confident
          top-1 class using its top1-top2 margin.
        """
        probs_stack = torch.stack(probs_list, dim=0)  # [K, B, C]

        if (
            hasattr(self, 'class_source_weight_last_epoch')
            and self.class_source_weight_last_epoch is not None
        ):
            base = self.class_source_weight_last_epoch.to(probs_stack.device)
        else:
            base = self.class_source_weight_ema.to(probs_stack.device)
        base = self._normalize_class_source_weights(base)

        # Original non-dynamic path retained as a strict fallback.
        if not self.v7_conflict_fusion_enabled:
            return super(Trainer, self)._eval_class_weighted_fusion(probs_list)

        ent = -(
            torch.clamp(probs_stack, min=self.entropy_eps)
            * torch.log(torch.clamp(probs_stack, min=self.entropy_eps))
        ).sum(dim=2)  # [K, B]

        top2_prob, top2_idx = probs_stack.topk(k=min(2, self.num_classes), dim=2)
        top1_idx = top2_idx[:, :, 0]
        if self.num_classes > 1:
            margin = top2_prob[:, :, 0] - top2_prob[:, :, 1]
        else:
            margin = top2_prob[:, :, 0]

        # conflict[b] is true when at least one source predicts a different class.
        reference = top1_idx[0:1, :]
        conflict = (top1_idx != reference).any(dim=0)  # [B]

        prior_power = torch.where(
            conflict,
            torch.full_like(
                conflict.float(), self.v7_conflict_prior_power
            ),
            torch.full_like(
                conflict.float(), self.v7_agree_prior_power
            ),
        )

        log_prior = torch.log(base.clamp_min(self.entropy_eps)).view(
            self.num_source, 1, self.num_classes
        )
        score = prior_power.view(1, -1, 1) * log_prior

        if self.rw_eval_use_entropy:
            score = score - ent.unsqueeze(2) / max(
                self.rw_eval_tau, self.entropy_eps
            )

        # Only in conflict samples, give each branch a bonus on its own top-1
        # class.  The bonus is proportional to the branch margin.
        top1_one_hot = F.one_hot(
            top1_idx, num_classes=self.num_classes
        ).to(probs_stack.dtype)
        conflict_float = conflict.to(probs_stack.dtype).view(1, -1, 1)
        score = score + (
            conflict_float
            * self.v7_conflict_top1_margin_bonus
            * margin.unsqueeze(2)
            * top1_one_hot
        )

        temperature = max(
            self.v7_conflict_weight_temperature, self.entropy_eps
        )
        weights = torch.softmax(score / temperature, dim=0)
        fused_prob = (weights * probs_stack).sum(dim=0)
        fused_prob = fused_prob / (
            fused_prob.sum(dim=1, keepdim=True) + self.entropy_eps
        )

        # Diagnostic counters only; they do not affect model selection.
        with torch.no_grad():
            original_score = log_prior
            if self.rw_eval_use_entropy:
                original_score = original_score - ent.unsqueeze(2) / max(
                    self.rw_eval_tau, self.entropy_eps
                )
            original_weights = torch.softmax(original_score, dim=0)
            original_fused = (original_weights * probs_stack).sum(dim=0)
            original_pred = original_fused.argmax(dim=1)
            new_pred = fused_prob.argmax(dim=1)

            self._v7_eval_total_samples = int(
                getattr(self, '_v7_eval_total_samples', 0)
            ) + int(probs_stack.size(1))
            self._v7_eval_conflict_samples = int(
                getattr(self, '_v7_eval_conflict_samples', 0)
            ) + int(conflict.sum().item())
            self._v7_eval_changed_predictions = int(
                getattr(self, '_v7_eval_changed_predictions', 0)
            ) + int((original_pred != new_pred).sum().item())

        return fused_prob, weights

    def test(self):
        """
        Preserve the original target-test evaluation and best-model selection.
        Only add diagnostic logging for conflict-aware fusion.
        """
        self._v7_eval_total_samples = 0
        self._v7_eval_conflict_samples = 0
        self._v7_eval_changed_predictions = 0

        acc = super(Trainer, self).test()

        total = max(self._v7_eval_total_samples, 1)
        logging.info(
            'V7 Target-Test conflict diagnostics: conflict={}/{} ({:.2f}%), '
            'changed_vs_original_fusion={}/{} ({:.2f}%)'.format(
                self._v7_eval_conflict_samples,
                self._v7_eval_total_samples,
                100.0 * self._v7_eval_conflict_samples / total,
                self._v7_eval_changed_predictions,
                self._v7_eval_total_samples,
                100.0 * self._v7_eval_changed_predictions / total,
            )
        )
        return acc
