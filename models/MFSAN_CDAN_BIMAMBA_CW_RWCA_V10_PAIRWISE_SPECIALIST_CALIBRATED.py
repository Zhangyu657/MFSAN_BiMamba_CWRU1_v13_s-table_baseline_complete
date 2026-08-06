# -*- coding: utf-8 -*-
"""
MFSAN-CDAN-BiMamba-CW-RWCA-V10-PairwiseSpecialistCalibrated

V10 is a targeted refinement of V9.  It keeps the same backbone, three-source
MFSAN branches, V6/V7 reliability gates, five-term objective and the original
"test every epoch and save the target-test best checkpoint" protocol.

The changes directly address the V9 log:

1. Switchable specialist memory
   V9 could keep an obsolete specialist (for example class 6 was logged as
   PU1 while the final class weight clearly selected PU3).  V10 permits a
   protected specialist to switch after another source dominates for several
   consecutive epochs.

2. Pair-specific Hard-SupCon weights
   K001-KA30 and KA30-KB23 no longer share one hard-negative strength.  The
   default is mild for K001-KA30 and stronger for KA30-KB23.

3. Class-calibrated prototype radius
   Difficult classes have independent radius floors/caps.  A globally/class-
   weak source can receive a tighter radius cap, preventing a diffuse weak
   source from accepting every target pseudo-label.

4. Optional fault-safe normal-class guard
   A target sample is allowed to remain K001 only when the fused K001
   probability reaches a configurable threshold.  Otherwise the best fault
   class is selected.  This is an inference rule and does not add a loss.
"""

import logging
from typing import Dict, List, Tuple

import torch

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V9_SPECIALIST_RADIUS_ALIGNMENT import (
    Trainer as V9Trainer,
)


class Trainer(V9Trainer):
    """V10 trainer with switchable specialists and calibrated pair/radius rules."""

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # ------------------------------------------------------------------
        # V10 switchable specialist memory
        # ------------------------------------------------------------------
        self.v10_specialist_switch_enabled = bool(
            getattr(args, 'v10_specialist_switch_enabled', True)
        )
        self.v10_specialist_switch_epochs = max(
            1, int(getattr(args, 'v10_specialist_switch_epochs', 2))
        )
        self.v10_specialist_switch_min_weight = min(
            max(float(getattr(args, 'v10_specialist_switch_min_weight', 0.55)), 0.0),
            1.0,
        )
        self.v10_specialist_switch_min_gap = max(
            0.0, float(getattr(args, 'v10_specialist_switch_min_gap', 0.15))
        )
        self._v10_last_specialist_events: List[str] = []

        # ------------------------------------------------------------------
        # V10 pair-specific Hard-SupCon
        # ------------------------------------------------------------------
        self.v10_hard_pair_weights = self._parse_pair_weight_map(
            getattr(args, 'v10_hard_pair_weights', '0-3:1.10,3-4:1.30')
        )
        if not self.v10_hard_pair_weights:
            # Backward-compatible fallback to the V8/V9 pair set.
            self.v10_hard_pair_weights = {
                self._canonical_pair(a, b): max(
                    1.0, float(self.v8_hard_negative_weight)
                )
                for a, b in self.v8_hard_negative_pairs
            }

        # ------------------------------------------------------------------
        # V10 class-calibrated radius
        # ------------------------------------------------------------------
        self.v10_radius_class_min = self._build_class_value_vector(
            getattr(args, 'v10_radius_class_min', '0:0.025,3:0.025,4:0.025'),
            default=float(self.v9_radius_min),
            lower=0.0,
            upper=1.0,
        )
        self.v10_radius_class_max = self._build_class_value_vector(
            getattr(args, 'v10_radius_class_max', '0:0.040,3:0.050,4:0.050'),
            default=float(self.v9_radius_max),
            lower=0.0,
            upper=1.0,
        )
        self.v10_radius_weak_source_threshold = min(
            max(
                float(getattr(args, 'v10_radius_weak_source_threshold', 0.05)),
                0.0,
            ),
            1.0,
        )
        self.v10_radius_weak_source_cap_scale = min(
            max(
                float(getattr(args, 'v10_radius_weak_source_cap_scale', 0.80)),
                0.05,
            ),
            1.0,
        )

        # Guarantee per-class max >= min.
        self.v10_radius_class_max = torch.maximum(
            self.v10_radius_class_max, self.v10_radius_class_min
        )

        # ------------------------------------------------------------------
        # V10 optional normal-class decision guard
        # ------------------------------------------------------------------
        self.v10_normal_guard_enabled = bool(
            getattr(args, 'v10_normal_guard_enabled', True)
        )
        self.v10_normal_class = int(getattr(args, 'v10_normal_class', 0))
        self.v10_normal_min_prob = min(
            max(float(getattr(args, 'v10_normal_min_prob', 0.80)), 0.0), 1.0
        )
        self.v10_normal_guard_min_fault_prob = min(
            max(
                float(getattr(args, 'v10_normal_guard_min_fault_prob', 0.05)),
                0.0,
            ),
            1.0,
        )
        self._v10_guard_total = 0
        self._v10_guard_changed = 0

        logging.info(
            'Using model: '
            'MFSAN_CDAN_BIMAMBA_CW_RWCA_V10_PAIRWISE_SPECIALIST_CALIBRATED'
        )
        logging.info(
            'V10 specialist switching: enabled={} switch_epochs={} '
            'switch_min_weight={:.3f} switch_min_gap={:.3f}'.format(
                self.v10_specialist_switch_enabled,
                self.v10_specialist_switch_epochs,
                self.v10_specialist_switch_min_weight,
                self.v10_specialist_switch_min_gap,
            )
        )
        logging.info(
            'V10 pair-specific Hard-SupCon targets: {}'.format(
                self._format_v10_pair_weights(final=True)
            )
        )
        logging.info(
            'V10 class radius min={} max={} weak_threshold={:.3f} '
            'weak_cap_scale={:.3f}'.format(
                self._format_class_vector(self.v10_radius_class_min),
                self._format_class_vector(self.v10_radius_class_max),
                self.v10_radius_weak_source_threshold,
                self.v10_radius_weak_source_cap_scale,
            )
        )
        logging.info(
            'V10 normal guard: enabled={} class={} min_prob={:.3f} '
            'min_fault_prob={:.3f}'.format(
                self.v10_normal_guard_enabled,
                self.v10_normal_class,
                self.v10_normal_min_prob,
                self.v10_normal_guard_min_fault_prob,
            )
        )
        logging.info(
            'V10 effective objective remains five terms: '
            'CE + MMD + CDAN + calibrated-radius CLMMD + pairwise Hard-SupCon.'
        )
        logging.info(
            'Evaluation policy: eval_each_epoch={} select_best_on_target={}'.format(
                bool(getattr(args, 'eval_each_epoch', True)),
                bool(getattr(args, 'select_best_on_target', True)),
            )
        )

    # ------------------------------------------------------------------
    # Parsing/formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_pair(a: int, b: int) -> Tuple[int, int]:
        a, b = int(a), int(b)
        return (a, b) if a <= b else (b, a)

    def _parse_pair_weight_map(self, value) -> Dict[Tuple[int, int], float]:
        result: Dict[Tuple[int, int], float] = {}
        if value is None:
            return result
        if isinstance(value, dict):
            iterator = value.items()
        else:
            text = str(value).strip()
            if not text:
                return result
            iterator = []
            for item in text.split(','):
                item = item.strip()
                if not item or ':' not in item or '-' not in item:
                    continue
                pair_text, weight_text = item.split(':', 1)
                a_text, b_text = pair_text.split('-', 1)
                iterator.append(((int(a_text), int(b_text)), float(weight_text)))
        for pair, weight in iterator:
            if isinstance(pair, str):
                a_text, b_text = pair.split('-', 1)
                pair = (int(a_text), int(b_text))
            a, b = self._canonical_pair(pair[0], pair[1])
            if a == b or not (0 <= a < self.num_classes) or not (
                0 <= b < self.num_classes
            ):
                continue
            result[(a, b)] = max(1.0, float(weight))
        return result

    def _build_class_value_vector(
        self, value, default: float, lower: float, upper: float
    ) -> torch.Tensor:
        vec = torch.full(
            (self.num_classes,), float(default), device=self.device
        )
        text = '' if value is None else str(value).strip()
        if text:
            for item in text.split(','):
                item = item.strip()
                if not item or ':' not in item:
                    continue
                c_text, v_text = item.split(':', 1)
                c = int(c_text)
                if 0 <= c < self.num_classes:
                    vec[c] = min(max(float(v_text), lower), upper)
        return vec

    def _format_class_vector(self, vec: torch.Tensor) -> str:
        return ','.join(
            'c{}={:.3f}'.format(c, float(vec[c].item()))
            for c in range(self.num_classes)
            if c in self.v9_prototype_filter_class_set
        )

    def _pair_weight_at_epoch(self, target: float) -> float:
        epoch = int(getattr(self, '_cur_epoch', 1))
        if epoch < self.v9_hard_supcon_start_epoch:
            return 1.0
        progress = (
            epoch - self.v9_hard_supcon_start_epoch + 1
        ) / float(self.v9_hard_supcon_ramp_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 1.0 + progress * (max(float(target), 1.0) - 1.0)

    def _format_v10_pair_weights(self, final: bool = False) -> str:
        items = []
        for (a, b), target in sorted(self.v10_hard_pair_weights.items()):
            weight = target if final else self._pair_weight_at_epoch(target)
            items.append('{}-{}={:.3f}'.format(a, b, weight))
        return ', '.join(items) if items else 'none'

    # ------------------------------------------------------------------
    # Checkpoint marker
    # ------------------------------------------------------------------
    def _checkpoint_dict(self):
        ckpt = super(Trainer, self)._checkpoint_dict()
        ckpt.update({
            'v10_pairwise_specialist_calibrated': True,
            'v10_hard_pair_weights': {
                '{}-{}'.format(a, b): float(w)
                for (a, b), w in self.v10_hard_pair_weights.items()
            },
        })
        return ckpt

    # ------------------------------------------------------------------
    # Switchable specialist memory
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_v9_specialist_memory(self, epoch_weights: torch.Tensor) -> None:
        if not self.v9_specialist_protection_enabled:
            return
        epoch = int(getattr(self, '_cur_epoch', 1))
        if epoch < self.v9_specialist_start_epoch:
            return

        cw = self._normalize_class_source_weights(epoch_weights.detach())
        events: List[str] = []

        for c in range(self.num_classes):
            vec = cw[:, c]
            sorted_w, sorted_idx = torch.sort(vec, descending=True)
            best = int(sorted_idx[0].item())
            best_value = float(sorted_w[0].item())
            second_value = (
                float(sorted_w[1].item()) if self.num_source > 1 else 0.0
            )
            top_gap = best_value - second_value
            protected = int(self._v9_protected_specialist_source[c].item())

            if protected < 0:
                eligible = (
                    best_value >= self.v9_specialist_min_weight
                    and top_gap >= self.v9_specialist_min_gap
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
                        events.append(
                            'c{}:protect {}'.format(c, self._source_name(best))
                        )
                else:
                    self._v9_specialist_candidate_source[c] = -1
                    self._v9_specialist_candidate_streak[c] = 0
                continue

            protected_weight = float(vec[protected].item())

            # V10: allow a stale specialist to switch to a newly dominant source.
            switch_gap = best_value - protected_weight
            switch_eligible = (
                self.v10_specialist_switch_enabled
                and best != protected
                and best_value >= self.v10_specialist_switch_min_weight
                and top_gap >= self.v9_specialist_min_gap
                and switch_gap >= self.v10_specialist_switch_min_gap
            )
            if switch_eligible:
                if int(self._v9_specialist_candidate_source[c].item()) == best:
                    self._v9_specialist_candidate_streak[c] += 1
                else:
                    self._v9_specialist_candidate_source[c] = best
                    self._v9_specialist_candidate_streak[c] = 1
                if int(self._v9_specialist_candidate_streak[c].item()) >= (
                    self.v10_specialist_switch_epochs
                ):
                    old = protected
                    self._v9_protected_specialist_source[c] = best
                    self._v9_specialist_candidate_source[c] = -1
                    self._v9_specialist_candidate_streak[c] = 0
                    self._v9_specialist_release_streak[c] = 0
                    events.append(
                        'c{}:switch {}->{}'.format(
                            c, self._source_name(old), self._source_name(best)
                        )
                    )
                    continue
            else:
                self._v9_specialist_candidate_source[c] = -1
                self._v9_specialist_candidate_streak[c] = 0

            # Retain V9 release behavior when no strong replacement is ready.
            if protected_weight < self.v9_specialist_release_weight:
                self._v9_specialist_release_streak[c] += 1
                if int(self._v9_specialist_release_streak[c].item()) >= (
                    self.v9_specialist_release_epochs
                ):
                    self._v9_protected_specialist_source[c] = -1
                    self._v9_specialist_candidate_source[c] = -1
                    self._v9_specialist_candidate_streak[c] = 0
                    self._v9_specialist_release_streak[c] = 0
                    events.append(
                        'c{}:release {}'.format(c, self._source_name(protected))
                    )
            else:
                self._v9_specialist_release_streak[c] = 0

        self._v10_last_specialist_events = events
        if events:
            logging.info('V10 specialist events: %s', ' | '.join(events))

    # ------------------------------------------------------------------
    # Pair-specific Hard-SupCon matrix
    # ------------------------------------------------------------------
    def _build_hard_pair_matrix(self, labels: torch.Tensor) -> torch.Tensor:
        n = labels.numel()
        pair_weights = torch.ones(
            n, n, device=labels.device, dtype=torch.float32
        )
        if not self.v8_hard_supcon_enabled:
            return pair_weights

        row = labels.view(-1, 1)
        col = labels.view(1, -1)
        for (a, b), target in self.v10_hard_pair_weights.items():
            weight = self._pair_weight_at_epoch(target)
            if weight <= 1.0 + 1e-12:
                continue
            hard = ((row == a) & (col == b)) | ((row == b) & (col == a))
            pair_weights[hard] = torch.maximum(
                pair_weights[hard],
                torch.full_like(pair_weights[hard], float(weight)),
            )
        return pair_weights

    # ------------------------------------------------------------------
    # Class-calibrated radius threshold
    # ------------------------------------------------------------------
    def _v9_radius_threshold(self, source_idx: int, class_idx: int) -> torch.Tensor:
        mean = self._v9_radius_mean[source_idx, class_idx]
        std = torch.sqrt(
            self._v9_radius_var[source_idx, class_idx].clamp_min(0.0)
        )
        threshold = mean + self.v9_radius_std_scale * std

        class_min = self.v10_radius_class_min[class_idx]
        class_max = self.v10_radius_class_max[class_idx]

        # A source with very low current source-class reliability should not
        # receive a wide radius merely because its source cluster is diffuse.
        class_weight = None
        active = getattr(self, '_v7_active_supcon_class_weights', None)
        if active is not None and active.shape == (
            self.num_source, self.num_classes
        ):
            class_weight = float(active[source_idx, class_idx].item())
        if (
            class_weight is not None
            and class_weight < self.v10_radius_weak_source_threshold
        ):
            class_max = torch.maximum(
                class_min,
                class_max * self.v10_radius_weak_source_cap_scale,
            )

        return torch.minimum(torch.maximum(threshold, class_min), class_max)

    # ------------------------------------------------------------------
    # Optional fault-safe normal-class guard
    # ------------------------------------------------------------------
    def _eval_class_weighted_fusion(self, probs_list):
        fused_prob, weights = super(Trainer, self)._eval_class_weighted_fusion(
            probs_list
        )
        if (
            not self.v10_normal_guard_enabled
            or not (0 <= self.v10_normal_class < self.num_classes)
            or self.num_classes <= 1
        ):
            return fused_prob, weights

        normal = self.v10_normal_class
        pred = fused_prob.argmax(dim=1)
        fault_prob = fused_prob.clone()
        fault_prob[:, normal] = -1.0
        best_fault_prob, _ = fault_prob.max(dim=1)
        guard_mask = (
            (pred == normal)
            & (fused_prob[:, normal] < self.v10_normal_min_prob)
            & (best_fault_prob >= self.v10_normal_guard_min_fault_prob)
        )

        self._v10_guard_total += int(fused_prob.size(0))
        self._v10_guard_changed += int(guard_mask.sum().item())

        if int(guard_mask.sum().item()) == 0:
            return fused_prob, weights

        guarded = fused_prob.clone()
        guarded[guard_mask, normal] = 0.0
        guarded[guard_mask] = guarded[guard_mask] / (
            guarded[guard_mask].sum(dim=1, keepdim=True) + self.entropy_eps
        )
        return guarded, weights

    def test(self):
        self._v10_guard_total = 0
        self._v10_guard_changed = 0
        acc = super(Trainer, self).test()
        total = max(self._v10_guard_total, 1)
        logging.info(
            'V10 normal-guard diagnostics: changed={}/{} ({:.2f}%) '
            'normal_min_prob={:.3f}'.format(
                self._v10_guard_changed,
                self._v10_guard_total,
                100.0 * self._v10_guard_changed / total,
                self.v10_normal_min_prob,
            )
        )
        return acc

    # ------------------------------------------------------------------
    # Epoch diagnostics
    # ------------------------------------------------------------------
    def _finalize_v3_epoch_weights(self):
        super(Trainer, self)._finalize_v3_epoch_weights()
        logging.info(
            'V10 current pair-specific Hard-SupCon weights: %s',
            self._format_v10_pair_weights(final=False),
        )
        logging.info(
            'V10 switchable protected specialist map: %s',
            self._format_v9_protected_specialists(),
        )


__all__ = ['Trainer']
