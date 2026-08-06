# -*- coding: utf-8 -*-
"""
MFSAN-CDAN-BiMamba-CW-RWCA-V8-ConfusionProtoAlignment

V8 inherits V7 and keeps the original training/test protocol unchanged:

    - the held-out target test set is evaluated after every epoch;
    - target-test accuracy still selects the best checkpoint.

Compared with V7, V8 adds two targeted changes without adding a sixth loss:

1. Confusion-pair-aware Hard-negative SupCon
   The existing source supervised contrastive loss gives larger denominator
   weights to known confusion pairs, so these class boundaries receive stronger
   repulsion. Anchor classes are configurable, while all retained source samples
   remain in the contrastive pool as positives or negatives.

2. Prototype-guided CLMMD pseudo-label filtering
   Every source branch maintains EMA source-class prototypes in its branch
   feature space. A target sample participates in class-wise LMMD only when its
   classifier pseudo label agrees with the nearest valid source-class prototype
   and the prototype top1-top2 cosine margin is sufficiently large. Per-class
   confidence thresholds and CLMMD multipliers are configurable.

The effective objective remains exactly five terms:

    class/source weighted CE + MMD + CDAN + prototype-filtered CLMMD
    + confusion-aware Hard-negative SupCon
"""

import logging
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V7_CLASS_GATE_CONFLICT_FUSION import (
    Trainer as V7Trainer,
)


class Trainer(V7Trainer):
    """V8 trainer: V7 + hard-negative SupCon + prototype-filtered CLMMD."""

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # --------------------------------------------------------------
        # V8 Hard-negative SupCon
        # --------------------------------------------------------------
        self.v8_hard_supcon_enabled = bool(
            getattr(args, 'v8_hard_supcon_enabled', True)
        )
        self.v8_hard_negative_pairs = self._parse_pair_set(
            getattr(args, 'v8_hard_negative_pairs', '0-3,3-4,3-6,7-8')
        )
        self.v8_hard_negative_weight = max(
            1.0, float(getattr(args, 'v8_hard_negative_weight', 2.0))
        )
        self.v8_supcon_anchor_classes = self._parse_class_list(
            getattr(args, 'v8_supcon_anchor_classes', '0,1,3,4,6,7,8'),
            allow_all=True,
        )

        # --------------------------------------------------------------
        # V8 source-class prototypes and prototype-guided CLMMD
        # --------------------------------------------------------------
        self.v8_prototype_filter_enabled = bool(
            getattr(args, 'v8_prototype_filter_enabled', True)
        )
        self.v8_prototype_start_epoch = max(
            1, int(getattr(args, 'v8_prototype_start_epoch', 3))
        )
        self.v8_prototype_ema_momentum = min(
            max(float(getattr(args, 'v8_prototype_ema_momentum', 0.90)), 0.0),
            0.9999,
        )
        self.v8_prototype_margin = max(
            0.0, float(getattr(args, 'v8_prototype_margin', 0.05))
        )
        self.v8_prototype_min_updates = max(
            1, int(getattr(args, 'v8_prototype_min_updates', 1))
        )

        self.v8_prototype_conf_thresholds = torch.full(
            (self.num_classes,),
            float(self.pl_conf_thresh),
            device=self.device,
        )
        conf_overrides = self._parse_class_float_map(
            getattr(
                args,
                'v8_prototype_conf_overrides',
                '0:0.90,3:0.90,4:0.90,6:0.85',
            )
        )
        for c, value in conf_overrides.items():
            if 0 <= c < self.num_classes:
                self.v8_prototype_conf_thresholds[c] = min(
                    max(float(value), 0.0), 1.0
                )

        self.v8_clmmd_class_boost = torch.ones(
            self.num_classes, device=self.device
        )
        boost_map = self._parse_class_float_map(
            getattr(args, 'v8_clmmd_class_boost', '3:1.5,4:1.3,6:1.2')
        )
        for c, value in boost_map.items():
            if 0 <= c < self.num_classes:
                self.v8_clmmd_class_boost[c] = max(float(value), 0.0)

        self.v8_prototype_log_classes = self._parse_class_list(
            getattr(args, 'v8_prototype_log_classes', '0,3,4,6'),
            allow_all=True,
        )
        if self.v8_prototype_log_classes is None:
            self.v8_prototype_log_classes = list(range(self.num_classes))

        # Lazy feature dimension: [source, class, branch_feature_dim].
        self._v8_source_class_prototypes: Optional[torch.Tensor] = None
        self._v8_source_class_prototype_updates = torch.zeros(
            self.num_source,
            self.num_classes,
            dtype=torch.long,
            device=self.device,
        )

        # Per-epoch diagnostics. Counts are branch-specific because each target
        # sample is independently screened in every source-specific feature space.
        self._reset_v8_prototype_epoch_stats()

        logging.info(
            'Using model: '
            'MFSAN_CDAN_BIMAMBA_CW_RWCA_V8_CONFUSION_PROTO_ALIGNMENT'
        )
        logging.info(
            'V8 Hard-SupCon: enabled={} pairs={} hard_weight={:.4f} '
            'anchor_classes={}'.format(
                self.v8_hard_supcon_enabled,
                self._format_pair_set(self.v8_hard_negative_pairs),
                self.v8_hard_negative_weight,
                'all'
                if self.v8_supcon_anchor_classes is None
                else self.v8_supcon_anchor_classes,
            )
        )
        logging.info(
            'V8 prototype CLMMD: enabled={} start={} ema={:.4f} '
            'margin={:.4f} min_updates={} conf_thresholds={} '
            'class_boost={}'.format(
                self.v8_prototype_filter_enabled,
                self.v8_prototype_start_epoch,
                self.v8_prototype_ema_momentum,
                self.v8_prototype_margin,
                self.v8_prototype_min_updates,
                self._format_class_tensor(self.v8_prototype_conf_thresholds),
                self._format_class_tensor(self.v8_clmmd_class_boost),
            )
        )
        logging.info(
            'V8 effective objective remains five terms: '
            'CE + MMD + CDAN + prototype-filtered CLMMD + Hard-SupCon.'
        )
        logging.info(
            'Evaluation policy: eval_each_epoch={} select_best_on_target={}'.format(
                bool(getattr(args, 'eval_each_epoch', True)),
                bool(getattr(args, 'select_best_on_target', True)),
            )
        )

    # ------------------------------------------------------------------
    # Parsing and formatting utilities
    # ------------------------------------------------------------------
    def _parse_class_list(self, text, allow_all: bool = False):
        if text is None:
            return None if allow_all else []
        value = str(text).strip().lower()
        if allow_all and value in ('', 'all', 'none'):
            return None
        result: List[int] = []
        for item in str(text).split(','):
            item = item.strip()
            if not item:
                continue
            try:
                c = int(item)
            except ValueError:
                logging.warning('Ignore invalid class id: %s', item)
                continue
            if 0 <= c < self.num_classes:
                result.append(c)
            else:
                logging.warning(
                    'Ignore class id %s outside [0, %s].', c, self.num_classes - 1
                )
        return sorted(set(result))

    def _parse_pair_set(self, text) -> Set[Tuple[int, int]]:
        pairs: Set[Tuple[int, int]] = set()
        if text is None:
            return pairs
        for item in str(text).split(','):
            item = item.strip()
            if not item:
                continue
            sep = '-' if '-' in item else ':' if ':' in item else None
            if sep is None:
                logging.warning('Ignore invalid hard pair: %s', item)
                continue
            parts = item.split(sep)
            if len(parts) != 2:
                logging.warning('Ignore invalid hard pair: %s', item)
                continue
            try:
                a, b = int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                logging.warning('Ignore invalid hard pair: %s', item)
                continue
            if a == b or not (0 <= a < self.num_classes) or not (
                0 <= b < self.num_classes
            ):
                logging.warning('Ignore invalid hard pair: %s', item)
                continue
            pairs.add((min(a, b), max(a, b)))
        return pairs

    @staticmethod
    def _parse_class_float_map(text) -> Dict[int, float]:
        result: Dict[int, float] = {}
        if text is None:
            return result
        for item in str(text).split(','):
            item = item.strip()
            if not item or ':' not in item:
                continue
            key, value = item.split(':', 1)
            try:
                result[int(key.strip())] = float(value.strip())
            except ValueError:
                continue
        return result

    @staticmethod
    def _format_pair_set(pairs: Sequence[Tuple[int, int]]) -> str:
        if not pairs:
            return 'none'
        return ','.join('{}-{}'.format(a, b) for a, b in sorted(pairs))

    def _format_class_tensor(self, values: torch.Tensor) -> str:
        items = []
        values = values.detach().cpu().view(-1)
        for c in range(min(self.num_classes, values.numel())):
            items.append('c{}={:.3f}'.format(c, float(values[c].item())))
        return ','.join(items)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def _checkpoint_dict(self):
        ckpt = super(Trainer, self)._checkpoint_dict()
        ckpt.update({
            'v8_confusion_proto_alignment': True,
            'v8_hard_supcon_enabled': self.v8_hard_supcon_enabled,
            'v8_hard_negative_pairs': sorted(self.v8_hard_negative_pairs),
            'v8_hard_negative_weight': self.v8_hard_negative_weight,
            'v8_supcon_anchor_classes': self.v8_supcon_anchor_classes,
            'v8_prototype_filter_enabled': self.v8_prototype_filter_enabled,
            'v8_prototype_start_epoch': self.v8_prototype_start_epoch,
            'v8_prototype_ema_momentum': self.v8_prototype_ema_momentum,
            'v8_prototype_margin': self.v8_prototype_margin,
            'v8_prototype_min_updates': self.v8_prototype_min_updates,
            'v8_prototype_conf_thresholds': (
                self.v8_prototype_conf_thresholds.detach().cpu()
            ),
            'v8_clmmd_class_boost': self.v8_clmmd_class_boost.detach().cpu(),
            'v8_source_class_prototype_updates': (
                self._v8_source_class_prototype_updates.detach().cpu()
            ),
        })
        if self._v8_source_class_prototypes is not None:
            ckpt['v8_source_class_prototypes'] = (
                self._v8_source_class_prototypes.detach().cpu()
            )
        return ckpt

    def load_model(self):
        super(Trainer, self).load_model()
        ckpt = torch.load(self.args.load_path, map_location=self.device)

        prototypes = ckpt.get('v8_source_class_prototypes', None)
        updates = ckpt.get('v8_source_class_prototype_updates', None)
        if prototypes is not None:
            prototypes = prototypes.to(self.device).float()
            if (
                prototypes.dim() == 3
                and prototypes.size(0) == self.num_source
                and prototypes.size(1) == self.num_classes
            ):
                self._v8_source_class_prototypes = prototypes
            else:
                logging.warning(
                    'Ignore incompatible V8 prototypes with shape %s.',
                    tuple(prototypes.shape),
                )
        if updates is not None:
            updates = updates.to(self.device, dtype=torch.long)
            if updates.shape == (self.num_source, self.num_classes):
                self._v8_source_class_prototype_updates = updates

        logging.info(
            'Loaded V8 prototypes: valid=%d/%d.',
            int(
                (
                    self._v8_source_class_prototype_updates
                    >= self.v8_prototype_min_updates
                ).sum().item()
            ),
            self.num_source * self.num_classes,
        )

    # ------------------------------------------------------------------
    # Prototype utilities and prototype-filtered CLMMD
    # ------------------------------------------------------------------
    def _ensure_v8_prototypes(self, feature_dim: int) -> None:
        if self._v8_source_class_prototypes is None:
            self._v8_source_class_prototypes = torch.zeros(
                self.num_source,
                self.num_classes,
                feature_dim,
                device=self.device,
            )
            return
        if self._v8_source_class_prototypes.size(2) != feature_dim:
            raise RuntimeError(
                'V8 prototype feature dimension changed from {} to {}.'.format(
                    self._v8_source_class_prototypes.size(2), feature_dim
                )
            )

    @torch.no_grad()
    def _update_v8_source_prototypes(
        self,
        source_idx: int,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        if not (0 <= source_idx < self.num_source):
            return
        self._ensure_v8_prototypes(features.size(1))
        momentum = self.v8_prototype_ema_momentum
        for c in range(self.num_classes):
            mask = labels == c
            if int(mask.sum().item()) == 0:
                continue
            batch_proto = features[mask].detach().mean(dim=0)
            batch_proto = F.normalize(batch_proto.view(1, -1), dim=1).view(-1)
            updates = int(
                self._v8_source_class_prototype_updates[source_idx, c].item()
            )
            if updates == 0:
                new_proto = batch_proto
            else:
                old = self._v8_source_class_prototypes[source_idx, c]
                new_proto = momentum * old + (1.0 - momentum) * batch_proto
                new_proto = F.normalize(new_proto.view(1, -1), dim=1).view(-1)
            self._v8_source_class_prototypes[source_idx, c] = new_proto
            self._v8_source_class_prototype_updates[source_idx, c] += 1

    def _reset_v8_prototype_epoch_stats(self) -> None:
        shape = (self.num_source, self.num_classes)
        self._v8_proto_conf_candidates = torch.zeros(
            *shape, dtype=torch.long, device=self.device
        )
        self._v8_proto_accepted = torch.zeros(
            *shape, dtype=torch.long, device=self.device
        )
        self._v8_proto_reject_disagree = torch.zeros(
            *shape, dtype=torch.long, device=self.device
        )
        self._v8_proto_reject_margin = torch.zeros(
            *shape, dtype=torch.long, device=self.device
        )

    def _boost_clmmd_vector(self, loss_vec: torch.Tensor) -> torch.Tensor:
        return loss_vec * self.v8_clmmd_class_boost.to(loss_vec.device)

    def _classwise_lmmd_per_class(self, f_s, f_t, labels_s, probs_t):
        """
        Confidence + prototype agreement gated CLMMD.

        The active source index is exposed by the V6-Lite training loop. For
        early epochs or insufficient valid prototypes, the original V4
        confidence-only CLMMD is used as a safe fallback.
        """
        source_idx = int(getattr(self, '_active_source_idx_for_clmmd', -1))
        self._update_v8_source_prototypes(source_idx, f_s, labels_s)

        epoch = int(getattr(self, '_cur_epoch', 1))
        valid_source = 0 <= source_idx < self.num_source
        if (
            not self.v8_prototype_filter_enabled
            or epoch < self.v8_prototype_start_epoch
            or not valid_source
            or self._v8_source_class_prototypes is None
        ):
            loss_vec, valid_vec = super(Trainer, self)._classwise_lmmd_per_class(
                f_s, f_t, labels_s, probs_t
            )
            return self._boost_clmmd_vector(loss_vec), valid_vec

        proto_valid = (
            self._v8_source_class_prototype_updates[source_idx]
            >= self.v8_prototype_min_updates
        )
        # With fewer than two valid prototypes, a top1-top2 margin is not
        # meaningful; retain the original confidence-only behavior temporarily.
        if int(proto_valid.sum().item()) < 2:
            loss_vec, valid_vec = super(Trainer, self)._classwise_lmmd_per_class(
                f_s, f_t, labels_s, probs_t
            )
            return self._boost_clmmd_vector(loss_vec), valid_vec

        device = f_s.device
        loss_vec = torch.zeros(self.num_classes, device=device)
        valid_vec = torch.zeros(self.num_classes, device=device)

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

        for c in range(self.num_classes):
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
            mask_t = mask_conf & agree & margin_ok

            disagree_count = int((mask_conf & ~agree).sum().item())
            margin_reject_count = int(
                (mask_conf & agree & ~margin_ok).sum().item()
            )
            accept_count = int(mask_t.sum().item())
            self._v8_proto_reject_disagree[source_idx, c] += disagree_count
            self._v8_proto_reject_margin[source_idx, c] += margin_reject_count
            self._v8_proto_accepted[source_idx, c] += accept_count

            if accept_count < self.pl_min_target:
                continue

            wt_raw = probs_t[mask_t, c]
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
    # Confusion-pair-aware Hard-negative SupCon
    # ------------------------------------------------------------------
    def _build_hard_pair_matrix(self, labels: torch.Tensor) -> torch.Tensor:
        n = labels.numel()
        pair_weights = torch.ones(
            n, n, device=labels.device, dtype=torch.float32
        )
        if not self.v8_hard_supcon_enabled or not self.v8_hard_negative_pairs:
            return pair_weights
        for a, b in self.v8_hard_negative_pairs:
            hard = (
                ((labels.view(-1, 1) == a) & (labels.view(1, -1) == b))
                | ((labels.view(-1, 1) == b) & (labels.view(1, -1) == a))
            )
            pair_weights[hard] = self.v8_hard_negative_weight
        return pair_weights

    def _compute_source_supcon_loss(self, g_s_list, f_s_all, source_label_list):
        """Source-class weighted SupCon with confusion-pair hard negatives."""
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

        if not feature_parts:
            return torch.tensor(0.0, device=self.device)

        features = torch.cat(feature_parts, dim=0)
        labels = torch.cat(label_parts, dim=0)
        anchor_weights = torch.cat(anchor_weight_parts, dim=0).to(features.dtype)
        if features.size(0) <= 1:
            return torch.tensor(0.0, device=self.device)

        if self.v8_supcon_anchor_classes is None:
            anchor_class_mask = torch.ones_like(labels, dtype=torch.bool)
        else:
            anchor_class_mask = torch.zeros_like(labels, dtype=torch.bool)
            for c in self.v8_supcon_anchor_classes:
                anchor_class_mask |= labels == int(c)

        eps = max(float(getattr(self, 'entropy_eps', 1e-8)), 1e-12)
        temperature = max(float(self.supcon_temperature), eps)
        features = F.normalize(features, p=2, dim=1)
        logits = torch.matmul(features, features.t()) / temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        same_class = torch.eq(
            labels.view(-1, 1), labels.view(1, -1)
        ).to(features.dtype)
        self_mask = torch.ones_like(same_class)
        self_mask.fill_diagonal_(0.0)
        pos_mask = same_class * self_mask

        pair_weights = self._build_hard_pair_matrix(labels).to(features.dtype)
        exp_logits = torch.exp(logits) * self_mask * pair_weights
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True) + eps
        )
        pos_count = pos_mask.sum(dim=1)
        valid = (pos_count > 0) & anchor_class_mask
        if int(valid.sum().item()) == 0:
            return torch.tensor(0.0, device=self.device)

        anchor_loss = -(pos_mask * log_prob).sum(dim=1) / (pos_count + eps)
        valid_weights = anchor_weights[valid].clamp_min(eps)
        return (anchor_loss[valid] * valid_weights).sum() / (
            valid_weights.sum() + eps
        )

    # ------------------------------------------------------------------
    # Epoch diagnostics
    # ------------------------------------------------------------------
    def _finalize_v3_epoch_weights(self):
        super(Trainer, self)._finalize_v3_epoch_weights()

        valid_count = int(
            (
                self._v8_source_class_prototype_updates
                >= self.v8_prototype_min_updates
            ).sum().item()
        )
        logging.info(
            'V8 prototype bank: valid={}/{} update_range=[{},{}]'.format(
                valid_count,
                self.num_source * self.num_classes,
                int(self._v8_source_class_prototype_updates.min().item()),
                int(self._v8_source_class_prototype_updates.max().item()),
            )
        )

        log_classes = self.v8_prototype_log_classes or []
        for c in log_classes:
            for k in range(self.num_source):
                candidates = int(self._v8_proto_conf_candidates[k, c].item())
                accepted = int(self._v8_proto_accepted[k, c].item())
                disagree = int(self._v8_proto_reject_disagree[k, c].item())
                low_margin = int(self._v8_proto_reject_margin[k, c].item())
                rate = accepted / float(max(candidates, 1))
                logging.info(
                    'V8 prototype filter src{} class-{}: candidates={} '
                    'accepted={} accept_rate={:.4f} reject_disagree={} '
                    'reject_margin={} threshold={:.3f} boost={:.3f}'.format(
                        k,
                        c,
                        candidates,
                        accepted,
                        rate,
                        disagree,
                        low_margin,
                        float(self.v8_prototype_conf_thresholds[c].item()),
                        float(self.v8_clmmd_class_boost[c].item()),
                    )
                )

        self._reset_v8_prototype_epoch_stats()
