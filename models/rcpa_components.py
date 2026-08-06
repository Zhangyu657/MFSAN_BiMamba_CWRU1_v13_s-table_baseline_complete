# -*- coding: utf-8 -*-
"""Reusable components for the simplified RCPA multi-source model.

RCPA = Reliability-guided Class Prototype Alignment.

This module intentionally keeps the adaptation objective compact. It provides:
1. class-balanced source-domain batch sampling;
2. EMA prototype/statistics memory;
3. source-class reliability estimation;
4. minimum class-confusion loss.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler


class BalancedClassBatchSampler(Sampler[List[int]]):
    """Generate approximately class-balanced mini-batches.

    Samples are cycled and reshuffled independently within every class. A small
    amount of replacement can occur when a minority class is exhausted; this is
    preferable to producing batches with missing classes when class prototypes
    are estimated online.
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        self.labels = np.asarray(list(labels), dtype=np.int64)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        self.classes = np.asarray(sorted(np.unique(self.labels).tolist()), dtype=np.int64)
        if self.classes.size == 0:
            raise ValueError("BalancedClassBatchSampler received no labels.")
        if self.batch_size < self.classes.size:
            raise ValueError(
                f"batch_size={self.batch_size} must be >= number of classes={self.classes.size}."
            )

        self.class_indices = {
            int(c): np.flatnonzero(self.labels == c).astype(np.int64)
            for c in self.classes
        }
        if self.drop_last:
            self.num_batches = max(1, len(self.labels) // self.batch_size)
        else:
            self.num_batches = max(1, math.ceil(len(self.labels) / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterable[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        pools = {c: rng.permutation(idx) for c, idx in self.class_indices.items()}
        pointers = {c: 0 for c in self.class_indices}

        base = self.batch_size // len(self.classes)
        remainder = self.batch_size % len(self.classes)

        for batch_idx in range(self.num_batches):
            batch: List[int] = []
            remainder_classes = np.roll(self.classes, batch_idx)[:remainder]
            remainder_set = set(int(x) for x in remainder_classes.tolist())

            for c_np in self.classes:
                c = int(c_np)
                need = base + (1 if c in remainder_set else 0)
                selected: List[int] = []

                while len(selected) < need:
                    pool = pools[c]
                    pointer = pointers[c]
                    available = pool.size - pointer
                    take = min(need - len(selected), available)
                    if take > 0:
                        selected.extend(pool[pointer:pointer + take].tolist())
                        pointers[c] += take
                    if len(selected) < need:
                        pools[c] = rng.permutation(self.class_indices[c])
                        pointers[c] = 0

                batch.extend(selected)

            rng.shuffle(batch)
            yield batch


def normalized_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return sample-wise entropy normalized to [0, 1]."""
    probs = probs.clamp(min=eps, max=1.0)
    entropy = -(probs * probs.log()).sum(dim=1)
    return entropy / math.log(max(probs.size(1), 2))


def normalize_source_weights(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize [K] or [K, C] weights along the source dimension."""
    return weights / (weights.sum(dim=0, keepdim=True) + eps)


def column_standardize(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize a [K, C] matrix independently for every class column."""
    mean = values.mean(dim=0, keepdim=True)
    std = values.std(dim=0, keepdim=True, unbiased=False)
    return (values - mean) / (std + eps)


def adaptive_top2_source_gate(
    weights: torch.Tensor,
    enabled: bool = True,
    prune_gap: float = 0.05,
    bottom_floor: float = 0.01,
    max_source_weight: float = 0.65,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Suppress a clearly inferior source while preventing single-source monopoly.

    The first dimension is the source-domain dimension; all remaining dimensions
    are treated independently.  With three source domains, when the gap between
    the second- and third-ranked source weights exceeds ``prune_gap``, the third
    source is reduced to ``bottom_floor`` and the remaining mass is redistributed
    to the two stronger sources.  Independently, no source may exceed
    ``max_source_weight``.

    This realizes a ``two useful sources + one suppressed negative source``
    pattern without hard-coding a specific domain name.  For PU_1/PU_2/PU_3 ->
    PU_0, the gate can automatically suppress PU_3 when its measured reliability
    is consistently below PU_1 and PU_2.
    """
    if weights.dim() < 1:
        raise ValueError("weights must contain a source dimension.")

    normalized = normalize_source_weights(weights.clamp_min(eps), eps)
    if not bool(enabled) or normalized.size(0) != 3:
        return normalized

    floor = float(bottom_floor)
    cap = float(max_source_weight)
    gap_threshold = float(prune_gap)
    if not 0.0 <= floor < 1.0 / 3.0:
        raise ValueError("bottom_floor must be in [0, 1/3).")
    if not 1.0 / 3.0 <= cap < 1.0:
        raise ValueError("max_source_weight must be in [1/3, 1).")

    original_shape = normalized.shape
    flat = normalized.reshape(3, -1)
    sorted_values, sorted_indices = torch.sort(flat, dim=0, descending=True)
    prune_mask = (sorted_values[1] - sorted_values[2]) >= gap_threshold

    bottom_indices = sorted_indices[2].unsqueeze(0)
    bottom_mask = torch.zeros_like(flat).scatter_(0, bottom_indices, 1.0)
    top_values = flat * (1.0 - bottom_mask)
    top_values = top_values / top_values.sum(dim=0, keepdim=True).clamp_min(eps)
    pruned = (1.0 - floor) * top_values + floor * bottom_mask
    flat = torch.where(prune_mask.unsqueeze(0), pruned, flat)

    # Cap the strongest source and redistribute its excess proportionally to
    # the other two sources.  This blocks 0.99/0.005/0.005 collapse while still
    # allowing 0.50/0.49/0.01 negative-source suppression.
    max_values, max_indices = flat.max(dim=0, keepdim=True)
    over_mask = max_values.squeeze(0) > cap
    max_mask = torch.zeros_like(flat).scatter_(0, max_indices, 1.0)
    other_values = flat * (1.0 - max_mask)
    other_values = other_values / other_values.sum(dim=0, keepdim=True).clamp_min(eps)
    capped = cap * max_mask + (1.0 - cap) * other_values
    flat = torch.where(over_mask.unsqueeze(0), capped, flat)

    flat = flat / flat.sum(dim=0, keepdim=True).clamp_min(eps)
    return flat.reshape(original_shape)


class StableSourcePruningController(nn.Module):
    """Epoch-level negative-source controller with delayed confirmation.

    The controller never makes a hard pruning decision from one mini-batch.
    It first observes epoch-average *raw* source reliability weights.  A source
    is confirmed as negative only when it is ranked last for several
    consecutive epochs and the second-to-last gap remains sufficiently large.

    Before confirmation, every source keeps at least ``preconfirm_floor``
    probability.  During the initial warm-up epochs, uniform fusion is used so
    an unstable early classifier cannot poison pseudo labels.  Once confirmed,
    the negative source is assigned ``bottom_floor`` while the remaining mass
    is distributed over the other sources.  A release rule allows recovery if
    the confirmed source is no longer consistently the weakest one.
    """

    def __init__(
        self,
        num_sources: int,
        enabled: bool = True,
        warmup_epochs: int = 3,
        start_epoch: int = 4,
        confirm_epochs: int = 3,
        release_epochs: int = 3,
        confirm_gap: float = 0.08,
        preconfirm_floor: float = 0.10,
        bottom_floor: float = 0.01,
        max_source_weight: float = 0.65,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_sources = int(num_sources)
        self.enabled = bool(enabled)
        self.warmup_epochs = int(warmup_epochs)
        self.start_epoch = int(start_epoch)
        self.confirm_epochs = max(1, int(confirm_epochs))
        self.release_epochs = max(1, int(release_epochs))
        self.confirm_gap = float(confirm_gap)
        self.preconfirm_floor = float(preconfirm_floor)
        self.bottom_floor = float(bottom_floor)
        self.max_source_weight = float(max_source_weight)
        self.eps = float(eps)

        if self.num_sources < 2:
            raise ValueError("StableSourcePruningController requires >=2 sources.")
        if self.preconfirm_floor * self.num_sources >= 1.0:
            raise ValueError("preconfirm_floor * num_sources must be < 1.")
        if not 0.0 <= self.bottom_floor < 1.0 / self.num_sources:
            raise ValueError("bottom_floor must be in [0, 1/num_sources).")
        if not 1.0 / self.num_sources <= self.max_source_weight < 1.0:
            raise ValueError("max_source_weight must be in [1/num_sources, 1).")

        self.register_buffer("candidate_index", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("candidate_streak", torch.tensor(0, dtype=torch.long))
        self.register_buffer("confirmed_index", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("release_streak", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_epoch_weights", torch.full((self.num_sources,), 1.0 / self.num_sources))
        self.register_buffer("last_gap", torch.tensor(0.0))

    def _cap(self, weights: torch.Tensor) -> torch.Tensor:
        """Cap the strongest source and redistribute excess proportionally."""
        original_shape = weights.shape
        flat = normalize_source_weights(weights.clamp_min(self.eps), self.eps).reshape(
            self.num_sources, -1
        )
        max_values, max_indices = flat.max(dim=0, keepdim=True)
        over = max_values.squeeze(0) > self.max_source_weight
        if bool(over.any()):
            max_mask = torch.zeros_like(flat).scatter_(0, max_indices, 1.0)
            others = flat * (1.0 - max_mask)
            others = others / others.sum(dim=0, keepdim=True).clamp_min(self.eps)
            capped = self.max_source_weight * max_mask + (1.0 - self.max_source_weight) * others
            flat = torch.where(over.unsqueeze(0), capped, flat)
        return normalize_source_weights(flat, self.eps).reshape(original_shape)

    def apply(self, weights: torch.Tensor, epoch: int) -> torch.Tensor:
        """Apply the currently confirmed gate to [K], [K,C], or [K,B,C] weights."""
        normalized = normalize_source_weights(weights.clamp_min(self.eps), self.eps)
        if not self.enabled:
            return normalized
        if normalized.size(0) != self.num_sources:
            raise ValueError("The first dimension must equal num_sources.")

        epoch = int(epoch)
        if epoch <= self.warmup_epochs:
            return torch.full_like(normalized, 1.0 / self.num_sources)

        confirmed = int(self.confirmed_index.item())
        original_shape = normalized.shape
        flat = normalized.reshape(self.num_sources, -1)

        if confirmed < 0:
            # Affine floor preserves an exact minimum contribution after
            # normalization: floor + (1-K*floor)*normalized_raw.
            flat = normalize_source_weights(flat, self.eps)
            flat = self.preconfirm_floor + (
                1.0 - self.num_sources * self.preconfirm_floor
            ) * flat
            return self._cap(flat).reshape(original_shape)

        bottom_mask = torch.zeros_like(flat)
        bottom_mask[confirmed] = 1.0
        useful_mask = 1.0 - bottom_mask
        useful = flat * useful_mask
        useful = useful / useful.sum(dim=0, keepdim=True).clamp_min(self.eps)
        gated = (1.0 - self.bottom_floor) * useful + self.bottom_floor * bottom_mask

        # Preserve the confirmed source at bottom_floor while capping only
        # among the remaining useful sources.  A generic cap that redistributes
        # to every other row would incorrectly raise the suppressed source.
        useful_values = gated * useful_mask
        max_values, max_indices = useful_values.max(dim=0, keepdim=True)
        over = max_values.squeeze(0) > self.max_source_weight
        if bool(over.any()):
            max_mask = torch.zeros_like(flat).scatter_(0, max_indices, 1.0) * useful_mask
            remaining_mask = useful_mask * (1.0 - max_mask)
            remaining = gated * remaining_mask
            remaining = remaining / remaining.sum(dim=0, keepdim=True).clamp_min(self.eps)
            capped = (
                self.bottom_floor * bottom_mask
                + self.max_source_weight * max_mask
                + (1.0 - self.bottom_floor - self.max_source_weight) * remaining
            )
            gated = torch.where(over.unsqueeze(0), capped, gated)
        return normalize_source_weights(gated, self.eps).reshape(original_shape)

    @torch.no_grad()
    def update_epoch(self, raw_epoch_weights: torch.Tensor, epoch: int) -> dict:
        """Update confirmation state from one epoch-average raw reliability vector."""
        raw = normalize_source_weights(raw_epoch_weights.detach().float(), self.eps).reshape(-1)
        if raw.numel() != self.num_sources:
            raise ValueError("raw_epoch_weights must contain num_sources values.")
        self.last_epoch_weights.copy_(raw)

        sorted_values, sorted_indices = torch.sort(raw, descending=False)
        candidate = int(sorted_indices[0].item())
        gap = float((sorted_values[1] - sorted_values[0]).item())
        self.last_gap.fill_(gap)

        event = "observe"
        if not self.enabled or int(epoch) < self.start_epoch:
            return self.status(event=event)

        confirmed = int(self.confirmed_index.item())
        if confirmed < 0:
            if gap >= self.confirm_gap:
                if int(self.candidate_index.item()) == candidate:
                    self.candidate_streak.add_(1)
                else:
                    self.candidate_index.fill_(candidate)
                    self.candidate_streak.fill_(1)
            else:
                self.candidate_index.fill_(-1)
                self.candidate_streak.zero_()

            if int(self.candidate_streak.item()) >= self.confirm_epochs:
                self.confirmed_index.fill_(candidate)
                self.release_streak.zero_()
                event = "confirmed"
        else:
            still_weakest = candidate == confirmed and gap >= 0.5 * self.confirm_gap
            if still_weakest:
                self.release_streak.zero_()
            else:
                self.release_streak.add_(1)
                if int(self.release_streak.item()) >= self.release_epochs:
                    self.confirmed_index.fill_(-1)
                    self.candidate_index.fill_(candidate if gap >= self.confirm_gap else -1)
                    self.candidate_streak.fill_(1 if gap >= self.confirm_gap else 0)
                    self.release_streak.zero_()
                    event = "released"

        return self.status(event=event)

    def status(self, event: str = "none") -> dict:
        return {
            "event": str(event),
            "candidate_index": int(self.candidate_index.item()),
            "candidate_streak": int(self.candidate_streak.item()),
            "confirmed_index": int(self.confirmed_index.item()),
            "release_streak": int(self.release_streak.item()),
            "gap": float(self.last_gap.item()),
            "raw_weights": self.last_epoch_weights.detach().cpu().tolist(),
        }


def three_source_consensus_scores(
    probs_by_source: Sequence[torch.Tensor],
    confidence_threshold: float = 0.55,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate target transferability from agreement with the other two sources.

    For source ``k``, the other two classifiers define a reference only on
    samples where they predict the same class with sufficient confidence.  The
    score is the confidence-weighted agreement of source ``k`` with that
    reference.  This signal is target-side and therefore complements MMD and
    entropy, which can otherwise reward confidently wrong predictions.
    """
    if len(probs_by_source) != 3:
        return torch.full(
            (len(probs_by_source),),
            0.5,
            device=probs_by_source[0].device,
            dtype=probs_by_source[0].dtype,
        )

    predictions = []
    confidences = []
    for probs in probs_by_source:
        conf, pred = probs.detach().max(dim=1)
        predictions.append(pred)
        confidences.append(conf)

    scores = []
    for k in range(3):
        others = [idx for idx in range(3) if idx != k]
        ref_valid = (
            predictions[others[0]].eq(predictions[others[1]])
            & (confidences[others[0]] >= float(confidence_threshold))
            & (confidences[others[1]] >= float(confidence_threshold))
        )
        if int(ref_valid.sum().item()) < 2:
            scores.append(torch.tensor(0.5, device=probs_by_source[k].device, dtype=probs_by_source[k].dtype))
            continue
        reference = predictions[others[0]][ref_valid]
        agreement = predictions[k][ref_valid].eq(reference).float()
        weighted = agreement * confidences[k][ref_valid]
        scores.append(weighted.sum() / confidences[k][ref_valid].sum().clamp_min(eps))
    return torch.stack(scores)


class RCPAStatisticsMemory(nn.Module):
    """EMA memory for prototypes and reliability statistics."""

    def __init__(
        self,
        num_sources: int,
        num_classes: int,
        feature_dim: int,
        prototype_momentum: float = 0.90,
        statistic_momentum: float = 0.90,
        weight_momentum: float = 0.90,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_sources = int(num_sources)
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.prototype_momentum = float(prototype_momentum)
        self.statistic_momentum = float(statistic_momentum)
        self.weight_momentum = float(weight_momentum)
        self.eps = float(eps)

        shape = (self.num_sources, self.num_classes, self.feature_dim)
        stat_shape = (self.num_sources, self.num_classes)

        self.register_buffer("source_prototypes", torch.zeros(shape))
        self.register_buffer("target_prototypes", torch.zeros(shape))
        self.register_buffer("source_proto_valid", torch.zeros(stat_shape, dtype=torch.bool))
        self.register_buffer("target_proto_valid", torch.zeros(stat_shape, dtype=torch.bool))
        self.register_buffer("source_recognition", torch.full(stat_shape, 1.0 / self.num_classes))
        self.register_buffer("source_recognition_valid", torch.zeros(stat_shape, dtype=torch.bool))
        self.register_buffer("target_entropy", torch.ones(stat_shape))
        self.register_buffer("target_entropy_valid", torch.zeros(stat_shape, dtype=torch.bool))
        self.register_buffer("global_source_weights", torch.full((self.num_sources,), 1.0 / self.num_sources))
        self.register_buffer("class_source_weights", torch.full(stat_shape, 1.0 / self.num_sources))
        self.register_buffer("target_coverage", torch.tensor(0.0))

    @torch.no_grad()
    def _ema_vector(
        self,
        buffer: torch.Tensor,
        valid_buffer: torch.Tensor,
        index: Tuple[int, int],
        value: torch.Tensor,
        momentum: float,
    ) -> None:
        k, c = index
        value = value.detach()
        if bool(valid_buffer[k, c]):
            buffer[k, c].mul_(momentum).add_(value, alpha=1.0 - momentum)
        else:
            buffer[k, c].copy_(value)
            valid_buffer[k, c] = True

    @torch.no_grad()
    def _ema_scalar(
        self,
        buffer: torch.Tensor,
        valid_buffer: torch.Tensor,
        index: Tuple[int, int],
        value: torch.Tensor,
        momentum: float,
    ) -> None:
        k, c = index
        value = value.detach().float()
        if bool(valid_buffer[k, c]):
            buffer[k, c].mul_(momentum).add_(value, alpha=1.0 - momentum)
        else:
            buffer[k, c].copy_(value)
            valid_buffer[k, c] = True

    @torch.no_grad()
    def update_source(
        self,
        source_features: Sequence[torch.Tensor],
        source_probs: Sequence[torch.Tensor],
        source_labels: Sequence[torch.Tensor],
        min_samples: int = 1,
    ) -> None:
        for k, (features, probs, labels) in enumerate(zip(source_features, source_probs, source_labels)):
            for c in range(self.num_classes):
                mask = labels == c
                if int(mask.sum().item()) < int(min_samples):
                    continue
                self._ema_vector(
                    self.source_prototypes,
                    self.source_proto_valid,
                    (k, c),
                    features[mask].mean(dim=0),
                    self.prototype_momentum,
                )
                self._ema_scalar(
                    self.source_recognition,
                    self.source_recognition_valid,
                    (k, c),
                    probs[mask, c].mean(),
                    self.statistic_momentum,
                )

    @torch.no_grad()
    def update_target(
        self,
        target_features: Sequence[torch.Tensor],
        target_probs_by_source: Sequence[torch.Tensor],
        fused_probs: torch.Tensor,
        confidence_threshold: float,
        min_samples: int = 2,
        use_prototype_quality: bool = True,
        quality_temperature: float = 0.5,
        update_prototypes: bool = True,
        pseudo_labels: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        sample_quality: torch.Tensor | None = None,
    ) -> None:
        """Update target statistics using externally filtered pseudo labels.

        ``pseudo_labels``/``valid_mask`` are optional for backward
        compatibility. The robust RCPA trainer supplies a mask that already
        enforces fused confidence, prediction margin and multi-source
        agreement. This prevents high-confidence but inconsistent samples
        from entering the target prototype memory.
        """
        confidence, fused_pseudo = fused_probs.max(dim=1)
        pseudo = fused_pseudo if pseudo_labels is None else pseudo_labels
        confident = (
            confidence >= float(confidence_threshold)
            if valid_mask is None
            else valid_mask.bool()
        )

        if pseudo.shape != fused_pseudo.shape:
            raise ValueError(
                "pseudo_labels must have shape [batch_size], matching fused_probs."
            )
        if confident.shape != fused_pseudo.shape:
            raise ValueError(
                "valid_mask must have shape [batch_size], matching fused_probs."
            )

        quality_all = confidence.detach() if sample_quality is None else sample_quality.detach()
        if quality_all.shape != fused_pseudo.shape:
            raise ValueError(
                "sample_quality must have shape [batch_size], matching fused_probs."
            )
        quality_all = quality_all.clamp(min=self.eps)

        coverage_now = confident.float().mean()
        self.target_coverage.mul_(self.statistic_momentum).add_(coverage_now, alpha=1.0 - self.statistic_momentum)
        if not update_prototypes:
            return

        for k, (features, probs) in enumerate(zip(target_features, target_probs_by_source)):
            sample_entropy = normalized_entropy(probs, eps=self.eps)
            for c in range(self.num_classes):
                mask = confident & (pseudo == c)
                if int(mask.sum().item()) < int(min_samples):
                    continue

                selected_features = features[mask]
                quality = quality_all[mask].clone()
                if use_prototype_quality and bool(self.target_proto_valid[k, c]):
                    mem_proto = self.target_prototypes[k, c].view(1, -1)
                    similarity = F.cosine_similarity(selected_features.detach(), mem_proto, dim=1)
                    distance = (1.0 - similarity).clamp(min=0.0)
                    quality = quality * torch.exp(-distance / max(float(quality_temperature), self.eps))

                quality = quality / (quality.sum() + self.eps)
                proto = (selected_features * quality.unsqueeze(1)).sum(dim=0)
                self._ema_vector(
                    self.target_prototypes,
                    self.target_proto_valid,
                    (k, c),
                    proto,
                    self.prototype_momentum,
                )
                self._ema_scalar(
                    self.target_entropy,
                    self.target_entropy_valid,
                    (k, c),
                    (sample_entropy[mask] * quality).sum(),
                    self.statistic_momentum,
                )

    @torch.no_grad()
    def update_global_weights(
        self,
        mmd_distances: torch.Tensor,
        tau: float,
        target_entropies: torch.Tensor | None = None,
        source_recognition: torch.Tensor | None = None,
        entropy_weight: float = 1.0,
        recognition_weight: float = 0.30,
        adaptive_pruning: bool = False,
        prune_gap: float = 0.05,
        bottom_floor: float = 0.01,
        max_source_weight: float = 0.65,
    ) -> torch.Tensor:
        """Update global source reliability from discrepancy and confidence.

        The original high-PU_0 model did not rely on MMD alone: target
        uncertainty and source recognition ability also participated in source
        weighting.  Reintroducing these two signals lets a source such as PU_3
        be down-weighted before the class-prototype stage when its target
        predictions are uncertain or its source classifier is less reliable.
        """
        mmd = mmd_distances.detach().float()

        def standardize_vector(values: torch.Tensor) -> torch.Tensor:
            values = values.detach().float()
            return (values - values.mean()) / (
                values.std(unbiased=False) + self.eps
            )

        score = -standardize_vector(mmd)
        if target_entropies is not None:
            score = score - float(entropy_weight) * standardize_vector(
                target_entropies
            )
        if source_recognition is not None:
            score = score + float(recognition_weight) * standardize_vector(
                source_recognition
            )

        raw = torch.softmax(score / max(float(tau), self.eps), dim=0)
        raw = adaptive_top2_source_gate(
            raw,
            enabled=adaptive_pruning,
            prune_gap=prune_gap,
            bottom_floor=bottom_floor,
            max_source_weight=max_source_weight,
            eps=self.eps,
        )
        self.global_source_weights.mul_(self.weight_momentum).add_(
            raw, alpha=1.0 - self.weight_momentum
        )
        self.global_source_weights.copy_(
            normalize_source_weights(self.global_source_weights, self.eps)
        )
        self.global_source_weights.copy_(
            adaptive_top2_source_gate(
                self.global_source_weights,
                enabled=adaptive_pruning,
                prune_gap=prune_gap,
                bottom_floor=bottom_floor,
                max_source_weight=max_source_weight,
                eps=self.eps,
            )
        )
        return self.global_source_weights.clone()

    @torch.no_grad()
    def refresh_class_reliability(
        self,
        distance_weight: float,
        entropy_weight: float,
        recognition_weight: float,
        tau: float,
        global_prior_mix: float = 0.20,
        uniform_smoothing: float = 0.15,
        score_clip: float = 2.0,
        min_source_weight: float = 0.0,
        adaptive_pruning: bool = False,
        prune_gap: float = 0.05,
        bottom_floor: float = 0.01,
        max_source_weight: float = 0.65,
    ) -> torch.Tensor:
        """Refresh source-class reliability with controlled negative-source pruning."""
        distances = torch.ones_like(self.source_recognition)
        entropy = self.target_entropy.clone()
        recognition = self.source_recognition.clone()

        for k in range(self.num_sources):
            for c in range(self.num_classes):
                if bool(self.source_proto_valid[k, c] and self.target_proto_valid[k, c]):
                    s = self.source_prototypes[k, c].view(1, -1)
                    t = self.target_prototypes[k, c].view(1, -1)
                    distances[k, c] = 1.0 - F.cosine_similarity(s, t, dim=1)[0]

        for c in range(self.num_classes):
            valid_dist = self.source_proto_valid[:, c] & self.target_proto_valid[:, c]
            distances[~valid_dist, c] = distances[valid_dist, c].max() + 0.10 if bool(valid_dist.any()) else 1.0

            valid_ent = self.target_entropy_valid[:, c]
            entropy[~valid_ent, c] = entropy[valid_ent, c].max() if bool(valid_ent.any()) else 1.0

            valid_rec = self.source_recognition_valid[:, c]
            recognition[~valid_rec, c] = recognition[valid_rec, c].min() if bool(valid_rec.any()) else 1.0 / self.num_classes

        score = (
            -float(distance_weight) * column_standardize(distances, self.eps)
            -float(entropy_weight) * column_standardize(entropy, self.eps)
            +float(recognition_weight) * column_standardize(recognition, self.eps)
        )

        if float(score_clip) > 0.0:
            score = score.clamp(min=-float(score_clip), max=float(score_clip))

        raw_weights = torch.softmax(score / max(float(tau), self.eps), dim=0)

        prior_mix = min(1.0, max(0.0, float(global_prior_mix)))
        if prior_mix > 0.0:
            global_prior = self.global_source_weights.view(-1, 1).expand_as(raw_weights)
            raw_weights = (1.0 - prior_mix) * raw_weights + prior_mix * global_prior

        smoothing = min(1.0, max(0.0, float(uniform_smoothing)))
        if smoothing > 0.0:
            uniform = torch.full_like(raw_weights, 1.0 / self.num_sources)
            raw_weights = (1.0 - smoothing) * raw_weights + smoothing * uniform

        floor = max(0.0, float(min_source_weight))
        if floor > 0.0:
            if floor * self.num_sources >= 1.0:
                raise ValueError(
                    "min_source_weight * num_sources must be smaller than 1."
                )
            raw_weights = raw_weights.clamp(min=floor)
            raw_weights = normalize_source_weights(raw_weights, self.eps)

        raw_weights = adaptive_top2_source_gate(
            raw_weights,
            enabled=adaptive_pruning,
            prune_gap=prune_gap,
            bottom_floor=bottom_floor,
            max_source_weight=max_source_weight,
            eps=self.eps,
        )

        self.class_source_weights.mul_(self.weight_momentum).add_(raw_weights, alpha=1.0 - self.weight_momentum)
        self.class_source_weights.copy_(normalize_source_weights(self.class_source_weights, self.eps))

        if floor > 0.0:
            self.class_source_weights.clamp_(min=floor)
            self.class_source_weights.copy_(
                normalize_source_weights(self.class_source_weights, self.eps)
            )

        self.class_source_weights.copy_(
            adaptive_top2_source_gate(
                self.class_source_weights,
                enabled=adaptive_pruning,
                prune_gap=prune_gap,
                bottom_floor=bottom_floor,
                max_source_weight=max_source_weight,
                eps=self.eps,
            )
        )
        return self.class_source_weights.clone()

    @torch.no_grad()
    def source_readiness(self) -> float:
        values = []
        for k in range(self.num_sources):
            valid = self.source_recognition_valid[k]
            values.append(
                self.source_recognition[k, valid].mean()
                if bool(valid.any())
                else torch.tensor(0.0, device=self.source_recognition.device)
            )
        return float(torch.stack(values).min().item())


class SupervisedContrastiveLoss(nn.Module):
    """Single-view supervised contrastive loss.

    Samples sharing a label are positives; samples with different labels are
    negatives.  Anchors without another positive sample are excluded.
    """

    def __init__(self, temperature: float = 0.20, eps: float = 1e-8) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.dim() > 2:
            features = torch.flatten(features, start_dim=1)
        labels = labels.reshape(-1)
        if features.size(0) != labels.size(0):
            raise ValueError("features and labels must have the same batch size.")
        if features.size(0) <= 1:
            return features.sum() * 0.0

        features = F.normalize(features, p=2, dim=1)
        logits = features.mm(features.t()) / max(self.temperature, self.eps)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        same_class = labels[:, None].eq(labels[None, :]).float()
        self_mask = torch.eye(
            features.size(0), device=features.device, dtype=features.dtype
        )
        logits_mask = 1.0 - self_mask
        positive_mask = same_class * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps)
        )
        positive_count = positive_mask.sum(dim=1)
        valid_anchor = positive_count > 0
        if not bool(valid_anchor.any()):
            return features.sum() * 0.0

        mean_log_prob_positive = (positive_mask * log_prob).sum(dim=1) / (
            positive_count + self.eps
        )
        return -mean_log_prob_positive[valid_anchor].mean()


class MinimumClassConfusionLoss(nn.Module):
    """Minimum class-confusion loss computed from target probabilities."""

    def __init__(self, temperature: float = 2.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(min=self.eps, max=1.0)
        sharpened = probs.pow(1.0 / max(self.temperature, self.eps))
        sharpened = sharpened / (sharpened.sum(dim=1, keepdim=True) + self.eps)
        entropy = -(sharpened * sharpened.log()).sum(dim=1)
        sample_weight = 1.0 + torch.exp(-entropy)
        sample_weight = sample_weight * (sharpened.size(0) / (sample_weight.sum() + self.eps))
        confusion = sharpened.t().mm(sharpened * sample_weight.unsqueeze(1))
        confusion = confusion / (confusion.sum(dim=1, keepdim=True) + self.eps)
        off_diagonal = confusion.sum() - torch.diagonal(confusion).sum()
        return off_diagonal / max(confusion.size(0), 1)
