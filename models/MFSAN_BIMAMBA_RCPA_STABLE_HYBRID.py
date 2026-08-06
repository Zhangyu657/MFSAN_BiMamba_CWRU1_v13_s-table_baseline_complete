# -*- coding: utf-8 -*-
"""Stable-hybrid MFSAN-BiMamba-RCPA for the difficult PU_0 transfer task.

This model keeps the progressive RCPA framework and adds two corrections that
are directly motivated by the supplied PU_0 logs:




python train.py \
  --model_name MFSAN_CDAN_BIMAMBA_CW_RWCA_V6_STABLE_GATE \
  --source PU_0,PU_1,PU_3 \
  --target PU_2 \
  --train_mode multi_source \
  --data_dir /workspace/PU_TL_9_replace \
  --include_faults K001,KA04,KA16,KA30,KB23,KB24,KI04,KI16,KI17 \
  --signal_size 1024 \
  --target_test_size 0.40 \
  --target_split_mode time \
  --backbone BIMAMBA \
  --cuda_device 0 \
  --max_epoch 20 \
  --batch_size 64 \
  --num_workers 4 \
  --opt sgd \
  --lr 0.01 \
  --momentum 0.9 \
  --weight_decay 0.0005 \
  --lr_scheduler stepLR \
  --steps 10 \
  --gamma 0.2 \
  --dropout 0.0 \
  --lambda_cda 0.0 \
  --lambda_adv 0.02 \
  --lambda_grl 1.0 \
  --lambda_ent 0.005 \
  --lambda_clmmd 0.005 \
  --adv_hidden_dim 256 \
  --adv_detach_prob True \
  --adv_use_entropy_weight True \
  --adv_conf_thresh 0.0 \
  --pl_conf_thresh 0.80 \
  --pl_min_target 2 \
  --rw_tau 0.50 \
  --rw_mmd_weight 1.0 \
  --rw_ent_weight 1.0 \
  --rw_detach_weights True \
  --rw_ema_momentum 0.90 \
  --rw_eval_use_entropy True \
  --rw_eval_tau 0.50 \
  --cw_warmup_epochs 3 \
  --cw_alpha 0.30 \
  --cw_alpha_ramp_epochs 3 \
  --rec_score_weight 0.30 \
  --rec_score_mode prob \
  --rec_score_detach True \
  --lambda_supcon 0.01 \
  --supcon_temperature 0.20 \
  --supcon_start_epoch 3 \
  --supcon_feature_mode G \
  --supcon_focus_classes 1,3,8 \
  --lambda_mca 0.02 \
  --mca_start_epoch 3 \
  --mca_use_reliability True \
  --mca_detach_fused True \
  --v6_gate_enabled True \
  --v6_gate_start_epoch 4 \
  --v6_gate_confirm_epochs 3 \
  --v6_gate_release_epochs 3 \
  --v6_gate_confirm_gap 0.08 \
  --v6_gate_release_gap 0.03 \
  --v6_gate_preconfirm_floor 0.05 \
  --v6_gate_bottom_floor 0.01 \
  --v6_gate_max_source_weight 0.75 \
  --v6_gate_apply_to_supcon True \
  --v6_supcon_source_min_weight 0.05 \
  --v6_class_weight_power 1.20 \
  --v6_class_alignment_boost 1.00 \
  --v6_mca_pairwise_weight 0.25 \
  --save True \
  --save_best True \
  --save_dir ./ckpt/PU0_V6_StableGate \
  --random_state 2027










1. Stable negative-source suppression
   - no hard source pruning during early warm-up;
   - target-side three-classifier consensus is added to source reliability;
   - a source is suppressed only after it is ranked last for several complete
     epochs with a persistent reliability gap;
   - the decision can be released if later evidence changes.

2. Stage-aware recovery of the useful losses from the original high-accuracy
   model
   - global stage: MMD + CDAN + target entropy;
   - prototype stage: global terms + prototype/MCC + CLMMD + MCA + classifier
     consistency;
   - focused SupCon for KA04/KA16/KI17 is retained with epoch decay.

The objective is deliberately stage-aware rather than activating every loss
from the first mini-batch.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from tqdm import tqdm

import utils
from models.MFSAN_BIMAMBA_RCPA import Trainer as BaseRCPATrainer
from models.rcpa_components import (
    StableSourcePruningController,
    normalize_source_weights,
    normalized_entropy,
    three_source_consensus_scores,
)


class GradientReverseFunction(Function):
    """Gradient reversal used by CDAN."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, coeff: float) -> torch.Tensor:
        ctx.coeff = float(coeff)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coeff * grad_output, None


def grad_reverse(x: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
    return GradientReverseFunction.apply(x, coeff)


class DomainDiscriminator(nn.Module):
    """Conditional domain discriminator for one source-target pair."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        hidden2 = max(hidden_dim // 2, 8)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Trainer(BaseRCPATrainer):
    """RCPA trainer with stable source pruning and stage-aware hybrid losses."""

    def __init__(self, args):
        super().__init__(args)

        # ------------------------------------------------------------------
        # Stable source-pruning controller
        # ------------------------------------------------------------------
        self.global_consensus_weight = float(
            getattr(args, "rcpa_global_consensus_weight", 1.0)
        )
        self.consensus_confidence = float(
            getattr(args, "rcpa_consensus_confidence", 0.55)
        )
        self.source_gate = StableSourcePruningController(
            num_sources=self.num_source,
            enabled=bool(getattr(args, "rcpa_adaptive_source_pruning", True)),
            warmup_epochs=int(getattr(args, "rcpa_gate_warmup_epochs", 3)),
            start_epoch=int(getattr(args, "rcpa_gate_start_epoch", 4)),
            confirm_epochs=int(getattr(args, "rcpa_gate_confirm_epochs", 3)),
            release_epochs=int(getattr(args, "rcpa_gate_release_epochs", 3)),
            confirm_gap=float(getattr(args, "rcpa_gate_confirm_gap", 0.08)),
            preconfirm_floor=float(
                getattr(args, "rcpa_gate_preconfirm_floor", 0.10)
            ),
            bottom_floor=float(
                getattr(args, "rcpa_source_bottom_floor", 0.01)
            ),
            max_source_weight=float(
                getattr(args, "rcpa_max_source_weight", 0.65)
            ),
            eps=self.eps,
        ).to(self.device)

        # ------------------------------------------------------------------
        # Stage-aware auxiliary losses from the original V5-MCA model
        # ------------------------------------------------------------------
        self.lambda_cda = float(getattr(args, "lambda_cda", 0.0))
        self.lambda_ent = float(getattr(args, "lambda_ent", 0.005))
        self.lambda_adv = float(getattr(args, "lambda_adv", 0.02))
        self.lambda_grl = float(getattr(args, "lambda_grl", 1.0))
        self.lambda_clmmd = float(getattr(args, "lambda_clmmd", 0.005))
        self.lambda_mca = float(getattr(args, "lambda_mca", 0.02))
        self.lambda_consistency = float(
            getattr(args, "rcpa_lambda_consistency", 1.0)
        )

        self.cdan_start_epoch = int(
            getattr(args, "rcpa_cdan_start_epoch", 4)
        )
        self.clmmd_start_epoch = int(
            getattr(args, "rcpa_clmmd_start_epoch", 8)
        )
        self.mca_start_epoch = int(getattr(args, "rcpa_mca_start_epoch", 8))
        self.consistency_start_epoch = int(
            getattr(args, "rcpa_consistency_start_epoch", 8)
        )

        self.detach_prob = bool(getattr(args, "cda_detach_prob", True))
        self.adv_detach_prob = bool(getattr(args, "adv_detach_prob", True))
        self.adv_use_entropy_weight = bool(
            getattr(args, "adv_use_entropy_weight", True)
        )
        self.adv_conf_thresh = float(getattr(args, "adv_conf_thresh", 0.0))
        self.mca_use_reliability = bool(
            getattr(args, "mca_use_reliability", True)
        )
        self.mca_detach_fused = bool(getattr(args, "mca_detach_fused", True))
        self.mca_eps = float(getattr(args, "mca_eps", 1e-5))

        self.clmmd_kernel_num = int(getattr(args, "clmmd_kernel_num", 5))
        self.clmmd_kernel_mul = float(getattr(args, "clmmd_kernel_mul", 2.0))
        self.clmmd_min_source = int(getattr(args, "clmmd_min_source", 2))
        self.clmmd_min_target_weight = float(
            getattr(args, "clmmd_min_target_weight", 1e-3)
        )

        # Focused SupCon is useful for class boundaries, but in the simplified
        # RCPA model it was relatively too large.  Decay it after prototype
        # alignment starts.
        self.hard_supcon_final_factor = float(
            getattr(args, "rcpa_hard_supcon_final_factor", 0.20)
        )
        self.hard_supcon_decay_start_epoch = int(
            getattr(args, "rcpa_hard_supcon_decay_start_epoch", 8)
        )
        self.hard_supcon_decay_end_epoch = int(
            getattr(args, "rcpa_hard_supcon_decay_end_epoch", 15)
        )

        self.joint_dim = self.num_classes * self.feature_dim
        self.cda_mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
            kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)]
        )
        adv_hidden_dim = int(getattr(args, "adv_hidden_dim", 256))
        self.Ds = nn.ModuleList(
            [
                DomainDiscriminator(
                    input_dim=self.joint_dim,
                    hidden_dim=adv_hidden_dim,
                    dropout=float(args.dropout),
                )
                for _ in range(self.num_source)
            ]
        ).to(self.device)

        # Rebuild the optimizer because BaseRCPATrainer created it before the
        # discriminators existed.
        self.optimizer = self._get_optimizer([self.G, self.Fs, self.Cs, self.Ds])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        logging.info("Using model: MFSAN_BIMAMBA_RCPA_STABLE_HYBRID")
        logging.info(
            "Stable source gate: sources=%s warmup=%d start=%d confirm=%d "
            "release=%d gap=%.3f pre_floor=%.3f bottom_floor=%.3f cap=%.3f",
            list(self.src),
            self.source_gate.warmup_epochs,
            self.source_gate.start_epoch,
            self.source_gate.confirm_epochs,
            self.source_gate.release_epochs,
            self.source_gate.confirm_gap,
            self.source_gate.preconfirm_floor,
            self.source_gate.bottom_floor,
            self.source_gate.max_source_weight,
        )
        logging.info(
            "Global reliability adds target-consensus weight=%.3f "
            "(confidence threshold=%.3f)",
            self.global_consensus_weight,
            self.consensus_confidence,
        )
        logging.info(
            "Stage-aware hybrid losses: CDAN=%.4f ENT=%.4f CDA=%.4f "
            "CLMMD=%.4f MCA=%.4f CONSISTENCY=%.4f",
            self.lambda_adv,
            self.lambda_ent,
            self.lambda_cda,
            self.lambda_clmmd,
            self.lambda_mca,
            self.lambda_consistency,
        )
        logging.info(
            "Hybrid loss starts: CDAN/global=%d CLMMD=%d MCA=%d consistency=%d",
            self.cdan_start_epoch,
            self.clmmd_start_epoch,
            self.mca_start_epoch,
            self.consistency_start_epoch,
        )
        logging.info(
            "Hard SupCon schedule: base_lambda=%.5f decay=[%d,%d] final_factor=%.3f",
            self.lambda_hard_supcon,
            self.hard_supcon_decay_start_epoch,
            self.hard_supcon_decay_end_epoch,
            self.hard_supcon_final_factor,
        )

    # ==================================================================
    # State and checkpoints
    # ==================================================================
    def _set_to_train(self):
        super()._set_to_train()
        self.Ds.train()

    def _set_to_eval(self):
        super()._set_to_eval()
        self.Ds.eval()

    def _checkpoint_dict(self):
        ckpt = super()._checkpoint_dict()
        ckpt.update(
            {
                "model_name": "MFSAN_BIMAMBA_RCPA_STABLE_HYBRID",
                "Ds": self.Ds.state_dict(),
                "source_gate": self.source_gate.state_dict(),
                "lambda_adv": self.lambda_adv,
                "lambda_ent": self.lambda_ent,
                "lambda_clmmd": self.lambda_clmmd,
                "lambda_mca": self.lambda_mca,
                "lambda_consistency": self.lambda_consistency,
                "cur_epoch": int(self._cur_epoch),
            }
        )
        return ckpt

    def load_model(self):
        logging.info("Loading model from %s", self.args.load_path)
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.Fs.load_state_dict(ckpt["Fs"])
        self.Cs.load_state_dict(ckpt["Cs"])
        if "memory" in ckpt:
            self.memory.load_state_dict(ckpt["memory"], strict=True)
        if "Ds" in ckpt:
            self.Ds.load_state_dict(ckpt["Ds"], strict=True)
        else:
            logging.warning("Checkpoint has no domain discriminator state.")
        if "source_gate" in ckpt:
            self.source_gate.load_state_dict(ckpt["source_gate"], strict=True)
        self._current_stage = int(ckpt.get("current_stage", 2))
        self._stage1_start_epoch = ckpt.get("stage1_start_epoch", None)
        self._stage2_start_epoch = ckpt.get("stage2_start_epoch", None)
        self._last_rho = float(ckpt.get("last_rho", 0.0))
        self._cur_epoch = int(ckpt.get("cur_epoch", self.args.max_epoch))

    # ==================================================================
    # Stable source reliability and fusion
    # ==================================================================
    @staticmethod
    def _standardize_vector(values: torch.Tensor, eps: float) -> torch.Tensor:
        values = values.float()
        if values.numel() <= 1:
            return values * 0.0
        return (values - values.mean()) / (
            values.std(unbiased=False) + eps
        )

    @torch.no_grad()
    def _update_raw_global_reliability(
        self,
        mmd_tensor: torch.Tensor,
        target_entropy_tensor: torch.Tensor,
        source_recognition_tensor: torch.Tensor,
        consensus_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update ungated EMA source weights and return EMA, raw, score."""
        score = (
            -self._standardize_vector(mmd_tensor.detach(), self.eps)
            -self.global_entropy_weight
            * self._standardize_vector(target_entropy_tensor.detach(), self.eps)
            +self.global_recognition_weight
            * self._standardize_vector(source_recognition_tensor.detach(), self.eps)
            +self.global_consensus_weight
            * self._standardize_vector(consensus_tensor.detach(), self.eps)
        )
        raw = torch.softmax(
            score / max(self.global_weight_tau, self.eps), dim=0
        )
        momentum = float(self.memory.weight_momentum)
        self.memory.global_source_weights.mul_(momentum).add_(
            raw, alpha=1.0 - momentum
        )
        self.memory.global_source_weights.copy_(
            normalize_source_weights(self.memory.global_source_weights, self.eps)
        )
        return self.memory.global_source_weights.clone(), raw.clone(), score.clone()

    def _effective_global_weights(self) -> torch.Tensor:
        return self.source_gate.apply(
            self.memory.global_source_weights, epoch=self._cur_epoch
        ).detach()

    def _guided_class_weights(self, rho: float) -> torch.Tensor:
        global_weights = self._effective_global_weights().view(-1, 1).repeat(
            1, self.num_classes
        )
        raw_class = self.memory.class_source_weights
        guided = (1.0 - float(rho)) * global_weights + float(rho) * raw_class
        guided = normalize_source_weights(guided, self.eps)
        return self.source_gate.apply(guided, epoch=self._cur_epoch).detach()

    def _eval_fusion(
        self, probs_by_source: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base = self._guided_class_weights(self._last_rho).clamp_min(self.eps)
        stacked = torch.stack(probs_by_source, dim=0)
        if not self.dynamic_eval_fusion:
            fused = (stacked * base[:, None, :]).sum(dim=0)
            fused = fused / (fused.sum(dim=1, keepdim=True) + self.eps)
            return fused, base

        entropy = torch.stack(
            [normalized_entropy(p, self.eps) for p in probs_by_source], dim=0
        )
        dynamic = base[:, None, :] * torch.exp(
            -self.eval_entropy_eta * entropy
        ).unsqueeze(-1)
        dynamic = torch.softmax(
            torch.log(dynamic.clamp_min(self.eps))
            / max(self.eval_weight_tau, self.eps),
            dim=0,
        )
        dynamic = self.source_gate.apply(dynamic, epoch=self._cur_epoch)
        fused = (stacked * dynamic).sum(dim=0)
        fused = fused / (fused.sum(dim=1, keepdim=True) + self.eps)
        return fused, dynamic

    # ==================================================================
    # Hybrid loss utilities
    # ==================================================================
    def _target_entropy_loss(self, probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(min=self.eps, max=1.0)
        return -(probs * probs.log()).sum(dim=1).mean()

    def _entropy_weight(self, probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(min=self.eps, max=1.0)
        entropy = -(probs * probs.log()).sum(dim=1)
        weight = torch.exp(-entropy).detach()
        return weight / (weight.mean() + self.eps)

    def _joint_feature(
        self,
        features: torch.Tensor,
        probs: torch.Tensor,
        detach_prob: bool,
    ) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=1)
        probs_used = probs.detach() if detach_prob else probs
        joint = torch.bmm(probs_used.unsqueeze(2), features.unsqueeze(1))
        joint = joint.reshape(features.size(0), -1)
        return F.normalize(joint, p=2, dim=1)

    def _conditional_mmd(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_probs: torch.Tensor,
        target_probs: torch.Tensor,
    ) -> torch.Tensor:
        joint_s = self._joint_feature(
            source_features, source_probs, self.detach_prob
        )
        joint_t = self._joint_feature(
            target_features, target_probs, self.detach_prob
        )
        return self.cda_mkmmd(joint_s, joint_t)

    def _domain_adversarial_loss(
        self,
        source_index: int,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_probs: torch.Tensor,
        grl_coeff: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source_onehot = F.one_hot(
            source_labels, num_classes=self.num_classes
        ).float()

        if self.adv_conf_thresh > 0.0:
            confidence = target_probs.detach().max(dim=1).values
            target_mask = confidence >= self.adv_conf_thresh
            if int(target_mask.sum().item()) < 2:
                zero = source_features.sum() * 0.0
                return zero, zero.detach()
            target_features_used = target_features[target_mask]
            target_probs_used = target_probs[target_mask]
        else:
            target_features_used = target_features
            target_probs_used = target_probs

        joint_s = self._joint_feature(source_features, source_onehot, True)
        joint_t = self._joint_feature(
            target_features_used, target_probs_used, self.adv_detach_prob
        )
        joint = torch.cat([joint_s, joint_t], dim=0)
        labels = torch.cat(
            [
                torch.zeros(joint_s.size(0), dtype=torch.long, device=self.device),
                torch.ones(joint_t.size(0), dtype=torch.long, device=self.device),
            ],
            dim=0,
        )
        logits = self.Ds[source_index](grad_reverse(joint, grl_coeff))
        per_sample = F.cross_entropy(logits, labels, reduction="none")
        if self.adv_use_entropy_weight:
            weight_s = torch.ones(joint_s.size(0), device=self.device)
            weight_t = self._entropy_weight(target_probs_used)
            weights = torch.cat([weight_s, weight_t], dim=0)
            loss = (per_sample * weights).sum() / (weights.sum() + self.eps)
        else:
            loss = per_sample.mean()
        with torch.no_grad():
            accuracy = logits.argmax(dim=1).eq(labels).float().mean()
        return loss, accuracy

    def _gaussian_kernel_matrix(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_x = x.size(0)
        total = torch.cat([x, y], dim=0)
        l2 = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(dim=2)
        with torch.no_grad():
            base = l2.detach().mean().clamp_min(self.eps)
        kernel_sum = torch.zeros_like(l2)
        middle = self.clmmd_kernel_num // 2
        for index in range(self.clmmd_kernel_num):
            sigma = base * (self.clmmd_kernel_mul ** (index - middle))
            kernel_sum = kernel_sum + torch.exp(-l2 / (2.0 * sigma))
        return (
            kernel_sum[:n_x, :n_x],
            kernel_sum[:n_x, n_x:],
            kernel_sum[n_x:, n_x:],
        )

    def _classwise_lmmd_per_class(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_class_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss_vector = torch.zeros(self.num_classes, device=self.device)
        valid_vector = torch.zeros(self.num_classes, device=self.device)
        target_class_probs = target_class_probs.clamp(min=self.eps, max=1.0)
        target_class_probs = target_class_probs / (
            target_class_probs.sum(dim=1, keepdim=True) + self.eps
        )

        for class_index in range(self.num_classes):
            source_mask = source_labels == class_index
            source_count = int(source_mask.sum().item())
            if source_count < self.clmmd_min_source:
                continue
            target_weight_raw = target_class_probs[:, class_index]
            target_weight_sum = target_weight_raw.sum()
            if target_weight_sum.detach().item() < self.clmmd_min_target_weight:
                continue

            source_selected = source_features[source_mask]
            k_ss, k_st, k_tt = self._gaussian_kernel_matrix(
                source_selected, target_features
            )
            source_weight = torch.ones(
                source_count, device=self.device
            ) / float(source_count)
            target_weight = target_weight_raw / (
                target_weight_sum + self.eps
            )
            loss_class = (
                torch.sum(
                    source_weight[:, None] * source_weight[None, :] * k_ss
                )
                + torch.sum(
                    target_weight[:, None] * target_weight[None, :] * k_tt
                )
                - 2.0
                * torch.sum(
                    source_weight[:, None] * target_weight[None, :] * k_st
                )
            )
            loss_vector[class_index] = loss_class
            valid_vector[class_index] = 1.0
        return loss_vector, valid_vector

    def _classifier_consistency_loss(
        self,
        target_probs: Sequence[torch.Tensor],
        fused_probs: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._cur_epoch < self.consistency_start_epoch:
            return fused_probs.sum() * 0.0
        reference = fused_probs.detach()
        loss = fused_probs.sum() * 0.0
        for source_index in range(self.num_source):
            difference = torch.abs(target_probs[source_index] - reference)
            loss = loss + (
                difference * class_weights[source_index].view(1, -1)
            ).sum(dim=1).mean() / float(self.num_classes)
        return loss

    def _normalize_class_correlation(self, probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(min=self.eps, max=1.0)
        probs = probs / (probs.sum(dim=1, keepdim=True) + self.eps)
        correlation = probs.t().mm(probs) / max(float(probs.size(0)), 1.0)
        return correlation / (torch.norm(correlation, p="fro") + self.mca_eps)

    def _multi_classifier_alignment_loss(
        self,
        target_probs: Sequence[torch.Tensor],
        fused_probs: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._cur_epoch < self.mca_start_epoch or self.lambda_mca <= 0.0:
            return fused_probs.sum() * 0.0
        reference_probs = fused_probs.detach() if self.mca_detach_fused else fused_probs
        reference_correlation = self._normalize_class_correlation(reference_probs)
        source_weights = class_weights.mean(dim=1)
        source_weights = source_weights / (source_weights.sum() + self.eps)
        loss = fused_probs.sum() * 0.0
        denominator = fused_probs.sum() * 0.0
        for source_index in range(self.num_source):
            correlation = self._normalize_class_correlation(
                target_probs[source_index]
            )
            difference2 = (correlation - reference_correlation) ** 2
            if self.mca_use_reliability:
                pair_weight = torch.outer(
                    class_weights[source_index], class_weights[source_index]
                ).detach()
                pair_loss = (difference2 * pair_weight).sum() / (
                    pair_weight.sum() + self.mca_eps
                )
                loss = loss + source_weights[source_index].detach() * pair_loss
                denominator = denominator + source_weights[source_index].detach()
            else:
                loss = loss + difference2.mean()
                denominator = denominator + 1.0
        return loss / (denominator + self.mca_eps)

    def _hard_supcon_lambda_now(self) -> float:
        base = float(self.lambda_hard_supcon)
        start = int(self.hard_supcon_decay_start_epoch)
        end = max(start + 1, int(self.hard_supcon_decay_end_epoch))
        if self._cur_epoch <= start:
            return base
        if self._cur_epoch >= end:
            return base * self.hard_supcon_final_factor
        progress = (self._cur_epoch - start) / float(end - start)
        factor = 1.0 - progress * (1.0 - self.hard_supcon_final_factor)
        return base * factor

    # ==================================================================
    # Training
    # ==================================================================
    def _train_one_epoch(self, epoch_acc, epoch_loss):
        stage_counts = torch.zeros(3, device=self.device)
        class_weight_sum = torch.zeros(
            self.num_source, self.num_classes, device=self.device
        )
        effective_global_sum = torch.zeros(self.num_source, device=self.device)
        raw_global_sum = torch.zeros(self.num_source, device=self.device)
        reliability_score_sum = torch.zeros(self.num_source, device=self.device)
        mmd_component_sum = torch.zeros(self.num_source, device=self.device)
        entropy_component_sum = torch.zeros(self.num_source, device=self.device)
        recognition_component_sum = torch.zeros(self.num_source, device=self.device)
        consensus_component_sum = torch.zeros(self.num_source, device=self.device)
        valid_proto_pairs_sum = 0.0

        for batch_index in tqdm(range(self.num_iter), ascii=True):
            target_data, target_batch_meta = self._get_next_batch("train")
            source_data_list: List[torch.Tensor] = []
            source_label_list: List[torch.Tensor] = []
            for source in self.src:
                source_data, actual_labels = self._get_next_batch(
                    source, return_actual=True
                )
                source_label_list.append(
                    self._get_train_label(
                        actual_labels, label_set=self.src_labels_flat
                    )
                )
                source_data_list.append(source_data)

            self.optimizer.zero_grad()
            all_data = torch.cat(source_data_list + [target_data], dim=0)
            all_shared = self.G(all_data)
            split_sizes = [item.size(0) for item in source_data_list] + [
                target_data.size(0)
            ]
            split_shared = torch.split(all_shared, split_sizes, dim=0)
            shared_sources = list(split_shared[:-1])
            shared_target = split_shared[-1]

            source_features: List[torch.Tensor] = []
            target_features: List[torch.Tensor] = []
            source_logits: List[torch.Tensor] = []
            target_logits: List[torch.Tensor] = []
            source_probs: List[torch.Tensor] = []
            target_probs: List[torch.Tensor] = []
            mmd_losses: List[torch.Tensor] = []
            cls_losses: List[torch.Tensor] = []

            for source_index in range(self.num_source):
                feature_source = self.Fs[source_index](shared_sources[source_index])
                feature_target = self.Fs[source_index](shared_target)
                logits_source = self.Cs[source_index](feature_source)
                logits_target = self.Cs[source_index](feature_target)
                probs_source = F.softmax(logits_source, dim=1)
                probs_target = F.softmax(logits_target, dim=1)

                source_features.append(feature_source)
                target_features.append(feature_target)
                source_logits.append(logits_source)
                target_logits.append(logits_target)
                source_probs.append(probs_source)
                target_probs.append(probs_target)
                cls_losses.append(
                    F.cross_entropy(
                        logits_source,
                        source_label_list[source_index],
                        label_smoothing=self.label_smoothing,
                    )
                )
                mmd_losses.append(self.mkmmd(feature_source, feature_target))
                epoch_acc["Source Data"] += self._get_accuracy(
                    logits_source, source_label_list[source_index]
                ) / float(self.num_source)

            loss_cls = torch.stack(cls_losses).mean()
            mmd_tensor = torch.stack(mmd_losses)
            target_entropy_tensor = torch.stack(
                [normalized_entropy(probs, self.eps).mean() for probs in target_probs]
            )
            source_recognition_tensor = torch.stack(
                [
                    probs.gather(1, labels.view(-1, 1)).mean()
                    for probs, labels in zip(source_probs, source_label_list)
                ]
            )
            target_consensus_tensor = three_source_consensus_scores(
                target_probs,
                confidence_threshold=self.consensus_confidence,
                eps=self.eps,
            )
            raw_ema, raw_current, reliability_score = self._update_raw_global_reliability(
                mmd_tensor,
                target_entropy_tensor,
                source_recognition_tensor,
                target_consensus_tensor,
            )
            effective_global_weights = self.source_gate.apply(
                raw_ema, epoch=self._cur_epoch
            ).detach()

            (
                stage_before,
                rho_before,
                source_ready_before,
                target_coverage_before,
                valid_target_classes_before,
            ) = self._resolve_stage()
            prior_weights = self._guided_class_weights(rho_before)
            prior_fused = self._classwise_fusion(target_probs, prior_weights)
            (
                prior_pseudo,
                prior_mask,
                prior_quality,
                prior_pseudo_stats,
            ) = self._build_reliable_pseudo_labels(target_probs, prior_fused)

            self.memory.update_source(
                source_features,
                source_probs,
                source_label_list,
                min_samples=self.min_source_samples,
            )
            self.memory.update_target(
                target_features,
                target_probs,
                prior_fused,
                confidence_threshold=self.conf_threshold,
                min_samples=self.min_target_samples,
                use_prototype_quality=self.use_prototype_quality,
                quality_temperature=self.prototype_quality_temperature,
                update_prototypes=(stage_before >= 1),
                pseudo_labels=prior_pseudo,
                valid_mask=prior_mask,
                sample_quality=prior_quality,
            )
            # Class reliability is kept raw.  Stable global pruning is applied
            # only in _guided_class_weights, so one noisy batch cannot hard-prune
            # a different source for each class.
            self.memory.refresh_class_reliability(
                distance_weight=self.distance_weight,
                entropy_weight=self.entropy_weight,
                recognition_weight=self.recognition_weight,
                tau=self.reliability_tau,
                global_prior_mix=self.reliability_global_prior,
                uniform_smoothing=self.reliability_smoothing,
                score_clip=self.reliability_score_clip,
                min_source_weight=self.min_source_weight,
                adaptive_pruning=False,
            )

            stage, rho, source_ready, target_coverage, valid_target_classes = (
                self._resolve_stage()
            )
            stage_counts[stage] += 1.0
            class_weights = self._guided_class_weights(rho)
            fused_target_probs = self._classwise_fusion(target_probs, class_weights)
            pseudo_labels, valid_pseudo_mask, pseudo_quality, pseudo_stats = (
                self._build_reliable_pseudo_labels(target_probs, fused_target_probs)
            )

            loss_global = torch.sum(effective_global_weights * mmd_tensor)
            if stage == 2:
                loss_proto, valid_pairs = self._prototype_alignment_loss(
                    source_features,
                    target_features,
                    source_label_list,
                    pseudo_labels,
                    valid_pseudo_mask,
                    pseudo_quality,
                    class_weights,
                )
                loss_mcc = self.mcc_loss(fused_target_probs)
            else:
                loss_proto = fused_target_probs.sum() * 0.0
                loss_mcc = fused_target_probs.sum() * 0.0
                valid_pairs = 0
            valid_proto_pairs_sum += float(valid_pairs)

            adapt_strength = self._adaptation_strength(stage)
            progress = (
                (self._cur_epoch - 1) * self.num_iter + batch_index + 1
            ) / max(float(self.args.max_epoch * self.num_iter), 1.0)
            grl_coeff = self.lambda_grl * (
                2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
            )

            loss_adv = fused_target_probs.sum() * 0.0
            domain_accuracy = fused_target_probs.sum() * 0.0
            loss_cda = fused_target_probs.sum() * 0.0
            if stage >= 1 and self._cur_epoch >= self.cdan_start_epoch:
                for source_index in range(self.num_source):
                    adv_k, domain_acc_k = self._domain_adversarial_loss(
                        source_index,
                        source_features[source_index],
                        target_features[source_index],
                        source_label_list[source_index],
                        target_probs[source_index],
                        grl_coeff=grl_coeff,
                    )
                    loss_adv = loss_adv + effective_global_weights[source_index] * adv_k
                    domain_accuracy = domain_accuracy + effective_global_weights[source_index] * domain_acc_k
                    if self.lambda_cda > 0.0:
                        loss_cda = loss_cda + effective_global_weights[source_index] * self._conditional_mmd(
                            source_features[source_index],
                            target_features[source_index],
                            source_probs[source_index],
                            target_probs[source_index],
                        )

            loss_ent = (
                self._target_entropy_loss(fused_target_probs)
                if stage >= 1
                else fused_target_probs.sum() * 0.0
            )
            loss_consistency = self._classifier_consistency_loss(
                target_probs, fused_target_probs, class_weights
            ) if stage == 2 else fused_target_probs.sum() * 0.0

            loss_clmmd = fused_target_probs.sum() * 0.0
            if (
                stage == 2
                and self._cur_epoch >= self.clmmd_start_epoch
                and self.lambda_clmmd > 0.0
            ):
                numerator = fused_target_probs.sum() * 0.0
                denominator = fused_target_probs.sum() * 0.0
                target_soft = fused_target_probs.detach()
                for source_index in range(self.num_source):
                    loss_vector, valid_vector = self._classwise_lmmd_per_class(
                        source_features[source_index],
                        target_features[source_index],
                        source_label_list[source_index],
                        target_soft,
                    )
                    weight = class_weights[source_index] * valid_vector
                    numerator = numerator + (weight * loss_vector).sum()
                    denominator = denominator + weight.sum()
                if denominator.detach().item() > 0.0:
                    loss_clmmd = numerator / (denominator + self.eps)

            loss_mca = (
                self._multi_classifier_alignment_loss(
                    target_probs, fused_target_probs, class_weights
                )
                if stage == 2
                else fused_target_probs.sum() * 0.0
            )

            loss_hard_supcon = self._compute_hard_class_supcon(
                shared_sources=shared_sources,
                source_labels=source_label_list,
                shared_target=shared_target,
                pseudo_labels=pseudo_labels,
                valid_pseudo_mask=valid_pseudo_mask,
                pseudo_quality=pseudo_quality,
                stage=stage,
            )
            hard_lambda_now = self._hard_supcon_lambda_now()

            class_adapt = loss_proto + self.lambda_mcc * loss_mcc
            if stage == 0:
                rcpa_core = fused_target_probs.sum() * 0.0
                auxiliary = fused_target_probs.sum() * 0.0
                loss = loss_cls
            elif stage == 1:
                rcpa_core = loss_global
                auxiliary = (
                    self.lambda_adv * loss_adv
                    + self.lambda_ent * loss_ent
                    + self.lambda_cda * loss_cda
                )
                loss = (
                    loss_cls
                    + self.lambda_adapt * adapt_strength * rcpa_core
                    + adapt_strength * auxiliary
                )
            else:
                rcpa_core = (1.0 - rho) * loss_global + rho * class_adapt
                auxiliary = (
                    self.lambda_adv * loss_adv
                    + self.lambda_ent * loss_ent
                    + self.lambda_cda * loss_cda
                    + self.lambda_clmmd * loss_clmmd
                    + self.lambda_mca * loss_mca
                    + self.lambda_consistency * loss_consistency
                )
                loss = (
                    self.cls_stage2_weight * loss_cls
                    + self.lambda_adapt * adapt_strength * rcpa_core
                    + adapt_strength * auxiliary
                )
            loss = loss + hard_lambda_now * loss_hard_supcon

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite stable-hybrid loss: "
                    f"cls={loss_cls.item()} global={loss_global.item()} "
                    f"proto={loss_proto.item()} adv={loss_adv.item()} "
                    f"clmmd={loss_clmmd.item()} mca={loss_mca.item()}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.G.parameters())
                + list(self.Fs.parameters())
                + list(self.Cs.parameters())
                + list(self.Ds.parameters()),
                max_norm=float(getattr(self.args, "rcpa_grad_clip", 5.0)),
            )
            self.optimizer.step()

            epoch_loss["Total"] += loss.detach()
            epoch_loss["Source Classifier"] += loss_cls.detach()
            epoch_loss["RCPA Core"] += rcpa_core.detach()
            epoch_loss["Auxiliary Adaptation"] += auxiliary.detach()
            epoch_loss["Global MMD"] += loss_global.detach()
            epoch_loss["Prototype Alignment"] += loss_proto.detach()
            epoch_loss["MCC"] += loss_mcc.detach()
            epoch_loss["CDAN Domain"] += loss_adv.detach()
            epoch_loss["Target Entropy"] += loss_ent.detach()
            epoch_loss["Conditional MMD"] += loss_cda.detach()
            epoch_loss["CLMMD"] += loss_clmmd.detach()
            epoch_loss["MCA"] += loss_mca.detach()
            epoch_loss["Classifier Consistency"] += loss_consistency.detach()
            epoch_loss["Hard SupCon"] += loss_hard_supcon.detach()
            epoch_loss["Hard SupCon Lambda"] += torch.tensor(hard_lambda_now, device=self.device)
            epoch_loss["Hard SupCon Weighted"] += (hard_lambda_now * loss_hard_supcon).detach()
            epoch_loss["Adapt Strength"] += torch.tensor(adapt_strength, device=self.device)
            epoch_loss["Prototype Ratio rho"] += torch.tensor(rho, device=self.device)
            epoch_loss["Source Readiness"] += torch.tensor(source_ready, device=self.device)
            epoch_loss["Target Confident Coverage"] += torch.tensor(target_coverage, device=self.device)
            epoch_loss["Current Reliable Pseudo Coverage"] += torch.tensor(pseudo_stats["coverage"], device=self.device)
            epoch_loss["Pseudo Agreement Rate"] += torch.tensor(pseudo_stats["agreement_rate"], device=self.device)
            epoch_loss["Pseudo Margin Rate"] += torch.tensor(pseudo_stats["margin_rate"], device=self.device)
            epoch_loss["Pseudo Mean Quality"] += torch.tensor(pseudo_stats["mean_quality"], device=self.device)
            epoch_loss["Valid Target Classes"] += torch.tensor(float(valid_target_classes), device=self.device)
            epoch_loss["Valid Prototype Pairs"] += torch.tensor(float(valid_pairs), device=self.device)
            epoch_acc["Domain Data"] += domain_accuracy.detach().item()

            class_weight_sum += class_weights
            effective_global_sum += effective_global_weights
            raw_global_sum += raw_current
            reliability_score_sum += reliability_score
            mmd_component_sum += mmd_tensor.detach()
            entropy_component_sum += target_entropy_tensor.detach()
            recognition_component_sum += source_recognition_tensor.detach()
            consensus_component_sum += target_consensus_tensor.detach()

        denominator = max(float(self.num_iter), 1.0)
        avg_class_weights = class_weight_sum / denominator
        avg_effective = effective_global_sum / denominator
        avg_raw = raw_global_sum / denominator
        avg_score = reliability_score_sum / denominator
        avg_mmd = mmd_component_sum / denominator
        avg_entropy = entropy_component_sum / denominator
        avg_recognition = recognition_component_sum / denominator
        avg_consensus = consensus_component_sum / denominator

        gate_status = self.source_gate.update_epoch(avg_raw, self._cur_epoch)
        confirmed_index = gate_status["confirmed_index"]
        confirmed_name = (
            self.src[confirmed_index]
            if 0 <= confirmed_index < len(self.src)
            else "none"
        )
        candidate_index = gate_status["candidate_index"]
        candidate_name = (
            self.src[candidate_index]
            if 0 <= candidate_index < len(self.src)
            else "none"
        )

        logging.info(
            "RCPA stages in epoch: warmup=%d global=%d prototype=%d",
            int(stage_counts[0].item()),
            int(stage_counts[1].item()),
            int(stage_counts[2].item()),
        )
        for source_index, source_name in enumerate(self.src):
            logging.info(
                "Source reliability %s (src%d): mmd=%.5f entropy=%.5f "
                "recognition=%.5f consensus=%.5f score=%.5f raw_weight=%.5f "
                "effective_weight=%.5f",
                source_name,
                source_index,
                avg_mmd[source_index].item(),
                avg_entropy[source_index].item(),
                avg_recognition[source_index].item(),
                avg_consensus[source_index].item(),
                avg_score[source_index].item(),
                avg_raw[source_index].item(),
                avg_effective[source_index].item(),
            )
        logging.info(
            "Stable source gate: event=%s candidate=%s(src%d) streak=%d "
            "confirmed=%s(src%d) release_streak=%d gap=%.5f",
            gate_status["event"],
            candidate_name,
            candidate_index,
            gate_status["candidate_streak"],
            confirmed_name,
            confirmed_index,
            gate_status["release_streak"],
            gate_status["gap"],
        )
        for class_index in range(self.num_classes):
            logging.info(
                "RCPA class-%d source weights: %s",
                class_index,
                ", ".join(
                    f"{self.src[k]}(src{k})={avg_class_weights[k, class_index].item():.4f}"
                    for k in range(self.num_source)
                ),
            )
        logging.info(
            "Average valid prototype pairs per iteration: %.2f",
            valid_proto_pairs_sum / denominator,
        )
        if hasattr(self.G, "get_gate"):
            logging.info("BiMamba residual gate: %.6f", self.G.get_gate().item())
        return epoch_acc, epoch_loss
