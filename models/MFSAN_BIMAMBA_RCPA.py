# -*- coding: utf-8 -*-
"""MFSAN-BiMamba-RCPA.

A simplified multi-source unsupervised domain-adaptation model for
cross-condition bearing fault diagnosis.

Core design
-----------
1. MSCNN + small-gated BiMamba shared backbone.
2. One source-specific feature branch and classifier per source domain.
3. EMA source/target class prototypes and a source-class reliability matrix.
4. State-controlled progressive adaptation:
       source warm-up -> global MK-MMD -> class prototype alignment + MCC.
5. The same long-term reliability matrix is used for class-wise prediction
   fusion, while optional sample entropy only modifies evaluation fusion.

Compared with the previous V5 stack
-----------------------------------
CDAN/CDA, standalone CLMMD, target entropy loss, L1 consistency and MCA remain
removed.  Two focused mechanisms are retained because the PU_0 logs provide
direct evidence for them:
1. adaptive top-2 source gating to suppress a negative-transfer source;
2. hard-class supervised contrastive learning for KA04, KA16 and KI17.

    L = L_cls + lambda_adapt * L_adapt + lambda_hc * L_hard_supcon
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

import modules
import modules_bimamba
import utils
from train_utils import TrainerBase
from models.rcpa_components import (
    BalancedClassBatchSampler,
    MinimumClassConfusionLoss,
    RCPAStatisticsMemory,
    SupervisedContrastiveLoss,
    adaptive_top2_source_gate,
    normalized_entropy,
    normalize_source_weights,
)


class Trainer(TrainerBase):
    """Trainer for the RCPA model."""

    def __init__(self, args):
        super().__init__(args)
        if args.train_mode != "multi_source":
            raise ValueError("MFSAN_BIMAMBA_RCPA requires --train_mode multi_source.")

        self.src_labels_flat = sorted(
            set(label for domain_labels in args.label_sets[:-1] for label in domain_labels)
        )
        self.num_classes = len(self.src_labels_flat)
        if self.num_classes < 2:
            raise ValueError("RCPA requires at least two fault classes.")

        self.eps = 1e-8
        self._cur_epoch = 0
        self._current_stage = 0
        self._stage1_start_epoch = None
        self._stage2_start_epoch = None
        self._last_rho = 0.0

        # ------------------------------------------------------------------
        # Shared backbone
        # ------------------------------------------------------------------
        if args.backbone in ["CNN", "MSCNN_BiMamba_Att", "MS_BiMamba_Att", "BIMAMBA"]:
            self.G = modules_bimamba.MSCNNBiMambaAttBackbone(
                in_channel=1,
                stem_channels=int(getattr(args, "bimamba_stem_channels", 64)),
                mamba_dim=int(getattr(args, "bimamba_dim", 64)),
                mamba_depth=int(getattr(args, "bimamba_depth", 2)),
                mamba_d_state=int(getattr(args, "bimamba_d_state", 16)),
                mamba_d_conv=int(getattr(args, "bimamba_d_conv", 4)),
                mamba_expand=int(getattr(args, "bimamba_expand", 2)),
                dropout=float(args.dropout),
                gate_init=float(getattr(args, "bimamba_gate_init", 0.01)),
                gate_max=float(getattr(args, "bimamba_gate_max", 0.03)),
            ).to(self.device)
            actual_backbone = "MSCNN_BiMamba_Att_SmallGate"
        elif args.backbone in ["MSCNN_NO_BIMAMBA", "CNN_NO_BIMAMBA"]:
            self.G = modules.MSCNN(in_channel=1).to(self.device)
            actual_backbone = "MSCNN_No_BiMamba"
        elif args.backbone == "ResNet":
            self.G = modules.ResNet(in_channel=1, layers=[2, 2, 2, 2]).to(self.device)
            actual_backbone = "ResNet"
        else:
            raise ValueError(f"Unknown backbone: {args.backbone}")

        # ------------------------------------------------------------------
        # Source-specific branches
        # ------------------------------------------------------------------
        self.Fs = nn.ModuleList(
            [
                modules.MLP(
                    input_size=self.G.out_dim,
                    dropout=float(args.dropout),
                    num_layer=2,
                    output_layer=False,
                )
                for _ in range(self.num_source)
            ]
        ).to(self.device)
        self.Cs = nn.ModuleList(
            [
                modules.MLP(
                    input_size=self.Fs[k].feature_dim,
                    output_size=self.num_classes,
                    num_layer=1,
                    last=None,
                )
                for k in range(self.num_source)
            ]
        ).to(self.device)
        self.feature_dim = int(self.Fs[0].feature_dim)

        # One global discrepancy metric. No adversarial discriminator is built.
        self.mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
            kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)]
        )

        # ------------------------------------------------------------------
        # RCPA hyperparameters
        # ------------------------------------------------------------------
        self.lambda_adapt = float(getattr(args, "lambda_adapt", 0.50))
        self.lambda_mcc = float(getattr(args, "lambda_mcc", 0.03))
        self.cls_stage2_weight = float(getattr(args, "rcpa_cls_stage2_weight", 1.0))
        self.label_smoothing = float(getattr(args, "rcpa_label_smoothing", 0.05))

        self.global_weight_tau = float(getattr(args, "rcpa_global_weight_tau", 0.50))
        self.global_entropy_weight = float(
            getattr(args, "rcpa_global_entropy_weight", 1.0)
        )
        self.global_recognition_weight = float(
            getattr(args, "rcpa_global_recognition_weight", 0.30)
        )
        self.reliability_tau = float(getattr(args, "rcpa_reliability_tau", 0.70))
        self.distance_weight = float(getattr(args, "rcpa_distance_weight", 1.0))
        self.entropy_weight = float(getattr(args, "rcpa_entropy_weight", 0.25))
        self.recognition_weight = float(getattr(args, "rcpa_recognition_weight", 0.25))
        self.reliability_global_prior = float(
            getattr(args, "rcpa_reliability_global_prior", 0.25)
        )
        self.reliability_smoothing = float(
            getattr(args, "rcpa_reliability_smoothing", 0.15)
        )
        self.reliability_score_clip = float(
            getattr(args, "rcpa_reliability_score_clip", 2.0)
        )
        self.min_source_weight = float(
            getattr(args, "rcpa_min_source_weight", 0.01)
        )

        # Adaptive negative-source suppression.  This does not hard-code PU_3;
        # it suppresses whichever source is consistently ranked third by the
        # measured reliability, while capping the strongest source.
        self.adaptive_source_pruning = bool(
            getattr(args, "rcpa_adaptive_source_pruning", True)
        )
        self.apply_gate_to_global = bool(
            getattr(args, "rcpa_apply_gate_to_global", True)
        )
        self.source_prune_gap = float(
            getattr(args, "rcpa_source_prune_gap", 0.05)
        )
        self.source_bottom_floor = float(
            getattr(args, "rcpa_source_bottom_floor", 0.01)
        )
        self.max_source_weight = float(
            getattr(args, "rcpa_max_source_weight", 0.65)
        )

        # Hard-class boundary enhancement.  Class ids 1/2/8 correspond to
        # KA04/KA16/KI17 in the fixed nine-class PU label mapping.
        self.lambda_hard_supcon = float(
            getattr(args, "rcpa_lambda_hard_supcon", 0.01)
        )
        self.hard_supcon_temperature = float(
            getattr(args, "rcpa_hard_supcon_temperature", 0.20)
        )
        self.hard_supcon_start_epoch = int(
            getattr(args, "rcpa_hard_supcon_start_epoch", 3)
        )
        hard_classes_text = str(
            getattr(args, "rcpa_hard_classes", "1,2,8")
        ).strip()
        self.hard_classes = sorted(
            {
                int(value.strip())
                for value in hard_classes_text.split(",")
                if value.strip()
            }
        )
        self.hard_supcon_use_target = bool(
            getattr(args, "rcpa_hard_supcon_use_target", False)
        )
        self.hard_target_quality_threshold = float(
            getattr(args, "rcpa_hard_target_quality_threshold", 0.90)
        )
        self.hard_target_max_per_class = int(
            getattr(args, "rcpa_hard_target_max_per_class", 12)
        )

        self.conf_threshold = float(getattr(args, "rcpa_conf_threshold", 0.80))
        self.pseudo_margin = float(getattr(args, "rcpa_pseudo_margin", 0.15))
        self.pseudo_min_agreement = int(
            getattr(args, "rcpa_pseudo_min_agreement", min(2, self.num_source))
        )
        self.pseudo_source_confidence = float(
            getattr(args, "rcpa_pseudo_source_confidence", 0.55)
        )
        self.min_source_samples = int(getattr(args, "rcpa_min_source_samples", 2))
        self.min_target_samples = int(getattr(args, "rcpa_min_target_samples", 2))
        self.prototype_quality_temperature = float(
            getattr(args, "rcpa_prototype_quality_temperature", 0.50)
        )
        self.use_prototype_quality = bool(
            getattr(args, "rcpa_use_prototype_quality", True)
        )

        self.source_ready_threshold = float(
            getattr(args, "rcpa_source_ready_threshold", 0.75)
        )
        self.target_ready_threshold = float(
            getattr(args, "rcpa_target_ready_threshold", 0.40)
        )

        # Earliest transition epochs: readiness cannot trigger a stage before
        # these epochs. Force epochs remain latest fallback transition times.
        self.min_stage1_epoch = int(getattr(args, "rcpa_min_stage1_epoch", 4))
        self.force_stage1_epoch = int(getattr(args, "rcpa_force_stage1_epoch", 6))
        self.min_stage2_epoch = int(getattr(args, "rcpa_min_stage2_epoch", 8))
        self.force_stage2_epoch = int(getattr(args, "rcpa_force_stage2_epoch", 12))
        self.min_valid_target_classes = int(
            getattr(args, "rcpa_min_valid_target_classes", max(2, self.num_classes - 2))
        )
        self.force_stage2_min_valid_classes = int(
            getattr(
                args,
                "rcpa_force_stage2_min_valid_classes",
                max(2, (self.num_classes + 1) // 2),
            )
        )
        self.adapt_ramp_epochs = max(
            1, int(getattr(args, "rcpa_adapt_ramp_epochs", 3))
        )
        self.prototype_ramp_epochs = max(
            1, int(getattr(args, "rcpa_prototype_ramp_epochs", 8))
        )
        self.rho_max = float(getattr(args, "rcpa_rho_max", 0.60))

        if self.min_stage1_epoch < 1:
            raise ValueError("rcpa_min_stage1_epoch must be >= 1.")
        if self.min_stage2_epoch < self.min_stage1_epoch:
            raise ValueError("rcpa_min_stage2_epoch must be >= rcpa_min_stage1_epoch.")
        if self.force_stage1_epoch < self.min_stage1_epoch:
            raise ValueError("rcpa_force_stage1_epoch must be >= rcpa_min_stage1_epoch.")
        if self.force_stage2_epoch < self.min_stage2_epoch:
            raise ValueError("rcpa_force_stage2_epoch must be >= rcpa_min_stage2_epoch.")
        if not 1 <= self.min_valid_target_classes <= self.num_classes:
            raise ValueError(
                "rcpa_min_valid_target_classes must be in [1, num_classes]."
            )
        if not 1 <= self.force_stage2_min_valid_classes <= self.num_classes:
            raise ValueError(
                "rcpa_force_stage2_min_valid_classes must be in [1, num_classes]."
            )
        if not 0.0 <= self.rho_max < 1.0:
            raise ValueError("rcpa_rho_max must be in [0, 1).")
        if not 0 <= self.pseudo_min_agreement <= self.num_source:
            raise ValueError(
                "rcpa_pseudo_min_agreement must be in [0, num_sources]."
            )
        if not 0.0 <= self.pseudo_margin < 1.0:
            raise ValueError("rcpa_pseudo_margin must be in [0, 1).")
        if not 0.0 <= self.pseudo_source_confidence <= 1.0:
            raise ValueError("rcpa_pseudo_source_confidence must be in [0, 1].")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("rcpa_label_smoothing must be in [0, 1).")
        if not 0.0 <= self.source_bottom_floor < 1.0 / max(self.num_source, 1):
            raise ValueError(
                "rcpa_source_bottom_floor must be in [0, 1/num_sources)."
            )
        if not 1.0 / max(self.num_source, 1) <= self.max_source_weight < 1.0:
            raise ValueError(
                "rcpa_max_source_weight must be in [1/num_sources, 1)."
            )
        if self.source_prune_gap < 0.0:
            raise ValueError("rcpa_source_prune_gap must be non-negative.")
        if self.lambda_hard_supcon < 0.0:
            raise ValueError("rcpa_lambda_hard_supcon must be non-negative.")
        if self.hard_supcon_start_epoch < 1:
            raise ValueError("rcpa_hard_supcon_start_epoch must be >= 1.")
        if any(c < 0 or c >= self.num_classes for c in self.hard_classes):
            raise ValueError(
                f"rcpa_hard_classes must be valid class ids in [0, {self.num_classes - 1}]."
            )

        self.dynamic_eval_fusion = bool(
            getattr(args, "rcpa_dynamic_eval_fusion", False)
        )
        self.eval_entropy_eta = float(getattr(args, "rcpa_eval_entropy_eta", 0.30))
        self.eval_weight_tau = float(getattr(args, "rcpa_eval_weight_tau", 2.0))
        self.eval_each_epoch = bool(getattr(args, "rcpa_eval_each_epoch", False))
        self.track_best_target_test = bool(
            getattr(args, "rcpa_track_best_target_test", True)
        )
        self.select_best_by_target_test = bool(
            getattr(args, "rcpa_select_best_by_target_test", False)
        )
        self.balanced_source_batches = bool(
            getattr(args, "rcpa_balanced_source_batches", True)
        )

        prototype_momentum = float(getattr(args, "rcpa_prototype_momentum", 0.90))
        statistic_momentum = float(getattr(args, "rcpa_statistic_momentum", 0.90))
        weight_momentum = float(getattr(args, "rcpa_weight_momentum", 0.90))

        self.memory = RCPAStatisticsMemory(
            num_sources=self.num_source,
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            prototype_momentum=prototype_momentum,
            statistic_momentum=statistic_momentum,
            weight_momentum=weight_momentum,
            eps=self.eps,
        ).to(self.device)
        self.mcc_loss = MinimumClassConfusionLoss(
            temperature=float(getattr(args, "rcpa_mcc_temperature", 2.0)),
            eps=self.eps,
        ).to(self.device)
        self.hard_supcon_loss = SupervisedContrastiveLoss(
            temperature=self.hard_supcon_temperature,
            eps=self.eps,
        ).to(self.device)

        # Data and optimizer are initialized after all model modules exist.
        self._init_data()
        self.src = args.source_name
        self.optimizer = self._get_optimizer([self.G, self.Fs, self.Cs])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        # One epoch follows the longest source loader. Shorter loaders and the
        # target loader are cycled by TrainerBase._get_next_batch().
        self.num_iter = max(len(self.dataloaders[s]) for s in self.src)

        logging.info("Using model: MFSAN_BIMAMBA_RCPA")
        logging.info("Requested backbone: %s", args.backbone)
        logging.info("Actual backbone: %s", actual_backbone)
        logging.info("Backbone output dim: %d", self.G.out_dim)
        logging.info("Source-specific feature dim: %d", self.feature_dim)
        logging.info(
            "Top-level objective: L_cls + lambda_adapt * L_adapt "
            "+ lambda_hard_supcon * L_hard_supcon"
        )
        logging.info("lambda_adapt=%.6f, lambda_mcc=%.6f", self.lambda_adapt, self.lambda_mcc)
        logging.info(
            "Global source reliability: MMD + %.3f*target_entropy + "
            "%.3f*source_recognition, tau=%.3f",
            self.global_entropy_weight,
            self.global_recognition_weight,
            self.global_weight_tau,
        )
        logging.info(
            "RCPA class reliability weights: distance=%.3f entropy=%.3f "
            "recognition=%.3f tau=%.3f",
            self.distance_weight,
            self.entropy_weight,
            self.recognition_weight,
            self.reliability_tau,
        )
        logging.info(
            "Reliability stabilizers: global_prior=%.3f uniform_smoothing=%.3f "
            "score_clip=%.3f min_source_weight=%.3f",
            self.reliability_global_prior,
            self.reliability_smoothing,
            self.reliability_score_clip,
            self.min_source_weight,
        )
        logging.info(
            "Adaptive source pruning: enabled=%s global_gate=%s prune_gap=%.3f "
            "bottom_floor=%.3f max_source_weight=%.3f",
            self.adaptive_source_pruning,
            self.apply_gate_to_global,
            self.source_prune_gap,
            self.source_bottom_floor,
            self.max_source_weight,
        )
        logging.info(
            "Hard-class SupCon: lambda=%.4f temperature=%.3f start_epoch=%d "
            "classes=%s use_target=%s target_quality>=%.3f",
            self.lambda_hard_supcon,
            self.hard_supcon_temperature,
            self.hard_supcon_start_epoch,
            self.hard_classes,
            self.hard_supcon_use_target,
            self.hard_target_quality_threshold,
        )
        logging.info(
            "Reliable pseudo labels: confidence>=%.3f margin>=%.3f "
            "agreement>=%d source_confidence>=%.3f",
            self.conf_threshold,
            self.pseudo_margin,
            self.pseudo_min_agreement,
            self.pseudo_source_confidence,
        )
        logging.info(
            "Stage controller: source_ready=%.3f target_ready=%.3f "
            "min_stage1=%d force_stage1=%d min_stage2=%d force_stage2=%d "
            "min_valid_target_classes=%d force_min_valid_classes=%d",
            self.source_ready_threshold,
            self.target_ready_threshold,
            self.min_stage1_epoch,
            self.force_stage1_epoch,
            self.min_stage2_epoch,
            self.force_stage2_epoch,
            self.min_valid_target_classes,
            self.force_stage2_min_valid_classes,
        )
        logging.info(
            "Stage-2 mixing: rho_max=%.3f, prototype_ramp_epochs=%d; "
            "global alignment is always retained at >= %.3f",
            self.rho_max,
            self.prototype_ramp_epochs,
            1.0 - self.rho_max,
        )
        logging.info(
            "Target-test evaluation policy: eval_each_epoch=%s, "
            "track_best_report=%s, select_best_checkpoint=%s",
            self.eval_each_epoch,
            self.track_best_target_test,
            self.select_best_by_target_test,
        )
        if self.select_best_by_target_test:
            logging.warning(
                "rcpa_select_best_by_target_test=True: target-test labels will be "
                "used for checkpoint selection. This is convenient for legacy/debug "
                "comparison but is not recommended for final thesis experiments."
            )
        if hasattr(self.G, "uses_real_mamba"):
            logging.info(
                "BiMamba implementation: %s",
                "mamba_ssm" if self.G.uses_real_mamba else "lite_pytorch_fallback",
            )
        if hasattr(self.G, "get_gate"):
            logging.info("Initial BiMamba residual gate: %.6f", self.G.get_gate().item())

    # ==================================================================
    # Data
    # ==================================================================
    def _init_data(self):
        """Create source/target datasets and optional balanced source loaders."""
        import importlib

        args = self.args
        self.datasets = {}
        for i, source in enumerate(args.source_name):
            dataset_name, condition, _ = utils.get_info_from_name(source)
            if condition is not None:
                Dataset = importlib.import_module("data_loader.conditional_load").dataset
                self.datasets[source] = Dataset(
                    args, dataset_name, i, condition=condition
                ).data_preprare(is_src=True)
            else:
                Dataset = importlib.import_module("data_loader.load").dataset
                self.datasets[source] = Dataset(args, dataset_name, i).data_preprare(is_src=True)

        dataset_name, condition, _ = utils.get_info_from_name(args.target)
        if condition is not None:
            Dataset = importlib.import_module("data_loader.conditional_load").dataset
            self.datasets["train"], self.datasets["val"] = Dataset(
                args, dataset_name, -1, condition=condition
            ).data_preprare(is_src=False)
        else:
            Dataset = importlib.import_module("data_loader.load").dataset
            self.datasets["train"], self.datasets["val"] = Dataset(
                args, dataset_name, -1
            ).data_preprare(is_src=False)

        self.dataset_keys = args.source_name + ["train", "val"]
        self.source_batch_samplers: Dict[str, BalancedClassBatchSampler] = {}
        self.dataloaders = {}
        pin_memory = self.device.type == "cuda"

        for source_idx, source in enumerate(args.source_name):
            dataset = self.datasets[source]
            logging.info("Source set %s number of samples: %d", source, len(dataset))
            dataset.summary()

            if self.balanced_source_batches and hasattr(dataset, "actual_labels"):
                sampler = BalancedClassBatchSampler(
                    labels=dataset.actual_labels,
                    batch_size=args.batch_size,
                    drop_last=True,
                    seed=int(args.random_state) + source_idx * 1009,
                )
                self.source_batch_samplers[source] = sampler
                self.dataloaders[source] = DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    num_workers=args.num_workers,
                    pin_memory=pin_memory,
                )
            else:
                self.dataloaders[source] = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    drop_last=True,
                    pin_memory=pin_memory,
                )

        logging.info("Target train set number of samples: %d", len(self.datasets["train"]))
        self.datasets["train"].summary()
        logging.info("Target test set number of samples: %d", len(self.datasets["val"]))
        self.datasets["val"].summary()

        self.dataloaders["train"] = DataLoader(
            self.datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=pin_memory,
        )
        self.dataloaders["val"] = DataLoader(
            self.datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=pin_memory,
        )
        self.iters = {key: iter(loader) for key, loader in self.dataloaders.items()}

    # ==================================================================
    # State and checkpoint
    # ==================================================================
    def _set_to_train(self):
        self.G.train()
        self.Fs.train()
        self.Cs.train()
        self.memory.train()

    def _set_to_eval(self):
        self.G.eval()
        self.Fs.eval()
        self.Cs.eval()
        self.memory.eval()

    def _checkpoint_dict(self):
        return {
            "model_name": "MFSAN_BIMAMBA_RCPA",
            "G": self.G.state_dict(),
            "Fs": self.Fs.state_dict(),
            "Cs": self.Cs.state_dict(),
            "memory": self.memory.state_dict(),
            "current_stage": self._current_stage,
            "stage1_start_epoch": self._stage1_start_epoch,
            "stage2_start_epoch": self._stage2_start_epoch,
            "last_rho": self._last_rho,
            "num_sources": self.num_source,
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "lambda_adapt": self.lambda_adapt,
            "lambda_mcc": self.lambda_mcc,
        }

    def save_model(self):
        path = self.args.save_path + ".pth"
        torch.save(self._checkpoint_dict(), path)
        logging.info("Model saved to %s", path)

    def save_best_model(self):
        """Save the current parameters as the best checkpoint."""
        path = self.args.save_path + "_best.pth"
        torch.save(self._checkpoint_dict(), path)
        logging.info("Best model saved to %s", path)

    def load_model(self):
        logging.info("Loading model from %s", self.args.load_path)
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.Fs.load_state_dict(ckpt["Fs"])
        self.Cs.load_state_dict(ckpt["Cs"])
        if "memory" in ckpt:
            self.memory.load_state_dict(ckpt["memory"], strict=True)
        self._current_stage = int(ckpt.get("current_stage", 2))
        self._stage1_start_epoch = ckpt.get("stage1_start_epoch", None)
        self._stage2_start_epoch = ckpt.get("stage2_start_epoch", None)
        self._last_rho = float(ckpt.get("last_rho", 0.0))

    # ==================================================================
    # Reliability and fusion
    # ==================================================================
    def _adaptation_strength(self, stage: int) -> float:
        """Ramp adaptation from the epoch in which stage 1 actually starts."""
        if stage <= 0:
            return 0.0
        if self._stage1_start_epoch is None:
            self._stage1_start_epoch = self._cur_epoch
        progress = (
            self._cur_epoch - int(self._stage1_start_epoch) + 1
        ) / float(self.adapt_ramp_epochs)
        return min(1.0, max(0.0, progress))

    def _valid_target_class_count(self) -> int:
        """Count target classes with valid prototypes in every source branch."""
        valid_by_class = self.memory.target_proto_valid.all(dim=0)
        return int(valid_by_class.sum().item())

    def _resolve_stage(self) -> Tuple[int, float, float, float, int]:
        """
        Monotonic dynamic controller with earliest and latest transition epochs.

        Stage 0 -> 1:
          epoch >= min_stage1_epoch and
          (source readiness is sufficient or epoch reaches force_stage1_epoch).

        Stage 1 -> 2:
          epoch >= min_stage2_epoch and
          ((target coverage and valid-class coverage are sufficient) or
           epoch reaches force_stage2_epoch).
        """
        source_readiness = self.memory.source_readiness()
        target_coverage = float(self.memory.target_coverage.item())
        valid_target_classes = self._valid_target_class_count()
        epoch = int(self._cur_epoch)

        if self._current_stage == 0 and epoch >= self.min_stage1_epoch:
            ready = source_readiness >= self.source_ready_threshold
            forced = epoch >= self.force_stage1_epoch
            if ready or forced:
                self._current_stage = 1
                if self._stage1_start_epoch is None:
                    self._stage1_start_epoch = epoch
                logging.info(
                    "RCPA stage transition: warmup -> global at epoch %d "
                    "(source_ready=%.4f, ready=%s, forced=%s)",
                    epoch, source_readiness, ready, forced,
                )

        if self._current_stage == 1 and epoch >= self.min_stage2_epoch:
            coverage_ready = target_coverage >= self.target_ready_threshold
            classes_ready = valid_target_classes >= self.min_valid_target_classes
            ready = coverage_ready and classes_ready
            forced = (
                epoch >= self.force_stage2_epoch
                and valid_target_classes >= self.force_stage2_min_valid_classes
            )
            if ready or forced:
                self._current_stage = 2
                if self._stage2_start_epoch is None:
                    self._stage2_start_epoch = epoch
                logging.info(
                    "RCPA stage transition: global -> prototype at epoch %d "
                    "(target_coverage=%.4f, valid_target_classes=%d/%d, "
                    "ready=%s, forced=%s)",
                    epoch, target_coverage, valid_target_classes, self.num_classes,
                    ready, forced,
                )

        if self._current_stage < 2:
            rho = 0.0
        else:
            start = int(self._stage2_start_epoch or epoch)
            rho_raw = min(
                1.0,
                (epoch - start + 1) / float(self.prototype_ramp_epochs),
            )
            rho = self.rho_max * rho_raw

        self._last_rho = float(rho)

        return (
            self._current_stage,
            rho,
            source_readiness,
            target_coverage,
            valid_target_classes,
        )

    def _guided_class_weights(self, rho: float) -> torch.Tensor:
        global_weights = self.memory.global_source_weights.view(-1, 1).repeat(1, self.num_classes)
        class_weights = self.memory.class_source_weights
        guided = (1.0 - float(rho)) * global_weights + float(rho) * class_weights
        guided = normalize_source_weights(guided, self.eps)
        guided = adaptive_top2_source_gate(
            guided,
            enabled=getattr(self, "adaptive_source_pruning", False),
            prune_gap=getattr(self, "source_prune_gap", 0.05),
            bottom_floor=getattr(self, "source_bottom_floor", 0.01),
            max_source_weight=getattr(self, "max_source_weight", 0.65),
            eps=self.eps,
        )
        return guided.detach()

    def _build_reliable_pseudo_labels(
        self,
        probs_by_source: Sequence[torch.Tensor],
        fused_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Build pseudo labels using confidence, margin and source consensus."""
        confidence, pseudo = fused_probs.detach().max(dim=1)
        top2 = fused_probs.detach().topk(k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]

        stacked = torch.stack([p.detach() for p in probs_by_source], dim=0)
        source_confidence, source_prediction = stacked.max(dim=2)
        pseudo_expanded = pseudo.unsqueeze(0).expand_as(source_prediction)
        agreeing = (
            (source_prediction == pseudo_expanded)
            & (source_confidence >= self.pseudo_source_confidence)
        )
        agreement_count = agreeing.sum(dim=0)

        valid = (
            (confidence >= self.conf_threshold)
            & (margin >= self.pseudo_margin)
            & (agreement_count >= self.pseudo_min_agreement)
        )

        agreement_ratio = agreement_count.float() / max(float(self.num_source), 1.0)
        quality = confidence * margin.clamp(min=self.eps) * agreement_ratio.clamp(min=self.eps)
        quality = quality.clamp(min=self.eps)

        stats = {
            "coverage": float(valid.float().mean().item()),
            "agreement_rate": float(
                (agreement_count >= self.pseudo_min_agreement).float().mean().item()
            ),
            "margin_rate": float((margin >= self.pseudo_margin).float().mean().item()),
            "mean_quality": float(quality[valid].mean().item()) if bool(valid.any()) else 0.0,
        }
        return pseudo, valid, quality, stats

    def _classwise_fusion(
        self,
        probs_by_source: Sequence[torch.Tensor],
        class_weights: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.stack(probs_by_source, dim=0)  # [K, B, C]
        fused = (stacked * class_weights[:, None, :]).sum(dim=0)
        return fused / (fused.sum(dim=1, keepdim=True) + self.eps)

    def _eval_fusion(
        self, probs_by_source: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Training and evaluation must use the same guided mixture.  In stage 1
        # this is purely global; in stage 2 it is capped by rho_max, so the test
        # path never silently switches to a pure and potentially collapsed
        # class-reliability matrix.
        base = self._guided_class_weights(self._last_rho).clamp(min=self.eps)
        stacked = torch.stack(probs_by_source, dim=0)  # [K, B, C]

        if not self.dynamic_eval_fusion:
            fused = (stacked * base[:, None, :]).sum(dim=0)
            fused = fused / (fused.sum(dim=1, keepdim=True) + self.eps)
            return fused, base

        entropy = torch.stack([normalized_entropy(p, self.eps) for p in probs_by_source], dim=0)
        sample_factor = torch.exp(-self.eval_entropy_eta * entropy).unsqueeze(-1)
        dynamic = base[:, None, :] * sample_factor
        dynamic = torch.softmax(
            torch.log(dynamic.clamp(min=self.eps)) / max(self.eval_weight_tau, self.eps),
            dim=0,
        )
        dynamic = adaptive_top2_source_gate(
            dynamic,
            enabled=getattr(self, "adaptive_source_pruning", False),
            prune_gap=getattr(self, "source_prune_gap", 0.05),
            bottom_floor=getattr(self, "source_bottom_floor", 0.01),
            max_source_weight=getattr(self, "max_source_weight", 0.65),
            eps=self.eps,
        )
        fused = (stacked * dynamic).sum(dim=0)
        fused = fused / (fused.sum(dim=1, keepdim=True) + self.eps)
        return fused, dynamic

    # ==================================================================
    # Losses
    # ==================================================================
    def _compute_hard_class_supcon(
        self,
        shared_sources: Sequence[torch.Tensor],
        source_labels: Sequence[torch.Tensor],
        shared_target: torch.Tensor,
        pseudo_labels: torch.Tensor,
        valid_pseudo_mask: torch.Tensor,
        pseudo_quality: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        """Enhance the KA04/KA16/KI17 decision boundaries in shared space."""
        zero = shared_target.sum() * 0.0
        if (
            self.lambda_hard_supcon <= 0.0
            or self._cur_epoch < self.hard_supcon_start_epoch
            or not self.hard_classes
        ):
            return zero

        hard_ids = torch.tensor(
            self.hard_classes,
            device=self.device,
            dtype=source_labels[0].dtype,
        )
        selected_features: List[torch.Tensor] = []
        selected_labels: List[torch.Tensor] = []

        for features, labels in zip(shared_sources, source_labels):
            mask = torch.isin(labels, hard_ids)
            if bool(mask.any()):
                selected_features.append(features[mask])
                selected_labels.append(labels[mask])

        # Optional high-quality target positives.  Source-only is the default
        # because it was the stable configuration in the original high-PU_0
        # model.  The switch is retained for controlled follow-up experiments.
        if self.hard_supcon_use_target and stage >= 2:
            target_mask = (
                valid_pseudo_mask
                & torch.isin(pseudo_labels, hard_ids)
                & (pseudo_quality >= self.hard_target_quality_threshold)
            )
            for class_id in self.hard_classes:
                indices = torch.where(target_mask & (pseudo_labels == class_id))[0]
                if indices.numel() == 0:
                    continue
                keep = min(indices.numel(), self.hard_target_max_per_class)
                if indices.numel() > keep:
                    local_quality = pseudo_quality[indices]
                    top_local = torch.topk(local_quality, k=keep).indices
                    indices = indices[top_local]
                selected_features.append(shared_target[indices])
                selected_labels.append(pseudo_labels[indices])

        if not selected_features:
            return zero
        features = torch.cat(selected_features, dim=0)
        labels = torch.cat(selected_labels, dim=0)
        if features.size(0) <= 1:
            return zero
        return self.hard_supcon_loss(features, labels)

    def _prototype_alignment_loss(
        self,
        source_features: Sequence[torch.Tensor],
        target_features: Sequence[torch.Tensor],
        source_labels: Sequence[torch.Tensor],
        pseudo_labels: torch.Tensor,
        valid_pseudo_mask: torch.Tensor,
        pseudo_quality: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        numerator = torch.tensor(0.0, device=self.device)
        denominator = torch.tensor(0.0, device=self.device)
        valid_pairs = 0

        for k in range(self.num_source):
            for c in range(self.num_classes):
                source_mask = source_labels[k] == c
                target_mask = valid_pseudo_mask & (pseudo_labels == c)

                source_current_valid = int(source_mask.sum().item()) >= self.min_source_samples
                target_current_valid = int(target_mask.sum().item()) >= self.min_target_samples

                source_proto = None
                target_proto = None
                source_has_grad = False
                target_has_grad = False

                if source_current_valid:
                    source_proto = source_features[k][source_mask].mean(dim=0)
                    source_has_grad = True
                elif bool(self.memory.source_proto_valid[k, c]):
                    source_proto = self.memory.source_prototypes[k, c].detach()

                if target_current_valid:
                    selected = target_features[k][target_mask]
                    quality = pseudo_quality[target_mask].detach().clamp(min=self.eps)
                    if self.use_prototype_quality and bool(self.memory.target_proto_valid[k, c]):
                        mem_proto = self.memory.target_prototypes[k, c].view(1, -1)
                        distance = (
                            1.0
                            - F.cosine_similarity(selected.detach(), mem_proto, dim=1)
                        ).clamp(min=0.0)
                        quality = quality * torch.exp(
                            -distance / max(self.prototype_quality_temperature, self.eps)
                        )
                    quality = quality / (quality.sum() + self.eps)
                    target_proto = (selected * quality.unsqueeze(1)).sum(dim=0)
                    target_has_grad = True
                elif bool(self.memory.target_proto_valid[k, c]):
                    target_proto = self.memory.target_prototypes[k, c].detach()

                if source_proto is None or target_proto is None:
                    continue
                if not (source_has_grad or target_has_grad):
                    continue

                distance = 1.0 - F.cosine_similarity(
                    source_proto.view(1, -1), target_proto.view(1, -1), dim=1
                )[0]
                weight = class_weights[k, c]
                numerator = numerator + weight * distance
                denominator = denominator + weight
                valid_pairs += 1

        if valid_pairs == 0:
            return torch.tensor(0.0, device=self.device), 0
        return numerator / (denominator + self.eps), valid_pairs

    # ==================================================================
    # Train and evaluation
    # ==================================================================
    def train(self):
        """Train RCPA and optionally report/track target-test accuracy.

        Recommended thesis mode
        -----------------------
        ``rcpa_eval_each_epoch=False`` and
        ``rcpa_select_best_by_target_test=False``.
        The target test set is evaluated only once after fixed-epoch training.

        Debug/legacy comparison mode
        ----------------------------
        Set ``rcpa_eval_each_epoch=True`` to print target-test metrics after
        every epoch.  ``rcpa_track_best_target_test=True`` records the best
        reported value in the log.  A checkpoint is selected by those labels
        only when ``rcpa_select_best_by_target_test=True`` is explicitly set.
        """
        best_report_acc = float("-inf")
        best_report_epoch = 0
        last_test_acc = None

        for epoch in range(1, self.args.max_epoch + 1):
            self._cur_epoch = epoch
            for sampler in self.source_batch_samplers.values():
                sampler.set_epoch(epoch)

            logging.info("-----Epoch %d/%d-----", epoch, self.args.max_epoch)
            if self.lr_scheduler is not None:
                logging.info("current lr: %s", self.lr_scheduler.get_last_lr())

            self._set_to_train()
            epoch_acc = defaultdict(float)
            epoch_loss = defaultdict(float)
            epoch_acc, epoch_loss = self._train_one_epoch(epoch_acc, epoch_loss)
            self._log_epoch_info(epoch_loss, epoch_acc, self.num_iter)

            if self.eval_each_epoch:
                if not self.select_best_by_target_test:
                    logging.warning(
                        "Target-test metrics are being reported at every epoch. "
                        "They are not used for checkpoint selection, but repeated "
                        "inspection can still influence manual tuning."
                    )

                new_acc = float(self.test())
                last_test_acc = new_acc
                logging.info(
                    "Epoch %d target-test-acc %.4f",
                    epoch,
                    new_acc,
                )

                if self.track_best_target_test and new_acc >= best_report_acc:
                    best_report_acc = new_acc
                    best_report_epoch = epoch

                    if self.select_best_by_target_test:
                        if getattr(self.args, "save", False) and getattr(
                            self.args, "save_best", True
                        ):
                            self.save_best_model()
                        logging.info(
                            "Best model updated at epoch %d, target-test-acc %.4f",
                            best_report_epoch,
                            best_report_acc,
                        )
                    else:
                        logging.info(
                            "Best reported target-test result updated at epoch %d, "
                            "target-test-acc %.4f (reporting only; checkpoint unchanged)",
                            best_report_epoch,
                            best_report_acc,
                        )

                if self.track_best_target_test and best_report_epoch > 0:
                    if self.select_best_by_target_test:
                        logging.info(
                            "The best model epoch %d, target-test-acc %.4f",
                            best_report_epoch,
                            best_report_acc,
                        )
                    else:
                        logging.info(
                            "The best reported target-test epoch %d, "
                            "target-test-acc %.4f",
                            best_report_epoch,
                            best_report_acc,
                        )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        if self.eval_each_epoch:
            logging.info(
                "Training finished. The final epoch has already been evaluated "
                "on the target test set."
            )
            final_acc = last_test_acc
        else:
            logging.info(
                "Training finished. Running one final target-test evaluation."
            )
            final_acc = float(self.test())
            logging.info("Final target-test-acc %.4f", final_acc)

        if self.track_best_target_test and self.eval_each_epoch and best_report_epoch > 0:
            label = "best model" if self.select_best_by_target_test else "best reported result"
            logging.info(
                "Training summary: %s at epoch %d, target-test-acc %.4f; "
                "final-epoch target-test-acc %.4f",
                label,
                best_report_epoch,
                best_report_acc,
                float(final_acc),
            )

        return float(final_acc)

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        stage_counts = torch.zeros(3, device=self.device)
        class_weight_sum = torch.zeros(
            self.num_source, self.num_classes, device=self.device
        )
        global_weight_sum = torch.zeros(self.num_source, device=self.device)
        valid_proto_pairs_sum = 0.0

        for _ in tqdm(range(self.num_iter), ascii=True):
            target_data, _ = self._get_next_batch("train")
            source_data_list: List[torch.Tensor] = []
            source_label_list: List[torch.Tensor] = []

            for k, source in enumerate(self.src):
                source_data, actual_labels = self._get_next_batch(source, return_actual=True)
                train_labels = self._get_train_label(
                    actual_labels, label_set=self.src_labels_flat
                )
                source_data_list.append(source_data)
                source_label_list.append(train_labels)

            self.optimizer.zero_grad()

            all_data = torch.cat(source_data_list + [target_data], dim=0)
            all_shared = self.G(all_data)
            split_sizes = [x.size(0) for x in source_data_list] + [target_data.size(0)]
            split_features = torch.split(all_shared, split_sizes, dim=0)
            shared_sources = list(split_features[:-1])
            shared_target = split_features[-1]

            source_features: List[torch.Tensor] = []
            target_features: List[torch.Tensor] = []
            source_logits: List[torch.Tensor] = []
            target_logits: List[torch.Tensor] = []
            source_probs: List[torch.Tensor] = []
            target_probs: List[torch.Tensor] = []
            mmd_losses: List[torch.Tensor] = []
            cls_losses: List[torch.Tensor] = []

            for k in range(self.num_source):
                f_s = self.Fs[k](shared_sources[k])
                f_t = self.Fs[k](shared_target)
                y_s = self.Cs[k](f_s)
                y_t = self.Cs[k](f_t)
                p_s = F.softmax(y_s, dim=1)
                p_t = F.softmax(y_t, dim=1)

                source_features.append(f_s)
                target_features.append(f_t)
                source_logits.append(y_s)
                target_logits.append(y_t)
                source_probs.append(p_s)
                target_probs.append(p_t)
                cls_losses.append(
                    F.cross_entropy(
                        y_s,
                        source_label_list[k],
                        label_smoothing=self.label_smoothing,
                    )
                )
                mmd_losses.append(self.mkmmd(f_s, f_t))

                epoch_acc["Source Data"] += self._get_accuracy(
                    y_s, source_label_list[k]
                ) / float(self.num_source)

            loss_cls = torch.stack(cls_losses).mean()
            mmd_tensor = torch.stack(mmd_losses)
            target_entropy_tensor = torch.stack(
                [normalized_entropy(p, self.eps).mean() for p in target_probs]
            )
            source_recognition_tensor = torch.stack(
                [
                    p.gather(1, labels.view(-1, 1)).mean()
                    for p, labels in zip(source_probs, source_label_list)
                ]
            )

            # Update stable global source prior using MMD, target uncertainty
            # and source recognition ability, followed by adaptive top-2 gating.
            global_weights = self.memory.update_global_weights(
                mmd_tensor,
                tau=self.global_weight_tau,
                target_entropies=target_entropy_tensor,
                source_recognition=source_recognition_tensor,
                entropy_weight=self.global_entropy_weight,
                recognition_weight=self.global_recognition_weight,
                adaptive_pruning=(
                    self.adaptive_source_pruning and self.apply_gate_to_global
                ),
                prune_gap=getattr(self, "source_prune_gap", 0.05),
                bottom_floor=getattr(self, "source_bottom_floor", 0.01),
                max_source_weight=getattr(self, "max_source_weight", 0.65),
            )

            # Use the previous stable matrix for pseudo-label fusion, then update
            # memory and refresh reliability.  All memory updates are detached.
            stage_before, rho_before, _, _, _ = self._resolve_stage()
            prior_weights = self._guided_class_weights(rho_before)
            prior_fused = self._classwise_fusion(target_probs, prior_weights)
            prior_pseudo, prior_mask, prior_quality, prior_stats = (
                self._build_reliable_pseudo_labels(target_probs, prior_fused)
            )

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
            self.memory.refresh_class_reliability(
                distance_weight=self.distance_weight,
                entropy_weight=self.entropy_weight,
                recognition_weight=self.recognition_weight,
                tau=self.reliability_tau,
                global_prior_mix=self.reliability_global_prior,
                uniform_smoothing=self.reliability_smoothing,
                score_clip=self.reliability_score_clip,
                min_source_weight=self.min_source_weight,
                adaptive_pruning=self.adaptive_source_pruning,
                prune_gap=getattr(self, "source_prune_gap", 0.05),
                bottom_floor=getattr(self, "source_bottom_floor", 0.01),
                max_source_weight=getattr(self, "max_source_weight", 0.65),
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

            loss_global = torch.sum(global_weights * mmd_tensor)
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
                loss_proto = torch.tensor(0.0, device=self.device)
                loss_mcc = torch.tensor(0.0, device=self.device)
                valid_pairs = 0
            valid_proto_pairs_sum += float(valid_pairs)

            loss_hard_supcon = self._compute_hard_class_supcon(
                shared_sources=shared_sources,
                source_labels=source_label_list,
                shared_target=shared_target,
                pseudo_labels=pseudo_labels,
                valid_pseudo_mask=valid_pseudo_mask,
                pseudo_quality=pseudo_quality,
                stage=stage,
            )

            adapt_strength = self._adaptation_strength(stage)
            if stage == 0:
                loss_adapt = torch.tensor(0.0, device=self.device)
                loss = loss_cls
            elif stage == 1:
                loss_adapt = loss_global
                loss = loss_cls + self.lambda_adapt * adapt_strength * loss_adapt
            else:
                class_adapt = loss_proto + self.lambda_mcc * loss_mcc
                loss_adapt = (1.0 - rho) * loss_global + rho * class_adapt
                loss = (
                    self.cls_stage2_weight * loss_cls
                    + self.lambda_adapt * adapt_strength * loss_adapt
                )

            loss = loss + self.lambda_hard_supcon * loss_hard_supcon

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite RCPA loss detected: "
                    f"cls={loss_cls.item()}, global={loss_global.item()}, "
                    f"proto={loss_proto.item()}, mcc={loss_mcc.item()}, "
                    f"hard_supcon={loss_hard_supcon.item()}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.G.parameters())
                + list(self.Fs.parameters())
                + list(self.Cs.parameters()),
                max_norm=float(getattr(self.args, "rcpa_grad_clip", 5.0)),
            )
            self.optimizer.step()

            epoch_loss["Total"] += loss.detach()
            epoch_loss["Source Classifier"] += loss_cls.detach()
            epoch_loss["Adaptation"] += loss_adapt.detach()
            epoch_loss["Global MMD"] += loss_global.detach()
            epoch_loss["Prototype Alignment"] += loss_proto.detach()
            epoch_loss["MCC"] += loss_mcc.detach()
            epoch_loss["Hard SupCon"] += loss_hard_supcon.detach()
            epoch_loss["Hard SupCon Weighted"] += (
                self.lambda_hard_supcon * loss_hard_supcon
            ).detach()
            epoch_loss["Adapt Strength"] += torch.tensor(adapt_strength, device=self.device)
            epoch_loss["Prototype Ratio rho"] += torch.tensor(rho, device=self.device)
            epoch_loss["Source Readiness"] += torch.tensor(source_ready, device=self.device)
            epoch_loss["Target Confident Coverage"] += torch.tensor(target_coverage, device=self.device)
            epoch_loss["Current Reliable Pseudo Coverage"] += torch.tensor(
                pseudo_stats["coverage"], device=self.device
            )
            epoch_loss["Pseudo Agreement Rate"] += torch.tensor(
                pseudo_stats["agreement_rate"], device=self.device
            )
            epoch_loss["Pseudo Margin Rate"] += torch.tensor(
                pseudo_stats["margin_rate"], device=self.device
            )
            epoch_loss["Pseudo Mean Quality"] += torch.tensor(
                pseudo_stats["mean_quality"], device=self.device
            )
            epoch_loss["Valid Target Classes"] += torch.tensor(
                float(valid_target_classes), device=self.device
            )
            epoch_loss["Valid Prototype Pairs"] += torch.tensor(float(valid_pairs), device=self.device)

            class_weight_sum += class_weights
            global_weight_sum += global_weights

        denom = max(float(self.num_iter), 1.0)
        avg_class_weights = class_weight_sum / denom
        avg_global_weights = global_weight_sum / denom
        logging.info(
            "RCPA stages in epoch: warmup=%d global=%d prototype=%d",
            int(stage_counts[0].item()),
            int(stage_counts[1].item()),
            int(stage_counts[2].item()),
        )
        logging.info(
            "RCPA global source weights: %s",
            ", ".join(
                f"src{k}={avg_global_weights[k].item():.4f}"
                for k in range(self.num_source)
            ),
        )
        for c in range(self.num_classes):
            logging.info(
                "RCPA class-%d source weights: %s",
                c,
                ", ".join(
                    f"src{k}={avg_class_weights[k, c].item():.4f}"
                    for k in range(self.num_source)
                ),
            )
        logging.info("Average valid prototype pairs per iteration: %.2f", valid_proto_pairs_sum / denom)
        if hasattr(self.G, "get_gate"):
            logging.info("BiMamba residual gate: %.6f", self.G.get_gate().item())
        return epoch_acc, epoch_loss

    def _eval(self, data, actual_labels, correct, total):
        shared = self.G(data)
        probs = [
            F.softmax(self.Cs[k](self.Fs[k](shared)), dim=1)
            for k in range(self.num_source)
        ]
        fused, weights = self._eval_fusion(probs)
        pred = fused.argmax(dim=1)
        actual_pred = self._get_actual_label(pred, label_set=self.src_labels_flat)

        if hasattr(self, "_eval_pred_list"):
            self._eval_pred_list.append(actual_pred.detach().cpu())
        if hasattr(self, "_eval_label_list"):
            self._eval_label_list.append(actual_labels.detach().cpu())

        if not hasattr(self, "_eval_source_weight_sum"):
            self._eval_source_weight_sum = torch.zeros(self.num_source)
            self._eval_source_weight_count = 0
        if weights.dim() == 3:
            source_mean = weights.detach().mean(dim=(1, 2)).cpu()
        else:
            source_mean = weights.detach().mean(dim=1).cpu()
        self._eval_source_weight_sum += source_mean
        self._eval_source_weight_count += 1

        output = self._get_accuracy(actual_pred, actual_labels, return_acc=False)
        correct["acc"] += output[0]
        total["acc"] += output[1]
        return correct, total
