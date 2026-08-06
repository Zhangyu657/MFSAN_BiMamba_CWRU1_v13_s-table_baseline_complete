# -*- coding: utf-8 -*-
"""
V6-Lite PU0 全量可视化与错分诊断脚本

用途：
1. 复现 best.pth 在目标测试集上的预测；
2. 生成混淆矩阵、ROC/PR、置信度、校准、分支一致性、融合权重、
   特征空间、时间漂移、原始波形/频谱/时频图、输入梯度显著性等图；
3. 输出逐样本预测表、错分样本表、主要混淆对、最近邻分析和自动诊断摘要。

放置位置：项目根目录（与 train.py、models、data_loader 同级）。
推荐加载：*_best.pth，而不是最后一轮 *.pth。

说明：
- 可视化能够定位“错在哪里、何时错、哪个分支冲突、特征是否重叠、
  信号统计是否异常”，但不能仅凭单张图证明物理因果。
- 默认类别顺序必须与训练日志检测到的顺序一致：
  K001, KA04, KA16, KA30, KB23, KB24, KI04, KI16, KI17。
"""

import os
import re
import sys
import csv
import math
import json
import argparse
import logging
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    silhouette_samples,
)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

try:
    from scipy import stats
    from scipy.signal import spectrogram
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data_loader"))

EPS = 1e-12


def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("yes", "true", "t", "1", "y"):
        return True
    if str(v).lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    p = argparse.ArgumentParser(
        description="V6-Lite PU0 full visualization and misclassification diagnostics"
    )

    # ---------------- 基本实验 ----------------
    p.add_argument("--model_name", type=str,
                   default="MFSAN_CDAN_BIMAMBA_CW_RWCA_V6_LITE_PU0")
    p.add_argument("--source", type=str, default="PU_1,PU_2,PU_3")
    p.add_argument("--target", type=str, default="PU_0")
    p.add_argument("--train_mode", type=str, default="multi_source")
    p.add_argument("--data_dir", type=str, default="/workspace/PU_TL_9_replace")
    p.add_argument("--signal_size", type=int, default=1024)
    p.add_argument("--backbone", type=str, default="CNN")
    p.add_argument("--cuda_device", type=str, default="0")
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--log_path", type=str, default="")
    p.add_argument("--output_dir", type=str,
                   default="./visual_results/PU0_B1_full_diagnostics")

    # 重要：使用训练日志实际检测到的排序，不使用 include_faults 的输入顺序。
    p.add_argument(
        "--include_faults",
        type=str,
        default="K001,KA04,KA16,KA30,KB23,KB24,KI04,KI16,KI17",
    )
    p.add_argument("--exclude_faults", type=str, default="")
    p.add_argument("--target_test_size", type=float, default=0.40)
    p.add_argument("--target_split_mode", type=str, default="time")

    # ---------------- DataLoader / 优化器占位参数 ----------------
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--normlize_type", type=str, default="-1-1")
    p.add_argument("--random_state", type=int, default=2027)
    p.add_argument("--opt", type=str, default="sgd")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lr_scheduler", type=str, default="stepLR")
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--steps", type=str, default="10")
    p.add_argument("--tradeoff", type=str, default="exp,exp,exp")
    p.add_argument("--zeta", type=float, default=10.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--max_epoch", type=int, default=20)

    # ---------------- 损失及模型参数（与 B1 日志一致） ----------------
    p.add_argument("--lambda_l1", type=float, default=0.0)
    p.add_argument("--lambda_cda", type=float, default=0.0)
    p.add_argument("--lambda_ent", type=float, default=0.0)
    p.add_argument("--cda_detach_prob", type=str2bool, default=True)
    p.add_argument("--lambda_adv", type=float, default=0.02)
    p.add_argument("--lambda_grl", type=float, default=1.0)
    p.add_argument("--adv_hidden_dim", type=int, default=256)
    p.add_argument("--adv_detach_prob", type=str2bool, default=True)
    p.add_argument("--adv_use_entropy_weight", type=str2bool, default=True)
    p.add_argument("--adv_conf_thresh", type=float, default=0.0)

    p.add_argument("--bla_gate_init", type=float, default=0.01)
    p.add_argument("--bla_gate_max", type=float, default=0.03)
    p.add_argument("--bimamba_stem_channels", type=int, default=64)
    p.add_argument("--bimamba_dim", type=int, default=64)
    p.add_argument("--bimamba_depth", type=int, default=2)
    p.add_argument("--bimamba_d_state", type=int, default=16)
    p.add_argument("--bimamba_d_conv", type=int, default=4)
    p.add_argument("--bimamba_expand", type=int, default=2)
    p.add_argument("--bimamba_gate_init", type=float, default=0.01)
    p.add_argument("--bimamba_gate_max", type=float, default=0.03)

    p.add_argument("--rw_tau", type=float, default=0.5)
    p.add_argument("--rw_mmd_weight", type=float, default=1.0)
    p.add_argument("--rw_ent_weight", type=float, default=1.0)
    p.add_argument("--rw_detach_weights", type=str2bool, default=True)
    p.add_argument("--rw_ema_momentum", type=float, default=0.9)
    p.add_argument("--rw_eval_use_entropy", type=str2bool, default=True)
    p.add_argument("--rw_eval_tau", type=float, default=0.5)

    p.add_argument("--lambda_clmmd", type=float, default=0.005)
    p.add_argument("--clmmd_kernel_num", type=int, default=5)
    p.add_argument("--clmmd_kernel_mul", type=float, default=2.0)
    p.add_argument("--clmmd_fix_sigma", type=float, default=None)
    p.add_argument("--clmmd_min_source", type=int, default=2)
    p.add_argument("--clmmd_min_target_weight", type=float, default=0.001)
    p.add_argument("--pl_conf_thresh", type=float, default=0.80)
    p.add_argument("--pl_min_target", type=int, default=2)

    p.add_argument("--cw_warmup_epochs", type=int, default=3)
    p.add_argument("--cw_alpha", type=float, default=0.30)
    p.add_argument("--cw_alpha_ramp_epochs", type=int, default=3)

    p.add_argument("--lambda_supcon", type=float, default=0.01)
    p.add_argument("--supcon_temperature", type=float, default=0.20)
    p.add_argument("--supcon_start_epoch", type=int, default=3)
    p.add_argument("--supcon_feature_mode", type=str, default="G", choices=["G", "F"])
    p.add_argument("--supcon_focus_classes", type=str, default="1,3,8")

    p.add_argument("--rec_score_weight", type=float, default=0.30)
    p.add_argument("--rec_score_mode", type=str, default="prob",
                   choices=["prob", "acc", "mix"])
    p.add_argument("--rec_score_detach", type=str2bool, default=True)
    p.add_argument("--lambda_mca", type=float, default=0.0)
    p.add_argument("--mca_start_epoch", type=int, default=1)
    p.add_argument("--mca_use_reliability", type=str2bool, default=True)
    p.add_argument("--mca_detach_fused", type=str2bool, default=True)
    p.add_argument("--mca_eps", type=float, default=1e-5)

    # B1 门控参数；分析实验 A 时在命令行改成 confirm=3, pre_floor=0.05。
    p.add_argument("--v6_gate_enabled", type=str2bool, default=True)
    p.add_argument("--v6_gate_start_epoch", type=int, default=4)
    p.add_argument("--v6_gate_confirm_epochs", type=int, default=2)
    p.add_argument("--v6_gate_release_epochs", type=int, default=3)
    p.add_argument("--v6_gate_confirm_gap", type=float, default=0.08)
    p.add_argument("--v6_gate_release_gap", type=float, default=0.03)
    p.add_argument("--v6_gate_preconfirm_floor", type=float, default=0.01)
    p.add_argument("--v6_gate_bottom_floor", type=float, default=0.01)
    p.add_argument("--v6_gate_max_source_weight", type=float, default=0.75)
    p.add_argument("--v6_gate_apply_to_supcon", type=str2bool, default=True)
    p.add_argument("--v6_supcon_source_min_weight", type=float, default=0.05)
    p.add_argument("--v6_class_weight_power", type=float, default=1.20)
    p.add_argument("--v6_class_alignment_boost", type=float, default=1.0)
    p.add_argument("--v6_mca_pairwise_weight", type=float, default=0.25)

    # ---------------- 分析开关 ----------------
    p.add_argument("--feature_mode", type=str, default="F_mean", choices=["G", "F_mean"])
    p.add_argument("--tsne_max_per_class", type=int, default=250)
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_iter", type=int, default=1000)
    p.add_argument("--top_confusion_pairs", type=int, default=8)
    p.add_argument("--representative_errors", type=int, default=6)
    p.add_argument("--rolling_window", type=int, default=500)
    p.add_argument("--calibration_bins", type=int, default=15)
    p.add_argument("--sampling_rate", type=float, default=1.0,
                   help="未知时保持1.0，频率单位显示为cycles/sample；已知采样率可填Hz。")
    p.add_argument("--integrated_gradients_steps", type=int, default=16)
    p.add_argument("--skip_heavy", type=str2bool, default=False,
                   help="True时跳过t-SNE、频谱样本图和梯度显著性。")
    p.add_argument("--save_raw_npz", type=str2bool, default=False)

    args = p.parse_args()
    args.source_name = [x.strip() for x in args.source.split(",") if x.strip()]
    args.tradeoff = [x.strip() for x in args.tradeoff.split(",") if x.strip()]
    args.betas = tuple(args.betas)
    args.save = False
    args.save_best = False
    args.save_dir = "./ckpt"
    args.save_path = os.path.join(args.output_dir, "dummy")
    args.load_path = args.ckpt_path
    args.da_scenario = "closed-set"
    return args


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "visualization.log"),
                                mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_class_names(args):
    names = [x.strip() for x in args.include_faults.split(",") if x.strip()]
    if args.exclude_faults.strip():
        excluded = {x.strip() for x in args.exclude_faults.split(",") if x.strip()}
        names = [x for x in names if x not in excluded]
    if not names:
        raise ValueError("No class names. Check --include_faults.")
    return names


def prepare_label_sets(args, class_names):
    label_set = list(range(len(class_names)))
    args.label_sets = [label_set[:] for _ in args.source_name] + [label_set[:]]
    args.faults = [class_names[:] for _ in args.source_name] + [class_names[:]]
    args.fault_label = {name: idx for idx, name in enumerate(class_names)}
    args.class_names = class_names
    args.selected_faults = class_names
    args.fault_names = class_names
    args.num_classes = len(class_names)
    return args


def build_trainer(args):
    import importlib
    module = importlib.import_module(f"models.{args.model_name}")
    return module.Trainer(args)


def manual_load(trainer, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=trainer.device)
    for key, attr in (("G", "G"), ("Fs", "Fs"), ("Cs", "Cs"), ("Ds", "Ds")):
        if key in ckpt and hasattr(trainer, attr):
            getattr(trainer, attr).load_state_dict(ckpt[key])
    return ckpt


def load_checkpoint(trainer, ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    trainer.args.load_path = ckpt_path
    try:
        trainer.load_model()
        ckpt = torch.load(ckpt_path, map_location=trainer.device)
        logging.info("Checkpoint loaded by trainer.load_model().")
    except Exception as exc:
        logging.warning("trainer.load_model() failed; manual fallback: %s", exc)
        ckpt = manual_load(trainer, ckpt_path)
    return ckpt


def set_eval(trainer):
    trainer._set_to_eval()
    for obj_name in ("G", "Fs", "Cs", "Ds"):
        obj = getattr(trainer, obj_name, None)
        if obj is not None:
            try:
                obj.eval()
            except Exception:
                pass


def flatten_signal_batch(data):
    arr = data.detach().cpu().numpy().astype(np.float32)
    return arr.reshape(arr.shape[0], -1)


def expand_fusion_weights(weights, batch_size, num_source, num_classes, device):
    """统一为 [K, B, C]。"""
    if weights is None:
        return torch.full(
            (num_source, batch_size, num_classes),
            1.0 / num_source,
            device=device,
        )
    if weights.dim() == 2:  # [K, C]
        return weights.unsqueeze(1).expand(-1, batch_size, -1)
    if weights.dim() == 3:
        return weights
    raise ValueError(f"Unexpected fusion weight shape: {tuple(weights.shape)}")


def forward_diagnostics(trainer, data, feature_mode="F_mean", require_grad=False):
    ctx = torch.enable_grad() if require_grad else torch.no_grad()
    with ctx:
        feat_g = trainer.G(data)
        feat_list, logit_list, prob_list = [], [], []
        for i in range(trainer.num_source):
            feat_i = trainer.Fs[i](feat_g)
            logit_i = trainer.Cs[i](feat_i)
            prob_i = F.softmax(logit_i, dim=1)
            feat_list.append(feat_i)
            logit_list.append(logit_i)
            prob_list.append(prob_i)

        fusion_weights = None
        if hasattr(trainer, "_eval_class_weighted_fusion"):
            out = trainer._eval_class_weighted_fusion(prob_list)
            if isinstance(out, tuple):
                fused_prob = out[0]
                if len(out) > 1:
                    fusion_weights = out[1]
            else:
                fused_prob = out
        else:
            fused_prob = torch.stack(prob_list, dim=0).mean(dim=0)

        fusion_weights = expand_fusion_weights(
            fusion_weights,
            data.size(0),
            trainer.num_source,
            fused_prob.size(1),
            data.device,
        )

        pred_train = fused_prob.argmax(dim=1)
        pred_actual = trainer._get_actual_label(
            pred_train, label_set=trainer.src_labels_flat
        )

        branch_probs = torch.stack(prob_list, dim=1)  # [B, K, C]
        branch_pred_train = branch_probs.argmax(dim=2)  # [B, K]
        branch_pred_actual = []
        for k in range(trainer.num_source):
            branch_pred_actual.append(
                trainer._get_actual_label(
                    branch_pred_train[:, k], label_set=trainer.src_labels_flat
                )
            )
        branch_pred_actual = torch.stack(branch_pred_actual, dim=1)

        if feature_mode == "G":
            feat_plot = feat_g
        else:
            feat_plot = torch.stack(feat_list, dim=0).mean(dim=0)

        return {
            "fused_probs": fused_prob,
            "preds": pred_actual,
            "features": feat_plot,
            "shared_features": feat_g,
            "branch_probs": branch_probs,
            "branch_preds": branch_pred_actual,
            "fusion_weights": fusion_weights.permute(1, 0, 2),  # [B,K,C]
        }


def collect_target_diagnostics(trainer, args):
    set_eval(trainer)
    stores = defaultdict(list)
    loader = trainer.dataloaders["val"]

    for data, labels in loader:
        data_device = data.to(trainer.device, non_blocking=True)
        out = forward_diagnostics(trainer, data_device, args.feature_mode, False)
        stores["labels"].append(labels.cpu())
        stores["signals"].append(torch.from_numpy(flatten_signal_batch(data)))
        for key in ("fused_probs", "preds", "features", "branch_probs",
                    "branch_preds", "fusion_weights"):
            stores[key].append(out[key].detach().cpu())

    result = {}
    for key, chunks in stores.items():
        result[key] = torch.cat(chunks, dim=0).numpy()
    return result


def entropy_np(probs):
    p = np.clip(probs, EPS, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def confidence_margin(probs):
    sorted_p = np.sort(probs, axis=1)
    return sorted_p[:, -1], sorted_p[:, -1] - sorted_p[:, -2], sorted_p[:, -2]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_matrix_csv(matrix, row_names, col_names, path):
    pd.DataFrame(matrix, index=row_names, columns=col_names).to_csv(
        path, encoding="utf-8-sig"
    )


def plot_matrix(matrix, xlabels, ylabels, title, xlabel, ylabel, path,
                text_fmt=None, figsize=None):
    if figsize is None:
        figsize = (max(8, len(xlabels) * 1.05), max(6, len(ylabels) * 0.75))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)
    if text_fmt is not None:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, text_fmt(matrix[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_detailed_dataframe(data, class_names, source_names):
    labels = data["labels"].astype(int)
    preds = data["preds"].astype(int)
    probs = data["fused_probs"]
    branch_probs = data["branch_probs"]
    branch_preds = data["branch_preds"].astype(int)
    fusion_weights = data["fusion_weights"]

    conf, margin, second_prob = confidence_margin(probs)
    second_pred = np.argsort(probs, axis=1)[:, -2]
    ent = entropy_np(probs)
    correct = labels == preds
    branch_conf = branch_probs.max(axis=2)
    branch_agree_count = (branch_preds == preds[:, None]).sum(axis=1)
    unique_branch_count = np.array([len(set(row.tolist())) for row in branch_preds])

    rows = {
        "index": np.arange(len(labels)),
        "true_id": labels,
        "true_name": [class_names[i] for i in labels],
        "pred_id": preds,
        "pred_name": [class_names[i] for i in preds],
        "correct": correct.astype(int),
        "confidence": conf,
        "entropy": ent,
        "margin": margin,
        "second_pred_id": second_pred,
        "second_pred_name": [class_names[i] for i in second_pred],
        "second_probability": second_prob,
        "branch_agree_with_fused": branch_agree_count,
        "branch_unique_prediction_count": unique_branch_count,
    }

    for c, name in enumerate(class_names):
        rows[f"prob_{name}"] = probs[:, c]
    for k, source in enumerate(source_names):
        rows[f"{source}_pred_id"] = branch_preds[:, k]
        rows[f"{source}_pred_name"] = [class_names[i] for i in branch_preds[:, k]]
        rows[f"{source}_confidence"] = branch_conf[:, k]
        rows[f"{source}_weight_on_true"] = fusion_weights[np.arange(len(labels)), k, labels]
        rows[f"{source}_weight_on_pred"] = fusion_weights[np.arange(len(labels)), k, preds]

    return pd.DataFrame(rows)


def basic_metrics_and_plots(df, probs, class_names, out_dir):
    labels = df.true_id.to_numpy()
    preds = df.pred_id.to_numpy()
    ids = np.arange(len(class_names))

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=ids, zero_division=0
    )
    metrics = pd.DataFrame({
        "class_id": ids,
        "class_name": class_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "error_rate": 1.0 - recall,
    })
    metrics.to_csv(os.path.join(out_dir, "class_metrics.csv"), index=False,
                   encoding="utf-8-sig")

    summary = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=np.maximum(support, 1))),
    }
    with open(os.path.join(out_dir, "overall_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(labels, preds, labels=ids)
    cm_true = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_pred = cm / np.maximum(cm.sum(axis=0, keepdims=True), 1)
    save_matrix_csv(cm, class_names, class_names,
                    os.path.join(out_dir, "confusion_counts.csv"))
    save_matrix_csv(cm_true, class_names, class_names,
                    os.path.join(out_dir, "confusion_true_normalized.csv"))
    save_matrix_csv(cm_pred, class_names, class_names,
                    os.path.join(out_dir, "confusion_pred_normalized.csv"))
    plot_matrix(cm, class_names, class_names, "Confusion Matrix (Counts)",
                "Predicted", "True",
                os.path.join(out_dir, "01_confusion_counts.png"),
                text_fmt=lambda x: str(int(x)))
    plot_matrix(cm_true, class_names, class_names,
                "Confusion Matrix (Normalized by True Class)",
                "Predicted", "True",
                os.path.join(out_dir, "02_confusion_true_normalized.png"),
                text_fmt=lambda x: f"{x:.2f}")
    plot_matrix(cm_pred, class_names, class_names,
                "Confusion Matrix (Normalized by Predicted Class)",
                "Predicted", "True",
                os.path.join(out_dir, "03_confusion_pred_normalized.png"),
                text_fmt=lambda x: f"{x:.2f}")

    # PRF bar
    x = np.arange(len(class_names)); width = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Score"); ax.set_title("Per-class Precision / Recall / F1")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_per_class_prf.png"), dpi=300)
    plt.close(fig)

    # True vs predicted distribution
    true_count = np.bincount(labels, minlength=len(class_names))
    pred_count = np.bincount(preds, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 0.2, true_count, 0.4, label="True")
    ax.bar(x + 0.2, pred_count, 0.4, label="Predicted")
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Samples"); ax.set_title("True vs Predicted Class Distribution")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_true_vs_pred_distribution.png"), dpi=300)
    plt.close(fig)

    # Top confusion pairs
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), i, j, float(cm_true[i, j])))
    pairs.sort(reverse=True)
    pair_rows = []
    for count, i, j, rate in pairs:
        pair_rows.append({
            "true_id": i, "true_name": class_names[i],
            "pred_id": j, "pred_name": class_names[j],
            "count": count, "true_class_error_share": rate,
        })
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(os.path.join(out_dir, "top_confusion_pairs.csv"), index=False,
                   encoding="utf-8-sig")
    if len(pair_df):
        top = pair_df.head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(np.arange(len(top)), top["count"].to_numpy())
        ax.set_yticks(np.arange(len(top)))
        ax.set_yticklabels([f"{a} → {b}" for a, b in zip(top.true_name, top.pred_name)])
        ax.set_xlabel("Misclassified samples")
        ax.set_title("Top Misclassification Flows")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "06_top_misclassification_pairs.png"), dpi=300)
        plt.close(fig)

    # ROC / PR
    y_bin = label_binarize(labels, classes=ids)
    fig, ax = plt.subplots(figsize=(10, 8))
    roc_rows = []
    for c, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
        score = auc(fpr, tpr)
        roc_rows.append({"class": name, "auc": score})
        ax.plot(fpr, tpr, label=f"{name} AUC={score:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "07_roc_curves.png"), dpi=300)
    plt.close(fig)
    pd.DataFrame(roc_rows).to_csv(os.path.join(out_dir, "roc_auc.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    pr_rows = []
    for c, name in enumerate(class_names):
        pr, rc, _ = precision_recall_curve(y_bin[:, c], probs[:, c])
        ap = average_precision_score(y_bin[:, c], probs[:, c])
        pr_rows.append({"class": name, "average_precision": ap})
        ax.plot(rc, pr, label=f"{name} AP={ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("One-vs-Rest Precision-Recall Curves"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "08_pr_curves.png"), dpi=300)
    plt.close(fig)
    pd.DataFrame(pr_rows).to_csv(os.path.join(out_dir, "pr_average_precision.csv"), index=False)

    return metrics, cm, cm_true, pair_df, summary


def calibration_and_uncertainty(df, class_names, args, out_dir):
    correct = df.correct.to_numpy().astype(bool)
    conf = df.confidence.to_numpy()
    ent = df.entropy.to_numpy()
    margin = df.margin.to_numpy()

    # Confidence hist correct/wrong
    bins = np.linspace(0, 1, 31)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(conf[correct], bins=bins, alpha=0.65, density=True, label="Correct")
    ax.hist(conf[~correct], bins=bins, alpha=0.65, density=True, label="Wrong")
    ax.set_xlabel("Maximum probability"); ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution: Correct vs Wrong")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_confidence_correct_vs_wrong.png"), dpi=300)
    plt.close(fig)

    # Entropy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ent[correct], bins=30, alpha=0.65, density=True, label="Correct")
    ax.hist(ent[~correct], bins=30, alpha=0.65, density=True, label="Wrong")
    ax.set_xlabel("Prediction entropy"); ax.set_ylabel("Density")
    ax.set_title("Entropy Distribution: Correct vs Wrong")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "10_entropy_correct_vs_wrong.png"), dpi=300)
    plt.close(fig)

    # Margin
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(margin[correct], bins=30, alpha=0.65, density=True, label="Correct")
    ax.hist(margin[~correct], bins=30, alpha=0.65, density=True, label="Wrong")
    ax.set_xlabel("Top-1 minus Top-2 probability"); ax.set_ylabel("Density")
    ax.set_title("Prediction Margin: Correct vs Wrong")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "11_margin_correct_vs_wrong.png"), dpi=300)
    plt.close(fig)

    # Reliability diagram + ECE
    edges = np.linspace(0, 1, args.calibration_bins + 1)
    mids, accs, confs, counts = [], [], [], []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi if hi < 1 else conf <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(conf[mask].mean())
        mids.append((lo + hi) / 2)
        accs.append(bin_acc); confs.append(bin_conf); counts.append(n)
        ece += n / len(conf) * abs(bin_acc - bin_conf)
    cal_df = pd.DataFrame({"bin_mid": mids, "accuracy": accs,
                           "mean_confidence": confs, "count": counts})
    cal_df.to_csv(os.path.join(out_dir, "calibration_bins.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(confs, accs, marker="o", label=f"Model ECE={ece:.4f}")
    ax.set_xlabel("Mean confidence"); ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "12_reliability_diagram.png"), dpi=300)
    plt.close(fig)

    # Risk-coverage curve
    order = np.argsort(-conf)
    corr_sorted = correct[order].astype(float)
    coverage = np.arange(1, len(conf) + 1) / len(conf)
    risk = 1.0 - np.cumsum(corr_sorted) / np.arange(1, len(conf) + 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(coverage, risk)
    ax.set_xlabel("Coverage (fraction retained)"); ax.set_ylabel("Error rate")
    ax.set_title("Risk-Coverage Curve")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "13_risk_coverage.png"), dpi=300)
    plt.close(fig)

    # Per-class confidence / entropy / error
    grouped = df.groupby("true_name").agg(
        accuracy=("correct", "mean"),
        mean_confidence=("confidence", "mean"),
        mean_entropy=("entropy", "mean"),
        mean_margin=("margin", "mean"),
        samples=("correct", "size"),
    ).reindex(class_names)
    grouped.to_csv(os.path.join(out_dir, "per_class_uncertainty.csv"),
                   encoding="utf-8-sig")
    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 0.2, grouped.accuracy, 0.4, label="Accuracy")
    ax.bar(x + 0.2, grouped.mean_confidence, 0.4, label="Mean confidence")
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05); ax.set_title("Per-class Accuracy vs Confidence")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "14_per_class_accuracy_confidence.png"), dpi=300)
    plt.close(fig)

    high_conf_errors = df[(df.correct == 0)].sort_values("confidence", ascending=False)
    high_conf_errors.to_csv(os.path.join(out_dir, "high_confidence_errors.csv"),
                            index=False, encoding="utf-8-sig")
    low_conf_correct = df[(df.correct == 1)].sort_values("confidence")
    low_conf_correct.to_csv(os.path.join(out_dir, "low_confidence_correct.csv"),
                            index=False, encoding="utf-8-sig")
    return float(ece)


def branch_and_fusion_analysis(df, data, class_names, source_names, out_dir):
    labels = df.true_id.to_numpy().astype(int)
    fused_preds = df.pred_id.to_numpy().astype(int)
    branch_preds = data["branch_preds"].astype(int)
    branch_probs = data["branch_probs"]
    weights = data["fusion_weights"]
    k_count = len(source_names)

    accs = [accuracy_score(labels, branch_preds[:, k]) for k in range(k_count)]
    accs.append(accuracy_score(labels, fused_preds))
    names = source_names + ["Fused"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(np.arange(len(names)), accs)
    ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title("Individual Source Classifiers vs Fused Prediction")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "15_branch_vs_fused_accuracy.png"), dpi=300)
    plt.close(fig)

    # Each branch confusion and per-class accuracy
    per_branch_class_acc = np.zeros((k_count + 1, len(class_names)))
    for k in range(k_count):
        cm = confusion_matrix(labels, branch_preds[:, k], labels=np.arange(len(class_names)))
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        per_branch_class_acc[k] = np.diag(cmn)
        plot_matrix(cmn, class_names, class_names,
                    f"{source_names[k]} Branch Confusion (True-normalized)",
                    "Predicted", "True",
                    os.path.join(out_dir, f"16_branch_{k}_{source_names[k]}_confusion.png"),
                    text_fmt=lambda x: f"{x:.2f}")
    fused_cm = confusion_matrix(labels, fused_preds, labels=np.arange(len(class_names)))
    fused_cmn = fused_cm / np.maximum(fused_cm.sum(axis=1, keepdims=True), 1)
    per_branch_class_acc[-1] = np.diag(fused_cmn)
    plot_matrix(per_branch_class_acc, class_names, names,
                "Per-class Accuracy of Each Source Branch and Fusion",
                "True Class", "Predictor",
                os.path.join(out_dir, "17_branch_per_class_accuracy_heatmap.png"),
                text_fmt=lambda x: f"{x:.2f}")
    save_matrix_csv(per_branch_class_acc, names, class_names,
                    os.path.join(out_dir, "branch_per_class_accuracy.csv"))

    # Pairwise agreement
    agree = np.eye(k_count)
    for i in range(k_count):
        for j in range(k_count):
            agree[i, j] = np.mean(branch_preds[:, i] == branch_preds[:, j])
    plot_matrix(agree, source_names, source_names,
                "Pairwise Prediction Agreement between Source Branches",
                "Branch", "Branch",
                os.path.join(out_dir, "18_branch_pairwise_agreement.png"),
                text_fmt=lambda x: f"{x:.3f}")
    save_matrix_csv(agree, source_names, source_names,
                    os.path.join(out_dir, "branch_pairwise_agreement.csv"))

    # Disagreement and errors
    unique_count = df.branch_unique_prediction_count.to_numpy()
    correct = df.correct.to_numpy().astype(bool)
    levels = np.arange(1, k_count + 1)
    error_rate = []
    sample_count = []
    for level in levels:
        mask = unique_count == level
        sample_count.append(int(mask.sum()))
        error_rate.append(float((~correct[mask]).mean()) if mask.any() else np.nan)
    dis_df = pd.DataFrame({"unique_branch_predictions": levels,
                           "samples": sample_count, "error_rate": error_rate})
    dis_df.to_csv(os.path.join(out_dir, "branch_disagreement_error_rate.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(levels, np.nan_to_num(error_rate))
    ax.set_xticks(levels); ax.set_xlabel("Number of unique branch predictions")
    ax.set_ylabel("Error rate"); ax.set_title("Error Rate vs Branch Disagreement")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "19_error_vs_branch_disagreement.png"), dpi=300)
    plt.close(fig)

    # Fusion weights by true class; average on each sample's true class.
    true_weight = np.zeros((k_count, len(class_names)))
    pred_weight = np.zeros((k_count, len(class_names)))
    for c in range(len(class_names)):
        mask_true = labels == c
        mask_pred = fused_preds == c
        if mask_true.any():
            for k in range(k_count):
                true_weight[k, c] = weights[mask_true, k, c].mean()
        if mask_pred.any():
            for k in range(k_count):
                pred_weight[k, c] = weights[mask_pred, k, c].mean()
    plot_matrix(true_weight, class_names, source_names,
                "Average Source Fusion Weight on the True Class",
                "True Class", "Source",
                os.path.join(out_dir, "20_fusion_weight_by_true_class.png"),
                text_fmt=lambda x: f"{x:.3f}")
    plot_matrix(pred_weight, class_names, source_names,
                "Average Source Fusion Weight on the Predicted Class",
                "Predicted Class", "Source",
                os.path.join(out_dir, "21_fusion_weight_by_pred_class.png"),
                text_fmt=lambda x: f"{x:.3f}")
    save_matrix_csv(true_weight, source_names, class_names,
                    os.path.join(out_dir, "fusion_weight_by_true_class.csv"))

    # Average source weight correct vs wrong, on final predicted class.
    pred_class_weights = weights[np.arange(len(labels)), :, fused_preds]
    avg_correct = pred_class_weights[correct].mean(axis=0)
    avg_wrong = pred_class_weights[~correct].mean(axis=0)
    x = np.arange(k_count)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - 0.2, avg_correct, 0.4, label="Correct")
    ax.bar(x + 0.2, avg_wrong, 0.4, label="Wrong")
    ax.set_xticks(x); ax.set_xticklabels(source_names)
    ax.set_ylabel("Mean source weight on predicted class")
    ax.set_title("Fusion Source Weight: Correct vs Wrong Samples")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "22_fusion_weights_correct_vs_wrong.png"), dpi=300)
    plt.close(fig)

    # Which branch drove high-confidence errors?
    wrong = ~correct
    driver = np.argmax(pred_class_weights, axis=1)
    driver_rows = []
    for k, name in enumerate(source_names):
        m = wrong & (driver == k)
        driver_rows.append({
            "source": name,
            "wrong_samples_driven": int(m.sum()),
            "mean_error_confidence": float(df.confidence.to_numpy()[m].mean()) if m.any() else np.nan,
        })
    pd.DataFrame(driver_rows).to_csv(os.path.join(out_dir, "error_dominant_source.csv"), index=False)


def stratified_indices(labels, max_per_class, seed=2027):
    rng = np.random.default_rng(seed)
    result = []
    for c in sorted(np.unique(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        result.extend(idx.tolist())
    result = np.array(result, dtype=int)
    rng.shuffle(result)
    return result


def feature_space_analysis(df, features, class_names, args, out_dir):
    labels = df.true_id.to_numpy().astype(int)
    preds = df.pred_id.to_numpy().astype(int)
    correct = df.correct.to_numpy().astype(bool)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)

    # PCA by class
    pca = PCA(n_components=2, random_state=args.random_state)
    emb_pca = pca.fit_transform(x_scaled)
    fig, ax = plt.subplots(figsize=(10, 8))
    for c, name in enumerate(class_names):
        m = labels == c
        ax.scatter(emb_pca[m, 0], emb_pca[m, 1], s=7, alpha=0.45, label=name)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"PCA by True Class (explained={pca.explained_variance_ratio_.sum():.3f})")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "23_pca_by_true_class.png"), dpi=300)
    plt.close(fig)

    # PCA correct vs wrong
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(emb_pca[correct, 0], emb_pca[correct, 1], s=7, alpha=0.35, label="Correct")
    ax.scatter(emb_pca[~correct, 0], emb_pca[~correct, 1], s=18, alpha=0.75, label="Wrong")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA: Correct vs Wrong")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "24_pca_correct_vs_wrong.png"), dpi=300)
    plt.close(fig)

    # Class centroids and distances
    centroids = np.vstack([x_scaled[labels == c].mean(axis=0) for c in range(len(class_names))])
    dist = np.sqrt(((centroids[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
    plot_matrix(dist, class_names, class_names, "Euclidean Distance between Class Centroids",
                "Class", "Class", os.path.join(out_dir, "25_class_centroid_distance.png"),
                text_fmt=lambda x: f"{x:.2f}")
    save_matrix_csv(dist, class_names, class_names,
                    os.path.join(out_dir, "class_centroid_distance.csv"))

    # Silhouette per sample and class
    try:
        max_sil = min(8000, len(labels))
        idx_sil = stratified_indices(labels, max_sil // len(class_names), args.random_state)
        sil = silhouette_samples(x_scaled[idx_sil], labels[idx_sil], metric="euclidean")
        sil_df = pd.DataFrame({"true_id": labels[idx_sil], "silhouette": sil})
        per_class_sil = sil_df.groupby("true_id").silhouette.mean().reindex(range(len(class_names)))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(np.arange(len(class_names)), per_class_sil.to_numpy())
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_ylabel("Mean silhouette"); ax.set_title("Per-class Feature Separability")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "26_per_class_silhouette.png"), dpi=300)
        plt.close(fig)
    except Exception as exc:
        logging.warning("Silhouette analysis skipped: %s", exc)

    # Nearest-neighbor diagnosis: wrong sample's neighbors in feature space.
    nn = NearestNeighbors(n_neighbors=min(11, len(labels)), metric="euclidean")
    nn.fit(x_scaled)
    wrong_idx = np.where(~correct)[0]
    rows = []
    for idx in wrong_idx:
        distances, neigh = nn.kneighbors(x_scaled[idx:idx+1], return_distance=True)
        neigh = neigh[0][1:]; distances = distances[0][1:]
        neigh_labels = labels[neigh]
        neigh_preds = preds[neigh]
        rows.append({
            "index": int(idx),
            "true_name": class_names[labels[idx]],
            "pred_name": class_names[preds[idx]],
            "confidence": float(df.confidence.iloc[idx]),
            "nearest_true_class_fraction": float(np.mean(neigh_labels == labels[idx])),
            "nearest_pred_class_fraction": float(np.mean(neigh_labels == preds[idx])),
            "neighbor_true_names": "|".join(class_names[x] for x in neigh_labels),
            "neighbor_pred_names": "|".join(class_names[x] for x in neigh_preds),
            "neighbor_distances": "|".join(f"{x:.5f}" for x in distances),
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "wrong_sample_nearest_neighbors.csv"),
                              index=False, encoding="utf-8-sig")

    if not args.skip_heavy:
        idx = stratified_indices(labels, args.tsne_max_per_class, args.random_state)
        perplexity = min(args.tsne_perplexity, max(5.0, (len(idx) - 1) / 3.0))
        kwargs = dict(n_components=2, perplexity=perplexity, init="pca",
                      learning_rate="auto", random_state=args.random_state)
        try:
            tsne = TSNE(max_iter=args.tsne_iter, **kwargs)
        except TypeError:
            tsne = TSNE(n_iter=args.tsne_iter, **kwargs)
        emb = tsne.fit_transform(x_scaled[idx])
        y = labels[idx]; corr = correct[idx]
        fig, ax = plt.subplots(figsize=(10, 8))
        for c, name in enumerate(class_names):
            m = y == c
            ax.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.65, label=name)
        ax.set_title("t-SNE by True Class"); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "27_tsne_by_true_class.png"), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(emb[corr, 0], emb[corr, 1], s=8, alpha=0.35, label="Correct")
        ax.scatter(emb[~corr, 0], emb[~corr, 1], s=22, alpha=0.8, label="Wrong")
        ax.set_title("t-SNE: Correct vs Wrong")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "28_tsne_correct_vs_wrong.png"), dpi=300)
        plt.close(fig)

    return dist


def temporal_analysis(df, class_names, args, out_dir):
    """目标集按time切分，样本顺序可用于检查后半段工况漂移。"""
    correct = df.correct.to_numpy().astype(float)
    conf = df.confidence.to_numpy()
    n = len(df)
    window = min(max(50, args.rolling_window), max(50, n // 3))
    kernel = np.ones(window) / window
    rolling_acc = np.convolve(correct, kernel, mode="valid")
    rolling_conf = np.convolve(conf, kernel, mode="valid")

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(np.arange(len(rolling_acc)) + window - 1, rolling_acc)
    ax.set_xlabel("Target-test sample order")
    ax.set_ylabel("Rolling accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Rolling Accuracy over Target-test Order (window={window})")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "29_rolling_accuracy_time_order.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(np.arange(len(rolling_conf)) + window - 1, rolling_conf)
    ax.set_xlabel("Target-test sample order")
    ax.set_ylabel("Rolling mean confidence")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Rolling Confidence over Target-test Order (window={window})")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "30_rolling_confidence_time_order.png"), dpi=300)
    plt.close(fig)

    # Quartile x class error rate
    quartile = np.minimum((np.arange(n) * 4 // n), 3)
    labels = df.true_id.to_numpy().astype(int)
    matrix = np.full((4, len(class_names)), np.nan)
    for q in range(4):
        for c in range(len(class_names)):
            m = (quartile == q) & (labels == c)
            if m.any():
                matrix[q, c] = 1.0 - correct[m].mean()
    plot_matrix(np.nan_to_num(matrix), class_names,
                ["Q1", "Q2", "Q3", "Q4"],
                "Error Rate by Target-test Time Quartile and Class",
                "True Class", "Time Quartile",
                os.path.join(out_dir, "31_error_rate_time_quartile_class.png"),
                text_fmt=lambda x: f"{x:.3f}")
    save_matrix_csv(matrix, ["Q1", "Q2", "Q3", "Q4"], class_names,
                    os.path.join(out_dir, "error_rate_time_quartile_class.csv"))


def spectral_entropy(x):
    spec = np.abs(np.fft.rfft(x)) ** 2
    p = spec / (spec.sum() + EPS)
    return float(-(p * np.log(p + EPS)).sum() / np.log(len(p) + EPS))


def signal_feature_table(signals, df, sampling_rate):
    rows = []
    for i, x in enumerate(signals):
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        rms = math.sqrt(float(np.mean(x ** 2)) + EPS)
        mean_abs = float(abs_x.mean())
        peak = float(abs_x.max())
        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), d=1.0 / sampling_rate)
        power = spec ** 2
        spectral_centroid = float((freqs * power).sum() / (power.sum() + EPS))
        dominant_freq = float(freqs[int(np.argmax(power))])
        if SCIPY_AVAILABLE:
            skew = float(stats.skew(x, bias=False, nan_policy="omit"))
            kurtosis = float(stats.kurtosis(x, fisher=False, bias=False, nan_policy="omit"))
        else:
            centered = x - x.mean()
            sd = x.std() + EPS
            skew = float(np.mean((centered / sd) ** 3))
            kurtosis = float(np.mean((centered / sd) ** 4))
        rows.append({
            "index": i,
            "true_name": df.true_name.iloc[i],
            "pred_name": df.pred_name.iloc[i],
            "correct": int(df.correct.iloc[i]),
            "confidence": float(df.confidence.iloc[i]),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "rms": rms,
            "peak_abs": peak,
            "peak_to_peak": float(np.ptp(x)),
            "mean_abs": mean_abs,
            "skewness": skew,
            "kurtosis": kurtosis,
            "crest_factor": peak / (rms + EPS),
            "impulse_factor": peak / (mean_abs + EPS),
            "spectral_centroid": spectral_centroid,
            "dominant_frequency": dominant_freq,
            "spectral_entropy": spectral_entropy(x),
        })
    return pd.DataFrame(rows)


def cohens_d(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) +
                        (len(b) - 1) * np.var(b, ddof=1)) /
                       max(len(a) + len(b) - 2, 1))
    return float((np.mean(b) - np.mean(a)) / (pooled + EPS))


def signal_statistical_analysis(signals, df, class_names, out_dir, sampling_rate):
    sig_df = signal_feature_table(signals, df, sampling_rate)
    sig_df.to_csv(os.path.join(out_dir, "signal_features_all_samples.csv"),
                  index=False, encoding="utf-8-sig")
    feature_cols = [
        "std", "rms", "peak_abs", "peak_to_peak", "skewness", "kurtosis",
        "crest_factor", "impulse_factor", "spectral_centroid",
        "dominant_frequency", "spectral_entropy",
    ]

    effect = np.full((len(class_names), len(feature_cols)), np.nan)
    for c, name in enumerate(class_names):
        sub = sig_df[sig_df.true_name == name]
        for j, feat in enumerate(feature_cols):
            effect[c, j] = cohens_d(
                sub.loc[sub.correct == 1, feat],
                sub.loc[sub.correct == 0, feat],
            )
    plot_matrix(np.nan_to_num(effect), feature_cols, class_names,
                "Signal Feature Shift: Wrong minus Correct (Cohen's d)",
                "Signal feature", "True Class",
                os.path.join(out_dir, "32_signal_feature_effect_size.png"),
                text_fmt=lambda x: f"{x:+.2f}",
                figsize=(15, 8))
    save_matrix_csv(effect, class_names, feature_cols,
                    os.path.join(out_dir, "signal_feature_effect_size.csv"))

    # Global boxplots for the most interpretable features.
    for idx, feat in enumerate(["rms", "kurtosis", "crest_factor", "spectral_entropy"]):
        arrays = [sig_df.loc[sig_df.correct == 1, feat].dropna().to_numpy(),
                  sig_df.loc[sig_df.correct == 0, feat].dropna().to_numpy()]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(arrays, labels=["Correct", "Wrong"], showfliers=False)
        ax.set_ylabel(feat); ax.set_title(f"{feat}: Correct vs Wrong")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"33_{idx}_{feat}_correct_wrong.png"), dpi=300)
        plt.close(fig)
    return sig_df, effect


def plot_average_pair_signals(signals, df, pair_df, args, out_dir):
    pair_dir = ensure_dir(os.path.join(out_dir, "confusion_pair_signal_analysis"))
    max_pairs = min(args.top_confusion_pairs, len(pair_df))
    for rank in range(max_pairs):
        row = pair_df.iloc[rank]
        true_name, pred_name = row.true_name, row.pred_name
        m_wrong = (df.true_name == true_name) & (df.pred_name == pred_name)
        m_correct = (df.true_name == true_name) & (df.correct == 1)
        if m_wrong.sum() < 2 or m_correct.sum() < 2:
            continue
        xw = signals[m_wrong.to_numpy()]
        xc = signals[m_correct.to_numpy()]
        # 绝对值平均能减少相位抵消。
        mean_w = np.mean(np.abs(xw), axis=0)
        mean_c = np.mean(np.abs(xc), axis=0)
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(mean_c, label=f"Correct {true_name}")
        ax.plot(mean_w, label=f"{true_name} misclassified as {pred_name}")
        ax.set_xlabel("Sample point"); ax.set_ylabel("Mean absolute amplitude")
        ax.set_title(f"Waveform Envelope Comparison: {true_name} → {pred_name}")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(pair_dir, f"pair_{rank+1}_{true_name}_to_{pred_name}_waveform.png"), dpi=300)
        plt.close(fig)

        spec_c = np.mean(np.abs(np.fft.rfft(xc, axis=1)), axis=0)
        spec_w = np.mean(np.abs(np.fft.rfft(xw, axis=1)), axis=0)
        freqs = np.fft.rfftfreq(signals.shape[1], d=1.0 / args.sampling_rate)
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(freqs, spec_c, label=f"Correct {true_name}")
        ax.plot(freqs, spec_w, label=f"{true_name} misclassified as {pred_name}")
        ax.set_xlabel("Frequency" if args.sampling_rate != 1 else "Normalized frequency")
        ax.set_ylabel("Mean magnitude")
        ax.set_title(f"Spectrum Comparison: {true_name} → {pred_name}")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(pair_dir, f"pair_{rank+1}_{true_name}_to_{pred_name}_spectrum.png"), dpi=300)
        plt.close(fig)


def fused_prob_with_grad(trainer, x):
    feat_g = trainer.G(x)
    probs = []
    for i in range(trainer.num_source):
        feat_i = trainer.Fs[i](feat_g)
        probs.append(F.softmax(trainer.Cs[i](feat_i), dim=1))
    out = trainer._eval_class_weighted_fusion(probs) if hasattr(trainer, "_eval_class_weighted_fusion") else torch.stack(probs).mean(0)
    return out[0] if isinstance(out, tuple) else out


def input_gradient_saliency(trainer, signal, pred_id):
    x = torch.tensor(signal, dtype=torch.float32, device=trainer.device).view(1, 1, -1)
    x.requires_grad_(True)
    trainer.G.zero_grad(set_to_none=True)
    try:
        trainer.Fs.zero_grad(set_to_none=True); trainer.Cs.zero_grad(set_to_none=True)
    except Exception:
        pass
    prob = fused_prob_with_grad(trainer, x)
    score = torch.log(prob[0, int(pred_id)].clamp_min(EPS))
    score.backward()
    return np.abs(x.grad.detach().cpu().numpy().reshape(-1))


def integrated_gradients(trainer, signal, pred_id, steps=16):
    signal = np.asarray(signal, dtype=np.float32)
    baseline = np.zeros_like(signal)
    total_grad = np.zeros_like(signal, dtype=np.float64)
    for alpha in np.linspace(0.0, 1.0, steps, endpoint=True):
        current = baseline + alpha * (signal - baseline)
        grad = input_gradient_saliency(trainer, current, pred_id)
        total_grad += grad
    avg_grad = total_grad / max(steps, 1)
    return np.abs((signal - baseline) * avg_grad)


def representative_error_analysis(trainer, signals, df, args, out_dir):
    if args.skip_heavy:
        return
    error_dir = ensure_dir(os.path.join(out_dir, "representative_high_confidence_errors"))
    errors = df[df.correct == 0].sort_values("confidence", ascending=False).head(args.representative_errors)
    for rank, (_, row) in enumerate(errors.iterrows(), start=1):
        idx = int(row["index"])
        x = signals[idx]
        base = f"error_{rank}_idx{idx}_{row.true_name}_to_{row.pred_name}_conf{row.confidence:.4f}"

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(x); ax.set_xlabel("Sample point"); ax.set_ylabel("Amplitude")
        ax.set_title(f"High-confidence Error: true={row.true_name}, pred={row.pred_name}, conf={row.confidence:.4f}")
        fig.tight_layout(); fig.savefig(os.path.join(error_dir, base + "_waveform.png"), dpi=300)
        plt.close(fig)

        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), d=1.0 / args.sampling_rate)
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(freqs, spec)
        ax.set_xlabel("Frequency" if args.sampling_rate != 1 else "Normalized frequency")
        ax.set_ylabel("Magnitude"); ax.set_title("Magnitude Spectrum")
        fig.tight_layout(); fig.savefig(os.path.join(error_dir, base + "_spectrum.png"), dpi=300)
        plt.close(fig)

        if SCIPY_AVAILABLE:
            f, t, sxx = spectrogram(x, fs=args.sampling_rate, nperseg=min(128, len(x)), noverlap=min(96, len(x)//2))
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.pcolormesh(t, f, 10 * np.log10(sxx + EPS), shading="auto")
            ax.set_xlabel("Time"); ax.set_ylabel("Frequency")
            ax.set_title("Spectrogram of Misclassified Sample")
            fig.colorbar(im, ax=ax, label="Power (dB)")
            fig.tight_layout(); fig.savefig(os.path.join(error_dir, base + "_spectrogram.png"), dpi=300)
            plt.close(fig)

        try:
            sal = input_gradient_saliency(trainer, x, int(row.pred_id))
            ig = integrated_gradients(trainer, x, int(row.pred_id), args.integrated_gradients_steps)
            sal = sal / (sal.max() + EPS); ig = ig / (ig.max() + EPS)

            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(x, label="Signal")
            ax.plot(sal * (np.std(x) + EPS) + np.mean(x), alpha=0.7, label="Input-gradient saliency (scaled)")
            ax.set_xlabel("Sample point"); ax.set_title("Input-gradient Saliency")
            ax.legend(); fig.tight_layout()
            fig.savefig(os.path.join(error_dir, base + "_saliency.png"), dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(x, label="Signal")
            ax.plot(ig * (np.std(x) + EPS) + np.mean(x), alpha=0.7, label="Integrated gradients (scaled)")
            ax.set_xlabel("Sample point"); ax.set_title("Integrated Gradients")
            ax.legend(); fig.tight_layout()
            fig.savefig(os.path.join(error_dir, base + "_integrated_gradients.png"), dpi=300)
            plt.close(fig)
        except Exception as exc:
            logging.warning("Saliency failed for index %d: %s", idx, exc)


def parse_training_log(log_path, class_names, source_names, out_dir):
    if not log_path or not os.path.isfile(log_path):
        logging.warning("No valid --log_path; training-curve plots skipped.")
        return None

    epoch_rows = defaultdict(dict)
    class_f1_rows = defaultdict(dict)
    gate_rows = []
    current_epoch = None

    epoch_re = re.compile(r"Epoch\s+(\d+)/(\d+)")
    loss_re = re.compile(r"Train-Loss\s+(.+?):\s+([0-9eE+\-.]+)")
    acc_re = re.compile(r"Train-Acc\s+(.+?):\s+([0-9eE+\-.]+)")
    target_acc_re = re.compile(r"Target-Test-acc:\s+([0-9eE+\-.]+)")
    target_f1_re = re.compile(r"Target-Test-F1-macro:\s+([0-9eE+\-.]+)")
    class_f1_re = re.compile(r"Target-Test-Class-(\d+).*?F1:\s+([0-9eE+\-.]+)")
    lr_re = re.compile(r"current lr:\s*\[([0-9eE+\-.]+)")
    gate_re = re.compile(r"V6 stable gate: event=(\w+).*?gap=([0-9eE+\-.]+).*?raw=(.*)$")
    source_weight_re = re.compile(r"(PU_\d+|src\d+)\(src\d+\)=([0-9eE+\-.]+)")
    source_line_patterns = {
        "raw": "V6-Lite raw global source weights:",
        "effective": "V6-Lite effective global source weights:",
        "final": "V6-Lite final class-averaged source weights:",
    }

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = epoch_re.search(line)
            if m:
                current_epoch = int(m.group(1))
                epoch_rows[current_epoch]["epoch"] = current_epoch
                continue
            if current_epoch is None:
                continue
            m = lr_re.search(line)
            if m:
                epoch_rows[current_epoch]["lr"] = float(m.group(1))
            m = loss_re.search(line)
            if m:
                key = "loss_" + re.sub(r"[^A-Za-z0-9]+", "_", m.group(1)).strip("_")
                epoch_rows[current_epoch][key] = float(m.group(2))
            m = acc_re.search(line)
            if m:
                key = "train_acc_" + re.sub(r"[^A-Za-z0-9]+", "_", m.group(1)).strip("_")
                epoch_rows[current_epoch][key] = float(m.group(2))
            m = target_acc_re.search(line)
            if m:
                epoch_rows[current_epoch]["target_acc"] = float(m.group(1))
            m = target_f1_re.search(line)
            if m:
                epoch_rows[current_epoch]["target_macro_f1"] = float(m.group(1))
            m = class_f1_re.search(line)
            if m:
                class_f1_rows[current_epoch][int(m.group(1))] = float(m.group(2))
            for prefix, marker in source_line_patterns.items():
                if marker in line:
                    matches = source_weight_re.findall(line)
                    for src, value in matches:
                        epoch_rows[current_epoch][f"{prefix}_{src}"] = float(value)
            if "BiMamba-Att residual gate:" in line:
                try:
                    epoch_rows[current_epoch]["bimamba_gate"] = float(line.rsplit(":", 1)[1].strip())
                except Exception:
                    pass
            m = gate_re.search(line)
            if m:
                gate_rows.append({"epoch": current_epoch, "event": m.group(1),
                                  "gap": float(m.group(2)), "raw_text": m.group(3).strip()})

    hist = pd.DataFrame([epoch_rows[k] for k in sorted(epoch_rows)]).sort_values("epoch")
    hist.to_csv(os.path.join(out_dir, "training_history.csv"), index=False,
                encoding="utf-8-sig")
    pd.DataFrame(gate_rows).to_csv(os.path.join(out_dir, "gate_events.csv"), index=False,
                                   encoding="utf-8-sig")

    # Accuracy/F1
    fig, ax = plt.subplots(figsize=(11, 6))
    if "target_acc" in hist: ax.plot(hist.epoch, hist.target_acc, marker="o", label="Target accuracy")
    if "target_macro_f1" in hist: ax.plot(hist.epoch, hist.target_macro_f1, marker="o", label="Target Macro-F1")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_ylim(0, 1.02)
    ax.set_title("Target-test Accuracy and Macro-F1")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(out_dir, "34_training_target_metrics.png"), dpi=300)
    plt.close(fig)

    # Loss curves: plot available effective/weighted terms.
    preferred = [
        "loss_Source_Classifier", "loss_MMD", "loss_CDAN_Weighted",
        "loss_CLMMD_Weighted", "loss_SupCon_Weighted", "loss_Total",
    ]
    cols = [c for c in preferred if c in hist.columns]
    if cols:
        fig, ax = plt.subplots(figsize=(12, 7))
        for col in cols:
            ax.plot(hist.epoch, hist[col], marker="o", label=col.replace("loss_", ""))
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Training Loss Components")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "35_training_loss_components.png"), dpi=300)
        plt.close(fig)

    # Source weights raw/effective/final.
    for index, prefix in enumerate(("raw", "effective", "final"), start=36):
        cols = [c for c in hist.columns if c.startswith(prefix + "_")]
        if cols:
            fig, ax = plt.subplots(figsize=(11, 6))
            for col in cols:
                ax.plot(hist.epoch, hist[col], marker="o", label=col[len(prefix)+1:])
            ax.set_xlabel("Epoch"); ax.set_ylabel("Weight")
            ax.set_ylim(0, 1.0); ax.set_title(f"{prefix.capitalize()} Source Weights over Epochs")
            ax.legend(); fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{index}_{prefix}_source_weights.png"), dpi=300)
            plt.close(fig)

    # Per-class F1 heatmap.
    if class_f1_rows:
        mat = np.full((len(class_names), len(hist)), np.nan)
        epochs = hist.epoch.astype(int).tolist()
        for j, ep in enumerate(epochs):
            for c, value in class_f1_rows.get(ep, {}).items():
                if c < len(class_names): mat[c, j] = value
        plot_matrix(np.nan_to_num(mat), [str(x) for x in epochs], class_names,
                    "Per-class F1 over Epochs", "Epoch", "Class",
                    os.path.join(out_dir, "39_per_class_f1_over_epochs.png"),
                    text_fmt=None, figsize=(14, 8))
        save_matrix_csv(mat, class_names, [str(x) for x in epochs],
                        os.path.join(out_dir, "per_class_f1_over_epochs.csv"))

    return hist


def write_auto_report(df, metrics, pair_df, centroid_dist, effect, ece, out_dir):
    lines = []
    lines.append("V6-Lite 自动错分诊断摘要")
    lines.append("=" * 70)
    lines.append(f"样本数：{len(df)}")
    lines.append(f"准确率：{df.correct.mean():.6f}")
    lines.append(f"错误样本数：{int((df.correct == 0).sum())}")
    lines.append(f"ECE：{ece:.6f}")
    lines.append("")

    worst = metrics.sort_values("f1").head(3)
    lines.append("F1最低类别：")
    for _, r in worst.iterrows():
        lines.append(f"- {r.class_name}: P={r.precision:.4f}, R={r.recall:.4f}, F1={r.f1:.4f}")

    lines.append("")
    lines.append("主要错分流向：")
    for _, r in pair_df.head(8).iterrows():
        lines.append(f"- {r.true_name} → {r.pred_name}: {int(r['count'])} samples, "
                     f"占该真实类别 {r.true_class_error_share:.2%}")

    wrong = df[df.correct == 0]
    if len(wrong):
        high_conf = (wrong.confidence >= 0.90).mean()
        lines.append("")
        lines.append(f"错误中置信度≥0.90的比例：{high_conf:.2%}")
        lines.append(f"正确样本平均置信度：{df.loc[df.correct == 1, 'confidence'].mean():.4f}")
        lines.append(f"错误样本平均置信度：{wrong.confidence.mean():.4f}")
        lines.append(f"正确样本平均分支预测种类数：{df.loc[df.correct == 1, 'branch_unique_prediction_count'].mean():.4f}")
        lines.append(f"错误样本平均分支预测种类数：{wrong.branch_unique_prediction_count.mean():.4f}")

    # Closest centroid pairs excluding diagonal.
    d = centroid_dist.copy(); np.fill_diagonal(d, np.inf)
    flat = np.argsort(d, axis=None)[:6]
    lines.append("")
    lines.append("特征中心距离最近的类别对（潜在重叠）：")
    used = set()
    names = metrics.class_name.tolist()
    for idx in flat:
        i, j = np.unravel_index(idx, d.shape)
        key = tuple(sorted((i, j)))
        if key in used: continue
        used.add(key)
        lines.append(f"- {names[i]} ↔ {names[j]}: distance={d[i, j]:.4f}")
        if len(used) >= 3: break

    if effect is not None:
        lines.append("")
        lines.append("信号统计效应量绝对值最大的类别/特征组合，提示错分样本与正确样本存在分布差异；")
        lines.append("需结合对应波形、频谱和工况信息解释，不能直接视为物理因果。")

    lines.append("")
    lines.append("判断逻辑：")
    lines.append("1. 混淆矩阵确定错分方向；")
    lines.append("2. 置信度/熵/校准判断是边界不确定还是高置信系统性错误；")
    lines.append("3. 分支一致性与融合权重判断是否由某个源域分支主导；")
    lines.append("4. PCA/t-SNE/中心距离判断类别特征是否重叠；")
    lines.append("5. 时间顺序图判断目标域后段是否存在漂移；")
    lines.append("6. 波形/频谱/信号统计和显著性图定位输入层面的差异。")

    with open(os.path.join(out_dir, "AUTO_DIAGNOSIS.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    setup_logging(args.output_dir)
    class_names = parse_class_names(args)
    args = prepare_label_sets(args, class_names)

    logging.info("Model: %s", args.model_name)
    logging.info("Checkpoint: %s", args.ckpt_path)
    logging.info("Classes: %s", class_names)
    logging.info("Output: %s", args.output_dir)
    if not args.ckpt_path.endswith("_best.pth"):
        logging.warning("You are not loading *_best.pth; figures may not match the best epoch.")

    trainer = build_trainer(args)
    ckpt = load_checkpoint(trainer, args.ckpt_path)
    data = collect_target_diagnostics(trainer, args)

    df = build_detailed_dataframe(data, class_names, args.source_name)
    df.to_csv(os.path.join(args.output_dir, "predictions_detailed.csv"),
              index=False, encoding="utf-8-sig")

    metrics, cm, cm_true, pair_df, summary = basic_metrics_and_plots(
        df, data["fused_probs"], class_names, args.output_dir
    )
    ece = calibration_and_uncertainty(df, class_names, args, args.output_dir)
    branch_and_fusion_analysis(df, data, class_names, args.source_name, args.output_dir)
    centroid_dist = feature_space_analysis(
        df, data["features"], class_names, args, args.output_dir
    )
    temporal_analysis(df, class_names, args, args.output_dir)
    sig_df, effect = signal_statistical_analysis(
        data["signals"], df, class_names, args.output_dir, args.sampling_rate
    )

    if not args.skip_heavy:
        plot_average_pair_signals(data["signals"], df, pair_df, args, args.output_dir)
        representative_error_analysis(trainer, data["signals"], df, args, args.output_dir)

    parse_training_log(args.log_path, class_names, args.source_name, args.output_dir)
    write_auto_report(df, metrics, pair_df, centroid_dist, effect, ece, args.output_dir)

    if args.save_raw_npz:
        np.savez_compressed(
            os.path.join(args.output_dir, "raw_diagnostics.npz"),
            labels=data["labels"], preds=data["preds"],
            fused_probs=data["fused_probs"], features=data["features"],
            branch_probs=data["branch_probs"], branch_preds=data["branch_preds"],
            fusion_weights=data["fusion_weights"],
        )

    logging.info("All diagnostics completed.")
    logging.info("Main report: %s", os.path.join(args.output_dir, "AUTO_DIAGNOSIS.txt"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
