# -*- coding: utf-8 -*-
"""
plot_saved_results.py

功能：
1. 加载训练好的 MFSAN_CDA / MFSAN_CDAN 权重；
2. 在目标域验证集上重新推理；
3. 输出：
   - 混淆矩阵：confusion_matrix_counts.png
   - 归一化混淆矩阵：confusion_matrix_normalized.png
   - 每类 Precision / Recall / F1 柱状图：precision_recall_f1_bar.png
   - 目标域 t-SNE 类别分布图：tsne_target_by_class.png
   - 源域-目标域 t-SNE 域分布图：tsne_domain_distribution.png
   - 指标 CSV：metrics_report.csv
   - 预测结果 CSV：predictions.csv

推荐运行示例：


python plot_saved_results.py \
  --model_name MFSAN_CDAN_BIMAMBA_CW_RWCA_V4_SUPCON \
  --source PU_1,PU_2,PU_3 \
  --target PU_0 \
  --train_mode multi_source \
  --data_dir /workspace/PU_TL_9_replace \
  --signal_size 1024 \
  --backbone CNN \
  --cuda_device 0 \
  --ckpt_path "/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels_bimamba_dongtai_new_9labels_CW_RWCA_v4/ckpt/MFSAN_CDAN_BIMAMBA_CW_RWCA_V4_SUPCON/multi_source/[PU_1_PU_2_PU_3]To[PU_0]_0512-055536_best.pth" \
  --include_faults K001,KA04,KA16,KA30,KB23,KB24,KI04,KI16,KI17 \
  --batch_size 64 \
  --num_workers 4 \
  --feature_mode F_mean \
  --output_dir ./plot_results/V4_SUPCON_PU123_to_PU0_best_fixed_classes
"""

import os
import sys
import csv
import argparse
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
from sklearn.manifold import TSNE


# =========================================================
# 路径设置：保证可以 import models/MFSAN_CDA.py 里的 modules
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "models"))
sys.path.append(os.path.join(PROJECT_ROOT, "data_loader"))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot saved MFSAN/MFSAN-CDA/MFSAN-CDAN results.")

    # ========== 基本实验参数 ==========
    parser.add_argument("--model_name", type=str, default="MFSAN_CDA",
                        help="Model file name under models/, e.g. MFSAN_CDA or MFSAN_CDAN.")
    parser.add_argument("--source", type=str, default="PU_0,PU_2,PU_3")
    parser.add_argument("--target", type=str, default="PU_1")
    parser.add_argument("--train_mode", type=str, default="multi_source")
    parser.add_argument("--data_dir", type=str, default="/workspace/PU_TL")
    parser.add_argument("--signal_size", type=int, default=1024)
    parser.add_argument("--backbone", type=str, default="CNN")
    parser.add_argument("--cuda_device", type=str, default="0")

    # ========== 权重和输出 ==========
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to saved .pth checkpoint."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./plot_results",
        help="Directory to save figures and CSV files."
    )

    # ========== 9类筛选 ==========
    parser.add_argument(
        "--include_faults",
        type=str,
        default="K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18",
        help="Comma-separated class names to keep."
    )
    parser.add_argument(
        "--exclude_faults",
        type=str,
        default="",
        help="Comma-separated class names to remove."
    )

    # ========== DataLoader 参数 ==========
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--normlize_type", type=str, default="-1-1")
    parser.add_argument("--random_state", type=int, default=10)

    # ========== 模型训练参数：实例化 Trainer 时需要 ==========
    parser.add_argument("--opt", type=str, default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_scheduler", type=str, default="stepLR")
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--steps", type=str, default="10")
    parser.add_argument("--tradeoff", type=str, default="exp,exp,exp")
    parser.add_argument("--zeta", type=float, default=10.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max_epoch", type=int, default=2)

    # ========== CDA 参数 ==========
    parser.add_argument("--lambda_cda", type=float, default=0.02)
    parser.add_argument("--lambda_ent", type=float, default=0.005)
    parser.add_argument("--cda_detach_prob", type=str2bool, default=True)

    # ========== CDAN 参数：如果 model_name=MFSAN_CDAN 会用到 ==========
    parser.add_argument("--lambda_adv", type=float, default=0.02)
    parser.add_argument("--lambda_grl", type=float, default=1.0)
    parser.add_argument("--adv_hidden_dim", type=int, default=256)
    parser.add_argument("--adv_detach_prob", type=str2bool, default=True)
    parser.add_argument("--adv_use_entropy_weight", type=str2bool, default=True)
    parser.add_argument("--adv_conf_thresh", type=float, default=0.0)

    # ========== t-SNE 参数 ==========
    parser.add_argument("--feature_mode", type=str, default="F_mean",
                        choices=["G", "F_mean"],
                        help="Feature used for t-SNE: G is shared backbone feature; F_mean is averaged source-specific feature.")
    parser.add_argument("--tsne_max_per_class", type=int, default=300,
                        help="Max target samples per class used for target-class t-SNE.")
    parser.add_argument("--tsne_max_per_domain", type=int, default=1000,
                        help="Max samples per domain used for source-target domain t-SNE.")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=1000)

    args = parser.parse_args()

    # train.py 里一般会做这些转换，这里手动补上
    args.source_name = [x.strip() for x in args.source.split(",") if x.strip()]
    args.tradeoff = [x.strip() for x in args.tradeoff.split(",") if x.strip()]
    args.betas = tuple(args.betas)

    # 绘图脚本不训练，但 Trainer 初始化时需要这些字段
    args.save = False
    args.save_dir = "./ckpt"
    args.save_path = os.path.join(args.output_dir, "dummy_save_path")
    args.load_path = args.ckpt_path
    args.da_scenario = "closed-set"

    return args


def parse_class_names(args):
    if args.include_faults.strip():
        class_names = [x.strip() for x in args.include_faults.split(",") if x.strip()]
    else:
        # 如果不显式指定 include_faults，则从目标域目录扫描
        target_dir = os.path.join(args.data_dir, args.target)
        class_names = sorted([
            d for d in os.listdir(target_dir)
            if os.path.isdir(os.path.join(target_dir, d))
        ])

    if args.exclude_faults.strip():
        exclude = set([x.strip() for x in args.exclude_faults.split(",") if x.strip()])
        class_names = [x for x in class_names if x not in exclude]

    return class_names


def prepare_label_sets(args, class_names):
    """
    为画图脚本补齐 train.py 中原本会生成的字段。

    conditional_load.py 需要：
        args.faults[source_idx]
        args.fault_label

    TrainerBase / 标签映射需要：
        args.label_sets
    """
    num_classes = len(class_names)

    # 类别名称 -> 数字标签
    # 例如：
    # K001 -> 0
    # KA04 -> 1
    # KA16 -> 2
    # ...
    fault_label = {fault: idx for idx, fault in enumerate(class_names)}

    # 数字标签集合
    label_set = list(range(num_classes))

    # 每个 source + target 一个 label_set
    # 例如 3 个源域 + 1 个目标域：
    # args.label_sets = [[0..8], [0..8], [0..8], [0..8]]
    args.label_sets = [label_set[:] for _ in args.source_name] + [label_set[:]]

    # conditional_load.py 会读取 args.faults[source_idx]
    # 这里每个源域和目标域都使用同样的 9 类
    args.faults = [class_names[:] for _ in args.source_name] + [class_names[:]]

    # 关键新增：conditional_load.py 会读取 args.fault_label
    args.fault_label = fault_label

    # 一些兼容字段，防止其他地方使用
    args.class_names = class_names
    args.selected_faults = class_names
    args.fault_names = class_names
    args.num_classes = num_classes

    return args


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "plot_log.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )


def build_trainer(args):
    import importlib

    model_module = importlib.import_module(f"models.{args.model_name}")
    trainer = model_module.Trainer(args)

    return trainer


def load_checkpoint_to_trainer(trainer, ckpt_path, device):
    logging.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "G" in ckpt:
        trainer.G.load_state_dict(ckpt["G"])
    else:
        raise KeyError("Checkpoint missing key: G")

    if "Fs" in ckpt:
        trainer.Fs.load_state_dict(ckpt["Fs"])
    else:
        raise KeyError("Checkpoint missing key: Fs")

    if "Cs" in ckpt:
        trainer.Cs.load_state_dict(ckpt["Cs"])
    else:
        raise KeyError("Checkpoint missing key: Cs")

    # CDAN 模型会有 Ds；CDA 模型没有 Ds
    if hasattr(trainer, "Ds"):
        if "Ds" in ckpt:
            trainer.Ds.load_state_dict(ckpt["Ds"])
            logging.info("Loaded domain discriminator weights: Ds")
        else:
            logging.warning("Trainer has Ds but checkpoint has no Ds. This is okay only if you are not using CDAN weights.")

    logging.info("Checkpoint loaded successfully.")


@torch.no_grad()
def forward_target_batch(trainer, data, feature_mode="F_mean"):
    """
    对目标域 batch 进行前向推理：
    - 返回融合预测概率
    - 返回预测标签
    - 返回用于 t-SNE 的特征
    """
    feat_g = trainer.G(data)

    feat_list = []
    prob_list = []

    for i in range(trainer.num_source):
        feat_i = trainer.Fs[i](feat_g)
        logit_i = trainer.Cs[i](feat_i)
        prob_i = F.softmax(logit_i, dim=1)

        feat_list.append(feat_i)
        prob_list.append(prob_i)

    probs = torch.stack(prob_list, dim=0).mean(dim=0)
    pred_train_label = probs.argmax(dim=1)

    # 映射回 actual label，一般仍然是 0~C-1
    pred_actual_label = trainer._get_actual_label(
        pred_train_label,
        label_set=trainer.src_labels_flat
    )

    if feature_mode == "G":
        feat_plot = feat_g
    else:
        feat_plot = torch.stack(feat_list, dim=0).mean(dim=0)

    return probs, pred_actual_label, feat_plot


@torch.no_grad()
def collect_target_predictions(trainer, args):
    """
    只在目标域 val 上收集：
    - 真实标签
    - 预测标签
    - 预测概率
    - 特征
    """
    trainer._set_to_eval()

    all_labels = []
    all_preds = []
    all_probs = []
    all_features = []

    val_loader = trainer.dataloaders["val"]

    for data, actual_labels in val_loader:
        data = data.to(trainer.device)

        probs, preds, features = forward_target_batch(
            trainer,
            data,
            feature_mode=args.feature_mode
        )

        all_labels.append(actual_labels.cpu())
        all_preds.append(preds.detach().cpu())
        all_probs.append(probs.detach().cpu())
        all_features.append(features.detach().cpu())

    labels = torch.cat(all_labels, dim=0).numpy()
    preds = torch.cat(all_preds, dim=0).numpy()
    probs = torch.cat(all_probs, dim=0).numpy()
    features = torch.cat(all_features, dim=0).numpy()

    return labels, preds, probs, features


@torch.no_grad()
def collect_domain_features(trainer, args, domain_key, max_samples):
    """
    用于源域/目标域的 t-SNE 域分布图。
    domain_key:
    - source: PU_0 / PU_1 / PU_2 / PU_3
    - target validation: 'val'
    """
    trainer._set_to_eval()

    loader = trainer.dataloaders[domain_key]

    feat_list = []
    label_list = []
    count = 0

    for data, actual_labels in loader:
        data = data.to(trainer.device)

        _, _, features = forward_target_batch(
            trainer,
            data,
            feature_mode=args.feature_mode
        )

        feat_list.append(features.detach().cpu())
        label_list.append(actual_labels.detach().cpu())

        count += data.size(0)
        if count >= max_samples:
            break

    features = torch.cat(feat_list, dim=0)[:max_samples].numpy()
    labels = torch.cat(label_list, dim=0)[:max_samples].numpy()

    return features, labels


def save_predictions_csv(labels, preds, probs, class_names, output_dir):
    path = os.path.join(output_dir, "predictions.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["index", "true_id", "true_name", "pred_id", "pred_name"]
        header += [f"prob_{name}" for name in class_names]
        writer.writerow(header)

        for i, (y, p) in enumerate(zip(labels, preds)):
            y = int(y)
            p = int(p)
            row = [
                i,
                y,
                class_names[y] if 0 <= y < len(class_names) else str(y),
                p,
                class_names[p] if 0 <= p < len(class_names) else str(p),
            ]
            row += probs[i].tolist()
            writer.writerow(row)

    logging.info(f"Saved predictions CSV: {path}")


def save_metrics_csv(labels, preds, class_names, output_dir):
    num_classes = len(class_names)
    label_ids = list(range(num_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=label_ids,
        zero_division=0
    )

    acc = accuracy_score(labels, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, labels=label_ids, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, labels=label_ids, average="weighted", zero_division=0
    )

    path = os.path.join(output_dir, "metrics_report.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "precision", "recall", "f1", "support"])

        for i in range(num_classes):
            writer.writerow([
                i,
                class_names[i],
                f"{precision[i]:.6f}",
                f"{recall[i]:.6f}",
                f"{f1[i]:.6f}",
                int(support[i]),
            ])

        writer.writerow([])
        writer.writerow(["overall", "accuracy", f"{acc:.6f}"])
        writer.writerow(["overall", "macro_precision", f"{macro_p:.6f}"])
        writer.writerow(["overall", "macro_recall", f"{macro_r:.6f}"])
        writer.writerow(["overall", "macro_f1", f"{macro_f1:.6f}"])
        writer.writerow(["overall", "weighted_precision", f"{weighted_p:.6f}"])
        writer.writerow(["overall", "weighted_recall", f"{weighted_r:.6f}"])
        writer.writerow(["overall", "weighted_f1", f"{weighted_f1:.6f}"])

    logging.info(f"Saved metrics CSV: {path}")

    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Macro-F1: {macro_f1:.4f}")
    logging.info(f"Weighted-F1: {weighted_f1:.4f}")

    return precision, recall, f1, support


def plot_confusion_matrices(labels, preds, class_names, output_dir):
    num_classes = len(class_names)
    label_ids = list(range(num_classes))

    cm = confusion_matrix(labels, preds, labels=label_ids)

    # 1. 原始计数混淆矩阵
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix_counts.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved confusion matrix: {path}")

    # 2. 归一化混淆矩阵
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm)

    ax.set_title("Normalized Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix_normalized.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved normalized confusion matrix: {path}")


def plot_prf_bar(precision, recall, f1, class_names, output_dir):
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1-score")

    ax.set_title("Precision / Recall / F1-score per Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()

    path = os.path.join(output_dir, "precision_recall_f1_bar.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved Precision/Recall/F1 bar chart: {path}")


def stratified_sample_indices(labels, max_per_class):
    selected = []
    labels = np.asarray(labels)

    for cls in sorted(np.unique(labels)):
        idx = np.where(labels == cls)[0]
        if len(idx) > max_per_class:
            idx = np.random.choice(idx, size=max_per_class, replace=False)
        selected.extend(idx.tolist())

    selected = np.array(selected)
    np.random.shuffle(selected)
    return selected


def plot_tsne_target_by_class(features, labels, class_names, args):
    output_dir = args.output_dir

    idx = stratified_sample_indices(labels, args.tsne_max_per_class)

    x = features[idx]
    y = labels[idx]

    logging.info(f"Running t-SNE for target class distribution, samples={len(idx)}")

    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        max_iter=args.tsne_iter,
        init="pca",
        learning_rate="auto",
        random_state=args.random_state,
    )

    emb = tsne.fit_transform(x)

    fig, ax = plt.subplots(figsize=(10, 8))

    for cls in sorted(np.unique(y)):
        cls_idx = y == cls
        name = class_names[int(cls)] if int(cls) < len(class_names) else str(cls)
        ax.scatter(
            emb[cls_idx, 0],
            emb[cls_idx, 1],
            s=10,
            alpha=0.75,
            label=name
        )

    ax.set_title("t-SNE Feature Distribution on Target Domain by Class")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(markerscale=2, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    path = os.path.join(output_dir, "tsne_target_by_class.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved target t-SNE by class: {path}")


def plot_tsne_domain_distribution(trainer, args):
    """
    源域 + 目标域的域分布 t-SNE。
    用来展示迁移学习里的源域/目标域特征是否混合。
    """
    domain_features = []
    domain_labels = []

    domain_keys = args.source_name + ["val"]
    domain_names = args.source_name + [args.target + "_val"]

    for domain_id, domain_key in enumerate(domain_keys):
        max_samples = args.tsne_max_per_domain

        logging.info(f"Collecting features for domain t-SNE: {domain_key}")

        feats, _ = collect_domain_features(
            trainer,
            args,
            domain_key=domain_key,
            max_samples=max_samples
        )

        domain_features.append(feats)
        domain_labels.append(np.full(feats.shape[0], domain_id))

    x = np.concatenate(domain_features, axis=0)
    y = np.concatenate(domain_labels, axis=0)

    logging.info(f"Running t-SNE for domain distribution, samples={x.shape[0]}")

    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        max_iter=args.tsne_iter,
        init="pca",
        learning_rate="auto",
        random_state=args.random_state,
    )

    emb = tsne.fit_transform(x)

    fig, ax = plt.subplots(figsize=(10, 8))

    for domain_id, domain_name in enumerate(domain_names):
        idx = y == domain_id
        ax.scatter(
            emb[idx, 0],
            emb[idx, 1],
            s=10,
            alpha=0.65,
            label=domain_name
        )

    ax.set_title("t-SNE Feature Distribution across Source and Target Domains")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(markerscale=2, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    path = os.path.join(args.output_dir, "tsne_domain_distribution.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved domain t-SNE: {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    logging.info("=" * 80)
    logging.info("Plot saved results")
    logging.info("=" * 80)

    class_names = parse_class_names(args)
    args = prepare_label_sets(args, class_names)

    logging.info(f"Model: {args.model_name}")
    logging.info(f"Source: {args.source_name}")
    logging.info(f"Target: {args.target}")
    logging.info(f"Classes ({len(class_names)}): {class_names}")
    logging.info(f"Checkpoint: {args.ckpt_path}")
    logging.info(f"Output dir: {args.output_dir}")

    trainer = build_trainer(args)
    load_checkpoint_to_trainer(trainer, args.ckpt_path, trainer.device)

    labels, preds, probs, features = collect_target_predictions(trainer, args)

    # 保存预测结果
    save_predictions_csv(labels, preds, probs, class_names, args.output_dir)

    # 保存指标
    precision, recall, f1, support = save_metrics_csv(
        labels,
        preds,
        class_names,
        args.output_dir
    )

    # 混淆矩阵
    plot_confusion_matrices(labels, preds, class_names, args.output_dir)

    # Precision / Recall / F1 柱状图
    plot_prf_bar(precision, recall, f1, class_names, args.output_dir)

    # t-SNE：目标域按类别
    plot_tsne_target_by_class(features, labels, class_names, args)

    # t-SNE：源域和目标域按域分布
    plot_tsne_domain_distribution(trainer, args)

    logging.info("=" * 80)
    logging.info("All figures and CSV files are saved.")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()