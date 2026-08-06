import os
import sys
import csv
import argparse
import logging

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
# 路径设置：保证可以 import models / data_loader
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
    parser = argparse.ArgumentParser(
        description="Plot saved V5_MCA results: confusion matrix, per-class metrics, t-SNE, reliability weights."
    )

    # ========== 基本实验参数 ==========
    parser.add_argument(
        "--model_name",
        type=str,
        default="MFSAN_CDAN_BIMAMBA_CW_RWCA_V5_MCA",
        help="Model file name under models/.",
    )
    parser.add_argument("--source", type=str, default="PU_1,PU_2,PU_3")
    parser.add_argument("--target", type=str, default="PU_0")
    parser.add_argument("--train_mode", type=str, default="multi_source")
    parser.add_argument("--data_dir", type=str, default="/workspace/PU_TL_9_replace")
    parser.add_argument("--signal_size", type=int, default=1024)
    parser.add_argument("--backbone", type=str, default="CNN")
    parser.add_argument("--cuda_device", type=str, default="0")

    # ========== 权重和输出 ==========
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to saved .pth checkpoint. 推荐使用 xxx_best.pth。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./plot_results/V4_SUPCON_PU123_to_PU0_best_training_fusion",
        help="Directory to save figures and CSV files.",
    )

    # ========== 9类筛选 ==========
    parser.add_argument(
        "--include_faults",
        type=str,
        default="K001,KA04,KA16,KA30,KB23,KB24,KI04,KI16,KI17",
        help="Comma-separated class names to keep, or empty to scan target folder.",
    )
    parser.add_argument(
        "--exclude_faults",
        type=str,
        default="",
        help="Comma-separated class names to remove.",
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

    # ========== CDA / CDAN 参数：默认对齐训练日志 ==========
    parser.add_argument("--lambda_cda", type=float, default=0.0)
    parser.add_argument("--lambda_ent", type=float, default=0.005)
    parser.add_argument("--cda_detach_prob", type=str2bool, default=True)

    parser.add_argument("--lambda_adv", type=float, default=0.02)
    parser.add_argument("--lambda_grl", type=float, default=1.0)
    parser.add_argument("--adv_hidden_dim", type=int, default=256)
    parser.add_argument("--adv_detach_prob", type=str2bool, default=True)
    parser.add_argument("--adv_use_entropy_weight", type=str2bool, default=True)
    parser.add_argument("--adv_conf_thresh", type=float, default=0.0)

    # ========== BiMamba-SmallGate 参数 ==========
    parser.add_argument("--bla_gate_init", type=float, default=0.01)
    parser.add_argument("--bla_gate_max", type=float, default=0.03)
    parser.add_argument("--bimamba_stem_channels", type=int, default=64)
    parser.add_argument("--bimamba_dim", type=int, default=64)
    parser.add_argument("--bimamba_depth", type=int, default=2)
    parser.add_argument("--bimamba_d_state", type=int, default=16)
    parser.add_argument("--bimamba_d_conv", type=int, default=4)
    parser.add_argument("--bimamba_expand", type=int, default=2)
    parser.add_argument("--bimamba_gate_init", type=float, default=0.01)
    parser.add_argument("--bimamba_gate_max", type=float, default=0.03)

    # ========== V2/V3/V4 RWCA / CW-RWCA 参数 ==========
    parser.add_argument("--rw_tau", type=float, default=0.5)
    parser.add_argument("--rw_mmd_weight", type=float, default=1.0)
    parser.add_argument("--rw_ent_weight", type=float, default=1.0)
    parser.add_argument("--rw_detach_weights", type=str2bool, default=True)
    parser.add_argument("--rw_ema_momentum", type=float, default=0.9)
    parser.add_argument("--rw_eval_use_entropy", type=str2bool, default=True)
    parser.add_argument("--rw_eval_tau", type=float, default=0.5)

    parser.add_argument("--lambda_clmmd", type=float, default=0.005)
    parser.add_argument("--clmmd_kernel_num", type=int, default=5)
    parser.add_argument("--clmmd_kernel_mul", type=float, default=2.0)
    parser.add_argument("--clmmd_fix_sigma", type=float, default=None)
    parser.add_argument("--clmmd_min_source", type=int, default=2)
    parser.add_argument("--clmmd_min_target_weight", type=float, default=0.001)

    parser.add_argument("--cw_warmup_epochs", type=int, default=3)
    parser.add_argument("--cw_alpha", type=float, default=0.30)
    parser.add_argument("--cw_alpha_ramp_epochs", type=int, default=3)

    # V4 伪标签置信度门控
    parser.add_argument("--pl_conf_thresh", type=float, default=0.80)
    parser.add_argument("--pl_min_target", type=int, default=2)

    # V4_SUPCON 参数
    parser.add_argument("--lambda_supcon", type=float, default=0.02)
    parser.add_argument("--supcon_temperature", type=float, default=0.10)
    parser.add_argument("--supcon_start_epoch", type=int, default=1)
    parser.add_argument("--supcon_feature_mode", type=str, default="G", choices=["G", "F"])
    parser.add_argument("--supcon_focus_classes", type=str, default="1,2")

    # ========== V5 参数：MDIFN source per-class recognition + MSD-MCA multi-classifier alignment ==========
    parser.add_argument("--rec_score_weight", type=float, default=0.30)
    parser.add_argument("--rec_score_mode", type=str, default="prob", choices=["prob", "acc", "mix"])
    parser.add_argument("--rec_score_detach", type=str2bool, default=True)

    parser.add_argument("--lambda_mca", type=float, default=0.02)
    parser.add_argument("--mca_start_epoch", type=int, default=1)
    parser.add_argument("--mca_use_reliability", type=str2bool, default=True)
    parser.add_argument("--mca_detach_fused", type=str2bool, default=True)
    parser.add_argument("--mca_eps", type=float, default=1e-5)

    # ========== t-SNE 参数 ==========
    parser.add_argument(
        "--feature_mode",
        type=str,
        default="F_mean",
        choices=["G", "F_mean"],
        help="Feature used for t-SNE: G is shared backbone feature; F_mean is averaged source-specific feature.",
    )
    parser.add_argument(
        "--tsne_max_per_class",
        type=int,
        default=300,
        help="Max target samples per class used for target-class t-SNE.",
    )
    parser.add_argument(
        "--tsne_max_per_domain",
        type=int,
        default=1000,
        help="Max samples per domain used for source-target domain t-SNE.",
    )
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=1000)

    args = parser.parse_args()

    # train.py 里一般会做这些转换，这里手动补上
    args.source_name = [x.strip() for x in args.source.split(",") if x.strip()]
    args.tradeoff = [x.strip() for x in args.tradeoff.split(",") if x.strip()]
    args.betas = tuple(args.betas)

    # 绘图脚本不训练，但 Trainer 初始化时需要这些字段
    args.save = False
    args.save_best = False
    args.save_dir = "./ckpt"
    args.save_path = os.path.join(args.output_dir, "dummy_save_path")
    args.load_path = args.ckpt_path
    args.da_scenario = "closed-set"

    return args


def parse_class_names(args):
    if args.include_faults.strip():
        class_names = [x.strip() for x in args.include_faults.split(",") if x.strip()]
    else:
        target_dir = os.path.join(args.data_dir, args.target)
        class_names = sorted([
            d for d in os.listdir(target_dir)
            if os.path.isdir(os.path.join(target_dir, d))
        ])

    if args.exclude_faults.strip():
        exclude = set([x.strip() for x in args.exclude_faults.split(",") if x.strip()])
        class_names = [x for x in class_names if x not in exclude]

    if len(class_names) == 0:
        raise ValueError("No class names found. 请检查 --include_faults 或数据目录。")

    return class_names


def prepare_label_sets(args, class_names):
    """
    为画图脚本补齐 train.py 中原本会生成的字段。
    """
    num_classes = len(class_names)
    fault_label = {fault: idx for idx, fault in enumerate(class_names)}
    label_set = list(range(num_classes))

    args.label_sets = [label_set[:] for _ in args.source_name] + [label_set[:]]
    args.faults = [class_names[:] for _ in args.source_name] + [class_names[:]]
    args.fault_label = fault_label

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
        ],
    )


def build_trainer(args):
    import importlib

    model_module = importlib.import_module(f"models.{args.model_name}")
    trainer = model_module.Trainer(args)
    return trainer


def _manual_load_checkpoint_to_trainer(trainer, ckpt_path, device):
    """
    兜底加载：如果 trainer.load_model() 在某些老模型上不可用，就手动加载网络权重。
    但对 V3/V4/V4_SUPCON，优先不要走这个分支。
    """
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

    if hasattr(trainer, "Ds"):
        if "Ds" in ckpt:
            trainer.Ds.load_state_dict(ckpt["Ds"])
            logging.info("Loaded domain discriminator weights: Ds")
        else:
            logging.warning("Trainer has Ds but checkpoint has no Ds.")

    return ckpt


def load_checkpoint_to_trainer(trainer, ckpt_path, device):
    """
    关键修改：
    V4/V4_SUPCON 必须优先调用 trainer.load_model()，不要只手动加载 G/Fs/Cs/Ds。

    原因：
    训练验证阶段的 _eval_class_weighted_fusion() 依赖 checkpoint 中保存的：
        source_weight_ema
        class_source_weight_ema
        class_source_weight_last_epoch
        pl_conf_thresh
        pl_min_target
    这些状态如果不加载，画图结果就可能和训练 best val-acc 有轻微差异。
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logging.info(f"Loading checkpoint: {ckpt_path}")
    trainer.args.load_path = ckpt_path

    used_trainer_load_model = False
    if hasattr(trainer, "load_model"):
        try:
            trainer.load_model()
            used_trainer_load_model = True
            logging.info("Checkpoint loaded by trainer.load_model().")
        except Exception as e:
            logging.warning(
                "trainer.load_model() failed, fallback to manual loading. Error: {}".format(e)
            )

    if not used_trainer_load_model:
        ckpt = _manual_load_checkpoint_to_trainer(trainer, ckpt_path, device)
        logging.info("Checkpoint loaded manually: G/Fs/Cs/Ds only.")
    else:
        ckpt = torch.load(ckpt_path, map_location=device)

    # 打印 V3/V4 状态，方便确认是否真的加载了训练中的融合状态
    if hasattr(trainer, "source_weight_ema"):
        try:
            logging.info(
                "source_weight_ema: {}".format(
                    trainer.source_weight_ema.detach().cpu().numpy()
                )
            )
        except Exception:
            pass

    if hasattr(trainer, "class_source_weight_ema"):
        try:
            logging.info(
                "class_source_weight_ema shape: {}".format(
                    tuple(trainer.class_source_weight_ema.shape)
                )
            )
        except Exception:
            pass

    if hasattr(trainer, "class_source_weight_last_epoch"):
        obj = getattr(trainer, "class_source_weight_last_epoch")
        if obj is not None:
            try:
                logging.info(
                    "class_source_weight_last_epoch shape: {}".format(tuple(obj.shape))
                )
            except Exception:
                pass
        else:
            logging.warning("class_source_weight_last_epoch is None after loading.")

    if hasattr(trainer, "pl_conf_thresh"):
        logging.info(
            "V4 pseudo-label gate: threshold={}, min_target={}".format(
                getattr(trainer, "pl_conf_thresh", None),
                getattr(trainer, "pl_min_target", None),
            )
        )

    # 从 checkpoint 里也打印一下保存状态，便于排查
    for key in [
        "v3_eval_no_ema",
        "v4_pseudo_label_conf_gate",
        "pl_conf_thresh",
        "pl_min_target",
        "lambda_supcon",
        "supcon_focus_classes",
        "v5_mca",
        "rec_score_weight",
        "rec_score_mode",
        "lambda_mca",
        "mca_start_epoch",
    ]:
        if key in ckpt:
            logging.info(f"checkpoint[{key}] = {ckpt[key]}")

    logging.info("Checkpoint loaded successfully.")


@torch.no_grad()
def forward_target_batch(trainer, data, feature_mode="F_mean"):
    """
    对 batch 进行前向推理。

    关键修改：
    - 先得到每个源域分支的 prob_list；
    - 如果模型提供 _eval_class_weighted_fusion，则使用训练验证阶段一致的 V3/V4 融合；
    - 否则兜底使用三个源域简单平均。
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

    fusion_weights = None
    if hasattr(trainer, "_eval_class_weighted_fusion"):
        out = trainer._eval_class_weighted_fusion(prob_list)
        if isinstance(out, tuple):
            probs = out[0]
            if len(out) > 1:
                fusion_weights = out[1]
        else:
            probs = out
    else:
        probs = torch.stack(prob_list, dim=0).mean(dim=0)

    pred_train_label = probs.argmax(dim=1)

    pred_actual_label = trainer._get_actual_label(
        pred_train_label,
        label_set=trainer.src_labels_flat,
    )

    if feature_mode == "G":
        feat_plot = feat_g
    else:
        feat_plot = torch.stack(feat_list, dim=0).mean(dim=0)

    return probs, pred_actual_label, feat_plot


@torch.no_grad()
def collect_target_predictions(trainer, args):
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
            feature_mode=args.feature_mode,
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
            feature_mode=args.feature_mode,
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
        zero_division=0,
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



def plot_per_class_accuracy(labels, preds, class_names, output_dir):
    """
    每类 accuracy = 对角线样本数 / 该类真实样本数。
    对单标签多分类任务来说，它等价于每类 recall，但图名更直观。
    """
    num_classes = len(class_names)
    label_ids = list(range(num_classes))
    cm = confusion_matrix(labels, preds, labels=label_ids)
    support = cm.sum(axis=1)
    per_class_acc = np.diag(cm) / np.maximum(support, 1)

    path_csv = os.path.join(output_dir, "per_class_accuracy.csv")
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "correct", "support", "per_class_accuracy"])
        for i, name in enumerate(class_names):
            writer.writerow([i, name, int(cm[i, i]), int(support[i]), f"{per_class_acc[i]:.6f}"])

    x = np.arange(num_classes)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, per_class_acc)
    ax.set_title("Per-class Accuracy on Target Domain")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i, v in enumerate(per_class_acc):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path_png = os.path.join(output_dir, "per_class_accuracy_bar.png")
    fig.savefig(path_png, dpi=300)
    plt.close(fig)

    logging.info(f"Saved per-class accuracy CSV: {path_csv}")
    logging.info(f"Saved per-class accuracy bar: {path_png}")
    return per_class_acc


def _load_class_source_weight_from_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "class_source_weight_last_epoch" in ckpt:
        name = "class_source_weight_last_epoch"
        cw = ckpt[name]
    elif "class_source_weight_ema" in ckpt:
        name = "class_source_weight_ema"
        cw = ckpt[name]
    else:
        logging.warning("Checkpoint has no class_source_weight_last_epoch / class_source_weight_ema.")
        return None, None, ckpt

    if isinstance(cw, torch.Tensor):
        cw = cw.detach().cpu().numpy()
    else:
        cw = np.asarray(cw, dtype=np.float32)

    cw = cw.astype(np.float32)
    col_sum = cw.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1.0
    cw = cw / col_sum
    return cw, name, ckpt


def _save_matrix_csv(matrix, row_names, col_names, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["source/class"] + list(col_names))
        for i, row_name in enumerate(row_names):
            writer.writerow([row_name] + [float(x) for x in matrix[i]])


def plot_class_source_weight_figures(args, class_names):
    """
    保存：
    1. class_source_weight_matrix.csv
    2. class_source_weight_heatmap.png
    3. global_source_reliability_bar.png
    4. class_source_weight_deviation_matrix.csv
    5. class_source_weight_deviation_heatmap.png
    6. class_source_weight_deviation_lineplot.png
    """
    cw, weight_name, ckpt = _load_class_source_weight_from_ckpt(args.ckpt_path)
    if cw is None:
        return

    source_names = args.source_name
    if len(source_names) != cw.shape[0] or len(class_names) != cw.shape[1]:
        logging.warning(
            "Skip reliability figures because shape mismatch: sources=%d, classes=%d, cw_shape=%s",
            len(source_names), len(class_names), tuple(cw.shape)
        )
        return

    # ---------- Save raw weight CSV ----------
    csv_path = os.path.join(args.output_dir, "class_source_weight_matrix.csv")
    _save_matrix_csv(cw, source_names, class_names, csv_path)

    # ---------- Raw heatmap ----------
    fig, ax = plt.subplots(figsize=(1.35 * len(class_names), 1.0 * len(source_names) + 3))
    im = ax.imshow(cw, aspect="auto")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(source_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(source_names)
    ax.set_xlabel("Class")
    ax.set_ylabel("Source Domain")
    ax.set_title(f"Class-wise Source Reliability Heatmap\n({weight_name})")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Reliability Weight")
    for i in range(cw.shape[0]):
        for j in range(cw.shape[1]):
            val = cw[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=8)
    fig.tight_layout()
    path = os.path.join(args.output_dir, "class_source_weight_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved class-source reliability heatmap: {path}")

    # ---------- Global source reliability ----------
    global_weight = cw.mean(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(len(source_names)), global_weight)
    ax.set_xticks(np.arange(len(source_names)))
    ax.set_xticklabels(source_names)
    ax.set_ylabel("Average Reliability Weight")
    ax.set_xlabel("Source Domain")
    ax.set_title(f"Average Source Reliability Across Classes\n({weight_name})")
    for i, v in enumerate(global_weight):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.output_dir, "global_source_reliability_bar.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved global source reliability bar: {path}")

    # ---------- Deviation heatmap ----------
    row_mean = cw.mean(axis=1, keepdims=True)
    dev = cw - row_mean
    dev_csv_path = os.path.join(args.output_dir, "class_source_weight_deviation_matrix.csv")
    _save_matrix_csv(dev, source_names, class_names, dev_csv_path)

    max_abs = np.max(np.abs(dev))
    if max_abs < 1e-8:
        max_abs = 1e-8

    fig, ax = plt.subplots(figsize=(1.35 * len(class_names), 1.0 * len(source_names) + 3))
    im = ax.imshow(dev, aspect="auto", cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(source_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(source_names)
    ax.set_xlabel("Class")
    ax.set_ylabel("Source Domain")
    ax.set_title(f"Class-wise Deviation Heatmap\n({weight_name})")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Deviation from Source Mean")
    for i in range(dev.shape[0]):
        for j in range(dev.shape[1]):
            val = dev[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    color="white" if abs(val) > max_abs * 0.65 else "black", fontsize=8)
    fig.tight_layout()
    path = os.path.join(args.output_dir, "class_source_weight_deviation_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved class-source reliability deviation heatmap: {path}")

    # ---------- Deviation lineplot ----------
    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(1.35 * len(class_names), 6))
    for i, src in enumerate(source_names):
        ax.plot(x, dev[i], marker="o", label=src)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Deviation from Source Mean")
    ax.set_title(f"Class-wise Reliability Deviation Across Classes\n({weight_name})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    path = os.path.join(args.output_dir, "class_source_weight_deviation_lineplot.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved class-source reliability deviation lineplot: {path}")

    logging.info("Reliability figure source weights average: %s", ", ".join(
        [f"{source_names[i]}={global_weight[i]:.4f}" for i in range(len(source_names))]
    ))

def stratified_sample_indices(labels, max_per_class):
    selected = []
    labels = np.asarray(labels)

    rng = np.random.default_rng(10)
    for cls in sorted(np.unique(labels)):
        idx = np.where(labels == cls)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.extend(idx.tolist())

    selected = np.array(selected)
    rng.shuffle(selected)
    return selected


def build_tsne(args):
    kwargs = dict(
        n_components=2,
        perplexity=args.tsne_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.random_state,
    )
    try:
        return TSNE(max_iter=args.tsne_iter, **kwargs)
    except TypeError:
        return TSNE(n_iter=args.tsne_iter, **kwargs)


def plot_tsne_target_by_class(features, labels, class_names, args):
    output_dir = args.output_dir

    idx = stratified_sample_indices(labels, args.tsne_max_per_class)

    x = features[idx]
    y = labels[idx]

    logging.info(f"Running t-SNE for target class distribution, samples={len(idx)}")

    tsne = build_tsne(args)
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
            label=name,
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
            max_samples=max_samples,
        )

        domain_features.append(feats)
        domain_labels.append(np.full(feats.shape[0], domain_id))

    x = np.concatenate(domain_features, axis=0)
    y = np.concatenate(domain_labels, axis=0)

    logging.info(f"Running t-SNE for domain distribution, samples={x.shape[0]}")

    tsne = build_tsne(args)
    emb = tsne.fit_transform(x)

    fig, ax = plt.subplots(figsize=(10, 8))

    for domain_id, domain_name in enumerate(domain_names):
        idx = y == domain_id
        ax.scatter(
            emb[idx, 0],
            emb[idx, 1],
            s=10,
            alpha=0.65,
            label=domain_name,
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
    logging.info("Plot saved results with training-time V4 eval fusion")
    logging.info("=" * 80)

    class_names = parse_class_names(args)
    args = prepare_label_sets(args, class_names)

    logging.info(f"Model: {args.model_name}")
    logging.info(f"Source: {args.source_name}")
    logging.info(f"Target: {args.target}")
    logging.info(f"Classes ({len(class_names)}): {class_names}")
    logging.info(f"Checkpoint: {args.ckpt_path}")
    logging.info(f"Output dir: {args.output_dir}")
    logging.info("Fusion mode: trainer._eval_class_weighted_fusion if available; otherwise mean.")

    if not args.ckpt_path.endswith("_best.pth"):
        logging.warning(
            "当前加载的不是 _best.pth。若要复现训练日志中的 best acc，建议使用 xxx_best.pth。"
        )

    trainer = build_trainer(args)
    load_checkpoint_to_trainer(trainer, args.ckpt_path, trainer.device)

    labels, preds, probs, features = collect_target_predictions(trainer, args)

    save_predictions_csv(labels, preds, probs, class_names, args.output_dir)

    precision, recall, f1, support = save_metrics_csv(
        labels,
        preds,
        class_names,
        args.output_dir,
    )

    plot_confusion_matrices(labels, preds, class_names, args.output_dir)
    plot_prf_bar(precision, recall, f1, class_names, args.output_dir)
    plot_per_class_accuracy(labels, preds, class_names, args.output_dir)
    plot_class_source_weight_figures(args, class_names)
    plot_tsne_target_by_class(features, labels, class_names, args)
    plot_tsne_domain_distribution(trainer, args)

    logging.info("=" * 80)
    logging.info("All figures and CSV files are saved.")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
