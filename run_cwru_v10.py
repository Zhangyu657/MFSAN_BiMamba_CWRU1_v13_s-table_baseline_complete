#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one or all CWRU multi-source transfer tasks with safe general defaults."""

import argparse
import subprocess
import sys
from pathlib import Path

TASKS = {
    0: ("CWRU_1,CWRU_2,CWRU_3", "CWRU_0"),
    1: ("CWRU_0,CWRU_2,CWRU_3", "CWRU_1"),
    2: ("CWRU_0,CWRU_1,CWRU_3", "CWRU_2"),
    3: ("CWRU_0,CWRU_1,CWRU_2", "CWRU_3"),
}


def build_command(args, target_condition, seed):
    source, target = TASKS[target_condition]
    train_py = Path(__file__).resolve().parent / "train.py"
    return [
        sys.executable, str(train_py),
        "--model_name", "MFSAN_CDAN_BIMAMBA_CW_RWCA_V10_PAIRWISE_SPECIALIST_CALIBRATED",
        "--dataset_profile", "cwru",
        "--cwru_apply_general_defaults", "True",
        "--source", source,
        "--target", target,
        "--train_mode", "multi_source",
        "--data_dir", str(Path(args.data_dir).expanduser()),
        "--signal_size", str(args.signal_size),
        "--backbone", "CNN",
        "--cuda_device", args.cuda_device,
        "--max_epoch", str(args.max_epoch),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--random_state", str(seed),
        "--target_test_size", str(args.target_test_size),
        "--target_split_mode", "time",
        "--lambda_l1", "0.0",
        "--lambda_cda", "0.0",
        "--lambda_ent", "0.0",
        "--lambda_mca", "0.0",
        "--lambda_adv", "0.02",
        "--lambda_clmmd", "0.005",
        "--lambda_supcon", "0.01",
        "--supcon_temperature", "0.20",
        "--supcon_start_epoch", "3",
        "--supcon_feature_mode", "G",
        "--supcon_focus_classes", "all",
        "--pl_conf_thresh", "0.80",
        "--pl_min_target", "2",
        "--rec_score_weight", "0.30",
        "--v7_class_gate_enabled", "True",
        "--v7_class_gate_log_classes", "all",
        "--v7_conflict_fusion_enabled", "True",
        "--v8_hard_supcon_enabled", "False",
        "--v8_prototype_filter_enabled", "False",
        "--source_balance_data", "True",
        "--eval_each_epoch", str(args.eval_each_epoch),
        "--select_best_on_target", str(args.select_best_on_target),
        "--save_best", str(args.select_best_on_target),
        "--save_dir", args.save_dir,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to CWRU_TL or CWRU")
    parser.add_argument("--targets", default="0,1,2,3", help="Target conditions, e.g. 0 or 0,1,2,3")
    parser.add_argument("--seeds", default="2027", help="Seeds, e.g. 2027,2028,2029")
    parser.add_argument("--cuda_device", default="0", help="Use empty string for CPU")
    parser.add_argument("--max_epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--signal_size", type=int, default=1024)
    parser.add_argument("--target_test_size", type=float, default=0.40)
    parser.add_argument("--save_dir", default="./ckpt/CWRU_V10_GENERAL")
    parser.add_argument("--eval_each_epoch", choices=["True", "False"], default="False")
    parser.add_argument("--select_best_on_target", choices=["True", "False"], default="False")
    args = parser.parse_args()

    targets = [int(x) for x in args.targets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    for target in targets:
        if target not in TASKS:
            raise ValueError(f"Unknown target condition {target}; use 0,1,2,3")
        for seed in seeds:
            cmd = build_command(args, target, seed)
            print("\nRUN:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
