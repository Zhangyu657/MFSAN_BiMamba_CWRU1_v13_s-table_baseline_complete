#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the targeted CWRU_0+CWRU_2+CWRU_3 -> CWRU_1 repair experiment."""

import argparse
import subprocess
import sys
from pathlib import Path


def build_command(args, seed):
    train_py = Path(__file__).resolve().parent / 'train.py'
    return [
        sys.executable, str(train_py),
        '--model_name', 'MFSAN_CDAN_BIMAMBA_CW_RWCA_V11_CWRU1_CLASS_RESCUE',
        '--dataset_profile', 'cwru',
        # Do not let the general profile disable the CWRU_1-specific modules.
        '--cwru_apply_general_defaults', 'False',
        '--source', 'CWRU_0,CWRU_2,CWRU_3',
        '--target', 'CWRU_1',
        '--train_mode', 'multi_source',
        '--data_dir', str(Path(args.data_dir).expanduser()),
        '--include_faults',
        'ball_07,ball_14,ball_21,inner_07,inner_14,inner_21,normal,outer_07,outer_14,outer_21',
        '--signal_size', str(args.signal_size),
        '--backbone', 'CNN',
        '--cuda_device', args.cuda_device,
        '--max_epoch', str(args.max_epoch),
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--random_state', str(seed),
        '--target_test_size', str(args.target_test_size),
        '--target_split_mode', 'time',
        '--source_balance_data', 'True',

        # Keep the same five losses, but reduce late over-alignment.
        '--lambda_l1', '0.0',
        '--lambda_cda', '0.0',
        '--lambda_ent', '0.0',
        '--lambda_mca', '0.0',
        '--lambda_adv', str(args.lambda_adv),
        '--lambda_clmmd', str(args.lambda_clmmd),
        '--lambda_supcon', str(args.lambda_supcon),
        '--supcon_temperature', '0.20',
        '--supcon_start_epoch', '2',
        '--supcon_feature_mode', 'G',
        '--supcon_focus_classes', '0,1,2',
        '--pl_conf_thresh', '0.80',
        '--pl_min_target', '2',
        '--rec_score_weight', '0.30',

        # Delay irreversible source/class gates until the target boundary exists.
        '--v7_class_gate_enabled', 'True',
        '--v7_class_gate_start_epoch', '6',
        '--v7_class_gate_confirm_epochs', '4',
        '--v7_class_gate_log_classes', '0,1,2,6',
        '--v7_conflict_fusion_enabled', 'True',
        '--v9_specialist_protection_enabled', 'True',
        '--v9_specialist_start_epoch', '6',

        # Separate the three rolling-element severity classes.
        '--v8_hard_supcon_enabled', 'True',
        '--v8_hard_negative_pairs', '0-2,1-2',
        '--v8_hard_negative_weight', '1.25',
        '--v8_supcon_anchor_classes', '0,1,2',
        '--v9_hard_supcon_start_epoch', '2',
        '--v9_hard_supcon_ramp_epochs', '3',
        '--v10_hard_pair_weights', '0-2:1.15,1-2:1.25',

        # Conservative prototype filtering and class-2 CLMMD rescue.
        '--v8_prototype_filter_enabled', 'True',
        '--v8_prototype_start_epoch', '2',
        '--v8_prototype_ema_momentum', '0.90',
        '--v8_prototype_margin', '0.03',
        '--v8_prototype_min_updates', '1',
        '--v8_prototype_conf_overrides', '0:0.85,1:0.85,2:0.70',
        '--v8_clmmd_class_boost', f'2:{args.rescue_clmmd_boost}',
        '--v8_prototype_log_classes', '0,1,2',
        '--v9_prototype_filter_classes', '0,1,2',
        '--v9_radius_min', '0.02',
        '--v9_radius_max', '0.12',
        '--v9_prototype_min_similarity', '0.35',
        '--v9_prototype_soft_tau', '0.10',
        '--v10_radius_class_min', '2:0.02',
        '--v10_radius_class_max', '2:0.08',

        '--v11_cwru1_rescue_enabled', 'True',
        '--v11_rescue_class', '2',
        '--v11_confusion_classes', '0,1',
        '--v11_rescue_start_epoch', '2',
        '--v11_rescue_topk', '2',
        '--v11_rescue_min_class_prob', str(args.rescue_min_prob),
        '--v11_rescue_proto_margin', '0.03',
        '--v11_rescue_min_similarity', '0.35',
        '--v11_rescue_clmmd_boost', str(args.rescue_clmmd_boost),
        '--v11_eval_rescue_enabled', 'True',
        '--v11_eval_min_class_prob', str(args.eval_min_prob),
        '--v11_eval_competitor_ratio', str(args.eval_competitor_ratio),
        '--v11_eval_min_source_votes', '2',
        '--v11_eval_boost', str(args.eval_boost),

        # The user requested the PU-style target-test checkpoint selection.
        '--eval_each_epoch', str(args.eval_each_epoch),
        '--select_best_on_target', str(args.select_best_on_target),
        '--save_best', str(args.select_best_on_target),
        # Accuracy alone hid the class-2 collapse, so use a transparent
        # class-aware score while still reporting accuracy every epoch.
        '--best_metric', args.best_metric,
        '--best_focus_class', '2',
        '--best_accuracy_weight', '0.45',
        '--best_macro_f1_weight', '0.35',
        '--best_focus_recall_weight', '0.20',
        '--save_dir', args.save_dir,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--seeds', default='2027')
    parser.add_argument('--cuda_device', default='0')
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--signal_size', type=int, default=1024)
    parser.add_argument('--target_test_size', type=float, default=0.40)
    parser.add_argument('--lambda_adv', type=float, default=0.01)
    parser.add_argument('--lambda_clmmd', type=float, default=0.003)
    parser.add_argument('--lambda_supcon', type=float, default=0.0075)
    parser.add_argument('--rescue_min_prob', type=float, default=0.10)
    parser.add_argument('--rescue_clmmd_boost', type=float, default=1.50)
    parser.add_argument('--eval_min_prob', type=float, default=0.08)
    parser.add_argument('--eval_competitor_ratio', type=float, default=0.35)
    parser.add_argument('--eval_boost', type=float, default=2.00)
    parser.add_argument('--eval_each_epoch', choices=['True', 'False'], default='True')
    parser.add_argument('--select_best_on_target', choices=['True', 'False'], default='True')
    parser.add_argument(
        '--best_metric',
        choices=['accuracy', 'macro_f1', 'class_aware'],
        default='class_aware',
    )
    parser.add_argument('--save_dir', default='./ckpt/CWRU1_V11_FIX')
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    for seed in seeds:
        cmd = build_command(args, seed)
        print('\nRUN:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
