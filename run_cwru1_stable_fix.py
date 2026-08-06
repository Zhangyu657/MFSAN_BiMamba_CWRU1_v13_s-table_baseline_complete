#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the conservative CWRU_0+CWRU_2+CWRU_3 -> CWRU_1 experiment.

The neural-network architecture is unchanged.  This script only changes the
training schedule, source-weight floors, pseudo-label filtering, and checkpoint
selection parameters.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def build_command(args, seed):
    train_py = Path(__file__).resolve().parent / 'train.py'
    return [
        sys.executable, str(train_py),
        '--model_name', 'MFSAN_CDAN_BIMAMBA_CW_RWCA_V12_CWRU1_STABLE_RESCUE',
        '--dataset_profile', 'cwru',
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

        # Same five losses, with a weaker adaptation schedule to reduce late
        # negative transfer and normal-class collapse.
        '--lambda_l1', '0.0',
        '--lambda_cda', '0.0',
        '--lambda_ent', '0.0',
        '--lambda_mca', '0.0',
        '--lambda_adv', str(args.lambda_adv),
        '--lambda_clmmd', str(args.lambda_clmmd),
        '--lambda_supcon', str(args.lambda_supcon),
        '--supcon_temperature', '0.20',
        '--supcon_start_epoch', str(args.supcon_start_epoch),
        '--supcon_feature_mode', 'G',
        '--supcon_focus_classes', '0,1,2',
        '--pl_conf_thresh', '0.85',
        '--pl_min_target', '2',
        '--rec_score_weight', '0.30',

        # Keep a small contribution from every source.  The former run reduced
        # CWRU_0 to about 0.1%, effectively turning three-source adaptation into
        # two-source adaptation.
        '--v6_gate_enabled', 'True',
        '--v6_gate_start_epoch', '4',
        '--v6_gate_confirm_epochs', '3',
        '--v6_gate_bottom_floor', str(args.source_floor),
        '--v6_gate_preconfirm_floor', str(args.source_floor),
        '--v6_gate_max_source_weight', '0.70',

        # Delay class-wise irreversible decisions until a stable target
        # boundary exists.
        '--v7_class_gate_enabled', 'True',
        '--v7_class_gate_start_epoch', '8',
        '--v7_class_gate_confirm_epochs', '4',
        '--v7_class_gate_bottom_floor', str(args.class_floor),
        '--v7_class_gate_preconfirm_floor', str(args.class_floor),
        '--v7_class_gate_log_classes', '0,1,2,6',
        '--v7_conflict_fusion_enabled', 'True',
        '--v9_specialist_protection_enabled', 'True',
        '--v9_specialist_start_epoch', '8',

        # Mild, delayed separation for the three ball-fault severities.
        '--v8_hard_supcon_enabled', 'True',
        '--v8_hard_negative_pairs', '0-2,1-2',
        '--v8_hard_negative_weight', '1.10',
        '--v8_supcon_anchor_classes', '0,1,2',
        '--v9_hard_supcon_start_epoch', str(args.rescue_start_epoch),
        '--v9_hard_supcon_ramp_epochs', '4',
        '--v10_hard_pair_weights', '0-2:1.05,1-2:1.10',

        # Retain the original prototype mechanism, but avoid the former 1.5x
        # class-2 over-alignment.
        '--v8_prototype_filter_enabled', 'True',
        '--v8_prototype_start_epoch', '4',
        '--v8_prototype_ema_momentum', '0.95',
        '--v8_prototype_margin', '0.05',
        '--v8_prototype_min_updates', '2',
        '--v8_prototype_conf_overrides', '0:0.90,1:0.90,2:0.80',
        '--v8_clmmd_class_boost', '2:1.10',
        '--v8_prototype_log_classes', '0,1,2,6',
        '--v9_prototype_filter_classes', '0,1,2',
        '--v9_radius_min', '0.02',
        '--v9_radius_max', '0.10',
        '--v9_prototype_min_similarity', '0.40',
        '--v9_prototype_soft_tau', '0.10',
        '--v10_radius_class_min', '2:0.02',
        '--v10_radius_class_max', '2:0.06',

        # Normal-class guard must use the CWRU normal index 6.
        '--v10_normal_guard_enabled', 'True',
        '--v10_normal_class', '6',
        '--v10_normal_min_prob', '0.80',
        '--v10_normal_guard_min_fault_prob', '0.05',

        # V12 stable rescue: one specialist branch, strict normal exclusion,
        # strict prototype agreement, and at most a few samples per batch.
        '--v12_rescue_enabled', 'True',
        '--v12_rescue_class', '2',
        '--v12_normal_class', '6',
        '--v12_confusion_classes', '0,1',
        '--v12_rescue_source_indices', args.rescue_source_indices,
        '--v12_rescue_start_epoch', str(args.rescue_start_epoch),
        '--v12_rescue_end_epoch', str(args.rescue_end_epoch),
        '--v12_rescue_topk', '2',
        '--v12_rescue_min_class_prob', str(args.rescue_min_prob),
        '--v12_rescue_min_competitor_ratio', str(args.rescue_min_ratio),
        '--v12_rescue_max_normal_prob', str(args.rescue_max_normal_prob),
        '--v12_rescue_proto_margin', str(args.rescue_proto_margin),
        '--v12_rescue_normal_proto_margin', str(args.rescue_normal_proto_margin),
        '--v12_rescue_min_similarity', str(args.rescue_min_similarity),
        '--v12_rescue_radius_cap', str(args.rescue_radius_cap),
        '--v12_rescue_max_per_batch', str(args.rescue_max_per_batch),
        '--v12_rescue_min_target', '2',
        '--v12_rescue_mix_alpha', str(args.rescue_mix_alpha),
        '--v12_rescue_clmmd_boost', str(args.rescue_clmmd_boost),
        '--v12_rescue_score_tau', '0.10',
        '--v12_eval_rescue_enabled', str(args.eval_rescue),

        # Keep the requested PU-style target-test checkpoint selection.  The
        # class-aware score prevents a high overall accuracy from hiding a
        # completely collapsed ball_21 class.
        '--eval_each_epoch', str(args.eval_each_epoch),
        '--select_best_on_target', str(args.select_best_on_target),
        '--save_best', str(args.select_best_on_target),
        '--best_metric', args.best_metric,
        '--best_focus_class', '2',
        '--best_accuracy_weight', '0.40',
        '--best_macro_f1_weight', '0.35',
        '--best_focus_recall_weight', '0.25',
        '--save_dir', args.save_dir,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--seeds', default='2027')
    parser.add_argument('--cuda_device', default='0')
    parser.add_argument('--max_epoch', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--signal_size', type=int, default=1024)
    parser.add_argument('--target_test_size', type=float, default=0.40)

    parser.add_argument('--lambda_adv', type=float, default=0.008)
    parser.add_argument('--lambda_clmmd', type=float, default=0.002)
    parser.add_argument('--lambda_supcon', type=float, default=0.004)
    parser.add_argument('--supcon_start_epoch', type=int, default=4)
    parser.add_argument('--source_floor', type=float, default=0.03)
    parser.add_argument('--class_floor', type=float, default=0.03)

    parser.add_argument('--rescue_source_indices', default='2')
    parser.add_argument('--rescue_start_epoch', type=int, default=6)
    parser.add_argument('--rescue_end_epoch', type=int, default=10)
    parser.add_argument('--rescue_min_prob', type=float, default=0.20)
    parser.add_argument('--rescue_min_ratio', type=float, default=0.50)
    parser.add_argument('--rescue_max_normal_prob', type=float, default=0.20)
    parser.add_argument('--rescue_proto_margin', type=float, default=0.08)
    parser.add_argument('--rescue_normal_proto_margin', type=float, default=0.10)
    parser.add_argument('--rescue_min_similarity', type=float, default=0.45)
    parser.add_argument('--rescue_radius_cap', type=float, default=0.05)
    parser.add_argument('--rescue_max_per_batch', type=int, default=4)
    parser.add_argument('--rescue_mix_alpha', type=float, default=0.25)
    parser.add_argument('--rescue_clmmd_boost', type=float, default=1.10)
    parser.add_argument('--eval_rescue', choices=['True', 'False'], default='False')

    parser.add_argument('--eval_each_epoch', choices=['True', 'False'], default='True')
    parser.add_argument('--select_best_on_target', choices=['True', 'False'], default='True')
    parser.add_argument(
        '--best_metric',
        choices=['accuracy', 'macro_f1', 'class_aware'],
        default='class_aware',
    )
    parser.add_argument('--save_dir', default='./ckpt/CWRU1_V12_STABLE_FIX')
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(',') if item.strip()]
    for seed in seeds:
        command = build_command(args, seed)
        print('\nRUN:', ' '.join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
