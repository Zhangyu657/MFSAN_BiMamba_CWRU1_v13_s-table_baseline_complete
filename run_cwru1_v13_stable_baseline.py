#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the next-step CWRU_1 stabilized baseline experiment.

Task: CWRU_0 + CWRU_2 + CWRU_3 -> CWRU_1.
The network structure is unchanged. This launcher removes the V7-V12 rescue
stack and uses a conservative staged schedule for the original V6 losses.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def build_command(args, seed):
    train_py = Path(__file__).resolve().parent / 'train.py'
    return [
        sys.executable,
        str(train_py),
        '--model_name',
        'MFSAN_CDAN_BIMAMBA_CW_RWCA_V13_CWRU1_BASELINE_STABILIZED',
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
        '--source_balance_data', str(args.source_balance_data),
        '--normlize_type={}'.format(args.normlize_type),

        # Smoother optimizer than the former SGD lr=0.01 run.
        '--opt', args.optimizer,
        '--lr', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--lr_scheduler', args.lr_scheduler,
        '--gamma', str(args.gamma),
        '--steps', str(args.steps),

        # Slow adaptation ramp. The five loss families remain unchanged.
        '--zeta', str(args.zeta),
        '--mmd_weight', str(args.mmd_weight),
        '--mmd_start_epoch', str(args.mmd_start_epoch),
        '--lambda_adv', str(args.lambda_adv),
        '--adv_start_epoch', str(args.adv_start_epoch),
        '--lambda_clmmd', str(args.lambda_clmmd),
        '--clmmd_start_epoch', str(args.clmmd_start_epoch),
        '--lambda_supcon', '0.0',
        '--lambda_l1', '0.0',
        '--lambda_cda', '0.0',
        '--lambda_ent', '0.0',
        '--lambda_mca', '0.0',
        '--pl_conf_thresh', str(args.pl_conf_thresh),
        '--pl_min_target', str(args.pl_min_target),
        '--source_label_smoothing', str(args.source_label_smoothing),
        '--grad_clip_norm', str(args.grad_clip_norm),

        # Keep reliability weighting mild and avoid irreversible source removal.
        '--cw_warmup_epochs', str(args.cw_warmup_epochs),
        '--cw_alpha', str(args.cw_alpha),
        '--cw_alpha_ramp_epochs', str(args.cw_alpha_ramp_epochs),
        '--rec_score_weight', str(args.rec_score_weight),
        '--v6_gate_enabled', 'False',
        '--v6_class_weight_power', str(args.class_weight_power),
        '--v6_class_alignment_boost', '1.0',
        '--v6_mca_pairwise_weight', '0.0',
        '--v6_gate_apply_to_supcon', 'False',

        # PU-style target-test checkpoint selection, with class-2 emphasis.
        '--eval_each_epoch', str(args.eval_each_epoch),
        '--select_best_on_target', str(args.select_best_on_target),
        '--save_best', str(args.select_best_on_target),
        '--best_metric', args.best_metric,
        '--best_focus_class', '2',
        '--best_accuracy_weight', str(args.best_accuracy_weight),
        '--best_macro_f1_weight', str(args.best_macro_f1_weight),
        '--best_focus_recall_weight', str(args.best_focus_recall_weight),
        '--early_stop_patience', str(args.early_stop_patience),
        '--early_stop_min_epoch', str(args.early_stop_min_epoch),
        '--early_stop_min_delta', str(args.early_stop_min_delta),
        '--log_confusion_matrix', 'True',
        '--save_dir', args.save_dir,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--seeds', default='2027')
    parser.add_argument('--cuda_device', default='0')
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--signal_size', type=int, default=1024)
    parser.add_argument('--target_test_size', type=float, default=0.40)
    parser.add_argument('--source_balance_data', choices=['True', 'False'], default='True')
    parser.add_argument('--normlize_type', choices=['0-1', '-1-1', 'mean-std'], default='-1-1')

    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', choices=['fix', 'stepLR', 'step', 'exp'], default='fix')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--steps', default='5')

    parser.add_argument('--zeta', type=float, default=2.0)
    parser.add_argument('--mmd_weight', type=float, default=0.15)
    parser.add_argument('--mmd_start_epoch', type=int, default=2)
    parser.add_argument('--lambda_adv', type=float, default=0.002)
    parser.add_argument('--adv_start_epoch', type=int, default=4)
    parser.add_argument('--lambda_clmmd', type=float, default=0.0005)
    parser.add_argument('--clmmd_start_epoch', type=int, default=5)
    parser.add_argument('--pl_conf_thresh', type=float, default=0.95)
    parser.add_argument('--pl_min_target', type=int, default=3)
    parser.add_argument('--source_label_smoothing', type=float, default=0.05)
    parser.add_argument('--grad_clip_norm', type=float, default=5.0)

    parser.add_argument('--cw_warmup_epochs', type=int, default=5)
    parser.add_argument('--cw_alpha', type=float, default=0.10)
    parser.add_argument('--cw_alpha_ramp_epochs', type=int, default=5)
    parser.add_argument('--rec_score_weight', type=float, default=0.10)
    parser.add_argument('--class_weight_power', type=float, default=1.0)

    parser.add_argument('--eval_each_epoch', choices=['True', 'False'], default='True')
    parser.add_argument('--select_best_on_target', choices=['True', 'False'], default='True')
    parser.add_argument('--best_metric', choices=['accuracy', 'macro_f1', 'class_aware'], default='class_aware')
    parser.add_argument('--best_accuracy_weight', type=float, default=0.35)
    parser.add_argument('--best_macro_f1_weight', type=float, default=0.35)
    parser.add_argument('--best_focus_recall_weight', type=float, default=0.30)
    parser.add_argument('--early_stop_patience', type=int, default=4)
    parser.add_argument('--early_stop_min_epoch', type=int, default=6)
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4)
    parser.add_argument('--save_dir', default='./ckpt/CWRU1_V13_STABLE_BASELINE')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    if not seeds:
        raise ValueError('At least one integer seed is required.')

    for seed in seeds:
        command = build_command(args, seed)
        print('\nRUN:', ' '.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
