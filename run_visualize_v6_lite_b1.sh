#!/usr/bin/env bash
set -euo pipefail

# 在项目根目录运行本脚本。
EXP_ROOT="./ckpt/PU0_EXP_B1_PREFLOOR001_CONFIRM2"

CKPT_PATH="$(find "$EXP_ROOT" -type f -name '*_best.pth' | sort | tail -n 1)"
LOG_PATH="$(find "$EXP_ROOT" -type f -name '*.log' | sort | tail -n 1)"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "没有找到 *_best.pth：${EXP_ROOT}" >&2
  exit 1
fi

if [[ -z "${LOG_PATH}" ]]; then
  echo "没有找到训练日志，将只画模型预测相关图。" >&2
fi

python visualize_v6_lite_full_diagnostics.py \
  --model_name MFSAN_CDAN_BIMAMBA_CW_RWCA_V6_LITE_PU0 \
  --source PU_1,PU_2,PU_3 \
  --target PU_0 \
  --train_mode multi_source \
  --data_dir /workspace/PU_TL_9_replace \
  --signal_size 1024 \
  --backbone CNN \
  --cuda_device 0 \
  --batch_size 64 \
  --num_workers 4 \
  --target_test_size 0.40 \
  --target_split_mode time \
  --random_state 2027 \
  --include_faults K001,KA04,KA16,KA30,KB23,KB24,KI04,KI16,KI17 \
  --lambda_l1 0.0 \
  --lambda_cda 0.0 \
  --lambda_ent 0.0 \
  --lambda_adv 0.02 \
  --lambda_clmmd 0.005 \
  --lambda_supcon 0.01 \
  --supcon_temperature 0.20 \
  --supcon_start_epoch 3 \
  --supcon_feature_mode G \
  --supcon_focus_classes 1,3,8 \
  --pl_conf_thresh 0.80 \
  --pl_min_target 2 \
  --rec_score_weight 0.30 \
  --rec_score_mode prob \
  --lambda_mca 0.0 \
  --v6_gate_enabled True \
  --v6_gate_start_epoch 4 \
  --v6_gate_confirm_epochs 2 \
  --v6_gate_release_epochs 3 \
  --v6_gate_confirm_gap 0.08 \
  --v6_gate_release_gap 0.03 \
  --v6_gate_preconfirm_floor 0.01 \
  --v6_gate_bottom_floor 0.01 \
  --v6_gate_max_source_weight 0.75 \
  --v6_gate_apply_to_supcon True \
  --v6_supcon_source_min_weight 0.05 \
  --v6_class_weight_power 1.20 \
  --v6_class_alignment_boost 1.0 \
  --ckpt_path "$CKPT_PATH" \
  --log_path "$LOG_PATH" \
  --output_dir ./visual_results/PU0_B1_FULL_DIAGNOSTICS \
  --feature_mode F_mean \
  --top_confusion_pairs 8 \
  --representative_errors 6 \
  --tsne_max_per_class 250 \
  --rolling_window 500 \
  --integrated_gradients_steps 16 \
  --skip_heavy False
