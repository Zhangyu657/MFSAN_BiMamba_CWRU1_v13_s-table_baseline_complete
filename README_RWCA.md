# MFSAN-CDAN-BiMamba-RWCA patch

This patch adds a new model:

`models/MFSAN_CDAN_BIMAMBA_RWCA.py`

RWCA = Reliability-Weighted Class-wise Alignment.

## Main changes

1. Keep the current MSCNN-BiMamba-SmallGate backbone.
2. Keep multi-source task-specific branches Fs_i / Cs_i.
3. Compute source reliability weights from:
   - source-target MK-MMD distance;
   - target prediction entropy.
4. Use source reliability weights for:
   - source classification loss;
   - MMD alignment loss;
   - CDA conditional MMD loss;
   - CDAN adversarial domain loss;
   - classifier discrepancy / consistency loss.
5. Add class-wise LMMD / LJMMD-like subdomain alignment loss.
6. Use weighted source prediction fusion for final target prediction.
7. Add best checkpoint saving in `train_utils.py`.
8. Add RWCA and BiMamba command-line parameters in `opt.py`.

## Files

Copy these files into the project root:

- `models/MFSAN_CDAN_BIMAMBA_RWCA.py`
- `train_utils.py`
- `opt.py`

## Recommended command

```bash

python train.py \
  --model_name MFSAN_CDAN_BIMAMBA_CW_RWCA_V4 \
  --source PU_1,PU_2,PU_3 \
  --target PU_0 \
  --data_dir /workspace/PU_TL_9_replace \
  --train_mode multi_source \
  --cuda_device 0 \
  --max_epoch 10 \
  --batch_size 64 \
  --signal_size 1024 \
  --backbone CNN \
  --lambda_clmmd 0.005 \
  --pl_conf_thresh 0.80 \
  --pl_min_target 2



python train.py \
  --model_name MFSAN_CDAN_BIMAMBA_RWCA \
  --source PU_0,PU_1,PU_2 \
  --target PU_3 \
  --train_mode multi_source \
  --data_dir /workspace/PU_TL_9_replace \
  --signal_size 1024 \
  --backbone CNN \
  --cuda_device 0 \
  --max_epoch 10 \
  --lambda_adv 0.01 \
  --lambda_grl 0.5 \
  --lambda_cda 0.02 \
  --lambda_ent 0.005 \
  --lambda_clmmd 0.02 \
  --adv_detach_prob True \
  --adv_use_entropy_weight True \
  --adv_conf_thresh 0.8 \
  --rw_tau 0.5 \
  --rw_mmd_weight 1.0 \
  --rw_ent_weight 1.0 \
  --rw_detach_weights True \
  --include_faults K001,KA04,KA16,KA30,KB24,KB23,KI04,KI17,KI16


  python train.py \
  --model_name MFSAN_CDAN_BIMAMBA_RWCA \
  --source PU_1,PU_2,PU_3 \
  --target PU_0 \
  --train_mode multi_source \
  --data_dir /workspace/PU_TL_9_replace \
  --signal_size 1024 \
  --backbone CNN \
  --cuda_device 0 \
  --max_epoch 10 \
  --lambda_adv 0.01 \
  --lambda_grl 0.5 \
  --lambda_cda 0.02 \
  --lambda_ent 0.005 \
  --lambda_clmmd 0.01 \
  --adv_detach_prob True \
  --adv_use_entropy_weight True \
  --adv_conf_thresh 0.8 \
  --rw_tau 1.0 \
  --rw_mmd_weight 1.0 \
  --rw_ent_weight 1.0 \
  --rw_detach_weights True \
  --include_faults K001,KA04,KA16,KA30,KB24,KB23,KI04,KI17,KI16
```

## What to watch in the log

- `RWCA average train source weights`
- `RWCA EMA source weights`
- `Val source fusion weights`
- `CLMMD` and `CLMMD Weighted`
- `Best model saved to ..._best.pth`



05-10 08:22:43 model_name: MFSAN_CDAN_BIMAMBA_RWCA
05-10 08:22:43 source: PU_0,PU_1,PU_2
05-10 08:22:43 target: PU_3
05-10 08:22:43 data_dir: /workspace/PU_TL_9_replace
05-10 08:22:43 train_mode: multi_source
05-10 08:22:43 cuda_device: 0
05-10 08:22:43 max_epoch: 10
05-10 08:22:43 batch_size: 64
05-10 08:22:43 signal_size: 1024
05-10 08:22:43 random_state: 10
05-10 08:22:43 include_faults: K001,KA04,KA16,KA30,KB24,KB23,KI04,KI17,KI16
05-10 08:22:43 exclude_faults: 
05-10 08:22:43 opt: sgd
05-10 08:22:43 momentum: 0.9
05-10 08:22:43 betas: (0.9, 0.999)
05-10 08:22:43 weight_decay: 0.0005
05-10 08:22:43 lr: 0.01
05-10 08:22:43 lr_scheduler: stepLR
05-10 08:22:43 gamma: 0.2
05-10 08:22:43 steps: 10
05-10 08:22:43 backbone: CNN
05-10 08:22:43 num_workers: 4
05-10 08:22:43 normlize_type: -1-1
05-10 08:22:43 tradeoff: ['exp', 'exp', 'exp']
05-10 08:22:43 zeta: 10.0
05-10 08:22:43 dropout: 0.0
05-10 08:22:43 lambda_cda: 0.02
05-10 08:22:43 lambda_ent: 0.005
05-10 08:22:43 cda_detach_prob: True
05-10 08:22:43 lambda_adv: 0.01
05-10 08:22:43 lambda_grl: 0.5
05-10 08:22:43 adv_hidden_dim: 256
05-10 08:22:43 adv_detach_prob: True
05-10 08:22:43 adv_use_entropy_weight: True
05-10 08:22:43 adv_conf_thresh: 0.8
05-10 08:22:43 save: True
05-10 08:22:43 save_dir: ./ckpt
05-10 08:22:43 load_path: 
05-10 08:22:43 bla_gate_init: 0.01
05-10 08:22:43 bla_gate_max: 0.03
05-10 08:22:43 save_best: True
05-10 08:22:43 bimamba_stem_channels: 64
05-10 08:22:43 bimamba_dim: 64
05-10 08:22:43 bimamba_depth: 2
05-10 08:22:43 bimamba_d_state: 16
05-10 08:22:43 bimamba_d_conv: 4
05-10 08:22:43 bimamba_expand: 2
05-10 08:22:43 bimamba_gate_init: 0.01
05-10 08:22:43 bimamba_gate_max: 0.03
05-10 08:22:43 rw_tau: 0.5
05-10 08:22:43 rw_mmd_weight: 1.0
05-10 08:22:43 rw_ent_weight: 1.0
05-10 08:22:43 rw_detach_weights: True
05-10 08:22:43 rw_ema_momentum: 0.9
05-10 08:22:43 rw_eval_use_entropy: True
05-10 08:22:43 rw_eval_tau: 0.5
05-10 08:22:43 lambda_clmmd: 0.02
05-10 08:22:43 clmmd_kernel_num: 5
05-10 08:22:43 clmmd_kernel_mul: 2.0
05-10 08:22:43 clmmd_min_source: 2
05-10 08:22:43 clmmd_min_target_weight: 0.001
05-10 08:22:43 save_path: ./ckpt/MFSAN_CDAN_BIMAMBA_RWCA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0510-082243
05-10 08:22:43 Source PU_0 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB23' 'KB24' 'KI04' 'KI16' 'KI17']
05-10 08:22:43 Source PU_1 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB23' 'KB24' 'KI04' 'KI16' 'KI17']
05-10 08:22:43 Source PU_2 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB23' 'KB24' 'KI04' 'KI16' 'KI17']
05-10 08:22:43 Target PU_3 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB23' 'KB24' 'KI04' 'KI16' 'KI17']
05-10 08:22:43 The scenario is: closed-set domain adaptation
05-10 08:22:44 using 1 / 1 gpus
05-10 08:22:46 Using model: MFSAN_CDAN_BIMAMBA_RWCA
05-10 08:22:46 Requested backbone: CNN
05-10 08:22:46 Actual backbone: MSCNN_BiMamba_Att_SmallGate
05-10 08:22:46 Backbone output dim: 640
05-10 08:22:46 BiMamba implementation: mamba_ssm
05-10 08:22:46 Initial BiMamba-Att residual gate: 0.010000
05-10 08:22:46 Max BiMamba-Att residual gate: 0.030000
05-10 08:22:49 Source set PU_0 number of samples: 45073.
05-10 08:22:49 Source set PU_1 number of samples: 45035.
05-10 08:22:49 Source set PU_2 number of samples: 45200.
05-10 08:22:50 Training set number of samples: 36106.
05-10 08:22:50 Validation set number of samples: 9029.
05-10 08:22:51 MFSAN-CDAN-BiMamba-RWCA lambda_adv: 0.010000
05-10 08:22:51 MFSAN-CDAN-BiMamba-RWCA lambda_grl: 0.500000
05-10 08:22:51 MFSAN-CDAN-BiMamba-RWCA lambda_cda: 0.020000
05-10 08:22:51 MFSAN-CDAN-BiMamba-RWCA lambda_ent: 0.005000
05-10 08:22:51 MFSAN-CDAN-BiMamba-RWCA lambda_clmmd: 0.020000
05-10 08:22:51 MFSAN-CDAN-BiMamba-RWCA rw_tau: 0.500000
05-10 08:22:51 MFSAN-CDAN-BiMamba-RWCA rw_mmd_weight: 1.000000
05-10 08:22:51 MFSAN-CDAN-BiMamba-RWCA rw_ent_weight: 1.000000
05-10 08:22:51 MFSAN-CDAN-BiMamba-RWCA rw_detach_weights: True
05-10 08:22:51 MFSAN-CDAN-BiMamba-RWCA joint feature dim: 9 x 40 = 360
05-10 08:22:51 -----Epoch 1/10-----
05-10 08:22:51 current lr: [0.01, 0.01, 0.01, 0.01]
05-10 08:35:08 RWCA average train source weights: src0=0.0137, src1=0.2922, src2=0.6941
05-10 08:35:08 RWCA EMA source weights: src0=0.0003, src1=0.2860, src2=0.7137
05-10 08:35:08 BiMamba-Att residual gate: 0.010472
05-10 08:35:08 Max BiMamba-Att residual gate: 0.030000
05-10 08:35:08 MFSAN-CDAN-BiMamba-RWCA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_clmmd=0.020000, lambda_ent=0.005000
05-10 08:35:08 Train-Loss Source Classifier: 0.1329
05-10 08:35:08 Train-Loss MMD: 0.4676
05-10 08:35:08 Train-Loss CDD/L1: 0.0134
05-10 08:35:08 Train-Loss CDA MMD: 0.2263
05-10 08:35:08 Train-Loss CLMMD: 1.2125
05-10 08:35:08 Train-Loss Target Entropy: 0.5621
05-10 08:35:08 Train-Loss CDAN Domain: 0.6496
05-10 08:35:08 Train-Loss CDA Weighted: 0.0000
05-10 08:35:08 Train-Loss CLMMD Weighted: 0.0000
05-10 08:35:08 Train-Loss Entropy Weighted: 0.0000
05-10 08:35:08 Train-Loss CDAN Weighted: 0.0000
05-10 08:35:08 Train-Loss RW Weight src0: 0.0137
05-10 08:35:08 Train-Loss RW Weight src1: 0.2922
05-10 08:35:08 Train-Loss RW Weight src2: 0.6941
05-10 08:35:08 Train-Acc Source Data: 0.8399
05-10 08:35:08 Train-Acc Domain Data: 0.5524
05-10 08:35:10 Val-acc: 0.7282
05-10 08:35:10 Val-Class-0 | Precision: 0.9812 | Recall: 0.9910 | F1: 0.9861 | Support: 1003
05-10 08:35:10 Val-Class-1 | Precision: 0.7914 | Recall: 0.7170 | F1: 0.7524 | Support: 1000
05-10 08:35:10 Val-Class-2 | Precision: 0.9778 | Recall: 0.1320 | F1: 0.2326 | Support: 1000
05-10 08:35:10 Val-Class-3 | Precision: 0.9628 | Recall: 0.7500 | F1: 0.8432 | Support: 1000
05-10 08:35:10 Val-Class-4 | Precision: 0.4192 | Recall: 0.9264 | F1: 0.5772 | Support: 1005
05-10 08:35:10 Val-Class-5 | Precision: 0.6088 | Recall: 0.9980 | F1: 0.7562 | Support: 1001
05-10 08:35:10 Val-Class-6 | Precision: 0.9577 | Recall: 0.8831 | F1: 0.9189 | Support: 1001
05-10 08:35:10 Val-Class-7 | Precision: 0.9501 | Recall: 0.3570 | F1: 0.5190 | Support: 1014
05-10 08:35:10 Val-Class-8 | Precision: 0.7825 | Recall: 0.8020 | F1: 0.7921 | Support: 1005
05-10 08:35:10 Val-F1-macro: 0.7086
05-10 08:35:10 Val-F1-weighted: 0.7084
05-10 08:35:10 Val source fusion weights: src0=0.0000, src1=0.2937, src2=0.7063
05-10 08:35:10 Best model saved to ./ckpt/MFSAN_CDAN_BIMAMBA_RWCA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0510-082243_best.pth
05-10 08:35:10 Best model updated at epoch 1, val-acc 0.7282
05-10 08:35:10 The best model epoch 1, val-acc 0.7282
05-10 08:35:10 -----Epoch 2/10-----
05-10 08:35:10 current lr: [0.01, 0.01, 0.01, 0.01]
05-10 08:47:27 RWCA average train source weights: src0=0.0001, src1=0.4972, src2=0.5027
05-10 08:47:27 RWCA EMA source weights: src0=0.0001, src1=0.5037, src2=0.4962
05-10 08:47:27 BiMamba-Att residual gate: 0.010905
05-10 08:47:27 Max BiMamba-Att residual gate: 0.030000
05-10 08:47:27 MFSAN-CDAN-BiMamba-RWCA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_clmmd=0.020000, lambda_ent=0.005000
05-10 08:47:27 Train-Loss Source Classifier: 0.0094
05-10 08:47:27 Train-Loss MMD: 0.0943
05-10 08:47:27 Train-Loss CDD/L1: 0.0020
05-10 08:47:27 Train-Loss CDA MMD: 0.0949
05-10 08:47:27 Train-Loss CLMMD: 0.1446
05-10 08:47:27 Train-Loss Target Entropy: 0.1237
05-10 08:47:27 Train-Loss CDAN Domain: 0.6901
05-10 08:47:27 Train-Loss CDA Weighted: 0.0010
05-10 08:47:27 Train-Loss CLMMD Weighted: 0.0015
05-10 08:47:27 Train-Loss Entropy Weighted: 0.0003
05-10 08:47:27 Train-Loss CDAN Weighted: 0.0035
05-10 08:47:27 Train-Loss RW Weight src0: 0.0001
05-10 08:47:27 Train-Loss RW Weight src1: 0.4972
05-10 08:47:27 Train-Loss RW Weight src2: 0.5027
05-10 08:47:27 Train-Acc Source Data: 0.9171
05-10 08:47:27 Train-Acc Domain Data: 0.5415
05-10 08:47:29 Val-acc: 0.9777
05-10 08:47:29 Val-Class-0 | Precision: 0.9891 | Recall: 0.9970 | F1: 0.9930 | Support: 1003
05-10 08:47:29 Val-Class-1 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1000
05-10 08:47:29 Val-Class-2 | Precision: 0.9950 | Recall: 0.9970 | F1: 0.9960 | Support: 1000
05-10 08:47:29 Val-Class-3 | Precision: 0.9839 | Recall: 0.8580 | F1: 0.9167 | Support: 1000
05-10 08:47:29 Val-Class-4 | Precision: 0.9880 | Recall: 0.9801 | F1: 0.9840 | Support: 1005
05-10 08:47:29 Val-Class-5 | Precision: 0.9653 | Recall: 1.0000 | F1: 0.9823 | Support: 1001
05-10 08:47:29 Val-Class-6 | Precision: 0.8983 | Recall: 0.9880 | F1: 0.9410 | Support: 1001
05-10 08:47:29 Val-Class-7 | Precision: 0.9940 | Recall: 0.9862 | F1: 0.9901 | Support: 1014
05-10 08:47:29 Val-Class-8 | Precision: 0.9960 | Recall: 0.9930 | F1: 0.9945 | Support: 1005
05-10 08:47:29 Val-F1-macro: 0.9775
05-10 08:47:29 Val-F1-weighted: 0.9775
05-10 08:47:29 Val source fusion weights: src0=0.0000, src1=0.5015, src2=0.4985
05-10 08:47:29 Best model saved to ./ckpt/MFSAN_CDAN_BIMAMBA_RWCA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0510-082243_best.pth
05-10 08:47:29 Best model updated at epoch 2, val-acc 0.9777
05-10 08:47:29 The best model epoch 2, val-acc 0.9777
05-10 08:47:29 -----Epoch 3/10-----
05-10 08:47:29 current lr: [0.01, 0.01, 0.01, 0.01]
05-10 08:59:47 RWCA average train source weights: src0=0.0001, src1=0.5067, src2=0.4932
05-10 08:59:47 RWCA EMA source weights: src0=0.0001, src1=0.5073, src2=0.4926
05-10 08:59:47 BiMamba-Att residual gate: 0.011299
05-10 08:59:47 Max BiMamba-Att residual gate: 0.030000
05-10 08:59:47 MFSAN-CDAN-BiMamba-RWCA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_clmmd=0.020000, lambda_ent=0.005000
05-10 08:59:47 Train-Loss Source Classifier: 0.0108
05-10 08:59:47 Train-Loss MMD: 0.0882
05-10 08:59:47 Train-Loss CDD/L1: 0.0010
05-10 08:59:47 Train-Loss CDA MMD: 0.0885
05-10 08:59:47 Train-Loss CLMMD: 0.0920
05-10 08:59:47 Train-Loss Target Entropy: 0.1134
05-10 08:59:47 Train-Loss CDAN Domain: 0.6863
05-10 08:59:47 Train-Loss CDA Weighted: 0.0014
05-10 08:59:47 Train-Loss CLMMD Weighted: 0.0015
05-10 08:59:47 Train-Loss Entropy Weighted: 0.0005
05-10 08:59:47 Train-Loss CDAN Weighted: 0.0055
05-10 08:59:47 Train-Loss RW Weight src0: 0.0001
05-10 08:59:47 Train-Loss RW Weight src1: 0.5067
05-10 08:59:47 Train-Loss RW Weight src2: 0.4932
05-10 08:59:47 Train-Acc Source Data: 0.9096
05-10 08:59:47 Train-Acc Domain Data: 0.5581
05-10 08:59:49 Val-acc: 0.9795
05-10 08:59:49 Val-Class-0 | Precision: 0.9776 | Recall: 0.9990 | F1: 0.9882 | Support: 1003
05-10 08:59:49 Val-Class-1 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1000
05-10 08:59:49 Val-Class-2 | Precision: 0.9980 | Recall: 1.0000 | F1: 0.9990 | Support: 1000
05-10 08:59:49 Val-Class-3 | Precision: 0.9908 | Recall: 0.8570 | F1: 0.9190 | Support: 1000
05-10 08:59:49 Val-Class-4 | Precision: 0.9940 | Recall: 0.9831 | F1: 0.9885 | Support: 1005
05-10 08:59:49 Val-Class-5 | Precision: 0.9737 | Recall: 1.0000 | F1: 0.9867 | Support: 1001
05-10 08:59:49 Val-Class-6 | Precision: 0.9037 | Recall: 0.9940 | F1: 0.9467 | Support: 1001
05-10 08:59:49 Val-Class-7 | Precision: 0.9990 | Recall: 0.9832 | F1: 0.9911 | Support: 1014
05-10 08:59:49 Val-Class-8 | Precision: 0.9892 | Recall: 0.9990 | F1: 0.9941 | Support: 1005
05-10 08:59:49 Val-F1-macro: 0.9792
05-10 08:59:49 Val-F1-weighted: 0.9792
05-10 08:59:49 Val source fusion weights: src0=0.0000, src1=0.5064, src2=0.4936
05-10 08:59:49 Best model saved to ./ckpt/MFSAN_CDAN_BIMAMBA_RWCA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0510-082243_best.pth
05-10 08:59:49 Best model updated at epoch 3, val-acc 0.9795
05-10 08:59:49 The best model epoch 3, val-acc 0.9795
05-10 08:59:49 -----Epoch 4/10-----
05-10 08:59:49 current lr: [0.01, 0.01, 0.01, 0.01]
05-10 09:12:07 RWCA average train source weights: src0=0.0017, src1=0.5000, src2=0.4983
05-10 09:12:07 RWCA EMA source weights: src0=0.0289, src1=0.5524, src2=0.4187
05-10 09:12:07 BiMamba-Att residual gate: 0.011658
05-10 09:12:07 Max BiMamba-Att residual gate: 0.030000
05-10 09:12:07 MFSAN-CDAN-BiMamba-RWCA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_clmmd=0.020000, lambda_ent=0.005000
05-10 09:12:07 Train-Loss Source Classifier: 0.0146
05-10 09:12:07 Train-Loss MMD: 0.0849
05-10 09:12:07 Train-Loss CDD/L1: 0.0011
05-10 09:12:07 Train-Loss CDA MMD: 0.0855
05-10 09:12:07 Train-Loss CLMMD: 0.0732
05-10 09:12:07 Train-Loss Target Entropy: 0.1211
05-10 09:12:07 Train-Loss CDAN Domain: 0.6837
05-10 09:12:07 Train-Loss CDA Weighted: 0.0016
05-10 09:12:07 Train-Loss CLMMD Weighted: 0.0014
05-10 09:12:07 Train-Loss Entropy Weighted: 0.0006
05-10 09:12:07 Train-Loss CDAN Weighted: 0.0064
05-10 09:12:07 Train-Loss RW Weight src0: 0.0017
05-10 09:12:07 Train-Loss RW Weight src1: 0.5000
05-10 09:12:07 Train-Loss RW Weight src2: 0.4983
05-10 09:12:07 Train-Acc Source Data: 0.8974
05-10 09:12:07 Train-Acc Domain Data: 0.5617
05-10 09:12:10 Val-acc: 0.9781
05-10 09:12:10 Val-Class-0 | Precision: 0.9804 | Recall: 1.0000 | F1: 0.9901 | Support: 1003
05-10 09:12:10 Val-Class-1 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1000
05-10 09:12:10 Val-Class-2 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1000
05-10 09:12:10 Val-Class-3 | Precision: 0.9772 | Recall: 0.8560 | F1: 0.9126 | Support: 1000
05-10 09:12:10 Val-Class-4 | Precision: 0.9803 | Recall: 0.9881 | F1: 0.9841 | Support: 1005
05-10 09:12:10 Val-Class-5 | Precision: 0.9881 | Recall: 0.9990 | F1: 0.9935 | Support: 1001
05-10 09:12:10 Val-Class-6 | Precision: 0.8896 | Recall: 0.9980 | F1: 0.9407 | Support: 1001
05-10 09:12:10 Val-Class-7 | Precision: 1.0000 | Recall: 0.9901 | F1: 0.9950 | Support: 1014
05-10 09:12:10 Val-Class-8 | Precision: 1.0000 | Recall: 0.9711 | F1: 0.9854 | Support: 1005
05-10 09:12:10 Val-F1-macro: 0.9778
05-10 09:12:10 Val-F1-weighted: 0.9779
05-10 09:12:10 Val source fusion weights: src0=0.0006, src1=0.5701, src2=0.4293
05-10 09:12:10 The best model epoch 3, val-acc 0.9795
05-10 09:12:10 -----Epoch 5/10-----
05-10 09:12:10 current lr: [0.01, 0.01, 0.01, 0.01]
05-10 09:24:29 RWCA average train source weights: src0=0.0285, src1=0.4929, src2=0.4786
05-10 09:24:29 RWCA EMA source weights: src0=0.0472, src1=0.5494, src2=0.4034
05-10 09:24:29 BiMamba-Att residual gate: 0.011984
05-10 09:24:29 Max BiMamba-Att residual gate: 0.030000
05-10 09:24:29 MFSAN-CDAN-BiMamba-RWCA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_clmmd=0.020000, lambda_ent=0.005000
05-10 09:24:29 Train-Loss Source Classifier: 0.0383
05-10 09:24:29 Train-Loss MMD: 0.0757
05-10 09:24:29 Train-Loss CDD/L1: 0.0057
05-10 09:24:29 Train-Loss CDA MMD: 0.0765
05-10 09:24:29 Train-Loss CLMMD: 0.0708
05-10 09:24:29 Train-Loss Target Entropy: 0.1862
05-10 09:24:29 Train-Loss CDAN Domain: 0.6690
05-10 09:24:29 Train-Loss CDA Weighted: 0.0015
05-10 09:24:29 Train-Loss CLMMD Weighted: 0.0014
05-10 09:24:29 Train-Loss Entropy Weighted: 0.0009
05-10 09:24:29 Train-Loss CDAN Weighted: 0.0065
05-10 09:24:29 Train-Loss RW Weight src0: 0.0285
05-10 09:24:29 Train-Loss RW Weight src1: 0.4929
05-10 09:24:29 Train-Loss RW Weight src2: 0.4786
05-10 09:24:29 Train-Acc Source Data: 0.9128
05-10 09:24:29 Train-Acc Domain Data: 0.5644
05-10 09:24:31 Val-acc: 0.9804
05-10 09:24:31 Val-Class-0 | Precision: 0.9901 | Recall: 0.9990 | F1: 0.9945 | Support: 1003
05-10 09:24:31 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-10 09:24:31 Val-Class-2 | Precision: 0.9980 | Recall: 0.9990 | F1: 0.9985 | Support: 1000
05-10 09:24:31 Val-Class-3 | Precision: 0.9965 | Recall: 0.8620 | F1: 0.9244 | Support: 1000
05-10 09:24:31 Val-Class-4 | Precision: 0.9950 | Recall: 0.9930 | F1: 0.9940 | Support: 1005
05-10 09:24:31 Val-Class-5 | Precision: 0.9607 | Recall: 1.0000 | F1: 0.9799 | Support: 1001
05-10 09:24:31 Val-Class-6 | Precision: 0.9044 | Recall: 0.9830 | F1: 0.9421 | Support: 1001
05-10 09:24:31 Val-Class-7 | Precision: 0.9970 | Recall: 0.9882 | F1: 0.9926 | Support: 1014
05-10 09:24:31 Val-Class-8 | Precision: 0.9911 | Recall: 0.9990 | F1: 0.9950 | Support: 1005
05-10 09:24:31 Val-F1-macro: 0.9801
05-10 09:24:31 Val-F1-weighted: 0.9802
05-10 09:24:31 Val source fusion weights: src0=0.0150, src1=0.5698, src2=0.4153
05-10 09:24:31 Best model saved to ./ckpt/MFSAN_CDAN_BIMAMBA_RWCA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0510-082243_best.pth
05-10 09:24:31 Best model updated at epoch 5, val-acc 0.9804
05-10 09:24:31 The best model epoch 5, val-acc 0.9804
05-10 09:24:31 -----Epoch 6/10-----
05-10 09:24:31 current lr: [0.01, 0.01, 0.01, 0.01]
05-10 09:36:55 RWCA average train source weights: src0=0.0355, src1=0.4851, src2=0.4794
05-10 09:36:55 RWCA EMA source weights: src0=0.0404, src1=0.6690, src2=0.2906
05-10 09:36:55 BiMamba-Att residual gate: 0.012281
05-10 09:36:55 Max BiMamba-Att residual gate: 0.030000
05-10 09:36:55 MFSAN-CDAN-BiMamba-RWCA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_clmmd=0.020000, lambda_ent=0.005000
05-10 09:36:55 Train-Loss Source Classifier: 0.0216
05-10 09:36:55 Train-Loss MMD: 0.0722
05-10 09:36:55 Train-Loss CDD/L1: 0.0025
05-10 09:36:55 Train-Loss CDA MMD: 0.0729
05-10 09:36:55 Train-Loss CLMMD: 0.0578
05-10 09:36:55 Train-Loss Target Entropy: 0.1416
05-10 09:36:55 Train-Loss CDAN Domain: 0.6795
05-10 09:36:55 Train-Loss CDA Weighted: 0.0014
05-10 09:36:55 Train-Loss CLMMD Weighted: 0.0011
05-10 09:36:55 Train-Loss Entropy Weighted: 0.0007
05-10 09:36:55 Train-Loss CDAN Weighted: 0.0067
05-10 09:36:55 Train-Loss RW Weight src0: 0.0355
05-10 09:36:55 Train-Loss RW Weight src1: 0.4851
05-10 09:36:55 Train-Loss RW Weight src2: 0.4794
05-10 09:36:55 Train-Acc Source Data: 0.9765
05-10 09:36:55 Train-Acc Domain Data: 0.5766
05-10 09:36:57 Val-acc: 0.9778
05-10 09:36:57 Val-Class-0 | Precision: 0.9911 | Recall: 0.9990 | F1: 0.9950 | Support: 1003
05-10 09:36:57 Val-Class-1 | Precision: 0.9891 | Recall: 1.0000 | F1: 0.9945 | Support: 1000
05-10 09:36:57 Val-Class-2 | Precision: 0.9940 | Recall: 1.0000 | F1: 0.9970 | Support: 1000
05-10 09:36:57 Val-Class-3 | Precision: 0.9838 | Recall: 0.9120 | F1: 0.9465 | Support: 1000
05-10 09:36:57 Val-Class-4 | Precision: 0.9606 | Recall: 0.9950 | F1: 0.9775 | Support: 1005
05-10 09:36:57 Val-Class-5 | Precision: 0.9882 | Recall: 1.0000 | F1: 0.9940 | Support: 1001
05-10 09:36:57 Val-Class-6 | Precision: 0.9267 | Recall: 0.9850 | F1: 0.9550 | Support: 1001
05-10 09:36:57 Val-Class-7 | Precision: 1.0000 | Recall: 0.9122 | F1: 0.9541 | Support: 1014
05-10 09:36:57 Val-Class-8 | Precision: 0.9728 | Recall: 0.9980 | F1: 0.9853 | Support: 1005
05-10 09:36:57 Val-F1-macro: 0.9777
05-10 09:36:57 Val-F1-weighted: 0.9776
05-10 09:36:57 Val source fusion weights: src0=0.0273, src1=0.6780, src2=0.2947
05-10 09:36:57 The best model epoch 5, val-acc 0.9804
05-10 09:36:57 -----Epoch 7/10-----
05-10 09:36:57 current lr: [0.01, 0.01, 0.01, 0.01]
05-10 09:49:22 RWCA average train source weights: src0=0.0396, src1=0.4970, src2=0.4634
05-10 09:49:22 RWCA EMA source weights: src0=0.0425, src1=0.5890, src2=0.3685
05-10 09:49:22 BiMamba-Att residual gate: 0.012551
05-10 09:49:22 Max BiMamba-Att residual gate: 0.030000
05-10 09:49:22 MFSAN-CDAN-BiMamba-RWCA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_clmmd=0.020000, lambda_ent=0.005000
05-10 09:49:22 Train-Loss Source Classifier: 0.0175
05-10 09:49:22 Train-Loss MMD: 0.0711
05-10 09:49:22 Train-Loss CDD/L1: 0.0014
05-10 09:49:22 Train-Loss CDA MMD: 0.0716
05-10 09:49:22 Train-Loss CLMMD: 0.0526
05-10 09:49:22 Train-Loss Target Entropy: 0.1259
05-10 09:49:22 Train-Loss CDAN Domain: 0.6792
05-10 09:49:22 Train-Loss CDA Weighted: 0.0014
05-10 09:49:22 Train-Loss CLMMD Weighted: 0.0010
05-10 09:49:22 Train-Loss Entropy Weighted: 0.0006
05-10 09:49:22 Train-Loss CDAN Weighted: 0.0068
05-10 09:49:22 Train-Loss RW Weight src0: 0.0396
05-10 09:49:22 Train-Loss RW Weight src1: 0.4970
05-10 09:49:22 Train-Loss RW Weight src2: 0.4634
05-10 09:49:22 Train-Acc Source Data: 0.9889
05-10 09:49:22 Train-Acc Domain Data: 0.5784
05-10 09:49:24 Val-acc: 0.9941
05-10 09:49:24 Val-Class-0 | Precision: 0.9990 | Recall: 0.9980 | F1: 0.9985 | Support: 1003
05-10 09:49:24 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-10 09:49:24 Val-Class-2 | Precision: 0.9990 | Recall: 0.9980 | F1: 0.9985 | Support: 1000
05-10 09:49:24 Val-Class-3 | Precision: 0.9969 | Recall: 0.9650 | F1: 0.9807 | Support: 1000
05-10 09:49:24 Val-Class-4 | Precision: 0.9941 | Recall: 0.9980 | F1: 0.9960 | Support: 1005
05-10 09:49:24 Val-Class-5 | Precision: 0.9940 | Recall: 1.0000 | F1: 0.9970 | Support: 1001
05-10 09:49:24 Val-Class-6 | Precision: 0.9679 | Recall: 0.9950 | F1: 0.9813 | Support: 1001
05-10 09:49:24 Val-Class-7 | Precision: 0.9970 | Recall: 0.9931 | F1: 0.9951 | Support: 1014
05-10 09:49:24 Val-Class-8 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1005
05-10 09:49:24 Val-F1-macro: 0.9941
05-10 09:49:24 Val-F1-weighted: 0.9941
05-10 09:49:24 Val source fusion weights: src0=0.0324, src1=0.5973, src2=0.3703
05-10 09:49:24 Best model saved to ./ckpt/MFSAN_CDAN_BIMAMBA_RWCA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0510-082243_best.pth
05-10 09:49:24 Best model updated at epoch 7, val-acc 0.9941
05-10 09:49:24 The best model epoch 7, val-acc 0.9941
05-10 09:49:24 -----Epoch 8/10-----
05-10 09:49:24 current lr: [0.01, 0.01, 0.01, 0.01]
05-10 10:01:48 RWCA average train source weights: src0=0.0414, src1=0.4654, src2=0.4932
05-10 10:01:48 RWCA EMA source weights: src0=0.0261, src1=0.4016, src2=0.5723
05-10 10:01:48 BiMamba-Att residual gate: 0.012795
05-10 10:01:48 Max BiMamba-Att residual gate: 0.030000
05-10 10:01:48 MFSAN-CDAN-BiMamba-RWCA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_clmmd=0.020000, lambda_ent=0.005000
05-10 10:01:48 Train-Loss Source Classifier: 0.0154
05-10 10:01:48 Train-Loss MMD: 0.0721
05-10 10:01:48 Train-Loss CDD/L1: 0.0010
05-10 10:01:48 Train-Loss CDA MMD: 0.0729
05-10 10:01:48 Train-Loss CLMMD: 0.0451
05-10 10:01:48 Train-Loss Target Entropy: 0.1175
05-10 10:01:48 Train-Loss CDAN Domain: 0.6772
05-10 10:01:48 Train-Loss CDA Weighted: 0.0015
05-10 10:01:48 Train-Loss CLMMD Weighted: 0.0009
05-10 10:01:48 Train-Loss Entropy Weighted: 0.0006
05-10 10:01:48 Train-Loss CDAN Weighted: 0.0068
05-10 10:01:48 Train-Loss RW Weight src0: 0.0414
05-10 10:01:48 Train-Loss RW Weight src1: 0.4654
05-10 10:01:48 Train-Loss RW Weight src2: 0.4932
05-10 10:01:48 Train-Acc Source Data: 0.9933
05-10 10:01:48 Train-Acc Domain Data: 0.5835
05-10 10:01:50 Val-acc: 0.9911
05-10 10:01:50 Val-Class-0 | Precision: 0.9980 | Recall: 1.0000 | F1: 0.9990 | Support: 1003
05-10 10:01:50 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-10 10:01:50 Val-Class-2 | Precision: 1.0000 | Recall: 0.9980 | F1: 0.9990 | Support: 1000
05-10 10:01:50 Val-Class-3 | Precision: 0.9958 | Recall: 0.9480 | F1: 0.9713 | Support: 1000
05-10 10:01:50 Val-Class-4 | Precision: 0.9833 | Recall: 0.9980 | F1: 0.9906 | Support: 1005
05-10 10:01:50 Val-Class-5 | Precision: 0.9970 | Recall: 1.0000 | F1: 0.9985 | Support: 1001
05-10 10:01:50 Val-Class-6 | Precision: 0.9522 | Recall: 0.9950 | F1: 0.9731 | Support: 1001
05-10 10:01:50 Val-Class-7 | Precision: 1.0000 | Recall: 0.9842 | F1: 0.9920 | Support: 1014
05-10 10:01:50 Val-Class-8 | Precision: 0.9960 | Recall: 0.9970 | F1: 0.9965 | Support: 1005
05-10 10:01:50 Val-F1-macro: 0.9911
05-10 10:01:50 Val-F1-weighted: 0.9911
05-10 10:01:50 Val source fusion weights: src0=0.0213, src1=0.4042, src2=0.5744
05-10 10:01:50 The best model epoch 7, val-acc 0.9941
05-10 10:01:50 -----Epoch 9/10-----
05-10 10:01:50 current lr: [0.01, 0.01, 0.01, 0.01]
05-10 10:14:14 RWCA average train source weights: src0=0.0415, src1=0.4699, src2=0.4886
05-10 10:14:14 RWCA EMA source weights: src0=0.0438, src1=0.3158, src2=0.6403
05-10 10:14:14 BiMamba-Att residual gate: 0.013015
05-10 10:14:14 Max BiMamba-Att residual gate: 0.030000
05-10 10:14:14 MFSAN-CDAN-BiMamba-RWCA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_clmmd=0.020000, lambda_ent=0.005000
05-10 10:14:14 Train-Loss Source Classifier: 0.0149
05-10 10:14:14 Train-Loss MMD: 0.0710
05-10 10:14:14 Train-Loss CDD/L1: 0.0008
05-10 10:14:14 Train-Loss CDA MMD: 0.0719
05-10 10:14:14 Train-Loss CLMMD: 0.0424
05-10 10:14:14 Train-Loss Target Entropy: 0.1137
05-10 10:14:14 Train-Loss CDAN Domain: 0.6740
05-10 10:14:14 Train-Loss CDA Weighted: 0.0014
05-10 10:14:14 Train-Loss CLMMD Weighted: 0.0008
05-10 10:14:14 Train-Loss Entropy Weighted: 0.0006
05-10 10:14:14 Train-Loss CDAN Weighted: 0.0067
05-10 10:14:14 Train-Loss RW Weight src0: 0.0415
05-10 10:14:14 Train-Loss RW Weight src1: 0.4699
05-10 10:14:14 Train-Loss RW Weight src2: 0.4886
05-10 10:14:14 Train-Acc Source Data: 0.9943
05-10 10:14:14 Train-Acc Domain Data: 0.5942
05-10 10:14:16 Val-acc: 0.9914
05-10 10:14:16 Val-Class-0 | Precision: 0.9980 | Recall: 1.0000 | F1: 0.9990 | Support: 1003
05-10 10:14:16 Val-Class-1 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1000
05-10 10:14:16 Val-Class-2 | Precision: 0.9950 | Recall: 1.0000 | F1: 0.9975 | Support: 1000
05-10 10:14:16 Val-Class-3 | Precision: 0.9959 | Recall: 0.9710 | F1: 0.9833 | Support: 1000
05-10 10:14:16 Val-Class-4 | Precision: 0.9960 | Recall: 0.9920 | F1: 0.9940 | Support: 1005
05-10 10:14:16 Val-Class-5 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1001
05-10 10:14:16 Val-Class-6 | Precision: 0.9727 | Recall: 0.9980 | F1: 0.9852 | Support: 1001
05-10 10:14:16 Val-Class-7 | Precision: 1.0000 | Recall: 0.9615 | F1: 0.9804 | Support: 1014
05-10 10:14:16 Val-Class-8 | Precision: 0.9673 | Recall: 1.0000 | F1: 0.9834 | Support: 1005
05-10 10:14:16 Val-F1-macro: 0.9914
05-10 10:14:16 Val-F1-weighted: 0.9913
05-10 10:14:16 Val source fusion weights: src0=0.0352, src1=0.3163, src2=0.6485
05-10 10:14:16 The best model epoch 7, val-acc 0.9941
05-10 10:14:16 -----Epoch 10/10-----
05-10 10:14:16 current lr: [0.01, 0.01, 0.01, 0.01]
05-10 10:26:40 RWCA average train source weights: src0=0.0424, src1=0.4824, src2=0.4752
05-10 10:26:40 RWCA EMA source weights: src0=0.0354, src1=0.5329, src2=0.4317
05-10 10:26:40 BiMamba-Att residual gate: 0.013214
05-10 10:26:40 Max BiMamba-Att residual gate: 0.030000
05-10 10:26:40 MFSAN-CDAN-BiMamba-RWCA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_clmmd=0.020000, lambda_ent=0.005000
05-10 10:26:40 Train-Loss Source Classifier: 0.0145
05-10 10:26:40 Train-Loss MMD: 0.0702
05-10 10:26:40 Train-Loss CDD/L1: 0.0008
05-10 10:26:40 Train-Loss CDA MMD: 0.0712
05-10 10:26:40 Train-Loss CLMMD: 0.0391
05-10 10:26:40 Train-Loss Target Entropy: 0.1124
05-10 10:26:40 Train-Loss CDAN Domain: 0.6710
05-10 10:26:40 Train-Loss CDA Weighted: 0.0014
05-10 10:26:40 Train-Loss CLMMD Weighted: 0.0008
05-10 10:26:40 Train-Loss Entropy Weighted: 0.0006
05-10 10:26:40 Train-Loss CDAN Weighted: 0.0067
05-10 10:26:40 Train-Loss RW Weight src0: 0.0424
05-10 10:26:40 Train-Loss RW Weight src1: 0.4824
05-10 10:26:40 Train-Loss RW Weight src2: 0.4752
05-10 10:26:40 Train-Acc Source Data: 0.9955
05-10 10:26:40 Train-Acc Domain Data: 0.6019
05-10 10:26:42 Val-acc: 0.9973
05-10 10:26:42 Val-Class-0 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1003
05-10 10:26:42 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-10 10:26:42 Val-Class-2 | Precision: 0.9980 | Recall: 1.0000 | F1: 0.9990 | Support: 1000
05-10 10:26:42 Val-Class-3 | Precision: 0.9980 | Recall: 0.9890 | F1: 0.9935 | Support: 1000
05-10 10:26:42 Val-Class-4 | Precision: 0.9941 | Recall: 0.9980 | F1: 0.9960 | Support: 1005
05-10 10:26:42 Val-Class-5 | Precision: 0.9970 | Recall: 1.0000 | F1: 0.9985 | Support: 1001
05-10 10:26:42 Val-Class-6 | Precision: 0.9901 | Recall: 0.9980 | F1: 0.9940 | Support: 1001
05-10 10:26:42 Val-Class-7 | Precision: 1.0000 | Recall: 0.9931 | F1: 0.9965 | Support: 1014
05-10 10:26:42 Val-Class-8 | Precision: 1.0000 | Recall: 0.9980 | F1: 0.9990 | Support: 1005
05-10 10:26:42 Val-F1-macro: 0.9973
05-10 10:26:42 Val-F1-weighted: 0.9973
05-10 10:26:42 Val source fusion weights: src0=0.0296, src1=0.5361, src2=0.4343
05-10 10:26:42 Best model saved to ./ckpt/MFSAN_CDAN_BIMAMBA_RWCA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0510-082243_best.pth
05-10 10:26:42 Best model updated at epoch 10, val-acc 0.9973
05-10 10:26:42 The best model epoch 10, val-acc 0.9973
05-10 10:26:42 Model saved to ./ckpt/MFSAN_CDAN_BIMAMBA_RWCA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0510-082243.pth

