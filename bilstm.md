(MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels_bilstm# python train.py   --model_name MFSAN_CDAN_BLA   --source PU_0,PU_1,PU_2   --target PU_3   --train_mode multi_source   --data_dir /workspace/PU_TL   --signal_size 1024   --backbone CNN   --cuda_device 0   --max_epoch 15  --
lambda_adv 0.01   --lambda_grl 0.5   --lambda_cda 0.02   --lambda_ent 0.005   --adv_detach_prob True   --adv_use_entropy_weight True   --adv_conf_thresh 0.8   --include_fa
ults K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18
05-09 09:45:56 model_name: MFSAN_CDAN_BLA
05-09 09:45:56 source: PU_0,PU_1,PU_2
05-09 09:45:56 target: PU_3
05-09 09:45:56 data_dir: /workspace/PU_TL
05-09 09:45:56 train_mode: multi_source
05-09 09:45:56 cuda_device: 0
05-09 09:45:56 max_epoch: 15
05-09 09:45:56 batch_size: 64
05-09 09:45:56 signal_size: 1024
05-09 09:45:56 random_state: 10
05-09 09:45:56 include_faults: K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18
05-09 09:45:56 exclude_faults: 
05-09 09:45:56 opt: sgd
05-09 09:45:56 momentum: 0.9
05-09 09:45:56 betas: (0.9, 0.999)
05-09 09:45:56 weight_decay: 0.0005
05-09 09:45:56 lr: 0.01
05-09 09:45:56 lr_scheduler: stepLR
05-09 09:45:56 gamma: 0.2
05-09 09:45:56 steps: 10
05-09 09:45:56 backbone: CNN
05-09 09:45:56 num_workers: 4
05-09 09:45:56 normlize_type: -1-1
05-09 09:45:56 tradeoff: ['exp', 'exp', 'exp']
05-09 09:45:56 zeta: 10.0
05-09 09:45:56 dropout: 0.0
05-09 09:45:56 lambda_cda: 0.02
05-09 09:45:56 lambda_ent: 0.005
05-09 09:45:56 cda_detach_prob: True
05-09 09:45:56 lambda_adv: 0.01
05-09 09:45:56 lambda_grl: 0.5
05-09 09:45:56 adv_hidden_dim: 256
05-09 09:45:56 adv_detach_prob: True
05-09 09:45:56 adv_use_entropy_weight: True
05-09 09:45:56 adv_conf_thresh: 0.8
05-09 09:45:56 save: True
05-09 09:45:56 save_dir: ./ckpt
05-09 09:45:56 load_path: 
05-09 09:45:56 bla_gate_init: 0.01
05-09 09:45:56 bla_gate_max: 0.03
05-09 09:45:56 save_path: ./ckpt/MFSAN_CDAN_BLA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0509-094556
05-09 09:45:56 Source PU_0 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 09:45:56 Source PU_1 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 09:45:56 Source PU_2 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 09:45:56 Target PU_3 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 09:45:56 The scenario is: closed-set domain adaptation
05-09 09:45:57 using 1 / 1 gpus
05-09 09:45:59 Using model: MFSAN_CDAN_BLA
05-09 09:45:59 Requested backbone: CNN
05-09 09:45:59 Actual backbone: MSCNN_BiLSTM_Att_SmallGate
05-09 09:45:59 Backbone output dim: 640
05-09 09:45:59 Initial BiLSTM-Att residual gate: 0.010000
05-09 09:45:59 Max BiLSTM-Att residual gate: 0.030000
05-09 09:46:02 Source set PU_0 number of samples: 45103.
Label 0 has samples: 5001
Label 1 has samples: 5000
Label 2 has samples: 5005
Label 3 has samples: 5005
Label 4 has samples: 5012
Label 5 has samples: 5022
Label 6 has samples: 5019
Label 7 has samples: 5033
Label 8 has samples: 5006
05-09 09:46:02 Source set PU_1 number of samples: 45023.
Label 0 has samples: 5002
Label 1 has samples: 5006
Label 2 has samples: 5002
Label 3 has samples: 4999
Label 4 has samples: 5005
Label 5 has samples: 5004
Label 6 has samples: 5001
Label 7 has samples: 5004
Label 8 has samples: 5000
05-09 09:46:02 Source set PU_2 number of samples: 45143.
Label 0 has samples: 5001
Label 1 has samples: 5005
Label 2 has samples: 5013
Label 3 has samples: 5018
Label 4 has samples: 5008
Label 5 has samples: 5059
Label 6 has samples: 5000
Label 7 has samples: 5040
Label 8 has samples: 4999
05-09 09:46:03 Training set number of samples: 36053.
Label 0 has samples: 4011
Label 1 has samples: 4000
Label 2 has samples: 4000
Label 3 has samples: 4000
Label 4 has samples: 4004
Label 5 has samples: 4000
Label 6 has samples: 4004
Label 7 has samples: 4016
Label 8 has samples: 4018
05-09 09:46:03 Validation set number of samples: 9015.
Label 0 has samples: 1003
Label 1 has samples: 1000
Label 2 has samples: 1000
Label 3 has samples: 1000
Label 4 has samples: 1001
Label 5 has samples: 1000
Label 6 has samples: 1001
Label 7 has samples: 1005
Label 8 has samples: 1005
05-09 09:46:04 MFSAN-CDAN-BLA lambda_adv: 0.010000
05-09 09:46:04 MFSAN-CDAN-BLA lambda_grl: 0.500000
05-09 09:46:04 MFSAN-CDAN-BLA adv_detach_prob: True
05-09 09:46:04 MFSAN-CDAN-BLA adv_use_entropy_weight: True
05-09 09:46:04 MFSAN-CDAN-BLA adv_conf_thresh: 0.800000
05-09 09:46:04 MFSAN-CDAN-BLA lambda_cda: 0.020000
05-09 09:46:04 MFSAN-CDAN-BLA lambda_ent: 0.005000
05-09 09:46:04 MFSAN-CDAN-BLA joint feature dim: 9 x 40 = 360
05-09 09:46:04 -----Epoch 1/15-----
05-09 09:46:04 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [09:15<00:00,  3.80it/s]
05-09 09:55:20 Initial BiLSTM-Att residual gate: 0.010477
05-09 09:55:20 Max BiLSTM-Att residual gate: 0.030000
05-09 09:55:20 MFSAN-CDAN-BLA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 09:55:20 Train-Loss Source Classifier: 0.2086
05-09 09:55:20 Train-Loss MMD: 0.6913
05-09 09:55:20 Train-Loss L1: 0.0681
05-09 09:55:20 Train-Loss CDA MMD: 0.2818
05-09 09:55:20 Train-Loss Target Entropy: 0.8723
05-09 09:55:20 Train-Loss CDAN Domain: 0.7082
05-09 09:55:20 Train-Loss CDA Weighted: 0.0000
05-09 09:55:20 Train-Loss Entropy Weighted: 0.0000
05-09 09:55:20 Train-Loss CDAN Weighted: 0.0000
05-09 09:55:20 Train-Acc Source Data: 0.9254
05-09 09:55:20 Train-Acc Domain Data: 0.4389
100%|###################################################################################################################################| 141/141 [00:01<00:00, 108.75it/s]
05-09 09:55:22 Val-acc: 0.6757
05-09 09:55:22 Val-Class-0 | Precision: 0.9348 | Recall: 0.9292 | F1: 0.9320 | Support: 1003
05-09 09:55:22 Val-Class-1 | Precision: 0.9055 | Recall: 0.6130 | F1: 0.7311 | Support: 1000
05-09 09:55:22 Val-Class-2 | Precision: 0.8577 | Recall: 0.2290 | F1: 0.3615 | Support: 1000
05-09 09:55:22 Val-Class-3 | Precision: 0.9776 | Recall: 0.8310 | F1: 0.8984 | Support: 1000
05-09 09:55:22 Val-Class-4 | Precision: 0.7443 | Recall: 0.9421 | F1: 0.8316 | Support: 1001
05-09 09:55:22 Val-Class-5 | Precision: 0.1162 | Recall: 0.0280 | F1: 0.0451 | Support: 1000
05-09 09:55:22 Val-Class-6 | Precision: 0.9681 | Recall: 0.8492 | F1: 0.9047 | Support: 1001
05-09 09:55:22 Val-Class-7 | Precision: 0.9970 | Recall: 0.6647 | F1: 0.7976 | Support: 1005
05-09 09:55:22 Val-Class-8 | Precision: 0.3147 | Recall: 0.9920 | F1: 0.4778 | Support: 1005
05-09 09:55:22 Val-F1-macro: 0.6644
05-09 09:55:22 Val-F1-weighted: 0.6645
05-09 09:55:22 The best model epoch 1, val-acc 0.6757
05-09 09:55:22 -----Epoch 2/15-----
05-09 09:55:22 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [09:17<00:00,  3.79it/s]
05-09 10:04:41 Initial BiLSTM-Att residual gate: 0.010913
05-09 10:04:41 Max BiLSTM-Att residual gate: 0.030000
05-09 10:04:41 MFSAN-CDAN-BLA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 10:04:41 Train-Loss Source Classifier: 0.0251
05-09 10:04:41 Train-Loss MMD: 0.1084
05-09 10:04:41 Train-Loss L1: 0.0142
05-09 10:04:41 Train-Loss CDA MMD: 0.1084
05-09 10:04:41 Train-Loss Target Entropy: 0.1922
05-09 10:04:41 Train-Loss CDAN Domain: 0.6855
05-09 10:04:41 Train-Loss CDA Weighted: 0.0007
05-09 10:04:41 Train-Loss Entropy Weighted: 0.0003
05-09 10:04:41 Train-Loss CDAN Weighted: 0.0023
05-09 10:04:41 Train-Acc Source Data: 0.9940
05-09 10:04:41 Train-Acc Domain Data: 0.5600
100%|###################################################################################################################################| 141/141 [00:01<00:00, 112.57it/s]
05-09 10:04:42 Val-acc: 0.9642
05-09 10:04:42 Val-Class-0 | Precision: 0.9940 | Recall: 0.9910 | F1: 0.9925 | Support: 1003
05-09 10:04:42 Val-Class-1 | Precision: 0.9990 | Recall: 0.9990 | F1: 0.9990 | Support: 1000
05-09 10:04:42 Val-Class-2 | Precision: 0.9940 | Recall: 0.9980 | F1: 0.9960 | Support: 1000
05-09 10:04:42 Val-Class-3 | Precision: 0.9673 | Recall: 0.9180 | F1: 0.9420 | Support: 1000
05-09 10:04:42 Val-Class-4 | Precision: 0.9747 | Recall: 1.0000 | F1: 0.9872 | Support: 1001
05-09 10:04:42 Val-Class-5 | Precision: 0.9651 | Recall: 0.8560 | F1: 0.9073 | Support: 1000
05-09 10:04:42 Val-Class-6 | Precision: 0.9282 | Recall: 0.9690 | F1: 0.9482 | Support: 1001
05-09 10:04:42 Val-Class-7 | Precision: 0.9950 | Recall: 0.9811 | F1: 0.9880 | Support: 1005
05-09 10:04:42 Val-Class-8 | Precision: 0.8723 | Recall: 0.9652 | F1: 0.9164 | Support: 1005
05-09 10:04:42 Val-F1-macro: 0.9641
05-09 10:04:42 Val-F1-weighted: 0.9641
05-09 10:04:43 The best model epoch 2, val-acc 0.9642
05-09 10:04:43 -----Epoch 3/15-----
05-09 10:04:43 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [09:18<00:00,  3.78it/s]
05-09 10:14:02 Initial BiLSTM-Att residual gate: 0.011309
05-09 10:14:02 Max BiLSTM-Att residual gate: 0.030000
05-09 10:14:02 MFSAN-CDAN-BLA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 10:14:02 Train-Loss Source Classifier: 0.0155
05-09 10:14:02 Train-Loss MMD: 0.0939
05-09 10:14:02 Train-Loss L1: 0.0054
05-09 10:14:02 Train-Loss CDA MMD: 0.0950
05-09 10:14:02 Train-Loss Target Entropy: 0.1186
05-09 10:14:02 Train-Loss CDAN Domain: 0.6817
05-09 10:14:02 Train-Loss CDA Weighted: 0.0012
05-09 10:14:02 Train-Loss Entropy Weighted: 0.0004
05-09 10:14:02 Train-Loss CDAN Weighted: 0.0042
05-09 10:14:02 Train-Acc Source Data: 0.9984
05-09 10:14:02 Train-Acc Domain Data: 0.5696
100%|###################################################################################################################################| 141/141 [00:01<00:00, 104.06it/s]
05-09 10:14:04 Val-acc: 0.9857
05-09 10:14:04 Val-Class-0 | Precision: 1.0000 | Recall: 0.9960 | F1: 0.9980 | Support: 1003
05-09 10:14:04 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 10:14:04 Val-Class-2 | Precision: 0.9930 | Recall: 0.9980 | F1: 0.9955 | Support: 1000
05-09 10:14:04 Val-Class-3 | Precision: 0.9705 | Recall: 0.9880 | F1: 0.9792 | Support: 1000
05-09 10:14:04 Val-Class-4 | Precision: 0.9891 | Recall: 0.9990 | F1: 0.9940 | Support: 1001
05-09 10:14:04 Val-Class-5 | Precision: 0.9550 | Recall: 0.9760 | F1: 0.9654 | Support: 1000
05-09 10:14:04 Val-Class-6 | Precision: 0.9959 | Recall: 0.9650 | F1: 0.9802 | Support: 1001
05-09 10:14:04 Val-Class-7 | Precision: 1.0000 | Recall: 0.9642 | F1: 0.9818 | Support: 1005
05-09 10:14:04 Val-Class-8 | Precision: 0.9696 | Recall: 0.9851 | F1: 0.9773 | Support: 1005
05-09 10:14:04 Val-F1-macro: 0.9857
05-09 10:14:04 Val-F1-weighted: 0.9857
05-09 10:14:04 The best model epoch 3, val-acc 0.9857
05-09 10:14:04 -----Epoch 4/15-----
05-09 10:14:04 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [09:17<00:00,  3.79it/s]
05-09 10:23:23 Initial BiLSTM-Att residual gate: 0.011671
05-09 10:23:23 Max BiLSTM-Att residual gate: 0.030000
05-09 10:23:23 MFSAN-CDAN-BLA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 10:23:23 Train-Loss Source Classifier: 0.0150
05-09 10:23:23 Train-Loss MMD: 0.0927
05-09 10:23:23 Train-Loss L1: 0.0035
05-09 10:23:23 Train-Loss CDA MMD: 0.0940
05-09 10:23:23 Train-Loss Target Entropy: 0.1113
05-09 10:23:23 Train-Loss CDAN Domain: 0.6836
05-09 10:23:23 Train-Loss CDA Weighted: 0.0015
05-09 10:23:23 Train-Loss Entropy Weighted: 0.0004
05-09 10:23:23 Train-Loss CDAN Weighted: 0.0054
05-09 10:23:23 Train-Acc Source Data: 0.9991
05-09 10:23:23 Train-Acc Domain Data: 0.5606
100%|###################################################################################################################################| 141/141 [00:01<00:00, 112.15it/s]
05-09 10:23:24 Val-acc: 0.9901
05-09 10:23:24 Val-Class-0 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1003
05-09 10:23:24 Val-Class-1 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1000
05-09 10:23:24 Val-Class-2 | Precision: 0.9940 | Recall: 1.0000 | F1: 0.9970 | Support: 1000
05-09 10:23:24 Val-Class-3 | Precision: 0.9949 | Recall: 0.9660 | F1: 0.9802 | Support: 1000
05-09 10:23:24 Val-Class-4 | Precision: 0.9990 | Recall: 0.9990 | F1: 0.9990 | Support: 1001
05-09 10:23:24 Val-Class-5 | Precision: 0.9838 | Recall: 0.9710 | F1: 0.9774 | Support: 1000
05-09 10:23:24 Val-Class-6 | Precision: 0.9717 | Recall: 0.9950 | F1: 0.9832 | Support: 1001
05-09 10:23:24 Val-Class-7 | Precision: 1.0000 | Recall: 0.9881 | F1: 0.9940 | Support: 1005
05-09 10:23:24 Val-Class-8 | Precision: 0.9708 | Recall: 0.9920 | F1: 0.9813 | Support: 1005
05-09 10:23:24 Val-F1-macro: 0.9901
05-09 10:23:24 Val-F1-weighted: 0.9901
05-09 10:23:24 The best model epoch 4, val-acc 0.9901
05-09 10:23:24 -----Epoch 5/15-----
05-09 10:23:24 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [09:17<00:00,  3.79it/s]
05-09 10:32:44 Initial BiLSTM-Att residual gate: 0.011998
05-09 10:32:44 Max BiLSTM-Att residual gate: 0.030000
05-09 10:32:44 MFSAN-CDAN-BLA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 10:32:44 Train-Loss Source Classifier: 0.0153
05-09 10:32:44 Train-Loss MMD: 0.0915
05-09 10:32:44 Train-Loss L1: 0.0028
05-09 10:32:44 Train-Loss CDA MMD: 0.0939
05-09 10:32:44 Train-Loss Target Entropy: 0.1120
05-09 10:32:44 Train-Loss CDAN Domain: 0.6821
05-09 10:32:44 Train-Loss CDA Weighted: 0.0017
05-09 10:32:44 Train-Loss Entropy Weighted: 0.0005
05-09 10:32:44 Train-Loss CDAN Weighted: 0.0061
05-09 10:32:44 Train-Acc Source Data: 0.9994
05-09 10:32:44 Train-Acc Domain Data: 0.5672
100%|###################################################################################################################################| 141/141 [00:01<00:00, 112.31it/s]
05-09 10:32:45 Val-acc: 0.9898
05-09 10:32:45 Val-Class-0 | Precision: 1.0000 | Recall: 0.9900 | F1: 0.9950 | Support: 1003
05-09 10:32:45 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 10:32:45 Val-Class-2 | Precision: 0.9960 | Recall: 1.0000 | F1: 0.9980 | Support: 1000
05-09 10:32:45 Val-Class-3 | Precision: 0.9842 | Recall: 0.9970 | F1: 0.9906 | Support: 1000
05-09 10:32:45 Val-Class-4 | Precision: 0.9804 | Recall: 1.0000 | F1: 0.9901 | Support: 1001
05-09 10:32:45 Val-Class-5 | Precision: 0.9897 | Recall: 0.9610 | F1: 0.9751 | Support: 1000
05-09 10:32:45 Val-Class-6 | Precision: 0.9990 | Recall: 0.9790 | F1: 0.9889 | Support: 1001
05-09 10:32:45 Val-Class-7 | Precision: 1.0000 | Recall: 0.9900 | F1: 0.9950 | Support: 1005
05-09 10:32:45 Val-Class-8 | Precision: 0.9605 | Recall: 0.9910 | F1: 0.9755 | Support: 1005
05-09 10:32:45 Val-F1-macro: 0.9898
05-09 10:32:45 Val-F1-weighted: 0.9898
05-09 10:32:45 The best model epoch 4, val-acc 0.9901
05-09 10:32:45 -----Epoch 6/15-----
05-09 10:32:45 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [09:18<00:00,  3.78it/s]
05-09 10:42:05 Initial BiLSTM-Att residual gate: 0.012294
05-09 10:42:05 Max BiLSTM-Att residual gate: 0.030000
05-09 10:42:05 MFSAN-CDAN-BLA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 10:42:05 Train-Loss Source Classifier: 0.0160
05-09 10:42:05 Train-Loss MMD: 0.0891
05-09 10:42:05 Train-Loss L1: 0.0025
05-09 10:42:05 Train-Loss CDA MMD: 0.0915
05-09 10:42:05 Train-Loss Target Entropy: 0.1132
05-09 10:42:05 Train-Loss CDAN Domain: 0.6795
05-09 10:42:05 Train-Loss CDA Weighted: 0.0017
05-09 10:42:05 Train-Loss Entropy Weighted: 0.0005
05-09 10:42:05 Train-Loss CDAN Weighted: 0.0064
05-09 10:42:05 Train-Acc Source Data: 0.9994
05-09 10:42:05 Train-Acc Domain Data: 0.5727
100%|###################################################################################################################################| 141/141 [00:01<00:00, 112.06it/s]
05-09 10:42:06 Val-acc: 0.9946
05-09 10:42:06 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 10:42:06 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 10:42:06 Val-Class-2 | Precision: 0.9970 | Recall: 1.0000 | F1: 0.9985 | Support: 1000
05-09 10:42:06 Val-Class-3 | Precision: 0.9969 | Recall: 0.9800 | F1: 0.9884 | Support: 1000
05-09 10:42:06 Val-Class-4 | Precision: 1.0000 | Recall: 0.9990 | F1: 0.9995 | Support: 1001
05-09 10:42:06 Val-Class-5 | Precision: 0.9890 | Recall: 0.9910 | F1: 0.9900 | Support: 1000
05-09 10:42:06 Val-Class-6 | Precision: 0.9842 | Recall: 0.9970 | F1: 0.9906 | Support: 1001
05-09 10:42:06 Val-Class-7 | Precision: 1.0000 | Recall: 0.9910 | F1: 0.9955 | Support: 1005
05-09 10:42:06 Val-Class-8 | Precision: 0.9842 | Recall: 0.9930 | F1: 0.9886 | Support: 1005
05-09 10:42:06 Val-F1-macro: 0.9946
05-09 10:42:06 Val-F1-weighted: 0.9946
05-09 10:42:06 The best model epoch 6, val-acc 0.9946
05-09 10:42:06 -----Epoch 7/15-----
05-09 10:42:06 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [09:18<00:00,  3.78it/s]
05-09 10:51:26 Initial BiLSTM-Att residual gate: 0.012563
05-09 10:51:26 Max BiLSTM-Att residual gate: 0.030000
05-09 10:51:26 MFSAN-CDAN-BLA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 10:51:26 Train-Loss Source Classifier: 0.0160
05-09 10:51:26 Train-Loss MMD: 0.0892
05-09 10:51:26 Train-Loss L1: 0.0022
05-09 10:51:26 Train-Loss CDA MMD: 0.0923
05-09 10:51:26 Train-Loss Target Entropy: 0.1133
05-09 10:51:26 Train-Loss CDAN Domain: 0.6768
05-09 10:51:26 Train-Loss CDA Weighted: 0.0018
05-09 10:51:26 Train-Loss Entropy Weighted: 0.0006
05-09 10:51:26 Train-Loss CDAN Weighted: 0.0066
05-09 10:51:26 Train-Acc Source Data: 0.9995
05-09 10:51:26 Train-Acc Domain Data: 0.5812
100%|###################################################################################################################################| 141/141 [00:01<00:00, 112.16it/s]
05-09 10:51:27 Val-acc: 0.9942
05-09 10:51:27 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 10:51:27 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 10:51:27 Val-Class-2 | Precision: 0.9930 | Recall: 1.0000 | F1: 0.9965 | Support: 1000
05-09 10:51:27 Val-Class-3 | Precision: 0.9901 | Recall: 0.9960 | F1: 0.9930 | Support: 1000
05-09 10:51:27 Val-Class-4 | Precision: 0.9980 | Recall: 1.0000 | F1: 0.9990 | Support: 1001
05-09 10:51:27 Val-Class-5 | Precision: 0.9708 | Recall: 0.9990 | F1: 0.9847 | Support: 1000
05-09 10:51:27 Val-Class-6 | Precision: 0.9990 | Recall: 0.9880 | F1: 0.9935 | Support: 1001
05-09 10:51:27 Val-Class-7 | Precision: 1.0000 | Recall: 0.9881 | F1: 0.9940 | Support: 1005
05-09 10:51:27 Val-Class-8 | Precision: 0.9980 | Recall: 0.9771 | F1: 0.9874 | Support: 1005
05-09 10:51:27 Val-F1-macro: 0.9942
05-09 10:51:27 Val-F1-weighted: 0.9942
05-09 10:51:27 The best model epoch 6, val-acc 0.9946
05-09 10:51:27 -----Epoch 8/15-----
05-09 10:51:27 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [09:18<00:00,  3.78it/s]
05-09 11:00:47 Initial BiLSTM-Att residual gate: 0.012806
05-09 11:00:47 Max BiLSTM-Att residual gate: 0.030000
05-09 11:00:47 MFSAN-CDAN-BLA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 11:00:47 Train-Loss Source Classifier: 0.0169
05-09 11:00:47 Train-Loss MMD: 0.0890
05-09 11:00:47 Train-Loss L1: 0.0022
05-09 11:00:47 Train-Loss CDA MMD: 0.0931
05-09 11:00:47 Train-Loss Target Entropy: 0.1193
05-09 11:00:47 Train-Loss CDAN Domain: 0.6735
05-09 11:00:47 Train-Loss CDA Weighted: 0.0018
05-09 11:00:47 Train-Loss Entropy Weighted: 0.0006
05-09 11:00:47 Train-Loss CDAN Weighted: 0.0066
05-09 11:00:47 Train-Acc Source Data: 0.9994
05-09 11:00:47 Train-Acc Domain Data: 0.5885
100%|###################################################################################################################################| 141/141 [00:01<00:00, 103.27it/s]
05-09 11:00:48 Val-acc: 0.9930
05-09 11:00:48 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 11:00:48 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 11:00:48 Val-Class-2 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 11:00:48 Val-Class-3 | Precision: 0.9949 | Recall: 0.9830 | F1: 0.9889 | Support: 1000
05-09 11:00:48 Val-Class-4 | Precision: 1.0000 | Recall: 0.9990 | F1: 0.9995 | Support: 1001
05-09 11:00:48 Val-Class-5 | Precision: 0.9621 | Recall: 0.9890 | F1: 0.9753 | Support: 1000
05-09 11:00:48 Val-Class-6 | Precision: 0.9940 | Recall: 0.9950 | F1: 0.9945 | Support: 1001
05-09 11:00:48 Val-Class-7 | Precision: 0.9990 | Recall: 0.9990 | F1: 0.9990 | Support: 1005
05-09 11:00:48 Val-Class-8 | Precision: 0.9879 | Recall: 0.9721 | F1: 0.9799 | Support: 1005
05-09 11:00:48 Val-F1-macro: 0.9930
05-09 11:00:48 Val-F1-weighted: 0.9930
05-09 11:00:48 The best model epoch 6, val-acc 0.9946
05-09 11:00:48 -----Epoch 9/15-----
05-09 11:00:48 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [09:17<00:00,  3.79it/s]
05-09 11:10:08 Initial BiLSTM-Att residual gate: 0.013025
05-09 11:10:08 Max BiLSTM-Att residual gate: 0.030000
05-09 11:10:08 MFSAN-CDAN-BLA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 11:10:08 Train-Loss Source Classifier: 0.0160
05-09 11:10:08 Train-Loss MMD: 0.0882
05-09 11:10:08 Train-Loss L1: 0.0020
05-09 11:10:08 Train-Loss CDA MMD: 0.0928
05-09 11:10:08 Train-Loss Target Entropy: 0.1192
05-09 11:10:08 Train-Loss CDAN Domain: 0.6697
05-09 11:10:08 Train-Loss CDA Weighted: 0.0018
05-09 11:10:08 Train-Loss Entropy Weighted: 0.0006
05-09 11:10:08 Train-Loss CDAN Weighted: 0.0067
05-09 11:10:08 Train-Acc Source Data: 0.9997
05-09 11:10:08 Train-Acc Domain Data: 0.5966
100%|###################################################################################################################################| 141/141 [00:01<00:00, 111.46it/s]
05-09 11:10:09 Val-acc: 0.9952
05-09 11:10:09 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 11:10:09 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 11:10:09 Val-Class-2 | Precision: 1.0000 | Recall: 0.9990 | F1: 0.9995 | Support: 1000
05-09 11:10:09 Val-Class-3 | Precision: 0.9852 | Recall: 0.9990 | F1: 0.9921 | Support: 1000
05-09 11:10:09 Val-Class-4 | Precision: 1.0000 | Recall: 0.9990 | F1: 0.9995 | Support: 1001
05-09 11:10:09 Val-Class-5 | Precision: 0.9766 | Recall: 1.0000 | F1: 0.9881 | Support: 1000
05-09 11:10:09 Val-Class-6 | Precision: 0.9990 | Recall: 0.9850 | F1: 0.9920 | Support: 1001
05-09 11:10:09 Val-Class-7 | Precision: 1.0000 | Recall: 0.9970 | F1: 0.9985 | Support: 1005
05-09 11:10:09 Val-Class-8 | Precision: 0.9970 | Recall: 0.9781 | F1: 0.9874 | Support: 1005
05-09 11:10:09 Val-F1-macro: 0.9952
05-09 11:10:09 Val-F1-weighted: 0.9952
05-09 11:10:09 The best model epoch 9, val-acc 0.9952
05-09 11:10:09 -----Epoch 10/15-----
05-09 11:10:09 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [09:18<00:00,  3.78it/s]
05-09 11:19:29 Initial BiLSTM-Att residual gate: 0.013223
05-09 11:19:29 Max BiLSTM-Att residual gate: 0.030000
05-09 11:19:29 MFSAN-CDAN-BLA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 11:19:29 Train-Loss Source Classifier: 0.0169
05-09 11:19:29 Train-Loss MMD: 0.0862
05-09 11:19:29 Train-Loss L1: 0.0020
05-09 11:19:29 Train-Loss CDA MMD: 0.0918
05-09 11:19:29 Train-Loss Target Entropy: 0.1218
05-09 11:19:29 Train-Loss CDAN Domain: 0.6659
05-09 11:19:29 Train-Loss CDA Weighted: 0.0018
05-09 11:19:29 Train-Loss Entropy Weighted: 0.0006
05-09 11:19:29 Train-Loss CDAN Weighted: 0.0066
05-09 11:19:29 Train-Acc Source Data: 0.9995
05-09 11:19:29 Train-Acc Domain Data: 0.6035
100%|###################################################################################################################################| 141/141 [00:01<00:00, 111.44it/s]
05-09 11:19:30 Val-acc: 0.9933
05-09 11:19:30 Val-Class-0 | Precision: 1.0000 | Recall: 0.9990 | F1: 0.9995 | Support: 1003
05-09 11:19:30 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 11:19:30 Val-Class-2 | Precision: 1.0000 | Recall: 0.9970 | F1: 0.9985 | Support: 1000
05-09 11:19:30 Val-Class-3 | Precision: 0.9949 | Recall: 0.9810 | F1: 0.9879 | Support: 1000
05-09 11:19:30 Val-Class-4 | Precision: 0.9980 | Recall: 0.9990 | F1: 0.9985 | Support: 1001
05-09 11:19:30 Val-Class-5 | Precision: 0.9812 | Recall: 0.9910 | F1: 0.9861 | Support: 1000
05-09 11:19:30 Val-Class-6 | Precision: 0.9833 | Recall: 0.9970 | F1: 0.9901 | Support: 1001
05-09 11:19:30 Val-Class-7 | Precision: 1.0000 | Recall: 0.9960 | F1: 0.9980 | Support: 1005
05-09 11:19:30 Val-Class-8 | Precision: 0.9830 | Recall: 0.9801 | F1: 0.9816 | Support: 1005
05-09 11:19:30 Val-F1-macro: 0.9933
05-09 11:19:30 Val-F1-weighted: 0.9933
05-09 11:19:30 The best model epoch 9, val-acc 0.9952
05-09 11:19:30 -----Epoch 11/15-----
05-09 11:19:30 current lr: [0.002, 0.002, 0.002, 0.002]
100%|##################################################################################################################################| 2112/2112 [09:18<00:00,  3.78it/s]
05-09 11:28:50 Initial BiLSTM-Att residual gate: 0.013260
05-09 11:28:50 Max BiLSTM-Att residual gate: 0.030000
05-09 11:28:50 MFSAN-CDAN-BLA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 11:28:50 Train-Loss Source Classifier: 0.0139
05-09 11:28:50 Train-Loss MMD: 0.0827
05-09 11:28:50 Train-Loss L1: 0.0014
05-09 11:28:50 Train-Loss CDA MMD: 0.0903
05-09 11:28:50 Train-Loss Target Entropy: 0.1151
05-09 11:28:50 Train-Loss CDAN Domain: 0.6597
05-09 11:28:50 Train-Loss CDA Weighted: 0.0018
05-09 11:28:50 Train-Loss Entropy Weighted: 0.0006
05-09 11:28:50 Train-Loss CDAN Weighted: 0.0066
05-09 11:28:50 Train-Acc Source Data: 1.0000
05-09 11:28:50 Train-Acc Domain Data: 0.6272
100%|###################################################################################################################################| 141/141 [00:01<00:00, 113.51it/s]
05-09 11:28:51 Val-acc: 0.9957
05-09 11:28:51 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 11:28:51 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 11:28:51 Val-Class-2 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1000
05-09 11:28:51 Val-Class-3 | Precision: 0.9960 | Recall: 0.9980 | F1: 0.9970 | Support: 1000
05-09 11:28:51 Val-Class-4 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1001
05-09 11:28:51 Val-Class-5 | Precision: 0.9727 | Recall: 0.9980 | F1: 0.9852 | Support: 1000
05-09 11:28:51 Val-Class-6 | Precision: 0.9980 | Recall: 0.9960 | F1: 0.9970 | Support: 1001
05-09 11:28:51 Val-Class-7 | Precision: 0.9980 | Recall: 1.0000 | F1: 0.9990 | Support: 1005
05-09 11:28:51 Val-Class-8 | Precision: 0.9980 | Recall: 0.9692 | F1: 0.9833 | Support: 1005
05-09 11:28:51 Val-F1-macro: 0.9957
05-09 11:28:51 Val-F1-weighted: 0.9957
05-09 11:28:51 The best model epoch 11, val-acc 0.9957
05-09 11:28:51 -----Epoch 12/15-----
05-09 11:28:51 current lr: [0.002, 0.002, 0.002, 0.002]
100%|##################################################################################################################################| 2112/2112 [09:18<00:00,  3.78it/s]
05-09 11:38:11 Initial BiLSTM-Att residual gate: 0.013296
05-09 11:38:11 Max BiLSTM-Att residual gate: 0.030000
05-09 11:38:11 MFSAN-CDAN-BLA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 11:38:11 Train-Loss Source Classifier: 0.0140
05-09 11:38:11 Train-Loss MMD: 0.0821
05-09 11:38:11 Train-Loss L1: 0.0013
05-09 11:38:11 Train-Loss CDA MMD: 0.0904
05-09 11:38:11 Train-Loss Target Entropy: 0.1178
05-09 11:38:11 Train-Loss CDAN Domain: 0.6568
05-09 11:38:11 Train-Loss CDA Weighted: 0.0018
05-09 11:38:11 Train-Loss Entropy Weighted: 0.0006
05-09 11:38:11 Train-Loss CDAN Weighted: 0.0066
05-09 11:38:11 Train-Acc Source Data: 1.0000
05-09 11:38:11 Train-Acc Domain Data: 0.6348
100%|###################################################################################################################################| 141/141 [00:01<00:00, 111.14it/s]
05-09 11:38:12 Val-acc: 0.9971
05-09 11:38:12 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 11:38:12 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 11:38:12 Val-Class-2 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 11:38:12 Val-Class-3 | Precision: 0.9960 | Recall: 0.9980 | F1: 0.9970 | Support: 1000
05-09 11:38:12 Val-Class-4 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1001
05-09 11:38:12 Val-Class-5 | Precision: 0.9920 | Recall: 0.9880 | F1: 0.9900 | Support: 1000
05-09 11:38:12 Val-Class-6 | Precision: 0.9980 | Recall: 0.9960 | F1: 0.9970 | Support: 1001
05-09 11:38:12 Val-Class-7 | Precision: 1.0000 | Recall: 0.9990 | F1: 0.9995 | Support: 1005
05-09 11:38:12 Val-Class-8 | Precision: 0.9881 | Recall: 0.9930 | F1: 0.9906 | Support: 1005
05-09 11:38:12 Val-F1-macro: 0.9971
05-09 11:38:12 Val-F1-weighted: 0.9971
05-09 11:38:12 The best model epoch 12, val-acc 0.9971
05-09 11:38:12 -----Epoch 13/15-----
05-09 11:38:12 current lr: [0.002, 0.002, 0.002, 0.002]
100%|##################################################################################################################################| 2112/2112 [09:18<00:00,  3.78it/s]
05-09 11:47:32 Initial BiLSTM-Att residual gate: 0.013331
05-09 11:47:32 Max BiLSTM-Att residual gate: 0.030000
05-09 11:47:32 MFSAN-CDAN-BLA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 11:47:32 Train-Loss Source Classifier: 0.0143
05-09 11:47:32 Train-Loss MMD: 0.0812
05-09 11:47:32 Train-Loss L1: 0.0013
05-09 11:47:32 Train-Loss CDA MMD: 0.0899
05-09 11:47:32 Train-Loss Target Entropy: 0.1195
05-09 11:47:32 Train-Loss CDAN Domain: 0.6550
05-09 11:47:32 Train-Loss CDA Weighted: 0.0018
05-09 11:47:32 Train-Loss Entropy Weighted: 0.0006
05-09 11:47:32 Train-Loss CDAN Weighted: 0.0065
05-09 11:47:32 Train-Acc Source Data: 1.0000
05-09 11:47:32 Train-Acc Domain Data: 0.6373
100%|###################################################################################################################################| 141/141 [00:01<00:00, 113.25it/s]
05-09 11:47:33 Val-acc: 0.9967
05-09 11:47:33 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 11:47:33 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 11:47:33 Val-Class-2 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 11:47:33 Val-Class-3 | Precision: 0.9950 | Recall: 0.9990 | F1: 0.9970 | Support: 1000
05-09 11:47:33 Val-Class-4 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1001
05-09 11:47:33 Val-Class-5 | Precision: 0.9919 | Recall: 0.9840 | F1: 0.9880 | Support: 1000
05-09 11:47:33 Val-Class-6 | Precision: 0.9990 | Recall: 0.9950 | F1: 0.9970 | Support: 1001
05-09 11:47:33 Val-Class-7 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1005
05-09 11:47:33 Val-Class-8 | Precision: 0.9852 | Recall: 0.9920 | F1: 0.9886 | Support: 1005
05-09 11:47:33 Val-F1-macro: 0.9967
05-09 11:47:33 Val-F1-weighted: 0.9967
05-09 11:47:33 The best model epoch 12, val-acc 0.9971
05-09 11:47:33 -----Epoch 14/15-----
05-09 11:47:33 current lr: [0.002, 0.002, 0.002, 0.002]
100%|##################################################################################################################################| 2112/2112 [09:18<00:00,  3.78it/s]
05-09 11:56:53 Initial BiLSTM-Att residual gate: 0.013366
05-09 11:56:53 Max BiLSTM-Att residual gate: 0.030000
05-09 11:56:53 MFSAN-CDAN-BLA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 11:56:53 Train-Loss Source Classifier: 0.0142
05-09 11:56:53 Train-Loss MMD: 0.0792
05-09 11:56:53 Train-Loss L1: 0.0012
05-09 11:56:53 Train-Loss CDA MMD: 0.0886
05-09 11:56:53 Train-Loss Target Entropy: 0.1194
05-09 11:56:53 Train-Loss CDAN Domain: 0.6529
05-09 11:56:53 Train-Loss CDA Weighted: 0.0018
05-09 11:56:53 Train-Loss Entropy Weighted: 0.0006
05-09 11:56:53 Train-Loss CDAN Weighted: 0.0065
05-09 11:56:53 Train-Acc Source Data: 1.0000
05-09 11:56:53 Train-Acc Domain Data: 0.6415
100%|###################################################################################################################################| 141/141 [00:01<00:00, 105.82it/s]
05-09 11:56:54 Val-acc: 0.9973
05-09 11:56:54 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 11:56:54 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 11:56:54 Val-Class-2 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 11:56:54 Val-Class-3 | Precision: 0.9960 | Recall: 0.9980 | F1: 0.9970 | Support: 1000
05-09 11:56:54 Val-Class-4 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1001
05-09 11:56:54 Val-Class-5 | Precision: 0.9881 | Recall: 0.9950 | F1: 0.9915 | Support: 1000
05-09 11:56:54 Val-Class-6 | Precision: 0.9980 | Recall: 0.9960 | F1: 0.9970 | Support: 1001
05-09 11:56:54 Val-Class-7 | Precision: 0.9990 | Recall: 0.9990 | F1: 0.9990 | Support: 1005
05-09 11:56:54 Val-Class-8 | Precision: 0.9950 | Recall: 0.9881 | F1: 0.9915 | Support: 1005
05-09 11:56:54 Val-F1-macro: 0.9973
05-09 11:56:54 Val-F1-weighted: 0.9973
05-09 11:56:54 The best model epoch 14, val-acc 0.9973
05-09 11:56:54 -----Epoch 15/15-----
05-09 11:56:54 current lr: [0.002, 0.002, 0.002, 0.002]
100%|##################################################################################################################################| 2112/2112 [09:18<00:00,  3.78it/s]
05-09 12:06:14 Initial BiLSTM-Att residual gate: 0.013400
05-09 12:06:14 Max BiLSTM-Att residual gate: 0.030000
05-09 12:06:14 MFSAN-CDAN-BLA active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 12:06:14 Train-Loss Source Classifier: 0.0145
05-09 12:06:14 Train-Loss MMD: 0.0814
05-09 12:06:14 Train-Loss L1: 0.0012
05-09 12:06:14 Train-Loss CDA MMD: 0.0917
05-09 12:06:14 Train-Loss Target Entropy: 0.1239
05-09 12:06:14 Train-Loss CDAN Domain: 0.6503
05-09 12:06:14 Train-Loss CDA Weighted: 0.0018
05-09 12:06:14 Train-Loss Entropy Weighted: 0.0006
05-09 12:06:14 Train-Loss CDAN Weighted: 0.0065
05-09 12:06:14 Train-Acc Source Data: 1.0000
05-09 12:06:14 Train-Acc Domain Data: 0.6453
100%|###################################################################################################################################| 141/141 [00:01<00:00, 112.45it/s]
05-09 12:06:15 Val-acc: 0.9977
05-09 12:06:15 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 12:06:15 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 12:06:15 Val-Class-2 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 12:06:15 Val-Class-3 | Precision: 0.9950 | Recall: 1.0000 | F1: 0.9975 | Support: 1000
05-09 12:06:15 Val-Class-4 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1001
05-09 12:06:15 Val-Class-5 | Precision: 0.9900 | Recall: 0.9950 | F1: 0.9925 | Support: 1000
05-09 12:06:15 Val-Class-6 | Precision: 1.0000 | Recall: 0.9950 | F1: 0.9975 | Support: 1001
05-09 12:06:15 Val-Class-7 | Precision: 0.9990 | Recall: 0.9990 | F1: 0.9990 | Support: 1005
05-09 12:06:15 Val-Class-8 | Precision: 0.9950 | Recall: 0.9900 | F1: 0.9925 | Support: 1005
05-09 12:06:15 Val-F1-macro: 0.9977
05-09 12:06:15 Val-F1-weighted: 0.9977
05-09 12:06:15 The best model epoch 15, val-acc 0.9977
05-09 12:06:16 Model saved to ./ckpt/MFSAN_CDAN_BLA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0509-094556.pth
(MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels_bilstm# 