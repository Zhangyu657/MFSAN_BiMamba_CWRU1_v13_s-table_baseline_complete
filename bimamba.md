MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels_bimamba# python train.py \
>   --model_name MFSAN_CDAN_BIMAMBA \
>   --source PU_0,PU_1,PU_2 \
>   --target PU_3 \
>   --train_mode multi_source \
>   --data_dir /workspace/PU_TL \
>   --signal_size 1024 \
>   --backbone CNN \
>   --cuda_device 0 \
>   --max_epoch 15 \
>   --lambda_adv 0.01 \
>   --lambda_grl 0.5 \
>   --lambda_cda 0.02 \
>   --lambda_ent 0.005 \
>   --adv_detach_prob True \
>   --adv_use_entropy_weight True \
>   --adv_conf_thresh 0.8 \
>   --include_faults K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18
05-09 12:23:00 model_name: MFSAN_CDAN_BIMAMBA
05-09 12:23:00 source: PU_0,PU_1,PU_2
05-09 12:23:00 target: PU_3
05-09 12:23:00 data_dir: /workspace/PU_TL
05-09 12:23:00 train_mode: multi_source
05-09 12:23:00 cuda_device: 0
05-09 12:23:00 max_epoch: 15
05-09 12:23:00 batch_size: 64
05-09 12:23:00 signal_size: 1024
05-09 12:23:00 random_state: 10
05-09 12:23:00 include_faults: K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18
05-09 12:23:00 exclude_faults: 
05-09 12:23:00 opt: sgd
05-09 12:23:00 momentum: 0.9
05-09 12:23:00 betas: (0.9, 0.999)
05-09 12:23:00 weight_decay: 0.0005
05-09 12:23:00 lr: 0.01
05-09 12:23:00 lr_scheduler: stepLR
05-09 12:23:00 gamma: 0.2
05-09 12:23:00 steps: 10
05-09 12:23:00 backbone: CNN
05-09 12:23:00 num_workers: 4
05-09 12:23:00 normlize_type: -1-1
05-09 12:23:00 tradeoff: ['exp', 'exp', 'exp']
05-09 12:23:00 zeta: 10.0
05-09 12:23:00 dropout: 0.0
05-09 12:23:00 lambda_cda: 0.02
05-09 12:23:00 lambda_ent: 0.005
05-09 12:23:00 cda_detach_prob: True
05-09 12:23:00 lambda_adv: 0.01
05-09 12:23:00 lambda_grl: 0.5
05-09 12:23:00 adv_hidden_dim: 256
05-09 12:23:00 adv_detach_prob: True
05-09 12:23:00 adv_use_entropy_weight: True
05-09 12:23:00 adv_conf_thresh: 0.8
05-09 12:23:00 save: True
05-09 12:23:00 save_dir: ./ckpt
05-09 12:23:00 load_path: 
05-09 12:23:00 bla_gate_init: 0.01
05-09 12:23:00 bla_gate_max: 0.03
05-09 12:23:00 save_path: ./ckpt/MFSAN_CDAN_BIMAMBA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0509-122300
05-09 12:23:00 Source PU_0 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 12:23:00 Source PU_1 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 12:23:00 Source PU_2 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 12:23:00 Target PU_3 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 12:23:00 The scenario is: closed-set domain adaptation
05-09 12:23:01 using 1 / 1 gpus
05-09 12:23:03 Using model: MFSAN_CDAN_BIMAMBA
05-09 12:23:03 Requested backbone: CNN
05-09 12:23:03 Actual backbone: MSCNN_BiMamba_Att_SmallGate
05-09 12:23:03 Backbone output dim: 640
05-09 12:23:03 BiMamba implementation: mamba_ssm
05-09 12:23:03 Initial BiMamba-Att residual gate: 0.010000
05-09 12:23:03 Max BiMamba-Att residual gate: 0.030000
05-09 12:23:06 Source set PU_0 number of samples: 45103.
Label 0 has samples: 5001
Label 1 has samples: 5000
Label 2 has samples: 5005
Label 3 has samples: 5005
Label 4 has samples: 5012
Label 5 has samples: 5022
Label 6 has samples: 5019
Label 7 has samples: 5033
Label 8 has samples: 5006
05-09 12:23:06 Source set PU_1 number of samples: 45023.
Label 0 has samples: 5002
Label 1 has samples: 5006
Label 2 has samples: 5002
Label 3 has samples: 4999
Label 4 has samples: 5005
Label 5 has samples: 5004
Label 6 has samples: 5001
Label 7 has samples: 5004
Label 8 has samples: 5000
05-09 12:23:06 Source set PU_2 number of samples: 45143.
Label 0 has samples: 5001
Label 1 has samples: 5005
Label 2 has samples: 5013
Label 3 has samples: 5018
Label 4 has samples: 5008
Label 5 has samples: 5059
Label 6 has samples: 5000
Label 7 has samples: 5040
Label 8 has samples: 4999
05-09 12:23:07 Training set number of samples: 36053.
Label 0 has samples: 4011
Label 1 has samples: 4000
Label 2 has samples: 4000
Label 3 has samples: 4000
Label 4 has samples: 4004
Label 5 has samples: 4000
Label 6 has samples: 4004
Label 7 has samples: 4016
Label 8 has samples: 4018
05-09 12:23:07 Validation set number of samples: 9015.
Label 0 has samples: 1003
Label 1 has samples: 1000
Label 2 has samples: 1000
Label 3 has samples: 1000
Label 4 has samples: 1001
Label 5 has samples: 1000
Label 6 has samples: 1001
Label 7 has samples: 1005
Label 8 has samples: 1005
05-09 12:23:08 MFSAN-CDAN-BiMamba lambda_adv: 0.010000
05-09 12:23:08 MFSAN-CDAN-BiMamba lambda_grl: 0.500000
05-09 12:23:08 MFSAN-CDAN-BiMamba adv_detach_prob: True
05-09 12:23:08 MFSAN-CDAN-BiMamba adv_use_entropy_weight: True
05-09 12:23:08 MFSAN-CDAN-BiMamba adv_conf_thresh: 0.800000
05-09 12:23:08 MFSAN-CDAN-BiMamba lambda_cda: 0.020000
05-09 12:23:08 MFSAN-CDAN-BiMamba lambda_ent: 0.005000
05-09 12:23:08 MFSAN-CDAN-BiMamba joint feature dim: 9 x 40 = 360
05-09 12:23:08 -----Epoch 1/15-----
05-09 12:23:08 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [10:50<00:00,  3.25it/s]
05-09 12:33:58 BiMamba-Att residual gate: 0.010477
05-09 12:33:58 Max BiMamba-Att residual gate: 0.030000
05-09 12:33:58 MFSAN-CDAN-BiMamba active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 12:33:58 Train-Loss Source Classifier: 0.2125
05-09 12:33:58 Train-Loss MMD: 0.6271
05-09 12:33:58 Train-Loss L1: 0.0689
05-09 12:33:58 Train-Loss CDA MMD: 0.2828
05-09 12:33:58 Train-Loss Target Entropy: 0.8461
05-09 12:33:58 Train-Loss CDAN Domain: 0.6486
05-09 12:33:58 Train-Loss CDA Weighted: 0.0000
05-09 12:33:58 Train-Loss Entropy Weighted: 0.0000
05-09 12:33:58 Train-Loss CDAN Weighted: 0.0000
05-09 12:33:58 Train-Acc Source Data: 0.9264
05-09 12:33:58 Train-Acc Domain Data: 0.5268
100%|####################################################################################################################################| 141/141 [00:02<00:00, 70.03it/s]
05-09 12:34:01 Val-acc: 0.6573
05-09 12:34:01 Val-Class-0 | Precision: 0.7099 | Recall: 0.9980 | F1: 0.8297 | Support: 1003
05-09 12:34:01 Val-Class-1 | Precision: 0.7669 | Recall: 0.9770 | F1: 0.8593 | Support: 1000
05-09 12:34:01 Val-Class-2 | Precision: 0.9911 | Recall: 0.1110 | F1: 0.1996 | Support: 1000
05-09 12:34:01 Val-Class-3 | Precision: 0.9547 | Recall: 0.5690 | F1: 0.7130 | Support: 1000
05-09 12:34:01 Val-Class-4 | Precision: 0.8902 | Recall: 0.6074 | F1: 0.7221 | Support: 1001
05-09 12:34:01 Val-Class-5 | Precision: 0.3482 | Recall: 0.7970 | F1: 0.4846 | Support: 1000
05-09 12:34:01 Val-Class-6 | Precision: 0.9452 | Recall: 0.9301 | F1: 0.9376 | Support: 1001
05-09 12:34:01 Val-Class-7 | Precision: 0.8712 | Recall: 0.4577 | F1: 0.6001 | Support: 1005
05-09 12:34:01 Val-Class-8 | Precision: 0.4148 | Recall: 0.4697 | F1: 0.4405 | Support: 1005
05-09 12:34:01 Val-F1-macro: 0.6430
05-09 12:34:01 Val-F1-weighted: 0.6429
05-09 12:34:01 The best model epoch 1, val-acc 0.6573
05-09 12:34:01 -----Epoch 2/15-----
05-09 12:34:01 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [10:53<00:00,  3.23it/s]
05-09 12:44:56 BiMamba-Att residual gate: 0.010909
05-09 12:44:56 Max BiMamba-Att residual gate: 0.030000
05-09 12:44:56 MFSAN-CDAN-BiMamba active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 12:44:56 Train-Loss Source Classifier: 0.0245
05-09 12:44:56 Train-Loss MMD: 0.1093
05-09 12:44:56 Train-Loss L1: 0.0147
05-09 12:44:56 Train-Loss CDA MMD: 0.1082
05-09 12:44:56 Train-Loss Target Entropy: 0.2018
05-09 12:44:56 Train-Loss CDAN Domain: 0.6860
05-09 12:44:56 Train-Loss CDA Weighted: 0.0007
05-09 12:44:56 Train-Loss Entropy Weighted: 0.0003
05-09 12:44:56 Train-Loss CDAN Weighted: 0.0024
05-09 12:44:56 Train-Acc Source Data: 0.9942
05-09 12:44:56 Train-Acc Domain Data: 0.5599
100%|####################################################################################################################################| 141/141 [00:01<00:00, 70.65it/s]
05-09 12:44:58 Val-acc: 0.9526
05-09 12:44:58 Val-Class-0 | Precision: 0.9950 | Recall: 0.9960 | F1: 0.9955 | Support: 1003
05-09 12:44:58 Val-Class-1 | Precision: 0.9970 | Recall: 1.0000 | F1: 0.9985 | Support: 1000
05-09 12:44:58 Val-Class-2 | Precision: 0.9911 | Recall: 0.9990 | F1: 0.9950 | Support: 1000
05-09 12:44:58 Val-Class-3 | Precision: 0.9921 | Recall: 0.8780 | F1: 0.9316 | Support: 1000
05-09 12:44:58 Val-Class-4 | Precision: 0.9795 | Recall: 1.0000 | F1: 0.9896 | Support: 1001
05-09 12:44:58 Val-Class-5 | Precision: 0.9647 | Recall: 0.7660 | F1: 0.8540 | Support: 1000
05-09 12:44:58 Val-Class-6 | Precision: 0.9048 | Recall: 0.9870 | F1: 0.9441 | Support: 1001
05-09 12:44:58 Val-Class-7 | Precision: 0.9858 | Recall: 0.9701 | F1: 0.9779 | Support: 1005
05-09 12:44:58 Val-Class-8 | Precision: 0.8062 | Recall: 0.9771 | F1: 0.8835 | Support: 1005
05-09 12:44:58 Val-F1-macro: 0.9522
05-09 12:44:58 Val-F1-weighted: 0.9522
05-09 12:44:59 The best model epoch 2, val-acc 0.9526
05-09 12:44:59 -----Epoch 3/15-----
05-09 12:44:59 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [10:54<00:00,  3.23it/s]
05-09 12:55:54 BiMamba-Att residual gate: 0.011304
05-09 12:55:54 Max BiMamba-Att residual gate: 0.030000
05-09 12:55:54 MFSAN-CDAN-BiMamba active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 12:55:54 Train-Loss Source Classifier: 0.0152
05-09 12:55:54 Train-Loss MMD: 0.0943
05-09 12:55:54 Train-Loss L1: 0.0056
05-09 12:55:54 Train-Loss CDA MMD: 0.0957
05-09 12:55:54 Train-Loss Target Entropy: 0.1225
05-09 12:55:54 Train-Loss CDAN Domain: 0.6828
05-09 12:55:54 Train-Loss CDA Weighted: 0.0012
05-09 12:55:54 Train-Loss Entropy Weighted: 0.0004
05-09 12:55:54 Train-Loss CDAN Weighted: 0.0042
05-09 12:55:54 Train-Acc Source Data: 0.9982
05-09 12:55:54 Train-Acc Domain Data: 0.5625
100%|####################################################################################################################################| 141/141 [00:02<00:00, 69.76it/s]
05-09 12:55:57 Val-acc: 0.9879
05-09 12:55:57 Val-Class-0 | Precision: 1.0000 | Recall: 0.9980 | F1: 0.9990 | Support: 1003
05-09 12:55:57 Val-Class-1 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1000
05-09 12:55:57 Val-Class-2 | Precision: 0.9970 | Recall: 1.0000 | F1: 0.9985 | Support: 1000
05-09 12:55:57 Val-Class-3 | Precision: 0.9878 | Recall: 0.9730 | F1: 0.9804 | Support: 1000
05-09 12:55:57 Val-Class-4 | Precision: 1.0000 | Recall: 0.9990 | F1: 0.9995 | Support: 1001
05-09 12:55:57 Val-Class-5 | Precision: 0.9415 | Recall: 0.9980 | F1: 0.9689 | Support: 1000
05-09 12:55:57 Val-Class-6 | Precision: 0.9745 | Recall: 0.9920 | F1: 0.9832 | Support: 1001
05-09 12:55:57 Val-Class-7 | Precision: 0.9970 | Recall: 0.9891 | F1: 0.9930 | Support: 1005
05-09 12:55:57 Val-Class-8 | Precision: 0.9979 | Recall: 0.9423 | F1: 0.9693 | Support: 1005
05-09 12:55:57 Val-F1-macro: 0.9879
05-09 12:55:57 Val-F1-weighted: 0.9879
05-09 12:55:57 The best model epoch 3, val-acc 0.9879
05-09 12:55:57 -----Epoch 4/15-----
05-09 12:55:57 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [10:55<00:00,  3.22it/s]
05-09 13:06:54 BiMamba-Att residual gate: 0.011665
05-09 13:06:54 Max BiMamba-Att residual gate: 0.030000
05-09 13:06:54 MFSAN-CDAN-BiMamba active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 13:06:54 Train-Loss Source Classifier: 0.0149
05-09 13:06:54 Train-Loss MMD: 0.0937
05-09 13:06:54 Train-Loss L1: 0.0036
05-09 13:06:54 Train-Loss CDA MMD: 0.0947
05-09 13:06:54 Train-Loss Target Entropy: 0.1141
05-09 13:06:54 Train-Loss CDAN Domain: 0.6848
05-09 13:06:54 Train-Loss CDA Weighted: 0.0015
05-09 13:06:54 Train-Loss Entropy Weighted: 0.0005
05-09 13:06:54 Train-Loss CDAN Weighted: 0.0054
05-09 13:06:54 Train-Acc Source Data: 0.9991
05-09 13:06:54 Train-Acc Domain Data: 0.5546
100%|####################################################################################################################################| 141/141 [00:01<00:00, 71.07it/s]
05-09 13:06:56 Val-acc: 0.9929
05-09 13:06:56 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 13:06:56 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 13:06:56 Val-Class-2 | Precision: 0.9980 | Recall: 1.0000 | F1: 0.9990 | Support: 1000
05-09 13:06:56 Val-Class-3 | Precision: 0.9949 | Recall: 0.9750 | F1: 0.9848 | Support: 1000
05-09 13:06:56 Val-Class-4 | Precision: 1.0000 | Recall: 0.9990 | F1: 0.9995 | Support: 1001
05-09 13:06:56 Val-Class-5 | Precision: 0.9919 | Recall: 0.9770 | F1: 0.9844 | Support: 1000
05-09 13:06:56 Val-Class-6 | Precision: 0.9756 | Recall: 0.9970 | F1: 0.9862 | Support: 1001
05-09 13:06:56 Val-Class-7 | Precision: 0.9980 | Recall: 0.9960 | F1: 0.9970 | Support: 1005
05-09 13:06:56 Val-Class-8 | Precision: 0.9784 | Recall: 0.9920 | F1: 0.9852 | Support: 1005
05-09 13:06:56 Val-F1-macro: 0.9929
05-09 13:06:56 Val-F1-weighted: 0.9929
05-09 13:06:56 The best model epoch 4, val-acc 0.9929
05-09 13:06:56 -----Epoch 5/15-----
05-09 13:06:56 current lr: [0.01, 0.01, 0.01, 0.01]
 82%|##########################################################################################################6                       | 1733/2112 [08:58<01:57,  3.24it/s] 82%|##########################################################################################################7                       | 1734/2112 [08:58<01:57,  3.22it/s]100%|##################################################################################################################################| 2112/2112 [10:56<00:00,  3.22it/s]
05-09 13:17:54 BiMamba-Att residual gate: 0.011993
05-09 13:17:54 Max BiMamba-Att residual gate: 0.030000
05-09 13:17:54 MFSAN-CDAN-BiMamba active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 13:17:54 Train-Loss Source Classifier: 0.0156
05-09 13:17:54 Train-Loss MMD: 0.0938
05-09 13:17:54 Train-Loss L1: 0.0029
05-09 13:17:54 Train-Loss CDA MMD: 0.0954
05-09 13:17:54 Train-Loss Target Entropy: 0.1151
05-09 13:17:54 Train-Loss CDAN Domain: 0.6835
05-09 13:17:54 Train-Loss CDA Weighted: 0.0017
05-09 13:17:54 Train-Loss Entropy Weighted: 0.0005
05-09 13:17:54 Train-Loss CDAN Weighted: 0.0061
05-09 13:17:54 Train-Acc Source Data: 0.9993
05-09 13:17:54 Train-Acc Domain Data: 0.5607
100%|####################################################################################################################################| 141/141 [00:02<00:00, 70.33it/s]
05-09 13:17:57 Val-acc: 0.9940
05-09 13:17:57 Val-Class-0 | Precision: 0.9980 | Recall: 1.0000 | F1: 0.9990 | Support: 1003
05-09 13:17:57 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 13:17:57 Val-Class-2 | Precision: 0.9980 | Recall: 1.0000 | F1: 0.9990 | Support: 1000
05-09 13:17:57 Val-Class-3 | Precision: 0.9959 | Recall: 0.9720 | F1: 0.9838 | Support: 1000
05-09 13:17:57 Val-Class-4 | Precision: 0.9980 | Recall: 1.0000 | F1: 0.9990 | Support: 1001
05-09 13:17:57 Val-Class-5 | Precision: 0.9900 | Recall: 0.9890 | F1: 0.9895 | Support: 1000
05-09 13:17:57 Val-Class-6 | Precision: 0.9774 | Recall: 0.9950 | F1: 0.9861 | Support: 1001
05-09 13:17:57 Val-Class-7 | Precision: 0.9990 | Recall: 0.9980 | F1: 0.9985 | Support: 1005
05-09 13:17:57 Val-Class-8 | Precision: 0.9901 | Recall: 0.9920 | F1: 0.9911 | Support: 1005
05-09 13:17:57 Val-F1-macro: 0.9940
05-09 13:17:57 Val-F1-weighted: 0.9940
05-09 13:17:57 The best model epoch 5, val-acc 0.9940
05-09 13:17:57 -----Epoch 6/15-----
05-09 13:17:57 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [10:55<00:00,  3.22it/s]
05-09 13:28:54 BiMamba-Att residual gate: 0.012290
05-09 13:28:54 Max BiMamba-Att residual gate: 0.030000
05-09 13:28:54 MFSAN-CDAN-BiMamba active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 13:28:54 Train-Loss Source Classifier: 0.0155
05-09 13:28:54 Train-Loss MMD: 0.0901
05-09 13:28:54 Train-Loss L1: 0.0025
05-09 13:28:54 Train-Loss CDA MMD: 0.0920
05-09 13:28:54 Train-Loss Target Entropy: 0.1126
05-09 13:28:54 Train-Loss CDAN Domain: 0.6818
05-09 13:28:54 Train-Loss CDA Weighted: 0.0017
05-09 13:28:54 Train-Loss Entropy Weighted: 0.0005
05-09 13:28:54 Train-Loss CDAN Weighted: 0.0064
05-09 13:28:54 Train-Acc Source Data: 0.9995
05-09 13:28:54 Train-Acc Domain Data: 0.5650
100%|####################################################################################################################################| 141/141 [00:01<00:00, 71.73it/s]
05-09 13:28:57 Val-acc: 0.9876
05-09 13:28:57 Val-Class-0 | Precision: 0.9990 | Recall: 0.9990 | F1: 0.9990 | Support: 1003
05-09 13:28:57 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 13:28:57 Val-Class-2 | Precision: 1.0000 | Recall: 0.9960 | F1: 0.9980 | Support: 1000
05-09 13:28:57 Val-Class-3 | Precision: 0.9910 | Recall: 0.9960 | F1: 0.9935 | Support: 1000
05-09 13:28:57 Val-Class-4 | Precision: 0.9891 | Recall: 1.0000 | F1: 0.9945 | Support: 1001
05-09 13:28:57 Val-Class-5 | Precision: 0.9989 | Recall: 0.9190 | F1: 0.9573 | Support: 1000
05-09 13:28:57 Val-Class-6 | Precision: 1.0000 | Recall: 0.9820 | F1: 0.9909 | Support: 1001
05-09 13:28:57 Val-Class-7 | Precision: 0.9980 | Recall: 0.9960 | F1: 0.9970 | Support: 1005
05-09 13:28:57 Val-Class-8 | Precision: 0.9195 | Recall: 1.0000 | F1: 0.9581 | Support: 1005
05-09 13:28:57 Val-F1-macro: 0.9876
05-09 13:28:57 Val-F1-weighted: 0.9876
05-09 13:28:57 The best model epoch 5, val-acc 0.9940
05-09 13:28:57 -----Epoch 7/15-----
05-09 13:28:57 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [10:56<00:00,  3.22it/s]
05-09 13:39:54 BiMamba-Att residual gate: 0.012560
05-09 13:39:54 Max BiMamba-Att residual gate: 0.030000
05-09 13:39:54 MFSAN-CDAN-BiMamba active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 13:39:54 Train-Loss Source Classifier: 0.0157
05-09 13:39:54 Train-Loss MMD: 0.0900
05-09 13:39:54 Train-Loss L1: 0.0023
05-09 13:39:54 Train-Loss CDA MMD: 0.0926
05-09 13:39:54 Train-Loss Target Entropy: 0.1147
05-09 13:39:54 Train-Loss CDAN Domain: 0.6792
05-09 13:39:54 Train-Loss CDA Weighted: 0.0018
05-09 13:39:54 Train-Loss Entropy Weighted: 0.0006
05-09 13:39:54 Train-Loss CDAN Weighted: 0.0066
05-09 13:39:54 Train-Acc Source Data: 0.9996
05-09 13:39:54 Train-Acc Domain Data: 0.5736
100%|####################################################################################################################################| 141/141 [00:02<00:00, 69.68it/s]
05-09 13:39:57 Val-acc: 0.9753
05-09 13:39:57 Val-Class-0 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1003
05-09 13:39:57 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 13:39:57 Val-Class-2 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1000
05-09 13:39:57 Val-Class-3 | Precision: 0.9970 | Recall: 0.9810 | F1: 0.9889 | Support: 1000
05-09 13:39:57 Val-Class-4 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1001
05-09 13:39:57 Val-Class-5 | Precision: 0.9975 | Recall: 0.8020 | F1: 0.8891 | Support: 1000
05-09 13:39:57 Val-Class-6 | Precision: 0.9823 | Recall: 0.9990 | F1: 0.9906 | Support: 1001
05-09 13:39:57 Val-Class-7 | Precision: 0.9990 | Recall: 0.9970 | F1: 0.9980 | Support: 1005
05-09 13:39:57 Val-Class-8 | Precision: 0.8365 | Recall: 0.9980 | F1: 0.9102 | Support: 1005
05-09 13:39:57 Val-F1-macro: 0.9750
05-09 13:39:57 Val-F1-weighted: 0.9750
05-09 13:39:57 The best model epoch 5, val-acc 0.9940
05-09 13:39:57 -----Epoch 8/15-----
05-09 13:39:57 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [10:56<00:00,  3.22it/s]
05-09 13:50:55 BiMamba-Att residual gate: 0.012803
05-09 13:50:55 Max BiMamba-Att residual gate: 0.030000
05-09 13:50:55 MFSAN-CDAN-BiMamba active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 13:50:55 Train-Loss Source Classifier: 0.0167
05-09 13:50:55 Train-Loss MMD: 0.0887
05-09 13:50:55 Train-Loss L1: 0.0022
05-09 13:50:55 Train-Loss CDA MMD: 0.0919
05-09 13:50:55 Train-Loss Target Entropy: 0.1181
05-09 13:50:55 Train-Loss CDAN Domain: 0.6764
05-09 13:50:55 Train-Loss CDA Weighted: 0.0018
05-09 13:50:55 Train-Loss Entropy Weighted: 0.0006
05-09 13:50:55 Train-Loss CDAN Weighted: 0.0067
05-09 13:50:55 Train-Acc Source Data: 0.9994
05-09 13:50:55 Train-Acc Domain Data: 0.5825
100%|####################################################################################################################################| 141/141 [00:01<00:00, 70.73it/s]
05-09 13:50:57 Val-acc: 0.9828
05-09 13:50:57 Val-Class-0 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1003
05-09 13:50:57 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 13:50:57 Val-Class-2 | Precision: 0.9990 | Recall: 0.9530 | F1: 0.9754 | Support: 1000
05-09 13:50:57 Val-Class-3 | Precision: 0.9979 | Recall: 0.9570 | F1: 0.9770 | Support: 1000
05-09 13:50:57 Val-Class-4 | Precision: 1.0000 | Recall: 0.9690 | F1: 0.9843 | Support: 1001
05-09 13:50:57 Val-Class-5 | Precision: 0.9969 | Recall: 0.9750 | F1: 0.9858 | Support: 1000
05-09 13:50:57 Val-Class-6 | Precision: 0.9727 | Recall: 0.9970 | F1: 0.9847 | Support: 1001
05-09 13:50:57 Val-Class-7 | Precision: 0.9901 | Recall: 0.9960 | F1: 0.9931 | Support: 1005
05-09 13:50:57 Val-Class-8 | Precision: 0.9012 | Recall: 0.9980 | F1: 0.9471 | Support: 1005
05-09 13:50:57 Val-F1-macro: 0.9830
05-09 13:50:57 Val-F1-weighted: 0.9830
05-09 13:50:57 The best model epoch 5, val-acc 0.9940
05-09 13:50:57 -----Epoch 9/15-----
05-09 13:50:57 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [10:56<00:00,  3.22it/s]
05-09 14:01:55 BiMamba-Att residual gate: 0.013024
05-09 14:01:55 Max BiMamba-Att residual gate: 0.030000
05-09 14:01:55 MFSAN-CDAN-BiMamba active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 14:01:55 Train-Loss Source Classifier: 0.0160
05-09 14:01:55 Train-Loss MMD: 0.0887
05-09 14:01:55 Train-Loss L1: 0.0020
05-09 14:01:55 Train-Loss CDA MMD: 0.0925
05-09 14:01:55 Train-Loss Target Entropy: 0.1182
05-09 14:01:55 Train-Loss CDAN Domain: 0.6732
05-09 14:01:55 Train-Loss CDA Weighted: 0.0018
05-09 14:01:55 Train-Loss Entropy Weighted: 0.0006
05-09 14:01:55 Train-Loss CDAN Weighted: 0.0067
05-09 14:01:55 Train-Acc Source Data: 0.9996
05-09 14:01:55 Train-Acc Domain Data: 0.5892
100%|####################################################################################################################################| 141/141 [00:02<00:00, 69.33it/s]
05-09 14:01:58 Val-acc: 0.9967
05-09 14:01:58 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 14:01:58 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 14:01:58 Val-Class-2 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 14:01:58 Val-Class-3 | Precision: 0.9920 | Recall: 0.9970 | F1: 0.9945 | Support: 1000
05-09 14:01:58 Val-Class-4 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1001
05-09 14:01:58 Val-Class-5 | Precision: 0.9861 | Recall: 0.9940 | F1: 0.9900 | Support: 1000
05-09 14:01:58 Val-Class-6 | Precision: 0.9970 | Recall: 0.9950 | F1: 0.9960 | Support: 1001
05-09 14:01:58 Val-Class-7 | Precision: 0.9980 | Recall: 0.9990 | F1: 0.9985 | Support: 1005
05-09 14:01:58 Val-Class-8 | Precision: 0.9970 | Recall: 0.9851 | F1: 0.9910 | Support: 1005
05-09 14:01:58 Val-F1-macro: 0.9967
05-09 14:01:58 Val-F1-weighted: 0.9967
05-09 14:01:58 The best model epoch 9, val-acc 0.9967
05-09 14:01:58 -----Epoch 10/15-----
05-09 14:01:58 current lr: [0.01, 0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [10:56<00:00,  3.22it/s]
05-09 14:12:56 BiMamba-Att residual gate: 0.013224
05-09 14:12:56 Max BiMamba-Att residual gate: 0.030000
05-09 14:12:56 MFSAN-CDAN-BiMamba active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 14:12:56 Train-Loss Source Classifier: 0.0163
05-09 14:12:56 Train-Loss MMD: 0.0869
05-09 14:12:56 Train-Loss L1: 0.0018
05-09 14:12:56 Train-Loss CDA MMD: 0.0915
05-09 14:12:56 Train-Loss Target Entropy: 0.1197
05-09 14:12:56 Train-Loss CDAN Domain: 0.6693
05-09 14:12:56 Train-Loss CDA Weighted: 0.0018
05-09 14:12:56 Train-Loss Entropy Weighted: 0.0006
05-09 14:12:56 Train-Loss CDAN Weighted: 0.0067
05-09 14:12:56 Train-Acc Source Data: 0.9996
05-09 14:12:56 Train-Acc Domain Data: 0.5983
100%|####################################################################################################################################| 141/141 [00:01<00:00, 71.84it/s]
05-09 14:12:58 Val-acc: 0.9864
05-09 14:12:58 Val-Class-0 | Precision: 0.9990 | Recall: 0.9990 | F1: 0.9990 | Support: 1003
05-09 14:12:58 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 14:12:58 Val-Class-2 | Precision: 1.0000 | Recall: 0.9980 | F1: 0.9990 | Support: 1000
05-09 14:12:58 Val-Class-3 | Precision: 0.9948 | Recall: 0.9570 | F1: 0.9755 | Support: 1000
05-09 14:12:58 Val-Class-4 | Precision: 0.9980 | Recall: 1.0000 | F1: 0.9990 | Support: 1001
05-09 14:12:58 Val-Class-5 | Precision: 0.9632 | Recall: 0.9680 | F1: 0.9656 | Support: 1000
05-09 14:12:58 Val-Class-6 | Precision: 0.9596 | Recall: 0.9970 | F1: 0.9780 | Support: 1001
05-09 14:12:58 Val-Class-7 | Precision: 0.9990 | Recall: 0.9990 | F1: 0.9990 | Support: 1005
05-09 14:12:58 Val-Class-8 | Precision: 0.9650 | Recall: 0.9592 | F1: 0.9621 | Support: 1005
05-09 14:12:58 Val-F1-macro: 0.9864
05-09 14:12:58 Val-F1-weighted: 0.9863
05-09 14:12:58 The best model epoch 9, val-acc 0.9967
05-09 14:12:58 -----Epoch 11/15-----
05-09 14:12:58 current lr: [0.002, 0.002, 0.002, 0.002]
100%|##################################################################################################################################| 2112/2112 [10:56<00:00,  3.22it/s]
05-09 14:23:56 BiMamba-Att residual gate: 0.013261
05-09 14:23:56 Max BiMamba-Att residual gate: 0.030000
05-09 14:23:56 MFSAN-CDAN-BiMamba active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 14:23:56 Train-Loss Source Classifier: 0.0137
05-09 14:23:56 Train-Loss MMD: 0.0845
05-09 14:23:56 Train-Loss L1: 0.0014
05-09 14:23:56 Train-Loss CDA MMD: 0.0915
05-09 14:23:56 Train-Loss Target Entropy: 0.1142
05-09 14:23:56 Train-Loss CDAN Domain: 0.6623
05-09 14:23:56 Train-Loss CDA Weighted: 0.0018
05-09 14:23:56 Train-Loss Entropy Weighted: 0.0006
05-09 14:23:56 Train-Loss CDAN Weighted: 0.0066
05-09 14:23:56 Train-Acc Source Data: 1.0000
05-09 14:23:56 Train-Acc Domain Data: 0.6235
100%|####################################################################################################################################| 141/141 [00:02<00:00, 69.15it/s]
05-09 14:23:59 Val-acc: 0.9958
05-09 14:23:59 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 14:23:59 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 14:23:59 Val-Class-2 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 14:23:59 Val-Class-3 | Precision: 0.9970 | Recall: 0.9980 | F1: 0.9975 | Support: 1000
05-09 14:23:59 Val-Class-4 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1001
05-09 14:23:59 Val-Class-5 | Precision: 0.9764 | Recall: 0.9910 | F1: 0.9836 | Support: 1000
05-09 14:23:59 Val-Class-6 | Precision: 0.9980 | Recall: 0.9970 | F1: 0.9975 | Support: 1001
05-09 14:23:59 Val-Class-7 | Precision: 0.9990 | Recall: 0.9990 | F1: 0.9990 | Support: 1005
05-09 14:23:59 Val-Class-8 | Precision: 0.9919 | Recall: 0.9771 | F1: 0.9845 | Support: 1005
05-09 14:23:59 Val-F1-macro: 0.9958
05-09 14:23:59 Val-F1-weighted: 0.9958
05-09 14:23:59 The best model epoch 9, val-acc 0.9967
05-09 14:23:59 -----Epoch 12/15-----
05-09 14:23:59 current lr: [0.002, 0.002, 0.002, 0.002]
100%|##################################################################################################################################| 2112/2112 [10:56<00:00,  3.22it/s]
05-09 14:34:57 BiMamba-Att residual gate: 0.013297
05-09 14:34:57 Max BiMamba-Att residual gate: 0.030000
05-09 14:34:57 MFSAN-CDAN-BiMamba active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 14:34:57 Train-Loss Source Classifier: 0.0137
05-09 14:34:57 Train-Loss MMD: 0.0821
05-09 14:34:57 Train-Loss L1: 0.0013
05-09 14:34:57 Train-Loss CDA MMD: 0.0891
05-09 14:34:57 Train-Loss Target Entropy: 0.1154
05-09 14:34:57 Train-Loss CDAN Domain: 0.6612
05-09 14:34:57 Train-Loss CDA Weighted: 0.0018
05-09 14:34:57 Train-Loss Entropy Weighted: 0.0006
05-09 14:34:57 Train-Loss CDAN Weighted: 0.0066
05-09 14:34:57 Train-Acc Source Data: 1.0000
05-09 14:34:57 Train-Acc Domain Data: 0.6274
100%|####################################################################################################################################| 141/141 [00:01<00:00, 72.11it/s]
05-09 14:34:59 Val-acc: 0.9957
05-09 14:34:59 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 14:34:59 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 14:34:59 Val-Class-2 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 14:34:59 Val-Class-3 | Precision: 0.9970 | Recall: 1.0000 | F1: 0.9985 | Support: 1000
05-09 14:34:59 Val-Class-4 | Precision: 1.0000 | Recall: 0.9990 | F1: 0.9995 | Support: 1001
05-09 14:34:59 Val-Class-5 | Precision: 0.9754 | Recall: 0.9900 | F1: 0.9826 | Support: 1000
05-09 14:34:59 Val-Class-6 | Precision: 1.0000 | Recall: 0.9970 | F1: 0.9985 | Support: 1001
05-09 14:34:59 Val-Class-7 | Precision: 0.9990 | Recall: 0.9970 | F1: 0.9980 | Support: 1005
05-09 14:34:59 Val-Class-8 | Precision: 0.9899 | Recall: 0.9781 | F1: 0.9840 | Support: 1005
05-09 14:34:59 Val-F1-macro: 0.9957
05-09 14:34:59 Val-F1-weighted: 0.9957
05-09 14:34:59 The best model epoch 9, val-acc 0.9967
05-09 14:34:59 -----Epoch 13/15-----
05-09 14:34:59 current lr: [0.002, 0.002, 0.002, 0.002]
100%|##################################################################################################################################| 2112/2112 [10:56<00:00,  3.22it/s]
05-09 14:45:57 BiMamba-Att residual gate: 0.013333
05-09 14:45:57 Max BiMamba-Att residual gate: 0.030000
05-09 14:45:57 MFSAN-CDAN-BiMamba active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 14:45:57 Train-Loss Source Classifier: 0.0140
05-09 14:45:57 Train-Loss MMD: 0.0830
05-09 14:45:57 Train-Loss L1: 0.0012
05-09 14:45:57 Train-Loss CDA MMD: 0.0910
05-09 14:45:57 Train-Loss Target Entropy: 0.1185
05-09 14:45:57 Train-Loss CDAN Domain: 0.6598
05-09 14:45:57 Train-Loss CDA Weighted: 0.0018
05-09 14:45:57 Train-Loss Entropy Weighted: 0.0006
05-09 14:45:57 Train-Loss CDAN Weighted: 0.0066
05-09 14:45:57 Train-Acc Source Data: 1.0000
05-09 14:45:57 Train-Acc Domain Data: 0.6308
100%|####################################################################################################################################| 141/141 [00:02<00:00, 67.84it/s]
05-09 14:46:00 Val-acc: 0.9965
05-09 14:46:00 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 14:46:00 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 14:46:00 Val-Class-2 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 14:46:00 Val-Class-3 | Precision: 0.9960 | Recall: 0.9990 | F1: 0.9975 | Support: 1000
05-09 14:46:00 Val-Class-4 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1001
05-09 14:46:00 Val-Class-5 | Precision: 0.9783 | Recall: 0.9940 | F1: 0.9861 | Support: 1000
05-09 14:46:00 Val-Class-6 | Precision: 1.0000 | Recall: 0.9960 | F1: 0.9980 | Support: 1001
05-09 14:46:00 Val-Class-7 | Precision: 1.0000 | Recall: 0.9980 | F1: 0.9990 | Support: 1005
05-09 14:46:00 Val-Class-8 | Precision: 0.9940 | Recall: 0.9811 | F1: 0.9875 | Support: 1005
05-09 14:46:00 Val-F1-macro: 0.9965
05-09 14:46:00 Val-F1-weighted: 0.9965
05-09 14:46:00 The best model epoch 9, val-acc 0.9967
05-09 14:46:00 -----Epoch 14/15-----
05-09 14:46:00 current lr: [0.002, 0.002, 0.002, 0.002]
100%|##################################################################################################################################| 2112/2112 [10:56<00:00,  3.22it/s]
05-09 14:56:58 BiMamba-Att residual gate: 0.013368
05-09 14:56:58 Max BiMamba-Att residual gate: 0.030000
05-09 14:56:58 MFSAN-CDAN-BiMamba active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 14:56:58 Train-Loss Source Classifier: 0.0141
05-09 14:56:58 Train-Loss MMD: 0.0816
05-09 14:56:58 Train-Loss L1: 0.0012
05-09 14:56:58 Train-Loss CDA MMD: 0.0901
05-09 14:56:58 Train-Loss Target Entropy: 0.1195
05-09 14:56:58 Train-Loss CDAN Domain: 0.6581
05-09 14:56:58 Train-Loss CDA Weighted: 0.0018
05-09 14:56:58 Train-Loss Entropy Weighted: 0.0006
05-09 14:56:58 Train-Loss CDAN Weighted: 0.0066
05-09 14:56:58 Train-Acc Source Data: 1.0000
05-09 14:56:58 Train-Acc Domain Data: 0.6348
100%|####################################################################################################################################| 141/141 [00:02<00:00, 69.98it/s]
05-09 14:57:00 Val-acc: 0.9962
05-09 14:57:00 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 14:57:00 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 14:57:00 Val-Class-2 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 14:57:00 Val-Class-3 | Precision: 0.9970 | Recall: 1.0000 | F1: 0.9985 | Support: 1000
05-09 14:57:00 Val-Class-4 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1001
05-09 14:57:00 Val-Class-5 | Precision: 0.9821 | Recall: 0.9870 | F1: 0.9845 | Support: 1000
05-09 14:57:00 Val-Class-6 | Precision: 1.0000 | Recall: 0.9970 | F1: 0.9985 | Support: 1001
05-09 14:57:00 Val-Class-7 | Precision: 0.9990 | Recall: 0.9990 | F1: 0.9990 | Support: 1005
05-09 14:57:00 Val-Class-8 | Precision: 0.9880 | Recall: 0.9831 | F1: 0.9855 | Support: 1005
05-09 14:57:00 Val-F1-macro: 0.9962
05-09 14:57:00 Val-F1-weighted: 0.9962
05-09 14:57:00 The best model epoch 9, val-acc 0.9967
05-09 14:57:00 -----Epoch 15/15-----
05-09 14:57:00 current lr: [0.002, 0.002, 0.002, 0.002]
100%|##################################################################################################################################| 2112/2112 [10:56<00:00,  3.22it/s]
05-09 15:07:59 BiMamba-Att residual gate: 0.013402
05-09 15:07:59 Max BiMamba-Att residual gate: 0.030000
05-09 15:07:59 MFSAN-CDAN-BiMamba active: lambda_adv=0.010000, lambda_grl=0.500000, lambda_cda=0.020000, lambda_ent=0.005000, adv_detach_prob=True, entropy_weight=True
05-09 15:07:59 Train-Loss Source Classifier: 0.0145
05-09 15:07:59 Train-Loss MMD: 0.0837
05-09 15:07:59 Train-Loss L1: 0.0012
05-09 15:07:59 Train-Loss CDA MMD: 0.0927
05-09 15:07:59 Train-Loss Target Entropy: 0.1225
05-09 15:07:59 Train-Loss CDAN Domain: 0.6570
05-09 15:07:59 Train-Loss CDA Weighted: 0.0019
05-09 15:07:59 Train-Loss Entropy Weighted: 0.0006
05-09 15:07:59 Train-Loss CDAN Weighted: 0.0066
05-09 15:07:59 Train-Acc Source Data: 1.0000
05-09 15:07:59 Train-Acc Domain Data: 0.6368
100%|####################################################################################################################################| 141/141 [00:01<00:00, 71.57it/s]
05-09 15:08:01 Val-acc: 0.9967
05-09 15:08:01 Val-Class-0 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1003
05-09 15:08:01 Val-Class-1 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 15:08:01 Val-Class-2 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1000
05-09 15:08:01 Val-Class-3 | Precision: 0.9960 | Recall: 1.0000 | F1: 0.9980 | Support: 1000
05-09 15:08:01 Val-Class-4 | Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1001
05-09 15:08:01 Val-Class-5 | Precision: 0.9880 | Recall: 0.9860 | F1: 0.9870 | Support: 1000
05-09 15:08:01 Val-Class-6 | Precision: 1.0000 | Recall: 0.9960 | F1: 0.9980 | Support: 1001
05-09 15:08:01 Val-Class-7 | Precision: 0.9990 | Recall: 0.9990 | F1: 0.9990 | Support: 1005
05-09 15:08:01 Val-Class-8 | Precision: 0.9871 | Recall: 0.9891 | F1: 0.9881 | Support: 1005
05-09 15:08:01 Val-F1-macro: 0.9967
05-09 15:08:01 Val-F1-weighted: 0.9967
05-09 15:08:01 The best model epoch 15, val-acc 0.9967
05-09 15:08:03 Model saved to ./ckpt/MFSAN_CDAN_BIMAMBA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0509-122300.pth
(MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels_bimamba# 