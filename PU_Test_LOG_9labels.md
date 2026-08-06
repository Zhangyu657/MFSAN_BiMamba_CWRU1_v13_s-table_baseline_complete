这是原始版本未改进版本

(MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels# python train.py \
>   --model_name MFSAN_CDA \
>   --source PU_0,PU_1,PU_2 \
>   --target PU_3 \
>   --train_mode multi_source \
>   --data_dir /workspace/PU_TL \
>   --signal_size 1024 \
>   --backbone CNN \
>   --cuda_device 0 \
>   --max_epoch 2 \
>   --lambda_cda 0 \
>   --lambda_ent 0 \
>   --include_faults K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18
05-09 06:30:25 model_name: MFSAN_CDA
05-09 06:30:25 source: PU_0,PU_1,PU_2
05-09 06:30:25 target: PU_3
05-09 06:30:25 data_dir: /workspace/PU_TL
05-09 06:30:25 train_mode: multi_source
05-09 06:30:25 cuda_device: 0
05-09 06:30:25 max_epoch: 2
05-09 06:30:25 batch_size: 64
05-09 06:30:25 signal_size: 1024
05-09 06:30:25 random_state: 10
05-09 06:30:25 include_faults: K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18
05-09 06:30:25 exclude_faults: 
05-09 06:30:25 opt: sgd
05-09 06:30:25 momentum: 0.9
05-09 06:30:25 betas: (0.9, 0.999)
05-09 06:30:25 weight_decay: 0.0005
05-09 06:30:25 lr: 0.01
05-09 06:30:25 lr_scheduler: stepLR
05-09 06:30:25 gamma: 0.2
05-09 06:30:25 steps: 10
05-09 06:30:25 backbone: CNN
05-09 06:30:25 num_workers: 4
05-09 06:30:25 normlize_type: -1-1
05-09 06:30:25 tradeoff: ['exp', 'exp', 'exp']
05-09 06:30:25 zeta: 10.0
05-09 06:30:25 dropout: 0.0
05-09 06:30:25 lambda_cda: 0.0
05-09 06:30:25 lambda_ent: 0.0
05-09 06:30:25 cda_detach_prob: True
05-09 06:30:25 save: True
05-09 06:30:25 save_dir: ./ckpt
05-09 06:30:25 load_path: 
05-09 06:30:25 save_path: ./ckpt/MFSAN_CDA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0509-063025
05-09 06:30:25 Source PU_0 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 06:30:25 Source PU_1 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 06:30:25 Source PU_2 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 06:30:25 Target PU_3 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 06:30:25 The scenario is: closed-set domain adaptation
05-09 06:30:26 using 1 / 1 gpus
05-09 06:30:27 Using model: MFSAN_CDA
05-09 06:30:27 Using backbone: CNN
05-09 06:30:27 Backbone output dim: 640
05-09 06:30:30 Source set PU_0 number of samples: 45103.
Label 0 has samples: 5001
Label 1 has samples: 5000
Label 2 has samples: 5005
Label 3 has samples: 5005
Label 4 has samples: 5012
Label 5 has samples: 5022
Label 6 has samples: 5019
Label 7 has samples: 5033
Label 8 has samples: 5006
05-09 06:30:30 Source set PU_1 number of samples: 45023.
Label 0 has samples: 5002
Label 1 has samples: 5006
Label 2 has samples: 5002
Label 3 has samples: 4999
Label 4 has samples: 5005
Label 5 has samples: 5004
Label 6 has samples: 5001
Label 7 has samples: 5004
Label 8 has samples: 5000
05-09 06:30:30 Source set PU_2 number of samples: 45143.
Label 0 has samples: 5001
Label 1 has samples: 5005
Label 2 has samples: 5013
Label 3 has samples: 5018
Label 4 has samples: 5008
Label 5 has samples: 5059
Label 6 has samples: 5000
Label 7 has samples: 5040
Label 8 has samples: 4999
05-09 06:30:31 Training set number of samples: 36053.
Label 0 has samples: 4011
Label 1 has samples: 4000
Label 2 has samples: 4000
Label 3 has samples: 4000
Label 4 has samples: 4004
Label 5 has samples: 4000
Label 6 has samples: 4004
Label 7 has samples: 4016
Label 8 has samples: 4018
05-09 06:30:31 Validation set number of samples: 9015.
Label 0 has samples: 1003
Label 1 has samples: 1000
Label 2 has samples: 1000
Label 3 has samples: 1000
Label 4 has samples: 1001
Label 5 has samples: 1000
Label 6 has samples: 1001
Label 7 has samples: 1005
Label 8 has samples: 1005
05-09 06:30:32 CDA lambda_cda: 0.000000
05-09 06:30:32 CDA lambda_ent: 0.000000
05-09 06:30:32 CDA detach probabilities for joint feature: True
05-09 06:30:32 CDA joint feature dim: 9 x 40 = 360
05-09 06:30:32 -----Epoch 1/2-----
05-09 06:30:32 current lr: [0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [08:37<00:00,  4.08it/s]
05-09 06:39:10 MFSAN-CDA active: lambda_cda=0.000000, lambda_ent=0.000000, detach_prob=True
05-09 06:39:10 Train-Loss Source Classifier: 0.2705
05-09 06:39:10 Train-Loss MMD: 0.3889
05-09 06:39:10 Train-Loss L1: 0.0733
05-09 06:39:10 Train-Loss CDA MMD: 0.3189
05-09 06:39:10 Train-Loss Target Entropy: 0.7040
05-09 06:39:10 Train-Loss CDA Weighted: 0.0000
05-09 06:39:10 Train-Loss Entropy Weighted: 0.0000
05-09 06:39:10 Train-Acc Source Data: 0.8972
100%|###################################################################################################################################| 141/141 [00:01<00:00, 137.98it/s]
05-09 06:39:11 Val-acc: 0.6808
05-09 06:39:11 The best model epoch 1, val-acc 0.6808
05-09 06:39:11 -----Epoch 2/2-----
05-09 06:39:11 current lr: [0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [08:42<00:00,  4.05it/s]
05-09 06:47:54 MFSAN-CDA active: lambda_cda=0.000000, lambda_ent=0.000000, detach_prob=True
05-09 06:47:54 Train-Loss Source Classifier: 0.0463
05-09 06:47:54 Train-Loss MMD: 0.1002
05-09 06:47:54 Train-Loss L1: 0.0132
05-09 06:47:54 Train-Loss CDA MMD: 0.1027
05-09 06:47:54 Train-Loss Target Entropy: 0.2107
05-09 06:47:54 Train-Loss CDA Weighted: 0.0000
05-09 06:47:54 Train-Loss Entropy Weighted: 0.0000
05-09 06:47:54 Train-Acc Source Data: 0.9889
100%|###################################################################################################################################| 141/141 [00:01<00:00, 135.70it/s]
05-09 06:47:56 Val-acc: 0.9827
05-09 06:47:56 The best model epoch 2, val-acc 0.9827
05-09 06:47:57 Model saved to ./ckpt/MFSAN_CDA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0509-063025.pth
(MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels# 

这是加入CDA后的版本
(MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels# python train.py \
>   --model_name MFSAN_CDA \
>   --source PU_0,PU_1,PU_2 \
>   --target PU_3 \
>   --train_mode multi_source \
>   --data_dir /workspace/PU_TL \
>   --signal_size 1024 \
>   --backbone CNN \
>   --cuda_device 0 \
>   --max_epoch 2 \
>   --lambda_cda 0.02 \
>   --lambda_ent 0.005 \
>   --include_faults K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18
05-09 06:49:51 model_name: MFSAN_CDA
05-09 06:49:51 source: PU_0,PU_1,PU_2
05-09 06:49:51 target: PU_3
05-09 06:49:51 data_dir: /workspace/PU_TL
05-09 06:49:51 train_mode: multi_source
05-09 06:49:51 cuda_device: 0
05-09 06:49:51 max_epoch: 2
05-09 06:49:51 batch_size: 64
05-09 06:49:51 signal_size: 1024
05-09 06:49:51 random_state: 10
05-09 06:49:51 include_faults: K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18
05-09 06:49:51 exclude_faults: 
05-09 06:49:51 opt: sgd
05-09 06:49:51 momentum: 0.9
05-09 06:49:51 betas: (0.9, 0.999)
05-09 06:49:51 weight_decay: 0.0005
05-09 06:49:51 lr: 0.01
05-09 06:49:51 lr_scheduler: stepLR
05-09 06:49:51 gamma: 0.2
05-09 06:49:51 steps: 10
05-09 06:49:51 backbone: CNN
05-09 06:49:51 num_workers: 4
05-09 06:49:51 normlize_type: -1-1
05-09 06:49:51 tradeoff: ['exp', 'exp', 'exp']
05-09 06:49:51 zeta: 10.0
05-09 06:49:51 dropout: 0.0
05-09 06:49:51 lambda_cda: 0.02
05-09 06:49:51 lambda_ent: 0.005
05-09 06:49:51 cda_detach_prob: True
05-09 06:49:51 save: True
05-09 06:49:51 save_dir: ./ckpt
05-09 06:49:51 load_path: 
05-09 06:49:51 save_path: ./ckpt/MFSAN_CDA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0509-064951
05-09 06:49:51 Source PU_0 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 06:49:51 Source PU_1 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 06:49:51 Source PU_2 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 06:49:51 Target PU_3 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 06:49:51 The scenario is: closed-set domain adaptation
05-09 06:49:52 using 1 / 1 gpus
05-09 06:49:54 Using model: MFSAN_CDA
05-09 06:49:54 Using backbone: CNN
05-09 06:49:54 Backbone output dim: 640
05-09 06:49:57 Source set PU_0 number of samples: 45103.
Label 0 has samples: 5001
Label 1 has samples: 5000
Label 2 has samples: 5005
Label 3 has samples: 5005
Label 4 has samples: 5012
Label 5 has samples: 5022
Label 6 has samples: 5019
Label 7 has samples: 5033
Label 8 has samples: 5006
05-09 06:49:57 Source set PU_1 number of samples: 45023.
Label 0 has samples: 5002
Label 1 has samples: 5006
Label 2 has samples: 5002
Label 3 has samples: 4999
Label 4 has samples: 5005
Label 5 has samples: 5004
Label 6 has samples: 5001
Label 7 has samples: 5004
Label 8 has samples: 5000
05-09 06:49:57 Source set PU_2 number of samples: 45143.
Label 0 has samples: 5001
Label 1 has samples: 5005
Label 2 has samples: 5013
Label 3 has samples: 5018
Label 4 has samples: 5008
Label 5 has samples: 5059
Label 6 has samples: 5000
Label 7 has samples: 5040
Label 8 has samples: 4999
05-09 06:49:57 Training set number of samples: 36053.
Label 0 has samples: 4011
Label 1 has samples: 4000
Label 2 has samples: 4000
Label 3 has samples: 4000
Label 4 has samples: 4004
Label 5 has samples: 4000
Label 6 has samples: 4004
Label 7 has samples: 4016
Label 8 has samples: 4018
05-09 06:49:57 Validation set number of samples: 9015.
Label 0 has samples: 1003
Label 1 has samples: 1000
Label 2 has samples: 1000
Label 3 has samples: 1000
Label 4 has samples: 1001
Label 5 has samples: 1000
Label 6 has samples: 1001
Label 7 has samples: 1005
Label 8 has samples: 1005
05-09 06:49:58 CDA lambda_cda: 0.020000
05-09 06:49:58 CDA lambda_ent: 0.005000
05-09 06:49:58 CDA detach probabilities for joint feature: True
05-09 06:49:58 CDA joint feature dim: 9 x 40 = 360
05-09 06:49:58 -----Epoch 1/2-----
05-09 06:49:58 current lr: [0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [08:38<00:00,  4.08it/s]
05-09 06:58:36 MFSAN-CDA active: lambda_cda=0.020000, lambda_ent=0.005000, detach_prob=True
05-09 06:58:36 Train-Loss Source Classifier: 0.2705
05-09 06:58:36 Train-Loss MMD: 0.3889
05-09 06:58:36 Train-Loss L1: 0.0733
05-09 06:58:36 Train-Loss CDA MMD: 0.3189
05-09 06:58:36 Train-Loss Target Entropy: 0.7040
05-09 06:58:36 Train-Loss CDA Weighted: 0.0000
05-09 06:58:36 Train-Loss Entropy Weighted: 0.0000
05-09 06:58:36 Train-Acc Source Data: 0.8972
100%|###################################################################################################################################| 141/141 [00:01<00:00, 140.89it/s]
05-09 06:58:38 Val-acc: 0.6808
05-09 06:58:38 The best model epoch 1, val-acc 0.6808
05-09 06:58:38 -----Epoch 2/2-----
05-09 06:58:38 current lr: [0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [08:41<00:00,  4.05it/s]
05-09 07:07:20 MFSAN-CDA active: lambda_cda=0.020000, lambda_ent=0.005000, detach_prob=True
05-09 07:07:20 Train-Loss Source Classifier: 0.0461
05-09 07:07:20 Train-Loss MMD: 0.1003
05-09 07:07:20 Train-Loss L1: 0.0130
05-09 07:07:20 Train-Loss CDA MMD: 0.1025
05-09 07:07:20 Train-Loss Target Entropy: 0.2060
05-09 07:07:20 Train-Loss CDA Weighted: 0.0021
05-09 07:07:20 Train-Loss Entropy Weighted: 0.0010
05-09 07:07:20 Train-Acc Source Data: 0.9888
100%|###################################################################################################################################| 141/141 [00:01<00:00, 139.12it/s]
05-09 07:07:21 Val-acc: 0.9828
05-09 07:07:21 The best model epoch 2, val-acc 0.9828
05-09 07:07:22 Model saved to ./ckpt/MFSAN_CDA/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0509-064951.pth
(MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels# # 目标域 PU_0




(MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels# python train.py \
CDA \
  --source PU_1,PU_2,PU_3 \
  --target PU_0 >   --model_name MFSAN_CDA \
>   --source PU_1,PU_2,PU_3 \
>   --target PU_0 \
>   --train_mode multi_source \
>   --data_dir /workspace/PU_TL \
>   --signal_size 1024 \
>   --backbone CNN \
>   --cuda_device 0 \
>   --max_epoch 2 \
>   --lambda_cda 0.02 \
>   --lambda_ent 0.005 \
>   --include_faults K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18
05-09 07:11:29 model_name: MFSAN_CDA
05-09 07:11:29 source: PU_1,PU_2,PU_3
05-09 07:11:29 target: PU_0
05-09 07:11:29 data_dir: /workspace/PU_TL
05-09 07:11:29 train_mode: multi_source
05-09 07:11:29 cuda_device: 0
05-09 07:11:29 max_epoch: 2
05-09 07:11:29 batch_size: 64
05-09 07:11:29 signal_size: 1024
05-09 07:11:29 random_state: 10
05-09 07:11:29 include_faults: K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18
05-09 07:11:29 exclude_faults: 
05-09 07:11:29 opt: sgd
05-09 07:11:29 momentum: 0.9
05-09 07:11:29 betas: (0.9, 0.999)
05-09 07:11:29 weight_decay: 0.0005
05-09 07:11:29 lr: 0.01
05-09 07:11:29 lr_scheduler: stepLR
05-09 07:11:29 gamma: 0.2
05-09 07:11:29 steps: 10
05-09 07:11:29 backbone: CNN
05-09 07:11:29 num_workers: 4
05-09 07:11:29 normlize_type: -1-1
05-09 07:11:29 tradeoff: ['exp', 'exp', 'exp']
05-09 07:11:29 zeta: 10.0
05-09 07:11:29 dropout: 0.0
05-09 07:11:29 lambda_cda: 0.02
05-09 07:11:29 lambda_ent: 0.005
05-09 07:11:29 cda_detach_prob: True
05-09 07:11:29 save: True
05-09 07:11:29 save_dir: ./ckpt
05-09 07:11:29 load_path: 
05-09 07:11:29 save_path: ./ckpt/MFSAN_CDA/multi_source/[PU_1_PU_2_PU_3]To[PU_0]_0509-071129
05-09 07:11:29 Source PU_1 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 07:11:29 Source PU_2 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 07:11:29 Source PU_3 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 07:11:29 Target PU_0 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 07:11:29 The scenario is: closed-set domain adaptation
05-09 07:11:30 using 1 / 1 gpus
05-09 07:11:32 Using model: MFSAN_CDA
05-09 07:11:32 Using backbone: CNN
05-09 07:11:32 Backbone output dim: 640
05-09 07:11:35 Source set PU_1 number of samples: 45023.
Label 0 has samples: 5002
Label 1 has samples: 5006
Label 2 has samples: 5002
Label 3 has samples: 4999
Label 4 has samples: 5005
Label 5 has samples: 5004
Label 6 has samples: 5001
Label 7 has samples: 5004
Label 8 has samples: 5000
05-09 07:11:35 Source set PU_2 number of samples: 45143.
Label 0 has samples: 5001
Label 1 has samples: 5005
Label 2 has samples: 5013
Label 3 has samples: 5018
Label 4 has samples: 5008
Label 5 has samples: 5059
Label 6 has samples: 5000
Label 7 has samples: 5040
Label 8 has samples: 4999
05-09 07:11:35 Source set PU_3 number of samples: 45068.
Label 0 has samples: 5014
Label 1 has samples: 5000
Label 2 has samples: 5000
Label 3 has samples: 5000
Label 4 has samples: 5005
Label 5 has samples: 5000
Label 6 has samples: 5005
Label 7 has samples: 5021
Label 8 has samples: 5023
05-09 07:11:36 Training set number of samples: 36079.
Label 0 has samples: 4000
Label 1 has samples: 4000
Label 2 has samples: 4004
Label 3 has samples: 4004
Label 4 has samples: 4009
Label 5 has samples: 4017
Label 6 has samples: 4015
Label 7 has samples: 4026
Label 8 has samples: 4004
05-09 07:11:36 Validation set number of samples: 9024.
Label 0 has samples: 1001
Label 1 has samples: 1000
Label 2 has samples: 1001
Label 3 has samples: 1001
Label 4 has samples: 1003
Label 5 has samples: 1005
Label 6 has samples: 1004
Label 7 has samples: 1007
Label 8 has samples: 1002
05-09 07:11:36 CDA lambda_cda: 0.020000
05-09 07:11:36 CDA lambda_ent: 0.005000
05-09 07:11:36 CDA detach probabilities for joint feature: True
05-09 07:11:36 CDA joint feature dim: 9 x 40 = 360
05-09 07:11:36 -----Epoch 1/2-----
05-09 07:11:36 current lr: [0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [08:37<00:00,  4.08it/s]
05-09 07:20:13 MFSAN-CDA active: lambda_cda=0.020000, lambda_ent=0.005000, detach_prob=True
05-09 07:20:14 Train-Loss Source Classifier: 0.1928
05-09 07:20:14 Train-Loss MMD: 0.7158
05-09 07:20:14 Train-Loss L1: 0.0632
05-09 07:20:14 Train-Loss CDA MMD: 0.4917
05-09 07:20:14 Train-Loss Target Entropy: 0.6901
05-09 07:20:14 Train-Loss CDA Weighted: 0.0000
05-09 07:20:14 Train-Loss Entropy Weighted: 0.0000
05-09 07:20:14 Train-Acc Source Data: 0.9277
100%|###################################################################################################################################| 141/141 [00:00<00:00, 143.46it/s]
05-09 07:20:15 Val-acc: 0.4227
05-09 07:20:15 The best model epoch 1, val-acc 0.4227
05-09 07:20:15 -----Epoch 2/2-----
05-09 07:20:15 current lr: [0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2112/2112 [08:40<00:00,  4.06it/s]
05-09 07:28:56 MFSAN-CDA active: lambda_cda=0.020000, lambda_ent=0.005000, detach_prob=True
05-09 07:28:56 Train-Loss Source Classifier: 0.0444
05-09 07:28:56 Train-Loss MMD: 0.1095
05-09 07:28:56 Train-Loss L1: 0.0200
05-09 07:28:56 Train-Loss CDA MMD: 0.1289
05-09 07:28:56 Train-Loss Target Entropy: 0.3555
05-09 07:28:56 Train-Loss CDA Weighted: 0.0026
05-09 07:28:56 Train-Loss Entropy Weighted: 0.0018
05-09 07:28:56 Train-Acc Source Data: 0.9918
100%|###################################################################################################################################| 141/141 [00:01<00:00, 133.49it/s]
05-09 07:28:57 Val-acc: 0.9085
05-09 07:28:57 The best model epoch 2, val-acc 0.9085
05-09 07:28:58 Model saved to ./ckpt/MFSAN_CDA/multi_source/[PU_1_PU_2_PU_3]To[PU_0]_0509-071129.pth
(MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels# 


python train.py \
  --model_name MFSAN_CDA \
  --source PU_0,PU_1,PU_3 \
  --target PU_2 \
  --train_mode multi_source \
  --data_dir /workspace/PU_TL \
  --signal_size 1024 \
  --backbone CNN \
  --cuda_device 0 \
  --max_epoch 2 \
  --lambda_cda 0.02 \
  --lambda_ent 0.005 \
  --include_faults K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18


  (MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels# python train.py \
>   --model_name MFSAN_CDA \
>   --source PU_0,PU_1,PU_3 \
>   --target PU_2 \
>   --train_mode multi_source \
>   --data_dir /workspace/PU_TL \
>   --signal_size 1024 \
>   --backbone CNN \
>   --cuda_device 0 \
>   --max_epoch 2 \
>   --lambda_cda 0.02 \
>   --lambda_ent 0.005 \
>   --include_faults K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18
05-09 07:32:04 model_name: MFSAN_CDA
05-09 07:32:04 source: PU_0,PU_1,PU_3
05-09 07:32:04 target: PU_2
05-09 07:32:04 data_dir: /workspace/PU_TL
05-09 07:32:04 train_mode: multi_source
05-09 07:32:04 cuda_device: 0
05-09 07:32:04 max_epoch: 2
05-09 07:32:04 batch_size: 64
05-09 07:32:04 signal_size: 1024
05-09 07:32:04 random_state: 10
05-09 07:32:04 include_faults: K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18
05-09 07:32:04 exclude_faults: 
05-09 07:32:04 opt: sgd
05-09 07:32:04 momentum: 0.9
05-09 07:32:04 betas: (0.9, 0.999)
05-09 07:32:04 weight_decay: 0.0005
05-09 07:32:04 lr: 0.01
05-09 07:32:04 lr_scheduler: stepLR
05-09 07:32:04 gamma: 0.2
05-09 07:32:04 steps: 10
05-09 07:32:04 backbone: CNN
05-09 07:32:04 num_workers: 4
05-09 07:32:04 normlize_type: -1-1
05-09 07:32:04 tradeoff: ['exp', 'exp', 'exp']
05-09 07:32:04 zeta: 10.0
05-09 07:32:04 dropout: 0.0
05-09 07:32:04 lambda_cda: 0.02
05-09 07:32:04 lambda_ent: 0.005
05-09 07:32:04 cda_detach_prob: True
05-09 07:32:04 save: True
05-09 07:32:04 save_dir: ./ckpt
05-09 07:32:04 load_path: 
05-09 07:32:04 save_path: ./ckpt/MFSAN_CDA/multi_source/[PU_0_PU_1_PU_3]To[PU_2]_0509-073204
05-09 07:32:04 Source PU_0 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 07:32:04 Source PU_1 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 07:32:04 Source PU_3 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 07:32:04 Target PU_2 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 07:32:04 The scenario is: closed-set domain adaptation
05-09 07:32:04 using 1 / 1 gpus
05-09 07:32:06 Using model: MFSAN_CDA
05-09 07:32:06 Using backbone: CNN
05-09 07:32:06 Backbone output dim: 640
05-09 07:32:09 Source set PU_0 number of samples: 45103.
Label 0 has samples: 5001
Label 1 has samples: 5000
Label 2 has samples: 5005
Label 3 has samples: 5005
Label 4 has samples: 5012
Label 5 has samples: 5022
Label 6 has samples: 5019
Label 7 has samples: 5033
Label 8 has samples: 5006
05-09 07:32:09 Source set PU_1 number of samples: 45023.
Label 0 has samples: 5002
Label 1 has samples: 5006
Label 2 has samples: 5002
Label 3 has samples: 4999
Label 4 has samples: 5005
Label 5 has samples: 5004
Label 6 has samples: 5001
Label 7 has samples: 5004
Label 8 has samples: 5000
05-09 07:32:09 Source set PU_3 number of samples: 45068.
Label 0 has samples: 5014
Label 1 has samples: 5000
Label 2 has samples: 5000
Label 3 has samples: 5000
Label 4 has samples: 5005
Label 5 has samples: 5000
Label 6 has samples: 5005
Label 7 has samples: 5021
Label 8 has samples: 5023
05-09 07:32:10 Training set number of samples: 36112.
Label 0 has samples: 4000
Label 1 has samples: 4004
Label 2 has samples: 4010
Label 3 has samples: 4014
Label 4 has samples: 4006
Label 5 has samples: 4047
Label 6 has samples: 4000
Label 7 has samples: 4032
Label 8 has samples: 3999
05-09 07:32:10 Validation set number of samples: 9031.
Label 0 has samples: 1001
Label 1 has samples: 1001
Label 2 has samples: 1003
Label 3 has samples: 1004
Label 4 has samples: 1002
Label 5 has samples: 1012
Label 6 has samples: 1000
Label 7 has samples: 1008
Label 8 has samples: 1000
05-09 07:32:11 CDA lambda_cda: 0.020000
05-09 07:32:11 CDA lambda_ent: 0.005000
05-09 07:32:11 CDA detach probabilities for joint feature: True
05-09 07:32:11 CDA joint feature dim: 9 x 40 = 360
05-09 07:32:11 -----Epoch 1/2-----
05-09 07:32:11 current lr: [0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2111/2111 [08:37<00:00,  4.08it/s]
05-09 07:40:48 MFSAN-CDA active: lambda_cda=0.020000, lambda_ent=0.005000, detach_prob=True
05-09 07:40:49 Train-Loss Source Classifier: 0.2787
05-09 07:40:49 Train-Loss MMD: 0.2948
05-09 07:40:49 Train-Loss L1: 0.0871
05-09 07:40:49 Train-Loss CDA MMD: 0.2377
05-09 07:40:49 Train-Loss Target Entropy: 0.7027
05-09 07:40:49 Train-Loss CDA Weighted: 0.0000
05-09 07:40:49 Train-Loss Entropy Weighted: 0.0000
05-09 07:40:49 Train-Acc Source Data: 0.8947
100%|###################################################################################################################################| 142/142 [00:01<00:00, 140.97it/s]
05-09 07:40:50 Val-acc: 0.8434
05-09 07:40:50 The best model epoch 1, val-acc 0.8434
05-09 07:40:50 -----Epoch 2/2-----
05-09 07:40:50 current lr: [0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2111/2111 [08:41<00:00,  4.04it/s]
05-09 07:49:33 MFSAN-CDA active: lambda_cda=0.020000, lambda_ent=0.005000, detach_prob=True
05-09 07:49:33 Train-Loss Source Classifier: 0.0427
05-09 07:49:33 Train-Loss MMD: 0.0982
05-09 07:49:33 Train-Loss L1: 0.0115
05-09 07:49:33 Train-Loss CDA MMD: 0.0987
05-09 07:49:33 Train-Loss Target Entropy: 0.1624
05-09 07:49:33 Train-Loss CDA Weighted: 0.0020
05-09 07:49:33 Train-Loss Entropy Weighted: 0.0008
05-09 07:49:33 Train-Acc Source Data: 0.9898
100%|###################################################################################################################################| 142/142 [00:01<00:00, 134.83it/s]
05-09 07:49:34 Val-acc: 0.9931
05-09 07:49:34 The best model epoch 2, val-acc 0.9931
05-09 07:49:35 Model saved to ./ckpt/MFSAN_CDA/multi_source/[PU_0_PU_1_PU_3]To[PU_2]_0509-073204.pth
(MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels# 


python train.py \
  --model_name MFSAN_CDA \
  --source PU_0,PU_2,PU_3 \
  --target PU_1 \
  --train_mode multi_source \
  --data_dir /workspace/PU_TL \
  --signal_size 1024 \
  --backbone CNN \
  --cuda_device 0 \
  --max_epoch 2 \
  --lambda_cda 0.02 \
  --lambda_ent 0.005 \
  --include_faults K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18


  (MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels# python train.py \
>   --model_name MFSAN_CDA \
>   --source PU_0,PU_2,PU_3 \
>   --target PU_1 \
>   --train_mode multi_source \
>   --data_dir /workspace/PU_TL \
>   --signal_size 1024 \
>   --backbone CNN \
>   --cuda_device 0 \
>   --max_epoch 2 \
>   --lambda_cda 0.02 \
>   --lambda_ent 0.005 \
>   --include_faults K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18
05-09 07:57:43 model_name: MFSAN_CDA
05-09 07:57:43 source: PU_0,PU_2,PU_3
05-09 07:57:43 target: PU_1
05-09 07:57:43 data_dir: /workspace/PU_TL
05-09 07:57:43 train_mode: multi_source
05-09 07:57:43 cuda_device: 0
05-09 07:57:43 max_epoch: 2
05-09 07:57:43 batch_size: 64
05-09 07:57:43 signal_size: 1024
05-09 07:57:43 random_state: 10
05-09 07:57:43 include_faults: K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18
05-09 07:57:43 exclude_faults: 
05-09 07:57:43 opt: sgd
05-09 07:57:43 momentum: 0.9
05-09 07:57:43 betas: (0.9, 0.999)
05-09 07:57:43 weight_decay: 0.0005
05-09 07:57:43 lr: 0.01
05-09 07:57:43 lr_scheduler: stepLR
05-09 07:57:43 gamma: 0.2
05-09 07:57:43 steps: 10
05-09 07:57:43 backbone: CNN
05-09 07:57:43 num_workers: 4
05-09 07:57:43 normlize_type: -1-1
05-09 07:57:43 tradeoff: ['exp', 'exp', 'exp']
05-09 07:57:43 zeta: 10.0
05-09 07:57:43 dropout: 0.0
05-09 07:57:43 lambda_cda: 0.02
05-09 07:57:43 lambda_ent: 0.005
05-09 07:57:43 cda_detach_prob: True
05-09 07:57:43 save: True
05-09 07:57:43 save_dir: ./ckpt
05-09 07:57:43 load_path: 
05-09 07:57:43 save_path: ./ckpt/MFSAN_CDA/multi_source/[PU_0_PU_2_PU_3]To[PU_1]_0509-075743
05-09 07:57:43 Source PU_0 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 07:57:43 Source PU_2 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 07:57:43 Source PU_3 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 07:57:43 Target PU_1 detected 9 classes: ['K001' 'KA04' 'KA16' 'KA30' 'KB24' 'KB27' 'KI04' 'KI17' 'KI18']
05-09 07:57:43 The scenario is: closed-set domain adaptation
05-09 07:57:44 using 1 / 1 gpus
05-09 07:57:45 Using model: MFSAN_CDA
05-09 07:57:45 Using backbone: CNN
05-09 07:57:45 Backbone output dim: 640
05-09 07:57:48 Source set PU_0 number of samples: 45103.
Label 0 has samples: 5001
Label 1 has samples: 5000
Label 2 has samples: 5005
Label 3 has samples: 5005
Label 4 has samples: 5012
Label 5 has samples: 5022
Label 6 has samples: 5019
Label 7 has samples: 5033
Label 8 has samples: 5006
05-09 07:57:48 Source set PU_2 number of samples: 45143.
Label 0 has samples: 5001
Label 1 has samples: 5005
Label 2 has samples: 5013
Label 3 has samples: 5018
Label 4 has samples: 5008
Label 5 has samples: 5059
Label 6 has samples: 5000
Label 7 has samples: 5040
Label 8 has samples: 4999
05-09 07:57:48 Source set PU_3 number of samples: 45068.
Label 0 has samples: 5014
Label 1 has samples: 5000
Label 2 has samples: 5000
Label 3 has samples: 5000
Label 4 has samples: 5005
Label 5 has samples: 5000
Label 6 has samples: 5005
Label 7 has samples: 5021
Label 8 has samples: 5023
05-09 07:57:49 Training set number of samples: 36015.
Label 0 has samples: 4001
Label 1 has samples: 4004
Label 2 has samples: 4001
Label 3 has samples: 3999
Label 4 has samples: 4004
Label 5 has samples: 4003
Label 6 has samples: 4000
Label 7 has samples: 4003
Label 8 has samples: 4000
05-09 07:57:49 Validation set number of samples: 9008.
Label 0 has samples: 1001
Label 1 has samples: 1002
Label 2 has samples: 1001
Label 3 has samples: 1000
Label 4 has samples: 1001
Label 5 has samples: 1001
Label 6 has samples: 1001
Label 7 has samples: 1001
Label 8 has samples: 1000
05-09 07:57:50 CDA lambda_cda: 0.020000
05-09 07:57:50 CDA lambda_ent: 0.005000
05-09 07:57:50 CDA detach probabilities for joint feature: True
05-09 07:57:50 CDA joint feature dim: 9 x 40 = 360
05-09 07:57:50 -----Epoch 1/2-----
05-09 07:57:50 current lr: [0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2113/2113 [08:37<00:00,  4.08it/s]
05-09 08:06:28 MFSAN-CDA active: lambda_cda=0.020000, lambda_ent=0.005000, detach_prob=True
05-09 08:06:28 Train-Loss Source Classifier: 0.2692
05-09 08:06:28 Train-Loss MMD: 0.2849
05-09 08:06:28 Train-Loss L1: 0.0868
05-09 08:06:28 Train-Loss CDA MMD: 0.2414
05-09 08:06:28 Train-Loss Target Entropy: 0.7024
05-09 08:06:28 Train-Loss CDA Weighted: 0.0000
05-09 08:06:28 Train-Loss Entropy Weighted: 0.0000
05-09 08:06:28 Train-Acc Source Data: 0.8974
100%|###################################################################################################################################| 141/141 [00:01<00:00, 131.17it/s]
05-09 08:06:29 Val-acc: 0.9123
05-09 08:06:29 Val-Class-0 | Precision: 0.9819 | Recall: 0.9750 | F1: 0.9784 | Support: 1001
05-09 08:06:29 Val-Class-1 | Precision: 0.8828 | Recall: 0.9022 | F1: 0.8924 | Support: 1002
05-09 08:06:29 Val-Class-2 | Precision: 0.9435 | Recall: 0.9011 | F1: 0.9218 | Support: 1001
05-09 08:06:29 Val-Class-3 | Precision: 0.8997 | Recall: 0.9870 | F1: 0.9413 | Support: 1000
05-09 08:06:29 Val-Class-4 | Precision: 0.9987 | Recall: 0.7972 | F1: 0.8867 | Support: 1001
05-09 08:06:29 Val-Class-5 | Precision: 0.8379 | Recall: 0.9401 | F1: 0.8861 | Support: 1001
05-09 08:06:29 Val-Class-6 | Precision: 0.9544 | Recall: 0.9830 | F1: 0.9685 | Support: 1001
05-09 08:06:29 Val-Class-7 | Precision: 0.8311 | Recall: 0.9980 | F1: 0.9069 | Support: 1001
05-09 08:06:29 Val-Class-8 | Precision: 0.9297 | Recall: 0.7270 | F1: 0.8159 | Support: 1000
05-09 08:06:29 Val-F1-macro: 0.9109
05-09 08:06:29 Val-F1-weighted: 0.9109
05-09 08:06:29 The best model epoch 1, val-acc 0.9123
05-09 08:06:29 -----Epoch 2/2-----
05-09 08:06:29 current lr: [0.01, 0.01, 0.01]
100%|##################################################################################################################################| 2113/2113 [08:42<00:00,  4.04it/s]
05-09 08:15:13 MFSAN-CDA active: lambda_cda=0.020000, lambda_ent=0.005000, detach_prob=True
05-09 08:15:13 Train-Loss Source Classifier: 0.0405
05-09 08:15:13 Train-Loss MMD: 0.0973
05-09 08:15:13 Train-Loss L1: 0.0120
05-09 08:15:13 Train-Loss CDA MMD: 0.0986
05-09 08:15:13 Train-Loss Target Entropy: 0.1633
05-09 08:15:13 Train-Loss CDA Weighted: 0.0020
05-09 08:15:13 Train-Loss Entropy Weighted: 0.0008
05-09 08:15:13 Train-Acc Source Data: 0.9904
100%|###################################################################################################################################| 141/141 [00:01<00:00, 130.79it/s]
05-09 08:15:15 Val-acc: 0.9900
05-09 08:15:15 Val-Class-0 | Precision: 0.9960 | Recall: 0.9990 | F1: 0.9975 | Support: 1001
05-09 08:15:15 Val-Class-1 | Precision: 0.9990 | Recall: 1.0000 | F1: 0.9995 | Support: 1002
05-09 08:15:15 Val-Class-2 | Precision: 0.9990 | Recall: 0.9990 | F1: 0.9990 | Support: 1001
05-09 08:15:15 Val-Class-3 | Precision: 0.9859 | Recall: 0.9790 | F1: 0.9824 | Support: 1000
05-09 08:15:15 Val-Class-4 | Precision: 1.0000 | Recall: 0.9930 | F1: 0.9965 | Support: 1001
05-09 08:15:15 Val-Class-5 | Precision: 0.9398 | Recall: 0.9990 | F1: 0.9685 | Support: 1001
05-09 08:15:15 Val-Class-6 | Precision: 1.0000 | Recall: 0.9870 | F1: 0.9935 | Support: 1001
05-09 08:15:15 Val-Class-7 | Precision: 0.9990 | Recall: 0.9600 | F1: 0.9791 | Support: 1001
05-09 08:15:15 Val-Class-8 | Precision: 0.9950 | Recall: 0.9940 | F1: 0.9945 | Support: 1000
05-09 08:15:15 Val-F1-macro: 0.9901
05-09 08:15:15 Val-F1-weighted: 0.9901
05-09 08:15:15 The best model epoch 2, val-acc 0.9900
05-09 08:15:16 Model saved to ./ckpt/MFSAN_CDA/multi_source/[PU_0_PU_2_PU_3]To[PU_1]_0509-075743.pth
(MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels# 