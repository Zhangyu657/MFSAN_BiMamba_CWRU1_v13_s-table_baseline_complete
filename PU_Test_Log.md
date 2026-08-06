(MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library# python train.py   --model_name MFSAN   --source PU_0,PU_1,PU_2   --target PU_3   --train_mode multi_source   --data_dir /workspace/PU_TL   --signal_size 1024   --cuda_device 0   --max_epoch 2
05-08 12:01:45 model_name: MFSAN
05-08 12:01:45 source: PU_0,PU_1,PU_2
05-08 12:01:45 target: PU_3
05-08 12:01:45 data_dir: /workspace/PU_TL
05-08 12:01:45 train_mode: multi_source
05-08 12:01:45 cuda_device: 0
05-08 12:01:45 max_epoch: 2
05-08 12:01:45 batch_size: 64
05-08 12:01:45 signal_size: 1024
05-08 12:01:45 random_state: 10
05-08 12:01:45 opt: sgd
05-08 12:01:45 momentum: 0.9
05-08 12:01:45 betas: (0.9, 0.999)
05-08 12:01:45 weight_decay: 0.0005
05-08 12:01:45 lr: 0.01
05-08 12:01:45 lr_scheduler: stepLR
05-08 12:01:45 gamma: 0.2
05-08 12:01:45 steps: 10
05-08 12:01:45 backbone: CNN
05-08 12:01:45 num_workers: 4
05-08 12:01:45 normlize_type: -1-1
05-08 12:01:45 tradeoff: ['exp', 'exp', 'exp']
05-08 12:01:45 zeta: 10.0
05-08 12:01:45 dropout: 0.0
05-08 12:01:45 save: True
05-08 12:01:45 save_dir: ./ckpt
05-08 12:01:45 load_path: 
05-08 12:01:45 save_path: ./ckpt/MFSAN/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0508-120145
05-08 12:01:45 Source PU_0 detected 13 classes: ['K001' 'KA04' 'KA15' 'KA16' 'KA30' 'KB23' 'KB24' 'KB27' 'KI04' 'KI16'
 'KI17' 'KI18' 'KI21']
05-08 12:01:45 Source PU_1 detected 13 classes: ['K001' 'KA04' 'KA15' 'KA16' 'KA30' 'KB23' 'KB24' 'KB27' 'KI04' 'KI16'
 'KI17' 'KI18' 'KI21']
05-08 12:01:45 Source PU_2 detected 13 classes: ['K001' 'KA04' 'KA15' 'KA16' 'KA30' 'KB23' 'KB24' 'KB27' 'KI04' 'KI16'
 'KI17' 'KI18' 'KI21']
05-08 12:01:45 Target PU_3 detected 13 classes: ['K001' 'KA04' 'KA15' 'KA16' 'KA30' 'KB23' 'KB24' 'KB27' 'KI04' 'KI16'
 'KI17' 'KI18' 'KI21']
05-08 12:01:45 The scenario is: closed-set domain adaptation
05-08 12:01:45 using 1 / 1 gpus
05-08 12:01:51 Source set PU_0 number of samples: 65116.
Label 0 has samples: 5001
Label 1 has samples: 5000
Label 2 has samples: 4999
Label 3 has samples: 5005
Label 4 has samples: 5005
Label 5 has samples: 4998
Label 6 has samples: 5012
Label 7 has samples: 5022
Label 8 has samples: 5019
Label 9 has samples: 5000
Label 10 has samples: 5033
Label 11 has samples: 5006
Label 12 has samples: 5016
05-08 12:01:51 Source set PU_1 number of samples: 65046.
Label 0 has samples: 5002
Label 1 has samples: 5006
Label 2 has samples: 4999
Label 3 has samples: 5002
Label 4 has samples: 4999
Label 5 has samples: 4999
Label 6 has samples: 5005
Label 7 has samples: 5004
Label 8 has samples: 5001
Label 9 has samples: 5017
Label 10 has samples: 5004
Label 11 has samples: 5000
Label 12 has samples: 5008
05-08 12:01:51 Source set PU_2 number of samples: 65258.
Label 0 has samples: 5001
Label 1 has samples: 5005
Label 2 has samples: 5000
Label 3 has samples: 5013
Label 4 has samples: 5018
Label 5 has samples: 5030
Label 6 has samples: 5008
Label 7 has samples: 5059
Label 8 has samples: 5000
Label 9 has samples: 5085
Label 10 has samples: 5040
Label 11 has samples: 4999
Label 12 has samples: 5000
05-08 12:01:52 Training set number of samples: 52141.
Label 0 has samples: 4011
Label 1 has samples: 4000
Label 2 has samples: 4016
Label 3 has samples: 4000
Label 4 has samples: 4000
Label 5 has samples: 4016
Label 6 has samples: 4004
Label 7 has samples: 4000
Label 8 has samples: 4004
Label 9 has samples: 4055
Label 10 has samples: 4016
Label 11 has samples: 4018
Label 12 has samples: 4001
05-08 12:01:52 Validation set number of samples: 13040.
Label 0 has samples: 1003
Label 1 has samples: 1000
Label 2 has samples: 1005
Label 3 has samples: 1000
Label 4 has samples: 1000
Label 5 has samples: 1005
Label 6 has samples: 1001
Label 7 has samples: 1000
Label 8 has samples: 1001
Label 9 has samples: 1014
Label 10 has samples: 1005
Label 11 has samples: 1005
Label 12 has samples: 1001
05-08 12:01:53 -----Epoch 1/2-----
05-08 12:01:53 current lr: [0.01, 0.01, 0.01]
100%|##################################################################################################################################| 3052/3052 [12:15<00:00,  4.15it/s]
05-08 12:14:08 Train-Loss Source Classifier: 0.3264
05-08 12:14:08 Train-Loss MMD: 0.3860
05-08 12:14:08 Train-Loss L1: 0.0539
05-08 12:14:08 Train-Acc Source Data: 0.8821
100%|###################################################################################################################################| 204/204 [00:01<00:00, 141.76it/s]
05-08 12:14:10 Val-acc: 0.6260
05-08 12:14:10 The best model epoch 1, val-acc 0.6260
05-08 12:14:10 -----Epoch 2/2-----
05-08 12:14:10 current lr: [0.01, 0.01, 0.01]
100%|##################################################################################################################################| 3052/3052 [12:21<00:00,  4.12it/s]
05-08 12:26:32 Train-Loss Source Classifier: 0.0537
05-08 12:26:32 Train-Loss MMD: 0.1030
05-08 12:26:32 Train-Loss L1: 0.0126
05-08 12:26:32 Train-Acc Source Data: 0.9877
100%|###################################################################################################################################| 204/204 [00:01<00:00, 127.17it/s]
05-08 12:26:34 Val-acc: 0.9651
05-08 12:26:34 The best model epoch 2, val-acc 0.9651
05-08 12:26:35 Model saved to ./ckpt/MFSAN/multi_source/[PU_0_PU_1_PU_2]To[PU_3]_0508-120145.pth
(MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library# 



(MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library# python train.py \
>   --model_name MFSAN \
>   --source PU_0,PU_1,PU_3 \
>   --target PU_2 \
>   --train_mode multi_source \
>   --data_dir /workspace/PU_TL \
>   --signal_size 1024 \
>   --cuda_device 0 \
>   --max_epoch 2
05-08 12:31:50 model_name: MFSAN
05-08 12:31:50 source: PU_0,PU_1,PU_3
05-08 12:31:50 target: PU_2
05-08 12:31:50 data_dir: /workspace/PU_TL
05-08 12:31:50 train_mode: multi_source
05-08 12:31:50 cuda_device: 0
05-08 12:31:50 max_epoch: 2
05-08 12:31:50 batch_size: 64
05-08 12:31:50 signal_size: 1024
05-08 12:31:50 random_state: 10
05-08 12:31:50 opt: sgd
05-08 12:31:50 momentum: 0.9
05-08 12:31:50 betas: (0.9, 0.999)
05-08 12:31:50 weight_decay: 0.0005
05-08 12:31:50 lr: 0.01
05-08 12:31:50 lr_scheduler: stepLR
05-08 12:31:50 gamma: 0.2
05-08 12:31:50 steps: 10
05-08 12:31:50 backbone: CNN
05-08 12:31:50 num_workers: 4
05-08 12:31:50 normlize_type: -1-1
05-08 12:31:50 tradeoff: ['exp', 'exp', 'exp']
05-08 12:31:50 zeta: 10.0
05-08 12:31:50 dropout: 0.0
05-08 12:31:50 save: True
05-08 12:31:50 save_dir: ./ckpt
05-08 12:31:50 load_path: 
05-08 12:31:50 save_path: ./ckpt/MFSAN/multi_source/[PU_0_PU_1_PU_3]To[PU_2]_0508-123150
05-08 12:31:50 Source PU_0 detected 13 classes: ['K001' 'KA04' 'KA15' 'KA16' 'KA30' 'KB23' 'KB24' 'KB27' 'KI04' 'KI16'
 'KI17' 'KI18' 'KI21']
05-08 12:31:50 Source PU_1 detected 13 classes: ['K001' 'KA04' 'KA15' 'KA16' 'KA30' 'KB23' 'KB24' 'KB27' 'KI04' 'KI16'
 'KI17' 'KI18' 'KI21']
05-08 12:31:50 Source PU_3 detected 13 classes: ['K001' 'KA04' 'KA15' 'KA16' 'KA30' 'KB23' 'KB24' 'KB27' 'KI04' 'KI16'
 'KI17' 'KI18' 'KI21']
05-08 12:31:50 Target PU_2 detected 13 classes: ['K001' 'KA04' 'KA15' 'KA16' 'KA30' 'KB23' 'KB24' 'KB27' 'KI04' 'KI16'
 'KI17' 'KI18' 'KI21']
05-08 12:31:50 The scenario is: closed-set domain adaptation
05-08 12:31:50 using 1 / 1 gpus
05-08 12:31:55 Source set PU_0 number of samples: 65116.
Label 0 has samples: 5001
Label 1 has samples: 5000
Label 2 has samples: 4999
Label 3 has samples: 5005
Label 4 has samples: 5005
Label 5 has samples: 4998
Label 6 has samples: 5012
Label 7 has samples: 5022
Label 8 has samples: 5019
Label 9 has samples: 5000
Label 10 has samples: 5033
Label 11 has samples: 5006
Label 12 has samples: 5016
05-08 12:31:55 Source set PU_1 number of samples: 65046.
Label 0 has samples: 5002
Label 1 has samples: 5006
Label 2 has samples: 4999
Label 3 has samples: 5002
Label 4 has samples: 4999
Label 5 has samples: 4999
Label 6 has samples: 5005
Label 7 has samples: 5004
Label 8 has samples: 5001
Label 9 has samples: 5017
Label 10 has samples: 5004
Label 11 has samples: 5000
Label 12 has samples: 5008
05-08 12:31:55 Source set PU_3 number of samples: 65181.
Label 0 has samples: 5014
Label 1 has samples: 5000
Label 2 has samples: 5021
Label 3 has samples: 5000
Label 4 has samples: 5000
Label 5 has samples: 5021
Label 6 has samples: 5005
Label 7 has samples: 5000
Label 8 has samples: 5005
Label 9 has samples: 5069
Label 10 has samples: 5021
Label 11 has samples: 5023
Label 12 has samples: 5002
05-08 12:31:56 Training set number of samples: 52204.
Label 0 has samples: 4000
Label 1 has samples: 4004
Label 2 has samples: 4000
Label 3 has samples: 4010
Label 4 has samples: 4014
Label 5 has samples: 4024
Label 6 has samples: 4006
Label 7 has samples: 4047
Label 8 has samples: 4000
Label 9 has samples: 4068
Label 10 has samples: 4032
Label 11 has samples: 3999
Label 12 has samples: 4000
05-08 12:31:56 Validation set number of samples: 13054.
Label 0 has samples: 1001
Label 1 has samples: 1001
Label 2 has samples: 1000
Label 3 has samples: 1003
Label 4 has samples: 1004
Label 5 has samples: 1006
Label 6 has samples: 1002
Label 7 has samples: 1012
Label 8 has samples: 1000
Label 9 has samples: 1017
Label 10 has samples: 1008
Label 11 has samples: 1000
Label 12 has samples: 1000
05-08 12:31:57 -----Epoch 1/2-----
05-08 12:31:57 current lr: [0.01, 0.01, 0.01]
100%|##################################################################################################################################| 3051/3051 [12:12<00:00,  4.16it/s]
05-08 12:44:10 Train-Loss Source Classifier: 0.3292
05-08 12:44:10 Train-Loss MMD: 0.3494
05-08 12:44:10 Train-Loss L1: 0.0689
05-08 12:44:10 Train-Acc Source Data: 0.8824
100%|###################################################################################################################################| 204/204 [00:01<00:00, 141.55it/s]
05-08 12:44:12 Val-acc: 0.8589
05-08 12:44:12 The best model epoch 1, val-acc 0.8589
05-08 12:44:12 -----Epoch 2/2-----
05-08 12:44:12 current lr: [0.01, 0.01, 0.01]
100%|##################################################################################################################################| 3051/3051 [12:20<00:00,  4.12it/s]
05-08 12:56:33 Train-Loss Source Classifier: 0.0559
05-08 12:56:33 Train-Loss MMD: 0.1014
05-08 12:56:33 Train-Loss L1: 0.0114
05-08 12:56:33 Train-Acc Source Data: 0.9863
100%|###################################################################################################################################| 204/204 [00:01<00:00, 140.09it/s]
05-08 12:56:35 Val-acc: 0.9778
05-08 12:56:35 The best model epoch 2, val-acc 0.9778
05-08 12:56:36 Model saved to ./ckpt/MFSAN/multi_source/[PU_0_PU_1_PU_3]To[PU_2]_0508-123150.pth
(MAMBA_COPY2) root@VM-0-80-ubuntu:/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library# 