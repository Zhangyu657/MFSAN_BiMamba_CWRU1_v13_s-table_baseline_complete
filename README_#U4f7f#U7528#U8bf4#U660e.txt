本补丁新增模型：MFSAN_CDAN_BIMAMBA_CW_RWCA

修改方式：
1. 把 models/MFSAN_CDAN_BIMAMBA_CW_RWCA.py 复制到你项目的 models/ 目录下。
2. train.py 不需要修改，因为原项目通过 importlib 按 --model_name 动态导入模型文件。
3. opt.py 不需要修改，因为 --model_name 没有 choices 限制。

运行 PU0 目标域：
python train.py \
  --model_name MFSAN_CDAN_BIMAMBA_CW_RWCA \
  --source PU_1,PU_2,PU_3 \
  --target PU_0 \
  --data_dir /workspace/PU_TL_9_replace \
  --train_mode multi_source \
  --cuda_device 0 \
  --max_epoch 10 \
  --batch_size 64 \
  --signal_size 1024 \
  --random_state 10 \
  --include_faults K001,KA04,KA16,KA30,KB24,KB23,KI04,KI17,KI16 \
  --opt sgd \
  --lr 0.01 \
  --lr_scheduler stepLR \
  --steps 10 \
  --gamma 0.2 \
  --backbone CNN \
  --normlize_type=-1-1 \
  --lambda_cda 0.02 \
  --lambda_ent 0.005 \
  --lambda_adv 0.01 \
  --lambda_grl 0.5 \
  --adv_conf_thresh 0.8 \
  --rw_tau 1.0 \
  --rw_eval_tau 0.5 \
  --lambda_clmmd 0.01

核心变化：
原 RWCA: 每个源域一个全局权重 w_s。
新 CW-RWCA: 每个源域、每个类别一个权重 w_{s,c}。

重点观察日志：
CW-RWCA average train class-1 source weights
CW-RWCA EMA class-1 source weights
CW-RWCA average train class-2 source weights
CW-RWCA EMA class-2 source weights

PU 9类顺序通常为：
Class-0 K001
Class-1 KA04
Class-2 KA16
Class-3 KA30
Class-4 KB23
Class-5 KB24
Class-6 KI04
Class-7 KI16
Class-8 KI17
