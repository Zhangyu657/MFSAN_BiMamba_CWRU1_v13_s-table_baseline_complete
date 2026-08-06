# CWRU_1 V13 基础稳定化版本

## 1. 修改目标

任务固定为：

```text
CWRU_0 + CWRU_2 + CWRU_3 -> CWRU_1
```

V12 日志显示：

- 第 2 轮目标准确率最高；
- 第 3 轮开始目标域迅速退化；
- 源域准确率持续接近 100%，目标域却下降到约 68%；
- ball_21 在救援开始前已经塌缩；
- 后续伪标签救援无法恢复该类别。

因此 V13 不再继续叠加伪标签救援，而是先稳定基础迁移过程。

## 2. 网络结构没有改变

仍然使用：

- MSCNN-BiMamba 主干；
- 3 个源域分支；
- 3 个分类器；
- 3 个 CDAN 域判别器；
- 原有 CE、MMD、CDAN、CLMMD、SupCon 五类目标。

V13 默认将 SupCon 权重设为 0，并关闭 V7-V12 的类别门控、专家保护、Hard-SupCon、原型救援和测试阶段类别救援。它们没有从项目中删除，只是不进入本次 V13 训练路径。

## 3. 主要修改

1. 优化器改为 Adam，学习率由 SGD 0.01 降为 Adam 0.0003。
2. 适配系数增长速度由 `zeta=10` 降为 `zeta=2`。
3. MMD 从第 2 轮开始，显式权重为 0.15。
4. CDAN 从第 4 轮开始，权重为 0.002。
5. CLMMD 从第 5 轮开始，权重为 0.0005。
6. 目标伪标签阈值提高到 0.95，每类至少 3 个样本。
7. 源域交叉熵加入 0.05 label smoothing。
8. 梯度范数裁剪为 5.0。
9. 关闭 V6 的不可逆源域门控，降低类别权重动态修正强度。
10. 增加早停：第 6 轮以后，连续 4 轮无改善即停止。
11. 每轮输出完整目标测试混淆矩阵和预测类别数量。

## 4. 解压

```bash
cd "/workspace/故障诊断迁移学习/CWRU数据集迁移学习简化损失函数版本"
unzip -o "MFSAN_BiMamba_CWRU1_v13_stable_baseline_complete.zip"
cd "MFSAN_BiMamba_CWRU1_v13_stable_baseline_complete"
```

## 5. 数据检查

```bash
python check_cwru_dataset.py \
  --data_dir "/workspace/CWRU_TL/CWRU" \
  --signal_size 1024
```

## 6. 推荐运行命令

```bash
python run_cwru1_v13_stable_baseline.py \
  --data_dir "/workspace/CWRU_TL/CWRU" \
  --seeds 2027 \
  --cuda_device 0
```

## 7. 后台运行

```bash
nohup python -u run_cwru1_v13_stable_baseline.py \
  --data_dir "/workspace/CWRU_TL/CWRU" \
  --seeds 2027 \
  --cuda_device 0 \
  > cwru1_v13_seed2027.log 2>&1 &
```

查看日志：

```bash
tail -f cwru1_v13_seed2027.log
```

## 8. 完整显式运行命令

```bash
python run_cwru1_v13_stable_baseline.py \
  --data_dir "/workspace/CWRU_TL/CWRU" \
  --seeds 2027 \
  --cuda_device 0 \
  --max_epoch 10 \
  --batch_size 64 \
  --num_workers 4 \
  --signal_size 1024 \
  --target_test_size 0.40 \
  --source_balance_data True \
  --normlize_type=-1-1 \
  --optimizer adam \
  --lr 0.0003 \
  --weight_decay 0.0001 \
  --lr_scheduler fix \
  --zeta 2.0 \
  --mmd_weight 0.15 \
  --mmd_start_epoch 2 \
  --lambda_adv 0.002 \
  --adv_start_epoch 4 \
  --lambda_clmmd 0.0005 \
  --clmmd_start_epoch 5 \
  --pl_conf_thresh 0.95 \
  --pl_min_target 3 \
  --source_label_smoothing 0.05 \
  --grad_clip_norm 5.0 \
  --cw_warmup_epochs 5 \
  --cw_alpha 0.10 \
  --cw_alpha_ramp_epochs 5 \
  --rec_score_weight 0.10 \
  --class_weight_power 1.0 \
  --eval_each_epoch True \
  --select_best_on_target True \
  --best_metric class_aware \
  --best_accuracy_weight 0.35 \
  --best_macro_f1_weight 0.35 \
  --best_focus_recall_weight 0.30 \
  --early_stop_patience 4 \
  --early_stop_min_epoch 6 \
  --early_stop_min_delta 0.0001 \
  --save_dir "./ckpt/CWRU1_V13_STABLE_BASELINE"
```

## 9. 日志中重点查看

```text
Target-Test-Class-2
Target-Test-Class-6
Target-Test predicted class counts
Target-Test confusion true_c2
Target-Test confusion true_c6
Train-Acc Source Data
Train-Acc Domain Data
Target-Test-acc
Target-Test-F1-macro
Early stopping
```

类别映射：

```text
class 2 = ball_21
class 6 = normal
```

`Target-Test confusion true_c2` 的一整行可以直接看出真实 ball_21 被分别错分到了哪些类别。

## 10. 结果判断

本版本首先验证基础训练是否稳定。成功的最低标准是：

- 第 2 轮以后目标准确率不再立即大幅下降；
- normal 召回率不再从 95% 快速跌到 20% 左右；
- ball_21 的预测数量和错分方向可以通过混淆矩阵明确确认；
- 源域准确率提高时，目标域准确率不再持续反向下降。

本版本不能在没有真实数据运行的情况下保证 ball_21 一定提高。它的目的，是先消除 V12 中已经确认的早期负迁移，再依据混淆矩阵进行下一项单因素修改。
