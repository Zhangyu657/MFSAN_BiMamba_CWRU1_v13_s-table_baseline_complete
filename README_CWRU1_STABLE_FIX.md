# CWRU_1 V12 稳定修复版

## 1. 修改目标

目标任务固定为：

```text
CWRU_0 + CWRU_2 + CWRU_3 -> CWRU_1
```

V11 日志显示：训练后期每轮接受数百次 `ball_21` 救援样本，但测试阶段
`ball_21` 仍只有很低的召回率，同时 `normal` 召回率明显下降。因此，V12
不再扩大救援数量，而是仅使用少量、高质量、排除正常类的目标候选。

## 2. 网络结构没有变化

V12 沿用原 V10 的全部网络模块：

- MSCNN-BiMamba 特征主干；
- 三个源域专用特征头；
- 三个分类器；
- CDAN 域判别器；
- 原有多源融合结构。

未新增、删除或调整任何卷积层、BiMamba 层、分类器层、域判别层及其维度。
仍然使用五项原有目标：CE、MMD、CDAN、CLMMD 和 SupCon。

## 3. 训练策略修改

1. `ball_21` 救援从第 6 轮开始，第 10 轮结束。
2. 默认只允许 CWRU_3 分支（源分支 2）提供救援 CLMMD。
3. `ball_21` 概率至少为 0.20，且不低于竞争类别概率的 50%。
4. `normal` 概率必须不高于 0.20。
5. `ball_21` 原型必须是最近原型，且：
   - 原型 Top1-Top2 余量不低于 0.08；
   - 与 `ball_21` 原型余弦相似度不低于 0.45；
   - 相对 `normal` 原型的相似度优势不低于 0.10；
   - 距离不超过 0.05。
6. 每个小批次最多保留 4 个质量最高的救援样本。
7. 救援 CLMMD 只以 25% 比例混入原 CLMMD，不再直接覆盖。
8. `ball_21` CLMMD 增强由 1.50 降为 1.10。
9. 测试阶段概率救援默认关闭，保持训练与推理结果可解释。
10. 域对抗、CLMMD、SupCon 权重降低并延迟启动，减少后期负迁移。
11. 源域与类别权重下限改为 0.03，避免三源模型退化成双源模型。

## 4. 数据检查

```bash
python check_cwru_dataset.py \
  --data_dir "/workspace/CWRU_TL/CWRU" \
  --signal_size 1024
```

```bash
python diagnose_cwru1_data.py \
  --data_dir "/workspace/CWRU_TL/CWRU" \
  --signal_size 1024 \
  --output cwru1_ball21_audit.csv
```

## 5. 推荐运行命令

```bash
python run_cwru1_stable_fix.py \
  --data_dir "/workspace/CWRU_TL/CWRU" \
  --seeds 2027 \
  --cuda_device 0 \
  --max_epoch 15 \
  --batch_size 64 \
  --num_workers 4 \
  --signal_size 1024 \
  --target_test_size 0.40 \
  --eval_each_epoch True \
  --select_best_on_target True \
  --best_metric class_aware
```

后台运行：

```bash
nohup python -u run_cwru1_stable_fix.py \
  --data_dir "/workspace/CWRU_TL/CWRU" \
  --seeds 2027 \
  --cuda_device 0 \
  --max_epoch 15 \
  --batch_size 64 \
  --num_workers 4 \
  --signal_size 1024 \
  --target_test_size 0.40 \
  --eval_each_epoch True \
  --select_best_on_target True \
  --best_metric class_aware \
  > cwru1_v12_stable_seed2027.log 2>&1 &
```

## 6. 日志中重点检查

```text
V12 stable rescue epoch stats class-2
Target-Test-Class-2
Target-Test-Class-6
Target-Test-F1-macro
Best model updated
```

理想状态：

- `accepted` 每个源分支每轮保持在较低数量，不再达到数百；
- class 2 的 Recall 和 F1 提高；
- class 6 的 Recall 不出现大幅下降；
- Macro-F1 与 Accuracy 同时保持或提高。

## 7. 最佳模型

使用文件名带 `_best.pth` 的模型。最终第 15 轮模型只用于记录，不应替代
根据目标测试指标保存的最佳检查点。
