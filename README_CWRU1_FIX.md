# CWRU_1（ball_21 类别塌缩）修复版

## 1. 现有日志暴露的问题

任务：`CWRU_0 + CWRU_2 + CWRU_3 -> CWRU_1`。

原V10最佳准确率为91.50%，但类别2（`ball_21`）Recall只有2.13%，47个样本中约只有1个识别正确。与此同时，类别0和类别1的Recall达到100%但Precision较低，说明类别2很可能被吸收到`ball_07`或`ball_14`中。

原配置还存在两个问题：

1. CWRU通用配置关闭了Hard-SupCon和原型筛选，因此V10的困难类别机制实际上没有用于CWRU。
2. 最佳模型仅按Accuracy选择，正常类样本较多，可能掩盖单个故障类别完全失效。

## 2. 本版本修改内容

### 2.1 新模型V11

新增：

`models/MFSAN_CDAN_BIMAMBA_CW_RWCA_V11_CWRU1_CLASS_RESCUE.py`

不增加第六项损失，总目标仍是：

`CE + MMD + CDAN + CLMMD + SupCon`

V11只修改现有CLMMD和测试融合的使用方式：

- 类别2进入分类器Top-2；
- 类别2概率不低于0.10；
- 当前Top-1属于类别0、1或2；
- 最近源域原型也判断为类别2；
- 原型相似度、margin和源域半径均通过；

满足这些条件后，目标样本才作为类别2参与现有CLMMD。这样可以打破“类别2没有Top-1伪标签，因此永远无法类别对齐”的循环。

测试阶段还增加了保守校准：只有至少两个源分类器把类别2放进Top-2，并且类别2概率接近类别0/1时，才给予类别2有限提升。

### 2.2 类别感知最佳模型选择

修改：

- `train_utils.py`
- `models/MFSAN_CDAN_BIMAMBA_CW_RWCA_V2.py`
- `opt.py`

新增三种选择指标：

- `accuracy`：完全保持原项目；
- `macro_f1`：按Macro-F1选择；
- `class_aware`：综合Accuracy、Macro-F1和指定类别Recall。

CWRU_1修复脚本默认使用：

`0.45 × Accuracy + 0.35 × Macro-F1 + 0.20 × ball_21 Recall`

这仍然是用户要求的“每轮测试并选择最佳模型”，只是避免Accuracy掩盖类别2塌缩。

### 2.3 CWRU_1专用训练参数

新增：

`run_cwru1_fix.py`

关键设置：

- Hard-SupCon困难类别对：`0-2`和`1-2`；
- 原型筛选类别：`0,1,2`；
- 类别2 CLMMD倍率：1.50；
- `lambda_adv`从0.02降至0.01；
- `lambda_clmmd`从0.005降至0.003；
- `lambda_supcon`从0.01降至0.0075；
- 类别门控和专家保护延迟到第6轮，减少早期错误门控；
- 仍按连续时间前60%目标训练、后40%目标测试；
- 每轮使用目标测试集评估并保存最佳模型。

### 2.4 数据审计脚本

新增：

`diagnose_cwru1_data.py`

首先检查CWRU_1/ball_21是否确实为`223.mat`和`X223_DE_time`，并检测：

- 目录和文件编号；
- MATLAB驱动端变量；
- 信号长度；
- 重复文件；
- RMS、标准差、峰度、主频bin；
- 四个工况三个滚动体故障之间是否存在明显异常。

如果数据文件本身放错，模型调参无法解决。

## 3. 运行步骤

### 第一步：审计数据

```bash
cd "/workspace/故障诊断迁移学习/CWRU数据集迁移学习简化损失函数版本/MFSAN_BiMamba_CWRU1_fix_complete"

python diagnose_cwru1_data.py \
  --data_dir "/workspace/CWRU_TL/CWRU" \
  --signal_size 1024 \
  --output cwru1_ball21_audit.csv
```

应重点确认：

- `condition_1/ball_21/223.mat`存在；
- 内部变量是`X223_DE_time`；
- 没有与其他类别完全重复的SHA256；
- 长度和窗口数没有异常。

### 第二步：单随机种子运行

```bash
python run_cwru1_fix.py \
  --data_dir "/workspace/CWRU_TL/CWRU" \
  --seeds 2027 \
  --cuda_device 0 \
  --max_epoch 20 \
  --batch_size 64 \
  --signal_size 1024 \
  --target_test_size 0.40 \
  --eval_each_epoch True \
  --select_best_on_target True \
  --best_metric class_aware
```

### 第三步：正式三随机种子实验

```bash
nohup python run_cwru1_fix.py \
  --data_dir "/workspace/CWRU_TL/CWRU" \
  --seeds 2027,2028,2029 \
  --cuda_device 0 \
  --max_epoch 20 \
  --batch_size 64 \
  --signal_size 1024 \
  --target_test_size 0.40 \
  --eval_each_epoch True \
  --select_best_on_target True \
  --best_metric class_aware \
  > cwru1_v11_fix.log 2>&1 &
```

查看日志：

```bash
tail -f cwru1_v11_fix.log
```

## 4. 日志中需要关注的内容

```text
V11 rescue CLMMD epoch stats class-2:
```

- `candidates=0`持续出现：类别2甚至没有进入Top-2，应降低`v11_rescue_min_class_prob`或先检查数据；
- `candidates>0`但`accepted=0`：源原型与目标类别2不一致，应检查数据或适当放宽原型阈值；
- `accepted>0`：类别2已经开始参与CLMMD对齐；
- `valid_batches>0`：该轮确实产生了类别2对齐损失。

```text
V11 class-rescue diagnostics:
```

用于观察测试时有多少样本满足保守校准条件，以及实际改变了多少预测。

最佳模型日志将变成：

```text
Best model updated at epoch ..., target-test-acc ..., class_aware score ...
```

## 5. 建议的判断标准

不能只看Accuracy。至少同时比较：

- Accuracy；
- Macro-F1；
- ball_21 Recall；
- ball_21 F1；
- normal Recall；
- 三个随机种子的Mean ± Standard Deviation。

建议修复成功的最低目标：

- ball_21 Recall明显高于原来的2.13%；
- Macro-F1高于原来的85.76%；
- Accuracy不出现明显下降；
- normal Recall仍保持较高水平。

## 6. 验证情况

已完成：

- 全部修改文件语法检查；
- V10原有机制烟雾测试；
- V11测试校准烟雾测试；
- 使用四工况十类别合成MAT数据完成一次端到端CPU训练、测试和最佳模型保存。

未完成：

- 容器中没有你的真实`/workspace/CWRU_TL/CWRU`数据，因此无法保证真实Accuracy一定提高。真实效果必须通过你的数据重新运行，并根据V11候选/接受日志继续调整阈值。

## 7. 根据V11日志进行第二轮调参

专用运行脚本已经开放关键参数，不需要再修改Python文件。

若连续多轮出现：

```text
V11 rescue CLMMD epoch stats class-2: candidates=0
```

说明类别2连Top-2都很难进入，可以先试：

```bash
python run_cwru1_fix.py \
  --data_dir "/workspace/CWRU_TL/CWRU" \
  --seeds 2027 \
  --cuda_device 0 \
  --rescue_min_prob 0.05 \
  --eval_min_prob 0.05 \
  --best_metric class_aware
```

若`candidates>0`但`accepted=0`，先不要继续降低概率阈值，应优先检查数据审计结果；确认数据无误后，可小幅放宽原型条件（需要在`run_cwru1_fix.py`中调整`v11_rescue_proto_margin`或`v11_rescue_min_similarity`）。

若ball_21 Recall提高但Precision明显下降，说明救援过强，可提高：

```bash
--rescue_min_prob 0.15 \
--eval_min_prob 0.12 \
--eval_competitor_ratio 0.50 \
--eval_boost 1.50
```

若第5轮后仍出现大幅下降，可进一步降低适配强度：

```bash
--lambda_adv 0.005 \
--lambda_clmmd 0.002 \
--lambda_supcon 0.005
```

每次只改一组参数，并保留基线日志，避免无法判断是哪一项产生作用。
