# MFSAN-CDAN-BiMamba-DWF Patch

包含 3 个文件：

```text
models/MFSAN_CDAN_BIMAMBA_DWF.py
train_utils.py
opt.py
```

## 放置位置

复制到项目根目录：

```text
/workspace/故障诊断迁移学习/github迁移学习/TL-Fault-Diagnosis-Library_9labels_bimamba
```

其中：

```text
models/MFSAN_CDAN_BIMAMBA_DWF.py -> 项目/models/MFSAN_CDAN_BIMAMBA_DWF.py
train_utils.py -> 项目/train_utils.py
opt.py -> 项目/opt.py
```

## 推荐运行命令

```bash
python train.py \
  --model_name MFSAN_CDAN_BIMAMBA_DWF \
  --source PU_0,PU_1,PU_2 \
  --target PU_3 \
  --train_mode multi_source \
  --data_dir /workspace/PU_TL \
  --signal_size 1024 \
  --backbone CNN \
  --cuda_device 0 \
  --max_epoch 15 \
  --lambda_adv 0.01 \
  --lambda_grl 0.5 \
  --lambda_cda 0.02 \
  --lambda_ent 0.005 \
  --adv_detach_prob True \
  --adv_use_entropy_weight True \
  --adv_conf_thresh 0.8 \
  --dwf_tau 0.5 \
  --dwf_detach_weights True \
  --include_faults K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18




  python train.py \
  --model_name MFSAN_CDAN_BIMAMBA_DWF \
  --source PU_0,PU_1,PU_2 \
  --target PU_3 \
  --train_mode multi_source \
  --data_dir /workspace/PU_TL \
  --signal_size 1024 \
  --backbone CNN \
  --cuda_device 0 \
  --max_epoch 10 \
  --lambda_adv 0.01 \
  --lambda_grl 0.5 \
  --lambda_cda 0.02 \
  --lambda_ent 0.005 \
  --adv_detach_prob True \
  --adv_use_entropy_weight True \
  --adv_conf_thresh 0.8 \
  --dwf_tau 0.5 \
  --dwf_detach_weights True \
  --include_faults K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18
```

## 预期新增日志

```text
Using model: MFSAN_CDAN_BIMAMBA_DWF
MFSAN-CDAN-BiMamba-DWF DWF tau: 0.500000
DWF average source weights: src0=..., src1=..., src2=...
Val-DWF mean source weights: src0=..., src1=..., src2=...
Best model updated and saved to ..._best.pth
```
