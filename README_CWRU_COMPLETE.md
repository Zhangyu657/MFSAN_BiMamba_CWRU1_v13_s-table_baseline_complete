# CWRU complete adaptation version

This project is the uploaded multi-source BiMamba/RWCA model modified for the
folder structure:

```
CWRU_TL/CWRU/
  condition_0/{ball_07,...,outer_21}/*.mat
  condition_1/{ball_07,...,outer_21}/*.mat
  condition_2/{ball_07,...,outer_21}/*.mat
  condition_3/{ball_07,...,outer_21}/*.mat
```

## What was changed

1. Robust CWRU `.mat` reader: variable discovery no longer depends on filenames.
2. `--data_dir` accepts either the parent `CWRU_TL` or the `CWRU` directory itself.
3. Automatic 10-class mapping and automatic V10 normal-class index (`normal` = 6
   under alphabetical ordering).
4. Source-domain class balancing is enabled by the CWRU profile.
5. PU-specific hard-negative pairs/prototype calibration are disabled until a
   CWRU confusion matrix supports new class-pair choices.
6. Strict evaluation mode can test the target set only once after training.

## 1. Check the dataset

Linux/macOS:

```bash
python check_cwru_dataset.py --data_dir /workspace/CWRU_TL --signal_size 1024
```

Windows example:

```powershell
python check_cwru_dataset.py --data_dir D:\CWRU_TL --signal_size 1024
```

## 2. Run one task

Target condition 3, sources 0/1/2:

```bash
python run_cwru_v10.py --data_dir /workspace/CWRU_TL --targets 3 --seeds 2027
```

Windows:

```powershell
python run_cwru_v10.py --data_dir D:\CWRU_TL --targets 3 --seeds 2027
```

## 3. Run four tasks and three seeds

```bash
python run_cwru_v10.py   --data_dir /workspace/CWRU_TL   --targets 0,1,2,3   --seeds 2027,2028,2029
```

By default the launcher uses strict reporting:

- `--eval_each_epoch False`
- `--select_best_on_target False`

The target held-out set is evaluated once after the final epoch.  To reproduce
the old protocol, explicitly pass both options as `True`, but that uses target
labels for checkpoint selection and is not recommended for final thesis results.

## Class indices

| id | folder |
|---:|---|
| 0 | ball_07 |
| 1 | ball_14 |
| 2 | ball_21 |
| 3 | inner_07 |
| 4 | inner_14 |
| 5 | inner_21 |
| 6 | normal |
| 7 | outer_07 |
| 8 | outer_14 |
| 9 | outer_21 |

The model automatically changes from 9 to 10 outputs. CDAN joint features
change from `9 x 40 = 360` to `10 x 40 = 400` without manual layer edits.
