import os
import re
import shutil
from collections import defaultdict
from scipy.io import loadmat

RAW_ROOT = "/workspace/故障诊断使用一维输入/PU原始信号数据集"
OUT_ROOT = "/workspace/PU_TL_9/PU"

# 建议第一版先用这 4 个标准工况
CONDITIONS = {
    "condition_0": "N09_M07_F10",
    "condition_1": "N15_M07_F10",
    "condition_2": "N15_M01_F10",
    "condition_3": "N15_M07_F04",
}

# 9-class PU setting: remove KA15, KB23, KI16, KI21
CLASSES = ["K001", "KA04", "KA16", "KA30", "KB24", "KB27", "KI04", "KI17", "KI18"]

USE_SYMLINK = True


def is_mat_file(name):
    return name.lower().endswith(".mat")


def get_all_classes(raw_root):
    classes = []
    for name in sorted(os.listdir(raw_root)):
        p = os.path.join(raw_root, name)
        if os.path.isdir(p) and not name.startswith("__") and not name.startswith("."):
            classes.append(name)
    return classes


def get_condition_from_filename(filename):
    """
    从文件名中提取工况前缀，例如：
    N09_M07_F10_K001_1.mat -> N09_M07_F10
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"(N\d+_M\d+_F\d+)_", base)
    if m:
        return m.group(1)
    return None


def check_pu_mat_readable(path):
    """
    检查这个 mat 文件是否能被当前 TL 库的 load_methods.PU 读取。
    该库默认要求：变量名 = 文件名去掉 .mat
    """
    base = os.path.splitext(os.path.basename(path))[0]
    try:
        data = loadmat(path)
    except Exception as e:
        return False, f"loadmat failed: {e}"

    if base not in data:
        valid_keys = [k for k in data.keys() if not k.startswith("__")]
        return False, f"variable {base} not found, valid keys={valid_keys[:5]}"

    try:
        fl = data[base]
        signal = fl[0][0][2][0][6][2]
        if signal is None:
            return False, "signal is None"
    except Exception as e:
        return False, f"PU nested signal read failed: {e}"

    return True, "ok"


def link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)

    if USE_SYMLINK:
        try:
            os.symlink(src, dst)
            return
        except Exception:
            pass

    shutil.copy2(src, dst)


def main():
    classes = CLASSES if CLASSES is not None else get_all_classes(RAW_ROOT)

    print("RAW_ROOT:", RAW_ROOT)
    print("OUT_ROOT:", OUT_ROOT)
    print("Classes:", classes)
    print("Conditions:", CONDITIONS)

    # 统计每个类别、每个工况下的文件
    files_by_class_condition = defaultdict(lambda: defaultdict(list))

    for cls in classes:
        cls_dir = os.path.join(RAW_ROOT, cls)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] class dir not found: {cls_dir}")
            continue

        for root, _, files in os.walk(cls_dir):
            for f in files:
                if not is_mat_file(f):
                    continue

                cond = get_condition_from_filename(f)
                if cond is None:
                    continue

                path = os.path.join(root, f)
                files_by_class_condition[cls][cond].append(path)

    # 打印统计
    print("\n========== Raw file statistics ==========")
    for cls in classes:
        print(f"\nClass {cls}:")
        for cond_name, cond_prefix in CONDITIONS.items():
            n = len(files_by_class_condition[cls][cond_prefix])
            print(f"  {cond_name} / {cond_prefix}: {n} files")

    # 找出四个工况都存在数据的类别
    valid_classes = []
    invalid_classes = []

    for cls in classes:
        ok = True
        for _, cond_prefix in CONDITIONS.items():
            if len(files_by_class_condition[cls][cond_prefix]) == 0:
                ok = False
                break
        if ok:
            valid_classes.append(cls)
        else:
            invalid_classes.append(cls)

    print("\n========== Valid classes ==========")
    print(valid_classes)

    if invalid_classes:
        print("\n========== Invalid classes skipped ==========")
        print(invalid_classes)

    # 生成目标目录
    print("\n========== Building PU_TL dataset ==========")
    total = 0
    unreadable = []

    for cond_name, cond_prefix in CONDITIONS.items():
        for cls in valid_classes:
            src_list = sorted(files_by_class_condition[cls][cond_prefix])

            for src in src_list:
                ok, msg = check_pu_mat_readable(src)
                if not ok:
                    unreadable.append((src, msg))
                    continue

                dst = os.path.join(OUT_ROOT, cond_name, cls, os.path.basename(src))
                link_or_copy(src, dst)
                total += 1

    print(f"\nDone. Linked/copied {total} files to {OUT_ROOT}")

    if unreadable:
        print("\n========== Unreadable files ==========")
        for p, msg in unreadable[:50]:
            print(p, "=>", msg)
        print(f"Total unreadable: {len(unreadable)}")
    else:
        print("All linked/copied files are readable by current PU loader.")

    print("\nNext data_dir should be:")
    print("/workspace/PU_TL_9")


if __name__ == "__main__":
    main()