import os
import glob
import shutil
from scipy.io import loadmat

RAW_ROOT = "/workspace/CWRU"
OUT_ROOT = "/workspace/CWRU_TL/CWRU"

# 标准 10 类，使用 12k Drive End + Normal
# 外圈这里选 6 点方向的一组文件编号
MAPPING = {
    "condition_0": {
        "normal": "97",
        "inner_07": "105",
        "inner_14": "169",
        "inner_21": "209",
        "ball_07": "118",
        "ball_14": "185",
        "ball_21": "222",
        "outer_07": "130",
        "outer_14": "197",
        "outer_21": "234",
    },
    "condition_1": {
        "normal": "98",
        "inner_07": "106",
        "inner_14": "170",
        "inner_21": "210",
        "ball_07": "119",
        "ball_14": "186",
        "ball_21": "223",
        "outer_07": "131",
        "outer_14": "198",
        "outer_21": "235",
    },
    "condition_2": {
        "normal": "99",
        "inner_07": "107",
        "inner_14": "171",
        "inner_21": "211",
        "ball_07": "120",
        "ball_14": "187",
        "ball_21": "224",
        "outer_07": "132",
        "outer_14": "199",
        "outer_21": "236",
    },
    "condition_3": {
        "normal": "100",
        "inner_07": "108",
        "inner_14": "172",
        "inner_21": "212",
        "ball_07": "121",
        "ball_14": "188",
        "ball_21": "225",
        "outer_07": "133",
        "outer_14": "200",
        "outer_21": "237",
    },
}


def mat_file_id(path):
    """
    从 mat 文件内部变量名中识别 CWRU 编号。
    例如 X097_DE_time -> 97，X105_DE_time -> 105。
    这样即使原始文件名不是 105.mat，也能找到对应编号。
    """
    try:
        data = loadmat(path)
    except Exception:
        return None

    for key in data.keys():
        if "DE_time" in key:
            # X097_DE_time / X105_DE_time
            num = key.split("_DE_time")[0].replace("X", "")
            return str(int(num))
    return None


def build_index(raw_root):
    index = {}
    mat_files = glob.glob(os.path.join(raw_root, "**", "*.mat"), recursive=True)

    for path in mat_files:
        fid = mat_file_id(path)
        if fid is not None:
            index[fid] = path

    return index


def safe_link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.exists(dst):
        os.remove(dst)

    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def main():
    index = build_index(RAW_ROOT)

    print(f"Found {len(index)} CWRU mat files with DE_time keys.")

    missing = []

    for condition, class_map in MAPPING.items():
        for class_name, file_id in class_map.items():
            if file_id not in index:
                missing.append((condition, class_name, file_id))
                continue

            src = index[file_id]
            dst = os.path.join(OUT_ROOT, condition, class_name, f"{file_id}.mat")
            safe_link_or_copy(src, dst)
            print(f"{condition}/{class_name}: {file_id}.mat <- {src}")

    if missing:
        print("\nMissing files:")
        for item in missing:
            print(item)
    else:
        print("\nAll files prepared successfully.")

    print(f"\nOutput dataset root: {OUT_ROOT}")


if __name__ == "__main__":
    main()