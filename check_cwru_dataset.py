#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the prepared CWRU_TL/CWRU directory before training."""

import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from cwru_profile import CWRU_CLASSES


def resolve_root(value):
    path = Path(value).expanduser().resolve()
    nested = path / "CWRU"
    return nested if nested.is_dir() else path


def load_channel(path, channel):
    mat = loadmat(path)
    suffix = f"_{channel}_time".lower()
    keys = sorted(k for k in mat if not k.startswith("__") and k.lower().endswith(suffix))
    if not keys:
        raise KeyError(f"No *_{channel}_time variable; keys={[k for k in mat if not k.startswith('__')]}")
    signal = np.asarray(mat[keys[0]]).reshape(-1)
    return keys[0], signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="CWRU_TL or CWRU directory")
    parser.add_argument("--signal_size", type=int, default=1024)
    parser.add_argument("--channel", choices=["DE", "FE", "BA"], default="DE")
    args = parser.parse_args()

    root = resolve_root(args.data_dir)
    errors = []
    total_windows = Counter()
    print(f"Resolved CWRU root: {root}")

    for condition in range(4):
        for class_name in CWRU_CLASSES:
            folder = root / f"condition_{condition}" / class_name
            mats = sorted(folder.glob("*.mat")) if folder.is_dir() else []
            if not mats:
                errors.append(f"Missing .mat files: {folder}")
                continue
            for mat_path in mats:
                try:
                    key, signal = load_channel(mat_path, args.channel)
                    if not np.isfinite(signal).all():
                        raise ValueError("contains NaN/Inf")
                    windows = len(signal) // args.signal_size
                    if windows < 2:
                        raise ValueError(f"only {windows} complete windows")
                    total_windows[(condition, class_name)] += windows
                    print(
                        f"OK condition_{condition}/{class_name}/{mat_path.name}: "
                        f"key={key}, length={len(signal)}, windows={windows}"
                    )
                except Exception as exc:
                    errors.append(f"{mat_path}: {exc}")

    print("\nWindow summary before target split:")
    for condition in range(4):
        values = [total_windows[(condition, c)] for c in CWRU_CLASSES]
        print(f"condition_{condition}: min={min(values or [0])}, max={max(values or [0])}, {dict(zip(CWRU_CLASSES, values))}")

    if errors:
        print("\nFAILED:")
        for item in errors:
            print(" -", item)
        raise SystemExit(1)
    print("\nCWRU dataset check passed.")


if __name__ == "__main__":
    main()
