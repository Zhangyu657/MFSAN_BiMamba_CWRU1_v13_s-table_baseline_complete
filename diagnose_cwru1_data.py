#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the CWRU_1 ball_21 data before changing the model.

The script verifies file identity, MATLAB channel keys, signal length, duplicate
files and simple window statistics for ball_07/ball_14/ball_21 across all four
conditions.  It never uses labels for training; it is an offline data audit.
"""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from data_loader.load_methods import CWRU

BALL_CLASSES = ['ball_07', 'ball_14', 'ball_21']
EXPECTED_IDS = {
    0: {'ball_07': '118', 'ball_14': '185', 'ball_21': '222'},
    1: {'ball_07': '119', 'ball_14': '186', 'ball_21': '223'},
    2: {'ball_07': '120', 'ball_14': '187', 'ball_21': '224'},
    3: {'ball_07': '121', 'ball_14': '188', 'ball_21': '225'},
}


def resolve_root(value: str) -> Path:
    p = Path(value).expanduser().resolve()
    if (p / 'condition_0').is_dir():
        return p
    if (p / 'CWRU' / 'condition_0').is_dir():
        return p / 'CWRU'
    raise FileNotFoundError(
        f'Cannot find condition_0 under {p} or {p / "CWRU"}'
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def matlab_de_keys(path: Path):
    mat = loadmat(path)
    return sorted(
        key for key in mat.keys()
        if not key.startswith('__') and key.lower().endswith('_de_time')
    )


def window_features(signal: np.ndarray, window: int):
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = len(x) // window
    if n <= 0:
        raise ValueError(f'Signal length {len(x)} is shorter than window {window}')
    w = x[: n * window].reshape(n, window)
    mean = w.mean(axis=1)
    std = w.std(axis=1)
    rms = np.sqrt(np.mean(w * w, axis=1))
    centered = w - mean[:, None]
    fourth = np.mean(centered ** 4, axis=1)
    kurtosis = fourth / (std ** 4 + 1e-12)
    spectrum = np.abs(np.fft.rfft(w, axis=1))
    peak_bin = spectrum[:, 1:].argmax(axis=1) + 1
    return {
        'windows': n,
        'mean_abs_median': float(np.median(np.abs(mean))),
        'std_median': float(np.median(std)),
        'rms_median': float(np.median(rms)),
        'kurtosis_median': float(np.median(kurtosis)),
        'peak_bin_median': float(np.median(peak_bin)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--signal_size', type=int, default=1024)
    parser.add_argument('--output', default='cwru1_ball21_audit.csv')
    args = parser.parse_args()

    root = resolve_root(args.data_dir)
    rows = []
    hashes = {}
    errors = []

    for condition in range(4):
        for class_name in BALL_CLASSES:
            folder = root / f'condition_{condition}' / class_name
            files = sorted(folder.glob('*.mat'))
            if not files:
                errors.append(f'MISSING: {folder}')
                continue
            for path in files:
                digest = sha256(path)
                hashes.setdefault(digest, []).append(str(path))
                keys = matlab_de_keys(path)
                signal = CWRU(str(path))
                feats = window_features(signal, args.signal_size)
                expected = EXPECTED_IDS[condition][class_name]
                stem_ok = path.stem.lstrip('0') == expected.lstrip('0')
                row = {
                    'condition': condition,
                    'class_name': class_name,
                    'file': path.name,
                    'expected_id': expected,
                    'expected_id_match': stem_ok,
                    'de_keys': '|'.join(keys),
                    'signal_length': int(signal.shape[0]),
                    'sha256': digest,
                    **feats,
                }
                rows.append(row)

    output = Path(args.output).expanduser().resolve()
    if rows:
        with output.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f'Dataset root: {root}')
    print(f'Audit CSV: {output}')
    for row in rows:
        flag = 'OK' if row['expected_id_match'] else 'CHECK_ID'
        print(
            f"[{flag}] condition_{row['condition']}/{row['class_name']}/"
            f"{row['file']} len={row['signal_length']} windows={row['windows']} "
            f"rms={row['rms_median']:.6f} kurt={row['kurtosis_median']:.3f} "
            f"key={row['de_keys']}"
        )

    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    if duplicates:
        print('\nWARNING: byte-identical files found across dataset locations:')
        for group in duplicates:
            print('  ' + ' == '.join(group))
    else:
        print('\nNo byte-identical duplicates found among audited ball files.')

    if errors:
        print('\nERRORS:')
        for item in errors:
            print('  ' + item)
        raise SystemExit(2)

    target_rows = [
        r for r in rows
        if r['condition'] == 1 and r['class_name'] == 'ball_21'
    ]
    if not target_rows:
        raise SystemExit('CWRU_1 ball_21 was not found.')
    if not all(r['expected_id_match'] for r in target_rows):
        print('\nIMPORTANT: CWRU_1/ball_21 is not named as expected file 223.mat. '
              'Verify its internal X223_DE_time key and source mapping.')
    print('\nCWRU_1 ball_21 audit completed.')


if __name__ == '__main__':
    main()
