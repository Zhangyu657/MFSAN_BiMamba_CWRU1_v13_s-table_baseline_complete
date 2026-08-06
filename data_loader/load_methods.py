import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


def CWRU(item_path):
    """Load one CWRU MATLAB vibration file robustly.

    The original implementation inferred the MATLAB variable from a numeric
    filename such as ``105.mat``.  That fails when files are renamed.  This
    version inspects the variables in the file and selects a vibration channel
    directly.  The channel can be changed with the environment variable
    ``CWRU_CHANNEL`` (DE, FE or BA); DE is the default used by this project.

    Returns
    -------
    numpy.ndarray
        Float32 array with shape [signal_length, 1].
    """
    mat = loadmat(item_path)
    channel = os.environ.get("CWRU_CHANNEL", "DE").strip().upper()
    if channel not in {"DE", "FE", "BA"}:
        raise ValueError(
            f"Unsupported CWRU_CHANNEL={channel!r}; use DE, FE or BA."
        )

    suffix = f"_{channel}_time"
    candidates = sorted(
        key for key in mat.keys()
        if not key.startswith("__") and key.endswith(suffix)
    )

    # Some redistributed files use slightly different capitalization.  Use a
    # case-insensitive fallback while still requiring the requested channel.
    if not candidates:
        suffix_lower = suffix.lower()
        candidates = sorted(
            key for key in mat.keys()
            if not key.startswith("__") and key.lower().endswith(suffix_lower)
        )

    if not candidates:
        available = [key for key in mat.keys() if not key.startswith("__")]
        raise KeyError(
            f"No CWRU {channel}-end vibration variable (*{suffix}) found in "
            f"{item_path}. Available variables: {available}"
        )

    signal_key = candidates[0]
    signal = np.asarray(mat[signal_key], dtype=np.float32).reshape(-1, 1)
    if signal.size == 0:
        raise ValueError(f"Empty CWRU signal in {item_path}, key={signal_key}")
    if not np.isfinite(signal).all():
        raise ValueError(f"NaN/Inf found in CWRU signal {item_path}, key={signal_key}")
    return signal


def MFPT(item_path):
    f = item_path.split("/")[-2]
    if f == 'normal':
        signal = (loadmat(item_path)["bearing"][0][0][1])
    else:
        signal = (loadmat(item_path)["bearing"][0][0][2])

    return signal


def PU(item_path):
    name = os.path.basename(item_path).split(".")[0]
    fl = loadmat(item_path)[name]
    signal = fl[0][0][2][0][6][2]  #Take out the data
    signal = signal.reshape(-1,1)

    return signal


def XJTU(item_path):
    fl = pd.read_csv(item_path)
    signal = fl["Horizontal_vibration_signals"]
    signal = signal.values.reshape(-1,1)

    return signal


def IMS(item_path):
    channel = {'normal': 0,
               'inner': 4,
               'outer': 0,
               'ball': 6}
    f = item_path.split("/")[-2]
    signal = np.loadtxt(item_path)[:, channel[f]]

    return signal


def JNU(item_path):
    fl = pd.read_csv(item_path)
    signal = fl.values
    
    return signal
