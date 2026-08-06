import os
from collections import defaultdict

import numpy as np
import pandas as pd

import aug
import data_utils
import load_methods
import utils


def _iter_class_files(root, condition, class_name):
    """
    Return sorted .mat files for one class under one condition directory.

    Expected directory format:
        data_dir / dataset / condition_x / class_name / *.mat
    """
    data_dir = os.path.join(root, 'condition_%d' % condition, class_name)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Class directory not found: {data_dir}")

    files = []
    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if os.path.isfile(item_path) and item.lower().endswith('.mat'):
            files.append(item_path)
    return files


def _window_signal(signal, signal_size):
    """
    Non-overlapping sliding window, consistent with the original code.
    The original implementation used start += signal_size.
    """
    data = []
    start, end = 0, signal_size
    while end <= signal.shape[0]:
        data.append(signal[start:end])
        start += signal_size
        end += signal_size
    return data


def _load_windows_from_files(file_list, data_load, signal_size, label):
    """
    Load full signals from files and cut them into non-overlapping windows.
    """
    data, actual_labels = [], []
    for item_path in file_list:
        signal = data_load(item_path)
        windows = _window_signal(signal, signal_size)
        if not windows:
            raise ValueError(
                f"Signal shorter than one window: file={item_path}, "
                f"length={signal.shape[0]}, signal_size={signal_size}"
            )
        data.extend(windows)
        actual_labels.extend([label] * len(windows))
    return data, actual_labels


def _split_files_by_class(files_by_class, test_size=0.4, random_state=10):
    """
    Split original .mat files per class before windowing.

    This avoids the old behavior where all windows were generated first and then
    randomly split into train/test. If each class has only one file, that file is
    put into train and test will be empty for that class; in that case use
    target_split_mode=time instead.
    """
    rng = np.random.default_rng(random_state)
    train_files_by_class = {}
    test_files_by_class = {}

    for cls_name, file_list in files_by_class.items():
        file_list = list(sorted(file_list))
        n = len(file_list)
        indices = np.arange(n)
        rng.shuffle(indices)

        if n <= 1:
            train_idx = indices
            test_idx = np.array([], dtype=int)
        else:
            n_test = int(round(test_size * n))
            n_test = max(1, min(n - 1, n_test))
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]

        train_files_by_class[cls_name] = [file_list[i] for i in sorted(train_idx.tolist())]
        test_files_by_class[cls_name] = [file_list[i] for i in sorted(test_idx.tolist())]

    return train_files_by_class, test_files_by_class


def get_files(root, dataset, faults, fault_label, signal_size, condition=3):
    """
    Original source-domain loading behavior:
    read all files under the selected condition and cut each file into windows.
    """
    data, actual_labels = [], []
    data_load = getattr(load_methods, dataset)

    for _, name in enumerate(faults):
        file_list = _iter_class_files(root, condition, name)
        label = fault_label[name]
        cls_data, cls_labels = _load_windows_from_files(file_list, data_load, signal_size, label)
        data.extend(cls_data)
        actual_labels.extend(cls_labels)

    return data, actual_labels


def get_target_files_split_by_file(root, dataset, faults, fault_label, signal_size,
                                   condition=3, test_size=0.4, random_state=10):
    """
    Target split mode 1: split by original .mat file per class before windowing.

    Target train set is used as unlabeled target-domain training data;
    target test set is used for evaluation and save_best selection, following
    the user's requested setting.
    """
    data_load = getattr(load_methods, dataset)

    files_by_class = {}
    for name in faults:
        files_by_class[name] = _iter_class_files(root, condition, name)

    train_files_by_class, test_files_by_class = _split_files_by_class(
        files_by_class,
        test_size=test_size,
        random_state=random_state
    )

    train_data, train_labels = [], []
    test_data, test_labels = [], []

    for name in faults:
        label = fault_label[name]

        cls_train_data, cls_train_labels = _load_windows_from_files(
            train_files_by_class[name], data_load, signal_size, label
        )
        train_data.extend(cls_train_data)
        train_labels.extend(cls_train_labels)

        cls_test_data, cls_test_labels = _load_windows_from_files(
            test_files_by_class[name], data_load, signal_size, label
        )
        test_data.extend(cls_test_data)
        test_labels.extend(cls_test_labels)

    return train_data, train_labels, test_data, test_labels


def get_target_files_split_by_time(root, dataset, faults, fault_label, signal_size,
                                   condition=3, test_size=0.4):
    """
    Target split mode 2: split each original .mat signal into continuous segments
    before windowing.

    Example with test_size=0.4:
        first 60% of each original signal  -> unlabeled target train
        last  40% of each original signal  -> target test

    This is the recommended mode when each class/condition may contain only a
    small number of .mat files. It avoids random mixing of adjacent windows.
    """
    data_load = getattr(load_methods, dataset)
    train_ratio = 1.0 - float(test_size)

    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError(f"target_test_size must be between 0 and 1, got {test_size}")

    train_data, train_labels = [], []
    test_data, test_labels = [], []

    for name in faults:
        file_list = _iter_class_files(root, condition, name)
        label = fault_label[name]

        for item_path in file_list:
            signal = data_load(item_path)
            n = signal.shape[0]
            split_point = int(n * train_ratio)

            train_signal = signal[:split_point]
            test_signal = signal[split_point:]

            train_windows = _window_signal(train_signal, signal_size)
            test_windows = _window_signal(test_signal, signal_size)
            if not train_windows or not test_windows:
                raise ValueError(
                    f"CWRU target time split produced an empty side for {item_path}: "
                    f"length={n}, split={split_point}, signal_size={signal_size}. "
                    "Reduce --signal_size or change --target_test_size."
                )

            train_data.extend(train_windows)
            train_labels.extend([label] * len(train_windows))

            test_data.extend(test_windows)
            test_labels.extend([label] * len(test_windows))

    return train_data, train_labels, test_data, test_labels


def data_transforms(normlize_type="-1-1"):
    transforms = {
        'train': aug.Compose([
            aug.Reshape(),
            aug.Normalize(normlize_type),
            aug.Retype()
        ]),
        'val': aug.Compose([
            aug.Reshape(),
            aug.Normalize(normlize_type),
            aug.Retype()
        ])
    }
    return transforms


class dataset(object):

    def __init__(self, args, dataset, source_idx, condition=2, balance_data=False, test_size=0.2):
        self.args = args
        data_root = utils.resolve_dataset_root(args.data_dir, dataset)
        faults = args.faults[source_idx]
        signal_size = args.signal_size
        normlize_type = args.normlize_type
        fault_label = args.fault_label
        self.label_set = args.label_sets[source_idx]
        self.random_state = args.random_state
        self.balance_data = balance_data

        # New target split settings. The original code used test_size=0.2 after
        # sliding windows. For the paper-recommended setup, default is:
        # target train 60%, target test 40%, split before windowing.
        self.test_size = float(getattr(args, 'target_test_size', test_size))
        self.target_split_mode = getattr(args, 'target_split_mode', 'time')

        self.data_root = data_root
        self.dataset = dataset
        self.faults = faults
        self.fault_label = fault_label
        self.signal_size = signal_size
        self.condition = condition
        self.transform = data_transforms(normlize_type)

        # Source-domain data are still loaded as before: all files and all windows.
        # Target-domain data are split in data_preprare(is_src=False).
        self.data, self.actual_labels = get_files(
            data_root,
            dataset,
            faults,
            fault_label,
            signal_size,
            condition=condition
        )

    def data_preprare(self, is_src=False):
        if is_src:
            data_pd = pd.DataFrame({"data": self.data, "actual_labels": self.actual_labels})
            data_pd = data_utils.balance_data(data_pd) if self.balance_data else data_pd
            train_dataset = data_utils.dataset(list_data=data_pd, transform=self.transform['train'])
            return train_dataset

        # Target domain: use one of the strict split modes before windowing.
        if self.target_split_mode == 'time':
            train_data, train_labels, test_data, test_labels = get_target_files_split_by_time(
                self.data_root,
                self.dataset,
                self.faults,
                self.fault_label,
                self.signal_size,
                condition=self.condition,
                test_size=self.test_size
            )
        elif self.target_split_mode == 'file':
            train_data, train_labels, test_data, test_labels = get_target_files_split_by_file(
                self.data_root,
                self.dataset,
                self.faults,
                self.fault_label,
                self.signal_size,
                condition=self.condition,
                test_size=self.test_size,
                random_state=self.random_state
            )
        elif self.target_split_mode == 'window_random':
            # Old behavior: generate all windows first, then randomly split windows.
            data_pd = pd.DataFrame({"data": self.data, "actual_labels": self.actual_labels})
            data_pd = data_utils.balance_data(data_pd) if self.balance_data else data_pd
            train_pd, test_pd = data_utils.train_test_split_(
                data_pd,
                test_size=self.test_size,
                label_set=self.label_set,
                random_state=self.random_state
            )
            train_dataset = data_utils.dataset(list_data=train_pd, transform=self.transform['train'])
            test_dataset = data_utils.dataset(list_data=test_pd, transform=self.transform['val'])
            return train_dataset, test_dataset
        else:
            raise ValueError(
                f"Unknown target_split_mode={self.target_split_mode}. "
                f"Use one of: time, file, window_random."
            )

        train_pd = pd.DataFrame({"data": train_data, "actual_labels": train_labels})
        test_pd = pd.DataFrame({"data": test_data, "actual_labels": test_labels})

        train_pd = data_utils.balance_data(train_pd) if self.balance_data else train_pd
        test_pd = data_utils.balance_data(test_pd) if self.balance_data else test_pd

        train_dataset = data_utils.dataset(list_data=train_pd, transform=self.transform['train'])
        test_dataset = data_utils.dataset(list_data=test_pd, transform=self.transform['val'])
        return train_dataset, test_dataset
