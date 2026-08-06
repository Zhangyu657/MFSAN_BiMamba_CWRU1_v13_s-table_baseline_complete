# -*- coding: utf-8 -*-
"""CWRU-specific configuration helpers for the multi-source model."""

import logging
import os

CWRU_CLASSES = [
    "ball_07", "ball_14", "ball_21",
    "inner_07", "inner_14", "inner_21",
    "normal",
    "outer_07", "outer_14", "outer_21",
]


def is_cwru_experiment(args):
    names = list(getattr(args, "source_name", [])) + [getattr(args, "target", "")]
    detected = bool(names) and all(str(name).split("_")[0].upper() == "CWRU" for name in names)
    profile = str(getattr(args, "dataset_profile", "auto")).lower()
    return profile == "cwru" or (profile == "auto" and detected)


def apply_before_class_discovery(args):
    """Apply settings that must be known before folder/class discovery."""
    if not is_cwru_experiment(args):
        return args

    os.environ["CWRU_CHANNEL"] = str(getattr(args, "cwru_channel", "DE")).upper()
    if not str(getattr(args, "include_faults", "")).strip():
        args.include_faults = ",".join(CWRU_CLASSES)

    if bool(getattr(args, "cwru_apply_general_defaults", True)):
        # CWRU source files differ in length, so equalize labeled source windows.
        args.source_balance_data = True
        # Use all classes in ordinary SupCon.  The PU-specific hard pairs are not
        # assumed to transfer to CWRU before a CWRU confusion analysis exists.
        args.supcon_focus_classes = "all"
        args.v7_class_gate_log_classes = "all"
        args.v8_hard_supcon_enabled = False
        args.v8_hard_negative_pairs = ""
        args.v8_supcon_anchor_classes = "all"
        args.v8_prototype_filter_enabled = False
        args.v8_prototype_conf_overrides = ""
        args.v8_clmmd_class_boost = ""
        args.v8_prototype_log_classes = "all"
        args.v9_prototype_filter_classes = "all"
        args.v10_hard_pair_weights = ""
        args.v10_radius_class_min = ""
        args.v10_radius_class_max = ""

    return args


def apply_after_label_mapping(args):
    """Infer semantic class indices after alphabetical label mapping."""
    if not is_cwru_experiment(args):
        return args

    missing = [name for name in CWRU_CLASSES if name not in args.fault_label]
    if missing:
        raise ValueError(
            "CWRU class folders are incomplete. Missing after filtering: " + ", ".join(missing)
        )

    args.v10_normal_class = int(args.fault_label["normal"])
    args.class_names = [
        name for name, _ in sorted(args.fault_label.items(), key=lambda item: item[1])
    ]
    logging.info("CWRU profile active; channel=%s", os.environ.get("CWRU_CHANNEL", "DE"))
    logging.info("CWRU class mapping: %s", args.fault_label)
    logging.info("CWRU normal class index for V10 guard: %d", args.v10_normal_class)
    logging.info(
        "CWRU general defaults: source_balance=%s hard_supcon=%s prototype_filter=%s",
        args.source_balance_data,
        args.v8_hard_supcon_enabled,
        args.v8_prototype_filter_enabled,
    )
    return args
