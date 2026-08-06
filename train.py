"""
新9类轴承后的训练指令

python train.py \
  --model_name MFSAN_CDAN_BIMAMBA_DWF \
  --source PU_0,PU_1,PU_2 \
  --target PU_3 \
  --train_mode multi_source \
  --data_dir /workspace/PU_TL_9_replace \
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
  --include_faults K001,KA04,KA16,KA30,KB24,KB23,KI04,KI17,KI16


"""


import os
import sys
sys.path.extend(['./models', './data_loader'])
import torch
import logging
import importlib
import random
import numpy as np
from datetime import datetime

import utils
from opt import parse_args
from cwru_profile import apply_before_class_discovery, apply_after_label_mapping


def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
        
    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    return logger


def creat_file(args):
    """
    Create save path and logger.

    New:
    1. Add random seed to save folder.
    2. Add random seed to log/checkpoint file name.
    3. Avoid overwriting results when running different random seeds.
    """

    # seed tag
    if getattr(args, 'random_state', None) is not None:
        seed_tag = f"seed{args.random_state}"
    else:
        seed_tag = "seedNone"

    # source-target name
    source_tag = '_'.join(args.source_name)
    target_tag = args.target

    # timestamp
    time_tag = datetime.strftime(datetime.now(), '%m%d-%H%M%S')

    # file name, e.g.
    # [PU_0_PU_1_PU_2]To[PU_3]_seed42_0513-103011
    file_name = f"[{source_tag}]To[{target_tag}]_{seed_tag}_{time_tag}"

    # save directory, e.g.
    # ./checkpoint/MFSAN_CDAN_BIMAMBA_CW_RWCA_V4_SUPCON/multi_source/seed42
    save_dir = os.path.join(
        args.save_dir,
        args.model_name,
        args.train_mode,
        seed_tag
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # args.save_path will be used by logger and model checkpoint
    args.save_path = os.path.join(save_dir, file_name)

    # set logger
    logger = setlogger(args.save_path + '.log')

    # save args
    for k, v in args.__dict__.items():
        if k != 'source_name':
            logging.info("{}: {}".format(k, v))

    return logger, args


def _parse_fault_name_list(text):
    """
    Parse comma-separated class names from command line.

    Examples:
        --include_faults K001,KA04,KA16
        --exclude_faults KA15,KB23,KI16,KI21
    """
    if text is None:
        return []
    return [x.strip() for x in str(text).split(',') if x.strip()]


def _filter_faults_by_name(faults, args, domain_name):
    """
    Filter detected fault folders by explicit class names.

    Notes:
    1. This happens after the original name-index selection, so it is compatible
       with the original PU_0-012 style usage.
    2. The same include/exclude rule is applied to all source and target domains,
       which keeps the task as closed-set domain adaptation when every domain has
       the same selected classes.
    """
    faults = list(faults)
    include_faults = _parse_fault_name_list(getattr(args, 'include_faults', ''))
    exclude_faults = _parse_fault_name_list(getattr(args, 'exclude_faults', ''))

    if include_faults:
        missing = [x for x in include_faults if x not in faults]
        if missing:
            logging.warning('%s missing include_faults ignored: %s', domain_name, missing)
        include_set = set(include_faults)
        faults = [x for x in faults if x in include_set]

    if exclude_faults:
        missing = [x for x in exclude_faults if x not in faults]
        if missing:
            logging.info('%s exclude_faults not present or already removed: %s', domain_name, missing)
        exclude_set = set(exclude_faults)
        faults = [x for x in faults if x not in exclude_set]

    if len(faults) == 0:
        raise ValueError(
            f'No classes left for {domain_name}. '
            f'include_faults={include_faults}, exclude_faults={exclude_faults}'
        )

    return np.array(sorted(faults))


def get_fault(name, args):
    dataset, condition, selected_list = utils.get_info_from_name(name)
    if condition is not None:
        data_root = utils.resolve_dataset_root(args.data_dir, dataset)
        faults = np.array(sorted(os.listdir(os.path.join(data_root, 'condition_%d' % condition))))
    else:
        data_root = utils.resolve_dataset_root(args.data_dir, dataset)
        faults = np.array(sorted(os.listdir(data_root)))
    if selected_list:
        faults = faults[selected_list]

    # New: filter by class names, e.g. remove KA15/KB23/KI16/KI21 to make 9-class PU.
    faults = _filter_faults_by_name(faults, args, name)

    num_classes = len(faults)
    return faults, num_classes


def determine_da_scenario(label_sets):
    # Extract source and target labels
    source_labels = label_sets[:-1]
    target_labels = label_sets[-1]

    # Flatten the source labels and convert to a set to get unique labels
    source_labels_flat = set([label for sublist in source_labels for label in sublist])
    target_labels_set = set(target_labels)

    # Check conditions for different domain adaptation scenarios
    if source_labels_flat == target_labels_set:
        return 'closed-set'
    elif target_labels_set.issubset(source_labels_flat):
        return 'partial'
    elif source_labels_flat.issubset(target_labels_set):
        return 'open-set'
    else:
        return 'universal'


if __name__ == '__main__':
    os.environ['NUMEXPR_MAX_THREADS'] = '8'
    args = parse_args()
    if args.random_state is not None:
        random.seed(args.random_state)
        np.random.seed(args.random_state)
        torch.manual_seed(args.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.random_state)
            torch.cuda.manual_seed_all(args.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    args.source_name = [x.strip() for x in list(args.source.split(','))]
    if '' in args.source_name:
        args.source_name.remove('')

    # Apply dataset-specific defaults before class folder discovery.
    args = apply_before_class_discovery(args)

    if not args.load_path:
        if len(args.source_name) == 1:
            args.train_mode = 'single_source'
        if args.train_mode == 'single_source':
            assert len(args.source_name) == 1, "single_source mode needs one source"
        else:
            assert len(args.source_name) > 1, "source_combine and multi_source mode need more than one source"
    
    # creating directory
    logger, args = creat_file(args)
    
    # getting faults dictionary
    args.faults, args.num_classes = [], []
    for name in args.source_name + [args.target]:
        faults, num_classes = get_fault(name, args)
        args.faults.append(faults)
        args.num_classes.append(num_classes)
    for name, faults, nclasses in zip(args.source_name, args.faults[:-1], args.num_classes[:-1]):
        logging.info('Source {} detected {} classes: {}'.format(name, nclasses, faults))
    logging.info('Target {} detected {} classes: {}'.format(args.target, args.num_classes[-1], args.faults[-1]))
    
    # getting mapping of fault to label
    all_faults = set()
    for faults in args.faults:
        for item in faults:
            all_faults.add(item)
    args.fault_label = {}
    for i, fault in enumerate(sorted(all_faults)):
        args.fault_label[fault] = i
    # Semantic indices such as the CWRU normal class are only known now.
    args = apply_after_label_mapping(args)
    if args.train_mode == 'source_combine':
        source_faults_flat = sorted(list(set([fault for sublist in args.faults[:-1] for fault in sublist])))
        args.faults.insert(0, source_faults_flat)
        args.num_classes.insert(0, len(source_faults_flat))

    # getting sets of labels
    args.label_sets = list()
    for faults in args.faults:
        args.label_sets.append([args.fault_label[item] for item in faults])
    
    # determine current DA scenario
    args.da_scenario = determine_da_scenario(args.label_sets)
    logging.info('The scenario is: {} domain adaptation'.format(args.da_scenario))

    # training
    trainer = importlib.import_module(f"models.{args.model_name}").Trainer(args)
    if args.load_path:
        trainer.load_model()
        trainer.test()
        os.remove(args.save_path + '.log')
    else:
        trainer.train()
        if args.save:
            trainer.save_model()
        else:
            os.remove(args.save_path + '.log')
    logger.handlers.clear()
