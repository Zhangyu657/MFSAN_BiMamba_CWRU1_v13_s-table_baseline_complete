import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    if v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='From https://github.com/Feaxure-fresh/TL-Fault-Diagnosis-Library')
 
    # basic parameters
    parser.add_argument('--model_name', type=str, default='CNN',
                        help='Name of the model (in ./models directory)')
    parser.add_argument('--source', type=str, default='CWRU_0',
                        help='Source data, separated by "," (select specific conditions of the dataset with name_number, such as CWRU_0)')
    parser.add_argument('--target', type=str, default='CWRU_1',
                        help='Target data (select specific conditions of the dataset with name_number, such as CWRU_0)')
    parser.add_argument('--data_dir', type=str, default="./datasets",
                        help='Directory of the datasets')
    parser.add_argument('--train_mode', type=str, default='single_source',
                        choices=['single_source', 'source_combine', 'multi_source'],
                        help='Training mode (select correctly before training)')
    parser.add_argument('--cuda_device', type=str, default='0',
                        help='Allocate the device to use only one GPU (empty string means using cpu)')
    parser.add_argument('--max_epoch', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--signal_size', type=int, default=1024,
                        help='Signal length split by sliding window')
    parser.add_argument('--random_state', type=int, default=10,
                        help='Random state for the entire training')

    # dataset/profile parameters
    parser.add_argument('--dataset_profile', type=str, default='auto',
                        choices=['auto', 'generic', 'cwru'],
                        help='Dataset-specific safe defaults. auto detects CWRU from source/target names.')
    parser.add_argument('--cwru_apply_general_defaults', type=str2bool, default=True,
                        help='For CWRU, remove PU-specific class-pair calibration and infer the normal class automatically.')
    parser.add_argument('--source_balance_data', type=str2bool, default=False,
                        help='Downsample each labeled source class to the same number of windows.')
    parser.add_argument('--cwru_channel', type=str, default='DE', choices=['DE', 'FE', 'BA'],
                        help='CWRU vibration channel selected from MATLAB variables.')

    # target-domain split parameters
    parser.add_argument('--target_test_size', type=float, default=0.40,
                        help='Target-domain held-out test ratio. Default 0.40 means 60%% unlabeled target train and 40%% target test.')
    parser.add_argument('--target_split_mode', type=str, default='time',
                        choices=['time', 'file', 'window_random'],
                        help='How to split target domain before evaluation: '
                             'time=split each original .mat signal into continuous 60/40 segments before windowing; '
                             'file=split original .mat files per class before windowing; '
                             'window_random=old behavior, split after sliding windows.')

    # class filtering parameters
    parser.add_argument('--include_faults', type=str, default='',
                        help='Comma-separated fault/bearing class names to keep, e.g. K001,KA04,KA16. '
                             'If empty, all detected classes except excluded ones are used.')
    parser.add_argument('--exclude_faults', type=str, default='',
                        help='Comma-separated fault/bearing class names to remove, e.g. KA15,KB23,KI16,KI21.')

    # optimizer parameters          
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd', help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for sgd')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='Betas for adam')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for both sgd and adam')
   
    # learning rate parameters
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='stepLR',
                        help='Type of learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='Parameter for the learning rate scheduler (except "fix")')
    parser.add_argument('--steps', type=str, default='10',
                        help='Step of learning rate decay for "step" and "stepLR"')
    
    # optimization parameters
    parser.add_argument('--backbone', type=str, default='CNN', choices=['CNN', 'ResNet', 'MSCNN_BiMamba_Att', 'MS_BiMamba_Att', 'BIMAMBA'],
                        help='The backbone used to construct the training model (defined in ./modules)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--normlize_type', type=str, choices=['0-1', '-1-1', 'mean-std'], default='-1-1',
                        help='Data normalization methods')
    parser.add_argument('--tradeoff', type=list, default=['exp', 'exp', 'exp'],
                        help='Trade-off coefficients for loss terms. Use integer or "exp". '
                             'Default three terms: MMD, L1, new adaptation losses.')
    parser.add_argument('--zeta', type=float, default=10.0,
                        help='Parameter to control the increasing rate of "exp" tradeoff')

    # V13 stable-baseline optimization controls (training policy only)
    parser.add_argument('--mmd_weight', type=float, default=1.0,
                        help='Explicit multiplier for the existing MMD term; does not add a new loss.')
    parser.add_argument('--mmd_start_epoch', type=int, default=1,
                        help='First epoch applying MMD. Earlier epochs use source classification only for this term.')
    parser.add_argument('--adv_start_epoch', type=int, default=1,
                        help='First epoch applying the existing CDAN adversarial term.')
    parser.add_argument('--clmmd_start_epoch', type=int, default=1,
                        help='First epoch applying the existing class-wise MMD term.')
    parser.add_argument('--source_label_smoothing', type=float, default=0.0,
                        help='Label smoothing for source cross-entropy; 0 keeps the original behavior.')
    parser.add_argument('--grad_clip_norm', type=float, default=0.0,
                        help='Clip total gradient norm before optimizer.step; <=0 disables clipping.')
    parser.add_argument('--lambda_l1', type=float, default=0.0,
                        help='Weight of target prediction L1 consistency. V6-Lite-PU0 defaults to 0 and skips this loss.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout layer coefficient')

    # MFSAN-CDA parameters
    parser.add_argument('--lambda_cda', type=float, default=0.0,
                        help='Weight of optional conditional MMD loss for MFSAN_CDA / MFSAN_CDAN')
    parser.add_argument('--lambda_ent', type=float, default=0.0,
                        help='Weight of target entropy minimization loss for MFSAN_CDA / MFSAN_CDAN')
    parser.add_argument('--cda_detach_prob', type=str2bool, default=True,
                        help='Detach class probabilities when building CDA joint features')

    # MFSAN-CDAN adversarial domain parameters
    parser.add_argument('--lambda_adv', type=float, default=0.02,
                        help='Weight of conditional adversarial domain loss for MFSAN_CDAN')
    parser.add_argument('--lambda_grl', type=float, default=1.0,
                        help='Gradient reversal strength for MFSAN_CDAN')
    parser.add_argument('--adv_hidden_dim', type=int, default=256,
                        help='Hidden dimension of domain discriminator for MFSAN_CDAN')
    parser.add_argument('--adv_detach_prob', type=str2bool, default=True,
                        help='Detach class probabilities when building CDAN joint features')
    parser.add_argument('--adv_use_entropy_weight', type=str2bool, default=True,
                        help='Use entropy-based target sample weights in adversarial domain loss')
    parser.add_argument('--adv_conf_thresh', type=float, default=0.0,
                        help='Only target samples with max probability >= threshold participate in CDAN. 0 means use all samples.')
    
    # save and load
    parser.add_argument('--save', type=str2bool, default=True, help='Save logs and trained model checkpoints')
    parser.add_argument('--save_dir', type=str, default='./ckpt',
                        help='Directory to save logs and model checkpoints')
    parser.add_argument('--load_path', type=str, default='',
                        help='Load trained model checkpoints from this path (for testing, not for resuming training)')


    parser.add_argument('--bla_gate_init', type=float, default=0.01,
                    help='Initial gate value for BiLSTM-Attention residual branch.')

    parser.add_argument('--bla_gate_max', type=float, default=0.03,
                        help='Maximum gate value for BiLSTM-Attention residual branch.')


    # checkpoint parameters
    parser.add_argument('--save_best', type=str2bool, default=True,
                        help='Save best checkpoint according to validation accuracy during training.')

    parser.add_argument('--eval_each_epoch', type=str2bool, default=True,
                        help='Evaluate the held-out target set after every epoch.')
    parser.add_argument('--select_best_on_target', type=str2bool, default=True,
                        help='Use held-out target accuracy to select a best checkpoint. Set False for strict UDA reporting.')

    parser.add_argument('--best_metric', type=str, default='accuracy',
                        choices=['accuracy', 'macro_f1', 'class_aware'],
                        help='Metric used when select_best_on_target=True. class_aware combines accuracy, macro-F1 and one focus-class recall.')
    parser.add_argument('--best_focus_class', type=int, default=-1,
                        help='Focus class used by best_metric=class_aware; -1 disables the focus term.')
    parser.add_argument('--best_accuracy_weight', type=float, default=0.45,
                        help='Accuracy weight for best_metric=class_aware.')
    parser.add_argument('--best_macro_f1_weight', type=float, default=0.35,
                        help='Macro-F1 weight for best_metric=class_aware.')
    parser.add_argument('--best_focus_recall_weight', type=float, default=0.20,
                        help='Focus-class recall weight for best_metric=class_aware.')

    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='Stop after this many epochs without checkpoint-score improvement; 0 disables early stopping.')
    parser.add_argument('--early_stop_min_epoch', type=int, default=1,
                        help='Do not early-stop before this epoch.')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='Minimum checkpoint-score increase counted as an improvement.')
    parser.add_argument('--log_confusion_matrix', type=str2bool, default=True,
                        help='Log target-test prediction counts and the full confusion matrix after each evaluation.')

    # BiMamba small-gate backbone parameters
    parser.add_argument('--bimamba_stem_channels', type=int, default=64,
                        help='Stem channels for BiMamba-Attention auxiliary branch.')
    parser.add_argument('--bimamba_dim', type=int, default=64,
                        help='Token dimension of BiMamba blocks.')
    parser.add_argument('--bimamba_depth', type=int, default=2,
                        help='Number of BiMamba blocks.')
    parser.add_argument('--bimamba_d_state', type=int, default=16,
                        help='State dimension for mamba_ssm Mamba.')
    parser.add_argument('--bimamba_d_conv', type=int, default=4,
                        help='Local convolution width for mamba_ssm Mamba.')
    parser.add_argument('--bimamba_expand', type=int, default=2,
                        help='Expansion ratio for mamba_ssm Mamba.')
    parser.add_argument('--bimamba_gate_init', type=float, default=0.01,
                        help='Initial gate value for BiMamba-Attention residual branch.')
    parser.add_argument('--bimamba_gate_max', type=float, default=0.03,
                        help='Maximum gate value for BiMamba-Attention residual branch.')

    # RWCA: reliability-weighted class-wise alignment parameters
    parser.add_argument('--rw_tau', type=float, default=0.5,
                        help='Temperature for reliability source weighting. Smaller value makes weights sharper.')
    parser.add_argument('--rw_mmd_weight', type=float, default=1.0,
                        help='Weight of source-target MK-MMD distance in source reliability score.')
    parser.add_argument('--rw_ent_weight', type=float, default=1.0,
                        help='Weight of target prediction entropy in source reliability score.')
    parser.add_argument('--rw_detach_weights', type=str2bool, default=True,
                        help='Detach reliability weights before weighted loss aggregation.')
    parser.add_argument('--rw_ema_momentum', type=float, default=0.9,
                        help='EMA momentum for validation/test-time source reliability prior.')
    parser.add_argument('--rw_eval_use_entropy', type=str2bool, default=True,
                        help='During eval, combine EMA source reliability prior with per-sample target entropy.')
    parser.add_argument('--rw_eval_tau', type=float, default=0.5,
                        help='Temperature for eval-time entropy-weighted prediction fusion.')

    parser.add_argument('--lambda_clmmd', type=float, default=0.005,
                        help='Weight of class-wise LMMD / LJMMD-like subdomain alignment loss.')
    parser.add_argument('--clmmd_kernel_num', type=int, default=5,
                        help='Number of Gaussian kernels used in CLMMD.')
    parser.add_argument('--clmmd_kernel_mul', type=float, default=2.0,
                        help='Bandwidth multiplier for Gaussian kernels used in CLMMD.')
    parser.add_argument('--clmmd_min_source', type=int, default=2,
                        help='Minimum source samples required for a class to participate in CLMMD.')
    parser.add_argument('--clmmd_min_target_weight', type=float, default=1e-3,
                        help='Minimum target soft class weight required for a class to participate in CLMMD.')

    parser.add_argument('--pl_conf_thresh', type=float, default=0.80,
                    help='Confidence threshold for V4 pseudo-label gated target samples.')

    parser.add_argument('--pl_min_target', type=int, default=2,
                    help='Minimum high-confidence target samples required per class for V4 gated CLMMD/CW-RWCA.')

    parser.add_argument('--lambda_supcon', type=float, default=0.01,
                    help='Weight of source supervised contrastive loss.')

    parser.add_argument('--supcon_temperature', type=float, default=0.20,
                        help='Temperature for supervised contrastive loss.')

    parser.add_argument('--supcon_start_epoch', type=int, default=3,
                        help='Start epoch for supervised contrastive loss.')

    parser.add_argument('--supcon_feature_mode', type=str, default='G',
                        choices=['G', 'F'],
                        help='Feature mode for SupCon: G uses shared backbone feature, F uses source-specific feature.')

    parser.add_argument('--supcon_focus_classes', type=str, default='3,4',
                        help='Comma-separated class ids for SupCon focus. Use all or empty string for all classes.')



    # V5: MDIFN-style source per-class recognition score
    parser.add_argument('--rec_score_weight', type=float, default=0.30,
                        help='Weight of source per-class recognition score in V5 class-source reliability. 0 disables it.')
    parser.add_argument('--rec_score_mode', type=str, default='prob', choices=['prob', 'acc', 'mix'],
                        help='Recognition score mode: prob=mean true-class probability, acc=per-class accuracy, mix=average of both.')
    parser.add_argument('--rec_score_detach', type=str2bool, default=True,
                        help='Detach source per-class recognition score before class-source weighting.')

    # V5: MSD-MCA-style multi-classifier alignment
    parser.add_argument('--lambda_mca', type=float, default=0.0,
                        help='Weight of reliability-guided multi-classifier alignment loss.')
    parser.add_argument('--mca_start_epoch', type=int, default=1,
                        help='Start epoch for multi-classifier alignment loss.')
    parser.add_argument('--mca_use_reliability', type=str2bool, default=True,
                        help='Use class-source reliability weights in multi-classifier alignment.')
    parser.add_argument('--mca_detach_fused', type=str2bool, default=True,
                        help='Detach fused target prediction when used as reference in multi-classifier alignment.')
    parser.add_argument('--mca_eps', type=float, default=1e-5,
                        help='Numerical epsilon for multi-classifier alignment.')


    # CW-RWCA schedule (explicitly exposed for V6)
    parser.add_argument('--cw_warmup_epochs', type=int, default=3,
                        help='Epochs using mainly global RWCA before class-wise correction.')
    parser.add_argument('--cw_alpha', type=float, default=0.30,
                        help='Maximum participation ratio of class-wise reliability weights.')
    parser.add_argument('--cw_alpha_ramp_epochs', type=int, default=3,
                        help='Ramp epochs for class-wise reliability after warmup.')

    # V6 stable negative-source gate
    parser.add_argument('--v6_gate_enabled', type=str2bool, default=True,
                        help='Enable epoch-stable automatic negative-source gating.')
    parser.add_argument('--v6_gate_start_epoch', type=int, default=2,
                        help='First epoch allowed to accumulate negative-source evidence.')
    parser.add_argument('--v6_gate_confirm_epochs', type=int, default=2,
                        help='Consecutive epochs required to confirm a negative source.')
    parser.add_argument('--v6_gate_release_epochs', type=int, default=3,
                        help='Consecutive epochs required to release a confirmed source gate.')
    parser.add_argument('--v6_gate_confirm_gap', type=float, default=0.08,
                        help='Minimum reliability-weight gap between the worst and second-worst sources for confirmation.')
    parser.add_argument('--v6_gate_release_gap', type=float, default=0.03,
                        help='Gap below which the negative-source confirmation may be released.')
    parser.add_argument('--v6_gate_preconfirm_floor', type=float, default=0.005,
                        help='Minimum source weight before stable confirmation.')
    parser.add_argument('--v6_gate_bottom_floor', type=float, default=0.001,
                        help='Weight assigned to a confirmed negative source.')
    parser.add_argument('--v6_gate_max_source_weight', type=float, default=0.75,
                        help='Maximum weight allowed for any single source.')
    parser.add_argument('--v6_gate_apply_to_supcon', type=str2bool, default=True,
                        help='Remove a confirmed negative source from focused SupCon.')
    parser.add_argument('--v6_supcon_source_min_weight', type=float, default=0.005,
                        help='Minimum source reliability for participation in SupCon before confirmation.')

    # V6 class-level adaptation controls
    parser.add_argument('--v6_class_weight_power', type=float, default=1.20,
                        help='Mild power sharpening applied to final class-wise source weights.')
    parser.add_argument('--v6_class_alignment_boost', type=float, default=1.0,
                        help='Additional multiplier for CLMMD and MCA; 1.0 reproduces V5 strength.')
    parser.add_argument('--v6_mca_pairwise_weight', type=float, default=0.0,
                        help='Weight of active-source pairwise class-correlation alignment added to MCA.')


    # V7: class-level stable gate
    parser.add_argument('--v7_class_gate_enabled', type=str2bool, default=True,
                        help='Enable independent stable source gating for every class.')
    parser.add_argument('--v7_class_gate_start_epoch', type=int, default=4,
                        help='First epoch allowed to accumulate class-level negative-source evidence.')
    parser.add_argument('--v7_class_gate_confirm_epochs', type=int, default=3,
                        help='Consecutive epochs required to confirm a weak source for one class.')
    parser.add_argument('--v7_class_gate_release_epochs', type=int, default=3,
                        help='Consecutive epochs required to release one class-level source gate.')
    parser.add_argument('--v7_class_gate_confirm_gap', type=float, default=0.10,
                        help='Minimum second-worst minus worst class-source weight gap for confirmation.')
    parser.add_argument('--v7_class_gate_release_gap', type=float, default=0.04,
                        help='Gap threshold used when deciding whether to release a class gate.')
    parser.add_argument('--v7_class_gate_max_bad_weight', type=float, default=0.20,
                        help='A class-source pair must be no larger than this value to be confirmed weak.')
    parser.add_argument('--v7_class_gate_preconfirm_floor', type=float, default=0.005,
                        help='Minimum class-source weight before class-level confirmation.')
    parser.add_argument('--v7_class_gate_bottom_floor', type=float, default=0.01,
                        help='Weight assigned to a confirmed weak source for one class.')

    # V7: adaptive class-specialist rescue
    parser.add_argument('--v7_class_rescue_enabled', type=str2bool, default=True,
                        help='Increase class-wise reliability participation when one source is a clear class specialist.')
    parser.add_argument('--v7_class_rescue_max_alpha', type=float, default=0.70,
                        help='Maximum class-wise alpha for a class with strong specialist evidence.')
    parser.add_argument('--v7_class_rescue_gap_start', type=float, default=0.10,
                        help='Top1-top2 class reliability gap where specialist rescue begins.')
    parser.add_argument('--v7_class_rescue_gap_full', type=float, default=0.40,
                        help='Top1-top2 class reliability gap where specialist rescue reaches max alpha.')

    # V7: class-aware SupCon
    parser.add_argument('--v7_supcon_class_min_weight', type=float, default=0.02,
                        help='Minimum source-class weight for samples to participate in SupCon.')

    # V7: conflict-aware dynamic fusion
    parser.add_argument('--v7_conflict_fusion_enabled', type=str2bool, default=True,
                        help='Enable branch-conflict-aware dynamic target prediction fusion.')
    parser.add_argument('--v7_agree_prior_power', type=float, default=1.0,
                        help='Class prior power when all source classifiers agree.')
    parser.add_argument('--v7_conflict_prior_power', type=float, default=0.30,
                        help='Reduced class prior power when source classifiers disagree.')
    parser.add_argument('--v7_conflict_top1_margin_bonus', type=float, default=1.0,
                        help='Bonus on a conflicting branch top-1 class proportional to its top1-top2 margin.')
    parser.add_argument('--v7_conflict_weight_temperature', type=float, default=1.0,
                        help='Temperature for source weights in conflict-aware fusion.')
    parser.add_argument('--v7_class_gate_log_classes', type=str, default='0,3,4,6',
                        help='Comma-separated class ids for detailed V7 gate logging, or all.')


    # V8: confusion-pair-aware Hard-negative SupCon
    parser.add_argument('--v8_hard_supcon_enabled', type=str2bool, default=True,
                        help='Enable confusion-pair-aware hard-negative weighting inside the existing SupCon loss.')
    parser.add_argument('--v8_hard_negative_pairs', type=str, default='0-3,3-4,3-6,7-8',
                        help='Comma-separated undirected class pairs, e.g. 0-3,3-4,3-6,7-8.')
    parser.add_argument('--v8_hard_negative_weight', type=float, default=2.0,
                        help='Denominator weight for configured hard-negative class pairs in SupCon.')
    parser.add_argument('--v8_supcon_anchor_classes', type=str, default='0,1,3,4,6,7,8',
                        help='Classes used as SupCon anchors. Pair-partner samples remain in the contrastive pool.')

    # V8: prototype-guided CLMMD pseudo-label filtering
    parser.add_argument('--v8_prototype_filter_enabled', type=str2bool, default=True,
                        help='Require target pseudo labels to agree with source-class prototypes before CLMMD.')
    parser.add_argument('--v8_prototype_start_epoch', type=int, default=3,
                        help='First epoch using prototype agreement to filter CLMMD target samples.')
    parser.add_argument('--v8_prototype_ema_momentum', type=float, default=0.90,
                        help='EMA momentum for per-source, per-class branch-feature prototypes.')
    parser.add_argument('--v8_prototype_margin', type=float, default=0.05,
                        help='Minimum cosine-similarity top1-top2 margin for accepting a prototype pseudo label.')
    parser.add_argument('--v8_prototype_min_updates', type=int, default=1,
                        help='Minimum prototype updates required before one source-class prototype is valid.')
    parser.add_argument('--v8_prototype_conf_overrides', type=str, default='0:0.90,3:0.90,4:0.90,6:0.85',
                        help='Per-class CLMMD confidence thresholds, e.g. 0:0.90,3:0.90,4:0.90,6:0.85.')
    parser.add_argument('--v8_clmmd_class_boost', type=str, default='3:1.5,4:1.3,6:1.2',
                        help='Per-class multipliers applied inside the existing CLMMD term.')
    parser.add_argument('--v8_prototype_log_classes', type=str, default='0,3,4,6',
                        help='Comma-separated class ids for prototype-filter acceptance logging, or all.')


    # V9: historical class-specialist protection
    parser.add_argument('--v9_specialist_protection_enabled', type=str2bool, default=True,
                        help='Protect a repeatedly strongest source from class-level hard suppression.')
    parser.add_argument('--v9_specialist_start_epoch', type=int, default=4,
                        help='First epoch collecting class-specialist evidence.')
    parser.add_argument('--v9_specialist_confirm_epochs', type=int, default=2,
                        help='Consecutive epochs required to protect one source-class specialist.')
    parser.add_argument('--v9_specialist_min_weight', type=float, default=0.45,
                        help='Minimum pre-gate source weight required for specialist protection.')
    parser.add_argument('--v9_specialist_min_gap', type=float, default=0.10,
                        help='Minimum gap between the best and second-best source for specialist protection.')
    parser.add_argument('--v9_specialist_floor', type=float, default=0.05,
                        help='Minimum class weight retained for a protected specialist.')
    parser.add_argument('--v9_specialist_release_weight', type=float, default=0.15,
                        help='Protected specialist release threshold before consecutive release counting.')
    parser.add_argument('--v9_specialist_release_epochs', type=int, default=4,
                        help='Consecutive low-weight epochs required to release a protected specialist.')

    # V9: delayed and softly ramped Hard-negative SupCon
    parser.add_argument('--v9_hard_supcon_start_epoch', type=int, default=8,
                        help='First epoch applying hard-pair weighting inside SupCon.')
    parser.add_argument('--v9_hard_supcon_ramp_epochs', type=int, default=4,
                        help='Number of epochs used to ramp hard-pair weight from 1.0 to its final value.')

    # V9: source-radius-aware prototype filtering
    parser.add_argument('--v9_prototype_filter_classes', type=str, default='0,3,4',
                        help='Classes using radius-aware prototype filtering, or all.')
    parser.add_argument('--v9_radius_ema_momentum', type=float, default=0.90,
                        help='EMA momentum for source-class cosine-distance mean and variance.')
    parser.add_argument('--v9_radius_std_scale', type=float, default=2.0,
                        help='Accept radius equals source mean plus this multiple of source standard deviation.')
    parser.add_argument('--v9_radius_min', type=float, default=0.03,
                        help='Minimum accepted cosine-distance radius.')
    parser.add_argument('--v9_radius_max', type=float, default=0.30,
                        help='Maximum accepted cosine-distance radius.')
    parser.add_argument('--v9_prototype_min_similarity', type=float, default=0.30,
                        help='Minimum cosine similarity to the selected source-class prototype.')
    parser.add_argument('--v9_prototype_soft_tau', type=float, default=0.10,
                        help='Temperature for distance-based soft target weights in CLMMD.')

    # V10: switchable class-specialist memory
    parser.add_argument('--v10_specialist_switch_enabled', type=str2bool, default=True,
                        help='Allow a protected class specialist to switch when another source becomes persistently dominant.')
    parser.add_argument('--v10_specialist_switch_epochs', type=int, default=2,
                        help='Consecutive dominant epochs required before switching the protected specialist.')
    parser.add_argument('--v10_specialist_switch_min_weight', type=float, default=0.55,
                        help='Minimum new-source class weight required for specialist switching.')
    parser.add_argument('--v10_specialist_switch_min_gap', type=float, default=0.15,
                        help='Minimum gap between the new source and the currently protected source.')

    # V10: pair-specific Hard-SupCon weights
    parser.add_argument('--v10_hard_pair_weights', type=str, default='0-3:1.10,3-4:1.30',
                        help='Pair-specific final Hard-SupCon weights, e.g. 0-3:1.10,3-4:1.30.')

    # V10: class-calibrated prototype radius
    parser.add_argument('--v10_radius_class_min', type=str, default='0:0.025,3:0.025,4:0.025',
                        help='Per-class minimum cosine-distance radius.')
    parser.add_argument('--v10_radius_class_max', type=str, default='0:0.040,3:0.050,4:0.050',
                        help='Per-class maximum cosine-distance radius.')
    parser.add_argument('--v10_radius_weak_source_threshold', type=float, default=0.05,
                        help='Source-class reliability below which the radius cap is tightened.')
    parser.add_argument('--v10_radius_weak_source_cap_scale', type=float, default=0.80,
                        help='Multiplier applied to the class radius cap for a weak source-class pair.')

    # V10: optional fault-safe normal decision guard
    parser.add_argument('--v10_normal_guard_enabled', type=str2bool, default=True,
                        help='Require sufficient fused probability before retaining the normal-class prediction.')
    parser.add_argument('--v10_normal_class', type=int, default=0,
                        help='Normal class index used by the V10 fault-safe decision guard.')
    parser.add_argument('--v10_normal_min_prob', type=float, default=0.80,
                        help='Minimum fused normal-class probability required to keep a normal prediction.')
    parser.add_argument('--v10_normal_guard_min_fault_prob', type=float, default=0.05,
                        help='Minimum best-fault probability required before overriding a low-confidence normal prediction.')


    # V11: CWRU-1 ball_21 rescue (class 2 by the standard alphabetical mapping)
    parser.add_argument('--v11_cwru1_rescue_enabled', type=str2bool, default=False,
                        help='Enable conservative prototype/top-k rescue for a collapsed CWRU target class.')
    parser.add_argument('--v11_rescue_class', type=int, default=2,
                        help='Target class id rescued by V11 (CWRU ball_21 is class 2).')
    parser.add_argument('--v11_confusion_classes', type=str, default='0,1',
                        help='Classes that may absorb the rescue class, e.g. 0,1 for ball_07/ball_14.')
    parser.add_argument('--v11_rescue_start_epoch', type=int, default=2,
                        help='First epoch enabling target top-k/prototype rescue inside CLMMD.')
    parser.add_argument('--v11_rescue_topk', type=int, default=2,
                        help='The rescue class must appear in this many top classifier ranks.')
    parser.add_argument('--v11_rescue_min_class_prob', type=float, default=0.10,
                        help='Minimum target probability of the rescue class before prototype rescue.')
    parser.add_argument('--v11_rescue_proto_margin', type=float, default=0.03,
                        help='Minimum source-prototype top1-top2 cosine margin for rescue.')
    parser.add_argument('--v11_rescue_min_similarity', type=float, default=0.35,
                        help='Minimum cosine similarity to the rescue-class source prototype.')
    parser.add_argument('--v11_rescue_clmmd_boost', type=float, default=1.50,
                        help='Multiplier for the rescued class component inside the existing CLMMD term.')
    parser.add_argument('--v11_eval_rescue_enabled', type=str2bool, default=True,
                        help='Enable conservative probability calibration for the rescued class at evaluation.')
    parser.add_argument('--v11_eval_min_class_prob', type=float, default=0.08,
                        help='Minimum fused probability required for evaluation-time rescue.')
    parser.add_argument('--v11_eval_competitor_ratio', type=float, default=0.35,
                        help='Rescue probability must reach this fraction of the strongest confusion-class probability.')
    parser.add_argument('--v11_eval_min_source_votes', type=int, default=2,
                        help='Minimum number of source branches that rank the rescue class in their top-k.')
    parser.add_argument('--v11_eval_boost', type=float, default=2.00,
                        help='Multiplicative boost applied to the rescue class before final renormalization.')


    # V12: stable CWRU-1 rescue without changing the network architecture
    parser.add_argument('--v12_rescue_enabled', type=str2bool, default=False,
                        help='Enable the delayed, capped, normal-protected class rescue used by V12.')
    parser.add_argument('--v12_rescue_class', type=int, default=2,
                        help='Target class rescued by V12; CWRU ball_21 is class 2.')
    parser.add_argument('--v12_normal_class', type=int, default=6,
                        help='Normal class index used by the V12 exclusion guards.')
    parser.add_argument('--v12_confusion_classes', type=str, default='0,1',
                        help='Classes that absorb the rescue class, e.g. ball_07 and ball_14.')
    parser.add_argument('--v12_rescue_source_indices', type=str, default='2',
                        help='Comma-separated source branch indices allowed to supply rescue CLMMD; source 2 is CWRU_3 in the dedicated task.')
    parser.add_argument('--v12_rescue_start_epoch', type=int, default=6,
                        help='First epoch enabling stable rescue.')
    parser.add_argument('--v12_rescue_end_epoch', type=int, default=10,
                        help='Last epoch enabling stable rescue, limiting late pseudo-label drift.')
    parser.add_argument('--v12_rescue_topk', type=int, default=2,
                        help='The rescue class must appear in the branch top-k predictions.')
    parser.add_argument('--v12_rescue_min_class_prob', type=float, default=0.20,
                        help='Minimum rescue-class probability for a target candidate.')
    parser.add_argument('--v12_rescue_min_competitor_ratio', type=float, default=0.50,
                        help='Rescue probability must reach this fraction of the strongest confusion-class probability.')
    parser.add_argument('--v12_rescue_max_normal_prob', type=float, default=0.20,
                        help='Reject target candidates whose normal-class probability exceeds this value.')
    parser.add_argument('--v12_rescue_proto_margin', type=float, default=0.08,
                        help='Minimum top1-top2 source-prototype cosine margin.')
    parser.add_argument('--v12_rescue_normal_proto_margin', type=float, default=0.10,
                        help='Minimum cosine-similarity advantage of rescue prototype over normal prototype.')
    parser.add_argument('--v12_rescue_min_similarity', type=float, default=0.45,
                        help='Minimum cosine similarity to the rescue-class source prototype.')
    parser.add_argument('--v12_rescue_radius_cap', type=float, default=0.05,
                        help='Maximum allowed source-prototype radius for rescued target samples.')
    parser.add_argument('--v12_rescue_max_per_batch', type=int, default=4,
                        help='Maximum number of rescued target samples per source branch and mini-batch.')
    parser.add_argument('--v12_rescue_min_target', type=int, default=2,
                        help='Minimum selected rescue samples required to compute the rescue CLMMD component.')
    parser.add_argument('--v12_rescue_mix_alpha', type=float, default=0.25,
                        help='Blend ratio of rescue CLMMD with the original class-wise CLMMD value.')
    parser.add_argument('--v12_rescue_clmmd_boost', type=float, default=1.10,
                        help='Small multiplier applied to the filtered rescue CLMMD component.')
    parser.add_argument('--v12_rescue_score_tau', type=float, default=0.10,
                        help='Temperature used to rank filtered rescue candidates before the per-batch cap.')
    parser.add_argument('--v12_eval_rescue_enabled', type=str2bool, default=False,
                        help='Optional evaluation-time calibration; disabled in the recommended experiment.')
    parser.add_argument('--v12_eval_min_class_prob', type=float, default=0.20,
                        help='Minimum fused rescue-class probability for optional evaluation calibration.')
    parser.add_argument('--v12_eval_competitor_ratio', type=float, default=0.60,
                        help='Optional evaluation rescue-to-competitor probability ratio.')
    parser.add_argument('--v12_eval_min_source_votes', type=int, default=2,
                        help='Minimum source branches voting for the rescue class during optional evaluation calibration.')
    parser.add_argument('--v12_eval_boost', type=float, default=1.25,
                        help='Optional small evaluation-time rescue probability multiplier.')

    args = parser.parse_args()
    return args
