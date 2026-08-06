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
    parser.add_argument('--dropout', type=float, default=0., help='Dropout layer coefficient')

    # MFSAN-CDA parameters
    parser.add_argument('--lambda_cda', type=float, default=0.0,
                        help='Weight of optional conditional MMD loss for MFSAN_CDA / MFSAN_CDAN')
    parser.add_argument('--lambda_ent', type=float, default=0.005,
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

    parser.add_argument('--lambda_clmmd', type=float, default=0.02,
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

    parser.add_argument('--lambda_supcon', type=float, default=0.02,
                    help='Weight of source supervised contrastive loss.')

    parser.add_argument('--supcon_temperature', type=float, default=0.10,
                        help='Temperature for supervised contrastive loss.')

    parser.add_argument('--supcon_start_epoch', type=int, default=1,
                        help='Start epoch for supervised contrastive loss.')

    parser.add_argument('--supcon_feature_mode', type=str, default='G',
                        choices=['G', 'F'],
                        help='Feature mode for SupCon: G uses shared backbone feature, F uses source-specific feature.')

    parser.add_argument('--supcon_focus_classes', type=str, default='1,2',
                        help='Comma-separated class ids for SupCon focus. Use all or empty string for all classes.')



    # V5: MDIFN-style source per-class recognition score
    parser.add_argument('--rec_score_weight', type=float, default=0.30,
                        help='Weight of source per-class recognition score in V5 class-source reliability. 0 disables it.')
    parser.add_argument('--rec_score_mode', type=str, default='prob', choices=['prob', 'acc', 'mix'],
                        help='Recognition score mode: prob=mean true-class probability, acc=per-class accuracy, mix=average of both.')
    parser.add_argument('--rec_score_detach', type=str2bool, default=True,
                        help='Detach source per-class recognition score before class-source weighting.')

    # V5: MSD-MCA-style multi-classifier alignment
    parser.add_argument('--lambda_mca', type=float, default=0.02,
                        help='Weight of reliability-guided multi-classifier alignment loss.')
    parser.add_argument('--mca_start_epoch', type=int, default=1,
                        help='Start epoch for multi-classifier alignment loss.')
    parser.add_argument('--mca_use_reliability', type=str2bool, default=True,
                        help='Use class-source reliability weights in multi-classifier alignment.')
    parser.add_argument('--mca_detach_fused', type=str2bool, default=True,
                        help='Detach fused target prediction when used as reference in multi-classifier alignment.')
    parser.add_argument('--mca_eps', type=float, default=1e-5,
                        help='Numerical epsilon for multi-classifier alignment.')

    args = parser.parse_args()
    return args
