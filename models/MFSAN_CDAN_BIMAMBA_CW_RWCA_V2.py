# -*- coding: utf-8 -*-
"""
MFSAN-CDAN-BiMamba-CW-RWCA-V2-V2

CW-RWCA-V2 = Global-guided Class-wise Reliability-Weighted Class Alignment

在第一版 CW-RWCA 基础上加入“全局 RWCA 先验约束”。

第一版 CW-RWCA 允许每个类别自由选择源域，但在 PU_0 这种难目标域上，
早期伪标签不稳定会把 PU_3 这类全局不可靠源域重新抬高，造成负迁移。

V2 的核心做法：
1. 先按原 RWCA 计算全局源域可靠性 w_global[s]；
2. 再计算类别级源域可靠性 w_class[s,c]；
3. 用 w_global 约束 w_class：
   w_final[s,c] = normalize((1-alpha)*w_global[s] + alpha*w_class[s,c])；
4. 前若干 epoch 只使用全局权重，之后逐步引入类别级权重。

推荐模型名：MFSAN_CDAN_BIMAMBA_CW_RWCA_V2_V2
"""

import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import utils
import modules
import modules_bimamba
from train_utils import TrainerBase


class GradientReverseFunction(Function):
    """Gradient Reversal Layer."""

    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coeff * grad_output, None


def grad_reverse(x, coeff=1.0):
    return GradientReverseFunction.apply(x, coeff)


class DomainDiscriminator(nn.Module):
    """
    Conditional domain discriminator.

    Input : joint feature [B, C*D]
    Output: domain logits [B, 2], where 0=source, 1=target
    """

    def __init__(self, input_dim, hidden_dim=256, dropout=0.0):
        super(DomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 8)),
            nn.BatchNorm1d(max(hidden_dim // 2, 8)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 8), 2),
        )

    def forward(self, x):
        return self.net(x)


class Trainer(TrainerBase):

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        if args.train_mode != 'multi_source':
            raise ValueError('MFSAN_CDAN_BIMAMBA_CW_RWCA_V2 is designed for --train_mode multi_source.')

        self.src_labels_flat = sorted(list(set([label for sublist in args.label_sets[:-1] for label in sublist])))
        self.num_classes = len(self.src_labels_flat)

        # ========== Shared backbone: MSCNN + BiMamba-Attention + SmallGate ==========
        if args.backbone in ['CNN', 'MSCNN_BiMamba_Att', 'MS_BiMamba_Att', 'BIMAMBA']:
            self.G = modules_bimamba.MSCNNBiMambaAttBackbone(
                in_channel=1,
                stem_channels=int(getattr(args, 'bimamba_stem_channels', 64)),
                mamba_dim=int(getattr(args, 'bimamba_dim', 64)),
                mamba_depth=int(getattr(args, 'bimamba_depth', 2)),
                mamba_d_state=int(getattr(args, 'bimamba_d_state', 16)),
                mamba_d_conv=int(getattr(args, 'bimamba_d_conv', 4)),
                mamba_expand=int(getattr(args, 'bimamba_expand', 2)),
                dropout=args.dropout,
                gate_init=float(getattr(args, 'bimamba_gate_init', 0.01)),
                gate_max=float(getattr(args, 'bimamba_gate_max', 0.03)),
            ).to(self.device)
            actual_backbone = 'MSCNN_BiMamba_Att_SmallGate'
        elif args.backbone == 'ResNet':
            self.G = modules.ResNet(in_channel=1, layers=[2, 2, 2, 2]).to(self.device)
            actual_backbone = 'ResNet'
        else:
            raise Exception(f"unknown backbone type {args.backbone}")

        logging.info('Using model: MFSAN_CDAN_BIMAMBA_CW_RWCA_V2')
        logging.info('Requested backbone: {}'.format(args.backbone))
        logging.info('Actual backbone: {}'.format(actual_backbone))
        logging.info('Backbone output dim: {}'.format(self.G.out_dim))

        if hasattr(self.G, 'uses_real_mamba'):
            logging.info('BiMamba implementation: {}'.format(
                'mamba_ssm' if self.G.uses_real_mamba else 'lite_pytorch_fallback'
            ))
        if hasattr(self.G, 'get_gate'):
            logging.info('Initial BiMamba-Att residual gate: {:.6f}'.format(self.G.get_gate().detach().item()))
            if hasattr(self.G, 'gate_max'):
                logging.info('Max BiMamba-Att residual gate: {:.6f}'.format(self.G.gate_max))

        # ========== Multi-source task-specific branches Fs_i / Cs_i ==========
        self.Fs = nn.ModuleList([
            modules.MLP(
                input_size=self.G.out_dim,
                dropout=args.dropout,
                num_layer=2,
                output_layer=False
            )
            for _ in range(self.num_source)
        ]).to(self.device)

        self.Cs = nn.ModuleList([
            modules.MLP(
                input_size=self.Fs[i].feature_dim,
                output_size=self.num_classes,
                num_layer=1,
                last=None
            )
            for i in range(self.num_source)
        ]).to(self.device)

        self.feature_dim = self.Fs[0].feature_dim
        self.joint_dim = self.num_classes * self.feature_dim

        # Feature-level MK-MMD and conditional joint MK-MMD
        self.mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
            kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)]
        )
        self.cda_mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
            kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)]
        )

        # CDAN discriminators: one discriminator per source domain
        adv_hidden_dim = int(getattr(args, 'adv_hidden_dim', 256))
        self.Ds = nn.ModuleList([
            DomainDiscriminator(
                input_dim=self.joint_dim,
                hidden_dim=adv_hidden_dim,
                dropout=args.dropout
            )
            for _ in range(self.num_source)
        ]).to(self.device)

        self._init_data()

        self.src = ['concat_source'] if args.train_mode == 'source_combine' else args.source_name
        self.optimizer = self._get_optimizer([self.G, self.Fs, self.Cs, self.Ds])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        self.num_iter = sum([len(self.dataloaders[s]) for s in self.src])

        # ========== Hyperparameters ==========
        self.lambda_cda = float(getattr(args, 'lambda_cda', 0.0))
        self.lambda_ent = float(getattr(args, 'lambda_ent', 0.005))
        self.detach_prob = bool(getattr(args, 'cda_detach_prob', True))

        self.lambda_adv = float(getattr(args, 'lambda_adv', 0.01))
        self.lambda_grl = float(getattr(args, 'lambda_grl', 0.5))
        self.adv_detach_prob = bool(getattr(args, 'adv_detach_prob', True))
        self.adv_use_entropy_weight = bool(getattr(args, 'adv_use_entropy_weight', True))
        self.adv_conf_thresh = float(getattr(args, 'adv_conf_thresh', 0.0))

        # Reliability-weighted fusion / alignment
        self.rw_tau = float(getattr(args, 'rw_tau', 0.5))
        self.rw_mmd_weight = float(getattr(args, 'rw_mmd_weight', 1.0))
        self.rw_ent_weight = float(getattr(args, 'rw_ent_weight', 1.0))
        self.rw_detach_weights = bool(getattr(args, 'rw_detach_weights', True))
        self.rw_ema_momentum = float(getattr(args, 'rw_ema_momentum', 0.9))
        self.rw_eval_use_entropy = bool(getattr(args, 'rw_eval_use_entropy', True))
        self.rw_eval_tau = float(getattr(args, 'rw_eval_tau', self.rw_tau))

        # Class-wise LMMD / LJMMD-like subdomain alignment
        self.lambda_clmmd = float(getattr(args, 'lambda_clmmd', 0.02))
        self.clmmd_kernel_num = int(getattr(args, 'clmmd_kernel_num', 5))
        self.clmmd_kernel_mul = float(getattr(args, 'clmmd_kernel_mul', 2.0))
        self.clmmd_min_source = int(getattr(args, 'clmmd_min_source', 2))
        self.clmmd_min_target_weight = float(getattr(args, 'clmmd_min_target_weight', 1e-3))

        # ========== V2: Global-guided CW control ==========
        # 这些参数不需要修改 opt.py；如果 opt.py 中没有对应命令行参数，就使用这里的默认值。
        # cw_warmup_epochs: 前几个 epoch 只使用全局 RWCA 权重，避免目标域早期伪标签污染类别级权重。
        # cw_alpha: 类别级权重最大参与比例。建议 0.2~0.3，不建议太大。
        # cw_alpha_ramp_epochs: warmup 后用几个 epoch 逐步把 alpha 拉到 cw_alpha。
        self.cw_warmup_epochs = int(getattr(args, 'cw_warmup_epochs', 3))
        self.cw_alpha = float(getattr(args, 'cw_alpha', 0.30))
        self.cw_alpha_ramp_epochs = int(getattr(args, 'cw_alpha_ramp_epochs', 3))

        self.entropy_eps = 1e-5

        # Global source reliability EMA, kept for compatibility and concise logging.
        self.source_weight_ema = torch.ones(self.num_source, device=self.device) / float(self.num_source)

        # CW-RWCA核心：class_source_weight_ema[k, c] 表示第 k 个源域对第 c 类的可靠性。
        # 每一列按源域归一化，即 sum_k w[k, c] = 1。
        self.class_source_weight_ema = (
            torch.ones(self.num_source, self.num_classes, device=self.device) / float(self.num_source)
        )

        # 默认重点打印 Class-1 / Class-2，因为 PU_0 实验里主要是这两类拖后腿。
        self._cw_log_classes = [c for c in [1, 2] if c < self.num_classes]
        self._last_train_source_weights = None
        self._last_val_source_weights = None

        logging.info('MFSAN-CDAN-BiMamba-CW-RWCA-V2 lambda_adv: {:.6f}'.format(self.lambda_adv))
        logging.info('MFSAN-CDAN-BiMamba-CW-RWCA-V2 lambda_grl: {:.6f}'.format(self.lambda_grl))
        logging.info('MFSAN-CDAN-BiMamba-CW-RWCA-V2 lambda_cda: {:.6f}'.format(self.lambda_cda))
        logging.info('MFSAN-CDAN-BiMamba-CW-RWCA-V2 lambda_ent: {:.6f}'.format(self.lambda_ent))
        logging.info('MFSAN-CDAN-BiMamba-CW-RWCA-V2 lambda_clmmd: {:.6f}'.format(self.lambda_clmmd))
        logging.info('MFSAN-CDAN-BiMamba-CW-RWCA-V2 rw_tau: {:.6f}'.format(self.rw_tau))
        logging.info('MFSAN-CDAN-BiMamba-CW-RWCA-V2 rw_mmd_weight: {:.6f}'.format(self.rw_mmd_weight))
        logging.info('MFSAN-CDAN-BiMamba-CW-RWCA-V2 rw_ent_weight: {:.6f}'.format(self.rw_ent_weight))
        logging.info('MFSAN-CDAN-BiMamba-CW-RWCA-V2 rw_detach_weights: {}'.format(self.rw_detach_weights))
        logging.info('MFSAN-CDAN-BiMamba-CW-RWCA-V2 joint feature dim: {} x {} = {}'.format(
            self.num_classes, self.feature_dim, self.joint_dim
        ))

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def _checkpoint_dict(self):
        return {
            'G': self.G.state_dict(),
            'Fs': self.Fs.state_dict(),
            'Cs': self.Cs.state_dict(),
            'Ds': self.Ds.state_dict(),
            'source_weight_ema': self.source_weight_ema.detach().cpu(),
            'class_source_weight_ema': self.class_source_weight_ema.detach().cpu(),
            'lambda_adv': self.lambda_adv,
            'lambda_grl': self.lambda_grl,
            'lambda_cda': self.lambda_cda,
            'lambda_ent': self.lambda_ent,
            'lambda_clmmd': self.lambda_clmmd,
            'rw_tau': self.rw_tau,
            'rw_mmd_weight': self.rw_mmd_weight,
            'rw_ent_weight': self.rw_ent_weight,
            'cw_warmup_epochs': self.cw_warmup_epochs,
            'cw_alpha': self.cw_alpha,
            'cw_alpha_ramp_epochs': self.cw_alpha_ramp_epochs,
        }

    def save_model(self):
        path = self.args.save_path + '.pth'
        torch.save(self._checkpoint_dict(), path)
        logging.info('Model saved to {}'.format(path))

    def save_best_model(self):
        path = self.args.save_path + '_best.pth'
        torch.save(self._checkpoint_dict(), path)
        logging.info('Best model saved to {}'.format(path))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        self.G.load_state_dict(ckpt['G'])
        self.Fs.load_state_dict(ckpt['Fs'])
        self.Cs.load_state_dict(ckpt['Cs'])
        if 'Ds' in ckpt:
            self.Ds.load_state_dict(ckpt['Ds'])
        else:
            logging.warning('No domain discriminator weights found in checkpoint.')
        if 'source_weight_ema' in ckpt:
            self.source_weight_ema = ckpt['source_weight_ema'].to(self.device).float()
            self.source_weight_ema = self.source_weight_ema / (self.source_weight_ema.sum() + self.entropy_eps)
            logging.info('Loaded source_weight_ema: {}'.format(self.source_weight_ema.detach().cpu().numpy()))

        if 'class_source_weight_ema' in ckpt:
            cw = ckpt['class_source_weight_ema'].to(self.device).float()
            if cw.dim() == 2 and cw.size(0) == self.num_source and cw.size(1) == self.num_classes:
                self.class_source_weight_ema = self._normalize_class_source_weights(cw)
                self.source_weight_ema = self._source_weights_from_class_weights(self.class_source_weight_ema)
                logging.info('Loaded class_source_weight_ema with shape {}'.format(tuple(cw.shape)))
            else:
                logging.warning('Ignore class_source_weight_ema due to incompatible shape: {}'.format(tuple(cw.shape)))
        else:
            # 兼容旧 RWCA checkpoint：如果只有全局源域权重，就扩展成每类相同的权重。
            self.class_source_weight_ema = self.source_weight_ema.view(-1, 1).repeat(1, self.num_classes)
            self.class_source_weight_ema = self._normalize_class_source_weights(self.class_source_weight_ema)

    # ------------------------------------------------------------------
    # Basic states
    # ------------------------------------------------------------------
    def _set_to_train(self):
        self.G.train()
        self.Fs.train()
        self.Cs.train()
        self.Ds.train()

    def _set_to_eval(self):
        self.G.eval()
        self.Fs.eval()
        self.Cs.eval()
        self.Ds.eval()

    # ------------------------------------------------------------------
    # Loss utilities
    # ------------------------------------------------------------------
    def _target_entropy(self, probs):
        probs = torch.clamp(probs, min=self.entropy_eps, max=1.0)
        return -(probs * torch.log(probs)).sum(dim=1).mean()

    def _entropy_scalar(self, probs):
        probs = torch.clamp(probs, min=self.entropy_eps, max=1.0)
        return -(probs * torch.log(probs)).sum(dim=1).mean()

    def _entropy_vector(self, probs):
        probs = torch.clamp(probs, min=self.entropy_eps, max=1.0)
        return -(probs * torch.log(probs)).sum(dim=1)

    def _entropy_weight(self, probs):
        probs = torch.clamp(probs, min=self.entropy_eps, max=1.0)
        ent = -(probs * torch.log(probs)).sum(dim=1)
        w = torch.exp(-ent).detach()
        w = w / (w.mean() + self.entropy_eps)
        return w

    def _joint_feature(self, features, probs, detach_prob=None):
        if detach_prob is None:
            detach_prob = self.detach_prob
        features = F.normalize(features, p=2, dim=1)
        if detach_prob:
            probs = probs.detach()
        joint = torch.bmm(probs.unsqueeze(2), features.unsqueeze(1))
        joint = joint.view(features.size(0), -1)
        joint = F.normalize(joint, p=2, dim=1)
        return joint

    def _conditional_mmd(self, f_s, f_t, prob_s, prob_t):
        joint_s = self._joint_feature(f_s, prob_s, detach_prob=self.detach_prob)
        joint_t = self._joint_feature(f_t, prob_t, detach_prob=self.detach_prob)
        return self.cda_mkmmd(joint_s, joint_t)

    def _domain_adversarial_loss(self, cur_src_idx, f_s, f_t, source_labels, prob_t, grl_coeff):
        device = f_s.device
        prob_s_onehot = F.one_hot(source_labels, num_classes=self.num_classes).float().to(device)

        if self.adv_conf_thresh > 0:
            conf_t, _ = torch.max(prob_t.detach(), dim=1)
            mask_t = conf_t >= self.adv_conf_thresh
            if mask_t.sum().item() < 2:
                zero = torch.tensor(0.0, device=device)
                return zero, zero
            f_t_used = f_t[mask_t]
            prob_t_used = prob_t[mask_t]
        else:
            f_t_used = f_t
            prob_t_used = prob_t

        joint_s = self._joint_feature(f_s, prob_s_onehot, detach_prob=True)
        joint_t = self._joint_feature(f_t_used, prob_t_used, detach_prob=self.adv_detach_prob)

        joint = torch.cat([joint_s, joint_t], dim=0)
        domain_labels = torch.cat([
            torch.zeros(joint_s.size(0), dtype=torch.long, device=device),
            torch.ones(joint_t.size(0), dtype=torch.long, device=device)
        ], dim=0)

        domain_logits = self.Ds[cur_src_idx](grad_reverse(joint, coeff=grl_coeff))
        per_sample_loss = F.cross_entropy(domain_logits, domain_labels, reduction='none')

        if self.adv_use_entropy_weight:
            w_s = torch.ones(joint_s.size(0), device=device)
            w_t = self._entropy_weight(prob_t_used)
            sample_weights = torch.cat([w_s, w_t], dim=0)
            loss_adv = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + self.entropy_eps)
        else:
            loss_adv = per_sample_loss.mean()

        with torch.no_grad():
            domain_pred = domain_logits.argmax(dim=1)
            domain_acc = torch.eq(domain_pred, domain_labels).float().mean()
        return loss_adv, domain_acc

    # ------------------------------------------------------------------
    # Reliability weighting: global RWCA + class-wise CW-RWCA
    # ------------------------------------------------------------------
    def _standardize(self, x):
        if x.numel() <= 1:
            return x * 0.0
        return (x - x.mean()) / (x.std(unbiased=False) + self.entropy_eps)

    def _standardize_by_class(self, x):
        """
        x: [num_source, num_classes]
        对每个类别 c，沿源域维度做标准化，让每个类别独立选择可靠源域。
        """
        if x.size(0) <= 1:
            return x * 0.0
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, unbiased=False, keepdim=True)
        return (x - mean) / (std + self.entropy_eps)

    def _source_reliability_weights(self, mmd_losses, ent_losses):
        """
        Original RWCA global source-domain reliability weights.
        Kept as a fallback / comparison helper.
        """
        dist = torch.stack([x.detach() if self.rw_detach_weights else x for x in mmd_losses]).float()
        ent = torch.stack([x.detach() if self.rw_detach_weights else x for x in ent_losses]).float()

        dist_z = self._standardize(dist)
        ent_z = self._standardize(ent)
        score = -(self.rw_mmd_weight * dist_z + self.rw_ent_weight * ent_z)
        weights = torch.softmax(score / max(self.rw_tau, self.entropy_eps), dim=0)

        if self.rw_detach_weights:
            weights = weights.detach()
        return weights

    def _normalize_class_source_weights(self, class_weights):
        """
        class_weights: [K, C], K=num_source, C=num_classes.
        Normalize over source dimension for every class, so sum_k w[k,c] = 1.
        """
        class_weights = torch.clamp(class_weights, min=self.entropy_eps)
        class_weights = class_weights / (class_weights.sum(dim=0, keepdim=True) + self.entropy_eps)
        return class_weights

    def _source_weights_from_class_weights(self, class_weights):
        """
        Convert class-wise source weights [K,C] to global source weights [K]
        for scalar losses/logging. This is only an average prior; final prediction
        still uses class-wise weights.
        """
        src_weights = class_weights.mean(dim=1)
        src_weights = src_weights / (src_weights.sum() + self.entropy_eps)
        return src_weights

    def _get_cw_alpha(self):
        """
        V2 schedule: use global RWCA only in early epochs, then gradually inject
        class-wise correction. self._cur_epoch is set in _train_one_epoch().
        """
        epoch = int(getattr(self, '_cur_epoch', 1))
        if epoch <= self.cw_warmup_epochs:
            return 0.0
        if self.cw_alpha_ramp_epochs <= 0:
            return self.cw_alpha
        progress = (epoch - self.cw_warmup_epochs) / float(self.cw_alpha_ramp_epochs)
        progress = max(0.0, min(1.0, progress))
        return self.cw_alpha * progress

    def _global_guided_class_weights(self, global_weights, class_weights, alpha):
        """
        Combine original global RWCA prior and class-wise CW-RWCA weights.

        global_weights: [K]
        class_weights : [K,C]
        alpha=0      -> pure global RWCA
        alpha=1      -> pure class-wise RWCA
        alpha in (0,1) -> global-guided class-wise correction
        """
        global_weights = global_weights.to(class_weights.device).float()
        global_weights = torch.clamp(global_weights, min=self.entropy_eps)
        global_weights = global_weights / (global_weights.sum() + self.entropy_eps)

        class_weights = self._normalize_class_source_weights(class_weights.float())
        global_mat = global_weights.view(-1, 1).repeat(1, self.num_classes)

        final_weights = (1.0 - float(alpha)) * global_mat + float(alpha) * class_weights
        final_weights = self._normalize_class_source_weights(final_weights)

        if self.rw_detach_weights:
            final_weights = final_weights.detach()
        return final_weights

    def _class_source_reliability_weights(self, f_s_list, f_t_list, labels_s_list, probs_t_list,
                                          mmd_losses, ent_losses):
        """
        CW-RWCA: compute class-wise source reliability weights w_{s,c}.

        For each source s and class c:
        - Build source class prototype from labeled source features.
        - Build target pseudo class prototype using target soft probabilities.
        - Smaller prototype distance + lower class-conditioned entropy => higher weight.

        Return:
            class_weights: [num_source, num_classes], normalized over sources per class.
        """
        device = f_t_list[0].device
        dist_mat = torch.zeros(self.num_source, self.num_classes, device=device)
        ent_mat = torch.zeros(self.num_source, self.num_classes, device=device)

        for k in range(self.num_source):
            f_s = f_s_list[k]
            f_t = f_t_list[k]
            labels_s = labels_s_list[k]
            probs_t = torch.clamp(probs_t_list[k], min=self.entropy_eps, max=1.0)
            probs_t = probs_t / (probs_t.sum(dim=1, keepdim=True) + self.entropy_eps)

            if self.rw_detach_weights:
                f_s = f_s.detach()
                f_t = f_t.detach()
                probs_t = probs_t.detach()
                fallback_dist = mmd_losses[k].detach().float()
                fallback_ent = ent_losses[k].detach().float()
            else:
                fallback_dist = mmd_losses[k].float()
                fallback_ent = ent_losses[k].float()

            ent_vec = self._entropy_vector(probs_t)

            for c in range(self.num_classes):
                mask_s = labels_s == c
                n_s = int(mask_s.sum().item())
                wt_raw = probs_t[:, c]
                wt_sum = wt_raw.sum()

                if n_s >= self.clmmd_min_source and wt_sum.detach().item() >= self.clmmd_min_target_weight:
                    proto_s = f_s[mask_s].mean(dim=0)
                    wt = wt_raw / (wt_sum + self.entropy_eps)
                    proto_t = (f_t * wt.view(-1, 1)).sum(dim=0)

                    proto_s = F.normalize(proto_s, p=2, dim=0)
                    proto_t = F.normalize(proto_t, p=2, dim=0)

                    # Prototype distance. Lower means source class c is closer to target class c.
                    dist_c = torch.mean((proto_s - proto_t) ** 2)

                    # Class-conditioned target entropy. Lower means this source classifier is more confident.
                    ent_c = (ent_vec * wt).sum()
                else:
                    # If this class is absent/too weak in the current batch, fall back to global score
                    # with a small penalty to avoid over-trusting unreliable class estimates.
                    dist_c = fallback_dist + 1.0
                    ent_c = fallback_ent + 1.0

                dist_mat[k, c] = dist_c
                ent_mat[k, c] = ent_c

        dist_z = self._standardize_by_class(dist_mat)
        ent_z = self._standardize_by_class(ent_mat)
        score = -(self.rw_mmd_weight * dist_z + self.rw_ent_weight * ent_z)
        class_weights = torch.softmax(score / max(self.rw_tau, self.entropy_eps), dim=0)
        class_weights = self._normalize_class_source_weights(class_weights)

        if self.rw_detach_weights:
            class_weights = class_weights.detach()
        return class_weights

    def _update_class_source_weight_ema(self, class_weights):
        with torch.no_grad():
            cw = self._normalize_class_source_weights(class_weights.detach())
            self.class_source_weight_ema = (
                self.rw_ema_momentum * self.class_source_weight_ema
                + (1.0 - self.rw_ema_momentum) * cw
            )
            self.class_source_weight_ema = self._normalize_class_source_weights(self.class_source_weight_ema)
            self.source_weight_ema = self._source_weights_from_class_weights(self.class_source_weight_ema)

    def _update_source_weight_ema(self, weights):
        # Kept for old global RWCA compatibility. CW-RWCA training uses
        # _update_class_source_weight_ema instead.
        with torch.no_grad():
            w = weights.detach()
            self.source_weight_ema = (
                self.rw_ema_momentum * self.source_weight_ema
                + (1.0 - self.rw_ema_momentum) * w
            )
            self.source_weight_ema = self.source_weight_ema / (self.source_weight_ema.sum() + self.entropy_eps)
            self.class_source_weight_ema = self.source_weight_ema.view(-1, 1).repeat(1, self.num_classes)
            self.class_source_weight_ema = self._normalize_class_source_weights(self.class_source_weight_ema)

    def _weighted_fusion(self, probs_list, weights):
        # Original global weighted fusion. Kept as fallback.
        probs_stack = torch.stack(probs_list, dim=0)  # [K, B, C]
        fused_prob = (weights.view(-1, 1, 1) * probs_stack).sum(dim=0)
        fused_prob = fused_prob / (fused_prob.sum(dim=1, keepdim=True) + self.entropy_eps)
        return fused_prob

    def _class_weighted_fusion(self, probs_list, class_weights):
        """
        CW-RWCA prediction fusion.
        probs_list: list of [B,C]
        class_weights: [K,C]
        fused[:, c] = sum_k w[k,c] * p_k[:,c]
        """
        probs_stack = torch.stack(probs_list, dim=0)  # [K, B, C]
        cw = self._normalize_class_source_weights(class_weights.to(probs_stack.device))
        fused_prob = (cw.view(self.num_source, 1, self.num_classes) * probs_stack).sum(dim=0)
        fused_prob = fused_prob / (fused_prob.sum(dim=1, keepdim=True) + self.entropy_eps)
        return fused_prob

    def _eval_class_weighted_fusion(self, probs_list):
        """
        Eval-time class-wise prediction fusion.
        Base prior is class_source_weight_ema [K,C]. Optionally combine it with
        current sample entropy, producing weights [K,B,C].
        """
        probs_stack = torch.stack(probs_list, dim=0)  # [K, B, C]
        base = self.class_source_weight_ema.to(probs_stack.device)
        base = self._normalize_class_source_weights(base)

        if not self.rw_eval_use_entropy:
            fused_prob = (base.view(self.num_source, 1, self.num_classes) * probs_stack).sum(dim=0)
            fused_prob = fused_prob / (fused_prob.sum(dim=1, keepdim=True) + self.entropy_eps)
            return fused_prob, base

        ent = -(torch.clamp(probs_stack, min=self.entropy_eps) *
                torch.log(torch.clamp(probs_stack, min=self.entropy_eps))).sum(dim=2)  # [K,B]
        log_prior = torch.log(base.clamp_min(self.entropy_eps)).view(self.num_source, 1, self.num_classes)
        score = log_prior - ent.unsqueeze(2) / max(self.rw_eval_tau, self.entropy_eps)  # [K,B,C]
        weights = torch.softmax(score, dim=0)

        fused_prob = (weights * probs_stack).sum(dim=0)
        fused_prob = fused_prob / (fused_prob.sum(dim=1, keepdim=True) + self.entropy_eps)
        return fused_prob, weights

    # ------------------------------------------------------------------
    # Class-wise LMMD / LJMMD-like loss
    # ------------------------------------------------------------------
    def _gaussian_kernel_matrix(self, x, y):
        """Multi-kernel Gaussian kernel matrix between x and y."""
        n_x = x.size(0)
        n_y = y.size(0)
        total = torch.cat([x, y], dim=0)
        l2 = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(dim=2)
        with torch.no_grad():
            base = l2.detach().mean().clamp_min(self.entropy_eps)
        kernel_sum = 0.0
        mid = self.clmmd_kernel_num // 2
        for i in range(self.clmmd_kernel_num):
            sigma = base * (self.clmmd_kernel_mul ** (i - mid))
            kernel_sum = kernel_sum + torch.exp(-l2 / (2.0 * sigma))
        return kernel_sum[:n_x, :n_x], kernel_sum[:n_x, n_x:], kernel_sum[n_x:, n_x:]

    def _classwise_lmmd_per_class(self, f_s, f_t, labels_s, probs_t):
        """
        Return per-class soft LMMD losses.

        loss_vec[c] is the LMMD loss for class c.
        valid_vec[c] = 1 means this class has enough source samples and enough
        target soft mass in the current batch.
        """
        device = f_s.device
        loss_vec = torch.zeros(self.num_classes, device=device)
        valid_vec = torch.zeros(self.num_classes, device=device)

        probs_t = torch.clamp(probs_t, min=self.entropy_eps, max=1.0)
        probs_t = probs_t / (probs_t.sum(dim=1, keepdim=True) + self.entropy_eps)

        for c in range(self.num_classes):
            mask_s = labels_s == c
            n_s = int(mask_s.sum().item())
            if n_s < self.clmmd_min_source:
                continue

            wt_raw = probs_t[:, c]
            wt_sum = wt_raw.sum()
            if wt_sum.detach().item() < self.clmmd_min_target_weight:
                continue

            xs = f_s[mask_s]
            xt = f_t
            k_ss, k_st, k_tt = self._gaussian_kernel_matrix(xs, xt)

            ws = torch.ones(n_s, device=device) / float(n_s)
            wt = wt_raw / (wt_sum + self.entropy_eps)

            loss_c = (
                torch.sum(ws.view(-1, 1) * ws.view(1, -1) * k_ss)
                + torch.sum(wt.view(-1, 1) * wt.view(1, -1) * k_tt)
                - 2.0 * torch.sum(ws.view(-1, 1) * wt.view(1, -1) * k_st)
            )
            loss_vec[c] = loss_c
            valid_vec[c] = 1.0

        return loss_vec, valid_vec

    def _classwise_lmmd(self, f_s, f_t, labels_s, probs_t):
        """Backward-compatible scalar CLMMD."""
        loss_vec, valid_vec = self._classwise_lmmd_per_class(f_s, f_t, labels_s, probs_t)
        denom = valid_vec.sum()
        if denom.detach().item() > 0:
            return (loss_vec * valid_vec).sum() / (denom + self.entropy_eps)
        return torch.tensor(0.0, device=f_s.device)

    # ------------------------------------------------------------------
    # Train / Eval
    # ------------------------------------------------------------------
    def train(self):
        """Train with the V2 epoch state and configurable checkpoint metric."""
        args = self.args
        best_score = float('-inf')
        best_acc = 0.0
        best_epoch = 0
        no_improve_epochs = 0
        early_stop_patience = max(0, int(getattr(args, 'early_stop_patience', 0)))
        early_stop_min_epoch = max(1, int(getattr(args, 'early_stop_min_epoch', 1)))
        early_stop_min_delta = max(0.0, float(getattr(args, 'early_stop_min_delta', 0.0)))

        for epoch in range(1, args.max_epoch + 1):
            self._cur_epoch = epoch
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)

            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            from collections import defaultdict
            epoch_acc = defaultdict(float)
            self._set_to_train()
            epoch_loss = defaultdict(float)
            self.tradeoff = self._get_tradeoff(args.tradeoff, epoch)

            epoch_acc, epoch_loss = self._train_one_epoch(epoch_acc, epoch_loss)
            self._log_epoch_info(epoch_loss, epoch_acc, self.num_iter)

            if bool(getattr(args, 'eval_each_epoch', True)):
                new_acc = self.test()
                if bool(getattr(args, 'select_best_on_target', True)):
                    new_score = self._checkpoint_selection_score()
                    metric_name = str(getattr(args, 'best_metric', 'accuracy'))
                    improved = (
                        best_epoch == 0
                        or new_score > best_score + early_stop_min_delta
                    )
                    if improved:
                        best_score = new_score
                        best_acc = new_acc
                        best_epoch = epoch
                        no_improve_epochs = 0
                        if getattr(args, 'save', False) and getattr(args, 'save_best', True):
                            if hasattr(self, 'save_best_model'):
                                self.save_best_model()
                            else:
                                self.save_model()
                            logging.info(
                                'Best model updated at epoch {}, target-test-acc {:.4f}, '
                                '{} score {:.4f}'.format(
                                    best_epoch, best_acc, metric_name, best_score
                                )
                            )
                    else:
                        no_improve_epochs += 1
                    logging.info(
                        'The best model epoch {}, target-test-acc {:.4f}, '
                        '{} score {:.4f}'.format(
                            best_epoch, best_acc, metric_name, best_score
                        )
                    )
                else:
                    logging.info(
                        'Target test was reported but not used for checkpoint selection.'
                    )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if (
                early_stop_patience > 0
                and epoch >= early_stop_min_epoch
                and no_improve_epochs >= early_stop_patience
            ):
                logging.info(
                    'Early stopping at epoch {}: no {} score improvement larger than {:.6f} '
                    'for {} consecutive epochs. Best epoch={} score={:.4f}.'.format(
                        epoch,
                        str(getattr(args, 'best_metric', 'accuracy')),
                        early_stop_min_delta,
                        no_improve_epochs,
                        best_epoch,
                        best_score,
                    )
                )
                break

        if not bool(getattr(args, 'eval_each_epoch', True)):
            logging.info(
                'Training finished; evaluating the held-out target set once (strict mode).'
            )
            self.test()

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        self._cur_epoch = int(getattr(self, '_cur_epoch', 1))
        weight_sum = torch.zeros(self.num_source, device=self.device)
        global_weight_sum = torch.zeros(self.num_source, device=self.device)
        raw_class_weight_sum = torch.zeros(self.num_source, self.num_classes, device=self.device)
        class_weight_sum = torch.zeros(self.num_source, self.num_classes, device=self.device)
        alpha_sum = 0.0

        for _ in tqdm(range(self.num_iter), ascii=True):
            target_data, _ = self._get_next_batch('train')

            source_data_list = []
            source_label_list = []
            for k in range(self.num_source):
                source_data_k, source_labels_k = self._get_next_batch(self.src[k], return_actual=True)
                source_labels_k = self._get_train_label(source_labels_k, label_set=self.src_labels_flat)
                source_data_list.append(source_data_k)
                source_label_list.append(source_labels_k)

            self.optimizer.zero_grad()

            # Shared backbone in one forward pass
            data = torch.cat(source_data_list + [target_data], dim=0)
            g_all = self.G(data)
            split_sizes = [x.size(0) for x in source_data_list] + [target_data.size(0)]
            g_s_list = list(torch.split(g_all, split_sizes, dim=0)[:-1])
            g_t = torch.split(g_all, split_sizes, dim=0)[-1]

            loss_cls_vec_list = []
            loss_cls_list = []
            loss_mmd_list = []
            loss_cda_list = []
            loss_adv_list = []
            loss_clmmd_vec_list = []
            clmmd_valid_list = []
            loss_clmmd_scalar_list = []
            ent_list = []
            domain_acc_list = []
            probs_t_all = []
            f_s_all = []
            f_t_all = []

            adv_tradeoff = self.tradeoff[2] if len(self.tradeoff) > 2 else 1.0
            grl_coeff = self.lambda_grl * adv_tradeoff

            for k in range(self.num_source):
                f_s = self.Fs[k](g_s_list[k])
                f_t = self.Fs[k](g_t)
                f_s_all.append(f_s)
                f_t_all.append(f_t)

                y_s = self.Cs[k](f_s)
                y_t = self.Cs[k](f_t)
                p_s = F.softmax(y_s, dim=1)
                p_t = F.softmax(y_t, dim=1)
                probs_t_all.append(p_t)

                labels_s = source_label_list[k]

                loss_cls_vec_k = F.cross_entropy(y_s, labels_s, reduction='none')
                loss_cls_k = loss_cls_vec_k.mean()
                loss_mmd_k = self.mkmmd(f_s, f_t)
                loss_cda_k = self._conditional_mmd(f_s, f_t, p_s, p_t)
                loss_clmmd_vec_k, clmmd_valid_k = self._classwise_lmmd_per_class(f_s, f_t, labels_s, p_t)
                valid_den = clmmd_valid_k.sum()
                if valid_den.detach().item() > 0:
                    loss_clmmd_k = (loss_clmmd_vec_k * clmmd_valid_k).sum() / (valid_den + self.entropy_eps)
                else:
                    loss_clmmd_k = torch.tensor(0.0, device=self.device)
                loss_adv_k, domain_acc_k = self._domain_adversarial_loss(
                    cur_src_idx=k,
                    f_s=f_s,
                    f_t=f_t,
                    source_labels=labels_s,
                    prob_t=p_t,
                    grl_coeff=grl_coeff
                )
                ent_k = self._entropy_scalar(p_t)

                loss_cls_vec_list.append(loss_cls_vec_k)
                loss_cls_list.append(loss_cls_k)
                loss_mmd_list.append(loss_mmd_k)
                loss_cda_list.append(loss_cda_k)
                loss_adv_list.append(loss_adv_k)
                loss_clmmd_vec_list.append(loss_clmmd_vec_k)
                clmmd_valid_list.append(clmmd_valid_k)
                loss_clmmd_scalar_list.append(loss_clmmd_k)
                ent_list.append(ent_k)
                domain_acc_list.append(domain_acc_k if torch.is_tensor(domain_acc_k) else torch.tensor(domain_acc_k, device=self.device))

                epoch_acc['Source Data'] += self._get_accuracy(y_s, labels_s) / float(self.num_source)

            # V2: original global RWCA prior + class-wise correction.
            # global_src_weights protects training from unstable pseudo labels in hard target domains.
            global_src_weights = self._source_reliability_weights(loss_mmd_list, ent_list)
            raw_class_src_weights = self._class_source_reliability_weights(
                f_s_all, f_t_all, source_label_list, probs_t_all, loss_mmd_list, ent_list
            )
            cw_alpha_now = self._get_cw_alpha()
            class_src_weights = self._global_guided_class_weights(
                global_src_weights, raw_class_src_weights, alpha=cw_alpha_now
            )
            src_weights = self._source_weights_from_class_weights(class_src_weights)

            self._update_class_source_weight_ema(class_src_weights)
            global_weight_sum += global_src_weights.detach()
            raw_class_weight_sum += raw_class_src_weights.detach()
            weight_sum += src_weights.detach()
            class_weight_sum += class_src_weights.detach()
            alpha_sum += float(cw_alpha_now)

            # Class-wise weighted prediction fusion for target domain
            probs_t_fused = self._class_weighted_fusion(probs_t_all, class_src_weights)

            # CW-weighted source classification loss: samples of class c use w[k,c]
            cls_num = torch.tensor(0.0, device=self.device)
            cls_den = torch.tensor(0.0, device=self.device)
            for k in range(self.num_source):
                sample_w = class_src_weights[k, source_label_list[k]]
                cls_num = cls_num + (loss_cls_vec_list[k] * sample_w).sum()
                cls_den = cls_den + sample_w.sum()
            loss_cls = cls_num / (cls_den + self.entropy_eps)

            # Scalar source-level losses use class-averaged source weights
            loss_mmd = sum(src_weights[k] * loss_mmd_list[k] for k in range(self.num_source))
            loss_cda = sum(src_weights[k] * loss_cda_list[k] for k in range(self.num_source))
            loss_adv = sum(src_weights[k] * loss_adv_list[k] for k in range(self.num_source))

            # CW-weighted CLMMD: each source-class pair has its own reliability.
            clmmd_num = torch.tensor(0.0, device=self.device)
            clmmd_den = torch.tensor(0.0, device=self.device)
            for k in range(self.num_source):
                valid = clmmd_valid_list[k]
                clmmd_num = clmmd_num + (class_src_weights[k] * valid * loss_clmmd_vec_list[k]).sum()
                clmmd_den = clmmd_den + (class_src_weights[k] * valid).sum()
            if clmmd_den.detach().item() > 0:
                loss_clmmd = clmmd_num / (clmmd_den + self.entropy_eps)
            else:
                loss_clmmd = torch.tensor(0.0, device=self.device)

            # CDD / classifier consistency: class-wise source weights make difficult classes choose source more carefully.
            loss_l1 = torch.tensor(0.0, device=self.device)
            for k in range(self.num_source):
                abs_diff = torch.abs(probs_t_all[k] - probs_t_fused.detach())
                loss_l1 = loss_l1 + (abs_diff * class_src_weights[k].view(1, -1)).sum(dim=1).mean() / float(self.num_classes)

            loss_ent = self._target_entropy(probs_t_fused)
            domain_acc = sum(src_weights[k] * domain_acc_list[k] for k in range(self.num_source))

            new_tradeoff = adv_tradeoff
            loss = (
                loss_cls
                + self.tradeoff[0] * loss_mmd
                + self.tradeoff[1] * loss_l1
                + new_tradeoff * self.lambda_cda * loss_cda
                + new_tradeoff * self.lambda_ent * loss_ent
                + new_tradeoff * self.lambda_adv * loss_adv
                + new_tradeoff * self.lambda_clmmd * loss_clmmd
            )

            # Logging losses
            epoch_acc['Domain Data'] += domain_acc.detach().item()
            epoch_loss['Source Classifier'] += loss_cls.detach()
            epoch_loss['MMD'] += loss_mmd.detach()
            epoch_loss['CDD/L1'] += loss_l1.detach()
            epoch_loss['CDA MMD'] += loss_cda.detach()
            epoch_loss['CLMMD'] += loss_clmmd.detach()
            epoch_loss['Target Entropy'] += loss_ent.detach()
            epoch_loss['CDAN Domain'] += loss_adv.detach()
            epoch_loss['CDA Weighted'] += (new_tradeoff * self.lambda_cda * loss_cda).detach()
            epoch_loss['CLMMD Weighted'] += (new_tradeoff * self.lambda_clmmd * loss_clmmd).detach()
            epoch_loss['Entropy Weighted'] += (new_tradeoff * self.lambda_ent * loss_ent).detach()
            epoch_loss['CDAN Weighted'] += (new_tradeoff * self.lambda_adv * loss_adv).detach()

            epoch_loss['CW Alpha'] += torch.tensor(float(cw_alpha_now), device=self.device)
            for k in range(self.num_source):
                epoch_loss[f'Global Prior src{k}'] += global_src_weights[k].detach()
                epoch_loss[f'RW Weight src{k}'] += src_weights[k].detach()
            for c in self._cw_log_classes:
                for k in range(self.num_source):
                    epoch_loss[f'Raw CW Weight c{c} src{k}'] += raw_class_src_weights[k, c].detach()
                    epoch_loss[f'CW Weight c{c} src{k}'] += class_src_weights[k, c].detach()

            loss.backward()
            self.optimizer.step()

        denom_iter = max(float(self.num_iter), 1.0)
        avg_weights = (weight_sum / denom_iter).detach().cpu().numpy()
        avg_global_weights = (global_weight_sum / denom_iter).detach().cpu().numpy()
        avg_raw_class_weights = (raw_class_weight_sum / denom_iter).detach().cpu().numpy()
        avg_class_weights = (class_weight_sum / denom_iter).detach().cpu().numpy()
        ema_weights = self.source_weight_ema.detach().cpu().numpy()
        ema_class_weights = self.class_source_weight_ema.detach().cpu().numpy()
        avg_alpha = alpha_sum / denom_iter

        logging.info('CW-RWCA-V2 alpha average: {:.4f} | warmup_epochs={} | alpha_max={:.4f} | ramp_epochs={}'.format(
            avg_alpha, self.cw_warmup_epochs, self.cw_alpha, self.cw_alpha_ramp_epochs
        ))
        logging.info('CW-RWCA-V2 global prior source weights: {}'.format(
            ', '.join(['src{}={:.4f}'.format(i, avg_global_weights[i]) for i in range(self.num_source)])
        ))
        logging.info('CW-RWCA-V2 average guided source weights: {}'.format(
            ', '.join(['src{}={:.4f}'.format(i, avg_weights[i]) for i in range(self.num_source)])
        ))
        logging.info('CW-RWCA-V2 EMA guided source weights: {}'.format(
            ', '.join(['src{}={:.4f}'.format(i, ema_weights[i]) for i in range(self.num_source)])
        ))
        for c in self._cw_log_classes:
            logging.info('CW-RWCA-V2 raw train class-{} source weights: {}'.format(
                c, ', '.join(['src{}={:.4f}'.format(i, avg_raw_class_weights[i, c]) for i in range(self.num_source)])
            ))
            logging.info('CW-RWCA-V2 guided train class-{} source weights: {}'.format(
                c, ', '.join(['src{}={:.4f}'.format(i, avg_class_weights[i, c]) for i in range(self.num_source)])
            ))
            logging.info('CW-RWCA-V2 EMA guided class-{} source weights: {}'.format(
                c, ', '.join(['src{}={:.4f}'.format(i, ema_class_weights[i, c]) for i in range(self.num_source)])
            ))

        if hasattr(self.G, 'get_gate'):
            logging.info('BiMamba-Att residual gate: {:.6f}'.format(self.G.get_gate().detach().item()))
            if hasattr(self.G, 'gate_max'):
                logging.info('Max BiMamba-Att residual gate: {:.6f}'.format(self.G.gate_max))

        logging.info(
            'MFSAN-CDAN-BiMamba-CW-RWCA-V2 active: lambda_adv={:.6f}, lambda_grl={:.6f}, lambda_cda={:.6f}, lambda_clmmd={:.6f}, lambda_ent={:.6f}'.format(
                self.lambda_adv, self.lambda_grl, self.lambda_cda, self.lambda_clmmd, self.lambda_ent
            )
        )

        return epoch_acc, epoch_loss

    def _eval(self, data, actual_labels, correct, total):
        feat_tgt = self.G(data)
        probs_tgt = [
            F.softmax(self.Cs[i](self.Fs[i](feat_tgt)), dim=1)
            for i in range(self.num_source)
        ]

        fused_prob, weights = self._eval_class_weighted_fusion(probs_tgt)
        pred = fused_prob.argmax(dim=1)
        actual_pred = self._get_actual_label(pred, label_set=self.src_labels_flat)

        if hasattr(self, '_eval_pred_list') and hasattr(self, '_eval_label_list'):
            self._eval_pred_list.append(actual_pred.detach().cpu())
            self._eval_label_list.append(actual_labels.detach().cpu())

        if not hasattr(self, '_eval_source_weight_sum'):
            self._eval_source_weight_sum = torch.zeros(self.num_source)
            self._eval_source_weight_count = 0

        # weights can be [K,C] when rw_eval_use_entropy=False or [K,B,C] when True.
        if weights.dim() == 3:
            src_weight_mean = weights.detach().mean(dim=(1, 2)).cpu()
        else:
            src_weight_mean = weights.detach().mean(dim=1).cpu()
        self._eval_source_weight_sum += src_weight_mean
        self._eval_source_weight_count += 1

        output = self._get_accuracy(actual_pred, actual_labels, return_acc=False)
        correct['acc'] += output[0]
        total['acc'] += output[1]

        if self.args.da_scenario in ['open-set', 'universal']:
            output = self._get_accuracy(
                actual_pred,
                actual_labels,
                return_acc=False,
                idx=0,
                mode='closed-set'
            )
            correct['Closed-set-acc'] += output[0]
            total['Closed-set-acc'] += output[1]

        return correct, total
