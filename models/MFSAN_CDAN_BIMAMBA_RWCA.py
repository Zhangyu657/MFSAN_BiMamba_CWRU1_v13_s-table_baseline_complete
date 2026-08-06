# -*- coding: utf-8 -*-
"""
MFSAN-CDAN-BiMamba-RWCA

RWCA = Reliability-Weighted Class-wise Alignment

在当前 MFSAN-CDAN-BiMamba-SmallGate 主模型基础上，进一步加入：
1. 源域可靠性权重 w_i：由 source-target MK-MMD 距离 + 目标域预测熵共同决定；
2. 加权多源对齐：Σ w_i L_MMD_i、Σ w_i L_CDA_i、Σ w_i L_adv_i；
3. 加权多源分类：Σ w_i L_cls_i；
4. 加权分类器一致性 / CDD：Σ w_i |p_i - p_fused|；
5. 类别级子域对齐：L_CLMMD，用 source hard label + target soft probability 做 class-wise MK-MMD；
6. 最终预测：Σ w_i p_i，其中 eval 阶段使用训练得到的 source reliability EMA + 当前目标熵。

推荐模型名：MFSAN_CDAN_BIMAMBA_RWCA
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
            raise ValueError('MFSAN_CDAN_BIMAMBA_RWCA is designed for --train_mode multi_source.')

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

        logging.info('Using model: MFSAN_CDAN_BIMAMBA_RWCA')
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

        self.entropy_eps = 1e-5
        self.source_weight_ema = torch.ones(self.num_source, device=self.device) / float(self.num_source)
        self._last_train_source_weights = None
        self._last_val_source_weights = None

        logging.info('MFSAN-CDAN-BiMamba-RWCA lambda_adv: {:.6f}'.format(self.lambda_adv))
        logging.info('MFSAN-CDAN-BiMamba-RWCA lambda_grl: {:.6f}'.format(self.lambda_grl))
        logging.info('MFSAN-CDAN-BiMamba-RWCA lambda_cda: {:.6f}'.format(self.lambda_cda))
        logging.info('MFSAN-CDAN-BiMamba-RWCA lambda_ent: {:.6f}'.format(self.lambda_ent))
        logging.info('MFSAN-CDAN-BiMamba-RWCA lambda_clmmd: {:.6f}'.format(self.lambda_clmmd))
        logging.info('MFSAN-CDAN-BiMamba-RWCA rw_tau: {:.6f}'.format(self.rw_tau))
        logging.info('MFSAN-CDAN-BiMamba-RWCA rw_mmd_weight: {:.6f}'.format(self.rw_mmd_weight))
        logging.info('MFSAN-CDAN-BiMamba-RWCA rw_ent_weight: {:.6f}'.format(self.rw_ent_weight))
        logging.info('MFSAN-CDAN-BiMamba-RWCA rw_detach_weights: {}'.format(self.rw_detach_weights))
        logging.info('MFSAN-CDAN-BiMamba-RWCA joint feature dim: {} x {} = {}'.format(
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
            'lambda_adv': self.lambda_adv,
            'lambda_grl': self.lambda_grl,
            'lambda_cda': self.lambda_cda,
            'lambda_ent': self.lambda_ent,
            'lambda_clmmd': self.lambda_clmmd,
            'rw_tau': self.rw_tau,
            'rw_mmd_weight': self.rw_mmd_weight,
            'rw_ent_weight': self.rw_ent_weight,
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
    # Reliability weighting
    # ------------------------------------------------------------------
    def _standardize(self, x):
        if x.numel() <= 1:
            return x * 0.0
        return (x - x.mean()) / (x.std(unbiased=False) + self.entropy_eps)

    def _source_reliability_weights(self, mmd_losses, ent_losses):
        """
        Compute source-domain reliability weights from source-target MK-MMD distances and
        target prediction entropy.

        Lower MK-MMD + lower entropy => larger weight.
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

    def _update_source_weight_ema(self, weights):
        with torch.no_grad():
            w = weights.detach()
            self.source_weight_ema = (
                self.rw_ema_momentum * self.source_weight_ema
                + (1.0 - self.rw_ema_momentum) * w
            )
            self.source_weight_ema = self.source_weight_ema / (self.source_weight_ema.sum() + self.entropy_eps)

    def _weighted_fusion(self, probs_list, weights):
        probs_stack = torch.stack(probs_list, dim=0)  # [K, B, C]
        fused_prob = (weights.view(-1, 1, 1) * probs_stack).sum(dim=0)
        fused_prob = fused_prob / (fused_prob.sum(dim=1, keepdim=True) + self.entropy_eps)
        return fused_prob

    def _eval_weighted_fusion(self, probs_list):
        """
        Eval-time prediction fusion.
        Use training EMA source reliability as global source similarity prior.
        Optionally combine it with current target sample entropy to make per-sample weights.
        """
        probs_stack = torch.stack(probs_list, dim=0)  # [K, B, C]
        base = self.source_weight_ema.to(probs_stack.device)
        base = base / (base.sum() + self.entropy_eps)

        if not self.rw_eval_use_entropy:
            weights = base.view(-1, 1).expand(-1, probs_stack.size(1))
        else:
            ent = -(torch.clamp(probs_stack, min=self.entropy_eps) *
                    torch.log(torch.clamp(probs_stack, min=self.entropy_eps))).sum(dim=2)  # [K, B]
            log_prior = torch.log(base.clamp_min(self.entropy_eps)).view(-1, 1)
            score = log_prior - ent / max(self.rw_eval_tau, self.entropy_eps)
            weights = torch.softmax(score, dim=0)  # [K, B]

        fused_prob = (weights.unsqueeze(2) * probs_stack).sum(dim=0)
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

    def _classwise_lmmd(self, f_s, f_t, labels_s, probs_t):
        """
        Soft class-wise MK-MMD.
        Source uses hard labels; target uses soft class probabilities.
        """
        device = f_s.device
        total_loss = torch.tensor(0.0, device=device)
        valid_classes = 0

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
            total_loss = total_loss + loss_c
            valid_classes += 1

        if valid_classes > 0:
            total_loss = total_loss / float(valid_classes)
        return total_loss

    # ------------------------------------------------------------------
    # Train / Eval
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch_acc, epoch_loss):
        weight_sum = torch.zeros(self.num_source, device=self.device)

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

            loss_cls_list = []
            loss_mmd_list = []
            loss_cda_list = []
            loss_adv_list = []
            loss_clmmd_list = []
            ent_list = []
            domain_acc_list = []
            probs_t_all = []

            adv_tradeoff = self.tradeoff[2] if len(self.tradeoff) > 2 else 1.0
            grl_coeff = self.lambda_grl * adv_tradeoff

            for k in range(self.num_source):
                f_s = self.Fs[k](g_s_list[k])
                f_t = self.Fs[k](g_t)

                y_s = self.Cs[k](f_s)
                y_t = self.Cs[k](f_t)
                p_s = F.softmax(y_s, dim=1)
                p_t = F.softmax(y_t, dim=1)
                probs_t_all.append(p_t)

                labels_s = source_label_list[k]

                loss_cls_k = F.cross_entropy(y_s, labels_s)
                loss_mmd_k = self.mkmmd(f_s, f_t)
                loss_cda_k = self._conditional_mmd(f_s, f_t, p_s, p_t)
                loss_clmmd_k = self._classwise_lmmd(f_s, f_t, labels_s, p_t)
                loss_adv_k, domain_acc_k = self._domain_adversarial_loss(
                    cur_src_idx=k,
                    f_s=f_s,
                    f_t=f_t,
                    source_labels=labels_s,
                    prob_t=p_t,
                    grl_coeff=grl_coeff
                )
                ent_k = self._entropy_scalar(p_t)

                loss_cls_list.append(loss_cls_k)
                loss_mmd_list.append(loss_mmd_k)
                loss_cda_list.append(loss_cda_k)
                loss_adv_list.append(loss_adv_k)
                loss_clmmd_list.append(loss_clmmd_k)
                ent_list.append(ent_k)
                domain_acc_list.append(domain_acc_k if torch.is_tensor(domain_acc_k) else torch.tensor(domain_acc_k, device=self.device))

                epoch_acc['Source Data'] += self._get_accuracy(y_s, labels_s) / float(self.num_source)

            # Source reliability weights from source-target MK-MMD + target entropy
            src_weights = self._source_reliability_weights(loss_mmd_list, ent_list)
            self._update_source_weight_ema(src_weights)
            weight_sum += src_weights.detach()

            # Weighted prediction fusion for target domain
            probs_t_fused = self._weighted_fusion(probs_t_all, src_weights)

            # Reliability-weighted losses
            loss_cls = sum(src_weights[k] * loss_cls_list[k] for k in range(self.num_source))
            loss_mmd = sum(src_weights[k] * loss_mmd_list[k] for k in range(self.num_source))
            loss_cda = sum(src_weights[k] * loss_cda_list[k] for k in range(self.num_source))
            loss_adv = sum(src_weights[k] * loss_adv_list[k] for k in range(self.num_source))
            loss_clmmd = sum(src_weights[k] * loss_clmmd_list[k] for k in range(self.num_source))

            # CDD / classifier consistency: each source classifier approaches weighted ensemble
            loss_l1 = sum(
                src_weights[k] * torch.abs(probs_t_all[k] - probs_t_fused.detach()).mean()
                for k in range(self.num_source)
            )

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

            for k in range(self.num_source):
                epoch_loss[f'RW Weight src{k}'] += src_weights[k].detach()

            loss.backward()
            self.optimizer.step()

        avg_weights = (weight_sum / max(float(self.num_iter), 1.0)).detach().cpu().numpy()
        ema_weights = self.source_weight_ema.detach().cpu().numpy()
        logging.info('RWCA average train source weights: {}'.format(
            ', '.join(['src{}={:.4f}'.format(i, avg_weights[i]) for i in range(self.num_source)])
        ))
        logging.info('RWCA EMA source weights: {}'.format(
            ', '.join(['src{}={:.4f}'.format(i, ema_weights[i]) for i in range(self.num_source)])
        ))

        if hasattr(self.G, 'get_gate'):
            logging.info('BiMamba-Att residual gate: {:.6f}'.format(self.G.get_gate().detach().item()))
            if hasattr(self.G, 'gate_max'):
                logging.info('Max BiMamba-Att residual gate: {:.6f}'.format(self.G.gate_max))

        logging.info(
            'MFSAN-CDAN-BiMamba-RWCA active: lambda_adv={:.6f}, lambda_grl={:.6f}, lambda_cda={:.6f}, lambda_clmmd={:.6f}, lambda_ent={:.6f}'.format(
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

        fused_prob, weights = self._eval_weighted_fusion(probs_tgt)
        pred = fused_prob.argmax(dim=1)
        actual_pred = self._get_actual_label(pred, label_set=self.src_labels_flat)

        if hasattr(self, '_eval_pred_list') and hasattr(self, '_eval_label_list'):
            self._eval_pred_list.append(actual_pred.detach().cpu())
            self._eval_label_list.append(actual_labels.detach().cpu())

        if not hasattr(self, '_eval_source_weight_sum'):
            self._eval_source_weight_sum = torch.zeros(self.num_source)
            self._eval_source_weight_count = 0
        self._eval_source_weight_sum += weights.detach().mean(dim=1).cpu()
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
