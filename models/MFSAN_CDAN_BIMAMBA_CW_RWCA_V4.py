# -*- coding: utf-8 -*-
"""
MFSAN-CDAN-BiMamba-CW-RWCA-V4

V4 = V3 + pseudo-label confidence gating.

核心改动：
1. 继承 V3：eval 阶段继续使用 current epoch average guided weights，不再直接使用 EMA；
2. 在 class-wise source reliability 计算中加入目标域伪标签置信度门控；
3. 在 CLMMD / LMMD 子域对齐中加入目标域伪标签置信度门控；
4. 只让高置信目标样本参与类别原型估计和类别级 MMD，减少错误伪标签对类别级对齐的污染。

使用方式：
把本文件放到：
    models/MFSAN_CDAN_BIMAMBA_CW_RWCA_V4.py
然后运行：
    --model_name MFSAN_CDAN_BIMAMBA_CW_RWCA_V4

依赖：
    models/MFSAN_CDAN_BIMAMBA_CW_RWCA_V3.py

默认参数：
    pl_conf_thresh = 0.80
    pl_min_target = 2

说明：
如果 opt.py 没有注册 pl_conf_thresh / pl_min_target，代码会自动使用默认值；
如果你希望通过命令行调参，建议在 opt.py 里增加这两个参数。
"""

import logging

import torch
import torch.nn.functional as F

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V3 import Trainer as V3Trainer


class Trainer(V3Trainer):
    """
    CW-RWCA-V4 Trainer.

    在 V3 基础上只重写两个核心函数：
    1. _class_source_reliability_weights()
       类别级源域可靠性估计时，目标域类别原型只使用高置信伪标签样本。

    2. _classwise_lmmd_per_class()
       CLMMD 子域对齐时，只使用高置信伪标签目标样本。

    这样做的目的：
    - 保留 V3 的 eval no EMA 机制；
    - 减少低置信 / 错误伪标签对类别原型和类别级 MMD 的污染；
    - 让 Class-wise RWCA 在目标域伪标签尚不稳定时更保守。
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # 目标域伪标签置信度门控阈值：
        # 当 max_prob >= pl_conf_thresh 且 pseudo_label == 当前类别 c 时，
        # 目标样本才参与 class c 的原型估计 / LMMD。
        self.pl_conf_thresh = float(getattr(args, 'pl_conf_thresh', 0.80))

        # 每个类别至少需要多少个高置信目标样本才参与该类别的可靠性估计 / CLMMD。
        self.pl_min_target = int(getattr(args, 'pl_min_target', 2))

        # 是否启用伪标签置信度门控。
        # 如果想关闭门控，可以传 --pl_conf_thresh 0.0。
        self.use_pl_conf_gate = self.pl_conf_thresh > 0.0

        logging.info('Using model: MFSAN_CDAN_BIMAMBA_CW_RWCA_V4')
        logging.info(
            'CW-RWCA-V4 pseudo-label confidence gate: enabled={}, threshold={:.4f}, min_target={}'.format(
                self.use_pl_conf_gate,
                self.pl_conf_thresh,
                self.pl_min_target,
            )
        )

    def _checkpoint_dict(self):
        """
        在 V3 checkpoint 基础上额外保存 V4 的伪标签门控参数。
        """
        ckpt = super(Trainer, self)._checkpoint_dict()
        ckpt['v4_pseudo_label_conf_gate'] = True
        ckpt['pl_conf_thresh'] = self.pl_conf_thresh
        ckpt['pl_min_target'] = self.pl_min_target
        return ckpt

    def load_model(self):
        """
        兼容加载 V3 / V4 checkpoint。

        V3 的 load_model 会加载：
            G / Fs / Cs / Ds
            source_weight_ema
            class_source_weight_ema
            class_source_weight_last_epoch

        这里额外尝试恢复 V4 的 pl_conf_thresh / pl_min_target。
        如果 checkpoint 中没有这两个字段，则保留当前 args/default 设置。
        """
        super(Trainer, self).load_model()

        try:
            ckpt = torch.load(self.args.load_path, map_location=self.device)

            if 'pl_conf_thresh' in ckpt:
                self.pl_conf_thresh = float(ckpt['pl_conf_thresh'])
            if 'pl_min_target' in ckpt:
                self.pl_min_target = int(ckpt['pl_min_target'])

            self.use_pl_conf_gate = self.pl_conf_thresh > 0.0

            logging.info(
                'Loaded V4 pseudo-label gate params: enabled={}, threshold={:.4f}, min_target={}'.format(
                    self.use_pl_conf_gate,
                    self.pl_conf_thresh,
                    self.pl_min_target,
                )
            )
        except Exception as e:
            logging.warning(
                'Failed to load V4 pseudo-label gate params from checkpoint; keep current args/default. Error: {}'.format(e)
            )

    def _high_conf_target_mask(self, probs_t, class_id):
        """
        根据目标域预测概率，返回 class_id 对应的高置信伪标签样本 mask。

        Args:
            probs_t:  [B, C]，目标域预测概率。
            class_id: int，当前类别编号。

        Returns:
            mask_t: [B] bool tensor。
        """
        if not self.use_pl_conf_gate:
            # 不启用门控时，退回到 soft assignment：所有目标样本都可参与。
            return torch.ones(probs_t.size(0), dtype=torch.bool, device=probs_t.device)

        conf_t, pseudo_t = probs_t.detach().max(dim=1)
        mask_t = (conf_t >= self.pl_conf_thresh) & (pseudo_t == int(class_id))
        return mask_t

    # ------------------------------------------------------------------
    # V4: Confidence-gated class-wise source reliability
    # ------------------------------------------------------------------
    def _class_source_reliability_weights(self, f_s_list, f_t_list, labels_s_list, probs_t_list,
                                          mmd_losses, ent_losses):
        """
        CW-RWCA-V4：计算类别级源域可靠性权重 w_{s,c}。

        相比 V2/V3：
        - V2/V3：target class prototype 使用所有目标样本，并用 probs_t[:, c] 做 soft weight；
        - V4：只使用高置信目标样本：
              max_prob >= pl_conf_thresh 且 pseudo_label == c。

        这样可以避免低置信目标样本污染类别原型。
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

                # V4 核心：只取当前类别 c 的高置信目标伪标签样本。
                mask_t = self._high_conf_target_mask(probs_t, c)
                n_t = int(mask_t.sum().item())

                if n_s >= self.clmmd_min_source and n_t >= self.pl_min_target:
                    proto_s = f_s[mask_s].mean(dim=0)

                    # 高置信目标样本内部仍然保留 probs_t[:, c] 作为 soft weight。
                    wt_raw = probs_t[mask_t, c]
                    wt_sum = wt_raw.sum()

                    if wt_sum.detach().item() >= self.clmmd_min_target_weight:
                        f_t_c = f_t[mask_t]
                        wt = wt_raw / (wt_sum + self.entropy_eps)
                        proto_t = (f_t_c * wt.view(-1, 1)).sum(dim=0)

                        proto_s = F.normalize(proto_s, p=2, dim=0)
                        proto_t = F.normalize(proto_t, p=2, dim=0)

                        # 原型距离越小，说明该源域在类别 c 上越接近目标域。
                        dist_c = torch.mean((proto_s - proto_t) ** 2)

                        # 高置信目标样本上的类别条件熵越低，说明该源域分类器越稳定。
                        ent_c = (ent_vec[mask_t] * wt).sum()
                    else:
                        # 目标类别 soft 权重太弱，回退到全局可靠性并加惩罚。
                        dist_c = fallback_dist + 1.0
                        ent_c = fallback_ent + 1.0
                else:
                    # 高置信目标样本太少时，不强行计算类别原型。
                    # 这样比使用大量低置信伪标签更安全。
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

    # ------------------------------------------------------------------
    # V4: Confidence-gated CLMMD / LJMMD-like loss
    # ------------------------------------------------------------------
    def _classwise_lmmd_per_class(self, f_s, f_t, labels_s, probs_t):
        """
        返回每个类别的 soft LMMD loss 和有效类别标记。

        相比 V2/V3：
        - V2/V3：每个 class c 使用所有 target 样本，以 probs_t[:, c] 做 soft weight；
        - V4：每个 class c 只使用满足以下条件的目标样本：
              max_prob >= pl_conf_thresh and pseudo_label == c。

        这样可以减少错误伪标签对类别级对齐的污染。
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

            # V4 核心：当前类别 c 只使用高置信伪标签目标样本。
            mask_t = self._high_conf_target_mask(probs_t, c)
            n_t = int(mask_t.sum().item())
            if n_t < self.pl_min_target:
                continue

            wt_raw = probs_t[mask_t, c]
            wt_sum = wt_raw.sum()
            if wt_sum.detach().item() < self.clmmd_min_target_weight:
                continue

            xs = f_s[mask_s]
            xt = f_t[mask_t]
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