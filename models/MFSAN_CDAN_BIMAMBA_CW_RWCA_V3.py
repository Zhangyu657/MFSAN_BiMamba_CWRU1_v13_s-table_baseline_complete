# -*- coding: utf-8 -*-
"""
MFSAN-CDAN-BiMamba-CW-RWCA-V3

V3 = V2 + eval no EMA.

核心改动：
1. 训练阶段仍然沿用 V2 的 global-guided class-wise RWCA；
2. 每个 epoch 内累积当前 batch 的 guided class-wise source weights；
3. epoch 结束后保存当前 epoch 平均 class-wise source weights；
4. eval / validation 阶段优先使用当前 epoch 平均权重，而不是历史 EMA 权重；
5. 这样可以缓解 V2 中“训练权重合理，但 eval EMA 权重滞后”的问题。

使用方式：
把本文件放到：
    models/MFSAN_CDAN_BIMAMBA_CW_RWCA_V3.py

运行时使用：
    --model_name MFSAN_CDAN_BIMAMBA_CW_RWCA_V3

注意：
本文件依赖你工程中已经存在的：
    models/MFSAN_CDAN_BIMAMBA_CW_RWCA_V2.py
"""

import logging
import torch

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V2 import Trainer as V2Trainer


class Trainer(V2Trainer):
    """
    CW-RWCA-V3 Trainer.

    不重写 V2 的主训练逻辑，只通过三个小改动修复 eval EMA 滞后：
    1. __init__ 初始化 last_epoch 权重；
    2. _update_class_source_weight_ema 中累计当前 epoch 的 guided weights；
    3. _eval_class_weighted_fusion 中优先使用 last_epoch 权重。
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # V3: eval 阶段使用当前 epoch 的平均 guided class weights。
        # 初始化时先用 EMA 兜底，等第一个 epoch 训练完后会被更新。
        self.class_source_weight_last_epoch = self.class_source_weight_ema.detach().clone()

        # 当前 epoch 的权重累积器。
        self._v3_collect_epoch_weights = False
        self._v3_class_weight_sum = None
        self._v3_class_weight_count = 0

        logging.info('Using model: MFSAN_CDAN_BIMAMBA_CW_RWCA_V3')
        logging.info('CW-RWCA-V3 eval fusion uses current epoch average guided weights instead of EMA.')

    def _checkpoint_dict(self):
        """
        在 V2 checkpoint 基础上额外保存 last_epoch 权重。
        """
        ckpt = super(Trainer, self)._checkpoint_dict()
        ckpt['class_source_weight_last_epoch'] = self.class_source_weight_last_epoch.detach().cpu()
        ckpt['v3_eval_no_ema'] = True
        return ckpt

    def load_model(self):
        """
        兼容 V2 checkpoint：
        如果 checkpoint 里有 last_epoch 权重就加载；
        没有就用 EMA 作为兜底。
        """
        super(Trainer, self).load_model()

        try:
            ckpt = torch.load(self.args.load_path, map_location=self.device)

            if 'class_source_weight_last_epoch' in ckpt:
                cw_last = ckpt['class_source_weight_last_epoch'].to(self.device).float()

                if (
                    cw_last.dim() == 2
                    and cw_last.size(0) == self.num_source
                    and cw_last.size(1) == self.num_classes
                ):
                    self.class_source_weight_last_epoch = self._normalize_class_source_weights(cw_last)
                    logging.info(
                        'Loaded class_source_weight_last_epoch with shape {}'.format(
                            tuple(cw_last.shape)
                        )
                    )
                else:
                    self.class_source_weight_last_epoch = self.class_source_weight_ema.detach().clone()
                    logging.warning(
                        'Ignore class_source_weight_last_epoch due to incompatible shape: {}'.format(
                            tuple(cw_last.shape)
                        )
                    )
            else:
                self.class_source_weight_last_epoch = self.class_source_weight_ema.detach().clone()
                logging.info(
                    'No class_source_weight_last_epoch in checkpoint; fallback to class_source_weight_ema.'
                )

        except Exception as e:
            self.class_source_weight_last_epoch = self.class_source_weight_ema.detach().clone()
            logging.warning(
                'Failed to load class_source_weight_last_epoch, fallback to EMA. Error: {}'.format(e)
            )

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        """
        沿用 V2 的 _train_one_epoch。

        V2 的训练过程中，每个 iteration 都会调用：
            self._update_class_source_weight_ema(class_src_weights)

        V3 重写了 _update_class_source_weight_ema，
        所以可以在不改 V2 主训练逻辑的情况下，
        顺手累计当前 epoch 的 guided class-wise weights。
        """
        self._v3_collect_epoch_weights = True
        self._v3_class_weight_sum = torch.zeros(
            self.num_source,
            self.num_classes,
            device=self.device
        )
        self._v3_class_weight_count = 0

        epoch_acc, epoch_loss = super(Trainer, self)._train_one_epoch(epoch_acc, epoch_loss)

        # 一个 epoch 训练结束后，把“当前 epoch 平均 guided weights”保存下来。
        with torch.no_grad():
            if self._v3_class_weight_count > 0:
                avg_cw = self._v3_class_weight_sum / float(self._v3_class_weight_count)
                self.class_source_weight_last_epoch = self._normalize_class_source_weights(
                    avg_cw.detach().clone()
                )

                src_last = self._source_weights_from_class_weights(
                    self.class_source_weight_last_epoch
                )

                logging.info(
                    'CW-RWCA-V3 last-epoch guided source weights for eval: {}'.format(
                        ', '.join([
                            'src{}={:.4f}'.format(i, src_last[i].detach().item())
                            for i in range(self.num_source)
                        ])
                    )
                )

                for c in self._cw_log_classes:
                    logging.info(
                        'CW-RWCA-V3 last-epoch guided class-{} source weights for eval: {}'.format(
                            c,
                            ', '.join([
                                'src{}={:.4f}'.format(
                                    i,
                                    self.class_source_weight_last_epoch[i, c].detach().item()
                                )
                                for i in range(self.num_source)
                            ])
                        )
                    )
            else:
                self.class_source_weight_last_epoch = self.class_source_weight_ema.detach().clone()
                logging.warning(
                    'CW-RWCA-V3 did not collect epoch weights; fallback eval weights to EMA.'
                )

        self._v3_collect_epoch_weights = False
        self._v3_class_weight_sum = None
        self._v3_class_weight_count = 0

        return epoch_acc, epoch_loss

    def _update_class_source_weight_ema(self, class_weights):
        """
        V2 每个 iteration 会调用这个函数更新 EMA。

        V3 在不破坏 V2 EMA 的前提下，顺手累计当前 epoch 的真实 guided weights。
        """
        with torch.no_grad():
            cw = self._normalize_class_source_weights(class_weights.detach())

            if getattr(self, '_v3_collect_epoch_weights', False):
                if self._v3_class_weight_sum is None:
                    self._v3_class_weight_sum = torch.zeros_like(cw, device=self.device)
                    self._v3_class_weight_count = 0

                self._v3_class_weight_sum += cw
                self._v3_class_weight_count += 1

        # 保留 V2 的 EMA 更新逻辑，方便日志和 checkpoint 兼容。
        super(Trainer, self)._update_class_source_weight_ema(class_weights)

    def _eval_class_weighted_fusion(self, probs_list):
        """
        Eval-time class-wise prediction fusion.

        V2 原逻辑：
            base = self.class_source_weight_ema

        V3 修改：
            base = self.class_source_weight_last_epoch

        这样验证阶段直接使用当前 epoch 的平均 guided class-wise weights，
        避免 EMA 滞后导致 eval fusion 偏移。
        """
        probs_stack = torch.stack(probs_list, dim=0)  # [K, B, C]

        if (
            hasattr(self, 'class_source_weight_last_epoch')
            and self.class_source_weight_last_epoch is not None
        ):
            base = self.class_source_weight_last_epoch.to(probs_stack.device)
        else:
            base = self.class_source_weight_ema.to(probs_stack.device)

        base = self._normalize_class_source_weights(base)

        if not self.rw_eval_use_entropy:
            fused_prob = (
                base.view(self.num_source, 1, self.num_classes)
                * probs_stack
            ).sum(dim=0)

            fused_prob = fused_prob / (
                fused_prob.sum(dim=1, keepdim=True) + self.entropy_eps
            )

            return fused_prob, base

        ent = -(
            torch.clamp(probs_stack, min=self.entropy_eps)
            * torch.log(torch.clamp(probs_stack, min=self.entropy_eps))
        ).sum(dim=2)  # [K, B]

        log_prior = torch.log(base.clamp_min(self.entropy_eps)).view(
            self.num_source,
            1,
            self.num_classes
        )

        score = log_prior - ent.unsqueeze(2) / max(
            self.rw_eval_tau,
            self.entropy_eps
        )  # [K, B, C]

        weights = torch.softmax(score, dim=0)

        fused_prob = (weights * probs_stack).sum(dim=0)
        fused_prob = fused_prob / (
            fused_prob.sum(dim=1, keepdim=True) + self.entropy_eps
        )

        return fused_prob, weights