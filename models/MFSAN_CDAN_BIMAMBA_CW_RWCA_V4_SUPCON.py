# -*- coding: utf-8 -*-
"""
MFSAN-CDAN-BiMamba-CW-RWCA-V4-SupCon

V4-SupCon = V4 + source supervised contrastive learning.

核心目的：
1. 保留 V4 的全部能力：
   - V3 eval no EMA；
   - V4 高置信伪标签门控；
   - CW-RWCA 类别级源域可靠性；
   - CLMMD / CDAN / MMD / target entropy。
2. 在源域监督分类之外，额外加入 SupCon 监督对比损失：
   - 拉近同类源域样本特征；
   - 拉远不同类源域样本特征；
   - 尤其用于增强 KA04 / KA16 这类高混淆类别的类间分离。

推荐使用方式：
    --model_name MFSAN_CDAN_BIMAMBA_CW_RWCA_V4_SUPCON

推荐初始参数：
    --lambda_supcon 0.02
    --supcon_temperature 0.10
    --supcon_start_epoch 1
    --supcon_feature_mode G
    --supcon_focus_classes 1,2


python train.py \
  --model_name MFSAN_CDAN_BIMAMBA_CW_RWCA_V4_SUPCON \
  --source PU_1,PU_2,PU_3 \
  --target PU_0 \
  --data_dir /workspace/PU_TL_9_replace \
  --train_mode multi_source \
  --cuda_device 0 \
  --max_epoch 10 \
  --batch_size 64 \
  --signal_size 1024 \
  --backbone CNN \
  --lambda_clmmd 0.005 \
  --pl_conf_thresh 0.80 \
  --pl_min_target 2 \
  --lambda_supcon 0.02 \
  --supcon_temperature 0.10 \
  --supcon_start_epoch 1 \
  --supcon_feature_mode G \
  --random_state 42 \
  --supcon_focus_classes 1,2

如果 opt.py 暂时没有注册这些参数，也可以不传，代码会使用默认值。
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.MFSAN_CDAN_BIMAMBA_CW_RWCA_V4 import Trainer as V4Trainer


class SupConLoss(nn.Module):
    """
    单视图 Supervised Contrastive Loss。

    Args:
        features: [B, D]
        labels:   [B]

    说明：
    - 同一类别样本互为正样本；
    - 不同类别样本互为负样本；
    - 如果某个 anchor 在当前 batch 中没有同类正样本，则该 anchor 不参与平均；
    - 返回标量 loss。
    """

    def __init__(self, temperature=0.10, eps=1e-8):
        super(SupConLoss, self).__init__()
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(self, features, labels):
        device = features.device

        if features.dim() > 2:
            features = torch.flatten(features, start_dim=1)

        labels = labels.contiguous().view(-1)
        batch_size = features.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        features = F.normalize(features, p=2, dim=1)

        logits = torch.matmul(features, features.t()) / max(self.temperature, self.eps)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        labels_col = labels.view(-1, 1)
        pos_mask = torch.eq(labels_col, labels_col.t()).float().to(device)

        # 去掉自己和自己。
        logits_mask = torch.ones_like(pos_mask, device=device)
        logits_mask.fill_diagonal_(0)
        pos_mask = pos_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0

        if valid.sum().detach().item() == 0:
            return torch.tensor(0.0, device=device)

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_count + self.eps)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


class Trainer(V4Trainer):
    """
    V4 + SupCon Trainer。

    改动点：
    - 重写 _train_one_epoch，在原 V2/V3/V4 主训练循环基础上加入 loss_supcon；
    - 默认只对 source shared feature g_s 计算 SupCon，因为 G 是共享特征空间，跨源域可比；
    - 可通过 supcon_focus_classes 指定只强化某些高混淆类别，比如 KA04/KA16 对应标签 1,2。
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        self.lambda_supcon = float(getattr(args, 'lambda_supcon', 0.02))
        self.supcon_temperature = float(getattr(args, 'supcon_temperature', 0.10))
        self.supcon_start_epoch = int(getattr(args, 'supcon_start_epoch', 1))
        self.supcon_feature_mode = str(getattr(args, 'supcon_feature_mode', 'G')).strip()

        # 例如：--supcon_focus_classes 1,2
        # 空字符串 / all 表示对所有类别做 SupCon。
        focus = str(getattr(args, 'supcon_focus_classes', '1,2')).strip()
        if focus == '' or focus.lower() == 'all':
            self.supcon_focus_classes = None
        else:
            self.supcon_focus_classes = sorted(list(set([
                int(x.strip()) for x in focus.split(',') if x.strip() != ''
            ])))

        self.supcon_loss = SupConLoss(
            temperature=self.supcon_temperature,
            eps=getattr(self, 'entropy_eps', 1e-8)
        ).to(self.device)

        logging.info('Using model: MFSAN_CDAN_BIMAMBA_CW_RWCA_V4_SUPCON')
        logging.info(
            'SupCon enabled: lambda_supcon={:.6f}, temperature={:.4f}, start_epoch={}, feature_mode={}, focus_classes={}'.format(
                self.lambda_supcon,
                self.supcon_temperature,
                self.supcon_start_epoch,
                self.supcon_feature_mode,
                'all' if self.supcon_focus_classes is None else self.supcon_focus_classes,
            )
        )

    def _checkpoint_dict(self):
        ckpt = super(Trainer, self)._checkpoint_dict()
        ckpt['v4_supcon'] = True
        ckpt['lambda_supcon'] = self.lambda_supcon
        ckpt['supcon_temperature'] = self.supcon_temperature
        ckpt['supcon_start_epoch'] = self.supcon_start_epoch
        ckpt['supcon_feature_mode'] = self.supcon_feature_mode
        ckpt['supcon_focus_classes'] = self.supcon_focus_classes
        return ckpt

    def _filter_supcon_features(self, features, labels):
        """
        根据 supcon_focus_classes 筛选用于 SupCon 的样本。
        默认 focus_classes=1,2，对应当前 PU 9类设置中的 KA04 / KA16。
        """
        if self.supcon_focus_classes is None:
            return features, labels

        mask = torch.zeros_like(labels, dtype=torch.bool, device=labels.device)
        for c in self.supcon_focus_classes:
            mask = mask | (labels == int(c))

        if mask.sum().detach().item() <= 1:
            return None, None

        return features[mask], labels[mask]

    def _compute_source_supcon_loss(self, g_s_list, f_s_all, source_label_list):
        """
        计算源域监督对比损失。

        feature_mode:
        - G: 使用共享主干特征 g_s，推荐默认；
        - F: 使用源域特定分支特征 f_s，不推荐作为默认，因为不同 Fs 的空间未必完全可比。
        """
        cur_epoch = int(getattr(self, '_cur_epoch', 1))
        if self.lambda_supcon <= 0.0 or cur_epoch < self.supcon_start_epoch:
            return torch.tensor(0.0, device=self.device)

        mode = self.supcon_feature_mode.upper()
        if mode == 'F':
            feat_list = f_s_all
        else:
            feat_list = g_s_list

        features = torch.cat(feat_list, dim=0)
        labels = torch.cat(source_label_list, dim=0)

        features, labels = self._filter_supcon_features(features, labels)
        if features is None:
            return torch.tensor(0.0, device=self.device)

        return self.supcon_loss(features, labels)

    def _finalize_v3_epoch_weights(self):
        """
        因为本类重写了 _train_one_epoch，不能再依赖 V3 的 super 包装。
        这里手动保留 V3 的 eval no EMA 行为。
        """
        with torch.no_grad():
            if getattr(self, '_v3_class_weight_count', 0) > 0:
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

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        self._cur_epoch = int(getattr(self, '_cur_epoch', 1))

        # 手动开启 V3 epoch 权重收集。
        self._v3_collect_epoch_weights = True
        self._v3_class_weight_sum = torch.zeros(
            self.num_source,
            self.num_classes,
            device=self.device
        )
        self._v3_class_weight_count = 0

        weight_sum = torch.zeros(self.num_source, device=self.device)
        global_weight_sum = torch.zeros(self.num_source, device=self.device)
        raw_class_weight_sum = torch.zeros(self.num_source, self.num_classes, device=self.device)
        class_weight_sum = torch.zeros(self.num_source, self.num_classes, device=self.device)
        alpha_sum = 0.0
        supcon_sum = torch.tensor(0.0, device=self.device)

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
            g_split = torch.split(g_all, split_sizes, dim=0)
            g_s_list = list(g_split[:-1])
            g_t = g_split[-1]

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
                domain_acc_list.append(
                    domain_acc_k if torch.is_tensor(domain_acc_k) else torch.tensor(domain_acc_k, device=self.device)
                )

                epoch_acc['Source Data'] += self._get_accuracy(y_s, labels_s) / float(self.num_source)

            # V2/V3/V4: global RWCA prior + class-wise correction + V4 target confidence gate.
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

            # CW-weighted source classification loss
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

            # CW-weighted CLMMD
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

            # CDD / classifier consistency
            loss_l1 = torch.tensor(0.0, device=self.device)
            for k in range(self.num_source):
                abs_diff = torch.abs(probs_t_all[k] - probs_t_fused.detach())
                loss_l1 = loss_l1 + (abs_diff * class_src_weights[k].view(1, -1)).sum(dim=1).mean() / float(self.num_classes)

            loss_ent = self._target_entropy(probs_t_fused)
            domain_acc = sum(src_weights[k] * domain_acc_list[k] for k in range(self.num_source))

            # SupCon：增强源域共享特征类别分离，默认聚焦 KA04/KA16，即 label 1/2。
            loss_supcon = self._compute_source_supcon_loss(g_s_list, f_s_all, source_label_list)
            supcon_sum = supcon_sum + loss_supcon.detach()

            new_tradeoff = adv_tradeoff
            loss = (
                loss_cls
                + self.tradeoff[0] * loss_mmd
                + self.tradeoff[1] * loss_l1
                + new_tradeoff * self.lambda_cda * loss_cda
                + new_tradeoff * self.lambda_ent * loss_ent
                + new_tradeoff * self.lambda_adv * loss_adv
                + new_tradeoff * self.lambda_clmmd * loss_clmmd
                + self.lambda_supcon * loss_supcon
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
            epoch_loss['SupCon'] += loss_supcon.detach()
            epoch_loss['SupCon Weighted'] += (self.lambda_supcon * loss_supcon).detach()

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
        avg_supcon = (supcon_sum / denom_iter).detach().item()

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

        logging.info(
            'SupCon average: {:.6f} | weighted={:.6f} | focus_classes={} | feature_mode={}'.format(
                avg_supcon,
                self.lambda_supcon * avg_supcon,
                'all' if self.supcon_focus_classes is None else self.supcon_focus_classes,
                self.supcon_feature_mode,
            )
        )

        if hasattr(self.G, 'get_gate'):
            logging.info('BiMamba-Att residual gate: {:.6f}'.format(self.G.get_gate().detach().item()))
            if hasattr(self.G, 'gate_max'):
                logging.info('Max BiMamba-Att residual gate: {:.6f}'.format(self.G.gate_max))

        logging.info(
            'MFSAN-CDAN-BiMamba-CW-RWCA-V4-SupCon active: lambda_adv={:.6f}, lambda_grl={:.6f}, lambda_cda={:.6f}, lambda_clmmd={:.6f}, lambda_ent={:.6f}, lambda_supcon={:.6f}'.format(
                self.lambda_adv, self.lambda_grl, self.lambda_cda, self.lambda_clmmd, self.lambda_ent, self.lambda_supcon
            )
        )

        # 手动完成 V3 eval no EMA 的 last epoch 权重更新。
        self._finalize_v3_epoch_weights()

        return epoch_acc, epoch_loss
