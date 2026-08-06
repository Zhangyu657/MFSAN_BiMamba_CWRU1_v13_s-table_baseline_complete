# -*- coding: utf-8 -*-
'''
MFSAN-CDAN-BLA

在 MFSAN-CDAN 的基础上，进一步引入 BiLSTM + Attention 特征增强分支。

模型创新点：
1. 保留原始 MFSAN 多源迁移框架：
   - 共享特征提取器 G；
   - 每个源域一个 Fs_i / Cs_i；
   - 源域分类损失 L_cls；
   - 源域-目标域边缘 MMD 对齐 L_MMD；
   - 目标域多分类器一致性损失 L_L1；
   - 测试阶段多分类器 softmax 融合。

2. 引入 BiLSTM-Attention 增强共享特征提取器：
   - 原始 MSCNN 主分支：提取多尺度局部冲击特征；
   - BiLSTM 辅助分支：建模长程时序依赖；
   - Temporal Attention：筛选关键时间片段；
   - Channel Attention：强化故障敏感通道；
   - Gate 残差融合：feat = feat_mscnn + gate * feat_bilstm_att。

3. 保留 CDAN 条件域对抗模块：
   - joint feature = p \otimes normalize(f)；
   - 源域使用真实标签 one-hot，目标域使用当前分类器预测概率；
   - joint feature 送入 GRL 梯度反转层；
   - 域判别器 D_i 区分 source / target；
   - 特征提取器通过 GRL 学习域不变条件特征。

4. 保留目标域条件熵最小化和可选 Conditional MMD。

推荐运行：
python train.py \
  --model_name MFSAN_CDAN_BLA \
  --source PU_0,PU_1,PU_2 \
  --target PU_3 \
  --train_mode multi_source \
  --data_dir /workspace/PU_TL \
  --signal_size 1024 \
  --backbone CNN \
  --cuda_device 0 \
  --max_epoch 2 \
  --lambda_adv 0.01 \
  --lambda_grl 0.5 \
  --lambda_cda 0 \
  --lambda_ent 0.005 \
  --include_faults K001,KA04,KA16,KA30,KB24,KB27,KI04,KI17,KI18

说明：
- 为了不修改 opt.py 的 backbone choices，本模型在 --backbone CNN 时默认使用
  “MSCNN + BiLSTM-Attention gate” 增强 backbone。
- 若你已经把 opt.py 放开了，也可以传 --backbone MSCNN_BiLSTM_Att。
'''

import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import utils
import modules
import modules_bla
from train_utils import TrainerBase


class GradientReverseFunction(Function):
    """
    Gradient Reversal Layer.

    Forward: identity.
    Backward: multiply gradient by -coeff.
    """

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
    Domain discriminator for conditional adversarial alignment.

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
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


class Trainer(TrainerBase):

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        if args.train_mode != 'multi_source':
            raise ValueError('MFSAN_CDAN_BLA is designed for --train_mode multi_source.')

        self.src_labels_flat = sorted(list(set([label for sublist in args.label_sets[:-1] for label in sublist])))
        num_classes = len(self.src_labels_flat)
        self.num_classes = num_classes

        # ========== Backbone: MSCNN + BiLSTM-Attention gated fusion ==========
        # 这里为了减少 opt.py 改动：--backbone CNN 时也使用增强 backbone。
        # 如果你想做原始 CNN baseline，请继续使用 model_name=MFSAN_CDA 或 MFSAN_CDAN。
        if args.backbone in ['CNN', 'MSCNN_BiLSTM_Att', 'MS_BiLSTM_Att', 'BLA']:
            self.G = modules_bla.MSCNNBiLSTMAttBackbone(
                in_channel=1,
                stem_channels=int(getattr(args, 'bla_stem_channels', 64)),
                lstm_hidden=int(getattr(args, 'bilstm_hidden', 64)),
                lstm_layers=int(getattr(args, 'bilstm_layers', 1)),
                dropout=args.dropout,
                gate_init=float(getattr(args, 'bla_gate_init', 0.01)),
                gate_max=float(getattr(args, 'bla_gate_max', 0.03)),
            ).to(self.device)
            actual_backbone = 'MSCNN_BiLSTM_Att_SmallGate'
        elif args.backbone == 'ResNet':
            # 不推荐这版用 ResNet；保留兼容。
            self.G = modules.ResNet(in_channel=1, layers=[2, 2, 2, 2]).to(self.device)
            actual_backbone = 'ResNet'
        else:
            raise Exception(f"unknown backbone type {args.backbone}")

        logging.info('Using model: MFSAN_CDAN_BLA')
        logging.info('Requested backbone: {}'.format(args.backbone))
        logging.info('Actual backbone: {}'.format(actual_backbone))
        logging.info('Backbone output dim: {}'.format(self.G.out_dim))
        if hasattr(self.G, 'get_gate'):
            logging.info('Initial BiLSTM-Att residual gate: {:.6f}'.format(
                self.G.get_gate().detach().item()
            ))
            if hasattr(self.G, 'gate_max'):
                logging.info('Max BiLSTM-Att residual gate: {:.6f}'.format(
                    self.G.gate_max
                ))
        # Source-specific feature extractors and classifiers, same as original MFSAN
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
                output_size=num_classes,
                num_layer=1,
                last=None
            )
            for i in range(self.num_source)
        ]).to(self.device)

        # Original MFSAN feature-level MK-MMD
        self.mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
            kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)]
        )

        # Optional Conditional MMD
        self.cda_mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
            kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)]
        )

        # ========== CDAN domain discriminators ==========
        self.feature_dim = self.Fs[0].feature_dim
        self.joint_dim = self.num_classes * self.feature_dim
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

        if args.train_mode == 'source_combine':
            self.src = ['concat_source']
        else:
            self.src = args.source_name

        self.optimizer = self._get_optimizer([self.G, self.Fs, self.Cs, self.Ds])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        self.num_iter = sum([len(self.dataloaders[s]) for s in self.src])

        # ========== Hyper-parameters ==========
        self.lambda_cda = float(getattr(args, 'lambda_cda', 0.0))
        self.lambda_ent = float(getattr(args, 'lambda_ent', 0.005))
        self.detach_prob = bool(getattr(args, 'cda_detach_prob', True))

        self.lambda_adv = float(getattr(args, 'lambda_adv', 0.01))
        self.lambda_grl = float(getattr(args, 'lambda_grl', 0.5))
        self.adv_detach_prob = bool(getattr(args, 'adv_detach_prob', True))
        self.adv_use_entropy_weight = bool(getattr(args, 'adv_use_entropy_weight', True))
        self.adv_conf_thresh = float(getattr(args, 'adv_conf_thresh', 0.0))

        self.entropy_eps = 1e-5

        logging.info('MFSAN-CDAN-BLA lambda_adv: {:.6f}'.format(self.lambda_adv))
        logging.info('MFSAN-CDAN-BLA lambda_grl: {:.6f}'.format(self.lambda_grl))
        logging.info('MFSAN-CDAN-BLA adv_detach_prob: {}'.format(self.adv_detach_prob))
        logging.info('MFSAN-CDAN-BLA adv_use_entropy_weight: {}'.format(self.adv_use_entropy_weight))
        logging.info('MFSAN-CDAN-BLA adv_conf_thresh: {:.6f}'.format(self.adv_conf_thresh))
        logging.info('MFSAN-CDAN-BLA lambda_cda: {:.6f}'.format(self.lambda_cda))
        logging.info('MFSAN-CDAN-BLA lambda_ent: {:.6f}'.format(self.lambda_ent))
        logging.info('MFSAN-CDAN-BLA joint feature dim: {} x {} = {}'.format(
            num_classes, self.feature_dim, self.joint_dim
        ))

    def save_model(self):
        torch.save({
            'G': self.G.state_dict(),
            'Fs': self.Fs.state_dict(),
            'Cs': self.Cs.state_dict(),
            'Ds': self.Ds.state_dict(),
            'lambda_adv': self.lambda_adv,
            'lambda_grl': self.lambda_grl,
            'lambda_cda': self.lambda_cda,
            'lambda_ent': self.lambda_ent,
        }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))

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

    def _target_entropy(self, probs):
        probs = torch.clamp(probs, min=self.entropy_eps, max=1.0)
        return -(probs * torch.log(probs)).sum(dim=1).mean()

    def _entropy_weight(self, probs):
        """
        w(x) = exp(-H(p(x))). Higher confidence -> larger weight.
        Normalize to mean 1 to avoid changing domain-loss scale too much.
        """
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
        batch_s = f_s.size(0)
        device = f_s.device

        # Source uses true one-hot labels as condition
        prob_s_onehot = F.one_hot(source_labels, num_classes=self.num_classes).float().to(device)

        # Optional target confidence filtering
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

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        for i in tqdm(range(self.num_iter), ascii=True):
            cur_src_idx = int(i % self.num_source)

            target_data, _ = self._get_next_batch('train')
            source_data, source_labels = self._get_next_batch(self.src[cur_src_idx], return_actual=True)
            source_labels = self._get_train_label(source_labels, label_set=self.src_labels_flat)

            self.optimizer.zero_grad()

            data = torch.cat((source_data, target_data), dim=0)

            g = self.G(data)
            g_s, g_t = g.chunk(2, dim=0)

            f = self.Fs[cur_src_idx](g)
            f_s, f_t = f.chunk(2, dim=0)

            y_s = self.Cs[cur_src_idx](f_s)
            y_t_cur = self.Cs[cur_src_idx](f_t)

            y_t_all = [self.Cs[k](self.Fs[k](g_t)) for k in range(self.num_source)]
            probs_t_all = [F.softmax(t, dim=1) for t in y_t_all]

            # Original MFSAN losses
            loss_c = F.cross_entropy(y_s, source_labels)
            loss_mmd = self.mkmmd(f_s, f_t)

            loss_l1 = 0.0
            for k in range(self.num_source - 1):
                for j in range(k + 1, self.num_source):
                    loss_l1 += torch.abs(probs_t_all[k] - probs_t_all[j]).mean()
            loss_l1 /= self.num_source

            # Optional Conditional MMD
            probs_s = F.softmax(y_s, dim=1)
            probs_t_cur = F.softmax(y_t_cur, dim=1)
            loss_cda = self._conditional_mmd(f_s, f_t, probs_s, probs_t_cur)

            # Target entropy minimization
            probs_t_fused = torch.stack(probs_t_all, dim=0).mean(dim=0)
            loss_ent = self._target_entropy(probs_t_fused)

            # CDAN adversarial loss
            adv_tradeoff = self.tradeoff[2] if len(self.tradeoff) > 2 else 1.0
            grl_coeff = self.lambda_grl * adv_tradeoff
            loss_adv, domain_acc = self._domain_adversarial_loss(
                cur_src_idx=cur_src_idx,
                f_s=f_s,
                f_t=f_t,
                source_labels=source_labels,
                prob_t=probs_t_cur,
                grl_coeff=grl_coeff
            )

            new_tradeoff = adv_tradeoff

            loss = (
                loss_c
                + self.tradeoff[0] * loss_mmd
                + self.tradeoff[1] * loss_l1
                + new_tradeoff * self.lambda_cda * loss_cda
                + new_tradeoff * self.lambda_ent * loss_ent
                + new_tradeoff * self.lambda_adv * loss_adv
            )

            # Log information
            epoch_acc['Source Data'] += self._get_accuracy(y_s, source_labels)
            epoch_acc['Domain Data'] += domain_acc.detach().item()

            epoch_loss['Source Classifier'] += loss_c
            epoch_loss['MMD'] += loss_mmd
            epoch_loss['L1'] += loss_l1
            epoch_loss['CDA MMD'] += loss_cda
            epoch_loss['Target Entropy'] += loss_ent
            epoch_loss['CDAN Domain'] += loss_adv
            epoch_loss['CDA Weighted'] += (new_tradeoff * self.lambda_cda * loss_cda).detach()
            epoch_loss['Entropy Weighted'] += (new_tradeoff * self.lambda_ent * loss_ent).detach()
            epoch_loss['CDAN Weighted'] += (new_tradeoff * self.lambda_adv * loss_adv).detach()

            loss.backward()
            self.optimizer.step()

        if hasattr(self.G, 'get_gate'):
            logging.info('Initial BiLSTM-Att residual gate: {:.6f}'.format(
                self.G.get_gate().detach().item()
            ))
            if hasattr(self.G, 'gate_max'):
                logging.info('Max BiLSTM-Att residual gate: {:.6f}'.format(
                    self.G.gate_max
                ))

        logging.info(
            'MFSAN-CDAN-BLA active: lambda_adv={:.6f}, lambda_grl={:.6f}, lambda_cda={:.6f}, lambda_ent={:.6f}, adv_detach_prob={}, entropy_weight={}'.format(
                self.lambda_adv,
                self.lambda_grl,
                self.lambda_cda,
                self.lambda_ent,
                self.adv_detach_prob,
                self.adv_use_entropy_weight
            )
        )

        return epoch_acc, epoch_loss

    def _eval(self, data, actual_labels, correct, total):
        feat_tgt = self.G(data)

        logits_tgt = [
            F.softmax(self.Cs[i](self.Fs[i](feat_tgt)), dim=1)
            for i in range(self.num_source)
        ]

        pred = torch.sum(torch.stack(logits_tgt), dim=0).argmax(dim=1)
        actual_pred = self._get_actual_label(pred, label_set=self.src_labels_flat)

        # For F1 calculation in modified train_utils.test()
        if hasattr(self, '_eval_pred_list') and hasattr(self, '_eval_label_list'):
            self._eval_pred_list.append(actual_pred.detach().cpu())
            self._eval_label_list.append(actual_labels.detach().cpu())

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
