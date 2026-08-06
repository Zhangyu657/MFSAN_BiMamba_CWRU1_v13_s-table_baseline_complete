'''
DW-MFSAN-CL

Based on conservative DW-MFSAN.

改进点：
1. 保留原始 MFSAN 的训练循环、MMD 实现、L1 一致性、测试融合、保存/加载逻辑；
2. 保留保守动态源域权重：
   - 分类损失不加权；
   - 只对 MMD 域对齐损失加权；
   - 动态权重与平均权重混合；
   - 设置最小源域权重保护；
3. 新增 SupCon 监督对比学习：
   - 在共享特征 g_s 上计算源域监督对比损失；
   - 同类样本拉近，不同类样本拉远；
   - 增强共享特征的类别判别性；
4. 总损失：
   L = Lcls + lambda_mmd * alpha_i * MMD_i + lambda_l1 * L1 + lambda_supcon * Lsupcon
'''

import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import utils
import modules
from train_utils import TrainerBase


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    输入：
        features: [B, D]
        labels:   [B]

    作用：
        同一类别样本互为正样本；
        不同类别样本作为负样本；
        增强特征空间中的类内聚合与类间分离。

    说明：
        这里使用单视图 SupCon，适合直接作用在源域共享特征 g_s 上。
    """

    def __init__(self, temperature=0.1, eps=1e-8):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features, labels):
        device = features.device

        if features.dim() > 2:
            features = torch.flatten(features, start_dim=1)

        labels = labels.contiguous().view(-1)
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        # 特征归一化，避免尺度影响相似度
        features = F.normalize(features, dim=1)

        # 相似度矩阵 [B, B]
        logits = torch.div(torch.matmul(features, features.T), self.temperature)

        # 数值稳定
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # mask[i, j] = 1 表示 i 和 j 是同类
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 去掉自己和自己
        logits_mask = torch.ones_like(mask).to(device)
        logits_mask.fill_diagonal_(0)

        mask = mask * logits_mask

        # exp logits，去掉对角线
        exp_logits = torch.exp(logits) * logits_mask

        # log prob
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)

        # 每个 anchor 的正样本数量
        positive_count = mask.sum(dim=1)

        # 如果某些 anchor 在 batch 内没有同类正样本，则不参与平均
        valid_mask = positive_count > 0

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positive_count + self.eps)

        loss = -mean_log_prob_pos[valid_mask].mean()

        return loss


class Trainer(TrainerBase):

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        if args.train_mode != 'multi_source':
            raise ValueError('DW_MFSAN_CL is designed for --train_mode multi_source.')

        self.src_labels_flat = sorted(list(set([label for sublist in args.label_sets[:-1] for label in sublist])))
        num_classes = len(self.src_labels_flat)

        if args.backbone == 'CNN':
            self.G = modules.MSCNN(in_channel=1).to(self.device)
        elif args.backbone == 'ResNet':
            self.G = modules.ResNet(in_channel=1, layers=[2, 2, 2, 2]).to(self.device)
        else:
            raise Exception(f"unknown backbone type {args.backbone}")

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

        # 保留原始 MFSAN 的 MK-MMD，保证公平对比
        self.mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
            kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)]
        )

        # SupCon
        self.supcon = SupConLoss(temperature=0.1).to(self.device)

        self._init_data()

        if args.train_mode == 'source_combine':
            self.src = ['concat_source']
        else:
            self.src = args.source_name

        self.optimizer = self._get_optimizer([self.G, self.Fs, self.Cs])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        # 保留原始 MFSAN 的迭代次数逻辑
        self.num_iter = sum([len(self.dataloaders[s]) for s in self.src])

        # ========== Conservative Dynamic Source Weight settings ==========
        self.mmd_ema = torch.ones(self.num_source, device=self.device)

        self.mmd_ema_momentum = 0.90

        self.dw_tau = 1.0

        # 动态权重混合系数
        # 0.5 表示一半平均权重，一半动态权重
        self.dw_rho = 0.5

        # 最小源域权重保护
        self.min_source_weight = 0.15

        self.keep_loss_scale = True

        # ========== SupCon settings ==========
        # 第一版建议小一点，避免对 CE 和 MMD 造成过强干扰。
        # 如果效果一般，可以后面尝试 0.02 / 0.05 / 0.1。
        self.lambda_supcon = 0.05

    def save_model(self):
        torch.save({
            'G': self.G.state_dict(),
            'Fs': self.Fs.state_dict(),
            'Cs': self.Cs.state_dict(),
            'mmd_ema': self.mmd_ema.detach().cpu()
        }, self.args.save_path + '.pth')

        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)

        self.G.load_state_dict(ckpt['G'])
        self.Fs.load_state_dict(ckpt['Fs'])
        self.Cs.load_state_dict(ckpt['Cs'])

        if 'mmd_ema' in ckpt:
            self.mmd_ema = ckpt['mmd_ema'].to(self.device)

    def _set_to_train(self):
        self.G.train()
        self.Fs.train()
        self.Cs.train()

    def _set_to_eval(self):
        self.G.eval()
        self.Fs.eval()
        self.Cs.eval()

    def _update_mmd_ema(self, cur_src_idx, loss_mmd):
        """
        更新当前源域的 MMD 运行均值。
        """
        with torch.no_grad():
            cur_value = loss_mmd.detach()

            self.mmd_ema[cur_src_idx] = (
                self.mmd_ema_momentum * self.mmd_ema[cur_src_idx]
                + (1.0 - self.mmd_ema_momentum) * cur_value
            )

    def _compute_dynamic_weights(self):
        """
        根据 mmd_ema 计算动态源域权重。

        1. 根据 MMD 得到动态权重 alpha_dynamic；
        2. 与平均权重 uniform_alpha 混合；
        3. 加入最小权重保护。

        MMD 越小，说明源域越接近目标域，动态权重越大。
        """
        d = self.mmd_ema.detach()

        uniform_alpha = torch.ones_like(d) / len(d)

        d_std = torch.std(d, unbiased=False)
        if d_std.item() < 1e-8:
            alpha = uniform_alpha
        else:
            d_norm = (d - d.mean()) / (d_std + 1e-8)
            alpha_dynamic = torch.softmax(-d_norm / self.dw_tau, dim=0)

            alpha = (1.0 - self.dw_rho) * uniform_alpha + self.dw_rho * alpha_dynamic

        if self.min_source_weight is not None and self.min_source_weight > 0:
            alpha = torch.clamp(alpha, min=self.min_source_weight)
            alpha = alpha / alpha.sum()

        return alpha

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        weight_sum = torch.zeros(self.num_source, device=self.device)
        weight_count = 0

        mmd_sum = torch.zeros(self.num_source, device=self.device)
        mmd_count = torch.zeros(self.num_source, device=self.device)

        for i in tqdm(range(self.num_iter), ascii=True):
            cur_src_idx = int(i % self.num_source)

            target_data, _ = self._get_next_batch('train')
            source_data, source_labels = self._get_next_batch(self.src[cur_src_idx], return_actual=True)
            source_labels = self._get_train_label(source_labels, label_set=self.src_labels_flat)

            self.optimizer.zero_grad()

            data = torch.cat((source_data, target_data), dim=0)

            # 共享特征
            g = self.G(data)
            g_s, g_t = g.chunk(2, dim=0)

            # 当前源域特定分支
            f = self.Fs[cur_src_idx](g)
            f_s, f_t = f.chunk(2, dim=0)

            y_s = self.Cs[cur_src_idx](f_s)

            # 目标域经过所有源域分支，计算 L1 一致性
            y_t = [self.Cs[k](self.Fs[k](g_t)) for k in range(self.num_source)]

            # 原始损失
            loss_c = F.cross_entropy(y_s, source_labels)
            loss_mmd = self.mkmmd(f_s, f_t)

            logits_tgt = [F.softmax(t, dim=1) for t in y_t]

            loss_l1 = 0.0
            for k in range(self.num_source - 1):
                for j in range(k + 1, self.num_source):
                    loss_l1 += torch.abs(logits_tgt[k] - logits_tgt[j]).mean()

            # 保留原始 MFSAN 写法
            loss_l1 /= self.num_source

            # SupCon：在共享源域特征 g_s 上做监督对比学习
            # 这样可以增强共享特征空间的类别判别性，不破坏源域特定分支结构。
            loss_supcon = self.supcon(g_s, source_labels)

            # 动态权重
            self._update_mmd_ema(cur_src_idx, loss_mmd)
            alpha = self._compute_dynamic_weights()

            cur_alpha = alpha[cur_src_idx]

            if self.keep_loss_scale:
                # alpha=[1/3,1/3,1/3] 时 cur_weight=1，等价原始 MFSAN 的 MMD 权重
                cur_weight = cur_alpha * self.num_source
            else:
                cur_weight = cur_alpha

            # DW-MFSAN-CL total loss
            # 分类损失不加权；
            # MMD 使用保守动态权重；
            # L1 保持原始；
            # SupCon 小权重加入。
            loss = (
                loss_c
                + self.tradeoff[0] * cur_weight * loss_mmd
                + self.tradeoff[1] * loss_l1
                + self.lambda_supcon * loss_supcon
            )

            # log information
            epoch_acc['Source Data'] += self._get_accuracy(y_s, source_labels)

            epoch_loss['Source Classifier'] += loss_c
            epoch_loss['MMD'] += loss_mmd
            epoch_loss['L1'] += loss_l1
            epoch_loss['DW MMD'] += cur_weight.detach() * loss_mmd.detach()
            epoch_loss['SupCon'] += loss_supcon.detach()

            loss.backward()
            self.optimizer.step()

            weight_sum += alpha.detach()
            weight_count += 1

            mmd_sum[cur_src_idx] += loss_mmd.detach()
            mmd_count[cur_src_idx] += 1

        if weight_count > 0:
            avg_weight = (weight_sum / weight_count).detach().cpu().tolist()

            weight_str = ', '.join([
                f'{self.src[k]}={avg_weight[k]:.4f}'
                for k in range(self.num_source)
            ])

            logging.info('Dynamic Source Weights: ' + weight_str)

        avg_mmd_list = []
        for k in range(self.num_source):
            if mmd_count[k].item() > 0:
                avg_mmd = (mmd_sum[k] / mmd_count[k]).detach().cpu().item()
            else:
                avg_mmd = 0.0

            avg_mmd_list.append(avg_mmd)

        mmd_str = ', '.join([
            f'{self.src[k]}={avg_mmd_list[k]:.4f}'
            for k in range(self.num_source)
        ])

        ema_str = ', '.join([
            f'{self.src[k]}={self.mmd_ema[k].detach().cpu().item():.4f}'
            for k in range(self.num_source)
        ])

        logging.info('Epoch Source-Target MMD: ' + mmd_str)
        logging.info('EMA Source-Target MMD: ' + ema_str)
        logging.info('SupCon lambda: {:.4f}'.format(self.lambda_supcon))

        return epoch_acc, epoch_loss

    def _eval(self, data, actual_labels, correct, total):
        feat_tgt = self.G(data)

        logits_tgt = [
            F.softmax(self.Cs[i](self.Fs[i](feat_tgt)), dim=1)
            for i in range(self.num_source)
        ]

        # 测试阶段保持原始 MFSAN 的求和融合
        pred = torch.sum(torch.stack(logits_tgt), dim=0).argmax(dim=1)

        actual_pred = self._get_actual_label(pred, label_set=self.src_labels_flat)

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