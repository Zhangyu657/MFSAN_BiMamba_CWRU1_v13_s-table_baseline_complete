'''
MFSAN-CDA

Based on the original MFSAN.

改进点：
1. 保留原始 MFSAN 的多源训练循环、MSCNN/ResNet backbone、MK-MMD、L1 一致性、测试融合、保存/加载逻辑；
2. 新增 Conditional MMD（CDA）：
   - 使用当前源域分支得到源域/目标域特征 f_s / f_t；
   - 使用当前分类器输出源域/目标域类别概率 p_s / p_t；
   - 构造 joint feature = p \otimes normalize(f)，近似实现“特征-类别联合表示”；
   - 对 joint_s 与 joint_t 做 MK-MMD，实现类别条件层面的源目标对齐；
3. 新增目标域条件熵最小化：
   - 对所有源域分类器在目标域上的预测概率做平均融合；
   - 最小化融合预测的熵，让目标域预测更确定；
4. 不引入动态源域权重，不引入 SupCon，不改变测试阶段的 MFSAN 多分类器融合方式。

推荐先用：
python train.py \
  --model_name MFSAN_CDA \
  --source PU_0,PU_1,PU_2 \
  --target PU_3 \
  --train_mode multi_source \
  --data_dir /workspace/PU_TL \
  --signal_size 1024 \
  --backbone CNN \
  --cuda_device 0 \
  --max_epoch 2
'''

import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import utils
import modules
from train_utils import TrainerBase


class Trainer(TrainerBase):

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        if args.train_mode != 'multi_source':
            raise ValueError('MFSAN_CDA is designed for --train_mode multi_source.')

        self.src_labels_flat = sorted(list(set([label for sublist in args.label_sets[:-1] for label in sublist])))
        num_classes = len(self.src_labels_flat)
        self.num_classes = num_classes

        # ========== Backbone: keep original MFSAN-CNN by default ==========
        if args.backbone == 'CNN':
            self.G = modules.MSCNN(in_channel=1).to(self.device)
        elif args.backbone == 'ResNet':
            self.G = modules.ResNet(in_channel=1, layers=[2, 2, 2, 2]).to(self.device)
        else:
            raise Exception(f"unknown backbone type {args.backbone}")

        logging.info('Using model: MFSAN_CDA')
        logging.info('Using backbone: {}'.format(args.backbone))
        logging.info('Backbone output dim: {}'.format(self.G.out_dim))

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

        # Conditional MMD uses a separate MK-MMD object because the joint feature scale/dim differs
        self.cda_mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
            kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)]
        )

        self._init_data()

        if args.train_mode == 'source_combine':
            self.src = ['concat_source']
        else:
            self.src = args.source_name

        self.optimizer = self._get_optimizer([self.G, self.Fs, self.Cs])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        # Keep original MFSAN iteration logic
        self.num_iter = sum([len(self.dataloaders[s]) for s in self.src])

        # ========== CDA hyper-parameters ==========
        # lambda_cda: conditional MMD strength. Start conservative to avoid hurting the strong MFSAN baseline.
        # lambda_ent: target entropy minimization strength. Keep small; too large may cause over-confident wrong pseudo-labels.
        self.lambda_cda = float(getattr(args, 'lambda_cda', 0.10))
        self.lambda_ent = float(getattr(args, 'lambda_ent', 0.01))
        self.detach_prob = bool(getattr(args, 'cda_detach_prob', True))
        self.entropy_eps = 1e-5

        logging.info('CDA lambda_cda: {:.6f}'.format(self.lambda_cda))
        logging.info('CDA lambda_ent: {:.6f}'.format(self.lambda_ent))
        logging.info('CDA detach probabilities for joint feature: {}'.format(self.detach_prob))
        logging.info('CDA joint feature dim: {} x {} = {}'.format(
            num_classes, self.Fs[0].feature_dim, num_classes * self.Fs[0].feature_dim
        ))

    def save_model(self):
        torch.save({
            'G': self.G.state_dict(),
            'Fs': self.Fs.state_dict(),
            'Cs': self.Cs.state_dict(),
            'lambda_cda': self.lambda_cda,
            'lambda_ent': self.lambda_ent,
        }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.G.load_state_dict(ckpt['G'])
        self.Fs.load_state_dict(ckpt['Fs'])
        self.Cs.load_state_dict(ckpt['Cs'])

    def _set_to_train(self):
        self.G.train()
        self.Fs.train()
        self.Cs.train()

    def _set_to_eval(self):
        self.G.eval()
        self.Fs.eval()
        self.Cs.eval()

    def _target_entropy(self, probs):
        """
        Target conditional entropy minimization.

        probs: [B, C], softmax probabilities.
        Lower entropy means more confident target prediction.
        """
        probs = torch.clamp(probs, min=self.entropy_eps, max=1.0)
        return -(probs * torch.log(probs)).sum(dim=1).mean()

    def _joint_feature(self, features, probs):
        """
        Build feature-label joint representation for Conditional MMD.

        features: [B, D]
        probs:    [B, C]
        output:   [B, C*D]

        joint[b] = probs[b] \otimes normalize(features[b])
        This mimics the feature-label multilinear/Kronecker interaction used in conditional alignment,
        but keeps the implementation simple and fully compatible with the current MFSAN MK-MMD.
        """
        features = F.normalize(features, p=2, dim=1)

        if self.detach_prob:
            probs = probs.detach()

        joint = torch.bmm(probs.unsqueeze(2), features.unsqueeze(1))
        joint = joint.view(features.size(0), -1)
        return joint

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        for i in tqdm(range(self.num_iter), ascii=True):
            cur_src_idx = int(i % self.num_source)

            target_data, _ = self._get_next_batch('train')
            source_data, source_labels = self._get_next_batch(self.src[cur_src_idx], return_actual=True)
            source_labels = self._get_train_label(source_labels, label_set=self.src_labels_flat)

            self.optimizer.zero_grad()

            data = torch.cat((source_data, target_data), dim=0)

            # Shared feature extractor
            g = self.G(data)
            g_s, g_t = g.chunk(2, dim=0)

            # Current source-specific branch
            f = self.Fs[cur_src_idx](g)
            f_s, f_t = f.chunk(2, dim=0)

            y_s = self.Cs[cur_src_idx](f_s)
            y_t_cur = self.Cs[cur_src_idx](f_t)

            # Target data through all source-specific branches for original MFSAN L1 consistency
            y_t_all = [self.Cs[k](self.Fs[k](g_t)) for k in range(self.num_source)]
            probs_t_all = [F.softmax(t, dim=1) for t in y_t_all]

            # ========== Original MFSAN losses ==========
            loss_c = F.cross_entropy(y_s, source_labels)
            loss_mmd = self.mkmmd(f_s, f_t)

            loss_l1 = 0.0
            for k in range(self.num_source - 1):
                for j in range(k + 1, self.num_source):
                    loss_l1 += torch.abs(probs_t_all[k] - probs_t_all[j]).mean()
            loss_l1 /= self.num_source

            # ========== New loss 1: Conditional MMD ==========
            # Source label distribution uses source classifier prediction, not one-hot labels, to keep source/target form consistent.
            # Probabilities are detached inside _joint_feature by default, so CDA mainly aligns features under class condition.
            probs_s = F.softmax(y_s, dim=1)
            probs_t_cur = F.softmax(y_t_cur, dim=1)

            joint_s = self._joint_feature(f_s, probs_s)
            joint_t = self._joint_feature(f_t, probs_t_cur)
            loss_cda = self.cda_mkmmd(joint_s, joint_t)

            # ========== New loss 2: Target entropy minimization ==========
            # Use the fused prediction of all source classifiers to match the original MFSAN test-time ensemble logic.
            probs_t_fused = torch.stack(probs_t_all, dim=0).mean(dim=0)
            loss_ent = self._target_entropy(probs_t_fused)

            # tradeoff[0]: original MMD ramp-up
            # tradeoff[1]: original L1 ramp-up
            # tradeoff[2]: CDA/entropy ramp-up. If args.tradeoff has only two elements, fall back to 1.0.
            cda_tradeoff = self.tradeoff[2] if len(self.tradeoff) > 2 else 1.0

            loss = (
                loss_c
                + self.tradeoff[0] * loss_mmd
                + self.tradeoff[1] * loss_l1
                + cda_tradeoff * self.lambda_cda * loss_cda
                + cda_tradeoff * self.lambda_ent * loss_ent
            )

            # Log information
            epoch_acc['Source Data'] += self._get_accuracy(y_s, source_labels)
            epoch_loss['Source Classifier'] += loss_c
            epoch_loss['MMD'] += loss_mmd
            epoch_loss['L1'] += loss_l1
            epoch_loss['CDA MMD'] += loss_cda
            epoch_loss['Target Entropy'] += loss_ent
            epoch_loss['CDA Weighted'] += (cda_tradeoff * self.lambda_cda * loss_cda).detach()
            epoch_loss['Entropy Weighted'] += (cda_tradeoff * self.lambda_ent * loss_ent).detach()

            loss.backward()
            self.optimizer.step()

        logging.info(
            'MFSAN-CDA active: lambda_cda={:.6f}, lambda_ent={:.6f}, detach_prob={}'.format(
                self.lambda_cda, self.lambda_ent, self.detach_prob
            )
        )

        return epoch_acc, epoch_loss

    def _eval(self, data, actual_labels, correct, total):
        feat_tgt = self.G(data)

        logits_tgt = [
            F.softmax(self.Cs[i](self.Fs[i](feat_tgt)), dim=1)
            for i in range(self.num_source)
        ]

        # Keep original MFSAN test-time sum fusion
        pred = torch.sum(torch.stack(logits_tgt), dim=0).argmax(dim=1)

        actual_pred = self._get_actual_label(pred, label_set=self.src_labels_flat)

        # =========================================================
        # 新增：保存验证/测试阶段的预测标签和真实标签，用于计算 F1
        # 注意：actual_pred 是已经映射回 actual label 的预测结果
        # =========================================================
        if hasattr(self, "_eval_pred_list") and hasattr(self, "_eval_label_list"):
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
