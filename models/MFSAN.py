"""
MFSAN with optional CNN-BiMamba backbone.

Original paper:
Zhu, Y., Zhuang, F. and Wang, D., 2019, July. Aligning domain-specific distribution
and classifier for cross-domain classification from multiple sources.

This version keeps the original MFSAN training logic unchanged:
    L = L_cls + lambda_mmd * MMD + lambda_l1 * L1

Only the shared feature extractor G can be changed by --backbone:
    --backbone CNN          : original MSCNN backbone
    --backbone ResNet       : original ResNet backbone
    --backbone CNN_BiMamba  : new CNN + BiMamba backbone
"""

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

        self.src_labels_flat = sorted(list(set([label for sublist in args.label_sets[:-1] for label in sublist])))
        num_classes = len(self.src_labels_flat)

        # ========== Shared feature extractor G ==========
        # Keep MFSAN framework unchanged. Only replace G when needed.
        if args.backbone == 'CNN':
            self.G = modules.MSCNN(in_channel=1).to(self.device)
        elif args.backbone == 'ResNet':
            self.G = modules.ResNet(in_channel=1, layers=[2, 2, 2, 2]).to(self.device)
        elif args.backbone == 'CNN_BiMamba':
            self.G = modules.CNNBiMambaBackbone(
                in_channel=1,
                d_model=128,
                depth=2,
                d_state=16,
                d_conv=4,
                expand=2,
                dropout=args.dropout,
                out_dim=512
            ).to(self.device)
        else:
            raise Exception(f"unknown backbone type {args.backbone}")

        logging.info('Using backbone: {}'.format(args.backbone))
        logging.info('Backbone output dim: {}'.format(self.G.out_dim))

        # ========== Source-specific feature branches Fs ==========
        self.Fs = nn.ModuleList([
            modules.MLP(
                input_size=self.G.out_dim,
                dropout=args.dropout,
                num_layer=2,
                output_layer=False
            )
            for _ in range(self.num_source)
        ]).to(self.device)

        # ========== Source-specific classifiers Cs ==========
        self.Cs = nn.ModuleList([
            modules.MLP(
                input_size=self.Fs[i].feature_dim,
                output_size=num_classes,
                num_layer=1,
                last=None
            )
            for i in range(self.num_source)
        ]).to(self.device)

        # ========== Original MK-MMD ==========
        self.mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
            kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)]
        )

        self._init_data()

        if args.train_mode == 'source_combine':
            self.src = ['concat_source']
        else:
            self.src = args.source_name
        
        self.optimizer = self._get_optimizer([self.G, self.Fs, self.Cs])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        self.num_iter = sum([len(self.dataloaders[s]) for s in self.src])
    
    def save_model(self):
        torch.save({
            'G': self.G.state_dict(),
            'Fs': self.Fs.state_dict(),
            'Cs': self.Cs.state_dict(),
            'backbone': self.args.backbone
        }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path, map_location=self.device)
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

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        for i in tqdm(range(self.num_iter), ascii=True):
            # obtain data
            cur_src_idx = int(i % self.num_source)
            target_data, _ = self._get_next_batch('train')
            source_data, source_labels = self._get_next_batch(self.src[cur_src_idx], return_actual=True)
            source_labels = self._get_train_label(source_labels, label_set=self.src_labels_flat)
            
            # forward
            self.optimizer.zero_grad()
            data = torch.cat((source_data, target_data), dim=0)

            # shared features
            g = self.G(data)
            g_s, g_t = g.chunk(2, dim=0)

            # current source-specific branch
            f = self.Fs[cur_src_idx](g)
            f_s, f_t = f.chunk(2, dim=0)
            y_s = self.Cs[cur_src_idx](f_s)

            # target predictions through all source-specific branches
            y_t = [self.Cs[i](self.Fs[i](g_t)) for i in range(self.num_source)]
            
            # compute original MFSAN losses
            loss_c = F.cross_entropy(y_s, source_labels)
            loss_mmd = self.mkmmd(f_s, f_t)

            logits_tgt = [F.softmax(t, dim=1) for t in y_t]
            loss_l1 = 0.0
            for k in range(self.num_source - 1):
                for j in range(k + 1, self.num_source):
                    loss_l1 += torch.abs(logits_tgt[k] - logits_tgt[j]).mean()
            loss_l1 /= self.num_source

            # Original MFSAN total loss. Do not add DW/SupCon here.
            loss = loss_c + self.tradeoff[0] * loss_mmd + self.tradeoff[1] * loss_l1
            
            # log information
            epoch_acc['Source Data'] += self._get_accuracy(y_s, source_labels)
            epoch_loss['Source Classifier'] += loss_c
            epoch_loss['MMD'] += loss_mmd
            epoch_loss['L1'] += loss_l1

            # backward
            loss.backward()
            self.optimizer.step()

        return epoch_acc, epoch_loss
    
    def _eval(self, data, actual_labels, correct, total):
        feat_tgt = self.G(data)

        logits_tgt = [
            F.softmax(self.Cs[i](self.Fs[i](feat_tgt)), dim=1)
            for i in range(self.num_source)
        ]

        # Original MFSAN test-time ensemble fusion
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
