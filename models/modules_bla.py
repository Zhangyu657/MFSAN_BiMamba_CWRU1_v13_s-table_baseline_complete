# -*- coding: utf-8 -*-
"""
modules_bla.py

新增：MSCNN + BiLSTM + Attention 门控增强特征提取器。

设计目的：
1. 不替换原始 MSCNN 主干，避免破坏当前强 baseline；
2. 增加 BiLSTM 分支建模一维振动信号的长程时序依赖；
3. 使用 Temporal Attention + Channel Attention 筛选关键时间片段和故障敏感通道；
4. 使用小 gate 残差融合：feat = feat_mscnn + gate * feat_bilstm_att。

使用方式：
    import modules_bla
    self.G = modules_bla.MSCNNBiLSTMAttBackbone(...)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import modules


class TemporalAttention(nn.Module):
    """
    Temporal attention over BiLSTM output sequence.

    Input : h_seq [B, T, D]
    Output: context [B, D], alpha [B, T]
    """

    def __init__(self, input_dim):
        super(TemporalAttention, self).__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, h_seq):
        score = self.score(h_seq).squeeze(-1)          # [B, T]
        alpha = F.softmax(score, dim=1)                # [B, T]
        context = torch.sum(h_seq * alpha.unsqueeze(-1), dim=1)
        return context, alpha


class ChannelAttention1D(nn.Module):
    """
    Simple channel attention for vector features.

    Input : x [B, D]
    Output: x * sigmoid(MLP(x))
    """

    def __init__(self, channels, reduction=4):
        super(ChannelAttention1D, self).__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.net(x)
        return x * weight, weight


class BiLSTMAttentionBranch(nn.Module):
    """
    BiLSTM + Temporal Attention + Channel Attention auxiliary branch.

    Input : x [B, 1, L]
    Output: feature [B, out_dim]
    """

    def __init__(
        self,
        in_channel=1,
        stem_channels=64,
        lstm_hidden=64,
        lstm_layers=1,
        out_dim=640,
        dropout=0.0,
        bidirectional=True,
    ):
        super(BiLSTMAttentionBranch, self).__init__()
        self.bidirectional = bidirectional
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.out_dim = out_dim

        # 轻量 CNN stem：先把原始一维信号压成局部 token 序列
        self.stem = nn.Sequential(
            nn.Conv1d(in_channel, stem_channels // 2, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(stem_channels // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(stem_channels // 2, stem_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.lstm = nn.LSTM(
            input_size=stem_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_dim = lstm_hidden * (2 if bidirectional else 1)
        self.temporal_att = TemporalAttention(lstm_dim)

        self.proj = nn.Sequential(
            nn.Linear(lstm_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.channel_att = ChannelAttention1D(out_dim, reduction=4)

    def forward(self, x):
        # x: [B, 1, L]
        z = self.stem(x)             # [B, C, T]
        z = z.transpose(1, 2)        # [B, T, C]
        h_seq, _ = self.lstm(z)      # [B, T, 2H]
        context, alpha = self.temporal_att(h_seq)
        feat = self.proj(context)
        feat, ch_weight = self.channel_att(feat)
        return feat


class MSCNNBiLSTMAttBackbone(nn.Module):
    """
    MSCNN 主干 + BiLSTM-Attention 辅助分支 + small-gate 残差融合。

    输出维度与原始 MSCNN 保持一致，方便无缝接入 MFSAN / MFSAN-CDAN：
        self.out_dim = MSCNN.out_dim

    融合：
        feat = feat_mscnn + gate * feat_bilstm_att

    改进点：
    1. gate_init 从 0.05 降到 0.01；
    2. 新增 gate_max，限制 BiLSTM-Attention 分支最大贡献；
    3. 使用 gate = gate_max * sigmoid(raw_gate)，而不是简单 sigmoid；
       这样 gate 始终在 [0, gate_max] 内，且不会因为 clamp 导致梯度完全截断。
    """

    def __init__(
        self,
        in_channel=1,
        stem_channels=64,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.0,
        gate_init=0.01,
        gate_max=0.03,
    ):
        super(MSCNNBiLSTMAttBackbone, self).__init__()

        self.mscnn = modules.MSCNN(in_channel=in_channel)
        self.out_dim = self.mscnn.out_dim

        self.bilstm_att = BiLSTMAttentionBranch(
            in_channel=in_channel,
            stem_channels=stem_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            out_dim=self.out_dim,
            dropout=dropout,
            bidirectional=True,
        )

        # ========= Small-Gate =========
        # gate 被限制在 [0, gate_max]，避免 BiLSTM-Attention 分支过强干扰 MSCNN 主干。
        self.gate_max = float(gate_max)
        self.gate_max = max(self.gate_max, 1e-6)

        gate_init = float(gate_init)
        gate_init = min(max(gate_init, 1e-6), self.gate_max - 1e-6)

        # gate = gate_max * sigmoid(raw_gate)
        # 因此 sigmoid(raw_gate) = gate_init / gate_max
        ratio = gate_init / self.gate_max
        ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)

        raw = math.log(ratio / (1.0 - ratio))
        self.raw_gate = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

        self.fusion_norm = nn.BatchNorm1d(self.out_dim)

    def get_gate(self):
        """
        Return current effective gate in [0, gate_max].
        """
        return self.gate_max * torch.sigmoid(self.raw_gate)

    def forward(self, x):
        feat_mscnn = self.mscnn(x)
        feat_bla = self.bilstm_att(x)

        gate = self.get_gate()
        feat = feat_mscnn + gate * feat_bla
        feat = self.fusion_norm(feat)

        return feat