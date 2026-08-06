# -*- coding: utf-8 -*-
"""
modules_bimamba.py

新增：MSCNN + BiMamba + Attention 小门控增强特征提取器。

设计目的：
1. 不替换原始 MSCNN 主干，避免破坏当前强 baseline；
2. 用 BiMamba 分支替代原来的 BiLSTM 分支，建模长程时序依赖；
3. 使用 Temporal Attention + Channel Attention 筛选关键时间片段和故障敏感通道；
4. 使用 small-gate 残差融合：
       feat = feat_mscnn + gate * feat_bimamba_att
5. 如果环境中安装了 mamba_ssm，则优先使用真实 Mamba；
   如果没有安装，则自动使用纯 PyTorch 的 LiteBiMambaMixer 兜底，避免导入报错。

使用方式：
    import modules_bimamba
    self.G = modules_bimamba.MSCNNBiMambaAttBackbone(...)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import modules


try:
    from mamba_ssm import Mamba
    HAS_MAMBA_SSM = True
except Exception:
    Mamba = None
    HAS_MAMBA_SSM = False


class TemporalAttention(nn.Module):
    """
    Temporal attention over sequence features.

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


class LiteMambaMixer(nn.Module):
    """
    纯 PyTorch 兜底版 Mamba-like token mixer。

    注意：
    - 这不是官方 mamba_ssm 的选择性扫描实现；
    - 它保留了 Mamba 风格的“归一化 + 门控 + 深度卷积 + 投影”序列混合结构；
    - 主要用于 mamba_ssm 不可用时保证代码能跑通。
    """

    def __init__(self, d_model, d_conv=4, expand=2, dropout=0.0):
        super(LiteMambaMixer, self).__init__()
        inner_dim = int(d_model * expand)
        padding = d_conv // 2

        self.in_proj = nn.Linear(d_model, inner_dim * 2)
        self.dwconv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=d_conv,
            padding=padding,
            groups=inner_dim,
            bias=True
        )
        self.out_proj = nn.Linear(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        u, v = self.in_proj(x).chunk(2, dim=-1)        # [B, T, inner]

        u = u.transpose(1, 2)                          # [B, inner, T]
        u = self.dwconv(u)

        # 当 kernel_size 为偶数且 padding=d_conv//2 时，长度可能多 1，裁剪回原长度。
        if u.size(-1) > x.size(1):
            u = u[..., :x.size(1)]

        u = u.transpose(1, 2)                          # [B, T, inner]
        y = F.silu(u) * torch.sigmoid(v)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block.

    如果 mamba_ssm 可用：
        正向 Mamba + 反向 Mamba；
    如果不可用：
        正向 LiteMambaMixer + 反向 LiteMambaMixer。

    Input / Output: [B, T, D]
    """

    def __init__(
        self,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
    ):
        super(BiMambaBlock, self).__init__()

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if HAS_MAMBA_SSM:
            self.fwd = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.bwd = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.impl = "mamba_ssm"
        else:
            self.fwd = LiteMambaMixer(
                d_model=d_model,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            self.bwd = LiteMambaMixer(
                d_model=d_model,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            self.impl = "lite_pytorch"

        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, D]
        z = self.norm(x)

        y_fwd = self.fwd(z)

        z_rev = torch.flip(z, dims=[1])
        y_bwd = self.bwd(z_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])

        y = 0.5 * (y_fwd + y_bwd)
        out = x + self.dropout(y)
        out = self.out_norm(out)
        return out


class BiMambaAttentionBranch(nn.Module):
    """
    BiMamba + Temporal Attention + Channel Attention auxiliary branch.

    Input : x [B, 1, L]
    Output: feature [B, out_dim]
    """

    def __init__(
        self,
        in_channel=1,
        stem_channels=64,
        mamba_dim=64,
        mamba_depth=2,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        out_dim=640,
        dropout=0.0,
    ):
        super(BiMambaAttentionBranch, self).__init__()
        self.out_dim = out_dim
        self.mamba_dim = mamba_dim
        self.mamba_depth = mamba_depth

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

        if stem_channels != mamba_dim:
            self.token_proj = nn.Linear(stem_channels, mamba_dim)
        else:
            self.token_proj = nn.Identity()

        self.blocks = nn.ModuleList([
            BiMambaBlock(
                d_model=mamba_dim,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                dropout=dropout,
            )
            for _ in range(mamba_depth)
        ])

        self.temporal_att = TemporalAttention(mamba_dim)

        self.proj = nn.Sequential(
            nn.Linear(mamba_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.channel_att = ChannelAttention1D(out_dim, reduction=4)

    def forward(self, x):
        # x: [B, 1, L]
        z = self.stem(x)                              # [B, C, T]
        z = z.transpose(1, 2)                         # [B, T, C]
        z = self.token_proj(z)                        # [B, T, D]

        for blk in self.blocks:
            z = blk(z)

        context, alpha = self.temporal_att(z)         # [B, D]
        feat = self.proj(context)                     # [B, out_dim]
        feat, ch_weight = self.channel_att(feat)

        return feat


class MSCNNBiMambaAttBackbone(nn.Module):
    """
    MSCNN 主干 + BiMamba-Attention 辅助分支 + small-gate 残差融合。

    输出维度与原始 MSCNN 保持一致：
        self.out_dim = MSCNN.out_dim

    融合：
        feat = feat_mscnn + gate * feat_bimamba_att

    Small-gate:
        gate = gate_max * sigmoid(raw_gate)
        gate 始终在 [0, gate_max] 内。
    """

    def __init__(
        self,
        in_channel=1,
        stem_channels=64,
        mamba_dim=64,
        mamba_depth=2,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        dropout=0.0,
        gate_init=0.01,
        gate_max=0.03,
    ):
        super(MSCNNBiMambaAttBackbone, self).__init__()

        self.mscnn = modules.MSCNN(in_channel=in_channel)
        self.out_dim = self.mscnn.out_dim

        self.bimamba_att = BiMambaAttentionBranch(
            in_channel=in_channel,
            stem_channels=stem_channels,
            mamba_dim=mamba_dim,
            mamba_depth=mamba_depth,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            out_dim=self.out_dim,
            dropout=dropout,
        )

        # ========= Small-Gate =========
        self.gate_max = float(gate_max)
        self.gate_max = max(self.gate_max, 1e-6)

        gate_init = float(gate_init)
        gate_init = min(max(gate_init, 1e-6), self.gate_max - 1e-6)

        ratio = gate_init / self.gate_max
        ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)

        raw = math.log(ratio / (1.0 - ratio))
        self.raw_gate = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

        self.fusion_norm = nn.BatchNorm1d(self.out_dim)

        self.uses_real_mamba = HAS_MAMBA_SSM

    def get_gate(self):
        """
        Return current effective gate in [0, gate_max].
        """
        return self.gate_max * torch.sigmoid(self.raw_gate)

    def forward(self, x):
        feat_mscnn = self.mscnn(x)
        feat_bimamba = self.bimamba_att(x)

        gate = self.get_gate()
        feat = feat_mscnn + gate * feat_bimamba
        feat = self.fusion_norm(feat)

        return feat
