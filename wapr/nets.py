# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# WAPR, SAPR, and WBPS. A and B are (N, C, H, W).
# WAPR、SAPR、WBPS。A 和 B 的形状是 (N, C, H, W)。
import math
import inspect
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from wapr import recipe
from wapr.model_metadata import unpack_engine
from wapr.pose_groups import wbps_group_sizes


def interpolate_raster_pe(pe, spatial_hw, n_tokens, base_hw=20):
    """
    # Resize the base grid stored in pe onto the current token map.

    ## Args

        - pe: a 3-D tensor (B, T, C). The resize path reshapes the first base_hw * base_hw tokens with batch 1, so B is 1 there. The resized table keeps pe.dtype.
        - spatial_hw: (h, w) in tokens, or None. None uses a square side when n_tokens is a perfect square, and otherwise skips the resize.
        - n_tokens: the token count to keep. When spatial_hw is set, h * w must equal n_tokens.
        - base_hw: the stored grid side. The default is 20, so the source grid has 400 tokens.

    ## Returns

        - The return is (1, n_tokens, C) after bilinear resize, or pe sliced on the token axis when the table is not resized.

    ---

    # 把 pe 里存着的基准网格缩放到当前 token 图上。

    ## 参数

        - pe: 三维张量 (B, T, C)。缩放分支将前 base_hw * base_hw 个 token 按 batch 1 重排，因此该分支要求 B 为 1。缩放后的表保持 pe.dtype。
        - spatial_hw: token 网格 (h, w)，或 None。None 在 n_tokens 是完全平方时采用这个边长，否则跳过缩放。
        - n_tokens: 要保留的 token 数。给出 spatial_hw 时，h * w 必须等于 n_tokens。
        - base_hw: 存下来的网格边长。默认是 20，因此源网格有 400 个 token。

    ## 返回

        - 双线性缩放后返回 (1, n_tokens, C)。
        - 不做缩放时，返回沿 token 轴切开的 pe。

"""
    base_t = int(base_hw) * int(base_hw)
    if pe.ndim != 3 or int(pe.shape[1]) < base_t:
        return pe[:, :n_tokens]
    if spatial_hw is None:
        side = int(round(n_tokens ** 0.5))
        if side * side == n_tokens:
            spatial_hw = (side, side)
        else:
            return pe[:, :n_tokens]
    h, w = int(spatial_hw[0]), int(spatial_hw[1])
    if h * w != n_tokens:
        raise ValueError("spatial %s != n_tokens %s" % ((h, w), n_tokens))
    if h == base_hw and w == base_hw:
        return pe[:, :base_t]
    c = int(pe.shape[-1])
    grid = pe[:, :base_t].reshape(1, base_hw, base_hw, c).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=(h, w), mode="bilinear", align_corners=True)
    return grid.to(dtype=pe.dtype).flatten(2).transpose(1, 2)


class ConvNormAct(nn.Module):
    """
    # Conv2d, batch norm, then ReLU.

        forward returns the activated map.

    ## Returns

        - Padding: (k - 1) // 2, so stride 1 keeps the spatial size. ObsStem.__init__ builds down1 and down2. PairFusion.__init__ builds widen.

    ---

    # 卷积、批归一化、再 ReLU。

        forward 返回激活后的特征图。

    ## 返回

        - 填充是 (k - 1) // 2，因此 stride 为 1 时空间尺寸不变。
        - ObsStem.__init__ 构建 down1 和 down2。
        - PairFusion.__init__ 构建 widen。

"""
    def __init__(self, in_ch, out_ch, k=3, stride=1, bias=True):
        """
        # Store a Conv2d, a BatchNorm2d, and an inplace ReLU.

        ## Args

            - in_ch: the input channel count.
            - out_ch: the output channel count.
            - k: the square kernel size. The default is 3, and the padding is (k - 1) // 2.
            - stride: the Conv2d stride. The default is 1, which keeps the spatial size.
            - bias: the Conv2d bias flag. The default is True.

        ## Returns

            - Returns None.

        ---

        # 存下 Conv2d、BatchNorm2d 和原地 ReLU。

        ## 参数

            - in_ch: 输入通道数。
            - out_ch: 输出通道数。
            - k: 正方形卷积核边长。默认是 3，填充是 (k - 1) // 2。
            - stride: Conv2d 的步长。默认是 1，这时空间尺寸不变。
            - bias: Conv2d 的偏置开关。默认是 True。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=pad, bias=bias)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        # Run conv, batch norm, and inplace ReLU, and return that map.

        ## Args

            - x: (N, in_ch, H, W), the same channel count this module was built with.

        ## Returns

            - The return is (N, out_ch, H_out, W_out).
            - Stride 1 keeps H and W.
            - Any other stride is the Conv2d stride.

        ---

        # 依次做卷积、批归一化和原地 ReLU，并返回这张特征图。

        ## 参数

            - x: (N, in_ch, H, W)，通道数与构建这个模块时的输入通道一致。

        ## 返回

            - 返回 (N, out_ch, H_out, W_out)。
            - stride: 1 时保持 H 和 W。其他 stride 就是 Conv2d 的步长。

"""
        return self.act(self.norm(self.conv(x)))


class BasicBlock(nn.Module):
    """
    # Residual block.

        Two 3x3 convolutions plus a residual.

        ResidualUnit.__init__ builds the single unit.

        Spatial size changes only when stride is not 1, or when a 1x1 downsample is required.

    ## Returns

        - expansion: 1, so the output width is channels.

    ---

    # 残差块。

        两层 3x3 卷积加一条残差。

        ResidualUnit.__init__ 构建这一个单元。

        只有 stride 不是 1，或需要 1x1 下采样时，空间尺寸才变。

    ## 返回

        - expansion: 1，因此输出宽度就是 channels。

"""
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1, bias=True):
        """
        # Build the residual block.

            Two 3x3 convolutions, their batch norms, and the residual path.

        ## Args

            - in_channels: the input channel count.
            - channels: the 3x3 output width. expansion is 1, so the residual width is channels.
            - stride: applied to the first 3x3 and to the 1x1 downsample. The second 3x3 uses stride 1. The default stride is 1.
            - dilation: both the dilation and the padding of the two 3x3 convolutions. The default is 1.
            - bias: the bias flag of the two 3x3 convolutions. The 1x1 downsample convolution is bias-free. The default is True.

        ## Returns

            - Returns None.

        ---

        # 构建残差块。

            两层 3x3 卷积、它们的批归一化，以及残差路径。

        ## 参数

            - in_channels: 输入通道数。
            - channels: 3x3 的输出宽度。expansion 是 1，因此残差宽度就是 channels。
            - stride 用在第一层 3x3 和 1x1 下采样上。
            - 第二层 3x3 的 stride 是 1。
            - 默认 stride 是 1。
            - dilation 同时是两层 3x3 的膨胀和填充。
            - 默认是 1。
            - bias: 两层 3x3 的偏置开关。1x1 下采样卷积没有偏置。默认是 True。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        out_channels = self.expansion * channels
        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1,
            padding=dilation, dilation=dilation, bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        if (stride != 1) or (in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        """
        # Add the residual to the second convolution and return inplace ReLU of that sum.

            x 是 (N, in_channels, H, W)。

        ## Args

            - x: (N, in_channels, H, W).

        ## Returns

            - The return is (N, channels, H_out, W_out).
            - Stride 1 with matching channels keeps the spatial size.
            - The first ReLU sits between the two convolutions.

        ---

        # 把残差加到第二层卷积上，返回这个和的原地 ReLU。

        ## 返回

            - 返回 (N, channels, H_out, W_out)。
            - stride: 1 且通道已经对齐时，空间尺寸不变。第一个 ReLU 放在两层卷积之间。

"""
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(identity + out, inplace=True)


class ResidualUnit(nn.Module):
    """
    # One BasicBlock.

        ObsStem and PairFusion both stack this unit.

        ObsStem.__init__ builds two units at 128 channels.

        PairFusion.__init__ builds two at 256 and two at 512.

    ---

    # 一个 BasicBlock。

        ObsStem 和 PairFusion 都堆这种单元。

        ObsStem.__init__ 构建两个 128 通道的单元。

        PairFusion.__init__ 构建两个 256 通道和两个 512 通道的单元。

"""
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, bias=True):
        """
        # Store one BasicBlock.

        ## Args

            - in_ch: the BasicBlock input channel count.
            - out_ch: the BasicBlock channels argument, the 3x3 output width.
            - stride: passed to BasicBlock. The default is 1.
            - dilation: passed to BasicBlock. The default is 1.
            - bias: passed to BasicBlock. The default is True.

        ## Returns

            - Returns None.

        ---

        # 存下一个 BasicBlock。

        ## 参数

            - in_ch: BasicBlock 的输入通道数。
            - out_ch: BasicBlock 的 channels 参数，也就是 3x3 的输出宽度。
            - stride 传给 BasicBlock。
            - 默认是 1。
            - dilation 传给 BasicBlock。
            - 默认是 1。
            - bias 传给 BasicBlock。
            - 默认是 True。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        self.unit = BasicBlock(in_ch, out_ch, stride=stride, dilation=dilation, bias=bias)

    def forward(self, x):
        """
        # Run x through one BasicBlock.

            x 是 (N, in_ch, H, W)。

        ## Args

            - x: (N, in_ch, H, W).

        ## Returns

            - The return is the BasicBlock output, (N, out_ch, H_out, W_out).

        ---

        # 让 x 经过一个 BasicBlock。

        ## 返回

            - 返回值是 BasicBlock 的输出，(N, out_ch, H_out, W_out)。

"""
        return self.unit(x)


class ObsStem(nn.Module):
    """
    # Two stride-2 convolutions, then two residual units at 128 channels.

        DualViewEncoder.__init__ builds one shared stem.

        down1 maps in_channels to 64 with a 7x7 kernel.

        down2 maps 64 to 128.

        The residual units keep 128 channels and the spatial size of down2.

    ---

    # 两次 stride 为 2 的卷积，再接两个 128 通道的残差单元。

        DualViewEncoder.__init__ 构建一个共用 stem。

        down1 用 7x7 把 in_channels 变成 64。

        down2 把 64 变成 128。

        残差单元保持 128 通道和 down2 的空间尺寸。

"""
    def __init__(self, in_channels, conv_bias=True):
        """
        # Build the two stride-2 downs and two residual units.

        ## Args

            - in_channels: the image-stack channel count passed to down1.
            - conv_bias: the BasicBlock bias of the two residual units. The default is True. The two down convolutions are built with bias True.

        ## Returns

            - Returns None.

        ---

        # 构建两次 stride 为 2 的下采样和两个残差单元。

        ## 参数

            - in_channels: 传给 down1 的图像堆叠通道数。
            - conv_bias: 两个残差单元里 BasicBlock 的偏置。默认是 True。两次下采样卷积按 bias True 构建。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        self.down1 = ConvNormAct(in_channels, 64, k=7, stride=2, bias=True)
        self.down2 = ConvNormAct(64, 128, k=3, stride=2, bias=True)
        self.res = nn.ModuleList(
            [
                ResidualUnit(128, 128, bias=conv_bias),
                ResidualUnit(128, 128, bias=conv_bias),
            ]
        )

    def forward(self, x):
        """
        # Downsample x twice, run both residual units, and return the 128-channel map.

            x 是 (N, in_channels, H, W)。

        ## Args

            - x: (N, in_channels, H, W).

        ## Returns

            - The return is (N, 128, H2, W2).
            - Each down convolution has stride 2.
            - The residual units keep that spatial size.

        ---

        # 对 x 下采样两次，跑完两个残差单元，返回 128 通道的特征图。

        ## 返回

            - 返回 (N, 128, H2, W2)。
            - 每次下采样卷积的 stride 都是 2。
            - 残差单元保持这个空间尺寸。

"""
        x = self.down1(x)
        x = self.down2(x)
        for blk in self.res:
            x = blk(x)
        return x


class PairFusion(nn.Module):
    """
    # Fuse a 256-channel pair, widen it to 512 with stride 2, then refine at 512.

        DualViewEncoder.__init__ builds it.

        The 256 channels are the two stem maps concatenated on the channel axis.

    ---

    # 融合 256 通道的一对特征，用 stride 2 扩到 512，再在 512 通道上细化。

        DualViewEncoder.__init__ 构建它。

        这 256 通道是两张 stem 特征沿通道拼起来的结果。

"""
    def __init__(self, conv_bias=True):
        """
        # Build two 256-channel units, a stride-2 widen to 512, and two 512-channel units.

        ## Args

            - conv_bias: the BasicBlock bias of the four residual units. The default is True. The widen convolution uses bias True.

        ## Returns

            - Returns None.

        ---

        # 构建两个 256 通道单元、一次 stride 为 2 的扩到 512，以及两个 512 通道单元。

        ## 参数

            - conv_bias: 四个残差单元里 BasicBlock 的偏置。默认是 True。扩通道卷积使用 bias True。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        self.mid = nn.ModuleList(
            [
                ResidualUnit(256, 256, bias=conv_bias),
                ResidualUnit(256, 256, bias=conv_bias),
            ]
        )
        self.widen = ConvNormAct(256, 512, k=3, stride=2, bias=True)
        self.deep = nn.ModuleList(
            [
                ResidualUnit(512, 512, bias=conv_bias),
                ResidualUnit(512, 512, bias=conv_bias),
            ]
        )

    def forward(self, x):
        """
        # Run the 256-channel units, the stride-2 widen, and the 512-channel units.

            x 是 (N, 256, H, W)。

        ## Args

            - x: (N, 256, H, W).

        ## Returns

            - The return is (N, 512, H_out, W_out).
            - The widen stride is 2.
            - The residual units keep the spatial size around them.

        ---

        # 先跑 256 通道单元，再做 stride 为 2 的扩通道，然后跑 512 通道单元。

        ## 返回

            - 返回 (N, 512, H_out, W_out)。
            - 扩通道的 stride 是 2。
            - 残差单元保持各自前后的空间尺寸。

"""
        for blk in self.mid:
            x = blk(x)
        x = self.widen(x)
        for blk in self.deep:
            x = blk(x)
        return x


class DualViewEncoder(nn.Module):
    """
    # Share one ObsStem across A and B, then fuse the two 128-channel maps.

        RefineNet.__init__ and WbpsNet.__init__ each build one encoder.

        The stem sees the batch concatenation of A and B.

        The fusion sees the channel concatenation.

    ---

    # A 和 B 共用一个 ObsStem，再把两张 128 通道的特征图融合。

        RefineNet.__init__ 和 WbpsNet.__init__ 各自构建一个编码器。

        stem 看到的是 A 和 B 在 batch 上的拼接。

        融合看到的是通道上的拼接。

"""
    def __init__(self, in_channels, conv_bias=True):
        """
        # Store one ObsStem and one PairFusion.

        ## Args

            - in_channels: the channel count of A and of B.
            - conv_bias: passed to the stem and the fusion. The default is True.

        ## Returns

            - Returns None.

        ---

        # 存下一个 ObsStem 和一个 PairFusion。

        ## 参数

            - in_channels: A 和 B 的通道数。
            - conv_bias 传给 stem 和融合。
            - 默认是 True。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        self.stem = ObsStem(in_channels, conv_bias=conv_bias)
        self.fusion = PairFusion(conv_bias=conv_bias)

    def forward(self, A, B):
        """
        # Encode A and B with one stem and return the fused 512-channel map.

        ## Args

            - A: (N, in_channels, H, W). N is A.shape[0].
            - B: (N, in_channels, H, W) and matches A.

        ## Returns

            - The return is (N, 512, h, w) from PairFusion.

        ---

        # 用同一个 stem 编码 A 和 B，返回融合后的 512 通道特征图。

        ## 参数

            - A: (N, in_channels, H, W)。N 是 A.shape[0]。
            - B: (N, in_channels, H, W)，并且与 A 形状相同。

        ## 返回

            - 返回值是 PairFusion 给出的 (N, 512, h, w)。

"""
        # Share the stem, then fuse the two views on the channel axis.
        # 共用 stem，再沿通道把两路视图拼起来融合。
        n = A.shape[0]
        x = self.stem(torch.cat([A, B], dim=0))
        a, b = x[:n], x[n:]
        return self.fusion(torch.cat((a, b), dim=1).contiguous())


class SinusoidPosEnc(nn.Module):
    """
    # Add a sinusoidal position table.

        The table is resized when the token grid is not 20 by 20.

        RefineNet.__init__ and WbpsNet.__init__ build it with dim 512 and max_len 400.

        The buffer table is (1, max_len, dim) float32.

    ---

    # 加上一张正弦位置表。

        token 网格不是 20 乘 20 时，这张表会被缩放。

        RefineNet.__init__ 和 WbpsNet.__init__ 用 dim 512 和 max_len 400 构建它。

        缓冲 table 是 (1, max_len, dim) float32。

"""
    def __init__(self, dim, max_len=400):
        """
        # Register the sinusoidal table.

        ## Args

            - dim: the table width. Even columns are sine and odd columns are cosine, so dim is even. Callers pass 512.
            - max_len: the token capacity. The default is 400. The stored table is (1, max_len, dim) float32.

        ## Returns

            - Returns None.

        ---

        # 登记正弦位置表。

        ## 参数

            - dim: 表的宽度。偶数列是正弦，奇数列是余弦，因此 dim 是偶数。调用方传入 512。
            - max_len: token 容量。默认是 400。存下的表是 (1, max_len, dim) float32。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        table = torch.zeros(max_len, dim, dtype=torch.float32)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        table[:, 0::2] = torch.sin(pos * div)
        table[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("table", table.unsqueeze(0), persistent=True)

    def forward(self, tokens, spatial_hw=None):
        """
        # Add the resized position table and return tokens of the same shape.

        ## Args

            - tokens: (B, T, dim). The table is moved to tokens.device and tokens.dtype before the add.
            - spatial_hw: (h, w) in tokens, or None. None is forwarded to interpolate_raster_pe.

        ## Returns

            - The return has the same shape as tokens.

        ---

        # 加上缩放后的位置表，返回与 tokens 同形状的结果。

        ## 参数

            - tokens: (B, T, dim)。相加之前，位置表会转到 tokens.device 和 tokens.dtype。
            - spatial_hw: token 网格 (h, w)，或 None。None 会原样传给 interpolate_raster_pe。

        ## 返回

            - 返回值与 tokens 形状相同。

"""
        pe = interpolate_raster_pe(self.table, spatial_hw, n_tokens=int(tokens.size(1)))
        return tokens + pe.to(device=tokens.device, dtype=tokens.dtype)


class PostNormTokenBlock(nn.Module):
    """
    # Self-attention, then a feed-forward.

        Layer norm follows each residual add.

        TokenStack.__init__ builds n_layers copies.

        RefineNet.__init__ builds trans_token and rot_token.

        WbpsNet.__init__ builds rank_pose and rank_rot.

    ---

    # 先自注意力，再前馈。

        每次残差相加之后做层归一化。

        TokenStack.__init__ 构建 n_layers 份。

        RefineNet.__init__ 构建 trans_token 和 rot_token。

        WbpsNet.__init__ 构建 rank_pose 和 rank_rot。

"""
    def __init__(self, dim=512, heads=4, ff=512, dropout=0.1):
        """
        # Store batch-first multi-head attention, the feed-forward, and two layer norms.

        ## Args

            - dim: the token width. The default is 512.
            - heads: the attention head count. The default is 4.
            - ff: the hidden width of the feed-forward. The default is 512.
            - dropout: used by attention and both residual dropouts. The default is 0.1.

        ## Returns

            - Returns None.

        ---

        # 存下 batch_first 的多头注意力、前馈和两个层归一化。

        ## 参数

            - dim: token 宽度。默认是 512。
            - heads: 注意力头数。默认是 4。
            - ff: 前馈的隐藏宽度。默认是 512。
            - dropout 用于注意力和两处残差 dropout。
            - 默认是 0.1。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(dim, ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        # Run post-norm attention, then the feed-forward.

            Attention weights are discarded.

            x 是 (B, T, dim)。

        ## Args

            - x: (B, T, dim).

        ## Returns

            - The return has the same shape as x.

        ---

        # 先做后归一化注意力，再做前馈。

            注意力权重被丢掉。

        ## 返回

            - 返回值与 x 形状相同。

"""
        x2, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout1(x2))
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm2(x + self.dropout2(x2))


class TokenStack(nn.Module):
    """
    # A list of PostNormTokenBlock layers.

        The default depth is 2.

        RefineNet.__init__ and WbpsNet.__init__ build it with n_layers 2.

        Each block uses the PostNormTokenBlock defaults.

    ---

    # 一列 PostNormTokenBlock。

        默认深度是 2。

        RefineNet.__init__ 和 WbpsNet.__init__ 用 n_layers 2 构建它。

        每一层都用 PostNormTokenBlock 的默认值。

"""
    def __init__(self, n_layers=2):
        """
        # Store n_layers PostNormTokenBlock modules.

        ## Args

            - n_layers: the depth. The default is 2.

        ## Returns

            - Returns None.

        ---

        # 存下 n_layers 个 PostNormTokenBlock。

        ## 参数

            - n_layers: 层数。默认是 2。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        self.blocks = nn.ModuleList([PostNormTokenBlock() for _ in range(n_layers)])

    def forward(self, x):
        """
        # Run every block in order and return the tokens.

            x 是 (B, T, dim)。

        ## Args

            - x: (B, T, dim).

        ## Returns

            - The return has the same shape as x.

        ---

        # 按顺序跑完每一层，返回 token。

        ## 返回

            - 返回值与 x 形状相同。

"""
        for blk in self.blocks:
            x = blk(x)
        return x


class LearnedQueryPool(nn.Module):
    """
    # One learned query attends over the tokens and returns one vector per batch row.

        RefineNet.__init__ builds trans_pool and rot_pool.

        WbpsNet.__init__ builds pool_pose and pool_rot.

        All four use dim 512 and 4 heads.

    ---

    # 一个可学习 query 对 token 做注意力，每个 batch 行返回一个向量。

        RefineNet.__init__ 构建 trans_pool 和 rot_pool。

        WbpsNet.__init__ 构建 pool_pose 和 pool_rot。

        四个都用 dim 512 和 4 个头。

"""
    def __init__(self, dim=512, heads=4):
        """
        # Store one query parameter and the Q, K, and V linears.

            The query parameter is (1, 1, dim), drawn from randn and multiplied by 0.02.

        ## Args

            - dim: the token width and the query width. The default is 512. The attention view requires dim to be divisible by heads.
            - heads: the head count. head_dim is dim // heads. The default is 4.

        ## Returns

            - Returns None.

        ---

        # 存下一个 query 参数，以及 Q、K、V 三个线性层。

            query 参数是 (1, 1, dim)，取自 randn 再乘 0.02。

        ## 参数

            - dim: token 宽度，也是 query 宽度。默认是 512。注意力的 view 要求 dim 能被 heads 整除。
            - heads: 头数。head_dim 是 dim // heads。默认是 4。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.wq = nn.Linear(dim, dim, bias=True)
        self.wk = nn.Linear(dim, dim, bias=True)
        self.wv = nn.Linear(dim, dim, bias=True)

    def forward(self, tokens):
        """
        # Attend from the learned query and return one vector per batch row.

            tokens 是 (B, T, dim)。

        ## Args

            - tokens: (B, T, dim).

        ## Returns

            - The return is (B, dim).

        ---

        # 用可学习 query 做注意力，每个 batch 行返回一个向量。

        ## 返回

            - 返回 (B, dim)。

"""
        b, t, c = tokens.shape
        q = self.wq(self.query).expand(b, 1, c)
        k = self.wk(tokens)
        v = self.wv(tokens)
        h, dh = self.heads, self.head_dim
        q = q.view(b, 1, h, dh).transpose(1, 2)
        k = k.view(b, t, h, dh).transpose(1, 2)
        v = v.view(b, t, h, dh).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (dh ** -0.5)
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v)
        return y.transpose(1, 2).reshape(b, c)


# WAPR and SAPR share this refinement net. Their weights and rotation steps differ.
# WAPR 与 SAPR 共用这一修正网络，权重和旋转步长不同。
class RefineNet(nn.Module):
    """
    # WAPR and SAPR refinement network.

        forward returns raw translation and rotation heads.

        load_net builds it for every recipe.channels name other than wbps.

        export_public_weights.public_state builds it the same way.

        in_channels comes from recipe.channels.

    ---

    # WAPR 和 SAPR 的修正网络。

        forward 返回未经缩放的平移头和旋转头。

        除了 wbps，recipe.channels 里的每个名字都由 load_net 构建它。

        export_public_weights.public_state 以同样方式构建。

        in_channels 来自 recipe.channels。

"""
    def __init__(self, in_channels):
        """
        # Build the shared encoder, position table, token trunk, and the two 3-D heads.

        ## Args

            - in_channels: stored on self.in_channels and passed to DualViewEncoder. recipe.channels uses 7 for wapr_w_mask and 6 for wapr_wo_mask and sapr.

        ## Returns

            - Returns None.

        ---

        # 构建共用编码器、位置表、token 堆叠，以及两个三维输出头。

        ## 参数

            - in_channels 存在 self.in_channels 上，并传给 DualViewEncoder。
            - recipe.channels 里 wapr_w_mask 是 7，wapr_wo_mask 和 sapr 是 6。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        self.in_channels = int(in_channels)
        self.encoder = DualViewEncoder(in_channels)
        self.pos = SinusoidPosEnc(512, max_len=400)
        self.token_trunk = TokenStack(n_layers=2)
        self.trans_token = PostNormTokenBlock()
        self.rot_token = PostNormTokenBlock()
        self.trans_pool = LearnedQueryPool(512, heads=4)
        self.rot_pool = LearnedQueryPool(512, heads=4)
        self.trans_out = nn.Linear(512, 3)
        self.rot_out = nn.Linear(512, 3)

    def forward(self, A, B):
        """
        # Encode one view pair.

            Raw trans and rot come back, each shaped (N, 3).

            A 是 (N, in_channels, H, W)。

        ## Args

            - A: (N, in_channels, H, W).
            - B: (N, in_channels, H, W) and matches A. The encoder map is tokenized as (N, H * W, 512).

        ## Returns

            - trans: (N, 3), the translation linear head.
            - rot: (N, 3), the rotation linear head.

        ---

        # 编码一对视图。

            返回未经缩放的 trans 和 rot，形状都是 (N, 3)。

        ## 参数

            - B: (N, in_channels, H, W)，并且与 A 形状相同。编码器输出被排成 token (N, H * W, 512)。

        ## 返回

            - trans: (N, 3)，平移线性头的输出。
            - rot: (N, 3)，旋转线性头的输出。

"""
        # trans and rot are each (N, 3). rot is later limited in the estimator.
        # trans 和 rot 都是 (N, 3)。rot 随后在 estimator 里限幅。
        feat = self.encoder(A, B)
        b, c, h, w = feat.shape
        tok = feat.reshape(b, c, h * w).permute(0, 2, 1)
        tok = self.token_trunk(self.pos(tok, spatial_hw=(h, w)))
        return {
            "trans": self.trans_out(self.trans_pool(self.trans_token(tok))),
            "rot": self.rot_out(self.rot_pool(self.rot_token(tok))),
        }


# WBPS scores each hypothesis within its group and against the other groups.
# WBPS 给每个候选姿态打组内分，以及相对其他组的组间分。
class WbpsNet(nn.Module):
    """
    # WBPS scorer.

        forward returns a within-group score and a between-group score for every hypothesis.

        load_net builds it when name is wbps.

        export_public_weights.public_state builds it for that same name.

        in_channels is recipe.channels wbps, which is 6.

    ---

    # WBPS 打分网络。

        forward 为每个候选姿态返回组内分和组间分。

        name 为 wbps 时由 load_net 构建。

        export_public_weights.public_state 对同一个名字构建它。

        in_channels 是 recipe.channels 里的 wbps，即 6。

"""
    def __init__(self, in_channels):
        """
        # Build the encoder, token trunk, two query pools, two rank blocks, and two scalar heads.

        ## Args

            - in_channels: stored on self.in_channels and passed to DualViewEncoder.

        ## Returns

            - Returns None.

        ---

        # 构建编码器、token 堆叠、两个查询池、两个排序块和两个标量头。

        ## 参数

            - in_channels 存在 self.in_channels 上，并传给 DualViewEncoder。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        self.in_channels = int(in_channels)
        self.encoder = DualViewEncoder(in_channels)
        self.pos = SinusoidPosEnc(512, max_len=400)
        self.token_trunk = TokenStack(n_layers=2)
        self.pool_pose = LearnedQueryPool(512, heads=4)
        self.pool_rot = LearnedQueryPool(512, heads=4)
        self.rank_pose = PostNormTokenBlock()
        self.rank_rot = PostNormTokenBlock()
        self.head_pose = nn.Linear(512, 1)
        self.head_rot = nn.Linear(512, 1)

    def forward(self, A, B, L):
        """
        # Score L hypotheses in each group and return both score rows.

        ## Args

            - A: (N, in_channels, H, W). N is divisible by L.
            - B: (N, in_channels, H, W) and matches A.
            - L: the hypothesis count in one group. It is a positive int. A non-positive L, or an N that is not divisible by L, raises ValueError.

        ## Returns

            - within_group: (N / L, L), the head on the rotation-pool tokens.
            - between_group: (N / L, L), the head on the pose-pool tokens.

        ---

        # 为每一组里的 L 个候选姿态打分，返回两行分数。

        ## 参数

            - A: (N, in_channels, H, W)。N 能被 L 整除。
            - B: (N, in_channels, H, W)，并且与 A 形状相同。
            - L: 一组里的候选数量。它是正整数。L 不是正数，或 N 不能被 L 整除时，抛出 ValueError。

        ## 返回

            - within_group: (N / L, L)，来自旋转池 token 上的那个头。
            - between_group: (N / L, L)，来自位姿池 token 上的那个头。

"""
        # L hypotheses form one group. Both outputs are (N/L, L).
        # L 个候选姿态组成一组。两个输出都是 (N/L, L)。
        L = int(L)
        if L <= 0 or A.shape[0] % L != 0:
            raise ValueError("batch %d L %d" % (int(A.shape[0]), L))
        feat = self.encoder(A, B)
        b, c, h, w = feat.shape
        tok = feat.reshape(b, c, h * w).permute(0, 2, 1)
        tok = self.token_trunk(self.pos(tok, spatial_hw=(h, w)))
        n = b // L
        pose_bank = self.pool_pose(tok).reshape(n, L, -1)
        rot_bank = self.pool_rot(tok).reshape(n, L, -1)
        between_group = self.head_pose(self.rank_pose(pose_bank)).reshape(n, L)
        within_group = self.head_rot(self.rank_rot(rot_bank)).reshape(n, L)
        return {"within_group": within_group, "between_group": between_group}


def _strip_module(src):
    """
    # Unwrap a checkpoint dict and drop one leading module.

        prefix.

    ## Args

        - src: a mapping of parameter names to tensors. A model_state_dict mapping is used when that value is a dict. Otherwise a state_dict mapping is used when that value is a dict.

    ## Returns

        - The return is a new dict.
        - A key that starts with module.
        - loses that prefix once.
        - Tensor values are the same objects.

    ---

    # 拆开权重字典，并去掉一层开头的 module. 前缀。

    ## 参数

        - src: 参数名到张量的映射。model_state_dict 的值本身是 dict 时用它。否则在 state_dict 的值是 dict 时用它。

    ## 返回

        - 返回一个新字典。
        - 以 module. 开头的键只去掉这一层前缀。
        - 张量值仍是原来的对象。

"""
    if isinstance(src, dict) and isinstance(src.get("model_state_dict"), dict):
        src = src["model_state_dict"]
    elif isinstance(src, dict) and isinstance(src.get("state_dict"), dict):
        src = src["state_dict"]
    out = {}
    for key, value in src.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        out[key] = value
    return out


def _map_convnorm(old, new, key):
    """
    # Map one stored conv-then-norm key onto ConvNormAct, or return None.

    ## Args

        - old: the stored prefix, including its trailing dot.
        - new: the ConvNormAct prefix, also with a trailing dot.
        - key: the full stored parameter name.

    ## Returns

        - The return is a string.
        - net.0.
        - under old becomes new plus conv.
        - net.1.
        - becomes new plus norm.
        - Any other key, including a key that misses old, returns None.

    ---

    # 把一条先卷积后归一化的存盘键名改到 ConvNormAct，无法对应时返回 None。

    ## 参数

        - old: 存盘前缀，带末尾的点。
        - new: ConvNormAct 的前缀，同样带末尾的点。
        - key: 完整的存盘参数名。

    ## 返回

        - 返回字符串。
        - old 下面的 net.0. 变成 new 加 conv.。
        - net.1. 变成 new 加 norm.。
        - 其他键，包括不以 old 开头的键，返回 None。

"""
    if not key.startswith(old):
        return None
    rest = key[len(old) :]
    if rest.startswith("net.0."):
        return new + "conv." + rest[len("net.0.") :]
    if rest.startswith("net.1."):
        return new + "norm." + rest[len("net.1.") :]
    return None


def _map_residual(old, new, key):
    """
    # Map one stored residual key onto ResidualUnit.unit, or return None.

    ## Args

        - old: the stored block prefix.
        - new: the ResidualUnit prefix.
        - key: the full stored parameter name.

    ## Returns

        - The return is a string.
        - conv1, conv2, bn1, and bn2 stay under new plus unit.
        - downsample: rewritten to unit.downsample. A key that misses old, or any other suffix, returns None.

    ---

    # 把一条存盘残差键名改到 ResidualUnit.unit，无法对应时返回 None。

    ## 参数

        - old: 存盘块的前缀。
        - new: ResidualUnit 的前缀。
        - key: 完整的存盘参数名。

    ## 返回

        - 返回字符串。
        - conv1、conv2、bn1、bn2 留在 new 加 unit 下面。
        - downsample 改写到 unit.downsample。
        - 不以 old 开头的键，以及其他后缀，返回 None。

"""
    if not key.startswith(old):
        return None
    rest = key[len(old) :]
    if rest.startswith("conv1.") or rest.startswith("conv2.") or rest.startswith("bn1.") or rest.startswith("bn2."):
        return new + "unit." + rest
    if rest.startswith("downsample."):
        return new + "unit.downsample." + rest[len("downsample.") :]
    return None


def _map_pool(old, new, key):
    """
    # Map one stored query-pool key onto LearnedQueryPool, or return None.

    ## Args

        - old: the stored pool name without a dot.
        - new: the LearnedQueryPool name.
        - key: the full stored parameter name. It must start with old plus a dot.

    ## Returns

        - The return is new plus the renamed suffix, or None.
        - q becomes query.
        - proj_q, proj_k, and proj_v become wq, wk, and wv, for weight and bias.

    ---

    # 把一条存盘查询池键名改到 LearnedQueryPool，无法对应时返回 None。

    ## 参数

        - old: 不带点的存盘池名。
        - new: LearnedQueryPool 的名字。
        - key: 完整的存盘参数名。它必须以 old 加一个点开头。

    ## 返回

        - 返回 new 加上改名后的后缀，或 None。
        - q 变成 query。
        - proj_q、proj_k、proj_v 的 weight 和 bias 变成 wq、wk、wv。

"""
    prefix = old + "."
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix) :]
    table = {
        "q": "query",
        "proj_q.weight": "wq.weight",
        "proj_q.bias": "wq.bias",
        "proj_k.weight": "wk.weight",
        "proj_k.bias": "wk.bias",
        "proj_v.weight": "wv.weight",
        "proj_v.bias": "wv.bias",
    }
    if rest not in table:
        return None
    return new + "." + table[rest]


def _map_encoder(key, stem0, stem1, res0, res1, mid0, mid1, widen, deep0, deep1):
    """
    # Map one stored encoder key onto DualViewEncoder, or return None.

    ## Args

        - key: the full stored parameter name.
        - stem0: the conv-norm prefix mapped to encoder.stem.down1.
        - stem1: the conv-norm prefix mapped to encoder.stem.down2.
        - res0: the residual prefix for encoder.stem.res.0.
        - res1: the residual prefix for encoder.stem.res.1.
        - mid0: the residual prefix for encoder.fusion.mid.0.
        - mid1: the residual prefix for encoder.fusion.mid.1.
        - widen: the conv-norm prefix for encoder.fusion.widen.
        - deep0: the residual prefix for encoder.fusion.deep.0.
        - deep1: the residual prefix for encoder.fusion.deep.1.

    ## Returns

        - The return is the renamed key, or None when none of those prefixes match.

    ---

    # 把一条存盘编码器键名改到 DualViewEncoder，无法对应时返回 None。

    ## 参数

        - key: 完整的存盘参数名。
        - stem0: 先卷积后归一化的前缀，对应 encoder.stem.down1。
        - stem1: 先卷积后归一化的前缀，对应 encoder.stem.down2。
        - res0: 残差前缀，对应 encoder.stem.res.0。
        - res1: 残差前缀，对应 encoder.stem.res.1。
        - mid0: 残差前缀，对应 encoder.fusion.mid.0。
        - mid1: 残差前缀，对应 encoder.fusion.mid.1。
        - widen: 先卷积后归一化的前缀，对应 encoder.fusion.widen。
        - deep0: 残差前缀，对应 encoder.fusion.deep.0。
        - deep1: 残差前缀，对应 encoder.fusion.deep.1。

    ## 返回

        - 返回改名后的键。
        - 这些前缀都不匹配时返回 None。

"""
    for old, new in ((stem0, "encoder.stem.down1."), (stem1, "encoder.stem.down2.")):
        mapped = _map_convnorm(old, new, key)
        if mapped:
            return mapped
    for old, new in (
        (res0, "encoder.stem.res.0."),
        (res1, "encoder.stem.res.1."),
        (mid0, "encoder.fusion.mid.0."),
        (mid1, "encoder.fusion.mid.1."),
        (deep0, "encoder.fusion.deep.0."),
        (deep1, "encoder.fusion.deep.1."),
    ):
        mapped = _map_residual(old, new, key)
        if mapped:
            return mapped
    return _map_convnorm(widen, "encoder.fusion.widen.", key)


def _remap_refine(src):
    """
    # Rename a WAPR or SAPR weight dict onto RefineNet.

        Unrecognized keys are omitted.

    ## Args

        - src: a name-to-tensor mapping, already unwrapped by the caller when it came from _strip_module.

    ## Returns

        - The return is a new dict.
        - Encoder keys, pos_embed.pe, shared.layers, the trans and rot heads, and the two pools are renamed.
        - Tensor values are the same objects.

    ---

    # 把 WAPR 或 SAPR 的权重字典改名到 RefineNet。

        认不出的键不写入结果。

    ## 参数

        - src: 名字到张量的映射。若它来自 _strip_module，调用方已经拆掉了外层包装。

    ## 返回

        - 返回一个新字典。
        - 编码器键、pos_embed.pe、shared.layers、平移和旋转头，以及两个池会被改名。
        - 张量值仍是原来的对象。

"""
    out = {}
    for key, value in src.items():
        mapped = _map_encoder(
            key,
            "encodeA.0.", "encodeA.1.",
            "encodeA.2.", "encodeA.3.",
            "encodeAB.0.", "encodeAB.1.",
            "encodeAB.2.", "encodeAB.3.", "encodeAB.4.",
        )
        if mapped is None and key == "pos_embed.pe":
            mapped = "pos.table"
        if mapped is None and key.startswith("shared.layers."):
            mapped = "token_trunk.blocks." + key[len("shared.layers.") :]
        if mapped is None and key.startswith("trans_head.0."):
            mapped = "trans_token." + key[len("trans_head.0.") :]
        if mapped is None and key.startswith("rot_head.0."):
            mapped = "rot_token." + key[len("rot_head.0.") :]
        if mapped is None and key.startswith("trans_head.1."):
            mapped = "trans_out." + key[len("trans_head.1.") :]
        if mapped is None and key.startswith("rot_head.1."):
            mapped = "rot_out." + key[len("rot_head.1.") :]
        if mapped is None:
            mapped = _map_pool("trans_pool", "trans_pool", key)
        if mapped is None:
            mapped = _map_pool("rot_pool", "rot_pool", key)
        if mapped is not None:
            out[mapped] = value
    return out


def _remap_wbps(src):
    """
    # Rename a WBPS weight dict onto WbpsNet.

        Unrecognized keys are omitted.

    ## Args

        - src: a name-to-tensor mapping.

    ## Returns

        - The return is a new dict.
        - Encoder keys, pos_embed.pe, att.layers, the two rank blocks, the two linear heads, and pool_1 and pool_2 are renamed.

    ---

    # 把 WBPS 的权重字典改名到 WbpsNet。

        认不出的键不写入结果。

    ## 参数

        - src: 名字到张量的映射。

    ## 返回

        - 返回一个新字典。
        - 编码器键、pos_embed.pe、att.layers、两个排序块、两个线性头，以及 pool_1 和 pool_2 会被改名。

"""
    out = {}
    for key, value in src.items():
        mapped = _map_encoder(
            key,
            "encoderA.0.", "encoderA.1.",
            "encoderA.2.", "encoderA.3.",
            "encoderAB.0.", "encoderAB.1.",
            "encoderAB.2.", "encoderAB.3.", "encoderAB.4.",
        )
        if mapped is None and key == "pos_embed.pe":
            mapped = "pos.table"
        if mapped is None and key.startswith("att.layers."):
            mapped = "token_trunk.blocks." + key[len("att.layers.") :]
        if mapped is None and key.startswith("att_cross."):
            mapped = "rank_pose." + key[len("att_cross.") :]
        if mapped is None and key.startswith("att_cross_2."):
            mapped = "rank_rot." + key[len("att_cross_2.") :]
        if mapped is None and key.startswith("linear."):
            mapped = "head_pose." + key[len("linear.") :]
        if mapped is None and key.startswith("linear_2."):
            mapped = "head_rot." + key[len("linear_2.") :]
        if mapped is None:
            mapped = _map_pool("pool_1", "pool_pose", key)
        if mapped is None:
            mapped = _map_pool("pool_2", "pool_rot", key)
        if mapped is not None:
            out[mapped] = value
    return out


def _load_weights(module, path, kind):
    """
    # Load a checkpoint into a RefineNet or WbpsNet.

        kind wbps renames with _remap_wbps.

        Any other kind renames with _remap_refine.

        A dict that already contains encoder.stem.down1.conv.weight is loaded without renaming.

        A key starting with encodeAB.5.fc or encoderAB.5.fc raises RuntimeError.

        A missing destination key, a mismatched tensor shape, or a stem input width different from module.in_channels also raises RuntimeError.

        Extra keys are ignored.

    ## Args

        - module: the network. It must expose in_channels and encoder.stem.down1.conv.weight.
        - path: a file torch.load can read on CPU. weights_only is False.

    ## Returns

        - Returns None.

    ---

    # 把权重载入 RefineNet 或 WbpsNet。

        kind 为 wbps 时用 _remap_wbps 改名。

        其他 kind 用 _remap_refine 改名。

        字典里已经有 encoder.stem.down1.conv.weight 时，直接载入，不再改名。

        以 encodeAB.5.fc 或 encoderAB.5.fc 开头的键会抛出 RuntimeError。

        目标键缺失、张量形状不一致，或 stem 输入宽度与 module.in_channels 不同，也会抛出 RuntimeError。

        多余的键被忽略。

    ## 参数

        - module: 这个网络。它必须有 in_channels 和 encoder.stem.down1.conv.weight。
        - path: torch.load 能在 CPU 上读取的文件。weights_only 是 False。

    ## 返回

        - 返回 None。

"""
    # Keys that already match this module are kept. Any other layout is renamed onto it.
    # 键名已经和本模块一致就直接用。其他布局会改名到本模块上。
    # PyTorch 1.11 predates weights_only; keep the same state-dict loader there.
    # PyTorch 1.11 尚无 weights_only；该版本继续使用相同的参数字典加载流程。
    load_options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = False
    src = _strip_module(torch.load(path, **load_options))
    for key in src:
        if key.startswith("encodeAB.5.fc.") or key.startswith("encoderAB.5.fc."):
            raise RuntimeError("checkpoint does not match")
    if "encoder.stem.down1.conv.weight" in src:
        mapped = src
    elif kind == "wbps":
        mapped = _remap_wbps(src)
    else:
        mapped = _remap_refine(src)
    dst = module.state_dict()
    filtered = {}
    for key, value in mapped.items():
        if key not in dst:
            continue
        if tuple(value.shape) != tuple(dst[key].shape):
            raise RuntimeError("checkpoint does not match")
        filtered[key] = value
    missing, _unexpected = module.load_state_dict(filtered, strict=False)
    if missing:
        raise RuntimeError("checkpoint does not match")
    stem_in = int(module.encoder.stem.down1.conv.weight.shape[1])
    if stem_in != int(module.in_channels):
        raise RuntimeError("checkpoint does not match")


class _Engine:
    """
    # One TensorRT CUDA engine.

        run binds FP32 output buffers and returns them by name.

        load_net builds it from an existing engine file.

        WAPR and SAPR use output names trans and rot.

        WBPS uses within_group and between_group.

        max_batch is the profile maximum of input A.

    ---

    # 一份 TensorRT CUDA 引擎。

        run 绑定 FP32 输出缓冲，并按名字返回它们。

        load_net 从一个已经存在的引擎文件构建它。

        WAPR 和 SAPR 的输出名是 trans 和 rot。

        WBPS 的输出名是 within_group 和 between_group。

        max_batch 是输入 A 的 profile 最大值。

"""
    def __init__(self, path, out_names):
        """
        # Deserialize one engine file and keep its execution context.

        ## Args

            - path: a native TensorRT engine or WAPR engine container. A failed deserialize raises RuntimeError.
            - out_names: the output tensor names, stored as a list. The input profile read here is A, profile 0. max_batch is that profile's maximum shape at dimension 0.

        ## Returns

            - Returns None.

        ---

        # 反序列化一份引擎文件，并保留它的执行上下文。

        ## 参数

            - path: 原生 TensorRT 引擎或 WAPR 引擎封装的路径。反序列化失败时抛出 RuntimeError。
            - out_names: 输出张量名，按列表存下。这里读取的输入 profile 是 A 的 profile 0。max_batch 是该 profile 最大形状的第 0 维。

        ## 返回

            - 返回 None。

"""
        import tensorrt as trt

        # tensorrt ships without type information, so these classes are looked up by name.
        # tensorrt 没有类型信息，所以按名字取出这些类。
        Logger = getattr(trt, "Logger")
        Runtime = getattr(trt, "Runtime")
        logger = Logger(getattr(Logger, "ERROR"))
        runtime = Runtime(logger)
        blob, self.metadata = unpack_engine(path.read_bytes())
        self.engine = runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError("engine")
        self.context = self.engine.create_execution_context()
        self.out_names = list(out_names)
        min_shape, opt_shape, max_shape = self.engine.get_tensor_profile_shape("A", 0)
        self.min_batch = int(min_shape[0])
        self.opt_batch = int(opt_shape[0])
        self.max_batch = int(max_shape[0])

    def run(self, feeds):
        """
        # Bind every feed, allocate one FP32 output per name, and execute on the current CUDA stream.

        ## Args

            - feeds: an input name to a CUDA tensor. NetRunner passes A and B. The first tensor supplies the device. This method does not cast the inputs and does not synchronize the stream.

        ## Returns

            - The return maps each name in out_names to a float32 tensor.
            - The shape comes from the execution context after the input shapes are set.

        ---

        # 绑定每一个输入，为每个输出名分配一块 FP32 缓冲，并在当前 CUDA stream 上执行。

        ## 参数

            - feeds 把输入名映射到 CUDA 张量。
            - NetRunner 传入 A 和 B。
            - 第一个张量提供 device。
            - 这个方法不转换输入的 dtype，也不同步 stream。

        ## 返回

            - 返回值把 out_names 里的每个名字映射到一个 float32 张量。
            - 形状来自设置完输入形状之后的执行上下文。

"""
        first = next(iter(feeds.values()))
        for name, tensor in feeds.items():
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        outputs = {}
        for name in self.out_names:
            shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
            buf = torch.empty(shape, device=first.device, dtype=torch.float32)
            self.context.set_tensor_address(name, int(buf.data_ptr()))
            outputs[name] = buf
        stream = torch.cuda.current_stream(first.device)
        self.context.execute_async_v3(stream.cuda_stream)
        return outputs


class NetRunner:
    """
    # Call the FP16 engine when one was loaded.

        A None engine calls the module.

        load_net builds it after the checkpoint is loaded.

        name wbps returns within_group and between_group.

        Every other loaded name returns trans and rot.

    ---

    # 载入了 FP16 引擎就走引擎。

        engine 为 None 时调用模块。

        权重载入之后由 load_net 构建。

        name 为 wbps 时返回 within_group 和 between_group。

        其他已载入的名字返回 trans 和 rot。

"""
    def __init__(self, name, module, engine):
        """
        # Store the network name, the module, and the optional engine.

        ## Args

            - name: wbps or a refine name from recipe.channels.
            - module: WbpsNet or RefineNet.
            - engine: an _Engine, or None when load_net was asked not to open one.

        ## Returns

            - Returns None.

        ---

        # 存下网络名字、模块，以及可选的引擎。

        ## 参数

            - name: wbps，或 recipe.channels 里的一个修正网络名字。
            - module: WbpsNet 或 RefineNet。
            - engine: _Engine；load_net 被要求不打开引擎时为 None。

        ## 返回

            - 返回 None。

"""
        self.name = name
        self.module = module
        self.engine = engine
        self.batch_engine = None

    def _engine_batch(self, A, B):
        """
        # Run A and B through the engine and return the named output dict.

            For name wbps the return has within_group and between_group.

            Otherwise it has trans and rot.

            Values are the engine tensors.

        ## Args

            - A: a tensor passed as an engine feed. This method does not make it contiguous.
            - B: a tensor passed as an engine feed, with the same leading size as A.

        ---

        # 让 A 和 B 走引擎，返回按名字整理好的输出字典。

            name 为 wbps 时，返回值含 within_group 和 between_group。

            否则含 trans 和 rot。

            值就是引擎张量。

        ## 参数

            - A: 作为引擎输入传入的张量。这个方法不会把它变成 contiguous。
            - B: 作为引擎输入传入的张量，第 0 维与 A 相同。

"""
        raw = self.engine.run({"A": A, "B": B})
        if self.name == "wbps":
            return {"within_group": raw["within_group"], "between_group": raw["between_group"]}
        return {"trans": raw["trans"], "rot": raw["rot"]}

    def __call__(self, A, B, L=None):
        """
        # Run one batch through the engine or the module and return the score or the refine dict.

        ## Args

            - A: a torch tensor (N, C, H, W). The engine path makes it contiguous before the run.
            - B: a torch tensor (N, C, H, W) with the same leading size as A. The engine path makes it contiguous too.
            - L: hypotheses per object. N must be divisible by L. Several groups use wbps_batch.engine in one forward; attention remains independent per object. Independent refinement rows below the engine minimum are padded and then trimmed.

        ## Returns

            - WBPS returns (N / L, L) scores; refinement returns one result per input row.
            - wbps keys are within_group and between_group.
            - Refine keys are trans and rot.

        ---

        # 让一个 batch 走引擎或走模块，返回打分字典或修正字典。

        ## 参数

            - A: torch 张量 (N, C, H, W)。走引擎时，运行前会把它变成 contiguous。
            - B: torch 张量 (N, C, H, W)，第 0 维与 A 相同。走引擎时也会把它变成 contiguous。
            - L: 每物体候选数量，N 必须能被 L 整除。多组由 wbps_batch.engine 一次前向处理，注意力保持每物体独立。独立修正行少于引擎最小 batch 时补齐，输出后裁去补充行。

        ## 返回

            - WBPS 返回 (N / L, L) 分数；修正网络为每个输入行返回一个结果。
            - wbps 的键是 within_group 和 between_group。
            - 修正网络的键是 trans 和 rot。

"""
        count = int(A.shape[0])
        group = count if L is None else int(L)
        if count <= 0 or int(B.shape[0]) != count:
            raise ValueError("A and B require the same positive row count")
        # Only WBPS has attention groups; refinement rows are independent.
        # 仅 WBPS 有注意力候选组；修正网络的各行相互独立。
        if self.name == "wbps" and (group not in wbps_group_sizes or count % group):
            raise ValueError("Expected complete supported pose groups; got N=%d, L=%d" % (count, group))
        if self.engine is not None:
            if count > int(self.engine.max_batch):
                raise ValueError(
                    "engine pose count %d is outside profile [%d, %d]"
                    % (count, self.engine.min_batch, self.engine.max_batch)
                )
            if count < int(self.engine.min_batch):
                if self.name == "wbps":
                    raise ValueError("WBPS cannot pad a hypothesis group / WBPS 不能补充候选组")
                # Evaluation refinement has no cross-row attention. Pad only the
                # engine buffer, then discard extra outputs; no scoring poses are added.
                # 评价模式的修正网络没有跨行注意力。仅补齐引擎缓冲并丢弃多余
                # 输出，不向评分阶段增加候选姿态。
                padding = int(self.engine.min_batch) - count
                padded_A = torch.cat((A, A[-1:].expand(padding, *A.shape[1:])), dim=0)
                padded_B = torch.cat((B, B[-1:].expand(padding, *B.shape[1:])), dim=0)
                raw = self._engine_batch(padded_A.contiguous(), padded_B.contiguous())
                return {name: value[:count] for name, value in raw.items()}
            if self.name == "wbps":
                if count != group:
                    # A separate object axis prevents attention from mixing identities.
                    # 独立物体维避免注意力混合不同实例；不退回逐物体调用。
                    if self.batch_engine is None:
                        path = Path(recipe.engine_file("wbps")).with_name("wbps_batch.engine")
                        if not path.is_file():
                            raise FileNotFoundError("Build the batched WBPS engine on this GPU: python -m wapr.export_batch_engine")
                        self.batch_engine = _Engine(path, ("within_group", "between_group"))
                    if count > self.batch_engine.max_batch:
                        raise ValueError("WBPS batch exceeds the engine row capacity")
                    layout = torch.empty((count // group, group, 1), device=A.device, dtype=torch.float32)
                    raw = self.batch_engine.run({"A": A.contiguous(), "B": B.contiguous(), "layout": layout})
                    expected = (count // group, group)
                    if any(tuple(value.shape) != expected for value in raw.values()):
                        raise RuntimeError("WBPS engine did not preserve the object groups")
                    return raw
            return self._engine_batch(A.contiguous(), B.contiguous())
        if self.name == "wbps":
            return self.module(A, B, L=group)
        return self.module(A, B)


def load_net(name, ckpt, engine=None, device="cpu"):
    """
    # Build one network, load its checkpoint, and return a NetRunner.

        name must be a key of recipe.channels.

        wbps builds WbpsNet.

        Every other name builds RefineNet.

        The channel count is recipe.channels[name].

        engine None leaves NetRunner.engine None.

        A path that is not a file raises FileNotFoundError.

        An existing file is opened as an _Engine.

        WAPR and SAPR output names are trans and rot.

        WBPS names are within_group and between_group.

    ## Args

        - ckpt: the checkpoint path passed to _load_weights.
        - device: passed to module.to. The default is cpu. The module is set to eval.

    ## Returns

        - The return is the NetRunner.
        - It is not None.

    ---

    # 构建一个网络，载入权重，返回 NetRunner。

        name 必须是 recipe.channels 的键。

        wbps 构建 WbpsNet。

        其他名字构建 RefineNet。

        通道数是 recipe.channels[name]。

        engine 为 None 时，NetRunner.engine 为 None。

        路径不是文件时抛出 FileNotFoundError。

        已经存在的文件会作为 _Engine 打开。

        WAPR 和 SAPR 的输出名是 trans 和 rot。

        WBPS 的名字是 within_group 和 between_group。

    ## 参数

        - ckpt: 传给 _load_weights 的权重路径。
        - device 传给 module.to。
        - 默认是 cpu。
        - 模块会被设为 eval。

    ## 返回

        - 返回 NetRunner。
        - 不是 None。

"""
    if name not in recipe.channels:
        raise KeyError(name)
    in_ch = int(recipe.channels[name])
    if name == "wbps":
        module = WbpsNet(in_ch)
        kind = "wbps"
        out_names = ("within_group", "between_group")
    else:
        module = RefineNet(in_ch)
        kind = "refine"
        out_names = ("trans", "rot")
    _load_weights(module, ckpt, kind)
    module.to(device)
    module.eval()
    runner_engine = None
    if engine is not None:
        engine_path = Path(engine)
        if not engine_path.is_file():
            raise FileNotFoundError(
                "%s FP16 engine is missing: %s. Build it with python -m wapr.export_engines on this GPU."
                % (name, engine_path)
            )
        runner_engine = _Engine(engine_path, out_names)
    return NetRunner(name, module, runner_engine)
