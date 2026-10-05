# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Export FP32 ONNX next to each weight, then a TensorRT 10 FP16 engine.
# 在每份权重旁边导出 FP32 ONNX，再写 TensorRT 10 的 FP16 引擎。
# Run from anywhere: python -m wapr.export_engines
# 任意目录可运行：python -m wapr.export_engines
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
# This file lives in wapr/, so the release root is its parent. Import does not depend on cwd.
# 本文件在 wapr/ 下，release 根目录是它的上一级。导入不依赖当前工作目录。
sys.path.insert(0, str(ROOT))

from wapr import recipe
from wapr.nets import load_net
from wapr.pose_groups import wbps_group_sizes


class _RefineWrap(nn.Module):
    """
    # Wrap WAPR and SAPR for ONNX.

        The module outputs are split into two tensors.

        main builds it for wapr_w_mask, wapr_wo_mask, and sapr.

        forward is what torch.onnx.export calls.

    ---

    # 把 WAPR 和 SAPR 包成 ONNX。

        模块的输出被拆成两个张量。

        main 为 wapr_w_mask、wapr_wo_mask 和 sapr 建立它。

        torch.onnx.export 调用的是 forward。
    """
    def __init__(self, module):
        """
        # Keep the loaded refine module.

        ## Args

            - module: a RefineNet on the export device. main passes runner.module.

        ## Returns

            - Returns None.

        ---

        # 保存已经载入的修正模块。

        ## 参数

            - module: 导出设备上的 RefineNet。main 传入 runner.module。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        self.module = module

    def forward(self, A, B):
        """
        # Run the refinement module.

            It returns trans and rot, each shaped (N, 3), before the estimator scales them.

            torch.onnx.export calls it through the wrapped module.

        ## Args

            - A and B: (N, C, crop_px, crop_px) float32. C is recipe.channels for that net.

        ---

        # 运行修正模块。

            返回 trans 和 rot，形状都是 (N, 3)。

            estimator 尚未对它们做缩放。

            torch.onnx.export 通过这个外壳调用它。

        ## 参数

            - A 和 B: (N, C, crop_px, crop_px) float32。C 是该网络在 recipe.channels 里的通道数。
        """
        out = self.module(A, B)
        return out["trans"], out["rot"]


def rank_tokens(tokens, block):
    """Run one WBPS rank block with a dynamic sequence axis.

    用动态序列维执行一次 WBPS 排序块；tokens 是 (1, L, 512)。
    nn.MultiheadAttention traces L as a constant. The -1 reshape keeps L
    dynamic in the exported graph while using the checkpoint's own weights.
    nn.MultiheadAttention 会把 L 记成常量；-1 reshape 保留动态 L，
    同时继续使用权重中的注意力、归一化和前馈权重。
    """
    attn = block.self_attn
    width = int(attn.embed_dim)
    heads = int(attn.num_heads)
    head_dim = width // heads
    qkv = F.linear(tokens, attn.in_proj_weight, attn.in_proj_bias)
    query, key, value = qkv.split(width, dim=-1)
    query = query.reshape(-1, heads, head_dim).unsqueeze(0).permute(0, 2, 1, 3)
    key = key.reshape(-1, heads, head_dim).unsqueeze(0).permute(0, 2, 1, 3)
    value = value.reshape(-1, heads, head_dim).unsqueeze(0).permute(0, 2, 1, 3)
    weights = torch.matmul(query, key.transpose(-2, -1)) * (head_dim ** -0.5)
    weights = torch.softmax(weights, dim=-1)
    mixed = torch.matmul(weights, value).permute(0, 2, 1, 3).reshape(1, -1, width)
    attended = F.linear(mixed, attn.out_proj.weight, attn.out_proj.bias)
    hidden = block.norm1(tokens + block.dropout1(attended))
    fed = block.linear2(block.dropout(block.activation(block.linear1(hidden))))
    return block.norm2(hidden + block.dropout2(fed))


class _ScoreWrap(nn.Module):
    """
    # Wrap WBPS for ONNX.

        Both outputs are (1, L) for one object with L hypotheses.

        main builds it for wbps.

        forward is what torch.onnx.export calls.

    ---

    # 把 WBPS 包成 ONNX。

        两个输出都是一个物体的 (1, L)，L 是候选数量。

        main 为 wbps 建立它。

        torch.onnx.export 调用的是 forward。
    """
    def __init__(self, module):
        """
        # Keep the WBPS module.

        ## Args

            - module: a WbpsNet.

        ## Returns

            - Returns None.

        ---

        # 保存 WBPS 模块。

        ## 参数

            - module: WbpsNet。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        self.module = module

    def forward(self, A, B):
        """
        # Run WBPS and return within_group and between_group, each (1, L).

            torch.onnx.export calls it through the wrapped module.

        ## Args

            - A and B: (L, C, crop_px, crop_px). L is one object's hypothesis count.

        ---

        # 跑 WBPS，返回 within_group 和 between_group，都是 (1, L)。

            torch.onnx.export 通过这个外壳调用它。

        ## 参数

            - A 和 B: (L, C, crop_px, crop_px)。L 是单个物体的候选数量。
        """
        score = self.module
        feat = score.encoder(A, B)
        b, c, h, w = feat.shape
        tok = feat.reshape(b, c, h * w).permute(0, 2, 1)
        tok = score.token_trunk(score.pos(tok, spatial_hw=(int(h), int(w))))
        pose_bank = score.pool_pose(tok).unsqueeze(0)
        rot_bank = score.pool_rot(tok).unsqueeze(0)
        between_group = score.head_pose(rank_tokens(pose_bank, score.rank_pose)).reshape(1, -1)
        within_group = score.head_rot(rank_tokens(rot_bank, score.rank_rot)).reshape(1, -1)
        return within_group, between_group


def _shape(batch, channels, px):
    """
    # One TensorRT profile shape, (batch, channels, px, px).

        batch, channels, and px are the three spatial sizes.

        All three are cast to int.

        main calls it for the minimum, optimum, and maximum profile.

    ## Returns

        - Returns that tuple of ints.

    ---

    # 一份 TensorRT profile 形状，(batch, channels, px, px)。

        batch、channels、px 是这三个尺寸。

        三个都转成 int。

        main 用它写最小、最优和最大三档 profile。

    ## 返回

        - 返回这个整数元组。
    """
    return (int(batch), int(channels), int(px), int(px))


def _build_fp16_engine(onnx_path, engine_path, min_shape, opt_shape, max_shape):
    """
    # Build one TensorRT 10 FP16 engine.

        Inputs stay FP32.

        Returns None.

    ## Args

        - onnx_path: the FP32 ONNX file.
        - engine_path: assets/weights/<name>.engine. min_shape, opt_shape, and max_shape are (batch, channels, px, px) for inputs A and B. A TensorRT version other than 10.x raises RuntimeError. A failed build raises RuntimeError.

    ---

    # 建一份 TensorRT 10 的 FP16 引擎。

        输入仍是 FP32。

        返回 None。

    ## 参数

        - onnx_path: FP32 的 ONNX。
        - engine_path: assets/weights/<name>.engine。
        - min_shape、opt_shape、max_shape: 输入 A 和 B 的 (batch, channels, px, px)。不是 TensorRT 10.x 就抛出 RuntimeError。构建失败也抛出 RuntimeError。

"""
    import tensorrt as trt

    if not str(trt.__version__).startswith("10."):
        raise RuntimeError("TensorRT 10.x is required")
    logger = trt.Logger(getattr(trt.Logger, "WARNING"))
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    profile = builder.create_optimization_profile()
    profile.set_shape("A", tuple(min_shape), tuple(opt_shape), tuple(max_shape))
    profile.set_shape("B", tuple(min_shape), tuple(opt_shape), tuple(max_shape))
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * 1024 ** 3)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("FP16 engine build failed")
    Path(engine_path).write_bytes(bytes(serialized))


def main():
    """
    # Write one FP32 ONNX beside each weight, opset 17, then a TensorRT 10 FP16 engine.

        The four names are wapr_w_mask, wapr_wo_mask, sapr, and wbps.

        A and B are (batch, channels, crop_px, crop_px).

        WBPS batch is one group's length L, never several objects stacked.

        Run it as python -m wapr.export_engines.

        The if __name__ block is the caller.

    ## Returns

        - Returns None.

    ---

    # 在每份权重旁边写一份 FP32 ONNX，opset 17，再写 TensorRT 10 的 FP16 引擎。

        四个名字是 wapr_w_mask、wapr_wo_mask、sapr 和 wbps。

        A 和 B 是 (batch, channels, crop_px, crop_px)。

        WBPS 的 batch 是一组的长度 L，不拼接多个物体。

        运行方式是 python -m wapr.export_engines。

        由 if __name__ 块调用。

    ## 返回

        - 返回 None。
    """
    device = "cuda:0"
    opset = 17
    group = int(recipe.n_view * recipe.n_inplane)
    if group not in wbps_group_sizes:
        raise ValueError("recipe pose group %d is unsupported" % group)
    min_batch = min(wbps_group_sizes)
    opt_batch = group
    max_batch = max(wbps_group_sizes)
    for name in ("wapr_w_mask", "wapr_wo_mask", "sapr", "wbps"):
        ckpt = recipe.weight_file(name)
        runner = load_net(name, ckpt, engine=None, device=device)
        channels = int(recipe.channels[name])
        px = int(recipe.crop_px)
        if name == "wbps":
            wrapped = _ScoreWrap(runner.module).to(device).eval()
            out_names = ["within_group", "between_group"]
            dyn_out = {key: {1: "batch"} for key in out_names}
            export_batch = group
        else:
            wrapped = _RefineWrap(runner.module).to(device).eval()
            out_names = ["trans", "rot"]
            dyn_out = {key: {0: "batch"} for key in out_names}
            export_batch = group
        # Trace one supported group; refinement rows remain independent in multi-instance batches.
        # 用一个合法组追踪导出；修正行在多实例批量计算中仍各自独立。
        A = torch.zeros(export_batch, channels, px, px, device=device)
        B = torch.zeros_like(A)
        onnx_path = ckpt.with_suffix(".onnx")
        torch.onnx.export(
            wrapped,
            (A, B),
            str(onnx_path),
            input_names=["A", "B"],
            output_names=out_names,
            dynamic_axes={"A": {0: "batch"}, "B": {0: "batch"}, **dyn_out},
            opset_version=opset,
            dynamo=False,
        )
        print("exported", name, "group", group, "max_batch", max_batch, flush=True)
        _build_fp16_engine(
            onnx_path,
            ckpt.with_suffix(".engine"),
            _shape(min_batch, channels, px),
            _shape(opt_batch, channels, px),
            _shape(max_batch, channels, px),
        )
        print("engine", name, flush=True)
    # Include the independent-object scorer for the default multi-instance path.
    # 同时构建独立物体批量计算评分引擎，供默认多实例路径使用。
    del wrapped, runner, A, B
    torch.cuda.empty_cache()
    from wapr.export_batch_engine import main as export_batch
    export_batch()

if __name__ == "__main__":
    main()
