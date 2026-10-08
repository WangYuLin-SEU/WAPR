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
import inspect
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
# This file lives in wapr/, so the release root is its parent. Import does not depend on cwd.
# 本文件在 wapr/ 下，release 根目录是它的上一级。导入不依赖当前工作目录。
sys.path.insert(0, str(ROOT))

from wapr import recipe
from wapr.model_metadata import checkpoint_metadata, write_onnx_metadata, read_onnx_metadata, pack_engine
from wapr.nets import load_net
from wapr.pose_groups import wbps_group_sizes


def _onnx_unflatten(graph, tensor, axis, sizes):
    """Replace one shape dimension with sizes while retaining runtime dimensions.

    用 sizes 替换一个形状维度，其余维度来自运行时 Shape，保留动态 batch。
    """
    from torch.onnx import symbolic_helper
    axis = symbolic_helper._get_const(axis, "i", "axis")
    rank = symbolic_helper._get_tensor_rank(tensor)
    if axis < 0:
        if rank is None:
            raise RuntimeError("Unflatten with a negative axis needs known rank / 负轴 unflatten 需要已知 rank")
        axis += rank
    if axis < 0 or (rank is not None and axis >= rank):
        raise ValueError("Unflatten axis outside tensor rank / unflatten 轴超出张量 rank")
    shape = graph.op("Shape", tensor)
    prefix = graph.op(
        "Slice", shape,
        graph.op("Constant", value_t=torch.tensor([0], dtype=torch.long)),
        graph.op("Constant", value_t=torch.tensor([axis], dtype=torch.long)),
    )
    suffix = graph.op(
        "Slice", shape,
        graph.op("Constant", value_t=torch.tensor([axis + 1], dtype=torch.long)),
        graph.op("Constant", value_t=torch.tensor([9223372036854775807], dtype=torch.long)),
    )
    if symbolic_helper._is_packed_list(sizes):
        elements = symbolic_helper._unpack_list(sizes)
        sizes = graph.op("Concat", *[
            symbolic_helper._unsqueeze_helper(graph, element, [0]) for element in elements
        ], axis_i=0)
    output_shape = graph.op("Concat", prefix, sizes, suffix, axis_i=0)
    return symbolic_helper._reshape_helper(graph, tensor, output_shape)


def _onnx_scaled_dot_product_attention(graph, query, key, value, mask=None,
                                       dropout=0.0, causal=False, scale=None, enable_gqa=False):
    """Export inference attention as its standard matrix operations.

    用标准矩阵算子导出推理注意力；不修改 forward，不近似掩码语义。
    """
    from torch.onnx import symbolic_helper
    dropout = symbolic_helper._get_const(dropout, "f", "dropout")
    causal = symbolic_helper._get_const(causal, "b", "causal")
    if not isinstance(enable_gqa, bool):
        enable_gqa = symbolic_helper._get_const(enable_gqa, "b", "enable_gqa")
    if dropout != 0.0 or enable_gqa:
        raise RuntimeError("Attention compatibility export requires dropout=0 and no GQA / 注意力兼容导出要求 dropout=0 且不启用 GQA")
    rank = symbolic_helper._get_tensor_rank(key)
    if rank is None or rank < 2:
        raise RuntimeError("Attention key rank must be known / 注意力 key rank 须已知")
    scalar_type = query.type().scalarType()
    dtype = {"Float": torch.float32, "Double": torch.float64, "Half": torch.float16, "BFloat16": torch.bfloat16}.get(scalar_type)
    onnx_dtype = {"Float": 1, "Double": 11, "Half": 10, "BFloat16": 16}.get(scalar_type)
    if dtype is None:
        raise RuntimeError("Attention input floating dtype must be known / 注意力输入浮点类型须已知")
    if scale is None or symbolic_helper._is_none(scale):
        width = graph.op("Gather", graph.op("Shape", query), graph.op("Constant", value_t=torch.tensor(-1, dtype=torch.long)), axis_i=0)
        factor = graph.op("Reciprocal", graph.op("Sqrt", graph.op("Cast", width, to_i=onnx_dtype)))
    else:
        factor = graph.op("Constant", value_t=torch.tensor(symbolic_helper._get_const(scale, "f", "scale"), dtype=dtype))
    axes = list(range(rank))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    logits = graph.op("MatMul", graph.op("Mul", query, factor), graph.op("Transpose", key, perm_i=axes))
    zero = graph.op("Constant", value_t=torch.tensor(0.0, dtype=dtype))
    negative_infinity = graph.op("Constant", value_t=torch.tensor(float("-inf"), dtype=dtype))
    has_mask = mask is not None and not symbolic_helper._is_none(mask)
    if causal:
        if has_mask:
            raise RuntimeError("Causal attention plus explicit mask is unsupported / 不支持因果注意力同时指定 mask")
        # Causal position comparisons remain dynamic for unequal query/key lengths.
        # 因果位置比较保留动态长度，支持 query/key 长度不同。
        query_length = graph.op("Gather", graph.op("Shape", query), graph.op("Constant", value_t=torch.tensor(-2, dtype=torch.long)), axis_i=0)
        key_length = graph.op("Gather", graph.op("Shape", key), graph.op("Constant", value_t=torch.tensor(-2, dtype=torch.long)), axis_i=0)
        start = graph.op("Constant", value_t=torch.tensor(0, dtype=torch.long))
        step = graph.op("Constant", value_t=torch.tensor(1, dtype=torch.long))
        rows = symbolic_helper._unsqueeze_helper(graph, graph.op("Range", start, query_length, step), [1])
        columns = symbolic_helper._unsqueeze_helper(graph, graph.op("Range", start, key_length, step), [0])
        bias = graph.op("Where", graph.op("LessOrEqual", columns, rows), zero, negative_infinity)
        logits = graph.op("Add", logits, bias)
    elif has_mask:
        bias = graph.op("Where", mask, zero, negative_infinity) if symbolic_helper._is_bool(mask) else mask
        logits = graph.op("Add", logits, bias)
    probabilities = graph.op("Softmax", logits, axis_i=-1)
    return graph.op("MatMul", probabilities, value)


def onnx_export_options():
    """Keep the legacy exporter and derive an opset supported by this torch.

    保留同一追踪导出器，按当前 torch 支持的最高版本确定 ONNX opset。
    """
    try:
        from torch.onnx import _constants
        maximum_opset = int(_constants.ONNX_MAX_OPSET)
    except (ImportError, AttributeError):
        from torch.onnx import symbolic_helper
        maximum_opset = int(symbolic_helper._onnx_main_opset)
    options = {"opset_version": min(17, maximum_opset)}
    # Query the exporter registry before adding missing standard-operator mappings.
    # 先查导出注册表，仅补缺失算子的标准 ONNX 表达；不修改网络或 PyTorch forward。
    try:
        try:
            from torch.onnx._internal import registration
        except ImportError:
            # Recent PyTorch moved the same legacy registry into its TorchScript exporter.
            # 新版 PyTorch 将同一追踪导出注册表移至 TorchScript 导出器目录。
            from torch.onnx._internal.torchscript_exporter import registration
        unflatten_supported = registration.registry.is_registered_op("aten::unflatten", options["opset_version"])
        attention_supported = registration.registry.is_registered_op("aten::scaled_dot_product_attention", options["opset_version"])
    except ImportError:
        from torch.onnx import symbolic_registry
        symbolic_registry.register_version("", options["opset_version"])
        unflatten_supported = symbolic_registry.is_registered_op("unflatten", "", options["opset_version"])
        attention_supported = symbolic_registry.is_registered_op("scaled_dot_product_attention", "", options["opset_version"])
    if not unflatten_supported:
        torch.onnx.register_custom_op_symbolic("aten::unflatten", _onnx_unflatten, options["opset_version"])
        print("WAPR_ONNX_COMPAT", "aten::unflatten -> Shape/Slice/Concat/Reshape", flush=True)
    if not attention_supported:
        torch.onnx.register_custom_op_symbolic("aten::scaled_dot_product_attention", _onnx_scaled_dot_product_attention, options["opset_version"])
        print("WAPR_ONNX_COMPAT", "aten::scaled_dot_product_attention -> MatMul/Softmax/MatMul", flush=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    print("WAPR_ONNX_EXPORT", {"torch": torch.__version__, **options}, flush=True)
    return options


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
        - engine_path: cache/weights/<name>.engine by default. min_shape, opt_shape, and max_shape are (batch, channels, px, px) for inputs A and B. A TensorRT version other than 10.x raises RuntimeError. A failed build raises RuntimeError.

    ---

    # 建一份 TensorRT 10 的 FP16 引擎。

        输入仍是 FP32。

        返回 None。

    ## 参数

        - onnx_path: FP32 的 ONNX。
        - engine_path: 默认 cache/weights/<name>.engine。
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
    Path(engine_path).write_bytes(pack_engine(bytes(serialized), read_onnx_metadata(onnx_path)))


def main():
    """
    # Write FP32 ONNX using an opset supported by torch (up to 17), then a TensorRT 10 FP16 engine.

        The four names are wapr_w_mask, wapr_wo_mask, sapr, and wbps.

        A and B are (batch, channels, crop_px, crop_px).

        WBPS batch is one group's length L, never several objects stacked.

        Run it as python -m wapr.export_engines.

        The if __name__ block is the caller.

    ## Returns

        - Returns None.

    ---

    # 在每份权重旁写 FP32 ONNX，使用 torch 支持且不高于 17 的 opset，再写 TensorRT 10 FP16 引擎。

        四个名字是 wapr_w_mask、wapr_wo_mask、sapr 和 wbps。

        A 和 B 是 (batch, channels, crop_px, crop_px)。

        WBPS 的 batch 是一组的长度 L，不拼接多个物体。

        运行方式是 python -m wapr.export_engines。

        由 if __name__ 块调用。

    ## 返回

        - 返回 None。
    """
    device = "cuda:0"
    export_options = onnx_export_options()
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
            **export_options,
        )
        print("exported", name, "group", group, "max_batch", max_batch, flush=True)
        write_onnx_metadata(onnx_path, checkpoint_metadata(ckpt, name))
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
