# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Export independent WBPS object groups in one TensorRT call.

将相互独立的 WBPS 物体候选组合并为一次 TensorRT 调用。
Run on the target GPU: python -m wapr.export_batch_engine.
在目标显卡运行：python -m wapr.export_batch_engine。
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from wapr import recipe
from wapr.nets import load_net
from wapr.pose_groups import wbps_group_sizes


def rank_batch_tokens(tokens, block):
    """Attention stays inside (objects, hypotheses, channels).

    注意力仅在 (物体数, 候选数量, 通道数) 的每个物体内部计算。
    """
    objects, group, width = tokens.shape
    attn = block.self_attn
    heads = int(attn.num_heads)
    head_dim = int(attn.embed_dim) // heads
    query, key, value = F.linear(tokens, attn.in_proj_weight, attn.in_proj_bias).split(int(attn.embed_dim), dim=-1)
    query = query.reshape(objects, group, heads, head_dim).permute(0, 2, 1, 3)
    key = key.reshape(objects, group, heads, head_dim).permute(0, 2, 1, 3)
    value = value.reshape(objects, group, heads, head_dim).permute(0, 2, 1, 3)
    attention = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) * head_dim ** -0.5, dim=-1)
    mixed = torch.matmul(attention, value).permute(0, 2, 1, 3).reshape(objects, group, width)
    attended = F.linear(mixed, attn.out_proj.weight, attn.out_proj.bias)
    hidden = block.norm1(tokens + block.dropout1(attended))
    fed = block.linear2(block.dropout(block.activation(block.linear1(hidden))))
    return block.norm2(hidden + block.dropout2(fed))


class BatchScoreWrap(nn.Module):
    """Use a small layout tensor to describe both dynamic group axes.

    用小型布局张量描述两个动态组维度，裁剪输入仍是展平的行。
    Layout values are unused; its shape is (objects, hypotheses, 1).
    布局值不参与计算；形状为 (物体数, 候选数量, 1)。
    """

    def __init__(self, module):
        """Keep the checkpoint module. / 保留已载入权重的网络。"""
        super().__init__()
        self.module = module

    def forward(self, A, B, layout):
        """Return independent (objects, hypotheses) scores.

        返回相互独立的 (物体数, 候选数量) 分数。
        """
        objects, group = layout.shape[:2]
        feat = self.module.encoder(A, B)
        rows, channels, height, width = feat.shape
        tokens = feat.reshape(rows, channels, height * width).permute(0, 2, 1)
        tokens = self.module.token_trunk(self.module.pos(tokens, spatial_hw=(height, width)))
        pose_bank = self.module.pool_pose(tokens).reshape(objects, group, channels)
        rot_bank = self.module.pool_rot(tokens).reshape(objects, group, channels)
        between = self.module.head_pose(rank_batch_tokens(pose_bank, self.module.rank_pose))
        within = self.module.head_rot(rank_batch_tokens(rot_bank, self.module.rank_rot))
        return within.reshape(objects, group), between.reshape(objects, group)


def main():
    """Build wbps_batch beside the existing engines, using unchanged weights.

    使用原权重，在已有引擎旁构建 wbps_batch；不修改原引擎。
    """
    import tensorrt as trt

    engine_dir = os.path.dirname(os.fspath(recipe.engine_file("wbps")))
    onnx_path = os.path.join(engine_dir, "wbps_batch.onnx")
    engine_path = os.path.join(engine_dir, "wbps_batch.engine")
    max_rows = max(wbps_group_sizes)
    group = int(recipe.n_view * recipe.n_inplane)
    objects = max(1, max_rows // group)
    runner = load_net("wbps", recipe.weight_file("wbps"), device="cuda:0")
    wrapper = BatchScoreWrap(runner.module).eval()
    tail = (int(recipe.channels["wbps"]), int(recipe.crop_px), int(recipe.crop_px))
    A = torch.zeros((objects * group, *tail), device="cuda:0")
    B = torch.zeros_like(A)
    layout = torch.empty((objects, group, 1), device="cuda:0")
    torch.onnx.export(
        wrapper, (A, B, layout), onnx_path, input_names=["A", "B", "layout"],
        output_names=["within_group", "between_group"], opset_version=17,
        dynamo=False,
        dynamic_axes={"A": {0: "rows"}, "B": {0: "rows"},
                      "layout": {0: "objects", 1: "hypotheses"},
                      "within_group": {0: "objects", 1: "hypotheses"},
                      "between_group": {0: "objects", 1: "hypotheses"}},
    )
    logger = trt.Logger(getattr(trt.Logger, "WARNING"))
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    profile = builder.create_optimization_profile()
    profile.set_shape("A", (2, *tail), tuple(A.shape), (max_rows, *tail))
    profile.set_shape("B", (2, *tail), tuple(B.shape), (max_rows, *tail))
    # The runtime checks objects * hypotheses == rows <= max_rows.
    # 运行时检查 物体数 * 候选数量 == 行数 <= max_rows。
    profile.set_shape("layout", (1, 2, 1), tuple(layout.shape), (max_rows // 2, max_rows, 1))
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * 1024 ** 3)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("WBPS batch engine build failed")
    Path(engine_path).write_bytes(bytes(serialized))
    print("WAPR_BATCH_ENGINE", {"engine": engine_path, "max_rows": max_rows,
                                "group_sizes": sorted(wbps_group_sizes)}, flush=True)


if __name__ == "__main__":
    main()
