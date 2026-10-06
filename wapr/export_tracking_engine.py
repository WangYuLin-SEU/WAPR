# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Export WBPS with six hypotheses per tracked instance and a dynamic object batch.

导出每个跟踪实例六个候选姿态、物体数可变的 WBPS 引擎。
Run explicitly on the inference GPU: python -m wapr.export_tracking_engine.
在推理显卡上显式运行：python -m wapr.export_tracking_engine。
The four existing engines and all checkpoint files remain unchanged.
不改动现有四份引擎或任何权重文件。
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from wapr import recipe
from wapr.export_engines import onnx_export_options
from wapr.model_metadata import checkpoint_metadata, write_onnx_metadata, read_onnx_metadata, pack_engine
from wapr.export_engines import _build_fp16_engine
from wapr.nets import load_net
from wapr.pose_groups import wbps_group_sizes


# Preserve the six candidates used by the one-instance tracking recipe.
# 保留单实例跟踪配置使用的六种候选。
tracking_hypotheses = (
    "previous", "depth_center", "wapr_center", "sapr_center",
    "sapr_wapr_center", "sapr_wapr_previous",
)


def rank_tracking_tokens(tokens, block):
    """Rank (objects, six hypotheses, channels) without cross-object attention.

    对 (物体数, 六个候选姿态, 通道数) 排序，注意力不跨越物体维。
    The explicit reshape keeps the object axis dynamic in ONNX/TensorRT.
    显式 reshape 使 ONNX/TensorRT 中的物体维保持动态。
    """
    group = len(tracking_hypotheses)
    attn = block.self_attn
    width = int(attn.embed_dim)
    heads = int(attn.num_heads)
    head_dim = width // heads
    query, key, value = F.linear(tokens, attn.in_proj_weight, attn.in_proj_bias).split(width, dim=-1)
    query = query.reshape(-1, group, heads, head_dim).permute(0, 2, 1, 3)
    key = key.reshape(-1, group, heads, head_dim).permute(0, 2, 1, 3)
    value = value.reshape(-1, group, heads, head_dim).permute(0, 2, 1, 3)
    attention = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) * head_dim ** -0.5, dim=-1)
    mixed = torch.matmul(attention, value).permute(0, 2, 1, 3).reshape(-1, group, width)
    attended = F.linear(mixed, attn.out_proj.weight, attn.out_proj.bias)
    hidden = block.norm1(tokens + block.dropout1(attended))
    fed = block.linear2(block.dropout(block.activation(block.linear1(hidden))))
    return block.norm2(hidden + block.dropout2(fed))


class TrackingScoreWrap(nn.Module):
    """Export the existing WBPS weights with a separate object batch axis.

    使用现有 WBPS 权重导出引擎，物体 batch 维与候选维相互独立。
    """

    def __init__(self, module):
        """Keep the loaded evaluation module. / 保存已载入的评估模块。"""
        super().__init__()
        self.module = module

    def forward(self, A, B):
        """Return two (objects, six) score tensors from flattened crop rows.

        将展平的裁剪行转换为两份 (物体数, 六个候选姿态) 分数。
        """
        group = len(tracking_hypotheses)
        feat = self.module.encoder(A, B)
        rows, channels, height, width = feat.shape
        tokens = feat.reshape(rows, channels, height * width).permute(0, 2, 1)
        tokens = self.module.token_trunk(self.module.pos(tokens, spatial_hw=(height, width)))
        pose_bank = self.module.pool_pose(tokens).reshape(-1, group, channels)
        rot_bank = self.module.pool_rot(tokens).reshape(-1, group, channels)
        between = self.module.head_pose(rank_tracking_tokens(pose_bank, self.module.rank_pose))
        within = self.module.head_rot(rank_tracking_tokens(rot_bank, self.module.rank_rot))
        return within.reshape(-1, group), between.reshape(-1, group)


def main():
    """Write only wbps_tracking.onnx and wbps_tracking.engine beside WBPS.

    只在 WBPS 文件旁写入 wbps_tracking.onnx 和 wbps_tracking.engine。
    """
    engine_dir = os.path.dirname(os.fspath(recipe.engine_file("wbps")))
    onnx_path = os.path.join(engine_dir, "wbps_tracking.onnx")
    engine_path = os.path.join(engine_dir, "wbps_tracking.engine")
    group = len(tracking_hypotheses)
    # Share the released engine's row ceiling, keeping whole instance groups.
    # 沿用发布引擎的行数上限，并保证每个实例的候选组完整。
    max_rows = max(wbps_group_sizes) // group * group
    opt_rows = max(group, int(recipe.n_view * recipe.n_inplane) // group * group)
    runner = load_net("wbps", recipe.weight_file("wbps"), device="cuda:0")
    wrapper = TrackingScoreWrap(runner.module).eval()
    shape = (opt_rows, int(recipe.channels["wbps"]), int(recipe.crop_px), int(recipe.crop_px))
    A = torch.zeros(shape, dtype=torch.float32, device="cuda:0")
    B = torch.zeros_like(A)
    torch.onnx.export(
        wrapper, (A, B), onnx_path, input_names=["A", "B"],
        output_names=["within_group", "between_group"], **onnx_export_options(),
        dynamic_axes={"A": {0: "pose_rows"}, "B": {0: "pose_rows"},
                      "within_group": {0: "objects"}, "between_group": {0: "objects"}},
    )
    write_onnx_metadata(onnx_path, checkpoint_metadata(recipe.weight_file("wbps"), "wbps"))
    _build_fp16_engine(onnx_path, engine_path, (group, *shape[1:]),
                       (opt_rows, *shape[1:]), (max_rows, *shape[1:]))
    print("WAPR_TRACKING_ENGINE", {"engine": engine_path, "group": group,
                                   "row_profile": [group, opt_rows, max_rows]}, flush=True)


if __name__ == "__main__":
    main()
