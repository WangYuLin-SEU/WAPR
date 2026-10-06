# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Region propagation for saved RGB-D robot streams, using wapr.region_tracking.

复用 wapr.region_tracking，为保存的机器人 RGB-D 序列传播区域。
Pose updates use wapr.tracking; this module supplies masks and full-frame renders.
位姿更新走 wapr.tracking；本模块提供区域和整帧渲染，不依赖 tracking_loc。
"""

import cv2
import numpy as np
from wapr import region_tracking as regions
from wapr.region_tracking import one_instance_mask_to_patches, patches_to_mask, translation_from_one_instance
from wapr.export_tracking_engine import tracking_hypotheses

HYPOTHESES = tuple((label,) for label in tracking_hypotheses)
# Preserve the saved-stream gates and encoder settings from the original recipe.
# 保留保存序列原有的门槛与编码器设置。
min_region_px = 150
min_matches = 8
cosine_min = 0.50
dino_patch = 14
# Keep the saved-stream depth policy explicit, in meters and pixels.
# 保留保存序列的显式深度策略，单位米和像素。
depth_agree_m = 0.020
region_dilate_px = 5


def load_dino():
    """Load the tracker checkpoint on CUDA. / 在 CUDA 上载入跟踪权重。"""
    return regions.load_tracker_dino("cuda:0")


def dino_tokens(model, rgb):
    """Encode once per camera frame. / 每个相机帧编码一次。"""
    return regions.dino_tokens(model, rgb, next(model.parameters()).device)


def centered_poses(poses, center):
    """Convert original object poses for centered runtime geometry.

    将原网格位姿转换为居中运行时几何的位姿。
    """
    shift = np.eye(4, dtype=np.float32)
    shift[:3, 3] = np.asarray(center, dtype=np.float32).reshape(3)
    return np.asarray(poses, dtype=np.float32) @ shift


def render_pose_rgbd(runtime, mesh_id, poses_centered, K, height, width):
    """Full-frame render, RGB 0..255 and depth in meters.

    整帧渲染，RGB 为 0..255，深度单位米。
    """
    poses = np.asarray(poses_centered, dtype=np.float32).reshape(-1, 4, 4)
    boxes = np.tile([0, 0, width, height], (len(poses), 1)).astype(np.float32)
    # The mixed raster path supports a rectangular full-frame resolution, even for one mesh.
    # 批量计算光栅路径支持矩形全帧分辨率，单网格也使用它。
    rgb, depth = runtime._render_mixed_tiles([mesh_id] * len(poses), poses, boxes, K,
                                           height, width, (height, width))
    rgb = rgb.detach().float().cpu().numpy()
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8), depth.detach().float().cpu().numpy()


def agreement_mask(rendered_depth, sensor_depth):
    """Depth agreement followed by the saved-stream dilation.

    深度一致区域，再执行保存序列的膨胀。
    """
    both = (rendered_depth > 1.0e-3) & (sensor_depth > 1.0e-3)
    mask = both & (np.abs(rendered_depth - sensor_depth) < depth_agree_m)
    if region_dilate_px > 0:
        kernel = np.ones((region_dilate_px, region_dilate_px), dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask


def visible_region(rendered_depth, sensor_depth):
    """Use depth agreement when enough pixels remain, otherwise the silhouette.

    深度一致像素足够时使用该区域，否则使用渲染轮廓。
    """
    mask = agreement_mask(rendered_depth, sensor_depth)
    if int(mask.sum()) >= min_region_px:
        return mask, "depth"
    return rendered_depth > 1.0e-3, "silhouette"


def propagate_one_instance_region(prev_tokens, prev_patch, template_tokens, template_patch,
                                  curr_tokens, grid, diameter_px, match_gate):
    """Match both histories with an explicit per-stream gate; no global mutation.

    用显式逐序列门槛匹配上一帧和模板；不修改全局参数。
    """
    return regions.propagate_one_instance_region(
        prev_tokens, prev_patch, template_tokens, template_patch, curr_tokens, grid, diameter_px,
        cosine_min=cosine_min, min_matches=match_gate, dino_patch=dino_patch,
    )
