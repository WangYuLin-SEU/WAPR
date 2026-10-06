# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""GPU silhouette rendering and greedy pose NMS.

GPU 轮廓渲染与贪心位姿 NMS；保持原图像素分辨率，不使用标注筛选预测。
"""
import numpy as np
import torch

from wapr.ogl import runtime_for


def greedy_mask_nms_across_categories(masks, scores, iou_thresh=0.5):
    """Return kept indices ranked by score for cross-category CUDA mask NMS.

    对外部掩码执行跨类别 CUDA NMS，按分数降序返回保留下标。
    """
    masks_t = torch.as_tensor(masks, device=torch.device("cuda", torch.cuda.current_device()))
    if masks_t.ndim == 2:
        masks_t = masks_t.unsqueeze(0)
    n = int(masks_t.shape[0])
    if n == 0:
        return []
    scores_t = torch.as_tensor(scores, device=masks_t.device, dtype=torch.float32).reshape(-1)
    # Zero class ids deliberately compare supplied masks across categories.
    # 类别编号统一为零，明确执行外部掩码的跨类别比较。
    kept = mask_iou_nms_indices_gpu(masks_t > 0.5, scores_t,
        torch.zeros(n, device=masks_t.device, dtype=torch.int64), iou_thresh)
    order = torch.argsort(scores_t[kept], descending=True, stable=True)
    return kept[order].cpu().tolist()


def rendered_mask_iou_nms(poses, mesh, K, height_px, width_px, iou_max):
    """Batch pose-contour NMS on the current CUDA device, preserving input order.

    在当前 CUDA 设备批量执行位姿轮廓 NMS，保留输入顺序。
    """
    return rendered_mask_iou_nms_gpu(poses, mesh, K, height_px, width_px, iou_max,
        torch.device("cuda", torch.cuda.current_device()))


@torch.inference_mode()
def render_pose_masks_gpu(mesh, poses_obj_to_cam, K, height_px, width_px, device):
    """Render independent object silhouettes on the original image pixel grid.

    mesh and poses use the same object frame, meters; K uses pixels. Returns
    CUDA bool masks (N, H, W). Integer crop origins preserve original pixel size.
    GPU rasterization uses pixel centers, unlike OpenCV integer
    triangle fill, so boundary pixels need not be identical.

    在原图像素网格上分别渲染每个位姿的物体剪影。mesh 与 poses 使用同一
    物体系及米制单位，K 为像素单位；返回 CUDA bool 掩码 (N, H, W)。整数
    裁剪起点保持原像素尺寸。GPU 按像素中心光栅化，与旧 OpenCV 整数三角形
    填充的边界像素可能不同。
    """
    poses = torch.as_tensor(poses_obj_to_cam, device=device, dtype=torch.float32).reshape(-1, 4, 4)
    count = len(poses)
    masks = torch.zeros((count, height_px, width_px), device=device, dtype=torch.bool)
    if count == 0:
        return masks
    vertices = torch.as_tensor(np.asarray(mesh.vertices), device=device, dtype=torch.float32)
    camera = torch.as_tensor(K, device=device, dtype=torch.float32).reshape(3, 3)
    with torch.autocast("cuda", enabled=False):
        points_cam = vertices[None] @ poses[:, :3, :3].transpose(1, 2) + poses[:, None, :3, 3]
        front = points_cam[..., 2] > 0
        z = torch.where(front, points_cam[..., 2], torch.ones_like(points_cam[..., 2]))
        pixels = (points_cam @ camera.T)[..., :2] / z[..., None]
        minimum = pixels.masked_fill(~front[..., None], float("inf")).amin(1)
        maximum = pixels.masked_fill(~front[..., None], float("-inf")).amax(1)
        visible = front.any(1)
        minimum = torch.where(visible[:, None], minimum, torch.zeros_like(minimum))
        maximum = torch.where(visible[:, None], maximum, torch.zeros_like(maximum))
        bounds = torch.tensor([width_px, height_px], device=device, dtype=torch.float32)
        origins = torch.minimum(minimum.floor().clamp_min(0), bounds).long()
        ends = torch.minimum(maximum.ceil().clamp_min(0), bounds).long()
        # The largest projected extent determines the render tile; no downsampling.
        # 最大投影尺寸唯一确定分块大小；每个分块像素仍对应一个原图像素。
        tile_px = max(1, int((ends - origins).clamp_min(0).max().item()))
        boxes = torch.cat((origins, origins + tile_px), dim=1).float()
        renderer = runtime_for(device)
        mesh_id = renderer.load_mesh_trimesh(mesh)
        rgb_tiles, depth_tiles = renderer.render_tiles(mesh_id, poses, boxes, K,
            height_px, width_px, tile_px)
        del rgb_tiles
        yy = origins[:, 1, None, None] + torch.arange(tile_px, device=device)[None, :, None]
        xx = origins[:, 0, None, None] + torch.arange(tile_px, device=device)[None, None, :]
        yy = yy.expand(count, tile_px, tile_px)
        xx = xx.expand(count, tile_px, tile_px)
        inside = (xx < width_px) & (yy < height_px)
        rows = torch.arange(count, device=device)[:, None, None]
        addresses = rows * height_px * width_px + yy * width_px + xx
        # Scatter only valid image pixels; no full-frame depth/RGB tensor is needed.
        # 只写入有效图像像素，避免为全部候选生成整帧深度或 RGB 浮点张量。
        masks.flatten().scatter_(0, addresses[inside], (depth_tiles > 0)[inside])
    return masks


@torch.inference_mode()
def mask_iou_nms_indices_gpu(masks, ranking_scores, obj_ids, iou_max, drop_empty=True):
    """Return kept input indices with stable, class-separated greedy mask NMS.

    masks: CUDA bool (N, H, W). Empty masks are removed unless drop_empty=False.
    ranking_scores are the caller's 2D or 6D scores. Strict IoU > iou_max
    suppresses; a suppressed row cannot suppress later rows. All comparisons
    and suppression state stay on GPU; returned indices preserve input order.

    masks 为 CUDA bool (N, H, W)。默认移除空掩码，drop_empty=False 时保留。
    ranking_scores 为调用方的 2D 或 6D 分数，按分数稳定降序排序，
    仅在同类别内以严格 IoU > iou_max 抑制。被抑制行不抑制后续行。比较与
    抑制状态均保留在 GPU，返回下标保持输入顺序。
    """
    if not masks.is_cuda or masks.dtype != torch.bool or masks.ndim != 3:
        raise ValueError("Expected CUDA bool masks (N, H, W)")
    if not 0 <= float(iou_max) <= 1:
        raise ValueError("iou_max must be between zero and one")
    count = len(masks)
    scores = torch.as_tensor(ranking_scores, device=masks.device, dtype=torch.float64).reshape(count)
    identities = torch.as_tensor(obj_ids, device=masks.device, dtype=torch.int64).reshape(count)
    if count == 0:
        return torch.empty(0, device=masks.device, dtype=torch.int64)
    pixels = masks.flatten(1)
    if pixels.shape[1] > 2 ** 24:
        raise ValueError("Exact mask intersections require at most 2**24 image pixels")
    with torch.autocast("cuda", enabled=False):
        binary = pixels.float()
        intersections = (binary @ binary.T).long()
        areas = pixels.sum(1, dtype=torch.int64)
        union = areas[:, None] + areas[None, :] - intersections
        # Binary FP32 products count exact integers here. Preserve the .5 boundary.
        # 此尺寸下二值 FP32 点积精确计数；0.5 阈值使用整数比较以保留等号边界。
        if float(iou_max) == 0.5:
            conflicts = 2 * intersections > union
        else:
            conflicts = intersections.double() > float(iou_max) * union.double()
        conflicts &= identities[:, None] == identities[None, :]
        order = torch.argsort(scores, descending=True, stable=True)
        conflicts = conflicts[order][:, order]
        dropped = areas[order] == 0 if drop_empty else torch.zeros(count, device=masks.device, dtype=torch.bool)
        keep_sorted = torch.zeros(count, device=masks.device, dtype=torch.bool)
        # Greedy selection has a row dependency. GPU updates avoid per-row .item()
        # synchronization and never copy full masks or pairwise overlaps to CPU.
        # 贪心选择存在逐行依赖；GPU 更新避免逐行 .item() 同步，不将整幅掩码
        # 或两两交叠矩阵复制到 CPU。
        for position in range(count):
            selected = ~dropped[position]
            keep_sorted[position] = selected
            dropped[position + 1:] |= conflicts[position, position + 1:] & selected
        keep_original = torch.zeros_like(keep_sorted)
        keep_original[order] = keep_sorted
        return torch.arange(count, device=masks.device)[keep_original]


def rendered_mask_iou_nms_gpu(poses, mesh, K, height_px, width_px, iou_max, device):
    """GPU rendered-mask NMS for pose dictionaries, retaining their input order.

    poses contain obj_id, score_6d and original-object-to-camera pose_4x4 in
    meters. Only the small final index list returns to CPU. No truth is read.

    poses 含 obj_id、score_6d 及米制原始物体系到相机的 pose_4x4。仅将最终
    小规模下标列表传回 CPU，不读取真值，输出保持原始输入顺序。
    """
    if not poses:
        return []
    masks = render_pose_masks_gpu(mesh, np.asarray([row["pose_4x4"] for row in poses]),
        K, height_px, width_px, device)
    kept = mask_iou_nms_indices_gpu(masks, [row["score_6d"] for row in poses],
        [row["obj_id"] for row in poses], iou_max)
    return [poses[index] for index in kept.cpu().tolist()]
