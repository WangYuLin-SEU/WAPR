# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Reusable RGB feature encoding, region association and depth translation.

可复用的 RGB 特征编码、区域关联与深度平移计算。
The caller owns track state, frame loops, thresholds and recovery decisions.
调用方持有轨迹状态，并明确编写帧循环、阈值与恢复决策。
"""

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from wapr.det2d import DINO_CHOICES, default_weights_dir, prepare_dino_weight, _load_official_dino


def load_tracker_dino(device, dino_name="vits14"):
    """Load the requested frozen DINOv2 encoder. / 载入指定型号的冻结 DINOv2 编码器。"""
    spec = DINO_CHOICES[dino_name]
    weight_path = os.path.join(default_weights_dir, spec["file"])
    from wapr.bootstrap import ensure_optional
    ensure_optional("dinov2")
    if not os.path.isfile(weight_path):
        prepare_dino_weight(default_weights_dir, dino=dino_name)
    print("TRACK_DINO", {"model": dino_name, "weight": weight_path}, flush=True)
    model = _load_official_dino(spec["hub"])
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def dino_tokens(model, rgb, device, dino_short_side=448, dino_patch=14):
    """Encode one frame shared by its tracks; return (grid_h, grid_w, C) and grid.

    RGB uint8 becomes normalized FP16 encoder input and FP32 matching tokens.
    一帧编码供所有轨迹共享，返回 (grid_h, grid_w, C) 特征与网格信息。
    RGB uint8 归一化为 FP16 编码输入，匹配特征为 FP32。
    """
    height, width = rgb.shape[:2]
    scale = float(dino_short_side) / float(min(height, width))
    resized_h = int(round(height * scale))
    resized_w = int(round(width * scale))
    resized_h -= resized_h % dino_patch
    resized_w -= resized_w % dino_patch
    resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
    image = torch.as_tensor(resized, device=device).permute(2, 0, 1).float().div_(255.0)
    mean = image.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = image.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    batch = ((image - mean) / std).unsqueeze(0)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        feat = model.get_intermediate_layers(batch, n=1, reshape=True)[0]
    tokens = feat[0].float().permute(1, 2, 0).contiguous()
    grid = {"gh": int(tokens.shape[0]), "gw": int(tokens.shape[1]),
            "nh": int(resized_h), "nw": int(resized_w), "height": int(height), "width": int(width)}
    return tokens, grid


def one_instance_mask_to_patches(mask, grid, patch_cover=0.20, dino_patch=14):
    """Mark patches whose instance coverage reaches patch_cover.

    标记实例掩码覆盖比例达到 patch_cover 的图块。
    """
    small = cv2.resize(mask.astype(np.uint8), (grid["nw"], grid["nh"]), interpolation=cv2.INTER_NEAREST)
    view = small.reshape(grid["gh"], dino_patch, grid["gw"], dino_patch)
    cover = view.mean(axis=(1, 3))
    return cover >= patch_cover


def patches_to_mask(rows, cols, grid, dino_patch=14):
    """Map patch indices to a bool mask at original resolution.

    将图块下标映射回原图分辨率的 bool 掩码。
    """
    mask = np.zeros((grid["height"], grid["width"]), dtype=bool)
    for row, col in zip(rows, cols):
        y0 = int(round(float(row) * dino_patch * grid["height"] / grid["nh"]))
        y1 = int(round(float(row + 1) * dino_patch * grid["height"] / grid["nh"]))
        x0 = int(round(float(col) * dino_patch * grid["width"] / grid["nw"]))
        x1 = int(round(float(col + 1) * dino_patch * grid["width"] / grid["nw"]))
        y0 = max(0, min(grid["height"], y0))
        y1 = max(0, min(grid["height"], y1))
        x0 = max(0, min(grid["width"], x0))
        x1 = max(0, min(grid["width"], x1))
        mask[y0:y1, x0:x1] = True
    return mask


def match_one_instance_patches(src_tokens, src_patch, dst_tokens, cosine_min=0.50, min_matches=8):
    """Match stored instance patches by cosine, or return None below the gate.

    按余弦匹配保存的实例图块；匹配数不足时返回 None。
    """
    src = src_tokens[src_patch]
    if int(src.shape[0]) == 0:
        return None
    src_n = F.normalize(src.float(), dim=-1)
    dst_n = F.normalize(dst_tokens.reshape(-1, src.shape[-1]).float(), dim=-1)
    score, index = (src_n @ dst_n.T).max(dim=1)
    keep = score >= cosine_min
    if int(keep.sum()) < min_matches:
        return None
    return index[keep].detach().cpu().numpy()


def cluster_matches(indices, grid, diameter_px, min_matches=8, dino_patch=14):
    """Return the densest cluster's mask, pixel center and match count.

    返回最密匹配簇的掩码、像素中心与匹配数。
    """
    grid_w = int(grid["gw"])
    cols = np.asarray(indices, dtype=np.int64) % grid_w
    rows = np.asarray(indices, dtype=np.int64) // grid_w
    x = (cols.astype(np.float64) + 0.5) * dino_patch * grid["width"] / grid["nw"]
    y = (rows.astype(np.float64) + 0.5) * dino_patch * grid["height"] / grid["nh"]
    # Preserve the original clustering heuristic; this is not a geometric identity.
    # 保留原有聚类启发式；该尺度不是几何恒等式推导的值。
    cell = max(float(diameter_px) * 0.5, 24.0)
    bin_x = np.floor(x / cell).astype(np.int64)
    bin_y = np.floor(y / cell).astype(np.int64)
    bin_x -= int(bin_x.min())
    bin_y -= int(bin_y.min())
    span = int(bin_x.max()) + 1
    keys = bin_y * span + bin_x
    counts = np.bincount(keys)
    mode = int(counts.argmax())
    mode_x = mode % span
    mode_y = mode // span
    near = (np.abs(bin_x - mode_x) <= 1) & (np.abs(bin_y - mode_y) <= 1)
    if int(near.sum()) < min_matches:
        return None
    keep_rows = rows[near]
    keep_cols = cols[near]
    order = np.unique(keep_rows.astype(np.int64) * grid_w + keep_cols.astype(np.int64))
    mask = patches_to_mask(order // grid_w, order % grid_w, grid, dino_patch=dino_patch)
    center = (float(np.median(x[near])), float(np.median(y[near])))
    return mask, center, int(near.sum())


def propagate_one_instance_region(prev_tokens, prev_patch, template_tokens, template_patch,
                                  curr_tokens, grid, diameter_px, cosine_min=0.50, min_matches=8,
                                  dino_patch=14):
    """Associate previous/template regions; return mask, pixel center and count.

    同时关联上一帧与模板区域，返回掩码、像素中心和匹配数。
    """
    found = []
    for src_tokens, src_patch in ((prev_tokens, prev_patch), (template_tokens, template_patch)):
        matched = match_one_instance_patches(src_tokens, src_patch, curr_tokens,
                                             cosine_min=cosine_min, min_matches=min_matches)
        if matched is not None:
            found.append(matched)
    if not found:
        return None
    indices = np.concatenate(found, axis=0)
    return cluster_matches(indices, grid, diameter_px, min_matches=min_matches, dino_patch=dino_patch)


def translation_from_one_instance(mask, uv, depth_m, K, rotation, vertices,
                                  min_region_px=150, front_percentile=30.0):
    """Estimate the original mesh origin in camera meters from visible front depth.

    由可见前表面深度估计原始网格原点在相机中的米制平移。
    """
    u, v = float(uv[0]), float(uv[1])
    ys, xs = np.nonzero(mask)
    if len(xs) < min_region_px:
        return None
    samples = depth_m[ys, xs]
    samples = samples[samples > 1.0e-3]
    if len(samples) < min_region_px // 2:
        return None
    z_front = float(np.median(samples))
    if not (0.05 < z_front < 5.0):
        return None
    cam = vertices @ np.asarray(rotation, dtype=np.float64).T
    near = cam[:, 2] <= np.percentile(cam[:, 2], front_percentile)
    offset = cam[near].mean(axis=0)
    x = (u - float(K[0, 2])) * z_front / float(K[0, 0])
    y = (v - float(K[1, 2])) * z_front / float(K[1, 1])
    front = np.array([x, y, z_front], dtype=np.float64)
    return (front - offset).astype(np.float32)
