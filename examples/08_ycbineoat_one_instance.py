# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 08. Previous: examples/07_write_bop_pose_csv.py. Next: examples/09_taco_many_instances.py
# 示例 08。上一例：examples/07_write_bop_pose_csv.py。下一例：examples/09_taco_many_instances.py
# Estimate frame 0, track one object, and compare DINOv2 recovery candidates.
# 第 0 帧估计位姿，跟踪一个物体，并比较 DINOv2 补救候选。
# Run: python examples/08_ycbineoat_one_instance.py
# 运行：python examples/08_ycbineoat_one_instance.py
# Page: docs/pose.html#track
# 页面：docs/pose.html#track
# The default fetches its original sequence on first use, not during installation.
# 默认在首次使用时获取原始序列，不在安装时下载。
# This file does not track a second instance.
# 本文件不跟踪第二个实例。
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

script_dir = os.path.dirname(os.path.abspath(__file__))
# This example sits in examples/, so the release root is its parent.
# 本示例在 examples/ 下，release 根目录是它的上一级。
release_dir = os.path.dirname(script_dir)
sys.path.insert(0, release_dir)

from wapr import WAPREstimator
from wapr import recipe
from wapr.bootstrap import ensure_optional
from wapr.det2d import WAPRDet2D, onboard_meshes
from wapr.download_assets import check_and_fetch_pack
from wapr.estimator import center_from_mesh, prepare_mesh
from wapr.resources import samples_dir
from wapr.source_setup import prepare_ycbineoat
from wapr.region_tracking import (
    load_tracker_dino, dino_tokens, one_instance_mask_to_patches, patches_to_mask,
    match_one_instance_patches, cluster_matches, propagate_one_instance_region,
    translation_from_one_instance,
)
from wapr.tracking import track_many_categories_many_instances, refine_batch
from wapr.view import visualize_6d_pose


# DINOv2 ViT-S/14 propagates the target region across frames. Resize the short side to 448 px, then crop to a multiple of the 14 px patch size.
# DINOv2 ViT-S/14 在帧间传播目标区域。先将短边缩放至 448 像素，再裁剪到 14 像素图块的整数倍。
dino_name = "vits14"
dino_short_side = 448
dino_patch = 14
# A patch belongs to the instance when the mask covers at least this fraction of it.
# 图块里 mask 至少盖住这个比例，才算这个实例的图块。
patch_cover = 0.20
# Cosine below this is not a match. Fewer than this many matches drops the frame's carry.
# 余弦低于这个值不算匹配。对上的图块少于此数，这一帧的区域传递作废。
cosine_min = 0.50
min_matches = 8
min_region_px = 150
# Nearest this percent of rotated vertices places the front surface relative to the origin.
# 旋转后最近的这个百分比顶点，用来放前表面相对原点的位置。
front_percentile = 30.0
# WBPS between_group is an error score; above this threshold, search DINOv2 patches.
# WBPS 的 between_group 是误差分数；超过该阈值时搜索 DINOv2 图块。
recover_between_group = 0.0
# Search limits and geometric gates remain explicit, with depth in meters.
# 搜索数量和几何阈值显式保留，深度单位为米。
local_candidates = 6
global_candidates = 6
minimum_similarity = 0.35
depth_gate_m = 0.18
local_radius_diameters = 0.55
candidate_separation_px = 25.0
region_area_drop_ratio = 0.70
# Each row is the label. The pose stack below follows this order.
# 每一行是名字。下面的位姿堆叠按这个顺序。
HYPOTHESES = (
    "prev",
    "center",
    "center+WAPR",
    "center+SAPR",
    "center+WAPR+SAPR",
    "prev+WAPR+SAPR",
)


def patch_candidates(template_features, current_tokens, grid, depth_m, pose, vertices, K, diameter_m):
    """Return spatially distinct, depth-consistent local and global DINOv2 patches.

    返回位置不同、深度一致的局部和全图 DINOv2 图块。
    """
    destination = F.normalize(current_tokens.reshape(-1, current_tokens.shape[-1]).float(), dim=-1)
    similarity = (template_features @ destination.T).amax(dim=0).cpu().tolist()
    rotated = vertices @ np.asarray(pose[:3, :3], dtype=np.float64).T
    expected_front_z = float(pose[2, 3]) + float(np.percentile(rotated[:, 2], front_percentile))
    prior_uv = (K @ pose[:3, 3])[:2] / float(pose[2, 3])
    radius_px = local_radius_diameters * diameter_m * float(K[0, 0]) / max(expected_front_z, 0.05)
    patches = []
    for index, score in enumerate(similarity):
        if score < minimum_similarity:
            continue
        row, col = divmod(index, grid["gw"])
        x = int(round((col + 0.5) * dino_patch * grid["width"] / grid["nw"]))
        y = int(round((row + 0.5) * dino_patch * grid["height"] / grid["nh"]))
        x = max(0, min(depth_m.shape[1] - 1, x))
        y = max(0, min(depth_m.shape[0] - 1, y))
        z = float(depth_m[y, x])
        if not np.isfinite(z) or z <= 0.0 or abs(z - expected_front_z) > depth_gate_m:
            continue
        distance_px = float(np.hypot(x - prior_uv[0], y - prior_uv[1]))
        patches.append((score, distance_px, x, y, z))
    local = sorted((item for item in patches if item[1] <= radius_px), key=lambda item: item[0], reverse=True)
    global_search = sorted(patches, key=lambda item: item[0], reverse=True)
    selected = []
    for ordered, limit in ((local, local_candidates), (global_search, global_candidates)):
        added = 0
        for item in ordered:
            if any(np.hypot(item[2] - other[2], item[3] - other[3]) < candidate_separation_px for other in selected):
                continue
            selected.append(item)
            added += 1
            if added == limit:
                break
    return selected


def front_point_seed(previous_pose, patch, vertices, K):
    """Keep the previous rotation and place the origin behind an observed surface point.

    保留上一旋转，根据观测到的前表面深度设置物体原点。
    """
    _score, _distance_px, x, y, z = patch
    rotated = vertices @ np.asarray(previous_pose[:3, :3], dtype=np.float64).T
    near = rotated[:, 2] <= np.percentile(rotated[:, 2], front_percentile)
    offset = rotated[near].mean(axis=0)
    front = np.array([(x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z])
    candidate = previous_pose.copy()
    candidate[:3, 3] = front - offset
    return candidate


def refine_search_group(estimator, base_pose, seeds, rgb, depth_m, K, mesh, center, diameter_m):
    """Compare the completed normal update with batch-refined search seeds.

    将完成的常规更新与批量修正后的搜索起点一起比较。
    """
    refined = refine_batch(estimator, "wapr_wo_mask", seeds, rgb, depth_m, K, mesh, center,
                           diameter_m, 1, recipe.wapr_max_rot_rad)
    refined = refine_batch(estimator, "sapr", refined, rgb, depth_m, K, mesh, center,
                           diameter_m, 2, recipe.sapr_max_rot_rad)
    group = np.concatenate([np.asarray(base_pose, dtype=np.float32)[None], refined], axis=0)
    pose_t = torch.as_tensor(group, device=estimator.device, dtype=torch.float32)
    blank_mask = np.zeros(depth_m.shape[:2], dtype=np.uint8)
    within, between = estimator._wbps(pose_t, rgb, depth_m, blank_mask, K, mesh,
                                      center, diameter_m, len(group))
    scores = within.reshape(-1).detach().float().cpu().numpy()
    valid = np.isfinite(group).all(axis=(1, 2)) & (group[:, 2, 3] > 0) & np.isfinite(scores)
    if not np.any(valid):
        raise RuntimeError("No valid DINOv2 search pose / 没有有效的 DINOv2 搜索位姿")
    chosen = int(np.argmax(np.where(valid, scores, -np.inf)))
    quality = float(between.max().detach().cpu())
    return group[chosen], chosen, quality


def frame_stems(seq_dir):
    """
    # List one sequence in time order.

    ## Args

        - seq_dir: the unpacked sequence. rgb/ holds the color frames. A stem is kept when depth/ has the same stem.

    ## Returns

        - Returns the stems, oldest first.

    ---

    # 按时间列出一段序列。

    ## 参数

        - seq_dir: 解压后的序列。彩色帧在 rgb/。depth/ 里有同一个文件名主干时才留下。

    ## 返回

        - 返回文件名主干，旧的在前。

"""
    rgb_dir = os.path.join(seq_dir, "rgb")
    depth_dir = os.path.join(seq_dir, "depth")
    names = sorted(
        name for name in os.listdir(rgb_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    stems = []
    for name in names:
        stem = os.path.splitext(name)[0]
        depth_path = os.path.join(depth_dir, stem + ".png")
        if os.path.isfile(depth_path):
            stems.append(stem)
    return stems


def read_rgbd(seq_dir, stem):
    """
    # Read one RGB-D frame from a YCBInEOAT sequence.

        Depth is uint16 millimeters, then divided by 1000.

    ## Returns

        - Returns rgb and depth_m. rgb is RGB. depth_m is float32 meters.

    ---

    # 从一段 YCBInEOAT 读一帧 RGB-D。

        深度是 uint16 毫米，再除以 1000。

    ## 返回

        - 返回 rgb 和 depth_m。rgb 是 RGB。depth_m 是 float32 米。

"""
    rgb_path = rgb_frame_path(seq_dir, stem)
    depth_path = os.path.join(seq_dir, "depth", stem + ".png")
    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(rgb_path)
    if depth is None:
        raise FileNotFoundError(depth_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth_m = depth.astype(np.float32) / 1000.0
    return rgb, depth_m


def rgb_frame_path(seq_dir, stem):
    """Resolve the enumerated RGB frame, rejecting ambiguous filenames.

    定位列出的 RGB 帧；同一主干有多个文件时明确拒绝。
    """
    paths = [os.path.join(seq_dir, "rgb", name) for name in os.listdir(os.path.join(seq_dir, "rgb"))
             if os.path.splitext(name)[0] == stem and name.lower().endswith((".png", ".jpg", ".jpeg"))]
    if len(paths) != 1:
        raise ValueError("Expected one RGB file for %s, found %d" % (stem, len(paths)))
    return paths[0]


if __name__ == "__main__":
    # Prepare optional detection before importing its mask codec.
    # 先准备可选检测功能，成功后再导入它的掩码编解码器。
    ensure_optional("det2d")
    from pycocotools import mask as mask_utils

    print("example 08  previous examples/07_write_bop_pose_csv.py  next examples/09_taco_many_instances.py", flush=True)
    # seq_dir is one unpacked YCBInEOAT sequence. mesh_path is that one object, meters.
    # seq_dir 是解压后的一段 YCBInEOAT。mesh_path 是这一个物体，米。
    # stride 1 keeps every frame. A larger stride skips frames.
    # stride 为 1 时逐帧。更大的 stride 会跳帧。
    # Detect frame 0 from RGB and the known mesh; later frames propagate that predicted region.
    # 第 0 帧由 RGB 和已知网格检测掩码；后续帧传播这块预测区域。
    # Author-approved default; downloaded BOP CAD is converted from mm to meters.
    # 作者确认的默认序列；下载的 BOP CAD 从毫米换为米。
    default_seq_dir = os.path.join(samples_dir(), "YCBInEOAT", "mustard_easy_00_02")
    seq_dir = default_seq_dir
    default_models_dir = os.path.join(samples_dir(), "bop", "ycbv", "models")
    mesh_path = os.path.join(default_models_dir, "obj_000005.ply")
    # The default diameter is read from BOP below; set meters for a custom mesh.
    # 默认直径从下方 BOP 元数据读取；自定义网格时在此填米单位的直径。
    diameter_m = 0.0
    stride = 1
    device = "cuda:0"
    # Fetch the default resources only when the recipe still selects them.
    # 仅在配方仍选择默认资源时下载，用户自选数据不会触发默认序列下载。
    if seq_dir == default_seq_dir:
        seq_dir = prepare_ycbineoat("mustard_easy_00_02")
    if mesh_path == os.path.join(default_models_dir, "obj_000005.ply"):
        check_and_fetch_pack("ycbv")
        if diameter_m == 0.0:
            with open(os.path.join(default_models_dir, "models_info.json")) as stream:
                default_model_info = json.load(stream)["5"]
            diameter_m = float(default_model_info["diameter"]) / 1000.0
    if not seq_dir or not mesh_path or float(diameter_m) <= 0.0:
        raise SystemExit(
            "Set seq_dir, mesh_path, and diameter_m in examples/08_ycbineoat_one_instance.py.\n"
            "diameter_m is meters and must be greater than 0.\n"
            "在 examples/08_ycbineoat_one_instance.py 里填 seq_dir、mesh_path 和 diameter_m。\n"
            "diameter_m 是米，必须大于 0。"
        )
    if int(stride) < 1:
        raise ValueError("stride")
    check_and_fetch_pack("wapr_sapr_wbps")
    if not os.path.isabs(seq_dir):
        seq_dir = os.path.join(release_dir, seq_dir)
    if not os.path.isabs(mesh_path):
        mesh_path = os.path.join(release_dir, mesh_path)
    stems = frame_stems(seq_dir)[::int(stride)]
    if len(stems) == 0:
        raise FileNotFoundError(seq_dir)
    K = np.loadtxt(os.path.join(seq_dir, "cam_K.txt"), dtype=np.float32).reshape(3, 3)
    raw = trimesh.load(mesh_path, force="mesh", process=False)
    if mesh_path == os.path.join(default_models_dir, "obj_000005.ply"):
        raw.apply_scale(0.001)
    vertices = np.asarray(raw.vertices, dtype=np.float64).copy()
    mesh = prepare_mesh(raw)
    center = center_from_mesh(mesh)
    mesh.metadata["name"] = os.path.splitext(os.path.basename(mesh_path))[0]
    # Detect the initial mask; subsequent recovery uses DINOv2 patches.
    # 检测初始掩码；后续补救使用 DINOv2 图块。
    stem0 = stems[0]
    rgb, depth_m = read_rgbd(seq_dir, stem0)
    # A saved visual click specifies the instance; rank only detector masks covering it.
    # 保存的目视点击点指定实例；只在覆盖该点的检测掩码中按分数排序。
    clicks_path = os.path.join(script_dir, "08_ycbineoat_init_clicks.json")
    with open(clicks_path, "r", encoding="utf-8") as stream:
        saved_clicks = json.load(stream)
    sequence_name = os.path.basename(os.path.normpath(seq_dir))
    selected = saved_clicks.get(sequence_name)
    if selected is None or selected["frame"] != stem0:
        raise RuntimeError("Save a frame-0 click for %s / 请为第 0 帧保存点击点" % sequence_name)
    click_u, click_v = (int(value) for value in selected["click_uv"])
    if not (0 <= click_u < rgb.shape[1] and 0 <= click_v < rgb.shape[0]):
        raise ValueError("Frame-0 click outside RGB / 第 0 帧点击点超出图像")
    template = onboard_meshes({1: mesh}, device=device)
    detector = WAPRDet2D(template, device=device, backend="trt")
    for _ in range(3):
        detector.detect_many_categories_many_instances(rgb)
    detections, timing = detector.detect_many_categories_many_instances(rgb, profile=True)
    if not detections:
        raise RuntimeError("No frame-0 detection for %s / 第 0 帧没有检测候选" % stem0)
    clicked_detections = []
    for detection in detections:
        visible = np.asarray(mask_utils.decode(detection["mask"])) > 0
        if visible.shape != rgb.shape[:2]:
            raise RuntimeError("Detection mask shape / 检测掩码尺寸错误")
        if visible[click_v, click_u] and int(np.count_nonzero(visible)) >= min_region_px:
            clicked_detections.append((detection, visible))
    if not clicked_detections:
        raise RuntimeError("No detector mask covers the saved click / 没有检测掩码覆盖保存的点击点")
    top, visible = max(clicked_detections, key=lambda item: float(item[0]["score_2d"]))
    mask0 = visible.astype(np.uint8) * 255
    # Save the predicted input and its detector score separately from dataset annotations.
    # 把预测输入及检测分数存到项目输出目录，与数据集标注分开。
    init_out = os.path.join(release_dir, "outputs", "tracking_init", os.path.basename(os.path.normpath(seq_dir)))
    os.makedirs(init_out, exist_ok=True)
    mask_out = os.path.join(init_out, "frame0_mask.png")
    if not cv2.imwrite(mask_out, mask0):
        raise OSError("Could not save predicted mask / 无法保存预测掩码: %s" % mask_out)
    with open(os.path.join(init_out, "frame0_detection.json"), "w", encoding="utf-8") as stream:
        json.dump({"frame": stem0, "source": "WAPRDet2D", "click_uv": [click_u, click_v],
                   "score_2d": float(top["score_2d"]),
                   "bbox_xywh": top["bbox"], "mask_px": int(np.count_nonzero(mask0)),
                   "detector_ms": float(timing["total_ms"]),
                   "annotated_mask_used": False, "annotated_pose_used": False,
                   "rgb_sha256": hashlib.sha256(open(rgb_frame_path(seq_dir, stem0), "rb").read()).hexdigest(),
                   "mask_sha256": hashlib.sha256(open(mask_out, "rb").read()).hexdigest()}, stream, indent=2)
    print("TRACK_INIT_2D", {"frame": stem0, "score_2d": float(top["score_2d"]),
                            "click_uv": [click_u, click_v], "mask_px": int(np.count_nonzero(mask0))}, flush=True)
    estimator = WAPREstimator(device=device)
    mesh = estimator.prepare_meshes([mesh])[0]
    center = center_from_mesh(mesh)
    mesh_setup_seconds = estimator.last_mesh_setup_seconds
    init_instances = [{"mesh": mesh, "diameter_m": float(diameter_m), "mask": mask0}]
    warmup_seconds = estimator.warmup_pose(rgb, depth_m, K, init_instances)
    print("POSE_PREPARATION", {"model_setup_seconds": estimator.model_setup_seconds,
          "mesh_setup_seconds": mesh_setup_seconds, "warmup_seconds": warmup_seconds}, flush=True)
    dino = load_tracker_dino(device, dino_name=dino_name)
    # Initialize from the detector mask; no dataset mask file is opened.
    # 使用检测掩码初始化；不读取数据集中的 mask 文件。
    # Estimate frame 0; recovery retains the previous rotation as its starting point.
    # 估计第 0 帧；补救以保留上一旋转的位姿为起点。
    torch.cuda.synchronize(device)
    init_started = time.perf_counter()
    init = estimator.estimate_one_category_one_instance(
        rgb, depth_m, K, mesh, float(diameter_m), mask=mask0,
    )
    torch.cuda.synchronize(device)
    init_hot_seconds = time.perf_counter() - init_started
    pose = np.asarray(init["pose_4x4"], dtype=np.float32)
    print(
        {"frame": stem0, "kind": "init", "pose_hot_seconds": init_hot_seconds,
         "t_m": pose[:3, 3].reshape(3).tolist()},
        flush=True,
    )
    tokens, grid = dino_tokens(dino, rgb, device, dino_short_side=dino_short_side, dino_patch=dino_patch)
    patch = one_instance_mask_to_patches(mask0 > 0, grid, patch_cover=patch_cover, dino_patch=dino_patch)
    template_features = F.normalize(tokens[torch.as_tensor(patch, device=tokens.device)].float(), dim=-1)
    # Warm the actual tracking and every possible search group; discard outputs.
    # 预热实际跟踪及每一种搜索组大小，丢弃输出。
    torch.cuda.synchronize(device)
    tracking_warm_started = time.perf_counter()
    for _ in range(3):
        tokens, grid = dino_tokens(dino, rgb, device, dino_short_side=dino_short_side, dino_patch=dino_patch)
        track_many_categories_many_instances(estimator, rgb, depth_m, K,
            [{"track_id": sequence_name, "mesh": mesh, "diameter_m": float(diameter_m), "pose_4x4": pose}])
        patch_candidates(template_features, tokens, grid, depth_m, pose, vertices, K, diameter_m)
        for count in range(1, local_candidates + global_candidates + 1):
            refine_search_group(estimator, pose, np.repeat(pose[None], count, axis=0),
                                rgb, depth_m, K, mesh, center, diameter_m)
    torch.cuda.synchronize(device)
    print("TRACK_WARMUP", {"seconds": time.perf_counter() - tracking_warm_started, "calls": 3}, flush=True)
    patch = one_instance_mask_to_patches(mask0 > 0, grid, patch_cover=patch_cover, dino_patch=dino_patch)
    template_tokens = tokens
    template_patch = patch
    prev_tokens = tokens
    prev_patch = patch
    initial_depth = (mask0 > 0) & np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(initial_depth):
        raise ValueError("Initial mask has no valid depth / 初始掩码内没有有效深度")
    previous_area_m2 = float(np.count_nonzero(patch)) * float(np.median(depth_m[initial_depth])) ** 2
    pending_region_loss = False
    last_rgb = rgb
    # Propagate every raw frame; update poses at the selected stride.
    # 每个原始帧传播区域；按选定步长更新位姿。
    for frame_index, stem in enumerate(frame_stems(seq_dir)[1:], start=1):
        rgb, depth_m = read_rgbd(seq_dir, stem)
        last_rgb = rgb
        tokens, grid = dino_tokens(dino, rgb, device, dino_short_side=dino_short_side, dino_patch=dino_patch)
        z_prev = max(float(pose[2, 3]), 0.05)
        diameter_px = float(diameter_m) * float(K[0, 0]) / z_prev
        propagated = propagate_one_instance_region(
            prev_tokens, prev_patch, template_tokens, template_patch, tokens, grid, diameter_px,
            cosine_min=cosine_min, min_matches=min_matches, dino_patch=dino_patch,
        )
        region = None
        translation = None
        nmatch = 0
        if propagated is None:
            pending_region_loss = True
        else:
            region, center_uv, nmatch = propagated
            prev_tokens = tokens
            prev_patch = one_instance_mask_to_patches(region, grid, patch_cover=patch_cover, dino_patch=dino_patch)
        if frame_index % int(stride):
            continue
        torch.cuda.synchronize(device)
        pose_started = time.perf_counter()
        previous_pose = pose.copy()
        reasons = []
        area_m2 = None
        if region is not None:
            translation = translation_from_one_instance(
                region, center_uv, depth_m, K, pose[:3, :3], vertices,
                min_region_px=min_region_px, front_percentile=front_percentile,
            )
            valid_depth = region & np.isfinite(depth_m) & (depth_m > 0)
            if np.any(valid_depth):
                observed_z = float(np.median(depth_m[valid_depth]))
                front_z = float(pose[2, 3]) + float(np.percentile((vertices @ pose[:3, :3].T)[:, 2], front_percentile))
                if abs(observed_z - front_z) > depth_gate_m:
                    reasons.append("depth_disagreement")
                area_m2 = float(np.count_nonzero(prev_patch)) * observed_z ** 2
                if area_m2 / previous_area_m2 < region_area_drop_ratio:
                    reasons.append("region_collapse")
        if pending_region_loss or translation is None:
            reasons.append("region_missing")
        center_pose = pose.copy()
        if translation is not None:
            center_pose[:3, 3] = translation
        # Finish the normal update first; it remains a search comparison candidate.
        # 先完成常规更新，保留其结果与搜索候选比较；不使用真值触发或筛选。
        tracked = track_many_categories_many_instances(estimator, rgb, depth_m, K,
            [{"track_id": sequence_name, "mesh": mesh, "diameter_m": float(diameter_m),
              "pose_4x4": previous_pose, "center_pose_4x4": center_pose}])[0]
        pose = np.asarray(tracked["pose_4x4"], dtype=np.float32)
        kind = HYPOTHESES[int(tracked["hypothesis_index"])]
        score = float(tracked["between_group"])
        if score > recover_between_group:
            reasons.append("wbps_quality")
        search_ms = 0.0
        candidate_count = 0
        recovery_accepted = False
        if reasons:
            search_started = time.perf_counter()
            patches = patch_candidates(template_features, tokens, grid, depth_m, previous_pose, vertices, K, diameter_m)
            candidate_count = len(patches)
            if patches:
                seeds = np.stack([front_point_seed(previous_pose, patch, vertices, K) for patch in patches])
                try:
                    pose, selected_index, score = refine_search_group(
                        estimator, pose, seeds, rgb, depth_m, K, mesh, center, diameter_m)
                    recovery_accepted = selected_index > 0
                    kind = "dino_recover" if recovery_accepted else "search_keep"
                except RuntimeError as error:
                    print("TRACK_RECOVERY_FAILED", {"frame": stem, "error": str(error)}, flush=True)
            # Search returns a pose rather than a new segmentation mask.
            # 搜索返回位姿，而非新分割掩码；继续维护自身的 DINOv2 区域状态。
            pending_region_loss = propagated is None
            torch.cuda.synchronize(device)
            search_ms = (time.perf_counter() - search_started) * 1000.0
        if translation is not None and area_m2 is not None:
            previous_area_m2 = area_m2
        torch.cuda.synchronize(device)
        pose_hot_seconds = time.perf_counter() - pose_started
        print(
            {
                "frame": stem,
                "kind": kind,
                "matches": int(nmatch),
                "between_group": score,
                "recovery_trigger": reasons,
                "candidate_count": candidate_count,
                "recovery_accepted": recovery_accepted,
                "search_ms": search_ms,
                "pose_hot_seconds": pose_hot_seconds,
                "t_m": pose[:3, 3].reshape(3).tolist(),
            },
            flush=True,
        )
    visualize_6d_pose(
        last_rgb,
        [{
            "name": mesh.metadata["name"],
            "pose_4x4": pose,
            "mesh": mesh,
        }],
        K,
        filename="one_instance_track.jpg",
    )
