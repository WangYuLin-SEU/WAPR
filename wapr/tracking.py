# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Update known track identities together on one RGB-D frame.

在同一 RGB-D 帧中批量更新身份已知的跟踪实例。
Initialization, region association, and lost-track recovery belong to the caller.
初始化、区域关联与丢失恢复由调用方负责。
"""

import os
from pathlib import Path

import numpy as np
import torch

from wapr import recipe
from wapr.estimator import _rgb_u8, center_from_mesh, poses_original_to_centered
from wapr.export_tracking_engine import tracking_hypotheses
from wapr.nets import _Engine
from wapr.ogl import depth2xyzmap_batch, make_crop_pair
from wapr.pose_groups import wbps_group_sizes


def refine_batch(estimator, net_name, poses, rgb, depth_m, K, mesh, center, diameter_m, iters, max_rot):
    """Refine one mesh's hypothesis rows as a batch, with the mask channel off.

    关闭掩码通道，批量计算修正同一网格的候选姿态。
    """
    pose_t = torch.as_tensor(poses, device=estimator.device, dtype=torch.float32)
    mask = np.zeros(depth_m.shape[:2], dtype=np.uint8)
    out = estimator._refine(
        estimator.nets[net_name], pose_t, rgb, depth_m, mask, K, mesh, center, diameter_m,
        iters=iters, max_rot=max_rot, crop_ratio=recipe.refine_crop_ratio, use_mask=False,
    )
    return out.detach().float().cpu().numpy()


def build_hypotheses(estimator, previous, center_pose, rgb, depth_m, K, mesh, center, diameter_m):
    """Build six candidates in tracking_hypotheses order, batching shared prefixes.

    按 tracking_hypotheses 顺序构造六个候选，共享前缀批量执行。
    """
    wapr_in = np.stack([center_pose, previous], axis=0)
    wapr_out = refine_batch(
        estimator, "wapr_wo_mask", wapr_in, rgb, depth_m, K, mesh, center, diameter_m,
        recipe.wapr_iters, recipe.wapr_max_rot_rad,
    )
    sapr_in = np.stack([center_pose, wapr_out[0], wapr_out[1]], axis=0)
    sapr_out = refine_batch(
        estimator, "sapr", sapr_in, rgb, depth_m, K, mesh, center, diameter_m,
        recipe.sapr_iters, recipe.sapr_max_rot_rad,
    )
    return np.stack([previous, center_pose, wapr_out[0], sapr_out[0], sapr_out[1], sapr_out[2]], axis=0)


def choose_pose(estimator, poses, rgb, depth_m, K, mesh, center, diameter_m):
    """Select the largest within_group score and return all between_group errors.

    选择 within_group 分数最大者，返回全部候选的 between_group 误差。
    """
    pose_t = torch.as_tensor(poses, device=estimator.device, dtype=torch.float32)
    mask = np.zeros(depth_m.shape[:2], dtype=np.uint8)
    within, between = estimator._wbps(
        pose_t, rgb, depth_m, mask, K, mesh, center, diameter_m, len(poses),
    )
    within_np = within.reshape(-1).detach().float().cpu().numpy()
    between_np = between.reshape(-1).detach().float().cpu().numpy()
    return int(np.argmax(within_np)), between_np


def between_group_of_pose(estimator, pose, rgb, depth_m, mask, K, mesh, center, diameter_m):
    """Score a complete original-object pose group (N, 4, 4), N >= 2.

    评估原始物体系下的完整位姿组 (N, 4, 4)，N >= 2。
    """
    pose_t = torch.as_tensor(pose, device=estimator.device, dtype=torch.float32).reshape(-1, 4, 4)
    count = int(pose_t.shape[0])
    if count < 2:
        raise ValueError("Pass a pose group, or reuse the selected candidate's existing WBPS score")
    _within, between = estimator._wbps(
        pose_t, rgb, depth_m, mask, K, mesh, center, float(diameter_m), count,
    )
    return between.reshape(-1).detach().float().cpu().numpy()


def track_many_categories_many_instances(estimator, rgb, depth_m, K, instances):
    """Batch the six-candidate pose update for one or several tracked instances.

    批量执行一个或多个跟踪实例的六候选位姿更新。

    Each item supplies a unique track_id, mesh, diameter_m, and previous
    pose_4x4 (object to camera, meters). Optional center_pose_4x4 supplies the
    same rotation with translation from a propagated region and sensor depth;
    without it, the previous pose is used. obj_id is optional metadata.
    每项提供唯一 track_id、mesh、diameter_m，以及上一帧 pose_4x4（物体到
    相机、米）。可选 center_pose_4x4 保留原旋转，平移来自传播区域与传感器
    深度；未提供时使用上一帧位姿。obj_id 是可选类别信息。

    WAPR/SAPR and WBPS run across instances; six hypotheses remain an independent
    score group per instance. TensorRT uses wbps_batch.engine, or the explicitly
    exported six-hypothesis tracking engine. Torch also scores groups together. Larger lists
    are chunked at the engine row limit. Empty input returns []. No GT or mask
    is read here, and no ID assignment, redetection, or automatic rescue occurs.
    within_group selects the pose; between_group is the complete group's maximum.
    WAPR/SAPR 与 WBPS 跨实例批量计算，每个实例的六个候选姿态仍独立评分。
    TensorRT 使用 wbps_batch.engine，或显式导出的六候选跟踪引擎；torch 路径同样批量计算评分。超过引擎行数
    上限时分块。空列表返回 []。此处不读取真值或 mask，不分配身份、不重新
    检测，也不自动恢复。输出顺序与输入一致，保留 track_id 和可选 obj_id。
    within_group 选择位姿；between_group 返回该实例完整候选组的最大值。
    """
    if not instances:
        return []
    rgb_u8 = _rgb_u8(rgb)
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    if depth.ndim != 2 or rgb_u8.shape[:2] != depth.shape:
        raise ValueError("RGB and depth sizes differ")
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics")
    track_ids = [item["track_id"] for item in instances]
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("track_id must be unique, including instances of the same category")
    meshes = []
    centers = []
    diameters = []
    previous = []
    centered = []
    prepared_meshes = estimator._pose_meshes([item["mesh"] for item in instances])
    prepared_ids = [estimator.renderer.cached_mesh_id(mesh) for mesh in prepared_meshes]
    for index, item in enumerate(instances):
        pose = np.asarray(item["pose_4x4"], dtype=np.float32)
        center_pose = np.asarray(item.get("center_pose_4x4", pose), dtype=np.float32)
        for candidate in (pose, center_pose):
            if candidate.shape != (4, 4) or not np.isfinite(candidate).all() or candidate[2, 3] <= 0:
                raise ValueError("Tracking requires a finite 4x4 pose with positive depth")
            if not np.allclose(candidate[3], [0, 0, 0, 1]):
                raise ValueError("Invalid homogeneous pose row")
        if not np.allclose(center_pose[:3, :3], pose[:3, :3]):
            raise ValueError("center_pose_4x4 must preserve the previous rotation")
        diameter = float(item["diameter_m"])
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError("diameter_m must be finite and positive")
        mesh = prepared_meshes[index]
        meshes.append(mesh)
        centers.append(center_from_mesh(mesh))
        diameters.append(diameter)
        previous.append(pose)
        centered.append(center_pose)

    group = len(tracking_hypotheses)
    wapr_engine = estimator.nets["wapr_wo_mask"].engine
    sapr_engine = estimator.nets["sapr"].engine
    batch_engine_path = os.path.join(os.path.dirname(os.fspath(recipe.engine_file("wbps"))), "wbps_batch.engine")
    if wapr_engine is not None and os.path.isfile(batch_engine_path):
        score_engine = None
        objects_per_batch = min(int(wapr_engine.max_batch) // 2,
                                int(sapr_engine.max_batch) // 3,
                                int(estimator.nets["wbps"].engine.max_batch) // group)
    elif wapr_engine is not None:
        # Load an independent resource only when the tracking entry is used.
        # 只在使用跟踪入口时载入独立引擎资源。
        score_engine = getattr(estimator, "tracking_score_engine", None)
        if score_engine is None:
            engine_path = os.path.join(os.path.dirname(os.fspath(recipe.engine_file("wbps"))),
                                       "wbps_tracking.engine")
            if not os.path.isfile(engine_path):
                raise FileNotFoundError("Run python -m wapr.export_tracking_engine on the inference GPU: " + engine_path)
            score_engine = _Engine(Path(engine_path), ("within_group", "between_group"))
            if score_engine.min_batch != group or score_engine.max_batch != max(wbps_group_sizes):
                raise RuntimeError("Tracking engine profile differs; rebuild wbps_tracking.engine")
            estimator.tracking_score_engine = score_engine
        objects_per_batch = min(int(wapr_engine.max_batch) // 2,
                                int(sapr_engine.max_batch) // 3, int(score_engine.max_batch) // group)
    else:
        score_engine = None
        # Bound crop storage by the same row budget as the released engines.
        # 裁剪存储沿用发布引擎相同的行数预算。
        objects_per_batch = max(wbps_group_sizes) // group
    if objects_per_batch < 1:
        raise RuntimeError("Engine profile cannot hold one tracking instance")
    rgb_device = torch.as_tensor(rgb_u8, device=estimator.device, dtype=torch.float32)
    depth_device = torch.as_tensor(depth, device=estimator.device, dtype=torch.float32)
    camera_device = torch.as_tensor(K, device=estimator.device, dtype=torch.float32)[None]
    observed_xyz = depth2xyzmap_batch(depth_device[None], camera_device).permute(0, 3, 1, 2)
    mask_device = torch.zeros_like(depth_device)
    results = []
    for start in range(0, len(instances), objects_per_batch):
        stop = min(start + objects_per_batch, len(instances))
        count = stop - start
        previous_t = torch.as_tensor(np.stack(previous[start:stop]), device=estimator.device)
        center_t = torch.as_tensor(np.stack(centered[start:stop]), device=estimator.device)
        batch_centers = np.stack(centers[start:stop]).astype(np.float32)
        batch_diameters = np.asarray(diameters[start:stop], dtype=np.float32)
        batch_meshes = meshes[start:stop]
        # Two WAPR prefixes and three SAPR prefixes are batched across all
        # instances; loops below only assemble mesh rows and output records.
        # 两个 WAPR 前缀和三个 SAPR 前缀跨全部实例批量计算；下面的循环只组织网格
        # 行与输出记录，不逐物体调用神经网络。
        wapr_in = torch.stack((center_t, previous_t), dim=1).reshape(count * 2, 4, 4)
        wapr_out = estimator._refine(
            estimator.nets["wapr_wo_mask"], wapr_in, rgb_device, depth_device, mask_device,
            K, batch_meshes[0], np.repeat(batch_centers, 2, axis=0), np.repeat(batch_diameters, 2),
            iters=recipe.wapr_iters, max_rot=recipe.wapr_max_rot_rad,
            crop_ratio=recipe.refine_crop_ratio, use_mask=False,
            meshes=[mesh for mesh in batch_meshes for _ in range(2)],
            group_size=2,
            observed_xyz=observed_xyz,
            mesh_ids=[mid for mid in prepared_ids[start:stop] for _ in range(2)],
        ).reshape(count, 2, 4, 4)
        sapr_in = torch.stack((center_t, wapr_out[:, 0], wapr_out[:, 1]), dim=1).reshape(count * 3, 4, 4)
        sapr_out = estimator._refine(
            estimator.nets["sapr"], sapr_in, rgb_device, depth_device, mask_device,
            K, batch_meshes[0], np.repeat(batch_centers, 3, axis=0), np.repeat(batch_diameters, 3),
            iters=recipe.sapr_iters, max_rot=recipe.sapr_max_rot_rad,
            crop_ratio=recipe.refine_crop_ratio, use_mask=False,
            meshes=[mesh for mesh in batch_meshes for _ in range(3)],
            group_size=3,
            observed_xyz=observed_xyz,
            mesh_ids=[mid for mid in prepared_ids[start:stop] for _ in range(3)],
        ).reshape(count, 3, 4, 4)
        candidates = torch.stack((previous_t, center_t, wapr_out[:, 0], sapr_out[:, 0],
                                  sapr_out[:, 1], sapr_out[:, 2]), dim=1)
        score_centers = np.repeat(batch_centers, group, axis=0)
        score_poses = poses_original_to_centered(candidates.reshape(count * group, 4, 4), score_centers)
        A, B, _poses, _diameters = make_crop_pair(
            rgb_device, depth_device, mask_device, score_poses, K, batch_meshes[0],
            np.repeat(batch_diameters, group), crop_ratio=recipe.wbps_crop_ratio,
            use_mask=False, device=estimator.device,
            meshes=[mesh for mesh in batch_meshes for _ in range(group)],
            observed_xyz=observed_xyz,
            mesh_ids=[mid for mid in prepared_ids[start:stop] for _ in range(group)],
        )
        with torch.no_grad():
            if score_engine is None:
                scores = estimator.nets["wbps"](A, B, L=group)
            else:
                scores = score_engine.run({"A": A, "B": B})
        if any(tuple(scores[name].shape) != (count, group)
               for name in ("within_group", "between_group")):
            raise RuntimeError("Tracking scorer must preserve the independent object/group axes")
        within = scores["within_group"].reshape(count, group)
        between = scores["between_group"].reshape(count, group)
        best = within.argmax(dim=1)
        selected = candidates[torch.arange(count, device=best.device), best].detach().float().cpu().numpy()
        # Quality uses the maximum along the complete candidate axis, separately
        # for each instance. within_group still selects the pose. Transfer once.
        # 质量分数沿完整候选轴取最大值，每实例独立；位姿仍由 within_group
        # 选择。每块只回传一次结果张量。
        group_between = between.max(dim=1).values.detach().float().cpu().numpy()
        choices = best.detach().cpu().numpy()
        for index in range(count):
            item = instances[start + index]
            pose = selected[index]
            row = {"track_id": item["track_id"], "pose_4x4": pose, "R": pose[:3, :3].copy(),
                   "t_m": pose[:3, 3].copy(), "hypothesis": tracking_hypotheses[int(choices[index])],
                   "hypothesis_index": int(choices[index]), "between_group": float(group_between[index]),
                   "score_6d": (100.0 - float(group_between[index])) / 200.0}
            if "obj_id" in item:
                row["obj_id"] = item["obj_id"]
            results.append(row)
    return results
