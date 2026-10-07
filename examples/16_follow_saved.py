# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 16. Previous: examples/15_xarm_cube.py. Next: examples/17_virtual_grasp.py
# 示例 16。上一例：examples/15_xarm_cube.py。下一例：examples/17_virtual_grasp.py
# The saved robot clips are videos. Later frames follow the previous pose.
# 存下来的机器人片段是视频。后面的帧跟着上一帧的位姿走。
# python examples/16_follow_saved.py
# 运行：python examples/16_follow_saved.py
"""Track saved robot RGB-D frames with shared features and batched pose updates.

共享特征并批量更新位姿，跟踪保存的机器人 RGB-D 画面。

The first frame uses the verified full prediction saved by the producer.
Each camera frame is encoded once by DINOv2. Tracks retain their own region
and identity state; all pose hypotheses render, refine and score together.
Observed depth supplies translation, and WBPS selects among six candidates
per object. Annotated poses are read only by the subsequent error report.
首帧使用生成脚本保存并核验的完整预测。每相机帧只编码一次 DINOv2。
各轨迹保留独立区域和身份状态，全部候选姿态共同渲染、修正与评分。
观测深度提供平移，WBPS 从每物体六个候选中选择。标注位姿仅由后续误差报告读取。
"""

import hashlib
import importlib
import json
import os
import shutil
import sys
import time

import cv2
import torch
import trimesh
import numpy as np


# The pose/camera helpers and region policy are published beside this recipe.
# 位姿、相机与区域策略均由相邻的公开示例提供。
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.dirname(EXAMPLES_DIR)
REGION_DIR = os.path.join(EXAMPLES_DIR, "16_follow_saved")
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)
if REGION_DIR not in sys.path:
    sys.path.insert(0, REGION_DIR)

known = importlib.import_module("13_known_mesh_place")
import region_tracking as tracker
from region_tracking import (
    HYPOTHESES,
    agreement_mask,
    centered_poses,
    dino_tokens,
    load_dino,
    min_region_px,
    one_instance_mask_to_patches,
    patches_to_mask,
    propagate_one_instance_region,
    render_pose_rgbd,
    translation_from_one_instance,
    visible_region,
)
from wapr.estimator import WAPREstimator, center_from_mesh, prepare_mesh
from wapr.tracking import track_many_categories_many_instances


DEVICE = "cuda:0"
# Object-to-camera poses already written by the full estimate. The backup keeps that file.
# 完整估计已经写好的物体到相机位姿。备份留着那份文件。
FULL_SUFFIX = "_full.json"
# Discard three complete region + pose + raster calls before timing each batch shape.
# 每种批形状计时前丢弃三次完整区域、位姿和光栅调用。
warmup_calls = 3


def read_rgb_depth(job):
    """RGB uint8 and depth in meters for one saved frame.

    一帧存下来的画面。RGB 是 uint8，深度是米。
    """

    bgr = cv2.imread(job["rgb"], cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("missing rgb %s" % job["rgb"])
    depth_m = np.load(job["depth"]).astype(np.float32)
    camera_k = np.asarray(job["K"], dtype=np.float64).reshape(3, 3)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth_m, camera_k


def backup_pose_file(path):
    """Keep a full estimate only while its recorded RGB-D inputs still agree.

    仅在保存的 RGB-D 输入一致时复用完整估计备份。
    """
    full_path = os.path.splitext(path)[0] + FULL_SUFFIX
    metadata_path = full_path + ".inputs.json"
    folder = os.path.dirname(path)
    input_paths = set()
    for name in ("jobs.json", "rows.json"):
        record_path = os.path.join(folder, name)
        if not os.path.isfile(record_path):
            continue
        input_paths.add(record_path)
        with open(record_path, encoding="utf-8") as stream:
            pending = [json.load(stream)]
        while pending:
            item = pending.pop()
            if isinstance(item, list):
                pending.extend(item)
            elif isinstance(item, dict):
                for key in ("rgb", "depth", "mask"):
                    if isinstance(item.get(key), str):
                        input_paths.add(item[key])
                pending.extend(item.values())
    for mesh_folder in (folder, os.path.join(folder, "meshes")):
        if os.path.isdir(mesh_folder):
            input_paths.update(os.path.join(mesh_folder, name) for name in os.listdir(mesh_folder)
                               if name.endswith(".ply"))
    input_hashes = {}
    for input_path in sorted(input_paths):
        with open(input_path, "rb") as stream:
            input_hashes[os.path.abspath(input_path)] = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()
    with open(path, "r", encoding="utf-8") as stream:
        rows = json.load(stream)
    if any("choice" in row for row in rows):
        if not os.path.isfile(full_path) or not os.path.isfile(metadata_path):
            raise RuntimeError("Rerun the producer (13–15) to create a verified full estimate / 请重跑 13–15 生成可核验的完整估计")
        with open(metadata_path, encoding="utf-8") as stream:
            stored = json.load(stream)
        with open(full_path, "rb") as stream:
            full_hash = hashlib.sha256(stream.read()).hexdigest()
        if stored.get("inputs") != input_hashes or stored.get("full_sha256") != full_hash:
            raise RuntimeError("Saved inputs changed; rerun 13–15 before tracking / 保存输入已改变，请先重跑 13–15")
        return full_path
    # A newly produced full estimate replaces the previous run's backup.
    # 新生成的完整估计刷新旧备份，不能保留上一轮结果。
    shutil.copy2(path, full_path)
    with open(full_path, "rb") as stream:
        full_hash = hashlib.sha256(stream.read()).hexdigest()
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump({"inputs": input_hashes, "full_sha256": full_hash,
                   "pose_coordinates": "original_object_to_camera_m"}, stream, indent=2)
    return full_path


def warm_tracking_path(estimator, runtime, dino, jobs, states, meshes, mesh_ids):
    """Discard full updates without advancing any persistent track state.

    丢弃完整更新，不推进任何持久跟踪状态。
    """

    rgb, depth_m, K = read_rgb_depth(jobs[0])
    height, width = depth_m.shape
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _call in range(warmup_calls):
        tokens, grid = dino_tokens(dino, rgb)
        instances = []
        for job in jobs:
            state = states[job["name"]]
            mesh, _center, vertices, diameter_m = meshes[job["name"]]
            pose = state["pose"]
            diameter_px = diameter_m * float(K[0, 0]) / max(float(pose[2, 3]), 0.05)
            propagated = propagate_one_instance_region(state["prev_tokens"], state["prev_patch"],
                state["template_tokens"], state["template_patch"], tokens, grid, diameter_px, state["gate"])
            center_pose = pose.copy()
            if propagated is not None:
                region, uv, _matches = propagated
                translation = translation_from_one_instance(region, uv, depth_m, K, pose[:3, :3], vertices)
                if translation is not None:
                    center_pose[:3, 3] = translation
            instances.append({"track_id": job["name"], "mesh": mesh, "diameter_m": diameter_m,
                              "pose_4x4": pose, "center_pose_4x4": center_pose})
        outputs = track_many_categories_many_instances(estimator, rgb, depth_m, K, instances)
        poses = np.stack([out["pose_4x4"] for out in outputs]).astype(np.float32)
        centers = np.stack([meshes[job["name"]][1] for job in jobs])
        poses[:, :3, 3] += (poses[:, :3, :3] @ centers[..., None]).squeeze(-1)
        boxes = np.tile([0, 0, width, height], (len(jobs), 1))
        ids = [mesh_ids[job["name"]] for job in jobs]
        _rgb, depths = runtime._render_mixed_tiles(ids, poses, boxes, K, height, width, (height, width))
        for rendered in depths.detach().cpu().numpy():
            region, _source = visible_region(rendered, depth_m)
            one_instance_mask_to_patches(region, grid)
    torch.cuda.synchronize()
    print("FOLLOW_WARMUP", {"calls": warmup_calls, "instances": len(jobs),
                           "seconds_separate": time.perf_counter() - started,
                           "backend": "configured_estimator"}, flush=True)


def load_pose_map(path):
    """{frame id: object-to-camera 4x4}.

    {帧编号: 物体到相机 4x4}。
    """
    with open(path, "r", encoding="utf-8") as stream:
        rows = json.load(stream)
    poses = {}
    for row in rows:
        if "pose" not in row:
            continue
        poses[row["id"]] = np.asarray(row["pose"], dtype=np.float64).reshape(4, 4)
    return poses


def write_pose_list(path, written):
    """Same list shape as the full-estimate file.

    和完整估计那个文件一样的列表。
    """
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(written, stream)
    print("WROTE_POSES", path, "n", len(written), flush=True)


def prepare_one_mesh(path):
    """Centered mesh for the refiner, and the original vertices for the translation.

    给修正网络的居中网格，以及给平移用的原始顶点。
    """

    raw = trimesh.load(path, force="mesh", process=False)
    vertices = np.asarray(raw.vertices, dtype=np.float64).copy()
    mesh = prepare_mesh(raw)
    center = np.asarray(center_from_mesh(mesh), dtype=np.float32).reshape(3)
    return mesh, center, vertices


def track_stream(estimator, runtime, dino, jobs, init_pose, mesh, center, diameter_m, vertices, mesh_id):
    """One object, one camera. Returns pose rows and the later-frame times in milliseconds.

    一个物体，一路相机。返回位姿行，以及后面每一帧的毫秒数。
    """

    pose = np.asarray(init_pose, dtype=np.float64).reshape(4, 4).copy()
    rgb, depth_m, camera_k = read_rgb_depth(jobs[0])
    height, width = depth_m.shape[:2]
    _rgb, rendered_depth = render_pose_rgbd(
        runtime, mesh_id, centered_poses(pose[None], center), camera_k, height, width,
    )
    region, region_source = visible_region(rendered_depth[0], depth_m)
    if int(region.sum()) < min_region_px:

        mask = cv2.imread(jobs[0]["mask"], cv2.IMREAD_GRAYSCALE)
        region = np.asarray(mask) > 0
        region_source = "mask"
    tokens, grid = dino_tokens(dino, rgb)
    patch = one_instance_mask_to_patches(region, grid)
    object_patches = int(patch.sum())
    if object_patches == 0:
        raise RuntimeError("empty region %s" % jobs[0]["id"])
    # YCBInEOAT keeps a match when at least 8 patches agree. A carrot, an
    # eggplant, or the cube covers fewer than 8 patches in this overview, so
    # that gate can never open. The gate used for this stream is the smaller
    # of 8 and the patches this object actually covers.
    # YCBInEOAT 至少 8 个图块对上才留下。胡萝卜、茄子和方块在这台总览里盖不满
    # 8 个图块，那个门槛永远打不开。这一条用的门槛是 8 和这个物体实际图块数里较小的那个。
    gate = min(int(tracker.min_matches), object_patches)
    print(
        "REGION", jobs[0]["id"], region_source,
        "patches", object_patches, "gate", gate, flush=True,
    )
    template_tokens = tokens
    template_patch = patch
    prev_tokens = tokens
    prev_patch = patch
    if len(jobs) > 1:
        warm_job = dict(jobs[1], name="object")
        warm_state = {"pose": pose, "prev_tokens": tokens, "prev_patch": patch,
                      "template_tokens": tokens, "template_patch": patch, "gate": gate}
        warm_tracking_path(estimator, runtime, dino, [warm_job], {"object": warm_state},
                           {"object": (mesh, center, vertices, diameter_m)}, {"object": mesh_id})
    written = [{"id": jobs[0]["id"], "pose": pose.tolist(), "choice": "init", "region": region_source}]
    times_ms = []
    # The published tracker asks for 8 agreeing patches. These overview objects
    # cover fewer than that, so this stream uses the patch count it actually has.
    # 发布的跟踪要求 8 个图块对上。这些总览里的物体盖不满这个数，所以这一条用它实际有的图块数。
    written, times_ms = _follow_later_frames(
            estimator, runtime, dino, jobs, pose, mesh, center, diameter_m, vertices,
            mesh_id, height, width, camera_k, tokens, grid, template_tokens, template_patch,
            prev_tokens, prev_patch, written, times_ms, gate,
    )
    return written, times_ms


def _follow_later_frames(
    estimator, runtime, dino, jobs, pose, mesh, center, diameter_m, vertices,
    mesh_id, height, width, camera_k, tokens, grid, template_tokens, template_patch,
    prev_tokens, prev_patch, written, times_ms, gate,
):
    """Frames after the first. The match gate is already set for this stream.

    第一帧之后。这一条的匹配门槛已经设好。
    """

    for step, job in enumerate(jobs[1:], start=1):
        rgb, depth_m, camera_k = read_rgb_depth(job)
        torch.cuda.synchronize()
        started = time.perf_counter()
        tokens, grid = dino_tokens(dino, rgb)
        z_prev = max(float(pose[2, 3]), 0.05)
        diameter_px = float(diameter_m) * float(camera_k[0, 0]) / z_prev
        propagated = propagate_one_instance_region(
            prev_tokens, prev_patch, template_tokens, template_patch, tokens, grid, diameter_px, gate,
        )
        if propagated is None:
            flat = np.flatnonzero(prev_patch)
            region = patches_to_mask(flat // grid["gw"], flat % grid["gw"], grid)
            translation = None
            nmatch = 0
        else:
            region, center_uv, nmatch = propagated
            translation = translation_from_one_instance(
                region, center_uv, depth_m, camera_k, pose[:3, :3], vertices,
            )
        center_pose = pose.copy()
        if translation is not None:
            center_pose[:3, 3] = translation
        result = track_many_categories_many_instances(estimator, rgb, depth_m, camera_k,
            [{"track_id": job["name"] if "name" in job else "object", "mesh": mesh,
              "diameter_m": diameter_m, "pose_4x4": pose, "center_pose_4x4": center_pose}])[0]
        choice = result["hypothesis_index"]
        pose = np.asarray(result["pose_4x4"], dtype=np.float64).copy()
        posed = centered_poses(pose[None], center)
        _rgb, posed_depth = render_pose_rgbd(runtime, mesh_id, posed, camera_k, height, width)
        refreshed, _source = visible_region(posed_depth[0], depth_m)
        rendered_px = int((posed_depth[0] > 1.0e-3).sum())
        agreed_px = int(agreement_mask(posed_depth[0], depth_m).sum())
        if rendered_px > 0 and agreed_px >= min_region_px:
            stored = refreshed
        else:
            stored = region
        prev_tokens = tokens
        prev_patch = one_instance_mask_to_patches(stored, grid)
        if int(prev_patch.sum()) == 0:
            prev_patch = one_instance_mask_to_patches(region, grid)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        times_ms.append(elapsed_ms)
        label = HYPOTHESES[choice][0]
        written.append({
            "id": job["id"],
            "pose": pose.tolist(),
            "choice": label,
            "matches": int(nmatch),
        })
        if step == 1 or step % 40 == 0:
            print(
                "FOLLOW", job["id"], label, "matches", int(nmatch),
                "ms", round(elapsed_ms, 1), flush=True,
            )
    return written, np.asarray(times_ms, dtype=np.float64)


def summarize_times(name, times_ms):
    """Median and p90 of the later frames. The init frame is not in this array.

    后面每一帧的中位数和 p90。第一帧不在这个数组里。
    """
    if len(times_ms) == 0:
        print("TIME", name, "n", 0, flush=True)
        return
    print(
        "TIME", name,
        "n", int(len(times_ms)),
        "median_ms", round(float(np.median(times_ms)), 1),
        "p90_ms", round(float(np.percentile(times_ms, 90)), 1),
        flush=True,
    )


def stream_world_error(rows, poses, camera_name, object_name, gt_key):
    """Median center, mesh +Z, and mesh +X, over frames that have a tracked pose.

    有跟踪位姿的帧上，中心、网格 +Z、网格 +X 的中位数。
    """
    centers = []
    axis_z = []
    axis_x = []
    for row in rows:
        camera = row["cameras"][camera_name]
        extrinsic = np.asarray(camera["extrinsic"], dtype=np.float64).reshape(4, 4)
        gt = np.asarray(row[gt_key], dtype=np.float64).reshape(4, 4)
        for job in camera["jobs"]:
            if job["name"] != object_name or job["id"] not in poses:
                continue
            pred = known.world_pose(extrinsic, poses[job["id"]])
            center_mm, z_deg = known.pose_error(pred, gt)
            x_deg = known.axis_angle_deg(pred[:3, 0], gt[:3, 0])
            centers.append(center_mm)
            axis_z.append(z_deg)
            axis_x.append(x_deg)
    if not centers:
        print("ERR", camera_name, object_name, "n", 0, flush=True)
        return
    print(
        "ERR", camera_name, object_name,
        "n", len(centers),
        "center_mm", round(float(np.median(centers)), 3),
        "axis_z_deg", round(float(np.median(axis_z)), 3),
        "axis_x_deg", round(float(np.median(axis_x)), 3),
        flush=True,
    )


def overview_error(rows, poses):
    """Median center and the two horizontal mesh axes. The pose is object-to-camera.

    中心和两根水平网格轴的中位数。位姿是物体到相机。
    """
    centers = []
    axis_z = []
    axis_x = []
    xy = []
    for row in rows:
        job_id = row.get("job")
        if job_id not in poses:
            continue
        pred = known.world_pose(np.asarray(row["render_extrinsic"]), poses[job_id])
        gt = np.asarray(row["gt"], dtype=np.float64).reshape(4, 4)
        center_mm, z_deg = known.pose_error(pred, gt)
        x_deg = known.axis_angle_deg(pred[:3, 0], gt[:3, 0])
        centers.append(center_mm)
        axis_z.append(z_deg)
        axis_x.append(x_deg)
        xy.append(float(np.linalg.norm(gt[:2, 3] - np.asarray(row["target"], dtype=np.float64)[:2]) * 1000.0))
    if not centers:
        return
    print(
        "ERR overview",
        "n", len(centers),
        "center_mm", round(float(np.median(centers)), 3),
        "axis_z_deg", round(float(np.median(axis_z)), 3),
        "axis_x_deg", round(float(np.median(axis_x)), 3),
        "last_xy_mm", round(xy[-1], 2),
        flush=True,
    )


def _track_camera_frames(estimator, runtime, dino, frames, init_poses, meshes, mesh_ids):
    """Share DINO and batch every object's pose update on each camera frame.

    每路相机逐帧共享 DINO 特征，并批量更新该帧全部物体位姿。
    Region association keeps the existing per-track gate and depth policy.
    区域关联保留原来的逐轨迹门槛和深度规则。
    """

    states = {}
    written = []
    times_ms = []
    warmed_shapes = set()
    for jobs in frames:
        if not jobs:
            continue
        rgb, depth_m, K = read_rgb_depth(jobs[0])
        height, width = depth_m.shape
        persistent_jobs = [job for job in jobs if job["name"] in states]
        warm_shape = (height, width, tuple(job["name"] for job in persistent_jobs))
        if persistent_jobs and warm_shape not in warmed_shapes:
            warm_tracking_path(estimator, runtime, dino, persistent_jobs, states, meshes, mesh_ids)
            warmed_shapes.add(warm_shape)
        torch.cuda.synchronize()
        started = time.perf_counter()
        tokens, grid = dino_tokens(dino, rgb)
        new_jobs = [job for job in jobs if job["name"] not in states]
        if new_jobs:
            poses = np.stack([init_poses[job["id"]] for job in new_jobs]).astype(np.float32)
            centers = np.stack([meshes[job["name"]][1] for job in new_jobs])
            poses[:, :3, 3] += (poses[:, :3, :3] @ centers[..., None]).squeeze(-1)
            boxes = np.tile([0, 0, width, height], (len(new_jobs), 1))
            ids = [mesh_ids[job["name"]] for job in new_jobs]
            _rgb, depths = runtime._render_mixed_tiles(ids, poses, boxes, K, height, width, (height, width))
            depths = depths.detach().cpu().numpy()
            for job, rendered in zip(new_jobs, depths):
                region, source = visible_region(rendered, depth_m)
                if int(region.sum()) < min_region_px:
                    mask = cv2.imread(job["mask"], cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise ValueError("Missing recorded prediction mask: " + job["mask"])
                    region = mask > 0
                    source = "recorded_prediction"
                patch = one_instance_mask_to_patches(region, grid)
                if not np.any(patch):
                    raise ValueError("Empty initial tracked region: " + job["id"])
                states[job["name"]] = {"pose": np.asarray(init_poses[job["id"]]).copy(),
                    "template_tokens": tokens, "template_patch": patch,
                    "prev_tokens": tokens, "prev_patch": patch,
                    "gate": min(int(tracker.min_matches), int(patch.sum()))}
                written.append({"id": job["id"], "pose": states[job["name"]]["pose"].tolist(),
                                "choice": "init", "region": source})
        initial_ids = {job["id"] for job in new_jobs}
        update_jobs = [job for job in jobs if job["id"] not in initial_ids]
        instances, propagated_regions, matches = [], [], []
        for job in update_jobs:
            state = states[job["name"]]
            mesh, _center, vertices, diameter_m = meshes[job["name"]]
            pose = state["pose"]
            diameter_px = diameter_m * float(K[0, 0]) / max(float(pose[2, 3]), 0.05)
            propagated = propagate_one_instance_region(state["prev_tokens"], state["prev_patch"],
                state["template_tokens"], state["template_patch"], tokens, grid, diameter_px, state["gate"])
            center_pose = pose.copy()
            if propagated is None:
                flat = np.flatnonzero(state["prev_patch"])
                region = patches_to_mask(flat // grid["gw"], flat % grid["gw"], grid)
                nmatch = 0
            else:
                region, center_uv, nmatch = propagated
                translation = translation_from_one_instance(region, center_uv, depth_m, K, pose[:3, :3], vertices)
                if translation is not None:
                    center_pose[:3, 3] = translation
            instances.append({"track_id": job["name"], "mesh": mesh, "diameter_m": diameter_m,
                              "pose_4x4": pose, "center_pose_4x4": center_pose})
            propagated_regions.append(region)
            matches.append(nmatch)
        # One neural pose update for all persistent tracks in this image.
        # 同图全部已有轨迹共用一次神经网络位姿更新。
        outputs = track_many_categories_many_instances(estimator, rgb, depth_m, K, instances)
        if update_jobs:
            poses = np.stack([out["pose_4x4"] for out in outputs]).astype(np.float32)
            centers = np.stack([meshes[job["name"]][1] for job in update_jobs])
            poses[:, :3, 3] += (poses[:, :3, :3] @ centers[..., None]).squeeze(-1)
            boxes = np.tile([0, 0, width, height], (len(update_jobs), 1))
            ids = [mesh_ids[job["name"]] for job in update_jobs]
            _rgb, depths = runtime._render_mixed_tiles(ids, poses, boxes, K, height, width, (height, width))
            depths = depths.detach().cpu().numpy()
            for job, out, rendered, region, nmatch in zip(update_jobs, outputs, depths, propagated_regions, matches):
                state = states[job["name"]]
                state["pose"] = np.asarray(out["pose_4x4"], dtype=np.float64)
                refreshed, _source = visible_region(rendered, depth_m)
                stored = refreshed if (np.any(rendered > 1e-3) and
                    int(agreement_mask(rendered, depth_m).sum()) >= min_region_px) else region
                patch = one_instance_mask_to_patches(stored, grid)
                state["prev_patch"] = patch if np.any(patch) else one_instance_mask_to_patches(region, grid)
                state["prev_tokens"] = tokens
                written.append({"id": job["id"], "pose": state["pose"].tolist(),
                    "choice": HYPOTHESES[out["hypothesis_index"]][0], "matches": int(nmatch)})
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - started) * 1000.0)
    return written, np.asarray(times_ms, dtype=np.float64)


def follow_bottle(estimator, runtime, dino, mesh_ids):
    """Table bottle, table box, and wrist bottle. Three streams, one pose file.

    桌上瓶子、桌上盒子、腕部瓶子。三条序列，一份位姿文件。
    """
    track_dir = os.path.join(RELEASE_DIR, "outputs", "sim_known_mesh_place", "tracking")
    pose_path = os.path.join(track_dir, "sequence_poses.json")
    full_path = backup_pose_file(pose_path)
    init_poses = load_pose_map(full_path)
    with open(os.path.join(track_dir, "rows.json"), "r", encoding="utf-8") as stream:
        rows = json.load(stream)
    streams = {}
    for row in rows:
        for camera_name, camera in row["cameras"].items():
            streams.setdefault(camera_name, []).append(camera["jobs"])
    meshes = {}
    for name in ("bottle", "box"):
        mesh_path = os.path.join(track_dir, "meshes", name + ".ply")
        mesh, center, vertices = prepare_one_mesh(mesh_path)
        mesh = estimator.prepare_meshes([mesh])[0]
        meshes[name] = (mesh, center, vertices, known.mesh_diameter_m(mesh))
        if name not in mesh_ids:
            mesh_ids[name] = runtime.load_mesh_trimesh(mesh, name=name)
    written = []
    all_times = []
    for camera_name, frames in streams.items():
        print("CAMERA_STREAM", camera_name, "frames", len(frames), flush=True)
        rows_out, times_ms = _track_camera_frames(estimator, runtime, dino, frames, init_poses, meshes, mesh_ids)
        written.extend(rows_out)
        all_times.append(times_ms)
        summarize_times(camera_name, times_ms)
    write_pose_list(pose_path, written)
    poses = {row["id"]: np.asarray(row["pose"], dtype=np.float64) for row in written}
    stream_world_error(rows, poses, "table_camera", "bottle", "gt_bottle")
    stream_world_error(rows, poses, "table_camera", "box", "gt_box")
    stream_world_error(rows, poses, "hand_camera", "bottle", "gt_bottle")
    nonempty_times = [item for item in all_times if len(item)]
    if nonempty_times:
        joined = np.concatenate(nonempty_times)
        summarize_times("bottle_all_later_frames", joined)


def follow_overview(estimator, runtime, dino, mesh_ids, folder, mesh_name):
    """One overview camera, one object. Jobs are already in frame order.

    一路总览相机，一个物体。任务列表已经按帧排好。
    """
    pose_path = os.path.join(folder, "jobs_poses.json")
    full_path = backup_pose_file(pose_path)
    init_poses = load_pose_map(full_path)
    with open(os.path.join(folder, "jobs.json"), "r", encoding="utf-8") as stream:
        jobs = json.load(stream)
    with open(os.path.join(folder, "rows.json"), "r", encoding="utf-8") as stream:
        rows = json.load(stream)
    mesh_path = os.path.join(folder, "mesh.ply")
    mesh, center, vertices = prepare_one_mesh(mesh_path)
    mesh = estimator.prepare_meshes([mesh])[0]
    diameter_m = known.mesh_diameter_m(mesh)
    if mesh_name not in mesh_ids:
        mesh_ids[mesh_name] = runtime.load_mesh_trimesh(mesh, name=mesh_name)
    init = init_poses[jobs[0]["id"]]
    print("STREAM", mesh_name, "frames", len(jobs), flush=True)
    written, times_ms = track_stream(
        estimator, runtime, dino, jobs, init, mesh, center, diameter_m, vertices, mesh_ids[mesh_name],
    )
    summarize_times(mesh_name, times_ms)
    write_pose_list(pose_path, written)
    poses = {row["id"]: np.asarray(row["pose"], dtype=np.float64) for row in written}
    overview_error(rows, poses)


if __name__ == "__main__":

    # 1. Load networks once. Initialize each stream with the verified predictions from 13–15.
    # 1. 网络仅载入一次，各序列使用 13–15 已核验的预测结果初始化。
    overview_only = len(sys.argv) >= 2 and sys.argv[1] == "overview"
    # Saved simulation outputs are prerequisites, not downloadable predictions.
    # 仿真输出是真实前置结果，不能用下载的预测替代；加载网络前先检查。
    prerequisite_files = []
    for folder in (os.path.join(RELEASE_DIR, "outputs", "sim_bridge_tasks", "carrot"),
                   os.path.join(RELEASE_DIR, "outputs", "sim_bridge_tasks", "eggplant"),
                   os.path.join(RELEASE_DIR, "outputs", "sim_xarm_cube", "cube")):
        prerequisite_files.extend(os.path.join(folder, name) for name in ("rows.json", "jobs.json", "jobs_poses.json", "mesh.ply"))
    if not overview_only:
        tracking_folder = os.path.join(RELEASE_DIR, "outputs", "sim_known_mesh_place", "tracking")
        prerequisite_files.extend(os.path.join(tracking_folder, name) for name in ("rows.json", "sequence_poses.json", "meshes/bottle.ply", "meshes/box.ply"))
    missing_inputs = [path for path in prerequisite_files if not os.path.isfile(path)]
    if missing_inputs:
        raise FileNotFoundError("Run examples 13–15 to produce saved inputs / 请先运行示例 13–15 生成前置结果:\n" + "\n".join(missing_inputs))
    estimator = WAPREstimator(device=DEVICE)
    # The pose crops stay on the OpenGL runtime inside the estimator.
    # The tracked region is a full-frame depth, the same CUDA raster path the
    # YCBInEOAT tracker uses. That rasterizer is not a second OpenGL context.
    # 位姿裁切仍走估计器里的 OpenGL。跟踪区域要的是整幅深度，和 YCBInEOAT
    # 跟踪使用 YCBInEOAT 跟踪器的 CUDA 光栅化路径，不创建第二个 OpenGL 上下文。
    from wapr.raster_fallback import NvRuntime

    runtime = NvRuntime(0)
    dino = load_dino()
    mesh_ids = {}
    if not overview_only:
        track_dir = os.path.join(RELEASE_DIR, "outputs", "sim_known_mesh_place", "tracking")
        pose_path = os.path.join(track_dir, "sequence_poses.json")
        full_path = backup_pose_file(pose_path)
        init_poses = load_pose_map(full_path)
        with open(os.path.join(track_dir, "rows.json"), "r", encoding="utf-8") as stream:
            rows = json.load(stream)
        streams = {}
        for row in rows:
            for camera_name, camera in row["cameras"].items():
                streams.setdefault(camera_name, []).append(camera["jobs"])
        meshes = {}
        for name in ("bottle", "box"):
            mesh_path = os.path.join(track_dir, "meshes", name + ".ply")
            mesh, center, vertices = prepare_one_mesh(mesh_path)
            mesh = estimator.prepare_meshes([mesh])[0]
            meshes[name] = (mesh, center, vertices, known.mesh_diameter_m(mesh))
            mesh_ids[name] = runtime.load_mesh_trimesh(mesh, name=name)
        written, all_times = [], []
        # 2. Each camera has independent states; all tracks on its current image update together.
        # 2. 每路相机独立保留状态；该相机当前图像上的所有轨迹一起更新。
        for camera_name, frames in streams.items():
            states, times_ms, warmed_shapes = {}, [], set()
            for jobs in frames:
                if not jobs:
                    continue
                rgb, depth_m, K = read_rgb_depth(jobs[0])
                height, width = depth_m.shape
                persistent_jobs = [job for job in jobs if job["name"] in states]
                warm_shape = (height, width, tuple(job["name"] for job in persistent_jobs))
                if persistent_jobs and warm_shape not in warmed_shapes:
                    warm_tracking_path(estimator, runtime, dino, persistent_jobs, states, meshes, mesh_ids)
                    warmed_shapes.add(warm_shape)
                torch.cuda.synchronize()
                started = time.perf_counter()
                # One image encoding shared by bottle, box, and every other track of this frame.
                # 图像只编码一次，供瓶子、盒子及该帧其他轨迹共同使用。
                tokens, grid = dino_tokens(dino, rgb)
                new_jobs = [job for job in jobs if job["name"] not in states]
                if new_jobs:
                    poses = np.stack([init_poses[job["id"]] for job in new_jobs]).astype(np.float32)
                    centers = np.stack([meshes[job["name"]][1] for job in new_jobs])
                    poses[:, :3, 3] += (poses[:, :3, :3] @ centers[..., None]).squeeze(-1)
                    boxes = np.tile([0, 0, width, height], (len(new_jobs), 1))
                    ids = [mesh_ids[job["name"]] for job in new_jobs]
                    _rgb, depths = runtime._render_mixed_tiles(ids, poses, boxes, K, height, width, (height, width))
                    for job, rendered in zip(new_jobs, depths.detach().cpu().numpy()):
                        region, source = visible_region(rendered, depth_m)
                        if int(region.sum()) < min_region_px:
                            mask = cv2.imread(job["mask"], cv2.IMREAD_GRAYSCALE)
                            if mask is None:
                                raise ValueError("Missing recorded prediction mask: " + job["mask"])
                            region, source = mask > 0, "recorded_prediction"
                        patch = one_instance_mask_to_patches(region, grid)
                        if not np.any(patch):
                            raise ValueError("Empty initial tracked region: " + job["id"])
                        states[job["name"]] = {"pose": np.asarray(init_poses[job["id"]]).copy(),
                            "template_tokens": tokens, "template_patch": patch, "prev_tokens": tokens,
                            "prev_patch": patch, "gate": min(int(tracker.min_matches), int(patch.sum()))}
                        written.append({"id": job["id"], "pose": states[job["name"]]["pose"].tolist(),
                                        "choice": "init", "region": source})
                initial_ids = {job["id"] for job in new_jobs}
                update_jobs = [job for job in jobs if job["id"] not in initial_ids]
                instances, propagated_regions, matches = [], [], []
                # This loop prepares states only; it never runs render/refine/score per object.
                # 本循环只整理各物体状态，不逐物体调用渲染、修正或评分。
                for job in update_jobs:
                    state = states[job["name"]]
                    mesh, _center, vertices, diameter_m = meshes[job["name"]]
                    pose = state["pose"]
                    diameter_px = diameter_m * float(K[0, 0]) / max(float(pose[2, 3]), 0.05)
                    propagated = propagate_one_instance_region(state["prev_tokens"], state["prev_patch"],
                        state["template_tokens"], state["template_patch"], tokens, grid, diameter_px, state["gate"])
                    center_pose = pose.copy()
                    if propagated is None:
                        flat = np.flatnonzero(state["prev_patch"])
                        region = patches_to_mask(flat // grid["gw"], flat % grid["gw"], grid)
                        nmatch = 0
                    else:
                        region, center_uv, nmatch = propagated
                        translation = translation_from_one_instance(region, center_uv, depth_m, K, pose[:3, :3], vertices)
                        if translation is not None:
                            center_pose[:3, 3] = translation
                    instances.append({"track_id": job["name"], "mesh": mesh, "diameter_m": diameter_m,
                                      "pose_4x4": pose, "center_pose_4x4": center_pose})
                    propagated_regions.append(region)
                    matches.append(nmatch)
                # 3. One WAPR/SAPR/WBPS update for all current tracks, with independent scoring groups.
                # 3. 当前所有轨迹批量执行一次 WAPR/SAPR/WBPS 更新，各轨迹保留独立评分组。
                outputs = track_many_categories_many_instances(estimator, rgb, depth_m, K, instances)
                if update_jobs:
                    poses = np.stack([out["pose_4x4"] for out in outputs]).astype(np.float32)
                    centers = np.stack([meshes[job["name"]][1] for job in update_jobs])
                    poses[:, :3, 3] += (poses[:, :3, :3] @ centers[..., None]).squeeze(-1)
                    boxes = np.tile([0, 0, width, height], (len(update_jobs), 1))
                    ids = [mesh_ids[job["name"]] for job in update_jobs]
                    _rgb, depths = runtime._render_mixed_tiles(ids, poses, boxes, K, height, width, (height, width))
                    for job, out, rendered, region, nmatch in zip(update_jobs, outputs,
                            depths.detach().cpu().numpy(), propagated_regions, matches):
                        state = states[job["name"]]
                        state["pose"] = np.asarray(out["pose_4x4"], dtype=np.float64)
                        refreshed, _source = visible_region(rendered, depth_m)
                        stored = refreshed if (np.any(rendered > 1e-3) and
                            int(agreement_mask(rendered, depth_m).sum()) >= min_region_px) else region
                        patch = one_instance_mask_to_patches(stored, grid)
                        state["prev_patch"] = patch if np.any(patch) else one_instance_mask_to_patches(region, grid)
                        state["prev_tokens"] = tokens
                        written.append({"id": job["id"], "pose": state["pose"].tolist(),
                                        "choice": HYPOTHESES[out["hypothesis_index"]][0], "matches": int(nmatch)})
                    torch.cuda.synchronize()
                    times_ms.append((time.perf_counter() - started) * 1000.0)
            times_ms = np.asarray(times_ms, dtype=np.float64)
            all_times.append(times_ms)
            summarize_times(camera_name, times_ms)
        write_pose_list(pose_path, written)
        poses = {row["id"]: np.asarray(row["pose"], dtype=np.float64) for row in written}
        stream_world_error(rows, poses, "table_camera", "bottle", "gt_bottle")
        stream_world_error(rows, poses, "table_camera", "box", "gt_box")
        stream_world_error(rows, poses, "hand_camera", "bottle", "gt_bottle")
        nonempty_times = [item for item in all_times if len(item)]
        if nonempty_times:
            summarize_times("bottle_all_later_frames", np.concatenate(nonempty_times))

    # 4. The remaining recordings contain one object per camera; retain their original region policy.
    # 4. 其余录像每路相机仅有一个物体，保留其原有区域传播策略。
    for folder, mesh_name in (
        (os.path.join(RELEASE_DIR, "outputs", "sim_bridge_tasks", "carrot"), "carrot"),
        (os.path.join(RELEASE_DIR, "outputs", "sim_bridge_tasks", "eggplant"), "eggplant"),
        (os.path.join(RELEASE_DIR, "outputs", "sim_xarm_cube", "cube"), "xarm_cube"),
    ):
        pose_path = os.path.join(folder, "jobs_poses.json")
        full_path = backup_pose_file(pose_path)
        init_poses = load_pose_map(full_path)
        with open(os.path.join(folder, "jobs.json"), "r", encoding="utf-8") as stream:
            jobs = json.load(stream)
        with open(os.path.join(folder, "rows.json"), "r", encoding="utf-8") as stream:
            rows = json.load(stream)
        mesh, center, vertices = prepare_one_mesh(os.path.join(folder, "mesh.ply"))
        mesh = estimator.prepare_meshes([mesh])[0]
        diameter_m = known.mesh_diameter_m(mesh)
        mesh_id = runtime.load_mesh_trimesh(mesh, name=mesh_name)
        pose = np.asarray(init_poses[jobs[0]["id"]], dtype=np.float64).reshape(4, 4).copy()
        rgb, depth_m, camera_k = read_rgb_depth(jobs[0])
        height, width = depth_m.shape[:2]
        _rgb, rendered_depth = render_pose_rgbd(runtime, mesh_id, centered_poses(pose[None], center),
                                               camera_k, height, width)
        region, region_source = visible_region(rendered_depth[0], depth_m)
        if int(region.sum()) < min_region_px:
            mask = cv2.imread(jobs[0]["mask"], cv2.IMREAD_GRAYSCALE)
            region, region_source = np.asarray(mask) > 0, "mask"
        tokens, grid = dino_tokens(dino, rgb)
        patch = one_instance_mask_to_patches(region, grid)
        if not np.any(patch):
            raise RuntimeError("empty region %s" % jobs[0]["id"])
        gate = min(int(tracker.min_matches), int(patch.sum()))
        template_tokens, template_patch = tokens, patch
        prev_tokens, prev_patch = tokens, patch
        if len(jobs) > 1:
            warm_job = dict(jobs[1], name="object")
            warm_state = {"pose": pose, "prev_tokens": tokens, "prev_patch": patch,
                          "template_tokens": tokens, "template_patch": patch, "gate": gate}
            warm_tracking_path(estimator, runtime, dino, [warm_job], {"object": warm_state},
                               {"object": (mesh, center, vertices, diameter_m)}, {"object": mesh_id})
        written = [{"id": jobs[0]["id"], "pose": pose.tolist(), "choice": "init", "region": region_source}]
        times_ms = []
        # Per-frame loop: propagate region, seed translation from observed depth, then refine pose.
        # 逐帧循环：传播区域，用观测深度初始化平移，再修正位姿。
        for step, job in enumerate(jobs[1:], start=1):
            rgb, depth_m, camera_k = read_rgb_depth(job)
            torch.cuda.synchronize()
            started = time.perf_counter()
            tokens, grid = dino_tokens(dino, rgb)
            diameter_px = diameter_m * float(camera_k[0, 0]) / max(float(pose[2, 3]), 0.05)
            propagated = propagate_one_instance_region(prev_tokens, prev_patch, template_tokens,
                                                        template_patch, tokens, grid, diameter_px, gate)
            translation = None
            if propagated is None:
                flat = np.flatnonzero(prev_patch)
                region = patches_to_mask(flat // grid["gw"], flat % grid["gw"], grid)
                nmatch = 0
            else:
                region, center_uv, nmatch = propagated
                translation = translation_from_one_instance(region, center_uv, depth_m,
                                                             camera_k, pose[:3, :3], vertices)
            center_pose = pose.copy()
            if translation is not None:
                center_pose[:3, 3] = translation
            # A single object is still a batch; its six hypotheses render/refine/score together.
            # 单物体也以批处理调用，其六个候选姿态共同渲染、修正与评分。
            result = track_many_categories_many_instances(estimator, rgb, depth_m, camera_k,
                [{"track_id": job.get("name", "object"), "mesh": mesh, "diameter_m": diameter_m,
                  "pose_4x4": pose, "center_pose_4x4": center_pose}])[0]
            pose = np.asarray(result["pose_4x4"], dtype=np.float64).copy()
            _rgb, posed_depth = render_pose_rgbd(runtime, mesh_id, centered_poses(pose[None], center),
                                                 camera_k, height, width)
            refreshed, _source = visible_region(posed_depth[0], depth_m)
            stored = refreshed if (int((posed_depth[0] > 1e-3).sum()) > 0 and
                int(agreement_mask(posed_depth[0], depth_m).sum()) >= min_region_px) else region
            prev_tokens = tokens
            prev_patch = one_instance_mask_to_patches(stored, grid)
            if not np.any(prev_patch):
                prev_patch = one_instance_mask_to_patches(region, grid)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - started) * 1000.0)
            written.append({"id": job["id"], "pose": pose.tolist(),
                            "choice": HYPOTHESES[result["hypothesis_index"]][0], "matches": int(nmatch)})
        summarize_times(mesh_name, np.asarray(times_ms, dtype=np.float64))
        write_pose_list(pose_path, written)
        poses = {row["id"]: np.asarray(row["pose"], dtype=np.float64) for row in written}
        # Annotated world poses are read only by this final error report.
        # 标注世界位姿仅由最终误差报告读取。
        overview_error(rows, poses)
    print("FOLLOW_DONE", flush=True)
