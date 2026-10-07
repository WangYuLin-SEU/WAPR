# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 09: TACO multi-object tracking. Previous: 08; next: 10 ROBI.
# 示例 09：TACO 多物体跟踪。上一例为 08，下一例为 10 ROBI。
# Run: python examples/09_taco_many_instances.py
# 运行：python examples/09_taco_many_instances.py
# Prepare the external sequence with examples/09_taco_many_instances/prepare_sample.py.
# 用 examples/09_taco_many_instances/prepare_sample.py 准备外部序列。
# Saved clicks predict first-frame masks; motion-capture poses are never read.
# 保存的点击点用于预测首帧掩码；不读取动捕位姿。
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import trimesh

script_dir = os.path.dirname(os.path.abspath(__file__))
release_dir = os.path.dirname(script_dir)
# Prefer this source checkout's public package. / 优先使用当前源码中的公开包。
sys.path.insert(0, release_dir)
from wapr import WAPREstimator, recipe
from wapr.download_assets import check_and_fetch_pack
from wapr.resources import samples_dir
from wapr.det2d import default_weights_dir
from ultralytics import SAM
from wapr.estimator import center_from_mesh
from wapr.region_tracking import (
    load_tracker_dino, dino_tokens, one_instance_mask_to_patches,
    propagate_one_instance_region, translation_from_one_instance,
)
from wapr.tracking import track_many_categories_many_instances
from wapr.view import save_pose_view


# Prepared excerpt root; all inputs derive from this path. / 小样根目录；输入由此派生。
data_root = os.path.join(samples_dir(), "taco")
output_dir = os.path.join(release_dir, "outputs", "taco_many_instances")
device = "cuda:0"
# Raw frame indices, end exclusive. The selected excerpt starts at raw frame 80.
# 原始帧号，终点不包含；选定摘录从原始第 80 帧开始，不将其写成原始第 0 帧。
first_frame, stop_frame, frame_stride = 80, 110, 1
objects = (("box", 93, "093_cm.obj"), ("roller", 76, "076_cm.obj"))
mesh_unit_m = 0.01  # TACO OBJ vertices are centimeters. / TACO OBJ 顶点单位为厘米。
depth_unit_m = 1.0 / 4000.0  # Stored uint16 depth. / 存储的 uint16 深度。
# Explicit experimental quality gate, not a library default or ADD criterion.
# 显式实验质量阈值，不是库默认值，也不是 ADD 达标判据。
recover_score_6d = 0.5
dino_name, dino_short_side, dino_patch = "vits14", 448, 14
patch_cover, cosine_min, min_matches = 0.20, 0.50, 8
min_region_px, front_percentile = 150, 30.0
warmup_calls = 3
# Untextured meshes: grayscale observation for pose and DINO, color for SAM/display.
# 无纹理网格：位姿及 DINO 使用灰度观测；SAM 和展示使用彩色图。
grayscale_pose = True
save_previews = True


def sam_masks(segmentor, rgb, points=None, boxes=None):
    """Predict all supplied object prompts with one shared image encoding.

    一次图像编码预测所有物体提示；返回原图尺寸的 bool 掩码及 SAM 分数。
    points is (objects, 1, 2); boxes is (objects, 4), xyxy in pixels.
    points 为 (物体数, 1, 2)，boxes 为 (物体数, 4) 像素 xyxy。
    """
    result = segmentor.predict(
        source=np.ascontiguousarray(rgb[..., ::-1]), points=points, bboxes=boxes,
        device=device, verbose=False, save=False, retina_masks=True,
    )[0]
    expected = len(points) if points is not None else len(boxes)
    if result.masks is None or len(result.masks.data) != expected:
        raise RuntimeError("SAM must preserve prompt order / SAM 必须保持提示顺序")
    masks = result.masks.data.detach().cpu().numpy() > 0
    if masks.shape[1:] != rgb.shape[:2]:
        raise ValueError("SAM mask size differs / SAM 掩码尺寸不符")
    return masks, result.boxes.conf.detach().cpu().numpy()


def rank_recovery_candidates(estimator, rgb, depth_m, K, tracks, candidates):
    """Rank complete independent recovery groups together on the GPU.

    GPU 批量评分独立的完整补救候选组；组内 top1，返回被选候选自身的分数。
    candidates maps track_id to distinct attempted poses; no singleton padding.
    candidates 将 track_id 映射到实际尝试位姿；不补齐单候选评分组。
    """
    selected = {}
    capacity = min(int(estimator.nets["wbps"].engine.max_batch), 120)
    # Different group lengths are separate batches, never padded attention groups.
    # 不同组长分别批量计算，不补齐注意力组。
    for group in sorted({len(rows) for rows in candidates.values()}):
        if group < 2:
            raise ValueError("Recovery WBPS group must contain at least two poses")
        track_ids = [key for key, rows in candidates.items() if len(rows) == group]
        objects_per_chunk = capacity // group
        for start in range(0, len(track_ids), objects_per_chunk):
            keys = track_ids[start:start + objects_per_chunk]
            poses = np.stack([row["pose_4x4"] for key in keys for row in candidates[key]])
            mesh_rows = [tracks[key]["mesh"] for key in keys for _ in range(group)]
            centers = np.stack([center_from_mesh(mesh) for mesh in mesh_rows]).astype(np.float32)
            diameters = np.asarray([tracks[key]["diameter_m"] for key in keys for _ in range(group)])
            within, between = estimator._wbps(
                torch.as_tensor(poses, device=device, dtype=torch.float32),
                rgb, depth_m, np.zeros(depth_m.shape, np.uint8), K, mesh_rows[0],
                centers, diameters, group, meshes=mesh_rows,
                mesh_ids=[estimator.renderer.cached_mesh_id(mesh) for mesh in mesh_rows],
            )
            assert tuple(within.shape) == tuple(between.shape) == (len(keys), group)
            best = within.argmax(dim=1)
            errors = between.gather(1, best[:, None]).flatten().detach().cpu().numpy()
            for key, choice, error in zip(keys, best.detach().cpu().numpy(), errors):
                row = dict(candidates[key][int(choice)])
                row["score_6d"] = (100.0 - float(error)) / 200.0
                row["between_group"] = float(error)
                selected[key] = row
    return selected


def mesh_counters(estimator):
    """Snapshot cold-resource counters. / 读取冷资源操作计数。"""
    return (estimator.mesh_prepare_count, estimator.renderer.mesh_pack_count,
            estimator.renderer.mesh_upload_count)


if __name__ == "__main__":
    check_and_fetch_pack("taco")
    # Input preparation and decoding precede every GPU timing interval.
    # 输入准备和解码先于所有 GPU 计时区间。
    rgb_path = os.path.join(data_root, "clip", "color.mp4")
    depth_path = os.path.join(data_root, "clip", "depth_u16_scale4000_512x376.npy")
    camera_path = os.path.join(data_root, "seq_cam", "egocentric_intrinsic.txt")
    clicks_path = os.path.join(script_dir, "09_taco_init_clicks.json")
    for path in (rgb_path, depth_path, camera_path, clicks_path):
        if not os.path.isfile(path):
            raise FileNotFoundError("Prepare TACO inputs first / 请先准备 TACO 输入: " + path)
    with open(clicks_path, encoding="utf-8") as stream:
        clicks = json.load(stream)
    if clicks["frame"] != first_frame:
        raise ValueError("Saved clicks belong to another frame / 点击点不属于当前首帧")
    if frame_stride < 1 or stop_frame <= first_frame:
        raise ValueError("Invalid frame interval / 帧区间无效")
    depth_u16 = np.load(depth_path, mmap_mode="r")
    if depth_u16.dtype != np.uint16 or depth_u16.ndim != 3 or stop_frame > len(depth_u16):
        raise ValueError("Expected uint16 (frames,H,W) depth / 需要 uint16 深度序列")
    K = np.loadtxt(camera_path).reshape(3, 3).astype(np.float32)
    if not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Invalid camera intrinsics / 相机内参无效")
    capture = cv2.VideoCapture(rgb_path)
    rgb_frames, depths_m, frame_ids = [], [], list(range(first_frame, stop_frame, frame_stride))
    if len(frame_ids) < 2:
        raise ValueError("Tracking needs at least two frames / 跟踪至少需要两帧")
    # Decode in order to retain exact frame indices. / 顺序解码，保持原始帧号。
    for frame_id in range(stop_frame):
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError("Missing RGB frame / 缺少 RGB 帧: %d" % frame_id)
        if frame_id in frame_ids:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != depth_u16.shape[1:] or [rgb.shape[1], rgb.shape[0]] != clicks["rgb_size_wh"]:
                raise ValueError("RGB, depth or click grid differs / RGB、深度或点击像素网格不符")
            rgb_frames.append(rgb)
            depths_m.append(depth_u16[frame_id].astype(np.float32) * depth_unit_m)
    capture.release()
    pose_frames = [np.repeat(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
                   if grayscale_pose else rgb for rgb in rgb_frames]
    raw_meshes, vertices_m, diameters_m = [], {}, {}
    for name, obj_id, mesh_file in objects:
        mesh_path = os.path.join(data_root, "meshes", mesh_file)
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * mesh_unit_m
        vertices_m[name] = np.asarray(mesh.vertices).copy()
        diameters_m[name] = float(np.linalg.norm(np.ptp(vertices_m[name], axis=0)))
        raw_meshes.append(mesh)
    point_prompts = np.asarray([[clicks["clicks"][name]] for name, _, _ in objects], np.int32)
    if np.any(point_prompts < 0) or np.any(point_prompts[..., 0] >= rgb_frames[0].shape[1]) or np.any(point_prompts[..., 1] >= rgb_frames[0].shape[0]):
        raise ValueError("Saved click outside RGB / 保存点击超出图像")
    os.makedirs(output_dir, exist_ok=True)
    torch.cuda.synchronize(device)
    setup_started = time.perf_counter()
    estimator = WAPREstimator(device=device)
    if any(net.engine is None for net in estimator.nets.values()):
        raise RuntimeError("This example requires TRT FP16 / 本例需要 TRT FP16 引擎")
    if not os.path.isfile(os.path.join(os.path.dirname(os.fspath(recipe.engine_file("wbps"))), "wbps_batch.engine")):
        raise FileNotFoundError("Run python -m wapr.export_batch_engine on this GPU")
    prepared = estimator.prepare_meshes(raw_meshes)
    dino = load_tracker_dino(device, dino_name=dino_name)
    sam_path = os.path.join(default_weights_dir, "sam2.1_l.pt")
    if not os.path.isfile(sam_path):
        raise FileNotFoundError("Prepare SAM 2 weights as in the detector install guide: " + sam_path)
    segmentor = SAM(sam_path)
    # Model setup occurs before the discarded full SAM warmups.
    # 模型构建在 SAM 预热前完成，所有预热结果丢弃。
    segmentor.predictor = segmentor._smart_load("predictor")(
        overrides={"model": sam_path, "device": device, "verbose": False, "task": "segment", "mode": "predict"})
    segmentor.predictor.setup_model(model=segmentor.model, verbose=False)
    torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - setup_started
    warm_started = time.perf_counter()
    for _ in range(warmup_calls):
        sam_masks(segmentor, rgb_frames[0], points=point_prompts)
    torch.cuda.synchronize(device)
    sam_warmup_seconds = time.perf_counter() - warm_started
    mask_started = time.perf_counter()
    masks0, sam_scores = sam_masks(segmentor, rgb_frames[0], points=point_prompts)
    torch.cuda.synchronize(device)
    init_mask_seconds = time.perf_counter() - mask_started
    for mask in masks0:
        if int(mask.sum()) < min_region_px or int(np.count_nonzero(depths_m[0][mask] > 0)) < min_region_px:
            raise RuntimeError("Initial predicted region lacks depth / 首帧预测区域缺少有效深度")
    init_instances = [{"mesh": mesh, "diameter_m": diameters_m[name], "mask": mask}
                      for (name, _, _), mesh, mask in zip(objects, prepared, masks0)]
    pose_warmup_seconds = estimator.warmup_pose(
        pose_frames[0], depths_m[0], K, init_instances, calls=warmup_calls)
    cold_before = mesh_counters(estimator)
    init_started = time.perf_counter()
    initial = estimator.estimate_many_categories_many_instances(pose_frames[0], depths_m[0], K, init_instances)
    torch.cuda.synchronize(device)
    init_pose_seconds = time.perf_counter() - init_started
    assert mesh_counters(estimator) == cold_before, "Cold mesh work inside initialization timing"
    # Initial feature seeding may be cold; record it as setup, never as hot inference.
    # 初始特征建库可能包含冷启动，归入准备阶段，不作为预热后推理计时。
    seed_started = time.perf_counter()
    tokens0, grid0 = dino_tokens(dino, pose_frames[0], device, dino_short_side=dino_short_side, dino_patch=dino_patch)
    torch.cuda.synchronize(device)
    dino_seed_setup_seconds = time.perf_counter() - seed_started
    tracks = {}
    for (name, obj_id, _), mesh, mask, result in zip(objects, prepared, masks0, initial):
        patch = one_instance_mask_to_patches(mask, grid0, patch_cover=patch_cover, dino_patch=dino_patch)
        if int(patch.sum()) < min_matches:
            raise RuntimeError("Initial region has too few DINO patches / 首帧区域图块不足: " + name)
        tracks[name] = {"track_id": name, "obj_id": obj_id, "mesh": mesh,
                        "diameter_m": diameters_m[name], "pose_4x4": result["pose_4x4"].copy(),
                        "prev_tokens": tokens0, "prev_patch": patch,
                        "template_tokens": tokens0, "template_patch": patch}
        cv2.imwrite(os.path.join(output_dir, name + "_init_mask.png"), mask.astype(np.uint8) * 255)
    # Warm every possible active/recovery batch size and two/three-pose scoring group.
    # 预热可能的活跃及补救批量大小，以及两、三候选评分组；不更新真实轨迹状态。
    torch.cuda.synchronize(device)
    warm_started = time.perf_counter()
    for _ in range(warmup_calls):
        curr_tokens, curr_grid = dino_tokens(dino, pose_frames[1], device, dino_short_side=dino_short_side, dino_patch=dino_patch)
        for name, _, _ in objects:
            track = tracks[name]
            diameter_px = track["diameter_m"] * float(K[0, 0]) / max(float(track["pose_4x4"][2, 3]), 0.05)
            propagate_one_instance_region(tokens0, track["prev_patch"], tokens0, track["template_patch"],
                                          curr_tokens, curr_grid, diameter_px, cosine_min=cosine_min,
                                          min_matches=min_matches, dino_patch=dino_patch)
        for count in range(1, len(objects) + 1):
            track_many_categories_many_instances(estimator, pose_frames[0], depths_m[0], K, list(tracks.values())[:count])
        for count in range(1, 2 * len(objects) + 1):
            prompt_rows = [init_instances[index % len(objects)] for index in range(count)]
            boxes = []
            for row in prompt_rows:
                ys, xs = np.nonzero(row["mask"])
                boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
            sam_masks(segmentor, rgb_frames[0], boxes=boxes)
            estimator.estimate_many_categories_many_instances(pose_frames[0], depths_m[0], K, prompt_rows)
        for group in (2, 3):
            for count in range(1, len(objects) + 1):
                candidates = {key: [{"pose_4x4": initial[index]["pose_4x4"]} for _ in range(group)]
                              for index, key in enumerate(list(tracks)[:count])}
                # Warm buffers only; identical test rows do not become real recovery candidates.
                # 只预热缓冲；相同测试行不作为真实补救候选，也不参与结果统计。
                rank_recovery_candidates(estimator, pose_frames[0], depths_m[0], K, tracks, candidates)
    torch.cuda.synchronize(device)
    tracking_warmup_seconds = time.perf_counter() - warm_started
    records = [{"frame": first_frame, "region_seconds": 0.0, "normal_seconds": init_pose_seconds,
                "reassociation_seconds": 0.0, "init_mask_seconds": init_mask_seconds,
                "tracks": [{"track_id": name, "obj_id": obj_id, "status": "INIT", "pose_updated": True,
                             "score_6d": float(result["score_6d"]), "pose_4x4": result["pose_4x4"].tolist()}
                            for (name, obj_id, _), result in zip(objects, initial)]}]
    print("TACO_SETUP", {"gpu": torch.cuda.get_device_name(device), "backend": "TensorRT FP16 + OGL",
                         "setup_seconds": setup_seconds, "sam_warmup_seconds": sam_warmup_seconds,
                         "pose_warmup_seconds": pose_warmup_seconds, "tracking_warmup_seconds": tracking_warmup_seconds,
                         "warmup_calls": warmup_calls, "tracks": list(tracks), "frames": [first_frame, stop_frame],
                         "stride": frame_stride, "threshold": recover_score_6d}, flush=True)
    for frame_id, rgb, rgb_pose, depth_m in zip(frame_ids[1:], rgb_frames[1:], pose_frames[1:], depths_m[1:]):
        torch.cuda.synchronize(device)
        cold_before = mesh_counters(estimator)
        frame_started = time.perf_counter()
        tokens, grid = dino_tokens(dino, rgb_pose, device, dino_short_side=dino_short_side, dino_patch=dino_patch)
        regions, active = {}, []
        for name, _, _ in objects:
            track = tracks[name]
            diameter_px = track["diameter_m"] * float(K[0, 0]) / max(float(track["pose_4x4"][2, 3]), 0.05)
            match = propagate_one_instance_region(
                track["prev_tokens"], track["prev_patch"], track["template_tokens"], track["template_patch"],
                tokens, grid, diameter_px, cosine_min=cosine_min, min_matches=min_matches, dino_patch=dino_patch)
            if match is None:
                continue
            region, uv, matches = match
            translation = translation_from_one_instance(region, uv, depth_m, K, track["pose_4x4"][:3, :3],
                vertices_m[name], min_region_px=min_region_px, front_percentile=front_percentile)
            if translation is None:
                continue
            regions[name] = region
            center_pose = track["pose_4x4"].copy()
            center_pose[:3, 3] = translation
            active.append(dict(track, center_pose_4x4=center_pose))
        torch.cuda.synchronize(device)
        region_seconds = time.perf_counter() - frame_started
        normal_started = time.perf_counter()
        updates = track_many_categories_many_instances(estimator, rgb_pose, depth_m, K, active)
        torch.cuda.synchronize(device)
        normal_seconds = time.perf_counter() - normal_started
        selected = {row["track_id"]: dict(row, status="TRACK") for row in updates}
        failed = [name for name, _, _ in objects
                  if name not in selected or selected[name]["score_6d"] < recover_score_6d]
        reassociation_seconds = 0.0
        if failed:
            rescue_started = time.perf_counter()
            recovery_rows, boxes, owners = [], [], []
            # Search previous and original templates separately across the entire image.
            # 将上一可靠帧及首帧模板分别在全图匹配，尝试其他候选区域。
            for name in failed:
                track = tracks[name]
                diameter_px = track["diameter_m"] * float(K[0, 0]) / max(float(track["pose_4x4"][2, 3]), 0.05)
                for source_tokens, source_patch in ((track["prev_tokens"], track["prev_patch"]),
                                                    (track["template_tokens"], track["template_patch"])):
                    match = propagate_one_instance_region(source_tokens, source_patch, source_tokens, source_patch,
                        tokens, grid, diameter_px, cosine_min=cosine_min, min_matches=min_matches, dino_patch=dino_patch)
                    if match is None:
                        continue
                    region = match[0]
                    ys, xs = np.nonzero(region)
                    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                    if any(owner == name and previous == box for owner, previous in zip(owners, boxes)):
                        continue
                    owners.append(name)
                    boxes.append(box)
            candidates = {}
            if boxes:
                masks, _sam_scores = sam_masks(segmentor, rgb, boxes=boxes)
                for name, mask in zip(owners, masks):
                    if int(mask.sum()) < min_region_px or int(np.count_nonzero(depth_m[mask] > 0)) < min_region_px:
                        continue
                    recovery_rows.append({"mesh": tracks[name]["mesh"], "diameter_m": tracks[name]["diameter_m"],
                                          "mask": mask, "track_id": name})
                rescued = estimator.estimate_many_categories_many_instances(rgb_pose, depth_m, K, recovery_rows)
                for row, result in zip(recovery_rows, rescued):
                    name = row["track_id"]
                    if name not in candidates:
                        baseline = selected.get(name, {"pose_4x4": tracks[name]["pose_4x4"], "status": "HELD"})
                        candidates[name] = [dict(baseline)]
                    candidates[name].append(dict(result, region=row["mask"], status="REASSOC"))
                ranked = rank_recovery_candidates(estimator, rgb_pose, depth_m, K, tracks, candidates)
                selected.update(ranked)
            torch.cuda.synchronize(device)
            reassociation_seconds = time.perf_counter() - rescue_started
        frame_tracks = []
        for name, obj_id, _ in objects:
            row = selected.get(name)
            accepted = row is not None and row["score_6d"] >= recover_score_6d and row["status"] != "HELD"
            status = row["status"] if accepted else "LOST"
            if accepted:
                # Commit only reliable updates; keep the original template for later recovery.
                # 仅写回可靠更新；保留首帧模板供后续重新关联。
                region = row.get("region", regions.get(name))
                patch = one_instance_mask_to_patches(region, grid, patch_cover=patch_cover, dino_patch=dino_patch)
                tracks[name]["pose_4x4"] = np.asarray(row["pose_4x4"], dtype=np.float32).copy()
                if int(patch.sum()) >= min_matches:
                    tracks[name]["prev_tokens"], tracks[name]["prev_patch"] = tokens, patch
            frame_tracks.append({"track_id": name, "obj_id": obj_id, "status": status, "pose_updated": accepted,
                                 "score_6d": float(row["score_6d"]) if row is not None else None,
                                 "reassociation_attempted": name in failed,
                                 "pose_4x4": tracks[name]["pose_4x4"].tolist()})
        torch.cuda.synchronize(device)
        hot_seconds = time.perf_counter() - frame_started
        cold_delta = [after - before for after, before in zip(mesh_counters(estimator), cold_before)]
        if any(cold_delta):
            raise RuntimeError("Cold mesh preparation inside warmed tracking / 预热后跟踪中出现网格冷准备")
        record = {"frame": frame_id, "region_seconds": region_seconds, "normal_seconds": normal_seconds,
                  "reassociation_seconds": reassociation_seconds, "hot_total_seconds": hot_seconds,
                  "mesh_prepare_pack_upload_delta": cold_delta, "tracks": frame_tracks}
        records.append(record)
        print("TACO_FRAME", {"frame": frame_id, "tracks": [(row["track_id"], row["status"]) for row in frame_tracks],
                             "normal_ms": round(normal_seconds * 1000, 1), "reassociation_ms": round(reassociation_seconds * 1000, 1),
                             "hot_total_ms": round(hot_seconds * 1000, 1)}, flush=True)
        if save_previews:
            visible = [{"name": row["track_id"], "mesh": tracks[row["track_id"]]["mesh"],
                        "pose_4x4": row["pose_4x4"], "score_6d": row["score_6d"]}
                       for row in frame_tracks if row["pose_updated"]]
            save_pose_view(rgb, visible, os.path.join(output_dir, "frame_%04d.jpg" % frame_id), K=K)
    report = {"gpu": torch.cuda.get_device_name(device), "backend": "TensorRT FP16 + OGL",
              "sequence": clicks["sequence"], "frame_stride": frame_stride, "grayscale_pose": grayscale_pose,
              "refine_steps": {"wapr": recipe.wapr_iters, "sapr": recipe.sapr_iters},
              "init_group_length": recipe.n_view * recipe.n_inplane, "tracking_group_length": 6,
              "recovery_threshold": recover_score_6d, "warmup_calls": warmup_calls,
              "setup_seconds": setup_seconds, "sam_warmup_seconds": sam_warmup_seconds,
              "dino_seed_setup_seconds": dino_seed_setup_seconds,
              "pose_warmup_seconds": pose_warmup_seconds, "tracking_warmup_seconds": tracking_warmup_seconds,
              "init_mask_seconds": init_mask_seconds, "init_pose_seconds": init_pose_seconds,
              "init_sam_scores": sam_scores.tolist(),
              "timing_scope": "hot_total includes region, pose and recovery; excludes decoding, setup, warmup, drawing and file writes",
              "annotation_inputs_used": False, "source_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
              "clicks": clicks, "frames": records}
    with open(os.path.join(output_dir, "tracking.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("TACO_OUTPUT", os.path.join(output_dir, "tracking.json"), flush=True)
