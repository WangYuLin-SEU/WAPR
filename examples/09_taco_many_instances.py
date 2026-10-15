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
from wapr.bootstrap import ensure_optional
from wapr.download_assets import check_and_fetch_pack
from wapr.resources import samples_dir, outputs_dir
from wapr.det2d import default_weights_dir
from wapr.estimator import center_from_mesh
from wapr.ogl import depth2xyzmap_batch
from wapr.view import save_pose_view


# Prepared excerpt root; all inputs derive from this path. / 小样根目录；输入由此派生。
data_root = os.path.join(samples_dir(), "taco")
output_dir = outputs_dir(__file__)
device = "cuda:0"
# Raw frame indices, end exclusive. The selected excerpt starts at raw frame 80.
# 原始帧号，终点不包含；选定摘录从原始第 80 帧开始，不将其写成原始第 0 帧。
first_frame, stop_frame, frame_stride = 80, 110, 1
objects = (("box", 93, "093_cm.obj"), ("roller", 76, "076_cm.obj"))
mesh_unit_m = 0.01  # TACO OBJ vertices are centimeters. / TACO OBJ 顶点单位为厘米。
depth_unit_m = 1.0 / 4000.0  # Stored uint16 depth. / 存储的 uint16 深度。
min_region_px = 150
warmup_calls = 3
# Untextured meshes: grayscale observation for pose refinement, color for SAM/display.
# 无纹理网格：位姿修正使用灰度观测；SAM 和展示使用彩色图。
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


def refine_tracks(estimator, poses, rgb, depth_m, K, meshes, centers, diameters):
    """Batch WAPR then SAPR updates and return the final poses.

    批量执行 WAPR 与 SAPR 修正，返回最后一轮位姿。
    """
    rgb_t = torch.as_tensor(rgb, device=device, dtype=torch.float32)
    depth_t = torch.as_tensor(depth_m, device=device, dtype=torch.float32)
    camera_t = torch.as_tensor(K, device=device, dtype=torch.float32)[None]
    observed_xyz = depth2xyzmap_batch(depth_t[None], camera_t).permute(0, 3, 1, 2)
    mask = torch.zeros_like(depth_t)
    mesh_ids = [estimator.renderer.cached_mesh_id(mesh) for mesh in meshes]
    poses = torch.as_tensor(poses, device=device, dtype=torch.float32)
    for net_name, steps, rotation_cap in (
        ("wapr_wo_mask", recipe.wapr_iters, recipe.wapr_max_rot_rad),
        ("sapr", recipe.sapr_iters, recipe.sapr_max_rot_rad),
    ):
        poses = estimator._refine(
            estimator.nets[net_name], poses, rgb_t, depth_t, mask, K, meshes[0],
            centers, diameters, iters=steps, max_rot=rotation_cap,
            crop_ratio=recipe.refine_crop_ratio, use_mask=False, meshes=meshes,
            group_size=1, observed_xyz=observed_xyz, mesh_ids=mesh_ids,
        )
    return poses


def mesh_counters(estimator):
    """Snapshot cold-resource counters. / 读取冷资源操作计数。"""
    return (estimator.mesh_prepare_count, estimator.renderer.mesh_pack_count,
            estimator.renderer.mesh_upload_count)


if __name__ == "__main__":
    recipe.visualize_path = output_dir
    # Ultralytics SAM belongs to the optional detection stack, not the base wheel.
    # Ultralytics SAM 属于可选检测环境，不属于基础 wheel 依赖。
    ensure_optional("det2d")
    from ultralytics import SAM

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
    pose_backend = recipe.resolve_backend()
    # TensorRT still requires the FP16 engines. Windows without TensorRT uses the torch path.
    # 有 TensorRT 时仍要求 FP16 引擎。未安装 TensorRT 的 Windows 走 torch 路径。
    if pose_backend == "trt":
        if any(net.engine is None for net in estimator.nets.values()):
            raise RuntimeError("This example requires TRT FP16 / 本例需要 TRT FP16 引擎")
        if not os.path.isfile(os.path.join(os.path.dirname(os.fspath(recipe.engine_file("wbps"))), "wbps_batch.engine")):
            raise FileNotFoundError("Run python -m wapr.export_batch_engine on this GPU")
    backend_label = "TensorRT FP16 + OGL" if pose_backend == "trt" else "PyTorch + OGL"
    prepared = estimator.prepare_meshes(raw_meshes)
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
    centers = np.stack([center_from_mesh(mesh) for mesh in prepared]).astype(np.float32)
    diameters = np.asarray([diameters_m[name] for name, _, _ in objects], dtype=np.float32)
    initial_poses = np.stack([result["pose_4x4"] for result in initial])
    for (name, _, _), mask in zip(objects, masks0):
        cv2.imwrite(os.path.join(output_dir, name + "_init_mask.png"), mask.astype(np.uint8) * 255)
    # Warm the tracking batch without advancing the sequence.
    # 预热跟踪批次，不推进序列状态。
    warm_started = time.perf_counter()
    for _ in range(warmup_calls):
        refine_tracks(estimator, initial_poses, pose_frames[0], depths_m[0], K, prepared, centers, diameters)
    torch.cuda.synchronize(device)
    tracking_warmup_seconds = time.perf_counter() - warm_started
    poses = initial_poses
    records = []
    for index, (frame_id, rgb, rgb_pose, depth_m) in enumerate(zip(frame_ids, rgb_frames, pose_frames, depths_m)):
        cold_before = mesh_counters(estimator)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        if index:
            # The previous frame's final pose initializes this frame.
            # 使用上一帧的最终位姿初始化当前帧。
            poses = refine_tracks(estimator, poses, rgb_pose, depth_m, K, prepared, centers, diameters)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started if index else init_pose_seconds
        assert mesh_counters(estimator) == cold_before, "Unexpected mesh preparation during tracking"
        pose_array = poses.detach().cpu().numpy() if torch.is_tensor(poses) else poses
        frame_tracks = [
            {"track_id": name, "obj_id": obj_id, "status": "INIT" if index == 0 else "TRACK",
             "pose_updated": True, "pose_4x4": pose.tolist()}
            for (name, obj_id, _), pose in zip(objects, pose_array)
        ]
        records.append({"frame": frame_id, "normal_seconds": elapsed, "tracks": frame_tracks})
        if save_previews:
            visible = [dict(row, mesh=mesh) for row, mesh in zip(frame_tracks, prepared)]
            save_pose_view(rgb, visible, os.path.join(output_dir, "frame_%04d.jpg" % frame_id), K=K)
        print("TACO_FRAME", {"frame": frame_id, "objects": len(frame_tracks)}, flush=True)
    report = {
        "gpu": torch.cuda.get_device_name(device), "backend": backend_label,
        "sequence": clicks["sequence"], "frame_stride": frame_stride,
        "grayscale_pose": grayscale_pose,
        "refine_steps": {"wapr": recipe.wapr_iters, "sapr": recipe.sapr_iters},
        "tracking_output": "final_sapr_pose", "warmup_calls": warmup_calls,
        "setup_seconds": setup_seconds, "sam_warmup_seconds": sam_warmup_seconds,
        "pose_warmup_seconds": pose_warmup_seconds, "tracking_warmup_seconds": tracking_warmup_seconds,
        "init_mask_seconds": init_mask_seconds, "init_pose_seconds": init_pose_seconds,
        "timing_scope": "pose updates only; excludes decoding, setup, warmup, drawing and writes",
        "annotation_inputs_used": False, "clicks": clicks,
        "source_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "frames": records,
    }
    with open(os.path.join(output_dir, "tracking.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("TACO_OUTPUT", os.path.join(output_dir, "tracking.json"), flush=True)
