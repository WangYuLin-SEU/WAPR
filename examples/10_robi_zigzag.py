# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 10. Previous: examples/09_taco_many_instances.py. Next: examples/11_reconstruct_object.py
# 示例 10。上一例：examples/09_taco_many_instances.py。下一例：examples/11_reconstruct_object.py
# Example 10. Multiple-instance pose estimation on one ROBI frame: 2D detection, then one pose batch.
# 示例 10。ROBI 单帧上的多实例位姿估计：先做 2D 检测，再批量估计位姿。
# python examples/10_robi_zigzag.py
# 运行：python examples/10_robi_zigzag.py
# Page: docs/pose.html#robi
# 页面：docs/pose.html#robi
# The frame is samples/robi/zigzag_scene4_view0/. It is Zigzag, scene 4, view 0, Ensenso.
# 这一帧在 samples/robi/zigzag_scene4_view0/。Zigzag，场景 4，第 0 帧，Ensenso。
# left.bmp is one channel. depth.png is uint16. Zigzag.stl is millimeters.
# left.bmp 是单通道。depth.png 是 uint16。Zigzag.stl 是毫米。
# depth_m and diameter_m are meters. K is pixels. Hypothesis counts stay in wapr/recipe.py.
# depth_m 和 diameter_m 是米。K 是像素。候选数量仍在 wapr/recipe.py。
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import trimesh

script_dir = os.path.dirname(os.path.abspath(__file__))
# This example sits in examples/, so the release root is its parent.
# 本示例在 examples/ 下，release 根目录是它的上一级。
release_dir = os.path.dirname(script_dir)
sys.path.insert(0, release_dir)

from wapr import WAPREstimator
from wapr import recipe
from wapr.download_assets import check_and_fetch_pack
from wapr.resources import samples_dir
from wapr.bop import read_image
from wapr.det2d import WAPRDet2D, onboard_meshes
from wapr.frame import estimate_frame_many_categories_many_instances
from wapr.view import pose_silhouette, visualize_6d_pose, visualize_2d_detection
from wapr.suppression import rendered_mask_iou_nms


def read_camera_yml(path):
    """
    # Read the Ensenso camera file and return its numbers.

    ## Args

        - path: camera.yml. Each line is key, colon, float. depth_unit is millimeters per stored count. fx, fy, cx, cy are pixels. width and height are pixels.

    ## Returns

        - Returns a dict of those floats.

    ---

    # 读入 Ensenso 相机文件，并返回其中的数。

    ## 参数

        - path: camera.yml。每一行是键、冒号、浮点数。depth_unit 是每个存储计数对应的毫米。fx、fy、cx、cy 是像素。width 和 height 是像素。

    ## 返回

        - 返回这些浮点数的 dict。

"""
    values = {}
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if not text or text.startswith("#") or ":" not in text:
                continue
            key, raw = text.split(":", 1)
            values[key.strip()] = float(raw.strip())
    needed = ("fx", "fy", "cx", "cy", "depth_unit", "width", "height")
    missing = [key for key in needed if key not in values]
    if missing:
        raise KeyError("%s missing %s" % (path, ", ".join(missing)))
    return values


if __name__ == "__main__":
    print("example 10  previous examples/09_taco_many_instances.py  next examples/11_reconstruct_object.py", flush=True)
    # Fetch one Ensenso frame on first use; no dataset enters the wheel.
    # 首次使用获取一帧 Ensenso，数据不进入 wheel。
    check_and_fetch_pack("robi")
    sample_dir = os.path.join(samples_dir(), "robi", "zigzag_scene4_view0")
    obj_id = 1
    object_name = "zigzag"
    # 76.2 mm is the Zigzag diameter in the ROBI evaluation. The threshold is 0.1 of that.
    # 76.2 mm 是 ROBI 评测里 Zigzag 的直径。阈值是它的 0.1 倍。
    diameter_m = 0.0762
    template_path = ""
    device = "cuda:0"
    det_backend = "trt"
    # visualize_det writes the 2D box and the mask. recipe.visualize gates the 6D box and the contour.
    # visualize_det 写出二维框和 mask。recipe.visualize 控制 6D 框和轮廓。
    visualize_det = True
    left_path = os.path.join(sample_dir, "left.bmp")
    depth_path = os.path.join(sample_dir, "depth.png")
    camera_path = os.path.join(sample_dir, "camera.yml")
    mesh_path = os.path.join(sample_dir, "Zigzag.stl")
    for path in (left_path, depth_path, camera_path, mesh_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    # Apply the Ensenso calibration before pairing depth pixels with the RGB image.
    # 先使用 Ensenso 标定，再将深度像素与 RGB 图像配对。
    camera = read_camera_yml(camera_path)
    width_px = int(camera["width"])
    height_px = int(camera["height"])
    # depth_unit is millimeters per uint16 count. Divide by 1000 to reach meters.
    # depth_unit 是每个 uint16 计数对应的毫米。除以 1000 得到米。
    depth_unit_mm = float(camera["depth_unit"])
    K = np.array(
        [
            [camera["fx"], 0.0, camera["cx"]],
            [0.0, camera["fy"], camera["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    left = read_image(left_path)
    if left.ndim == 2:
        rgb = np.stack([left, left, left], axis=-1)
    else:
        rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    depth_raw = read_image(depth_path)
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    if rgb.shape[0] != height_px or rgb.shape[1] != width_px:
        raise ValueError("left.bmp is %s, camera.yml is %d x %d" % (rgb.shape, width_px, height_px))
    if depth_raw.shape[0] != height_px or depth_raw.shape[1] != width_px:
        raise ValueError("depth.png is %s, camera.yml is %d x %d" % (depth_raw.shape, width_px, height_px))
    depth_m = depth_raw.astype(np.float32) * depth_unit_mm / 1000.0

    # The STL vertices are millimeters. The estimator takes meters.
    # STL 顶点是毫米。估计器接收的是米。
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * 0.001
    mesh.metadata["name"] = object_name
    mesh_only = {obj_id: mesh}
    template = onboard_meshes(mesh_only, template_path, device=device)
    print(
        "robi_zigzag",
        {
            "sample_dir": sample_dir,
            "obj_id": obj_id,
            "name": object_name,
            "diameter_m": diameter_m,
            "depth_unit_mm": depth_unit_mm,
            "template": template_path if template_path else "gpu",
            "n_view": recipe.n_view,
            "n_inplane": recipe.n_inplane,
            "wapr_iters": recipe.wapr_iters,
            "sapr_iters": recipe.sapr_iters,
            "det_backend": det_backend,
            "device": device,
            "visualize": recipe.visualize,
            "visualize_det": visualize_det,
            "visualize_path": recipe.visualize_path,
        },
        flush=True,
    )
    detector = WAPRDet2D(template, device=device, backend=det_backend)
    meshes = {obj_id: (mesh, diameter_m)}
    estimator = WAPREstimator(device=device)
    pose_mesh = estimator.prepare_meshes([mesh])[0]
    mesh_setup_seconds = estimator.last_mesh_setup_seconds
    pose_meshes = {obj_id: (pose_mesh, diameter_m)}
    print("rgb", tuple(rgb.shape), "depth_m", tuple(depth_m.shape), "K", tuple(K.shape), flush=True)
    # Search for an unknown number of Zigzag parts, then refine each detected pose.
    # 搜索数量未知的 Zigzag 零件，再修正每个检测实例的位姿。
    poses, timing, detections = estimate_frame_many_categories_many_instances(
        estimator, detector, rgb, depth_m, K, pose_meshes,
    )
    print("POSE_TIMING", timing, "model_setup_seconds", estimator.model_setup_seconds,
          "initial_mesh_setup_seconds", mesh_setup_seconds, flush=True)
    # Save every estimated pose before suppression; annotations are not loaded here.
    # 保存抑制前的全部估计位姿；本脚本不读取标注。
    estimated_poses = poses
    # Instance count is unknown. NMS on the rendered masks of this CAD uses mask IoU.
    # IoU above 0.5 suppresses the lower score_6d.
    # 实例数量未知。这份 CAD 的渲染掩膜做 NMS，用的是掩膜 IoU。
    # IoU 大于 0.5 时抑制 score_6d 更低的一条。
    render_nms_iou = 0.5
    n_before_nms = len(poses)
    poses = rendered_mask_iou_nms(poses, mesh, K, height_px, width_px, render_nms_iou)
    print(
        "det_ms", timing["total_ms"],
        "detections", len(detections),
        "instances_before_nms", n_before_nms,
        "instances", len(poses),
        "render_nms_iou", render_nms_iou,
        flush=True,
    )
    if visualize_det:
        visualize_2d_detection(rgb, detections, names={obj_id: object_name}, image=True, filename="robi_zigzag_det.jpg")
    for pose in poses:
        print(
            {
                "obj_id": pose["obj_id"],
                "name": pose.get("name", ""),
                "score_2d": pose["score_2d"],
                "score_6d": pose["score_6d"],
                "t_m": np.asarray(pose["t_m"]).reshape(3).tolist(),
                "time_s": pose["time_s"],
            },
            flush=True,
        )
    rows = []
    for pose in poses:
        row = dict(pose)
        row["mesh"] = meshes[int(pose["obj_id"])][0]
        rows.append(row)
    visualize_6d_pose(rgb, rows, K, filename="robi_zigzag.jpg")
    # Structured outputs let a later evaluator check the actual standard recipe.
    
    output_dir = os.path.join(release_dir, "outputs", "robi_zigzag")
    os.makedirs(output_dir, exist_ok=True)
    saved = {
        "protocol": "example08_standard_detector_and_masked_pose_batch",
        "pose_unit": "meters", "coordinate_frame": "object_to_camera",
        "annotated_pose_used": False, "annotated_mask_used": False,
        "annotated_instance_count_used": False,
        "K": K.reshape(-1).tolist(), "width": width_px, "height": height_px,
        "n_view": int(recipe.n_view), "n_inplane": int(recipe.n_inplane),
        "render_nms_iou": float(render_nms_iou),
        "image_sha256": hashlib.sha256(open(left_path, "rb").read()).hexdigest(),
        "mesh_sha256": hashlib.sha256(open(mesh_path, "rb").read()).hexdigest(),
        "backend": recipe.backend, "device": str(estimator.device),
        "timing_role": "single_call_diagnostic_including_first_inference; not_a_steady_state_benchmark",
        "raw_predictions": [], "retained_source_indices": [], "detections": [],
    }
    for index, pose in enumerate(estimated_poses):
        saved["raw_predictions"].append({
            "source_index": index, "obj_id": int(pose["obj_id"]),
            "pose_4x4": np.asarray(pose["pose_4x4"], dtype=np.float64).reshape(4, 4).tolist(),
            "score_2d": float(pose["score_2d"]), "score_6d": float(pose["score_6d"]),
            "bbox_xywh": [float(value) for value in pose["bbox_xywh"]],
        })
        if any(pose is retained for retained in poses):
            saved["retained_source_indices"].append(index)
    for detection in detections:
        encoded_mask = dict(detection["mask"])
        if isinstance(encoded_mask["counts"], bytes):
            encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")
        saved["detections"].append({
            "obj_id": int(detection["obj_id"]),
            "score_2d": float(detection["score_2d"]),
            "bbox_xywh": [float(value) for value in detection["bbox"]],
            "mask": encoded_mask,
        })
    with open(os.path.join(output_dir, "predictions.json"), "w", encoding="utf-8") as stream:
        json.dump(saved, stream, indent=2)
    print("saved_predictions", os.path.join(output_dir, "predictions.json"), flush=True)
