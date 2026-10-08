# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 12. Previous: examples/11_reconstruct_object.py. Next: examples/13_known_mesh_place.py
# 示例 12。上一例：examples/11_reconstruct_object.py。下一例：examples/13_known_mesh_place.py
"""Detect and place a saved reconstruction in another scene. / 检测并定位另一场景中的已重建物体。

Run: python examples/12_cross_scene_pose.py
运行：python examples/12_cross_scene_pose.py
"""
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

RELEASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "11_reconstruct_object")
# Import reusable mesh I/O, projection and appearance calculations, not a hidden runner.
# 导入可复用的网格读写、投影与外观计算；案例流程直接写在下方入口中。
if CASE_DIR not in sys.path:
    sys.path.insert(0, CASE_DIR)
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)

from pose_frame_align import load_viewer_mesh  # noqa: E402
from step01_point_mask import mesh_diameter_m, project_silhouette  # noqa: E402
from step04_dino_pose import dino_model, dino_cosine  # noqa: E402
from cross_scene_mustard import mask_iou  # noqa: E402
import wapr.det2d as detector_module
from wapr import recipe  # noqa: E402
from wapr.bootstrap import ensure_optional
from wapr.det2d import WAPRDet2D, onboard_meshes  # noqa: E402
from wapr.download_assets import check_and_fetch_pack
from wapr.estimator import (  # noqa: E402
    WAPREstimator, center_from_mesh, guess_translation, make_view_rots,
    poses_original_to_centered,
)
from wapr.ogl import depth2xyzmap_batch, make_crop_pair  # noqa: E402
from wapr.resources import resource_root, samples_dir, outputs_dir, cache_dir

# Author-approved default follows example 11's cracker reconstruction.
# 作者确认的默认值沿用示例 11 的饼干盒重建结果，此入口不重新建模。
OBJECT_NAME = "cracker"
# Consume example 11's writable output without writing into the installed package.
# 读取示例 11 的可写输出，不向安装包目录写文件。
MESH_STEM = os.path.join(outputs_dir("11_reconstruct_object"), OBJECT_NAME, "prediction", "mesh")
# External BOP data; the camera file supplies K and millimeter depth scale.
# 外部 BOP 数据；相机文件提供内参及毫米深度缩放。
BOP_ROOT = os.path.join(samples_dir(), "bop")
SCENE_ID = 50
IM_ID = 1130
SCENE_DIR = os.path.join(BOP_ROOT, "ycbv", "test", "%06d" % SCENE_ID)
FRAME_RGB = os.path.join(SCENE_DIR, "rgb", "%06d.png" % IM_ID)
FRAME_DEPTH = os.path.join(SCENE_DIR, "depth", "%06d.png" % IM_ID)
FRAME_CAMERA = os.path.join(SCENE_DIR, "scene_camera.json")
OBJ_ID = 2
DEVICE = "cuda:0"
OUT_DIR = os.path.join(outputs_dir(__file__), OBJECT_NAME)
# Optional evaluation mask is loaded only after the prediction is saved.
# 可选评测掩码只在预测结果保存后读取。
REFERENCE_MASK_PATH = None
# Keep the reconstruction example's WBPS gate before DINOv2 appearance ranking.
# 沿用重建示例的 WBPS 筛选条件，再按 DINOv2 外观相似度选择候选。
WITHIN_MIN = 0.0

if __name__ == "__main__":
    recipe.visualize_path = OUT_DIR
    # Prepare optional detection before importing its mask codec.
    # 先准备可选检测功能，成功后再导入它的掩码编解码器。
    ensure_optional("det2d")
    from pycocotools import mask as mask_utils

    if BOP_ROOT == os.path.join(samples_dir(), "bop"):
        check_and_fetch_pack("ycbv")
    # 1. Reuse the independent reconstruction from Scene A, including its texture.
    # 1. 读取 A 场景独立重建出的米制网格与贴图，不读取 B 场景的参考 CAD 或位姿。
    provenance_path = os.path.join(os.path.dirname(MESH_STEM), "provenance.json")
    for path in (MESH_STEM + ".bin", MESH_STEM + ".json", provenance_path,
                 FRAME_RGB, FRAME_DEPTH, FRAME_CAMERA):
        if not os.path.isfile(path):
            raise FileNotFoundError("Prepare example 11 and scene inputs / 请准备示例 11 与新场景输入: " + path)
    with open(provenance_path, encoding="utf-8") as stream:
        provenance = json.load(stream)
    if (provenance.get("reference_cad_used") is not False
            or provenance.get("annotated_pose_used") is not False
            or provenance.get("mask_source") not in ("sam2_point", "qwen_box_sam2")
            or provenance.get("unit") != "meters"
            or provenance.get("coordinate_frame") != "reconstructed_object"):
        raise ValueError("Use example 11's independent meter-scale prediction / 请使用示例 11 的独立米制预测")
    mesh = load_viewer_mesh(MESH_STEM)

    # 2. Load Scene B's RGB-D and camera. BOP depth_scale converts raw depth to mm.
    # 2. 读取 B 场景 RGB-D 与内参；BOP 原始深度乘 depth_scale 得毫米，再换成米。
    bgr = cv2.imread(FRAME_RGB, cv2.IMREAD_COLOR)
    raw_depth = cv2.imread(FRAME_DEPTH, cv2.IMREAD_UNCHANGED)
    if bgr is None or raw_depth is None or raw_depth.ndim != 2:
        raise ValueError("Invalid RGB-D inputs / RGB-D 输入无效")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with open(FRAME_CAMERA, encoding="utf-8") as stream:
        camera = json.load(stream)[str(IM_ID)]
    K = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
    depth_m = raw_depth.astype(np.float32) * float(camera.get("depth_scale", 1.0)) / 1000.0
    if rgb.shape[:2] != depth_m.shape or not np.isfinite(K).all():
        raise ValueError("Invalid RGB-D grid or K / RGB-D 网格或内参无效")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 3. Render Scene A's mesh into a CAD template bank. Reuse an unchanged bank.
    # 3. 为 A 场景的重建网格生成 CAD 模板库；网格、贴图和类别编号不变时复用缓存。
    # A changed encoder implementation selects another cache without overwriting earlier banks.
    # 编码实现变化时使用另一份缓存；保留此前的模板库，不覆盖已有用户文件。
    with open(detector_module.__file__, "rb") as stream:
        encoder_source_hash = hashlib.sha256(stream.read()).hexdigest()
    bank_dir = os.path.join(cache_dir(), "templates", "12_cross_scene_pose", OBJECT_NAME)
    os.makedirs(bank_dir, exist_ok=True)
    bank_path = os.path.join(bank_dir, "templates_" + encoder_source_hash + ".pt")
    bank_meta_path = bank_path + ".source.json"
    with open(MESH_STEM + ".json", encoding="utf-8") as stream:
        layout = json.load(stream)
    mesh_files = [MESH_STEM + ".bin", MESH_STEM + ".json"]
    if layout.get("mode") == "uv":
        mesh_files.append(os.path.join(os.path.dirname(MESH_STEM), layout["texture"]))
    digest = hashlib.sha256()
    for path in mesh_files:
        with open(path, "rb") as stream:
            digest.update(stream.read())
    mesh_hash = digest.hexdigest()
    bank_source = {"mesh_sha256": mesh_hash, "obj_id": int(OBJ_ID)}
    previous_source = None
    if os.path.isfile(bank_meta_path):
        with open(bank_meta_path, encoding="utf-8") as stream:
            previous_source = json.load(stream)
    if not os.path.isfile(bank_path) or previous_source != bank_source:
        onboard_meshes({OBJ_ID: mesh}, bank_path, device=DEVICE)
        with open(bank_meta_path, "w", encoding="utf-8") as stream:
            json.dump(bank_source, stream, indent=2)

    # 4. Predict regions from Scene B's RGB and the reconstructed-mesh templates.
    # 4. 使用 B 场景 RGB 和重建网格模板预测区域；场景标注、参考 mask 不参与检测。
    detector = WAPRDet2D(bank_path, device=DEVICE)
    for _ in range(3):
        detector.detect_many_categories_many_instances(rgb)
    instances, detection_timing = detector.detect_many_categories_many_instances(rgb)
    report = {
        "path_base": "release_root", "mesh_stem": os.path.relpath(MESH_STEM, RELEASE_DIR),
        "mesh_sha256": mesh_hash, "source_rgb": os.path.relpath(FRAME_RGB, RELEASE_DIR),
        "source_depth": os.path.relpath(FRAME_DEPTH, RELEASE_DIR),
        "camera_file": os.path.relpath(FRAME_CAMERA, RELEASE_DIR), "frame_key": str(IM_ID),
        "reconstruction_rerun": False, "reference_mask_used_for_selection": False,
        "selection": "highest_score_2d", "detections": len(instances),
        "detection_timing": detection_timing, "unit": "meters",
        "coordinate_frame": "reconstructed_object", "pose_convention": "object_to_camera",
    }
    report_path = os.path.join(OUT_DIR, "report.json")
    if not instances:
        report["status"] = "no_detection"
        for name in ("pose.npy", "mask.png", "overlay.png", "evaluation.json"):
            path = os.path.join(OUT_DIR, name)
            if os.path.isfile(path):
                os.remove(path)
        with open(report_path, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
        print("REPORT", report_path, "no_detection", flush=True)
        sys.exit(0)
    best_index = max(range(len(instances)), key=lambda index: float(instances[index]["score_2d"]))
    instance = instances[best_index]
    predicted_mask = np.asarray(mask_utils.decode(instance["mask"])) > 0
    if predicted_mask.ndim == 3:
        predicted_mask = predicted_mask[..., 0]
    print("SELECTED_REGION", best_index, "score_2d", instance["score_2d"], flush=True)

    # 5. Initialize every pose hypothesis from the predicted region and sensor depth.
    # 5. 用预测区域及传感器深度初始化全部候选姿态；4×3 为当前配置，不是实例数量。
    estimator = WAPREstimator(device=DEVICE)
    dino_started = time.perf_counter()
    dino_model()
    torch.cuda.synchronize(estimator.device)
    dino_setup_seconds = time.perf_counter() - dino_started
    prepared = estimator.prepare_meshes([mesh])[0]
    mesh_setup_seconds = estimator.last_mesh_setup_seconds
    center_m = center_from_mesh(prepared)
    diameter_m = mesh_diameter_m(prepared.vertices)
    translation_m = guess_translation(depth_m, predicted_mask, K)
    rotations = make_view_rots(recipe.n_view, recipe.n_inplane)
    candidate_count = int(rotations.shape[0])
    initial_poses = np.zeros((candidate_count, 4, 4), dtype=np.float32)
    initial_poses[:, 3, 3] = 1
    initial_poses[:, :3, :3] = rotations
    initial_poses[:, :3, 3] = translation_m.reshape(1, 3)
    poses = torch.as_tensor(initial_poses, device=estimator.device, dtype=torch.float32)
    warmup_seconds = estimator.warmup_pose(
        rgb, depth_m, K, [{"mesh": prepared, "diameter_m": diameter_m, "mask": predicted_mask}],
    )
    rgb_device = torch.as_tensor(rgb, device=estimator.device, dtype=torch.float32)
    depth_device = torch.as_tensor(depth_m, device=estimator.device, dtype=torch.float32)
    mask_device = torch.as_tensor(predicted_mask, device=estimator.device, dtype=torch.float32)
    K_device = torch.as_tensor(K, device=estimator.device, dtype=torch.float32)[None]
    observed_xyz = depth2xyzmap_batch(depth_device[None], K_device).permute(0, 3, 1, 2)
    mesh_ids = [estimator.renderer.cached_mesh_id(prepared)] * candidate_count

    # 6. Refine the whole group with masked WAPR, then SAPR; WBPS scores the group.
    # 6. 带 mask 的 WAPR 批量计算修正整组候选，SAPR 再修正，WBPS 对该独立物体组评分。
    torch.cuda.synchronize(estimator.device)
    pose_started = time.perf_counter()
    poses = estimator._refine(
        estimator.nets["wapr_w_mask"], poses, rgb_device, depth_device, mask_device, K,
        prepared, center_m, diameter_m, iters=recipe.wapr_iters,
        max_rot=recipe.wapr_max_rot_rad, crop_ratio=recipe.refine_crop_ratio,
        use_mask=True, group_size=candidate_count, observed_xyz=observed_xyz, mesh_ids=mesh_ids,
    )
    poses = estimator._refine(
        estimator.nets["sapr"], poses, rgb_device, depth_device, mask_device, K,
        prepared, center_m, diameter_m, iters=recipe.sapr_iters,
        max_rot=recipe.sapr_max_rot_rad, crop_ratio=recipe.refine_crop_ratio,
        use_mask=False, group_size=candidate_count, observed_xyz=observed_xyz, mesh_ids=mesh_ids,
    )
    within, between = estimator._wbps(
        poses, rgb_device, depth_device, mask_device, K, prepared, center_m, diameter_m,
        candidate_count, observed_xyz=observed_xyz, mesh_ids=mesh_ids,
    )
    within = within.reshape(-1).detach().float().cpu().numpy()
    between = between.reshape(-1).detach().float().cpu().numpy()
    torch.cuda.synchronize(estimator.device)
    refine_score_hot_seconds = time.perf_counter() - pose_started

    # 7. Rank the retained rendered hypotheses by DINOv2 appearance similarity.
    # 7. 将保留候选的渲染与照片交给 DINOv2 比较外观；若 WBPS 全未达阈值，保留全组供排序。
    poses_centered = poses_original_to_centered(poses, center_m)
    observed_crops, rendered_crops, _, _ = make_crop_pair(
        rgb, depth_m, predicted_mask, poses_centered, K, prepared, diameter_m,
        crop_ratio=recipe.wbps_crop_ratio, use_mask=True, device=estimator.device,
    )
    dino_warm_started = time.perf_counter()
    for _ in range(3):
        dino_cosine(observed_crops[:, :3], rendered_crops[:, :3], observed_crops[:, 3:4])
    torch.cuda.synchronize(estimator.device)
    dino_warmup_seconds = time.perf_counter() - dino_warm_started
    dino_started = time.perf_counter()
    cosine = dino_cosine(observed_crops[:, :3], rendered_crops[:, :3], observed_crops[:, 3:4])
    torch.cuda.synchronize(estimator.device)
    dino_hot_seconds = time.perf_counter() - dino_started
    keep = within > WITHIN_MIN
    if not bool(np.any(keep)):
        keep = np.ones(candidate_count, dtype=bool)
    selected_index = int(np.argmax(np.where(keep, cosine, -1.0)))
    pose = poses[selected_index].detach().float().cpu().numpy().astype(np.float64)
    selected = {
        "pose_4x4": pose.tolist(), "within_group": float(within[selected_index]),
        "score_6d": (100.0 - float(between.max())) / 200.0,
        "dino_cosine": float(cosine[selected_index]), "hypothesis": selected_index,
        "candidate_count": candidate_count,
        "timing": {"model_setup_seconds": estimator.model_setup_seconds,
                   "mesh_setup_seconds": mesh_setup_seconds, "warmup_seconds": warmup_seconds,
                   "refine_score_hot_seconds": refine_score_hot_seconds,
                   "backend": str(recipe.backend), "renderer": "ogl",
                   "refine_score_time_scope": "prepared GPU tensors; H2D preparation separate",
                   "dino_setup_seconds": dino_setup_seconds,
                   "dino_warmup_seconds": dino_warmup_seconds, "dino_hot_seconds": dino_hot_seconds,
                   "dino_time_scope": "prepared rendered crops; excludes crop generation, model loading and warmup"},
    }

    # 8. Save the prediction before opening any optional evaluation reference.
    # 8. 先保存物体到相机位姿、预测 mask 与叠图，再读取可选评测参考，避免真值参与选择。
    silhouette, overlay = project_silhouette(rgb, mesh, pose, K)
    np.save(os.path.join(OUT_DIR, "pose.npy"), pose)
    if (not cv2.imwrite(os.path.join(OUT_DIR, "overlay.png"), overlay)
            or not cv2.imwrite(os.path.join(OUT_DIR, "mask.png"), predicted_mask.astype(np.uint8) * 255)):
        raise OSError("Could not save prediction images / 无法保存预测图像")
    report.update({"status": "predicted", "evaluation_requested": REFERENCE_MASK_PATH is not None,
                   "selected_detection": best_index, "score_2d": float(instance["score_2d"]),
                   "bbox_xywh_px": instance["bbox"], "pose_4x4": pose.tolist(), "pose": selected})
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    evaluation_path = os.path.join(OUT_DIR, "evaluation.json")
    if REFERENCE_MASK_PATH is None:
        if os.path.isfile(evaluation_path):
            os.remove(evaluation_path)
    else:
        reference = cv2.imread(REFERENCE_MASK_PATH, cv2.IMREAD_GRAYSCALE)
        if reference is None or reference.shape != predicted_mask.shape:
            raise ValueError("Invalid optional evaluation mask / 可选评测掩码无效")
        evaluation = {"path_base": "release_root",
                      "reference_mask": os.path.relpath(REFERENCE_MASK_PATH, RELEASE_DIR),
                      "detection_iou": mask_iou(predicted_mask, reference > 0),
                      "pose_iou": mask_iou(silhouette > 0, reference > 0)}
        with open(evaluation_path, "w", encoding="utf-8") as stream:
            json.dump(evaluation, stream, indent=2)
        print("EVALUATION", evaluation, flush=True)
    print("SELECTED_POSE", selected_index, "within_group", selected["within_group"],
          "dino_cosine", selected["dino_cosine"], flush=True)
    print("REPORT", report_path, flush=True)
