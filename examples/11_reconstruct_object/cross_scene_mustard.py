# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Detect a YCBInEOAT reconstruction on one YCB-V frame.

用 YCBInEOAT 上重建的网格，在一张 YCB-V 图上做检测和位姿。

Scene A builds the mesh. Scene B never supplies that mesh, and the detector does
not read Scene B's object ids, boxes, or masks. Those masks are only scored
after the pose exists.
A 场景生成网格。B 场景不提供这份网格，检测器也不读 B 的物体编号、框和 mask。
那些 mask 只在位姿出来之后用来打分。
"""

import json
import os
import sys

import numpy as np
import torch
import wapr


DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.dirname(os.path.dirname(DEMO_DIR))
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from step01_point_mask import mesh_diameter_m, project_silhouette  # noqa: E402
from wapr.resources import outputs_dir, samples_dir, cache_dir


# Scene A supplies an independent meter-scale prediction with its saved appearance.
# A 场景提供独立的米制预测网格，并保留其保存的外观。
MESH_STEM = os.path.join(outputs_dir("11_reconstruct_object"), "mustard", "prediction", "mesh")
SCENE_A_RGB = os.path.join(
    os.path.join(samples_dir(), "ycbineoat"), "mustard0", "rgb", "1581120424100262102.png",
)
# Scene B is one YCB-V test frame. Five objects are visible. The mustard is obj 5.
# B 场景是 YCB-V 测试集的一帧。画面里有五个物体。芥末瓶是 5 号。
BOP_ROOT = os.path.join(samples_dir(), "bop")
SCENE_ID = 50
IM_ID = 1130
# The library stores this one mesh under the YCB-V mustard id. The id is a label
# on the bank, not a box given to the detector.
# 库里这份网格用 YCB-V 芥末瓶的编号保存。这个编号是库的标签，不是交给检测器的框。
OBJ_ID = 5
BANK_PATH = os.path.join(cache_dir(), "templates", "det2d", "cross_mustard_ycbineoat_box12s.pt")
OUT_DIR = os.path.join(outputs_dir("11_reconstruct_object"), "stages", "cross_scene_mustard")
# Exported recipes reuse the installed package font.
# 导出的配方复用安装包内字体。
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(wapr.__file__)), "fonts", "wqy-microhei.ttc")
DEVICE = "cuda:0"


def run_cross_scene(mesh_stem, rgb_path, depth_path, camera_path, frame_key,
                    obj_id, output_dir, device, reference_mask_path=None):
    """Use top score_2d, one pose batch, and optional post-prediction IoU.

    使用最高 score_2d、一次位姿候选批处理，以及可选的预测后 IoU 对照。
    """
    import hashlib
    import cv2
    from pycocotools import mask as mask_utils
    from wapr.det2d import WAPRDet2D, onboard_meshes
    from wapr.estimator import WAPREstimator
    from pose_frame_align import load_viewer_mesh
    from step04_dino_pose import select_pose_result

    provenance_path = os.path.join(os.path.dirname(mesh_stem), "provenance.json")
    for path in (mesh_stem + ".bin", mesh_stem + ".json", provenance_path,
                 rgb_path, depth_path, camera_path):
        if not os.path.isfile(path):
            raise FileNotFoundError("Prepare example 11 and scene inputs / 请准备示例 11 与新场景输入: " + path)
    with open(provenance_path, encoding="utf-8") as stream:
        provenance = json.load(stream)
    if (provenance.get("reference_cad_used") is not False
            or provenance.get("annotated_pose_used") is not False
            or provenance.get("mask_source") not in ("sam2_point", "qwen_box_sam2")
            or provenance.get("unit") != "meters"
            or provenance.get("coordinate_frame") != "reconstructed_object"):
        raise ValueError("Use the independent meter-scale prediction from 11 / 请使用示例 11 的独立米制预测")
    mesh = load_viewer_mesh(mesh_stem)
    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    raw_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if bgr is None or raw_depth is None or raw_depth.ndim != 2:
        raise ValueError("Invalid RGB-D inputs / RGB-D 输入无效")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with open(camera_path, encoding="utf-8") as stream:
        camera = json.load(stream)[str(frame_key)]
    k = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
    depth_m = raw_depth.astype(np.float32) * float(camera.get("depth_scale", 1.0)) / 1000.0
    if rgb.shape[:2] != depth_m.shape or not np.isfinite(k).all():
        raise ValueError("Invalid RGB-D grid or K / RGB-D 网格或内参无效")
    os.makedirs(output_dir, exist_ok=True)
    bank_path = os.path.join(output_dir, "templates.pt")
    bank_meta_path = os.path.join(output_dir, "templates_source.json")
    with open(mesh_stem + ".json", encoding="utf-8") as stream:
        layout = json.load(stream)
    mesh_files = [mesh_stem + ".bin", mesh_stem + ".json"]
    if layout.get("mode") == "uv":
        mesh_files.append(os.path.join(os.path.dirname(mesh_stem), layout["texture"]))
    digest = hashlib.sha256()
    for path in mesh_files:
        with open(path, "rb") as stream:
            digest.update(stream.read())
    mesh_hash = digest.hexdigest()
    bank_source = {"mesh_sha256": mesh_hash, "obj_id": int(obj_id)}
    previous_source = None
    if os.path.isfile(bank_meta_path):
        with open(bank_meta_path, encoding="utf-8") as stream:
            previous_source = json.load(stream)
    # Rebuild stale templates; a changed mesh or texture changes the detector bank.
    # 网格或贴图改变时重建模板库，避免复用旧模板。
    if not os.path.isfile(bank_path) or previous_source != bank_source:
        onboard_meshes({obj_id: mesh}, bank_path, device=device)
        with open(bank_meta_path, "w", encoding="utf-8") as stream:
            json.dump(bank_source, stream, indent=2)
    detector = WAPRDet2D(bank_path, device=device)
    for _ in range(3):
        detector.detect_many_categories_many_instances(rgb)
    instances, timing = detector.detect_many_categories_many_instances(rgb)
    report = {"path_base": "release_root", "mesh_stem": os.path.relpath(mesh_stem, RELEASE_DIR), "mesh_sha256": mesh_hash,
              "source_rgb": os.path.relpath(rgb_path, RELEASE_DIR), "source_depth": os.path.relpath(depth_path, RELEASE_DIR),
              "camera_file": os.path.relpath(camera_path, RELEASE_DIR), "frame_key": str(frame_key),
              "reconstruction_rerun": False, "reference_mask_used_for_selection": False,
              "selection": "highest_score_2d", "detections": len(instances),
              "detection_timing": timing, "unit": "meters",
              "coordinate_frame": "reconstructed_object", "pose_convention": "object_to_camera"}
    report_path = os.path.join(output_dir, "report.json")
    if not instances:
        report["status"] = "no_detection"
        # Clear stale prediction files when the new frame has no detection.
        # 新帧没有检测结果时清理旧预测文件，避免被误认为本次输出。
        for name in ("pose.npy", "mask.png", "overlay.png", "evaluation.json"):
            path = os.path.join(output_dir, name)
            if os.path.isfile(path):
                os.remove(path)
        with open(report_path, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
        print("REPORT", report_path, "no_detection", flush=True)
        return report
    # Neither scene object labels nor evaluation masks participate in selection.
    # 场景类别标注和评测掩码均不参与实例选择。
    best_index = max(range(len(instances)), key=lambda index: float(instances[index]["score_2d"]))
    instance = instances[best_index]
    predicted_mask = np.asarray(mask_utils.decode(instance["mask"])) > 0
    if predicted_mask.ndim == 3:
        predicted_mask = predicted_mask[..., 0]
    estimator = WAPREstimator(device=device)
    selected = select_pose_result(estimator, rgb, depth_m, predicted_mask, k, mesh)
    pose = np.asarray(selected["pose_4x4"], dtype=np.float64)
    silhouette, overlay = project_silhouette(rgb, mesh, pose, k)
    np.save(os.path.join(output_dir, "pose.npy"), pose)
    overlay_path = os.path.join(output_dir, "overlay.png")
    mask_path = os.path.join(output_dir, "mask.png")
    if not cv2.imwrite(overlay_path, overlay) or not cv2.imwrite(mask_path, predicted_mask.astype(np.uint8) * 255):
        raise OSError("Could not save prediction images / 无法保存预测图像")
    report.update({"status": "predicted", "evaluation_requested": reference_mask_path is not None,
                   "selected_detection": best_index,
                   "score_2d": float(instance["score_2d"]), "bbox_xywh_px": instance["bbox"],
                   "pose_4x4": pose.tolist(), "pose": selected})
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    # Load optional reference only after all prediction products are saved.
    # 全部预测产物保存后才读取可选参考掩码。
    if reference_mask_path is None:
        previous_evaluation = os.path.join(output_dir, "evaluation.json")
        if os.path.isfile(previous_evaluation):
            os.remove(previous_evaluation)
    else:
        reference = cv2.imread(reference_mask_path, cv2.IMREAD_GRAYSCALE)
        if reference is None or reference.shape != predicted_mask.shape:
            raise ValueError("Invalid optional evaluation mask / 可选评测掩码无效")
        evaluation = {"path_base": "release_root", "reference_mask": os.path.relpath(reference_mask_path, RELEASE_DIR),
                      "detection_iou": mask_iou(predicted_mask, reference > 0),
                      "pose_iou": mask_iou(silhouette > 0, reference > 0)}
        with open(os.path.join(output_dir, "evaluation.json"), "w", encoding="utf-8") as stream:
            json.dump(evaluation, stream, indent=2)
        print("EVALUATION", evaluation, flush=True)
    print("REPORT", report_path, flush=True)
    return report


def load_reconstructed_mesh():
    """
    # Return the Scene A prediction mesh.

    ## Args

        - There are no arguments.

    ## Returns

        - The return is a trimesh.
        - Vertices: meters. UV or vertex colors are kept. It is not None.

    ---

    # 返回 A 场景的预测网格。

    ## 参数

        - 没有参数。

    ## 返回

        - 返回值是 trimesh。
        - 顶点单位米。
        - UV 或顶点色都保留。
        - 不是 None。

"""
    from pose_frame_align import load_viewer_mesh

    return load_viewer_mesh(MESH_STEM)


def build_bank(mesh):
    """
    # Render the reconstructed mesh into a one-object template bank.

    ## Args

        - mesh: a trimesh, vertices in meters. It is not None. It is stored under OBJ_ID.

    ## Returns

        - Returns None.
        - The return is None.
        - The bank file is BANK_PATH.

    ---

    # 把重建网格渲成只有一个物体的模板库。

    ## 参数

        - mesh: trimesh，顶点单位米。不是 None。它存在 OBJ_ID 下面。

    ## 返回

        - 返回 None。
        - 返回值是 None。
        - 库文件是 BANK_PATH。

"""
    from wapr.det2d import onboard_meshes

    os.makedirs(os.path.dirname(BANK_PATH), exist_ok=True)
    onboard_meshes({OBJ_ID: mesh}, BANK_PATH, device=DEVICE)
    print("BANK", BANK_PATH, flush=True)


def load_scene_b():
    """
    # Return the YCB-V frame: RGB, depth, K, and the visible ground-truth masks.

    ## Args

        - There are no arguments.

    ## Returns

        - The return has four fields, and none is None.
        - rgb: (H, W, 3) uint8 RGB.
        - depth_m: (H, W) float32 meters.
        - k: (3, 3) float64 pixels.
        - masks: a list of dicts with obj_id int and mask (H, W) bool. These masks are not passed to the detector.

    ---

    # 返回 YCB-V 这一帧：RGB、深度、K，以及可见真值 mask。

    ## 参数

        - 没有参数。

    ## 返回

        - 返回值有四项，都不是 None。
        - rgb: (H, W, 3) uint8 RGB。
        - depth_m: (H, W) float32，米。
        - k: (3, 3) float64，像素。
        - masks: 字典列表，含 obj_id（整数）和 mask（(H, W) bool）。这些 mask 不传给检测器。

"""
    import cv2

    scene_dir = os.path.join(BOP_ROOT, "ycbv", "test", "%06d" % SCENE_ID)
    with open(os.path.join(scene_dir, "scene_camera.json"), encoding="utf-8") as stream:
        camera = json.load(stream)[str(IM_ID)]
    k = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
    depth_scale = float(camera.get("depth_scale", 1.0))
    rgb_path = os.path.join(scene_dir, "rgb", "%06d.png" % IM_ID)
    depth_path = os.path.join(scene_dir, "depth", "%06d.png" % IM_ID)
    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_m = depth.astype(np.float32) * depth_scale / 1000.0
    with open(os.path.join(scene_dir, "scene_gt.json"), encoding="utf-8") as stream:
        objects = json.load(stream)[str(IM_ID)]
    masks = []
    for index, obj in enumerate(objects):
        mask_path = os.path.join(scene_dir, "mask_visib", "%06d_%06d.png" % (IM_ID, index))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masks.append({"obj_id": int(obj["obj_id"]), "mask": mask > 0})
    return rgb, depth_m, k, masks


def mask_iou(left, right):
    """
    # Return the intersection over union of two masks.

    ## Args

        - left: a boolean array. It is not None.
        - right: a boolean array of the same shape. It is not None.

    ## Returns

        - The return is one float, unitless, from 0 to 1.
        - It is not None.
        - An empty union returns 0.0.

    ---

    # 返回两个 mask 的交并比。

    ## 参数

        - left: 布尔数组。不是 None。
        - right: 同形状的布尔数组。不是 None。

    ## 返回

        - 返回值是一个浮点，无量纲，从 0 到 1。
        - 不是 None。
        - 并集为空时返回 0.0。

"""
    union = np.logical_or(left, right).sum()
    if int(union) == 0:
        return 0.0
    return float(np.logical_and(left, right).sum()) / float(union)


def paint_score_badge(image, text, x, y, high):
    """
    # Paint one score chip above a box.

    ## Args

        - image: (H, W, 3) uint8 BGR. It is not None. The chip is drawn into it.
        - text: the score string. It is not None.
        - x: the box left edge in pixels. It is not None.
        - y: the box top edge in pixels. It is not None. The chip sits above this edge when there is room.
        - high: a bool. It is not None. True uses blue. False uses gray.

    ## Returns

        - Returns None.
        - The return is None.

    ---

    # 在框的上方画一块分数芯片。

    ## 参数

        - image: (H, W, 3) uint8 BGR。不是 None。芯片画进这张图。
        - text: 分数字符串。不是 None。
        - x: 框的左边缘，单位像素。不是 None。
        - y: 框的上边缘，单位像素。不是 None。有空位时芯片放在这条边的上方。
        - high: 布尔值。不是 None。真用蓝色。假用灰色。

    ## 返回

        - 返回 None。
        - 返回值是 None。

"""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thickness = 2
    (text_w, text_h), _base = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    height, width = image.shape[:2]
    left = int(round(x))
    top = int(round(y)) - text_h - 2 * pad
    if top < 2:
        top = int(round(y)) + 2
    left = max(2, min(left, width - text_w - 2 * pad - 2))
    top = max(2, min(top, height - text_h - 2 * pad - 2))
    color = (179, 93, 36) if high else (118, 118, 118)
    cv2.rectangle(
        image,
        (left, top),
        (left + text_w + 2 * pad, top + text_h + 2 * pad),
        color,
        -1,
    )
    cv2.putText(
        image, text,
        (left + pad, top + text_h + pad - 1),
        font, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )


def mark_score_rows(image, rows, score_key, draw_box):
    """
    # Draw each score, largest box first within a score group.

    ## Args

        - image: (H, W, 3) uint8 BGR. It is not None.
        - rows: a list of dicts. It is not None. Each dict has score_key as a float and bbox_xywh as four numbers in pixels, (x, y, w, h).
        - score_key: the dict key to read. It is not None.
        - draw_box: a bool. It is not None. True also draws the rectangle. False draws only the chip.

    ## Returns

        - Returns None.
        - The return is None.
        - The highest score is blue and is drawn last.

    ---

    # 画出每个分数。

    ## 参数

        - image: (H, W, 3) uint8 BGR。不是 None。
        - rows: 字典列表。不是 None。每个字典的 score_key 是浮点，bbox_xywh 是四个数，像素 (x, y, w, h)。
        - score_key: 要读取的字典键。不是 None。
        - draw_box: 布尔值。不是 None。真时同时画矩形。假时只画芯片。

    ## 返回

        - 同一分数组里面积大的框先画。
        - 返回 None。
        - 返回值是 None。
        - 最高分是蓝色，并且最后画。

"""
    import cv2

    best = max(float(row[score_key]) for row in rows)
    # Low scores first. The high box and its chip are drawn last, on top.
    # 先画低分。最高分的框和芯片最后画，压在上面。
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row[score_key]) == best,
            -(row["bbox_xywh"][2] * row["bbox_xywh"][3]),
        ),
    )
    for row in ordered:
        high = float(row[score_key]) == best
        x, y, w, h = [int(round(value)) for value in row["bbox_xywh"]]
        if draw_box:
            color = (179, 93, 36) if high else (150, 150, 150)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3 if high else 2)
        paint_score_badge(image, "%.2f" % float(row[score_key]), x, y, high)


def _box_distance(px, py, box):
    """
    # Return the squared pixel distance from a point to a box.

    ## Args

        - px: the point x in pixels. It is not None.
        - py: the point y in pixels. It is not None.
        - box: (x, y, w, h) in pixels. It is not None.

    ## Returns

        - The return is one float, squared pixels.
        - It is not None.
        - A point inside the box returns 0.

    ---

    # 返回一个点到一个框的像素距离的平方。

    ## 参数

        - px: 点的 x，单位像素。不是 None。
        - py: 点的 y，单位像素。不是 None。
        - box: 像素 (x, y, w, h)。不是 None。

    ## 返回

        - 返回值是一个浮点，像素的平方。
        - 不是 None。
        - 点在框内时返回 0。

"""
    x, y, w, h = box
    dx = max(x - px, 0, px - (x + w))
    dy = max(y - py, 0, py - (y + h))
    return dx * dx + dy * dy


def mute_low_pose_contours(photo, rows):
    """
    # Turn the lower pose contours gray and leave the highest one blue.

    ## Args

        - photo: (H, W, 3) uint8 BGR. It is not None. Blue contour pixels are recolored in place.
        - rows: a list of dicts. It is not None. Each dict has score_6d as a float and bbox_xywh as (x, y, w, h) pixels.

    ## Returns

        - Returns None.
        - The return is None.

    ---

    # 把较低的位姿轮廓改成灰色，最高的那条保持蓝色。

    ## 参数

        - photo: (H, W, 3) uint8 BGR。不是 None。蓝色轮廓像素就地改色。
        - rows: 字典列表。不是 None。每个字典的 score_6d 是浮点，bbox_xywh 是像素 (x, y, w, h)。

    ## 返回

        - 返回 None。
        - 返回值是 None。

"""
    import cv2

    best = max(float(row["score_6d"]) for row in rows)
    boxes = []
    for row in rows:
        x, y, w, h = [int(round(value)) for value in row["bbox_xywh"]]
        boxes.append((x, y, w, h, float(row["score_6d"]) == best))
    blue, green, red = [channel.astype(np.int16) for channel in cv2.split(photo)]
    contour = (
        (blue > 125)
        & (blue > red + 55)
        & (blue > green + 35)
        & (red < 100)
        & (green < 140)
    )
    high_box = next(box[:4] for box in boxes if box[4])
    ys, xs = np.where(contour)
    low = (150, 150, 150)
    # The dashed callout sits on the high box and can cross a lower box.
    # Pixels within this margin of the high box stay blue.
    # 虚线框贴着高分框，可能会穿过低分框。离高分框在这个边距内的像素保持蓝色。
    keep_margin_px = 18
    for py, px in zip(ys.tolist(), xs.tolist()):
        if _box_distance(px, py, high_box) <= keep_margin_px * keep_margin_px:
            continue
        nearest = min(range(len(boxes)), key=lambda index: _box_distance(px, py, boxes[index][:4]))
        if not boxes[nearest][4]:
            photo[py, px] = low




def draw_pose_zoom(pose_bgr, mesh, pose, K, path):
    """
    # Keep every contour and write the magnified reconstruction beside the photo.

    ## Args

        - pose_bgr: (H, W, 3) uint8 BGR, the photo with contours. It is not None.
        - mesh: a trimesh, vertices in meters. It is not None.
        - pose: (4, 4). Translation is meters. It is not None.
        - K: (3, 3) camera intrinsics in pixels. It is not None.
        - path: the output image path. It is not None.

    ## Returns

        - Returns None.
        - The return is None.

    ---

    # 留下每一条轮廓，并把放大的重建画在照片旁边。

    ## 参数

        - pose_bgr: (H, W, 3) uint8 BGR，带轮廓的照片。不是 None。
        - mesh: trimesh，顶点单位米。不是 None。
        - pose: (4, 4)。平移单位米。不是 None。
        - K: (3, 3) 相机内参，单位像素。不是 None。
        - path: 输出图片路径。不是 None。

    ## 返回

        - 返回 None。
        - 返回值是 None。

"""
    import cv2
    from step01_point_mask import POSE_ZOOM_H, POSE_ZOOM_W, draw_zoom_callout, render_mesh_crop

    height, width = pose_bgr.shape[:2]
    left, top, color, hit = render_mesh_crop(mesh, pose, K, height, width)
    box_h, box_w = hit.shape[:2]
    render = np.full((box_h, box_w, 3), 244, dtype=np.uint8)
    render[hit] = color[hit]
    zoom_w = int(POSE_ZOOM_W)
    zoom_h = int(POSE_ZOOM_H)
    scale = min(float(zoom_w) / float(box_w), float(zoom_h) / float(box_h))
    fitted_w = max(int(round(box_w * scale)), 1)
    fitted_h = max(int(round(box_h * scale)), 1)
    fitted = cv2.resize(render, (fitted_w, fitted_h), interpolation=cv2.INTER_NEAREST)
    panel = np.full((zoom_h, zoom_w, 3), 244, dtype=np.uint8)
    panel_x = (zoom_w - fitted_w) // 2
    panel_y = (zoom_h - fitted_h) // 2
    panel[panel_y:panel_y + fitted_h, panel_x:panel_x + fitted_w] = fitted
    gap = 16
    canvas_h = max(height, zoom_h)
    canvas_w = width + gap + zoom_w
    canvas = np.full((canvas_h, canvas_w, 3), 244, dtype=np.uint8)
    photo_y = (canvas_h - height) // 2
    canvas[photo_y:photo_y + height, :width] = pose_bgr
    zoom_top = (canvas_h - zoom_h) // 2
    canvas[zoom_top:zoom_top + zoom_h, width + gap:width + gap + zoom_w] = panel
    draw_zoom_callout(
        canvas,
        (left, top + photo_y, left + box_w - 1, top + photo_y + box_h - 1),
        (width + gap, zoom_top),
        (zoom_w, zoom_h),
    )
    cv2.imwrite(path, canvas)


def flow_label_upright():
    """
    # Return the extra view rotation that stands the printed label up.

    ## Args

        - There are no arguments.

    ## Returns

        - The return is (3, 3) float64, unitless.
        - It is not None.
        - It is applied in the view frame after label_yaw.

    ---

    # 返回把印刷标签立起来的额外视角旋转。

    ## 参数

        - 没有参数。

    ## 返回

        - 返回值是 (3, 3) float64，无量纲。
        - 不是 None。
        - 它在 label_yaw 之后作用于视角坐标系。

"""
    extra = -0.5 * np.pi
    cosine = float(np.cos(extra))
    sine = float(np.sin(extra))
    yaw = np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
    flip = np.diag([1.0, -1.0, -1.0])
    roll = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return flip.T @ yaw.T @ roll.T


def draw_figure(scene_a, mesh, scene_b, box_bgr, pose_bgr, path):
    """
    # Write the four-panel cross-scene figure.

    ## Args

        - scene_a: (H, W, 3) uint8 RGB of the build photo. It is not None.
        - mesh: a trimesh, vertices in meters. It is not None.
        - scene_b: (H, W, 3) uint8 RGB. It is not None. Only its height and width size the panels.
        - box_bgr: (H, W, 3) uint8 BGR, the detection picture. It is not None.
        - pose_bgr: (H, W, 3) uint8 BGR, the pose picture. It is not None.
        - path: the PNG path. It is not None.

    ## Returns

        - Returns None.
        - The return is None.

    ---

    # 写出跨场景的四格图。

    ## 参数

        - scene_a: 建网格那张照片的 (H, W, 3) uint8 RGB。不是 None。
        - mesh: trimesh，顶点单位米。不是 None。
        - scene_b: (H, W, 3) uint8 RGB。不是 None。只用它的高和宽来定每一格的尺寸。
        - box_bgr: (H, W, 3) uint8 BGR，检测图。不是 None。
        - pose_bgr: (H, W, 3) uint8 BGR，位姿图。不是 None。
        - path: PNG 路径。不是 None。

    ## 返回

        - 返回 None。
        - 返回值是 None。

"""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    from language_prompt import shaded_mesh

    height, width = scene_b.shape[:2]
    mesh_panel = shaded_mesh(mesh, width, height, flow_label_upright())
    panels = [
        cv2.cvtColor(scene_a, cv2.COLOR_RGB2BGR),
        mesh_panel,
        box_bgr,
        pose_bgr,
    ]
    labels = [
        ("YCBInEOAT", "重建这张"),
        ("Reconstructed mesh", "重建网格"),
        ("YCB-V detection", "YCB-V 上的检测"),
        ("WAPR pose", "WAPR 位姿"),
    ]
    scale = 2
    face_title = ImageFont.truetype(FONT_PATH, 26 * scale)
    face_sub = ImageFont.truetype(FONT_PATH, 15 * scale)
    face = ImageFont.truetype(FONT_PATH, 18 * scale)
    face_zh = ImageFont.truetype(FONT_PATH, 16 * scale)
    tile_w, tile_h = 320 * scale, 240 * scale
    gap = 28 * scale
    title_h = 78 * scale
    foot = 72 * scale
    canvas_w = 4 * tile_w + 5 * gap
    canvas_h = title_h + tile_h + foot
    board = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(board)
    draw.text((canvas_w / 2, 26 * scale), "Scene A mesh, Scene B image", font=face_title, fill=(28, 28, 28), anchor="mm")
    draw.text(
        (canvas_w / 2, 54 * scale),
        "A 场景的网格，拿到 B 场景的另一张图上。检测器看不到 B 的真值。",
        font=face_sub, fill=(70, 70, 70), anchor="mm",
    )
    for index, panel in enumerate(panels):
        small = cv2.resize(panel, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        left = gap + index * (tile_w + gap)
        board.paste(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), (left, title_h))
        en_text, zh_text = labels[index]
        draw.text((left + tile_w / 2, title_h + tile_h + 22 * scale), en_text, font=face, fill=(28, 28, 28), anchor="mm")
        draw.text((left + tile_w / 2, title_h + tile_h + 46 * scale), zh_text, font=face_zh, fill=(70, 70, 70), anchor="mm")
        if index < 3:
            x0 = left + tile_w + 4 * scale
            y = title_h + tile_h / 2
            draw.line([(x0, y), (x0 + gap - 14 * scale, y)], fill=(40, 40, 40), width=3 * scale)
            tip = x0 + gap - 10 * scale
            draw.polygon(
                [(tip, y), (tip - 10 * scale, y - 6 * scale), (tip - 10 * scale, y + 6 * scale)],
                fill=(40, 40, 40),
            )
    board.save(path, "PNG", optimize=True)


def run_scene_b(mesh):
    """
    # Detect on Scene B, estimate poses, and write the figures.

    ## Args

        - mesh: the reconstructed trimesh, vertices in meters. It is not None.

    ## Returns

        - Returns None.
        - The return is None.
        - detect.png, pose.png, flow.png, and result.json are written under OUT_DIR.
        - The shown instance is the highest score_2d. Ground-truth masks are only the printed IoU.

    ---

    # 在 B 场景上检测，估计位姿，并写出图。

    ## 参数

        - mesh: 重建 trimesh，顶点单位米。不是 None。

    ## 返回

        - 返回 None。
        - 返回值是 None。
        - detect.png、pose.png、flow.png 和 result.json 写到 OUT_DIR。
        - 显示的实例是 score_2d 最高的那一个。真值 mask 只用来打印 IoU。

"""
    import time
    import cv2
    from pycocotools import mask as mask_utils
    from wapr.det2d import WAPRDet2D
    from wapr.estimator import WAPREstimator

    os.makedirs(OUT_DIR, exist_ok=True)
    rgb, depth_m, k, gt_masks = load_scene_b()
    diameter_m = mesh_diameter_m(mesh.vertices)
    detector = WAPRDet2D(BANK_PATH, device=DEVICE, backend="trt")
    # Warm the actual cross-scene query before reporting its detector time.
    # 报告检测时间前，预热实际跨场景查询图像。
    for _ in range(3):
        detector.detect_many_categories_many_instances(rgb)
    instances, timing = detector.detect_many_categories_many_instances(rgb, profile=True)
    print("DET", len(instances), "ms", round(timing["total_ms"], 1), "candidates", timing["candidates"], flush=True)
    evidence = detector.last_evidence
    scores = evidence["scores"].detach().float().cpu().numpy()
    proposal_masks = evidence["masks"].detach().cpu().numpy() > 0
    proposal_rows = []
    for index in range(len(proposal_masks)):
        overlaps = [
            {"obj_id": int(row["obj_id"]), "iou": mask_iou(proposal_masks[index], row["mask"])}
            for row in gt_masks
        ]
        overlaps.sort(key=lambda row: -row["iou"])
        proposal_rows.append({"score_2d": float(scores[index, 0]), "best_gt": overlaps[0]})
    proposal_rows.sort(key=lambda row: -row["score_2d"])
    mustard = next(row["mask"] for row in gt_masks if row["obj_id"] == OBJ_ID)
    items = []
    det_masks = []
    for instance in instances:
        visible = np.asarray(mask_utils.decode(instance["mask"]))
        if visible.ndim == 3:
            visible = visible[..., 0]
        items.append({"mesh": mesh, "diameter_m": diameter_m, "mask": visible > 0})
        det_masks.append(visible > 0)
    estimator = WAPREstimator(device=DEVICE)
    prepared = estimator.prepare_meshes([mesh])[0]
    mesh_setup_seconds = estimator.last_mesh_setup_seconds
    for item in items:
        item["mesh"] = prepared
    warmup_seconds = estimator.warmup_pose(rgb, depth_m, k, items)
    torch.cuda.synchronize(DEVICE)
    started = time.perf_counter()
    estimated = estimator.estimate_many_categories_many_instances(rgb, depth_m, k, items)
    torch.cuda.synchronize(DEVICE)
    elapsed_s = time.perf_counter() - started
    print("POSE_TIMING", {"model_setup_seconds": estimator.model_setup_seconds,
          "mesh_setup_seconds": mesh_setup_seconds, "warmup_seconds": warmup_seconds,
          "pose_hot_seconds": elapsed_s}, flush=True)
    share_s = elapsed_s / float(len(estimated)) if estimated else 0.0
    pose_rows = []
    silhouettes = []
    for instance, det_mask, pose in zip(instances, det_masks, estimated):
        silhouette, _one = project_silhouette(rgb, mesh, np.asarray(pose["pose_4x4"]), k)
        silhouettes.append(silhouette)
        row = {
            "score_2d": float(instance["score_2d"]),
            "bbox_xywh": [float(value) for value in instance["bbox"]],
            "score_6d": float(pose["score_6d"]),
            "t_m": [float(value) for value in np.asarray(pose["t_m"]).reshape(-1)],
            "det_iou_mustard": mask_iou(det_mask, mustard),
            "pose_iou_mustard": mask_iou(silhouette > 0, mustard),
            "time_s": share_s,
        }
        pose_rows.append(row)
    best = max(range(len(pose_rows)), key=lambda index: pose_rows[index]["score_2d"])
    from step04_dino_pose import select_pose

    shown_pose, shown_within, shown_cosine, shown_hyp = select_pose(
        estimator, rgb, depth_m, det_masks[best], k, mesh,
    )
    shown_silhouette, _shown = project_silhouette(rgb, mesh, shown_pose, k)
    silhouettes[best] = shown_silhouette
    pose_rows[best]["pose_iou_mustard"] = mask_iou(shown_silhouette > 0, mustard)
    pose_rows[best]["within_group"] = shown_within
    pose_rows[best]["dino_cosine"] = shown_cosine
    pose_rows[best]["hyp"] = shown_hyp
    print(
        "SHOWN", "hyp", shown_hyp,
        "within", round(shown_within, 2),
        "dino", round(shown_cosine, 3),
        "pose_iou", round(pose_rows[best]["pose_iou_mustard"], 3),
        flush=True,
    )
    box_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mark_score_rows(box_bgr, pose_rows, "score_2d", draw_box=True)
    pose_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    best_pose = max(float(row["score_6d"]) for row in pose_rows)
    for silhouette, row in zip(silhouettes, pose_rows):
        high = float(row["score_6d"]) == best_pose
        contours, _hier = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pose_bgr, contours, -1, (179, 93, 36) if high else (150, 150, 150), 3 if high else 2)
        print(
            "POSE",
            "det", round(row["score_2d"], 3),
            "det_iou", round(row["det_iou_mustard"], 3),
            "score_6d", round(row["score_6d"], 3),
            "pose_iou", round(row["pose_iou_mustard"], 3),
            flush=True,
        )
    mark_score_rows(pose_bgr, pose_rows, "score_6d", draw_box=False)
    cv2.imwrite(os.path.join(OUT_DIR, "detect.png"), box_bgr)
    draw_pose_zoom(
        pose_bgr,
        mesh,
        shown_pose,
        k,
        os.path.join(OUT_DIR, "pose.png"),
    )
    scene_a = cv2.cvtColor(cv2.imread(SCENE_A_RGB, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    draw_figure(scene_a, mesh, rgb, box_bgr, pose_bgr, os.path.join(OUT_DIR, "flow.png"))
    report = {
        "path_base": "release_root", "scene_a": os.path.relpath(SCENE_A_RGB, RELEASE_DIR),
        "scene_b": {"dataset": "ycbv", "scene_id": SCENE_ID, "im_id": IM_ID},
        "mesh_diameter_m": diameter_m,
        "model_setup_seconds": estimator.model_setup_seconds,
        "mesh_setup_seconds": mesh_setup_seconds, "warmup_seconds": warmup_seconds,
        "pose_hot_seconds": elapsed_s, "pose_time_scope": "prepared-mesh TRT FP16 + OGL batch",
        "detector_ms": timing["total_ms"],
        "detector_candidates": timing["candidates"],
        "outputs": pose_rows,
        "top_proposals": proposal_rows[:8],
    }
    with open(os.path.join(OUT_DIR, "result.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("wrote", OUT_DIR, flush=True)




if __name__ == "__main__":
    # Run the script stages directly in the entry block.
    # 在入口块中直接执行脚本各阶段。
    scene_dir = os.path.join(BOP_ROOT, "ycbv", "test", "%06d" % SCENE_ID)
    run_cross_scene(
        MESH_STEM, os.path.join(scene_dir, "rgb", "%06d.png" % IM_ID),
        os.path.join(scene_dir, "depth", "%06d.png" % IM_ID),
        os.path.join(scene_dir, "scene_camera.json"), str(IM_ID), OBJ_ID, OUT_DIR, DEVICE,
    )
