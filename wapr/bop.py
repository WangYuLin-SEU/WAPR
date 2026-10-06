# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Read BOP frames, CADs and published detections; write metric pose results.

读取 BOP 帧、CAD 与已公布检测，输出米制位姿对应的 BOP 结果。
Dataset/frame selection and inference loops remain in the caller's recipe.
数据集、帧的选择与推理循环由调用方配置明确给出。
"""

import json
import os

import cv2
import numpy as np
import trimesh

from wapr.estimator import bbox_model_center


def read_image(path):
    """Read stored channels and bit depth unchanged. / 保留存储通道与位深读取图像。"""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image


def test_split(dataset):
    """Resolve the BOP depth-bearing test split. / 确定含深度图的 BOP 测试划分。"""
    if str(dataset).lower() in ("tless", "hb"):
        return "test_primesense"
    return "test"


def first_image(scene_dir, folder, im_id):
    """Find a six-digit BOP frame stem. / 查找六位编号的 BOP 帧文件。"""
    stem = "%06d" % int(im_id)
    for ext in (".png", ".jpg", ".tif", ".tiff"):
        path = os.path.join(scene_dir, folder, stem + ext)
        if os.path.isfile(path):
            return path
    return None


def load_bop_rgbd(bop_path, dataset, scene_id, im_id):
    """Return RGB, float32 depth in meters, and 3×3 pixel intrinsics.

    Stored depth times depth_scale is millimeters. Grayscale is repeated to RGB.
    返回 RGB、float32 米制深度与 3×3 像素内参。
    存储深度乘 depth_scale 得到毫米；灰度图重复为三个通道。
    """
    scene_dir = os.path.join(bop_path, dataset, test_split(dataset), "%06d" % int(scene_id))
    with open(os.path.join(scene_dir, "scene_camera.json"), "r") as stream:
        cam = json.load(stream)[str(int(im_id))]
    K = np.asarray(cam["cam_K"], dtype=np.float32).reshape(3, 3)
    depth_scale = float(cam.get("depth_scale", 1.0))
    rgb_path = first_image(scene_dir, "rgb", im_id) or first_image(scene_dir, "gray", im_id)
    depth_path = first_image(scene_dir, "depth", im_id)
    rgb = read_image(rgb_path)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = np.asarray(rgb[..., :3])
    depth = np.asarray(read_image(depth_path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth_m = depth * depth_scale / 1000.0
    return rgb, depth_m, K


def model_ids(bop_path, dataset):
    """List sorted ids from models/obj_######.ply. / 按文件名列出排序后的 CAD 编号。"""
    models = os.path.join(bop_path, dataset, "models")
    found = []
    for name in sorted(os.listdir(models)):
        if name.startswith("obj_") and name.endswith(".ply"):
            found.append(int(name[len("obj_"):-len(".ply")]))
    return found


def load_mesh_m(bop_path, dataset, obj_id):
    """Convert BOP millimeter geometry/diameter to meters and center the mesh.

    model_center preserves the original CAD origin used by output poses.
    将 BOP 毫米网格与直径换为米，并居中网格。
    model_center 保留输出位姿对应的原始 CAD 原点偏移。
    """
    models = os.path.join(bop_path, dataset, "models")
    ply = os.path.join(models, "obj_%06d.ply" % int(obj_id))
    loaded = trimesh.load(ply, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError("expected one mesh")
    mesh = loaded.copy()
    mesh.apply_scale(0.001)
    center = bbox_model_center(mesh.vertices)
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32) - center.reshape(1, 3)
    mesh.metadata["model_center"] = center
    mesh.metadata["name"] = "obj_%06d" % int(obj_id)
    with open(os.path.join(models, "models_info.json"), "r") as stream:
        info = json.load(stream)[str(int(obj_id))]
    diameter_m = float(info["diameter"]) / 1000.0
    return mesh, diameter_m


def rle_to_mask(rle):
    """Decode supplied COCO RLE, returning None for unavailable segmentation.

    解码给定的 COCO RLE；分割不可用时返回 None，供调用方使用检测框。
    """
    if not rle:
        return None
    counts = rle.get("counts")
    size = rle.get("size")
    if counts is None or size is None:
        return None
    if isinstance(counts, str):
        try:
            from pycocotools import mask as mask_utils
            return np.asarray(mask_utils.decode(rle), dtype=np.uint8)
        except Exception:
            return None
    binary = np.zeros(int(np.prod(size)), dtype=np.uint8)
    start = 0
    for i in range(len(counts) - 1):
        start += int(counts[i])
        end = start + int(counts[i + 1])
        binary[start:end] = (i + 1) % 2
    return binary.reshape(int(size[0]), int(size[1]), order="F")


def load_detections(path, score_thr=0.0):
    """Group published predicted boxes/masks by (scene_id, im_id), without GT.

    按 (scene_id, im_id) 组织公布的预测框与掩码，不读取真值。
    """
    with open(path, "r") as stream:
        raw = json.load(stream)
    if isinstance(raw, list):
        items = raw
    else:
        items = []
        for value in raw.values():
            if isinstance(value, list):
                items.extend(value)
            elif isinstance(value, dict):
                items.append(value)
    out = {}
    for det in items:
        scene_id = int(det.get("scene_id", det.get("sceneId", -1)))
        im_id = int(det.get("image_id", det.get("im_id", det.get("imageId", -1))))
        obj_id = int(det.get("category_id", det.get("obj_id", det.get("categoryId", -1))))
        score = float(det.get("score", det.get("confidence", 0.0)))
        if score < float(score_thr):
            continue
        bbox = det.get("bbox", det.get("bbox_est"))
        if bbox is None:
            continue
        mask = None
        if det.get("segmentation") is not None:
            mask = rle_to_mask(det["segmentation"])
        out.setdefault((scene_id, im_id), []).append(
            {"obj_id": obj_id, "bbox_xywh": [float(x) for x in bbox], "score_2d": score, "mask": mask}
        )
    return out


def keep_top_per_class(records, max_per_class):
    """Keep requested per-class counts in descending 2D score order.

    按 2D 分数降序保留各类别请求的实例数量。
    """
    grouped = {}
    for record in records:
        grouped.setdefault(int(record["obj_id"]), []).append(record)
    kept = []
    for obj_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: float(row["score_2d"]), reverse=True)
        cap = max_per_class.get(obj_id)
        kept.extend(ordered if cap is None else ordered[:int(cap)])
    return kept


def target_frames(bop_path, dataset):
    """Read unique official target frame keys, without object counts.

    读取官方目标文件中去重的帧编号，不向检测流程提供标注物体数量。
    """
    path = os.path.join(bop_path, dataset, "test_targets_bop19.json")
    with open(path, "r") as stream:
        targets = json.load(stream)
    return sorted({(int(row["scene_id"]), int(row["im_id"])) for row in targets})


def write_csv(path, rows):
    """Write row-major R, millimeter t, score_6d and whole-frame seconds.

    All rows of an image receive the caller's same whole-frame time.
    写出按行展开的 R、毫米 t、score_6d 与整帧秒数。
    同一图像的各行使用调用方给出的同一个整帧耗时。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as stream:
        stream.write("scene_id,im_id,obj_id,score,R,t,time\n")
        for row in rows:
            R = np.asarray(row["R"], dtype=np.float64).reshape(-1)
            t_mm = np.asarray(row["t_m"], dtype=np.float64).reshape(3) * 1000.0
            stream.write("%d,%d,%d,%.8f,%s,%s,%.6f\n" % (
                int(row["scene_id"]), int(row["im_id"]), int(row["obj_id"]),
                float(row["score_6d"]), " ".join("%.8f" % float(x) for x in R),
                " ".join("%.8f" % float(x) for x in t_mm), float(row["time"]),
            ))
