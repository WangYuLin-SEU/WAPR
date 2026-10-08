# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 02. Previous: examples/01_one_rgb_detect_segment.py. Next: examples/03_published_boxes_to_pose.py
# 示例 02。上一例：examples/01_one_rgb_detect_segment.py。下一例：examples/03_published_boxes_to_pose.py
# One category, one instance. One image and one object. python examples/02_one_category_one_instance.py
# 单类别、单实例。一张图、一个物体。运行：python examples/02_one_category_one_instance.py
# Page: docs/pose.html#one
# 页面：docs/pose.html#one
# The sample is samples/pose_lmo/, placed by python -m wapr.download_assets.
# 单帧示例在 samples/pose_lmo/，由 python -m wapr.download_assets 放到那里。
# depth_m and diameter_m are meters. K is pixels.
# depth_m 和 diameter_m 是米。K 是像素。
# This lesson supplies the annotated visible mask; it does not run a detector.
# 本课使用标注可见掩码作为输入，不执行自动检测。
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
import torch

ROOT = Path(__file__).resolve().parents[1]
# This example sits in examples/, so the release root is its parent.
# 本示例在 examples/ 下，release 根目录是它的上一级。
sys.path.insert(0, str(ROOT))

from wapr import WAPREstimator
from wapr import recipe
from wapr.download_assets import check_and_fetch_pack
from wapr.resources import samples_dir, outputs_dir
from wapr.view import visualize_6d_pose


if __name__ == "__main__":
    recipe.visualize_path = outputs_dir(__file__)
    print("example 02  previous examples/01_one_rgb_detect_segment.py  next examples/03_published_boxes_to_pose.py", flush=True)
    # Download the four pose checkpoints if any file is missing:
    # cache/weights/wapr_w_mask.pth, wapr_wo_mask.pth, sapr.pth, wbps.pth.
    # These are WAPR, SAPR, and WBPS. Detector weights are not in this pack.
    # 四份位姿权重仅在文件缺失时下载：
    # cache/weights/wapr_w_mask.pth、wapr_wo_mask.pth、sapr.pth、wbps.pth。
    # 这是 WAPR、SAPR 和 WBPS。检测器权重不在该数据包中。
    check_and_fetch_pack("wapr_sapr_wbps")
    # Fetch this lesson's LM-O frame, in meters, when one of its required files is missing.
    # 本课使用的 LM-O 帧以米为单位；任一必需文件缺失时重新获取该数据包。
    check_and_fetch_pack("pose_lmo")
    # Read one RGB-D observation and its metric object model; K stays in pixels.
    # 读取一帧 RGB-D 与米制物体模型；相机内参 K 仍使用像素单位。
    # sample_dir is the LM-O frame from SEU-WYL/WAPR. meta.json records K and diameter_m.
    # sample_dir 是 SEU-WYL/WAPR 里的 LM-O 一帧。meta.json 记录 K 和 diameter_m。
    # bbox_xywh is pixels, y down, and may stay empty.
    # bbox_xywh 是像素，y 向下，可保持为空。
    # Download helpers and this reader share the installed wheel's resource cache.
    # 下载入口与本读取流程共用已安装 wheel 的资源缓存。
    sample_dir = Path(samples_dir()) / "pose_lmo"
    meta_path = sample_dir / "meta.json"
    if not meta_path.is_file():
        raise SystemExit("missing %s. Run: python -m wapr.download_assets" % meta_path)
    meta = json.loads(meta_path.read_text())
    rgb_path = sample_dir / "rgb.png"
    depth_path = sample_dir / "depth.npy"
    mesh_path = sample_dir / "model.ply"
    mask_path = sample_dir / "mask.png"
    diameter_m = float(meta["diameter_m"])
    fx = float(meta["fx"])
    fy = float(meta["fy"])
    cx = float(meta["cx"])
    cy = float(meta["cy"])
    # Leave bbox_xywh as None to use mask.png. Fill [x, y, w, h], pixels, y down,
    # to ignore the mask and run from the bbox.
    # bbox_xywh 保持为 None 时使用 mask.png。若填入像素 [x, y, w, h]、y 向下，则忽略 mask，改由包围盒估计。
    bbox_xywh = None
    device = "cuda:0"
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depth_m = np.load(depth_path).astype(np.float32)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    # name is the label on the image. This sample is LM-O object 12, the holepuncher.
    # name 是图上的文字。这一帧是 LM-O 物体 12，打孔器。
    mesh.metadata["name"] = str(meta["name"]) if "name" in meta else "holepuncher"
    mask = None if bbox_xywh is not None else cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if bbox_xywh is None and mask is None:
        raise FileNotFoundError(mask_path)
    print(
        "input_protocol", "annotated_visible_mask" if bbox_xywh is None else "user_supplied_bbox",
        "pose_ground_truth_used", False,
        flush=True,
    )
    print(
        "single_image",
        {"visualize": recipe.visualize, "visualize_path": recipe.visualize_path},
        flush=True,
    )
    # Estimate one 6D pose, then draw that same returned pose on the input image.
    # 估计一条 6D 位姿，再把返回的这条位姿叠加到输入图像上。
    est = WAPREstimator(device=device)
    # One object still has a batch of hypotheses; prepare GL resources first.
    # 单物体仍有一批候选姿态；先准备 GL 资源，再预热实际位姿负载。
    pose_mesh = est.prepare_meshes([mesh])[0]
    mesh_setup_seconds = est.last_mesh_setup_seconds
    instances = [{"mesh": pose_mesh, "diameter_m": diameter_m,
                  "bbox_xywh": bbox_xywh, "mask": mask}]
    warmup_seconds = est.warmup_pose(rgb, depth_m, K, instances)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    out = est.estimate_one_category_one_instance(rgb, depth_m, K, pose_mesh, diameter_m, bbox_xywh=bbox_xywh, mask=mask)
    torch.cuda.synchronize(device)
    print("POSE_TIMING", {"model_setup_seconds": est.model_setup_seconds,
                          "mesh_setup_seconds": mesh_setup_seconds, "warmup_seconds": warmup_seconds,
                          "pose_hot_seconds": time.perf_counter() - started}, flush=True)
    print(out["pose_4x4"])
    print(out["score_6d"])
    visualize_6d_pose(
        rgb,
        [{
            "name": mesh.metadata["name"],
            "pose_4x4": out["pose_4x4"],
            "score_6d": out["score_6d"],
            "mesh": mesh,
            "bbox_xywh": bbox_xywh,
        }],
        K,
        filename="single_image.jpg",
    )
