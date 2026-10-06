# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 04. Previous: examples/03_published_boxes_to_pose.py. Next: examples/05_bop_6d_detection.py
# 示例 04。上一例：examples/03_published_boxes_to_pose.py。下一例：examples/05_bop_6d_detection.py
# 6D localization. Many known categories, each with a fixed count. python examples/04_6d_localization.py
# 6D 定位。多个已知类别，每个类别数量固定。运行：python examples/04_6d_localization.py
# Page: docs/pose.html#loc
# 页面：docs/pose.html#loc
# The categories and the counts are known before the image is read.
# 类别和数量在读图之前就已经知道。
# depth_m and diameter_m are meters. K is pixels.
# depth_m 和 diameter_m 是米。K 是像素。
import os
import sys

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
# This example sits in examples/, so the release root is its parent.
# 本示例在 examples/ 下，release 根目录是它的上一级。
release_dir = os.path.dirname(script_dir)
sys.path.insert(0, release_dir)

from wapr import WAPREstimator
from wapr import recipe
from wapr.det2d import WAPRDet2D, onboard_meshes
from wapr.bop import load_bop_rgbd, load_mesh_m
from wapr.download_assets import check_and_fetch_pack
from wapr.frame import estimate_frame_many_categories_many_instances
from wapr.view import visualize_6d_pose, visualize_2d_detection



if __name__ == "__main__":
    print("example 04  previous examples/03_published_boxes_to_pose.py  next examples/05_bop_6d_detection.py", flush=True)
    # bop_path is the external dataset root. Empty uses the one-frame pack.
    # bop_path 是外部数据集根目录。为空时用这单帧示例。
    bop_path = ""
    dataset = "lmo"
    scene_id = 2
    im_id = 307
    template_path = ""
    device = "cuda:0"
    det_backend = "trt"
    # visualize_det writes the 2D box and the mask. recipe.visualize gates the 6D box and the contour.
    # visualize_det 写出二维框和 mask。recipe.visualize 控制 6D 框和轮廓。
    visualize_det = True
    # The 2D illustration shows detections retained by the known-count localization.
    # This cap affects only the illustration; every detection enters pose estimation.
    # 2D 插图展示已知数量定位最终保留的检测；上限只影响插图，全部检测仍进入位姿估计。
    preview_top_k = 12
    # Known before this image is read.
    # LM-O test_targets_bop19.json, scene 2, image 307: eight categories, one each.
    # 读这张图之前就知道。
    # LM-O 的 test_targets_bop19.json，场景 2，第 307 帧：八个类别，每个 1 个。
    obj_ids = [1, 5, 6, 8, 9, 10, 11, 12]
    inst_count = {1: 1, 5: 1, 6: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1}
    # LM-O names for those ids. The ply files do not store these words.
    # 这些编号在 LM-O 里的名字。ply 里不存这些词。
    object_names = {
        1: "ape",
        5: "can",
        6: "cat",
        8: "driller",
        9: "duck",
        10: "eggbox",
        11: "glue",
        12: "holepuncher",
    }
    if not bop_path:
        check_and_fetch_pack(dataset)
        bop_path = os.path.join(release_dir, "samples", "bop")
    # The known category list defines the CAD bank; instance counts are applied later.
    # 已知类别列表决定 CAD 模板库的内容；实例数量在后续定位阶段约束。
    mesh_only = {}
    meshes = {}
    for obj_id in obj_ids:
        mesh, diameter_m = load_mesh_m(bop_path, dataset, obj_id)
        mesh.metadata["name"] = str(object_names[obj_id])
        mesh_only[obj_id] = mesh
        meshes[obj_id] = (mesh, diameter_m)
    # An empty template_path keeps the rendered views and DINOv2 features on the GPU.
    # template_path 为空时，渲染图和 DINOv2 特征留在显存。
    if template_path and os.path.isfile(template_path):
        template = template_path
    else:
        template = onboard_meshes(mesh_only, template_path, device=device)
    print(
        "6d_localization",
        {
            "dataset": dataset,
            "scene_id": int(scene_id),
            "im_id": int(im_id),
            "obj_ids": obj_ids,
            "inst_count": inst_count,
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
    estimator = WAPREstimator(device=device)
    prepared = estimator.prepare_meshes([pair[0] for pair in meshes.values()])
    mesh_setup_seconds = estimator.last_mesh_setup_seconds
    pose_meshes = {obj_id: (mesh, pair[1]) for (obj_id, pair), mesh in zip(meshes.items(), prepared)}
    rgb, depth_m, K = load_bop_rgbd(bop_path, dataset, scene_id, im_id)
    print("rgb", tuple(rgb.shape), "depth_m", tuple(depth_m.shape), "K", tuple(K.shape), flush=True)
    # Joint detection and pose estimation uses the declared per-category counts.
    # 联合检测与位姿估计使用预先给定的每类实例数量。
    # inst_count caps each category. It does not search for categories outside obj_ids.
    # inst_count 按类别截断。obj_ids 以外的类别不参与检索。
    poses, timing, detections = estimate_frame_many_categories_many_instances(
        estimator, detector, rgb, depth_m, K, pose_meshes,
        obj_ids=obj_ids,
        inst_count=inst_count,
    )
    print("det_ms", timing["total_ms"], "detections", len(detections), "instances", len(poses), flush=True)
    print("POSE_TIMING", timing, "model_setup_seconds", estimator.model_setup_seconds,
          "initial_mesh_setup_seconds", mesh_setup_seconds, flush=True)
    if visualize_det:
        preview_detections = []
        for pose in poses:
            for detection in detections:
                same_class = int(detection["obj_id"]) == int(pose["obj_id"])
                same_score = abs(float(detection["score_2d"]) - float(pose["score_2d"])) < 1.0e-6
                same_box = np.allclose(detection["bbox"], pose["bbox_xywh"], atol=1.0e-4)
                if same_class and same_score and same_box:
                    preview_detections.append(detection)
                    break
        if len(preview_detections) != len(poses):
            raise RuntimeError("Could not match each retained pose to its 2D detection")
        preview_detections = sorted(preview_detections, key=lambda row: -float(row["score_2d"]))[:preview_top_k]
        print("2d_preview", len(preview_detections), "retained of", len(detections), "raw detections", flush=True)
        visualize_2d_detection(rgb, preview_detections, names=object_names, image=True, filename="6d_localization_det.jpg")
    for pose in poses:
        print(
            {
                "obj_id": pose["obj_id"],
                "name": pose.get("name", ""),
                "score_2d": pose["score_2d"],
                "score_6d": pose["score_6d"],
                "t_m": np.asarray(pose["t_m"]).reshape(3).tolist(),
            },
            flush=True,
        )
    rows = []
    for pose in poses:
        row = dict(pose)
        row["mesh"] = meshes[int(pose["obj_id"])][0]
        rows.append(row)
    visualize_6d_pose(rgb, rows, K, filename="6d_localization.jpg")
