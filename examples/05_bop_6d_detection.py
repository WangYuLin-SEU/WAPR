# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 05. Previous: examples/04_6d_localization.py. Next: examples/06_custom_scene.py
# 示例 05。上一例：examples/04_6d_localization.py。下一例：examples/06_custom_scene.py
# BOP challenge, 6D detection. Many categories, many instances, count unknown, one BOP frame. python examples/05_bop_6d_detection.py
# BOP 挑战，6D 检测。多类别、多实例，数量事先不知道，BOP 的一帧。运行：python examples/05_bop_6d_detection.py
# Page: docs/pose.html#det
# 页面：docs/pose.html#det
# The detector is WAPRDet2D. Hypothesis counts and update counts stay in wapr/recipe.py.
# 检测器是 WAPRDet2D。候选数量和更新次数仍在 wapr/recipe.py。
import json
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
from wapr.bop import load_bop_rgbd, load_mesh_m, model_ids
from wapr.download_assets import check_and_fetch_pack
from wapr.resources import samples_dir
from wapr.frame import estimate_frame_many_categories_many_instances
from wapr.view import visualize_6d_pose, visualize_2d_detection


if __name__ == "__main__":
    print("example 05  previous examples/04_6d_localization.py  next examples/06_custom_scene.py", flush=True)
    # bop_path is the external dataset root.
    # An empty template_path keeps the rendered views and DINOv2 features on the GPU.
    # Set a path, such as outputs/cache/det2d/lmo.pt, only when the next run should load that file.
    # bop_path 是外部数据集根目录。
    # template_path 为空时，渲染图和 DINOv2 特征留在显存，不写磁盘。
    # 只有要给下次运行加载时，才填路径，例如 outputs/cache/det2d/lmo.pt。
    # det_backend is the detector. The pose backend stays in wapr/recipe.py.
    # det_backend 是检测器。姿态后端仍在 wapr/recipe.py。
    bop_path = ""
    dataset = "lmo"
    scene_id = 2
    im_id = 1
    template_path = ""
    device = "cuda:0"
    det_backend = "trt"
    # visualize_det writes the 2D box and the mask. recipe.visualize gates the 6D box and the contour.
    # visualize_det 写出二维框和 mask。recipe.visualize 控制 6D 框和轮廓。
    visualize_det = True
    # Keep all results in inference; cap only each image preview by its own score.
    # 推理保留全部结果；两张插图分别按各自分数限制显示数量。
    preview_top_k = 12
    # obj_id -> the string drawn on the image. An empty dict uses obj_000001.
    # obj_id 到图上文字。空字典时使用 obj_000001。
    object_names = {}
    # An empty bop_path uses the one-frame pack. It does not fetch the full test set.
    # That pack's models/ has the eight LM-O meshes, ids 1, 5, 6, 8, 9, 10, 11, 12.
    # bop_path 为空时用这单帧示例，不下载完整测试集。
    # 这个示例包的 models/ 是 LM-O 的八个网格，编号 1、5、6、8、9、10、11、12。
    if not bop_path:
        check_and_fetch_pack(dataset)
        bop_path = os.path.join(samples_dir(), "bop")
    # Build or reuse a bank for all available CAD classes; no instance count is supplied.
    # 为可用的全部 CAD 类别建立或复用模板库；不预设实例数量。
    # Views and features stay on the GPU unless template_path names a file.
    # 没有填写 template_path 时，渲染图和特征留在显存。
    if template_path and os.path.isfile(template_path):
        with open(template_path + ".json", "r") as stream:
            obj_ids = [int(obj_id) for obj_id in json.load(stream)["obj_ids"]]
        template = template_path
    else:
        mesh_only = {}
        for obj_id in model_ids(bop_path, dataset):
            mesh, _diameter_m = load_mesh_m(bop_path, dataset, obj_id)
            mesh_only[obj_id] = mesh
        if not mesh_only:
            raise SystemExit("no obj_*.ply under %s" % os.path.join(bop_path, dataset, "models"))
        template = onboard_meshes(mesh_only, template_path, device=device)
        if template_path:
            with open(template_path + ".json", "r") as stream:
                obj_ids = [int(obj_id) for obj_id in json.load(stream)["obj_ids"]]
        else:
            obj_ids = [int(obj_id) for obj_id in template["obj_ids"].tolist()]
    print(
        "bop_6d_detection",
        {
            "dataset": dataset,
            "scene_id": int(scene_id),
            "im_id": int(im_id),
            "template": template_path if template_path else "gpu",
            "objects": obj_ids,
            "n_view": recipe.n_view,
            "n_inplane": recipe.n_inplane,
            "wapr_iters": recipe.wapr_iters,
            "sapr_iters": recipe.sapr_iters,
            "det_backend": det_backend,
            "device": device,
            "visualize": recipe.visualize,
            "visualize_det": visualize_det,
            "visualize_path": recipe.visualize_path,
            "object_names": object_names,
        },
        flush=True,
    )
    detector = WAPRDet2D(template, device=device, backend=det_backend)
    meshes = {}
    for obj_id in obj_ids:
        mesh, diameter_m = load_mesh_m(bop_path, dataset, obj_id)
        if obj_id in object_names:
            mesh.metadata["name"] = str(object_names[obj_id])
        meshes[obj_id] = (mesh, diameter_m)
    estimator = WAPREstimator(device=device)
    prepared = estimator.prepare_meshes([pair[0] for pair in meshes.values()])
    mesh_setup_seconds = estimator.last_mesh_setup_seconds
    pose_meshes = {obj_id: (mesh, pair[1]) for (obj_id, pair), mesh in zip(meshes.items(), prepared)}
    rgb, depth_m, K = load_bop_rgbd(bop_path, dataset, scene_id, im_id)
    print("rgb", tuple(rgb.shape), "depth_m", tuple(depth_m.shape), "K", tuple(K.shape), flush=True)
    # This call discovers classes and instances before refining their 6D poses.
    # 这次调用先发现类别与实例，再修正各实例的 6D 位姿。
    poses, timing, detections = estimate_frame_many_categories_many_instances(estimator, detector, rgb, depth_m, K, pose_meshes)
    print("POSE_TIMING", timing, "model_setup_seconds", estimator.model_setup_seconds,
          "initial_mesh_setup_seconds", mesh_setup_seconds, flush=True)
    print("det_ms", timing["total_ms"], "detections", len(detections), "instances", len(poses), flush=True)
    if visualize_det:
        preview_detections = sorted(detections, key=lambda row: -float(row["score_2d"]))[:preview_top_k]
        print("2d_preview", len(preview_detections), "of", len(detections), flush=True)
        visualize_2d_detection(rgb, preview_detections, names=object_names, image=True, filename="6d_detection_det.jpg")
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
    preview_poses = sorted(rows, key=lambda row: -float(row["score_6d"]))[:preview_top_k]
    print("6d_preview", len(preview_poses), "of", len(rows), flush=True)
    visualize_6d_pose(rgb, preview_poses, K, filename="6d_detection.jpg")
