# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 07. Previous: examples/06_custom_scene.py. Next: examples/08_ycbineoat_one_instance.py
# 示例 07。上一例：examples/06_custom_scene.py。下一例：examples/08_ycbineoat_one_instance.py
# Example 07. Estimate poses for BOP targets and write a BOP CSV.
# 示例 07。估计 BOP 目标帧中的位姿并输出 BOP CSV。
# Page: docs/pose.html#bop
# 页面：docs/pose.html#bop
# Run: python examples/07_write_bop_pose_csv.py
# 运行：python examples/07_write_bop_pose_csv.py
# Both "json" and "det2d" inputs batch the retained instances from each frame.
# 两种检测输入都对同帧保留的实例批量估计。
# detector "det2d": WAPRDet2D, then estimate_frame_many_categories_many_instances.
# detector 为 "det2d"：先 WAPRDet2D，再 estimate_frame_many_categories_many_instances。
# detector="json": batch retained detections; JSON-path NMS compares across categories.
# detector="json"：保留的检测批量估计；JSON 路径的 NMS 跨类别比较。
# detector "json" reads det_json instead.
# detector 为 "json" 时改读 det_json。
# Run: python examples/07_write_bop_pose_csv.py, after setting bop_path and dataset.
# 运行：python examples/07_write_bop_pose_csv.py，先填好 bop_path 和 dataset。
# csv translation is millimeters. The csv score column is score_6d.
# csv 里的平移是毫米。csv 的 score 列写入 score_6d。
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# This example sits in examples/, so the release root is its parent.
# 本示例在 examples/ 下，release 根目录是它的上一级。
sys.path.insert(0, str(ROOT))

from wapr import WAPREstimator
from wapr import recipe
from wapr.bop import load_bop_rgbd, load_mesh_m, load_detections, model_ids, target_frames, write_csv
from wapr.estimator import mask_from_bbox
from wapr.det2d import WAPRDet2D, onboard_meshes
from wapr.download_assets import check_and_fetch_pack
from wapr.frame import estimate_frame_many_categories_many_instances
from wapr.suppression import greedy_mask_nms_across_categories


if __name__ == "__main__":
    # detector "det2d" is the default. "json" reads det_json.
    # detector 为 "det2d" 是默认。"json" 读 det_json。
    # score_thr and iou_thresh apply only to "json". det2d rejection and NMS live in wapr/det2d.py.
    # score_thr 和 iou_thresh 只用于 "json"。det2d 的拒识和 NMS 在 wapr/det2d.py。
    # det_backend is the detector. The pose backend stays in wapr/recipe.py.
    # det_backend 是检测器。姿态后端仍在 wapr/recipe.py。
    # An empty template_path keeps the bank on the GPU for this process.
    # Set a path, such as outputs/cache/det2d/<dataset>.pt, only to load a file next time.
    # template_path 为空时，这次运行的库留在显存。
    # 只有下次要直接加载文件时，才填路径，例如 outputs/cache/det2d/<dataset>.pt。
    bop_path = ""
    dataset = ""
    detector = "det2d"
    template_path = ""
    det_backend = "trt"
    det_json = ""
    out_csv = "poses.csv"
    device = "cuda:0"
    score_thr = 0.0
    iou_thresh = 0.5
    # An empty bop_path fetches that dataset's one frame. It does not fetch the full test set.
    # bop_path 为空时只取该数据集的一帧，不下载完整测试集。
    if not dataset:
        raise SystemExit("set dataset")
    if not bop_path:
        check_and_fetch_pack(dataset)
        bop_path = str(ROOT / "samples" / "bop")
    # Select one reproducible BOP export route; each writes rows to the same CSV format.
    # 选择一条可复现的 BOP 导出路径；两条路径均写出相同格式的 CSV 行。
    if detector == "det2d":
        # 1. Load a saved CAD bank or build one from the selected dataset's models.
        # 1. 载入已保存的 CAD 模板库，或由所选数据集的网格建立模板库。
        if template_path:
            with open(template_path + ".json", "r") as stream:
                obj_ids = [int(obj_id) for obj_id in json.load(stream)["obj_ids"]]
            template = template_path
            meshes = {obj_id: load_mesh_m(bop_path, dataset, obj_id) for obj_id in obj_ids}
        else:
            found = model_ids(bop_path, dataset)
            meshes = {obj_id: load_mesh_m(bop_path, dataset, obj_id) for obj_id in found}
            template = onboard_meshes({obj_id: pair[0] for obj_id, pair in meshes.items()}, "", device=device)
            obj_ids = [int(obj_id) for obj_id in template["obj_ids"].tolist()]
        frames = target_frames(bop_path, dataset)
        print("bop_test", {"detector": "det2d", "dataset": dataset,
            "template": template_path if template_path else "gpu", "frames": len(frames),
            "objects": obj_ids, "n_view": recipe.n_view, "n_inplane": recipe.n_inplane,
            "wapr_iters": recipe.wapr_iters, "sapr_iters": recipe.sapr_iters,
            "det_backend": det_backend, "device": device}, flush=True)
        # 2. Load both models once; iterate frames, batching every detection of each image.
        # 2. 两个模型各载入一次；逐帧读取，每张图的全部检测共同进入位姿批处理。
        det2d = WAPRDet2D(template, device=device, backend=det_backend)
        estimator = WAPREstimator(device=device)
        prepared = estimator.prepare_meshes([pair[0] for pair in meshes.values()])
        print("POSE_PREPARATION", {"model_setup_seconds": estimator.model_setup_seconds,
                                  "mesh_setup_seconds": estimator.last_mesh_setup_seconds}, flush=True)
        pose_meshes = {obj_id: (mesh, pair[1]) for (obj_id, pair), mesh in zip(meshes.items(), prepared)}
        rows = []
        for scene_id, im_id in frames:
            rgb, depth_m, K = load_bop_rgbd(bop_path, dataset, scene_id, im_id)
            # The frame entry reports hot stages separately from setup and warmup.
            # 帧入口分别记录预热后运行阶段、资源准备与预热，CSV 不包含预热前的调用。
            # Sum measured detection/filter and pose segments, excluding the warmup gap.
            # 相加实测检测/过滤与位姿阶段，排除中间预热间隔。
            poses, timing, _detections = estimate_frame_many_categories_many_instances(
                estimator, det2d, rgb, depth_m, K, pose_meshes,
            )
            frame_time_s = timing["hot_frame_seconds"]
            print("bop_frame", scene_id, im_id, "instances", len(poses), "det_ms", timing["total_ms"], flush=True)
            # Record construction does not perform per-object inference.
            # 此循环只组织输出行，不逐物体运行推理。
            for pose in poses:
                rows.append({"scene_id": scene_id, "im_id": im_id, "obj_id": pose["obj_id"],
                    "score_6d": pose["score_6d"], "R": pose["R"], "t_m": pose["t_m"], "time": frame_time_s})
    elif detector == "json":
        if not det_json:
            raise SystemExit("set det_json")
        # 1. Read predicted boxes/masks and apply their 2D score gate.
        # 1. 读取预测框与掩码，应用它们的 2D 分数门槛。
        dets = load_detections(det_json, score_thr=score_thr)
        estimator = WAPREstimator(device=device)
        meshes = {}
        warmed_shapes = set()
        rows = []
        for (scene_id, im_id), recs in sorted(dets.items()):
            mesh_setup_seconds = 0.0
            rgb, depth_m, K = load_bop_rgbd(bop_path, dataset, scene_id, im_id)
            height, width = depth_m.shape[:2]
            # 2. Cross-category NMS compares supplied masks; missing masks use boxes.
            # 2. 跨类别 NMS 比较外部掩码；掩码缺失时用框填充比较区域。
            nms_masks = []
            for rec in recs:
                if rec["mask"] is None:
                    nms_masks.append(mask_from_bbox(rec["bbox_xywh"], height, width))
                else:
                    nms_masks.append(rec["mask"])
            keep = greedy_mask_nms_across_categories(
                np.stack(nms_masks, 0), [rec["score_2d"] for rec in recs], iou_thresh,
            )
            instances = []
            for index in keep:
                rec = recs[index]
                obj_id = int(rec["obj_id"])
                if obj_id not in meshes:
                    source, diameter_m = load_mesh_m(bop_path, dataset, obj_id)
                    meshes[obj_id] = (estimator.prepare_meshes([source])[0], diameter_m)
                    mesh_setup_seconds += estimator.last_mesh_setup_seconds
                mesh, diameter_m = meshes[obj_id]
                instances.append({"mesh": mesh, "diameter_m": diameter_m,
                    "bbox_xywh": rec["bbox_xywh"], "mask": rec["mask"]})
            # 3. One pose call handles every retained instance. This timer excludes
            # the published detector's runtime and the JSON input filter above.
            # 3. 全部保留实例调用一次位姿估计。此计时不含已公布检测器的运行时间
            # 及上方 JSON 输入过滤耗时。
            shape = (tuple(depth_m.shape), tuple((estimator.renderer.cached_mesh_id(item["mesh"]), item.get("mask") is not None) for item in instances))
            warmup_seconds = 0.0
            if instances and shape not in warmed_shapes:
                warmup_seconds = estimator.warmup_pose(rgb, depth_m, K, instances)
                warmed_shapes.add(shape)
            print("POSE_PREPARATION", estimator.model_setup_seconds,
                  mesh_setup_seconds, warmup_seconds, flush=True)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            outputs = estimator.estimate_many_categories_many_instances(rgb, depth_m, K, instances)
            torch.cuda.synchronize(device)
            frame_time_s = time.perf_counter() - t0
            for index, out in zip(keep, outputs):
                rows.append({"scene_id": scene_id, "im_id": im_id, "obj_id": int(recs[index]["obj_id"]),
                    "score_6d": out["score_6d"], "R": out["R"], "t_m": out["t_m"], "time": frame_time_s})
    else:
        raise ValueError("detector must be det2d or json")
    # 4. Convert translations to millimeters and write the shared BOP CSV format.
    # Every row shares its frame's hot stage seconds: detection/filter + pose
    # for det2d, pose only for JSON. Setup and warmup are excluded in both.
    # 4. 将平移换为毫米，写出统一 BOP CSV。同帧共享预热后运行阶段秒数：det2d 为
    # 检测/过滤加位姿，JSON 仅为位姿；两条路径均排除准备与预热。
    write_csv(out_csv, rows)
