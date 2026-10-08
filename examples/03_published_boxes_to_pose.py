# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 03. Previous: examples/02_one_category_one_instance.py. Next: examples/04_6d_localization.py
# 示例 03。上一例：examples/02_one_category_one_instance.py。下一例：examples/04_6d_localization.py
# Published BOP 2D boxes feed one pose batch, without WAPRDet2D.
# 已公布的 BOP 2D 框进入一次位姿估计，不调用 WAPRDet2D。
# Page: docs/data.html#bop
# 页面：docs/data.html#bop
# Run: python examples/03_published_boxes_to_pose.py
# 运行：python examples/03_published_boxes_to_pose.py
# Boxes and masks come from that json. One call estimates every kept instance: WAPREstimator.estimate_many_categories_many_instances.
# 框和 mask 来自那份 json。一次调用估计全部留下的实例：WAPREstimator.estimate_many_categories_many_instances。
# This is not a loop of estimate_one_category_one_instance, and it does not call WAPRDet2D.
# 这不是逐条调用 estimate_one_category_one_instance，也不调用 WAPRDet2D。
# obj_ids chooses the classes. max_per_class caps how many poses each class returns.
# obj_ids 选择类别。max_per_class 限制每个类别返回几条位姿。
# frames lists the images. Each item is (scene_id, im_id).
# frames 列出图像。每一项是 (scene_id, im_id)。
import os
import sys
import time
import urllib.request

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
release_dir = os.path.dirname(script_dir)
sys.path.insert(0, release_dir)

from wapr import WAPREstimator
from wapr import recipe
from wapr.estimator import mask_from_bbox
from wapr.bop import keep_top_per_class, load_bop_rgbd, load_mesh_m, load_detections
from wapr.download_assets import check_and_fetch_pack
from wapr.resources import samples_dir, outputs_dir, cache_dir
from wapr.view import visualize_6d_pose


# LM-O files BOP republishes as default 2D detections.
# These are detection json files, not detector weights.
# LM-O 上 BOP 另行公布的默认 2D 检测。这是检测结果 json，不是检测器权重。
# cnos-fastsam: CNOS with FastSAM, method 370, zero-shot. Boxes come from the mask.
# cnos-fastsam：CNOS 配 FastSAM，方法页 370，zero-shot。框由 mask 得到。
bop_det_lmo = {
    "cnos-fastsam": {
        "file": "cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json",
        "url": (
            "https://huggingface.co/datasets/bop-benchmark/bop_extra/resolve/main/"
            "default_detections/classic_bop23_model_based_unseen/cnos-fastsam/"
            "cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json"
        ),
        "method_page": "https://bop.felk.cvut.cz/method_info/370/",
        "submission": "https://bop.felk.cvut.cz/sub_info/4003/",
    },
}


def fetch_bop_det(method):
    """Fetch the selected published BOP detection JSON when absent.

    缺失时获取所选方法已公布的 BOP 检测 JSON。
    """
    if method not in bop_det_lmo:
        known = ", ".join(sorted(bop_det_lmo))
        raise SystemExit("det_method must be one of: %s" % known)
    entry = bop_det_lmo[method]
    folder = os.path.join(cache_dir(), "bop_det")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, entry["file"])
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        print("bop_det present", method, path, flush=True)
        return path
    print("bop_det download", method, entry["submission"], flush=True)
    partial = path + ".partial"
    urllib.request.urlretrieve(entry["url"], partial)
    os.replace(partial, path)
    print("bop_det saved", path, os.path.getsize(path), flush=True)
    return path


if __name__ == "__main__":
    recipe.visualize_path = outputs_dir(__file__)
    # bop_path is the dataset root. Empty uses the one-frame pack under samples/bop/.
    # bop_path 是数据集根目录。为空时使用 samples/bop/ 里的单帧示例。
    # By default, use the included single-frame excerpt. Set use_full_published_json
    # to True to fetch the full official detection file into cache/bop_det/.
    # 默认使用随包提供的单帧摘录。设 use_full_published_json 为 True 才将完整官方
    # 检测文件下载到 cache/bop_det/。
    # frames must be an image that file contains. Image 1 of scene 2 is not in these files.
    # frames 必须是该文件所包含的图像。场景 2 第 1 帧不在这些文件中。
    # obj_ids is the eight LM-O objects. max_per_class keeps one mask per object.
    # score_thr is 0. The extra boxes stay in the file; they are not cut by a score floor.
    # obj_ids 是 LM-O 的八个物体。max_per_class 为每个物体保留一条 mask。
    # score_thr 是 0。其余检测框仍保留在文件中，并非由分数阈值剔除。
    bop_path = ""
    dataset = "lmo"
    det_method = "cnos-fastsam"
    use_full_published_json = False
    frames = [(2, 307)]
    obj_ids = [1, 5, 6, 8, 9, 10, 11, 12]
    max_per_class = {1: 1, 5: 1, 6: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1}
    device = "cuda:0"
    score_thr = 0.0
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
    # Explicitly keep this lesson's pose overlay enabled.
    # 显式开启本课的位姿叠图。
    recipe.visualize = True
    if not bop_path:
        check_and_fetch_pack(dataset)
        bop_path = os.path.join(samples_dir(), "bop")
    if use_full_published_json:
        det_json = fetch_bop_det(det_method)
    else:
        det_json = os.path.join(bop_path, dataset, "cnos-fastsam_scene2_im307.json")
    if not os.path.isabs(det_json):
        det_json = os.path.join(release_dir, det_json)
    if not os.path.isfile(det_json):
        raise FileNotFoundError(det_json)
    # Read published 2D evidence, then select the requested rows for each frame.
    # 读取已公布的 2D 检测，再按帧选出本实验请求的实例。
    print("bop_det", det_method, det_json, bop_det_lmo[det_method]["method_page"], flush=True)
    detections = load_detections(det_json, score_thr=score_thr)
    wanted = {int(obj_id) for obj_id in obj_ids}
    estimator = WAPREstimator(device=device)
    meshes = {}
    warmed_shapes = set()
    for scene_id, im_id in frames:
        mesh_setup_seconds = 0.0
        rgb, depth_m, K = load_bop_rgbd(bop_path, dataset, scene_id, im_id)
        print(
            "frame", int(scene_id), int(im_id),
            "rgb", tuple(rgb.shape),
            "depth_m", tuple(depth_m.shape),
            "K", tuple(K.shape),
            flush=True,
        )
        records = [
            record for record in detections.get((int(scene_id), int(im_id)), [])
            if int(record["obj_id"]) in wanted
        ]
        kept = keep_top_per_class(records, max_per_class)
        print("kept", len(kept), "of", len(records), "obj_ids", sorted(wanted), flush=True)
        # Pair each retained mask with its own metric CAD mesh for one pose batch.
        # 为每条保留的掩码配上对应的米制 CAD 网格，再组成一次位姿批处理。
        instances = []
        for record in kept:
            obj_id = int(record["obj_id"])
            if obj_id not in meshes:
                source, diameter_m = load_mesh_m(bop_path, dataset, obj_id)
                meshes[obj_id] = (estimator.prepare_meshes([source])[0], diameter_m)
                mesh_setup_seconds += estimator.last_mesh_setup_seconds
            mesh, diameter_m = meshes[obj_id]
            mask = record["mask"]
            if mask is None:
                mask = mask_from_bbox(record["bbox_xywh"], *depth_m.shape[:2])
            mesh.metadata["name"] = object_names.get(obj_id, mesh.metadata.get("name", "obj_%06d" % obj_id))
            instances.append({
                "mesh": mesh,
                "diameter_m": diameter_m,
                "mask": mask,
                "obj_id": obj_id,
                "name": mesh.metadata["name"],
                "bbox_xywh": record["bbox_xywh"],
                "score_2d": float(record["score_2d"]),
            })
        # One forward for every kept instance. The masks are already known, and each instance keeps its own mesh.
        # 一次前向估计全部留下的实例。mask 已经有了，每个实例用自己的网格。
        pose_instances = instances
        shape = (tuple(depth_m.shape), tuple(estimator.renderer.cached_mesh_id(item["mesh"]) for item in pose_instances))
        warmup_seconds = 0.0
        if pose_instances and shape not in warmed_shapes:
            warmup_seconds = estimator.warmup_pose(rgb, depth_m, K, pose_instances)
            warmed_shapes.add(shape)
        # Synchronize the complete batch; exclude mesh setup, warmup and drawing.
        # 同步完整批量调用；网格准备、预热和绘图均在预热后的计时外。
        torch.cuda.synchronize(estimator.device)
        pose_started = time.perf_counter()
        poses = estimator.estimate_many_categories_many_instances(rgb, depth_m, K, pose_instances)
        torch.cuda.synchronize(estimator.device)
        pose_hot_seconds = time.perf_counter() - pose_started
        print("POSE_TIMING", {"model_setup_seconds": estimator.model_setup_seconds,
              "mesh_setup_seconds": mesh_setup_seconds, "warmup_seconds": warmup_seconds,
              "pose_hot_seconds": pose_hot_seconds, "instances": len(instances)}, flush=True)
        rows = []
        for record, pose in zip(instances, poses):
            print(
                {
                    "scene_id": int(scene_id),
                    "im_id": int(im_id),
                    "obj_id": int(record["obj_id"]),
                    "score_2d": float(record["score_2d"]),
                    "score_6d": float(pose["score_6d"]),
                    "t_m": np.asarray(pose["t_m"]).reshape(3).tolist(),
                },
                flush=True,
            )
            rows.append({
                "obj_id": int(record["obj_id"]),
                "name": record["name"],
                "mesh": record["mesh"],
                "pose_4x4": pose["pose_4x4"],
                "bbox_xywh": record["bbox_xywh"],
                "score_2d": float(record["score_2d"]),
                "score_6d": float(pose["score_6d"]),
            })
        # Red is the estimated 3D box. Blue is the published 2D box. The file is under recipe.visualize_path.
        # 红色是估计的三维框。蓝色是已公布的二维框。文件写在 recipe.visualize_path 下。
        visualize_6d_pose(
            rgb, rows, K,
            filename="published_boxes_%06d_%06d.jpg" % (int(scene_id), int(im_id)),
        )
