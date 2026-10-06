# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 06. Previous: examples/05_bop_6d_detection.py. Next: examples/07_write_bop_pose_csv.py
# 示例 06。上一例：examples/05_bop_6d_detection.py。下一例：examples/07_write_bop_pose_csv.py
# Custom RGB-D folder layout, demonstrated with a downloaded LM-O frame.
# 自定义 RGB-D 目录格式，默认用下载的 LM-O 单帧演示。
# python examples/06_custom_scene.py
# 运行：python examples/06_custom_scene.py
# File names and fields: https://wangyulin-seu.github.io/WAPR/docs/data-own.html
# 文件名和字段：https://wangyulin-seu.github.io/WAPR/docs/data-own.html?lang=zh
# depth and meshes are meters after this loader. K is pixels.
# 这个加载器返回的深度和网格是米。K 是像素。
import json
import os
import shutil
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
release_dir = os.path.dirname(script_dir)
sys.path.insert(0, release_dir)

from wapr import WAPREstimator
from wapr import recipe
from wapr.det2d import WAPRDet2D, onboard_meshes
from wapr.download_assets import check_and_fetch_pack
from wapr.frame import estimate_frame_many_categories_many_instances
from wapr.scene_files import load_scene
from wapr.view import save_pose_view, visualize_2d_detection


if __name__ == "__main__":
    print("example 06  previous examples/05_bop_6d_detection.py  next examples/07_write_bop_pose_csv.py", flush=True)
    # The default is an adapter of LM-O scene 2, image 307, with eight existing meshes.
    # Replace scene_dir with your own folder; paths are relative to the release root.
    # For the wedge capture shown in Docs, supply its frame and mesh separately.
    # 默认使用 LM-O 场景 2 第 307 帧的目录格式适配样例，引用八份已有网格。
    # 将 scene_dir 换成自己的目录；相对路径从 release 根目录计算。
    # Docs 展示的三角块实拍案例需要另行提供其图像和网格。
    # template_path empty keeps the rendered views and DINOv2 features on the GPU.
    # Set a path only when the next run should load that file instead of rendering again.
    # A different mesh at an existing path is refused. Choose a new path.
    # template_path 为空时，渲染图和 DINOv2 特征留在显存，不写磁盘。
    # 只有要给下次运行加载时才填路径。已有路径上的网格变了会拒绝写入，需要换一个新路径。
    scene_dir = "samples/own_lmo"
    template_path = ""
    device = "cuda:0"
    det_backend = "trt"
    # These switches write the 2D and 6D previews independently of inference.
    # 这两个开关分别控制二维与六维预览图，不改变推理结果。
    visualize_det = True
    visualize_pose = True
    # Show the most informative candidates in images; inference keeps every result.
    # 图片只展示分数较高的候选；推理结果仍保留全部实例。
    preview_top_k = 12
    if not scene_dir:
        raise SystemExit(
            "Set scene_dir in examples/06_custom_scene.py.\n"
            "The folder needs rgb.png or rgb.jpg, depth.png or depth.npy,\n"
            "camera.json, and objects.json.\n"
            "在 examples/06_custom_scene.py 里填 scene_dir。\n"
            "目录里要有 rgb.png 或 rgb.jpg、depth.png 或 depth.npy、camera.json、objects.json。"
        )
    if not os.path.isabs(scene_dir):
        scene_dir = os.path.join(release_dir, scene_dir)
    # Create the default folder from the public LM-O sample, without copying CADs.
    # 从公开 LM-O 样例生成默认目录，网格继续引用原文件，不重复复制。
    if scene_dir == os.path.join(release_dir, "samples", "own_lmo") and not os.path.exists(scene_dir):
        check_and_fetch_pack("lmo")
        bop_dir = os.path.join(release_dir, "samples", "bop", "lmo")
        frame_dir = os.path.join(bop_dir, "test", "000002")
        with open(os.path.join(frame_dir, "scene_camera.json"), "r") as stream:
            camera = json.load(stream)["307"]
        with open(os.path.join(bop_dir, "models", "models_info.json"), "r") as stream:
            model_info = json.load(stream)
        camera_K = camera["cam_K"]
        custom_camera = {
            "fx": camera_K[0], "fy": camera_K[4],
            "cx": camera_K[2], "cy": camera_K[5],
            "depth_scale": camera["depth_scale"],
        }
        # Preserve the eight LM-O categories and original millimeter mesh units.
        # 保留八个 LM-O 类别以及原始毫米网格单位。
        model_names = {1: "ape", 5: "can", 6: "cat", 8: "driller", 9: "duck", 10: "eggbox", 11: "glue", 12: "holepuncher"}
        custom_objects = {"mesh_unit": "mm", "objects": []}
        for obj_id, name in model_names.items():
            custom_objects["objects"].append({
                "id": obj_id, "name": name,
                "file": "../bop/lmo/models/obj_%06d.ply" % obj_id,
                "diameter_mm": model_info[str(obj_id)]["diameter"],
            })
        os.makedirs(scene_dir)
        shutil.copyfile(os.path.join(frame_dir, "rgb", "000307.png"), os.path.join(scene_dir, "rgb.png"))
        shutil.copyfile(os.path.join(frame_dir, "depth", "000307.png"), os.path.join(scene_dir, "depth.png"))
        with open(os.path.join(scene_dir, "camera.json"), "w") as stream:
            json.dump(custom_camera, stream, indent=2)
        with open(os.path.join(scene_dir, "objects.json"), "w") as stream:
            json.dump(custom_objects, stream, indent=2)
        print("CUSTOM_SCENE_PREPARED", scene_dir, "LM-O scene 2 frame 307", flush=True)
    # The four-file scene contract resolves RGB, metric depth, K, meshes, and names.
    # 四文件场景约定解析 RGB、米制深度、K、网格与类别名称。
    rgb, depth_m, K, meshes, names = load_scene(scene_dir)
    print("rgb", tuple(rgb.shape), "depth_m", tuple(depth_m.shape), "K", tuple(K.shape), "classes", names, flush=True)
    print(
        "custom_scene",
        {
            "scene_dir": scene_dir,
            "objects": names,
            "template": template_path if template_path else "gpu",
            "n_view": recipe.n_view,
            "n_inplane": recipe.n_inplane,
            "wapr_iters": recipe.wapr_iters,
            "sapr_iters": recipe.sapr_iters,
            "det_backend": det_backend,
            "device": device,
            "visualize": recipe.visualize,
            "visualize_det": visualize_det,
            "visualize_pose": visualize_pose,
            "visualize_path": recipe.visualize_path,
        },
        flush=True,
    )
    mesh_only = {obj_id: pair[0] for obj_id, pair in meshes.items()}
    if template_path:
        onboard_meshes(mesh_only, template_path, device=device)
        template = template_path
    else:
        template = onboard_meshes(mesh_only, "", device=device)
    detector = WAPRDet2D(template, device=device, backend=det_backend)
    estimator = WAPREstimator(device=device)
    prepared = estimator.prepare_meshes([pair[0] for pair in meshes.values()])
    mesh_setup_seconds = estimator.last_mesh_setup_seconds
    pose_meshes = {obj_id: (mesh, pair[1]) for (obj_id, pair), mesh in zip(meshes.items(), prepared)}
    # One frame call runs detection followed by pose refinement for every match.
    # 一次帧级调用先检测，再为每条匹配结果修正位姿。
    poses, timing, detections = estimate_frame_many_categories_many_instances(estimator, detector, rgb, depth_m, K, pose_meshes)
    print("POSE_TIMING", timing, "model_setup_seconds", estimator.model_setup_seconds,
          "initial_mesh_setup_seconds", mesh_setup_seconds, flush=True)
    print("det_ms", timing["total_ms"], "detections", len(detections), "instances", len(poses), flush=True)
    if visualize_det:
        preview_detections = sorted(detections, key=lambda row: -float(row["score_2d"]))[:preview_top_k]
        print("2d_preview", len(preview_detections), "of", len(detections), flush=True)
        visualize_2d_detection(rgb, preview_detections, names=names, image=True, filename="custom_scene_det.jpg")
    rows = []
    for pose in poses:
        row = dict(pose)
        row["mesh"] = meshes[int(pose["obj_id"])][0]
        print(
            {
                "obj_id": pose["obj_id"],
                "name": pose.get("name", names.get(int(pose["obj_id"]), "")),
                "score_2d": pose["score_2d"],
                "score_6d": pose["score_6d"],
            },
            flush=True,
        )
        rows.append(row)
    # Use detection confidence in both images so they describe the same candidates.
    # 两张预览都按检测分数选取，便于对应同一批候选。
    preview_poses = sorted(rows, key=lambda row: -float(row["score_2d"]))[:preview_top_k]
    print("6d_preview", len(preview_poses), "of", len(rows), flush=True)
    if visualize_pose:
        pose_path = recipe.resolve_visualize_path("custom_scene.jpg")
        save_pose_view(rgb, preview_poses, pose_path, K=K, names=names)
