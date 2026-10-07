# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 11. Previous: examples/10_robi_zigzag.py. Next: examples/12_cross_scene_pose.py
# 示例 11。上一例：examples/10_robi_zigzag.py。下一例：examples/12_cross_scene_pose.py
# Reconstruction example 11. / 重建示例 11。
# Reconstruct and align one object in one RGB-D frame, using a point or sentence.
# 通过点或句子提示，在一帧 RGB-D 中重建并对齐一个物体。
# python examples/11_reconstruct_object.py
#
# Stages, in order:
#   mask, mesh, isotropic scale, visible dimensions, DINOv2 pose, RoMa shape, output.
# 阶段顺序：掩码、网格、各向同性尺度、可观测尺寸、DINOv2 位姿、RoMa 形状更新、输出。
# Step modules keep the functions, in examples/11_reconstruct_object/.
# This entry imports stage functions; incompatible SAM stacks use an isolated worker.
# 入口导入阶段函数；基础环境不兼容 SAM 时，仅该阶段使用独立工作进程。
# Language prompts and observation-only RoMa are stages in this same entry.
# 语言提示与仅用观测的 RoMa 均为本入口的内部阶段。
# Predicted meshes go under the resource cache's outputs/reconstruct_object/<object>/prediction/.
# 预测模型写入资源缓存中的 outputs/reconstruct_object/<object>/prediction/。
# Page assets are not rewritten. / 不改写页面现成资源。
import hashlib
import json
import os
import sys

import numpy as np

RELEASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Stages live in the folder with this script's name.
# 阶段在和本脚本同名的文件夹里。
CASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "11_reconstruct_object")
if CASE_DIR not in sys.path:
    sys.path.insert(0, CASE_DIR)
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)

from wapr.resources import resource_root

from step01_point_mask import (  # noqa: E402
    DATA_ROOT,
    align_scales,
    depth_extent_m,
    fit_axis_scales,
    mesh_diameter_m,
    reconstruct_mesh,
    save_viewer_mesh,
    scale_mesh,
    segment_point,
)
from step02_bake_mesh import apply_bake_budget, texture_config  # noqa: E402
from step03_box_lengths import (  # noqa: E402
    constrain_to_box,
    geometry_only,
    run_boxes,
)
from step04_dino_pose import select_pose_result  # noqa: E402
from roma_shape import run_rounds  # noqa: E402
from pose_frame_align import load_viewer_mesh  # noqa: E402

# Source RGB-D, intrinsics in pixels, and the output object name.
# 源图 RGB-D、像素内参，以及输出物体名。
OBJECT_NAME = "cracker"
FRAME_RGB = os.path.join(DATA_ROOT, "cracker_box_reorient", "rgb", "1579908945563078571.png")
FRAME_DEPTH = os.path.join(DATA_ROOT, "cracker_box_reorient", "depth", "1579908945563078571.png")
FRAME_K = os.path.join(DATA_ROOT, "cracker_box_reorient", "cam_K.txt")
# Raw depth values are millimeters. / 原始深度值单位为毫米。
DEPTH_UNIT_M = 0.001
# One prompt. Empty SENTENCE is the point. A sentence is the other form.
# 一个提示。SENTENCE 为空就是点。一句话是另一种形式。
# Two prompt forms share the same downstream stages. / 两种提示形式共享后续阶段。
SENTENCE = ""
# unipose: full UV bake, then the three UniPose9D edges. Published side lengths.
# outline: vertex colors, no bake, then the mask outline. Same size step as step 01.
# unipose：完整 UV 烘焙，再用 UniPose9D 的三条边。宣传页上的边长走这条。
# outline：保留顶点色供 DINO／展示，不烘焙，再贴 mask 轮廓；尺寸步骤与第 01 阶段相同。
# WAPR/SAPR/WBPS use the shared POSE_GEOMETRY_ONLY default in step01_point_mask.
# WAPR/SAPR/WBPS 使用 step01_point_mask 中共享的 POSE_GEOMETRY_ONLY 默认值。
SIZE_PATH_POINT = "unipose"
SIZE_PATH_SENTENCE = "outline"
SIZE_PATH = SIZE_PATH_SENTENCE if SENTENCE else SIZE_PATH_POINT
APPLY_ROMA = True

# 仅用观测在最终固定位姿下更新形状；参考模型诊断独立运行。
MAX_ROUNDS = 3
STOP_DELTA = 0.02
MIN_MATCHES = 400
DELTA_SCALE_BOUNDS = (0.66, 1.44)
TOTAL_SCALE_BOUNDS = (0.70, 1.30)
MATCH_PAD_PX = 36
MIN_CERT = 0.25
device = "cuda:0"
# Installed recipes share writable reconstruction outputs with example 12.
# 已安装配方与示例 12 共用可写的重建输出目录。
OUT_DIR = os.path.join(resource_root(), "outputs", "reconstruct_object")
POINTS_PATH = os.path.join(CASE_DIR, "selected_points.json")
with open(POINTS_PATH, "r", encoding="utf-8") as stream:
    saved_points = json.load(stream)
# A stored pixel selection is the prompt; no annotated pose is projected at runtime.
# 保存的像素点就是提示；运行时不从标注位姿投影生成提示。
CLICK_UV = tuple(saved_points["cracker_box_reorient"]["build"])

def query_mask(rgb, row):
    """
    # Return one SAM 2 mask from a sentence or a saved click.

    ## Args

        - rgb: (H, W, 3) uint8 RGB. It is not None.
        - row: contains click_uv in pixels. A nonempty SENTENCE uses Qwen and SAM 2 instead.

    ## Returns

        - The return is (H, W) uint8, 0 or 1.
        - It is not None.

    ---

    # 用句子或保存的点击点返回一张 SAM 2 掩码。

    ## 参数

        - rgb: (H, W, 3) uint8 RGB。不是 None。
        - row: 包含像素点击点 click_uv；SENTENCE 非空时使用千问定位与 SAM 2。

    ## 返回

        - 返回值是 (H, W) uint8，取值 0 或 1。
        - 不是 None。

"""
    if SENTENCE:
        import importlib

        language_case = importlib.import_module("language_prompt")
        box, reply = language_case.qwen_box(rgb, SENTENCE)
        print("sentence", SENTENCE, "box", [round(float(value), 1) for value in box], flush=True)
        print("reply", reply, flush=True)
        return language_case.mask_from_box(rgb, box)
    click = row["click_uv"]
    if click is not None:
        print("segment", click, flush=True)
        return segment_point(rgb, click)
    raise ValueError("Save a click or set SENTENCE / 请保存点击点或设置 SENTENCE；不读取历史叠加图掩码")


def build_mesh(rgb, mask, name):
    """
    # Return the vertex-color mesh or the full UV bake.

    ## Args

        - rgb: (H, W, 3) uint8 RGB. It is not None.
        - mask: (H, W) uint8, 0 or 1. It is not None.
        - name: the object name passed to texture_config. It is not None. SIZE_PATH "outline" ignores it and skips the bake. Any other SIZE_PATH than "unipose" raises ValueError.

    ## Returns

        - The return is a trimesh.
        - It is not None.
        - The SAM3D session is deleted before the return on the bake path.

    ---

    # 返回顶点色网格，或完整的 UV 烘焙网格。

    ## 参数

        - rgb: (H, W, 3) uint8 RGB。不是 None。
        - mask: (H, W) uint8，取值 0 或 1。不是 None。
        - name: 传给 texture_config 的物体名。不是 None。SIZE_PATH 为 "outline" 时忽略它，并且不烘焙。SIZE_PATH 既不是 "outline" 也不是 "unipose" 时抛出 ValueError。

    ## 返回

        - 返回值是 trimesh。
        - 不是 None。
        - 烘焙路径在返回前会放开 SAM3D 会话。

"""
    import torch

    # Use the same native compatibility boundary as the package bootstrap.
    # 与包内准备入口使用相同原生兼容边界；不在旧位姿 Torch 中导入 SAM 烘焙模块。
    supported_pair = (torch.__version__.split("+", 1)[0] == "2.5.1"
                      and torch.version.cuda in ("11.8", "12.1", "12.4"))
    if sys.version_info[:2] >= (3, 12) or sys.version_info[:2] < (3, 9) or not supported_pair:
        from wapr.sam3d_isolated import reconstruct_example
        return reconstruct_example(rgb, mask, CASE_DIR, device, SIZE_PATH, name)

    if SIZE_PATH == "outline":
        mesh = reconstruct_mesh(rgb, mask)
        print("mesh outline", len(mesh.vertices), "verts", flush=True)
        return mesh
    if SIZE_PATH != "unipose":
        raise ValueError(SIZE_PATH)
    from fast_sam3d.session import Sam3dSession

    apply_bake_budget()
    session = Sam3dSession(device=device, compile_model=False).load()
    output = session.reconstruct(rgb, mask, cfg=texture_config(name), allow_fallback=False)
    mesh = output["mesh"]
    # SAM3D has its own backend; its first-call profile is not WAPR hot timing.
    # SAM3D 使用自己的后端；首调用剖析不作为 WAPR 预热后推理统计。
    print("mesh bake", len(mesh.vertices), "verts", flush=True)
    del session
    torch.cuda.empty_cache()
    return mesh


def scale_log(rows):
    """
    # Return the scale search without the mesh arrays.

    ## Args

        - rows: the list of dicts from align_scales. It is not None. Each item has factor, diameter_m in meters, score_6d, and mask_iou.

    ## Returns

        - The return is a list of dicts with those four numbers, rounded.
        - It is not None.
        - Mesh arrays are omitted.

    ---

    # 返回不含网格数组的尺度搜索记录。

    ## 参数

        - rows: align_scales 返回的字典列表。不是 None。每一项含 factor、diameter_m（米）、score_6d 和 mask_iou。

    ## 返回

        - 返回值是含这四个数的字典列表，已四舍五入。
        - 不是 None。
        - 网格数组不写入。

"""
    logged = []
    for item in rows:
        logged.append({
            "factor": item["factor"],
            "diameter_m": round(float(item["diameter_m"]), 6),
            "score_6d": round(float(item["score_6d"]), 4),
            "mask_iou": round(float(item["mask_iou"]), 4),
        })
    return logged


def choose_size(estimator, rgb, depth_m, mask, k, mesh, row):
    """
    # Choose the isotropic scale, then the box edges or the outline.

    ## Args

        - estimator: a WAPREstimator. It is not None.
        - rgb: (H, W, 3) uint8 RGB. It is not None.
        - depth_m: (H, W) depth in meters. It is not None.
        - mask: (H, W). Nonzero pixels are the object. It is not None.
        - k: (3, 3) camera intrinsics in pixels. It is not None.
        - mesh: the trimesh from build_mesh. It is not None. The real mesh is not an argument.
        - row: unipose reads box_rotation (3, 3), box_translation (3,) meters and box_edge_m (3,) meters; it writes box_factors (3,) unitless.

    ## Returns

        - The return has four fields, and none is None.
        - metric: the sized trimesh, vertices in meters.
        - pose: (4, 4) scale-search pose, translation in meters; DINOv2 selects the final pose afterward.
        - report: a dict.
        - isotropic: the mesh before the three box edges. The outline path returns the same mesh for metric and isotropic.

    ---

    # 先选定各向同性尺度，再走框的三条边或走轮廓。

    ## 参数

        - estimator: WAPREstimator。不是 None。
        - rgb: (H, W, 3) uint8 RGB。不是 None。
        - depth_m: (H, W) 深度，单位米。不是 None。
        - mask: (H, W)。非零像素是物体。不是 None。
        - k: (3, 3) 相机内参，单位像素。不是 None。
        - mesh: build_mesh 返回的 trimesh。不是 None。真实网格不是参数。
        - row: unipose 读取 box_rotation（3, 3）、box_translation（3，米）与 box_edge_m（3，米），写入无量纲 box_factors。

    ## 返回

        - 返回值有四项，都不是 None。
        - metric: 定好尺寸的 trimesh，顶点单位米。
        - pose: （4, 4）尺度搜索位姿，平移单位米；随后由 DINOv2 选择最终位姿。
        - report: 字典。
        - isotropic: 三条边施加之前的网格。轮廓路径的 metric 和 isotropic 是同一份网格。

"""
    extent_m = depth_extent_m(depth_m, mask, k)
    source = mesh if SIZE_PATH == "outline" else geometry_only(mesh)
    best, scale_rows = align_scales(estimator, rgb, depth_m, mask, k, source, extent_m)
    pose = np.asarray(best["pose_4x4"], dtype=np.float64)
    logged = scale_log(scale_rows)
    if SIZE_PATH == "outline":
        adjusted, aspect = fit_axis_scales(best["mesh"], pose, depth_m, mask, k)
        report = {
            "size_path": "outline",
            "depth_extent_m": extent_m,
            "scale_factor": best["factor"],
            "score_6d": float(best["score_6d"]),
            "axis_scales": [round(float(value), 4) for value in aspect["scales"]],
            "observed_axes": [int(axis) for axis in aspect["observed_axes"]],
            "scale_rows": logged,
            "pose_timing": estimator.last_scale_timing,
        }
        return adjusted, pose, report, adjusted
    base_m = mesh_diameter_m(mesh.vertices)
    isotropic = scale_mesh(mesh, float(best["diameter_m"]) / base_m)
    vertices, factors, _posed_extent = constrain_to_box(
        isotropic.vertices,
        pose,
        row["box_rotation"],
        row["box_translation"],
        row["box_edge_m"],
    )
    metric = isotropic.copy()
    metric.vertices = vertices
    row["box_factors"] = np.asarray(factors, dtype=np.float64)
    report = {
        "size_path": "unipose",
        "depth_extent_m": extent_m,
        "scale_factor": best["factor"],
        "score_6d": float(best["score_6d"]),
        "box_factors": [round(float(value), 4) for value in factors],
        "box_edge_mm": [round(float(value) * 1000.0, 1) for value in row["box_edge_m"]],
        "scale_rows": logged,
        "pose_timing": estimator.last_scale_timing,
    }
    return metric, pose, report, isotropic




if __name__ == "__main__":
    # Run the script stages directly in the entry block.
    # 在入口块中直接执行脚本各阶段。
    import json
    import cv2
    import torch
    from wapr.estimator import WAPREstimator

    # Fetch the selected built-in sequence; preserve custom frame paths.
    # 获取选中的内置序列；保留用户自定义帧路径。
    from wapr.resources import samples_dir
    if DATA_ROOT == os.path.join(samples_dir(), "YCBInEOAT"):
        sequence = os.path.basename(os.path.dirname(FRAME_K))
        if sequence in ("cracker_box_reorient", "mustard_easy_00_02", "sugar_box1"):
            from wapr.source_setup import prepare_ycbineoat
            prepare_ycbineoat(sequence)

    # Select one recorded object and obtain its source RGB-D frame and mask.
    # 选择一个已记录物体，读取其源帧 RGB-D 与目标掩码。
    for path in (FRAME_RGB, FRAME_DEPTH, FRAME_K):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    bgr = cv2.imread(FRAME_RGB, cv2.IMREAD_COLOR)
    raw_depth = cv2.imread(FRAME_DEPTH, cv2.IMREAD_UNCHANGED)
    if bgr is None or raw_depth is None or raw_depth.ndim != 2:
        raise ValueError("Could not read RGB-D / 无法读取 RGB-D 输入")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_m = raw_depth.astype(np.float32) * DEPTH_UNIT_M
    k = np.loadtxt(FRAME_K, dtype=np.float64).reshape(3, 3)
    if depth_m.shape != rgb.shape[:2] or not np.isfinite(k).all():
        raise ValueError("Invalid RGB-D grid or K / RGB-D 网格或内参无效")
    row = {"name": OBJECT_NAME, "rgb": FRAME_RGB, "click_uv": CLICK_UV}
    print("INPUT", FRAME_RGB, "SIZE_PATH", SIZE_PATH, "RoMa", APPLY_ROMA, flush=True)
    if not SENTENCE and (CLICK_UV is None or not (0 <= CLICK_UV[0] < rgb.shape[1] and 0 <= CLICK_UV[1] < rgb.shape[0])):
        raise ValueError("Save a point inside this image / 请保存当前图像内的点击点")
    mask = query_mask(rgb, row)
    if mask.shape != depth_m.shape or not np.any(mask):
        raise ValueError("SAM 2 mask is empty or mismatched / SAM 2 掩码为空或尺寸不符")
    print("mask", int(mask.sum()), "px", flush=True)
    # The generated mesh has no metric scale until the size path below is applied.
    # 生成网格尚无米制尺寸，下面的尺寸路径会为其定标。
    mesh = build_mesh(rgb, mask, row["name"])

    # The UniPose9D route estimates a metric box from this same RGB-D observation.
    # UniPose9D 路径从同一帧 RGB-D 估计米制包围盒。
    if SIZE_PATH == "unipose":
        prepared = dict(row)
        prepared["rgb_u8"] = rgb
        prepared["depth_m"] = depth_m
        prepared["mask"] = mask > 0
        prepared["k"] = k
        run_boxes([prepared])
        row["box_rotation"] = prepared["box_rotation"]
        row["box_translation"] = prepared["box_translation"]
        row["box_edge_m"] = prepared["box_edge_m"]

    estimator = WAPREstimator(device=device)
    # Choose metric dimensions, then select the final source-image pose.
    # 先确定米制尺寸，再选择最终的源图位姿。
    metric, pose, size_report, isotropic = choose_size(
        estimator, rgb, depth_m, mask, k, mesh, row,
    )
    selected = select_pose_result(estimator, rgb, depth_m, mask, k, metric)
    scale_pose = pose.copy()
    pose = np.asarray(selected["pose_4x4"], dtype=np.float64)
    within_score = selected["within_group"]
    cosine = selected["dino_cosine"]
    hyp_index = selected["hypothesis"]
    print(
        "selected pose", "hyp", hyp_index,
        "within", round(within_score, 3),
        "dino", round(cosine, 4),
        flush=True,
    )

    object_dir = os.path.join(OUT_DIR, row["name"])
    prediction_dir = os.path.join(object_dir, "prediction")
    os.makedirs(prediction_dir, exist_ok=True)
    # RoMa changes only this source-image mesh; keep the DINOv2-selected pose fixed.
    # RoMa 只更新源图网格；DINOv2 选定的位姿保持固定。
    del estimator
    torch.cuda.empty_cache()
    roma_report = {"enabled": APPLY_ROMA, "rounds": []}
    if APPLY_ROMA:
        metric, roma_report = run_rounds(
            metric, pose, k, rgb, mask, device, prediction_dir,
            MAX_ROUNDS, STOP_DELTA, MIN_MATCHES, DELTA_SCALE_BOUNDS,
            TOTAL_SCALE_BOUNDS, MATCH_PAD_PX, MIN_CERT,
        )
    mask_path = os.path.join(prediction_dir, "mask.png")
    if not cv2.imwrite(mask_path, (mask > 0).astype(np.uint8) * 255):
        raise OSError(mask_path)
    # Save the final mesh in its original object frame with the matching camera pose.
    # 保存原物体坐标系下的最终网格及配套的物体到相机位姿。
    save_viewer_mesh(metric, os.path.join(prediction_dir, "mesh"))
    prediction_report = {
        "pose_4x4": np.asarray(pose, dtype=np.float64).tolist(),
        "unit": "meters",
        "coordinate_frame": "reconstructed_object",
        "pose_convention": "object_to_camera",
        "mask_source": "qwen_box_sam2" if SENTENCE else "sam2_point",
        "reference_cad_used": False,
        "annotated_pose_used": False,
        "size_path": SIZE_PATH,
        "mask_file": "mask.png",
        # Paths are relative to the release root; hashes still identify the inputs.
        # 路径相对于发布根目录，哈希仍标识实际输入文件。
        "source_path_base": "release_root",
        "source_rgb": os.path.relpath(FRAME_RGB, RELEASE_DIR),
        "source_depth": os.path.relpath(FRAME_DEPTH, RELEASE_DIR),
        "source_k": os.path.relpath(FRAME_K, RELEASE_DIR),
        "depth_unit_m": DEPTH_UNIT_M,
    }
    for label, path in (("rgb", FRAME_RGB), ("depth", FRAME_DEPTH), ("k", FRAME_K), ("mask", mask_path)):
        with open(path, "rb") as stream:
            prediction_report[label + "_sha256"] = hashlib.sha256(stream.read()).hexdigest()
    np.save(os.path.join(prediction_dir, "pose.npy"), pose)
    # Overlay uses the saved mesh, including any export simplification.
    # 叠图使用实际保存的网格，包含导出时可能发生的简化。
    from step01_point_mask import project_silhouette
    saved_mesh = load_viewer_mesh(os.path.join(prediction_dir, "mesh"))
    silhouette, overlay = project_silhouette(rgb, saved_mesh, pose, k)
    overlay_path = os.path.join(prediction_dir, "overlay.png")
    if not cv2.imwrite(overlay_path, overlay):
        raise OSError(overlay_path)
    report = {
        "object": row["name"],
        "prediction_mesh": os.path.relpath(os.path.join(prediction_dir, "mesh"), OUT_DIR),
        "reference_cad_used": False,
        "annotated_pose_used": False,
        "sentence": SENTENCE,
        "size": size_report,
        "selected_pose": selected,
        "selected_pose_scores_stage": "before_roma_shape_update",
        "pose_4x4": pose.tolist(),
        "scale_search_pose_4x4": np.asarray(scale_pose).tolist(),
        "unit": "meters",
        "coordinate_frame": "reconstructed_object",
        "pose_convention": "object_to_camera",
        "roma": roma_report,
        "output_mask_iou": float(np.logical_and(silhouette > 0, mask > 0).sum()
                                 / max(1, np.logical_or(silhouette > 0, mask > 0).sum())),
    }
    # save_viewer_mesh reserves stem.json for the binary mesh layout.
    # Keep the experiment report separate so the generated mesh remains loadable.
    # save_viewer_mesh 使用 stem.json 记录二进制网格布局；实验报告单独保存，避免覆盖网格元数据。
    prediction_report.update(report)
    prediction_report["prompt"] = {"sentence": SENTENCE} if SENTENCE else {"click_uv": list(row["click_uv"])}
    with open(os.path.join(prediction_dir, "provenance.json"), "w", encoding="utf-8") as stream:
        json.dump(prediction_report, stream, indent=2, ensure_ascii=False)
    with open(os.path.join(object_dir, "report.json"), "w", encoding="utf-8") as stream:
        json.dump(prediction_report, stream, indent=2, ensure_ascii=False)
    print("REPORT", os.path.join(object_dir, "report.json"), flush=True)
    torch.cuda.empty_cache()
