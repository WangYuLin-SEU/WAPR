# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Stage 03, called by examples/11_reconstruct_object.py; the entry measures box lengths alone.
# 第 03 段，由 examples/11_reconstruct_object.py 调用；入口块单独算边长。
# Stage 03 in the combined example; its standalone main is an independent box study.
# 综合示例的第 03 阶段；单独运行时是独立的包围盒实验。
# Isotropic WAPR size, then the three UniPose9D edge lengths. python examples/11_reconstruct_object/step03_box_lengths.py
# 先用 WAPR 定一个各向同性尺寸，再用 UniPose9D 的三条边长。运行：python examples/11_reconstruct_object/step03_box_lengths.py
"""Constrain a generated mesh with a WAPR pose and a UniPose9D box.

用 WAPR 位姿和 UniPose9D 的框，约束生成的网格。

WAPR supplies the camera pose of the generated vertices. UniPose9D supplies
the box center, the box axes, and the three edge lengths. Vertices are scaled
in that box frame, then mapped back through the same pose. The real CAD mesh
is not used to choose the scale. It is only the comparison after the fact.
WAPR 给出生成顶点的相机位姿。UniPose9D 给出框的中心、轴和三条边长。
顶点在这个框的坐标系里缩放，再沿同一个位姿变回去。真实 CAD 不参与选缩放，
只在做完之后用来比较。
"""

import json
import os
import sys

import cv2
import numpy as np


DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.dirname(os.path.dirname(DEMO_DIR))
# Resolve the upstream checkout under the release, independent of cwd.
# 上游源码由发布根目录定位，不依赖启动目录。
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)
from wapr.resources import cache_dir, samples_dir, outputs_dir

UNIPOSE_ROOT = os.path.join(cache_dir(), "sources", "UniPose9D")
UNIPOSE_INFER = os.path.join(UNIPOSE_ROOT, "infer")
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)
if UNIPOSE_INFER not in sys.path:
    sys.path.insert(0, UNIPOSE_INFER)

from pose_frame_align import (  # noqa: E402
    CRACKER_PAIR_YAW,
    draw_pair,
    extents_mm,
    into_cad_frame,
    load_viewer_mesh,
)
from step01_point_mask import (  # noqa: E402
    align_scales,
    depth_extent_m,
    mesh_diameter_m,
    project_silhouette,
    read_rgb_depth,
    save_viewer_mesh,
    scale_mesh,
    silhouette_iou,
)


DATA_ROOT = os.environ.get("RECON_DATA_ROOT", os.path.join(samples_dir(), "ycbineoat"))
DATA_ROOT = os.path.abspath(os.path.join(RELEASE_DIR, DATA_ROOT))
PAGE = os.path.join(outputs_dir("11_reconstruct_object"), "stages", "step01_point_mask")
MESH_DIR = os.path.join(outputs_dir("11_reconstruct_object"), "stages", "short_texture")
OUT_DIR = os.path.join(outputs_dir("11_reconstruct_object"), "stages", "box_constrain")
UNIPOSE_CKPT = os.path.join(UNIPOSE_ROOT, "checkpoints", "last.ckpt")
UNIPOSE_CFG = os.path.join(UNIPOSE_ROOT, "checkpoints", "config.yaml")
DEVICE = "cuda:0"

# A generated axis that is almost flat is left alone. Factors outside this
# range are clipped and reported. CAD does not set either number.
# 生成网格上几乎扁掉的轴保持不动。超出这个范围的倍率会被截断并写进记录。
# 这两个数都不是 CAD 定的。
MIN_EXTENT_M = 0.005
SCALE_MIN = 0.50
SCALE_MAX = 2.00
CHAMFER_SAMPLES = 4000


def first_rgb(sequence):
    """
    # Return the path of the first PNG in one YCBInEOAT sequence.

    ## Args

        - sequence: the sequence folder name under DATA_ROOT. It is not None.

    ## Returns

        - The return is one filesystem path string.
        - It is not None.
        - Names: sorted, and the first name that ends with .png is used.

    ---

    # 返回一条 YCBInEOAT 序列里第一张 PNG 的路径。

    ## 参数

        - sequence: DATA_ROOT 下的序列文件夹名。不是 None。

    ## 返回

        - 返回值是一个文件系统路径字符串。
        - 不是 None。
        - 文件名排序后，取第一个以 .png 结尾的名字。

"""
    rgb_dir = os.path.join(DATA_ROOT, sequence, "rgb")
    name = sorted(name for name in os.listdir(rgb_dir) if name.endswith(".png"))[0]
    return os.path.join(rgb_dir, name)


def mask_from_prompt(rgb, prompt_path):
    """
    # Return pixels that differ from the saved prompt overlay.

    ## Args

        - rgb: (H, W, 3) uint8 RGB. It is not None.
        - prompt_path: the path of the prompt image. It is not None. The image is read as BGR and compared with rgb converted to BGR.

    ## Returns

        - The return is (H, W) uint8, 0 or 1.
        - A channel sum above 8 is 1.
        - It is not None.

    ---

    # 返回和保存的提示叠图不一样的像素。

    ## 参数

        - rgb: (H, W, 3) uint8 RGB。不是 None。
        - prompt_path: 提示图路径。不是 None。图按 BGR 读入，再和转成 BGR 的 rgb 比较。

    ## 返回

        - 返回值是 (H, W) uint8，取值 0 或 1。
        - 通道差之和大于 8 的为 1。
        - 不是 None。

"""
    prompt = cv2.imread(prompt_path)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    diff = np.abs(bgr.astype(np.int16) - prompt.astype(np.int16)).sum(axis=2)
    return (diff > 8).astype(np.uint8)


def cases():
    """
    # Return the three showcase frames and their real-mesh stems.

    ## Args

        - There are no arguments.

    ## Returns

        - The return is a list of three dicts.
        - It is not None.
        - Each dict has name, rgb, prompt, cad, and k_sequence, all strings.
        - No field is None.

    ---

    # 返回宣传页上的三帧，以及各自真实网格的路径。

    ## 参数

        - 没有参数。

    ## 返回

        - 返回值是三个字典的列表。
        - 不是 None。
        - 每个字典含 name、rgb、prompt、cad、k_sequence，都是字符串。
        - 没有字段是 None。

"""
    return [
        {
            "name": "cracker",
            "rgb": first_rgb("cracker_box_reorient"),
            "prompt": os.path.join(PAGE, "prompt.png"),
            "cad": os.path.join(PAGE, "cad"),
            "k_sequence": "cracker_box_reorient",
        },
        {
            "name": "sugar",
            "rgb": os.path.join(DATA_ROOT, "sugar_box1", "rgb", "1579910587381007354.png"),
            "prompt": os.path.join(PAGE, "sugar_prompt.png"),
            "cad": os.path.join(PAGE, "sugar_cad"),
            "k_sequence": "sugar_box1",
        },
        {
            "name": "mustard",
            "rgb": first_rgb("mustard0"),
            "prompt": os.path.join(PAGE, "mustard_prompt.png"),
            "cad": os.path.join(PAGE, "mustard_cad"),
            "k_sequence": "mustard0",
        },
    ]


def geometry_only(mesh):
    """
    # Return a trimesh that keeps vertices and faces and drops the texture.

    ## Args

        - mesh: a trimesh. Vertices are meters. It is not None.

    ## Returns

        - The return is a new trimesh, vertices float64 meters, faces int64.
        - It is not None.
        - process: false, so the vertex order stays.

    ---

    # 返回只保留顶点和面、去掉贴图的 trimesh。

    ## 参数

        - mesh: trimesh。顶点单位米。不是 None。

    ## 返回

        - 返回值是一份新的 trimesh，顶点 float64、单位米，面为 int64。
        - 不是 None。
        - process: 假，顶点顺序保持不变。

"""
    import trimesh

    return trimesh.Trimesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )


def load_glb(path):
    """
    # Load one textured glb and return a single mesh.

    ## Args

        - path: the glb filesystem path. It is not None.

    ## Returns

        - The return is a trimesh.
        - It is not None.
        - A scene is concatenated into one mesh.
        - process: false.

    ---

    # 读入一个带贴图的 glb，并返回单独一个网格。

    ## 参数

        - path: glb 的文件系统路径。不是 None。

    ## 返回

        - 返回值是 trimesh。
        - 不是 None。
        - 场景会拼成一个网格。
        - process: 假。

"""
    import trimesh

    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded


def sorted_mm(values):
    """
    # Return the numbers sorted shortest first, rounded to 0.1.

    ## Args

        - values: a sequence of numbers. It is not None. This function does not convert units. Callers pass millimeters.

    ## Returns

        - The return is a list of floats.
        - It is not None.

    ---

    # 返回从小到大排序后的数，四舍五入到 0.1。

    ## 参数

        - values: 一串数。不是 None。这个函数不做单位换算。调用方传入的是毫米。

    ## 返回

        - 返回值是浮点数列表。
        - 不是 None。

"""
    return [round(float(value), 1) for value in sorted(values)]


def abs_sum(left, right):
    """
    # Return the sum of absolute differences, rounded to 0.1.

    ## Args

        - left: a sequence of numbers. It is not None. Callers pass millimeters.
        - right: a sequence of numbers, paired with left by zip. It is not None.

    ## Returns

        - The return is one float in the same unit as the inputs.
        - It is not None.

    ---

    # 返回绝对差之和，四舍五入到 0.1。

    ## 参数

        - left: 一串数。不是 None。调用方传入的是毫米。
        - right: 一串数，按 zip 和 left 配对。不是 None。

    ## 返回

        - 返回值是一个浮点，单位和输入相同。
        - 不是 None。

"""
    return round(sum(abs(a - b) for a, b in zip(left, right)), 1)


def constrain_to_box(vertices, pose_wapr, rotation_box, translation_box, edge_m):
    """
    # Scale vertices along the box axes and return the new vertices, factors, and extent.

    ## Args

        - vertices: (N, 3) float, meters, in the mesh frame. It is not None.
        - pose_wapr: (4, 4) float64. It maps a mesh vertex into the camera. Translation is meters. It is not None. The returned vertices still use this pose.
        - rotation_box: (3, 3) float64, the UniPose box axes. It is not None.
        - translation_box: (3,) float64, the box origin in the camera, meters. It is not None.
        - edge_m: (3,) float64, the three box edge lengths in meters. It is not None.

    ## Returns

        - The return has three fields, and none is None.
        - mesh_new: (N, 3) float64 meters in the original mesh frame.
        - factors: (3,) float64, unitless, clipped to SCALE_MIN and SCALE_MAX. An axis shorter than MIN_EXTENT_M keeps factor 1.
        - extent: (3,) float64 meters, the posed size before scaling.

    ---

    # 沿框的轴缩放顶点，返回新顶点、倍率和缩放前的尺寸。

    ## 参数

        - vertices: (N, 3) 浮点，米，网格坐标系。不是 None。
        - pose_wapr: (4, 4) float64。它把网格顶点变到相机。平移单位米。不是 None。返回的顶点仍用这个位姿。
        - rotation_box: (3, 3) float64，UniPose 框的轴。不是 None。
        - translation_box: (3,) float64，框原点在相机里的位置，单位米。不是 None。
        - edge_m: (3,) float64，框的三条边，单位米。不是 None。

    ## 返回

        - 返回值有三项，都不是 None。
        - mesh_new: (N, 3) float64，米，原始网格坐标系。
        - factors: (3,) float64，无量纲，截断在 SCALE_MIN 和 SCALE_MAX 之间。短于 MIN_EXTENT_M 的轴倍率保持 1。
        - extent: (3,) float64，米，缩放前摆好位姿的尺寸。

"""
    xyz = np.asarray(vertices, dtype=np.float64)
    hom = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    camera = (hom @ np.asarray(pose_wapr, dtype=np.float64).T)[:, :3]
    rotation_box = np.asarray(rotation_box, dtype=np.float64)
    translation_box = np.asarray(translation_box, dtype=np.float64).reshape(3)
    edge_m = np.asarray(edge_m, dtype=np.float64).reshape(3)
    local = (camera - translation_box.reshape(1, 3)) @ rotation_box
    low = local.min(axis=0)
    high = local.max(axis=0)
    extent = high - low
    center = 0.5 * (low + high)
    factors = np.ones(3, dtype=np.float64)
    for axis in range(3):
        if float(extent[axis]) < MIN_EXTENT_M:
            continue
        raw = float(edge_m[axis]) / float(extent[axis])
        factors[axis] = float(np.clip(raw, SCALE_MIN, SCALE_MAX))
    scaled = (local - center.reshape(1, 3)) * factors.reshape(1, 3)
    camera_new = scaled @ rotation_box.T + translation_box.reshape(1, 3)
    rotation = pose_wapr[:3, :3]
    translation = pose_wapr[:3, 3]
    mesh_new = (camera_new - translation.reshape(1, 3)) @ rotation
    return mesh_new, factors, extent


def with_vertex_colors(mesh):
    """
    # Return a copy whose vertices carry the texture color.

    ## Args

        - mesh: a trimesh. It is not None. A mesh with no UV or no image is returned unchanged.

    ## Returns

        - The return is a trimesh.
        - It is not None.
        - When a texture exists, visual is ColorVisuals with (N, 4) uint8 RGBA.
        - The input mesh is not modified.

    ---

    # 返回一份顶点上带贴图颜色的拷贝。

    ## 参数

        - mesh: trimesh。不是 None。没有 UV 或没有图像时，原样返回这份网格。

    ## 返回

        - 返回值是 trimesh。
        - 不是 None。
        - 有贴图时，visual 是 ColorVisuals，颜色为 (N, 4) uint8 RGBA。
        - 传入的网格不被修改。

"""
    import trimesh

    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    image = None
    if material is not None:
        image = getattr(material, "baseColorTexture", None)
        if image is None:
            image = getattr(material, "image", None)
    if uv is None or image is None:
        return mesh
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    uv = np.asarray(uv, dtype=np.float64)

    def sample(points):
        """
        # Sample the texture at UV points and return RGB.

        ## Args

            - points: (N, 2) float64 UV, x right, y up, both about 0–1. It is not None. y is flipped so 0 sits at the bottom of the image.

        ## Returns

            - The return is (N, 3) float64 RGB, about 0–255.
            - It is not None.

        ---

        # 在 UV 点上采样贴图，并返回 RGB。

        ## 参数

            - points: (N, 2) float64 UV，x 向右，y 向上，大约 0–1。不是 None。y 被翻转，所以 0 在图像下方。

        ## 返回

            - 返回值是 (N, 3) float64 RGB，大约 0–255。
            - 不是 None。

"""
        x = np.clip(np.rint(points[:, 0] * (width - 1)).astype(np.int32), 0, width - 1)
        y = np.clip(np.rint((1.0 - points[:, 1]) * (height - 1)).astype(np.int32), 0, height - 1)
        return rgb[y, x].astype(np.float64)

    faces = np.asarray(mesh.faces, dtype=np.int64)
    centroid = sample(uv[faces].mean(axis=1))
    usable = centroid.sum(axis=1) > 40.0
    totals = np.zeros((len(uv), 3), dtype=np.float64)
    weights = np.zeros(len(uv), dtype=np.float64)
    for face, color, keep in zip(faces, centroid, usable):
        if not keep:
            continue
        totals[face] += color
        weights[face] += 1.0
    colors = sample(uv)
    filled = weights > 0
    colors[filled] = totals[filled] / weights[filled, None]
    colors = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
    painted = mesh.copy()
    painted.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.concatenate([colors, alpha], axis=1),
    )
    return painted


def mean_nearest_mm(source, target):
    """
    # Return the symmetric mean nearest-point distance in millimeters.

    ## Args

        - source: a trimesh that can sample points. Vertices are meters. It is not None.
        - target: a trimesh that can sample points. Vertices are meters. It is not None.

    ## Returns

        - The return is one float, millimeters.
        - It is not None.
        - Each cloud uses at most CHAMFER_SAMPLES points.

    ---

    # 返回对称的平均最近点距离，单位毫米。

    ## 参数

        - source: 可以采样点的 trimesh。顶点单位米。不是 None。
        - target: 可以采样点的 trimesh。顶点单位米。不是 None。

    ## 返回

        - 返回值是一个浮点，毫米。
        - 不是 None。
        - 每团点最多用 CHAMFER_SAMPLES 个点。

"""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    left = np.asarray(source.sample(CHAMFER_SAMPLES, return_index=False), dtype=np.float64)
    right = np.asarray(target.sample(CHAMFER_SAMPLES, return_index=False), dtype=np.float64)
    if len(left) > CHAMFER_SAMPLES:
        left = left[rng.choice(len(left), CHAMFER_SAMPLES, replace=False)]
    if len(right) > CHAMFER_SAMPLES:
        right = right[rng.choice(len(right), CHAMFER_SAMPLES, replace=False)]
    left_to_right = cKDTree(right).query(left)[0]
    right_to_left = cKDTree(left).query(right)[0]
    return float(0.5 * (left_to_right.mean() + right_to_left.mean()) * 1000.0)


def run_boxes(rows):
    """
    # Run UniPose9D on each row and store the box on that row.

    ## Args

        - rows: a list of dicts. It is not None. Each dict needs rgb_u8 (H, W, 3) uint8 RGB, depth_m (H, W) meters, mask (H, W), k (3, 3) pixels, and name. The model is deleted before the return.

    ## Returns

        - Returns None.
        - The return is None.
        - Each row gains box_rotation (3, 3) float64, box_translation (3,) float64 meters, box_edge_m (3,) float64 meters, and box_inliers int.

    ---

    # 对每一行跑 UniPose9D，并把框写回该行。

    ## 参数

        - rows: 字典列表。不是 None。每个字典需要 rgb_u8（(H, W, 3) uint8 RGB）、depth_m（(H, W)，米）、mask（(H, W)）、k（(3, 3)，像素）和 name。返回前会放开模型。

    ## 返回

        - 返回 None。
        - 返回值是 None。
        - 每一行会多出 box_rotation（(3, 3) float64）、box_translation（(3,) float64，米）、box_edge_m（(3,) float64，米）和 box_inliers（整数）。

"""
    import torch
    from pathlib import Path
    # Keep UNIPOSE_INFER precedence; prepare optional dependencies on first use.
    # 保留 UNIPOSE_INFER 的源码优先级，仅在首次使用时准备可选依赖。
    from wapr.bootstrap import ensure_optional
    ensure_optional("unipose9d")
    from unipose9d_inference import estimate_pose, load_pose_model, set_seed

    set_seed(0)
    device = torch.device(DEVICE)
    checkpoint = UNIPOSE_CKPT
    configuration = UNIPOSE_CFG
    checkpoint_missing = not os.path.isfile(checkpoint)
    if not checkpoint_missing and checkpoint == os.path.join(UNIPOSE_ROOT, "checkpoints", "last.ckpt"):
        # Source snapshots contain an LFS pointer, not the model tensor file.
        # 源码快照可能只含 LFS 指针；保留该指针，从包内资源入口获取实际张量文件。
        with open(checkpoint, "rb") as stream:
            checkpoint_missing = stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1\n")
    if checkpoint_missing:
        if checkpoint != os.path.join(UNIPOSE_ROOT, "checkpoints", "last.ckpt"):
            raise FileNotFoundError("Custom UniPose checkpoint missing / 自定义 UniPose 权重缺失: " + checkpoint)
        # Prepare the pinned public weights only at this model-load boundary.
        # 仅在模型加载处准备固定公开权重；已有用户文件优先。
        from wapr.source_setup import prepare_unipose_weights
        prepared_weights = prepare_unipose_weights()
        checkpoint = prepared_weights["checkpoint"]
        if configuration == os.path.join(UNIPOSE_ROOT, "checkpoints", "config.yaml"):
            configuration = prepared_weights["config"]
    model, cfg = load_pose_model(Path(checkpoint), Path(configuration), device)
    image_size = int(cfg.get("image_size", 448))
    for row in rows:
        result = estimate_pose(
            row["rgb_u8"],
            row["depth_m"],
            row["mask"],
            row["k"],
            model,
            device,
            image_size=image_size,
            num_points=1024,
            num_tuples=1000,
            num_steps=32,
            num_repeats=10,
            ransac_hops=2,
            ransac_iterations=1000,
            scale_clustering=True,
        )
        row["box_rotation"] = np.asarray(result["rotation"], dtype=np.float64)
        row["box_translation"] = np.asarray(result["translation"], dtype=np.float64)
        row["box_edge_m"] = np.asarray(result["predicted_scale"], dtype=np.float64)
        row["box_inliers"] = int(result["inlier_count"])
        print(row["name"], "box_mm", sorted_mm(row["box_edge_m"] * 1000.0), flush=True)
    del model
    torch.cuda.empty_cache()


def redraw_pairs():
    """
    # Redraw the three comparison pictures from the saved meshes.

    ## Args

        - There are no arguments.

    ## Returns

        - Returns None.
        - The return is None.
        - Each picture is written under OUT_DIR.

    ---

    # 用已经写出的网格重画三张对照图。

    ## 参数

        - 没有参数。

    ## 返回

        - 返回 None。
        - 返回值是 None。
        - 每张图写到 OUT_DIR 下。

"""
    for row in cases():
        moved = load_viewer_mesh(os.path.join(OUT_DIR, row["name"]))
        cad = load_viewer_mesh(row["cad"])
        pair_yaw = CRACKER_PAIR_YAW if row["name"] == "cracker" else 0.0
        draw_pair(moved, cad, os.path.join(OUT_DIR, row["name"] + "_pair.png"), pair_yaw)
        print("pair", row["name"], flush=True)




if __name__ == "__main__":
    # Run the script stages directly in the entry block.
    # 在入口块中直接执行脚本各阶段。
    if os.environ.get("REDRAW_PAIR") == "1":
        redraw_pairs()
        raise SystemExit(0)
    import torch
    from wapr.estimator import WAPREstimator

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = cases()
    chosen = os.environ.get("BOX_ONLY", "")
    if chosen:
        rows = [row for row in rows if row["name"] == chosen]
    for row in rows:
        rgb, depth_m = read_rgb_depth(row["rgb"])
        row["rgb_u8"] = rgb
        row["depth_m"] = depth_m
        row["mask"] = mask_from_prompt(rgb, row["prompt"]) > 0
        row["k"] = np.loadtxt(
            os.path.join(DATA_ROOT, row["k_sequence"], "cam_K.txt"), dtype=np.float64,
        ).reshape(3, 3)
        row["mesh"] = load_glb(os.path.join(MESH_DIR, row["name"] + ".glb"))
        print(row["name"], "mask", int(row["mask"].sum()), "verts", len(row["mesh"].vertices), flush=True)
    run_boxes(rows)
    estimator = WAPREstimator(device=DEVICE)
    report = []
    for row in rows:
        mesh = row["mesh"]
        # Raw SAM 3D vertices are not meters. One isotropic size lets WAPR lock
        # the pose. The three box edges are applied after that pose, not before.
        # 原始 SAM 3D 顶点不是米。先用一个各向同性尺寸让 WAPR 锁住位姿。
        # 框的三条边在这个位姿之后再施加。
        extent_m = depth_extent_m(row["depth_m"], row["mask"], row["k"])
        best, _scale_rows = align_scales(
            estimator,
            row["rgb_u8"],
            row["depth_m"],
            row["mask"],
            row["k"],
            geometry_only(mesh),
            extent_m,
        )
        base_m = mesh_diameter_m(mesh.vertices)
        metric = scale_mesh(mesh, float(best["diameter_m"]) / base_m)
        pose = np.asarray(best["pose_4x4"], dtype=np.float64)
        before_mm = extents_mm(metric.vertices)
        constrained = metric.copy()
        new_vertices, factors, posed_extent_m = constrain_to_box(
            metric.vertices,
            pose,
            row["box_rotation"],
            row["box_translation"],
            row["box_edge_m"],
        )
        constrained.vertices = new_vertices
        cad = load_viewer_mesh(row["cad"])
        cad_diameter_m = mesh_diameter_m(cad.vertices)
        cad_pose_mesh = estimator.prepare_meshes([geometry_only(cad)])[0]
        estimator.warmup_pose(row["rgb_u8"], row["depth_m"], row["k"],
                              [{"mesh": cad_pose_mesh, "diameter_m": cad_diameter_m, "mask": row["mask"]}])
        cad_result = estimator.estimate_one_category_one_instance(
            row["rgb_u8"], row["depth_m"], row["k"], cad_pose_mesh, cad_diameter_m, mask=row["mask"],
        )
        pose_cad = np.asarray(cad_result["pose_4x4"], dtype=np.float64)
        moved, _relative = into_cad_frame(constrained, pose, pose_cad)
        before_moved, _before_relative = into_cad_frame(metric, pose, pose_cad)
        save_viewer_mesh(moved, os.path.join(OUT_DIR, row["name"]))
        _sil, overlay = project_silhouette(row["rgb_u8"], constrained, pose, row["k"])
        cv2.imwrite(os.path.join(OUT_DIR, row["name"] + "_overlay.png"), overlay)
        pair_yaw = CRACKER_PAIR_YAW if row["name"] == "cracker" else 0.0
        draw_pair(moved, cad, os.path.join(OUT_DIR, row["name"] + "_pair.png"), pair_yaw)
        after_mm = extents_mm(moved.vertices)
        cad_mm = extents_mm(cad.vertices)
        entry = {
            "name": row["name"],
            "wapr": round(float(best["score_6d"]), 3),
            "mask_iou": round(float(best["mask_iou"]), 3),
            "isotropic": float(best["factor"]),
            "cad_wapr": round(float(cad_result["score_6d"]), 3),
            "box_inliers": row["box_inliers"],
            "factors": [round(float(value), 3) for value in factors],
            "posed_extent_mm": [round(float(value) * 1000.0, 1) for value in posed_extent_m],
            "overlay_iou": round(silhouette_iou(
                project_silhouette(row["rgb_u8"], constrained, pose, row["k"])[0],
                row["mask"],
            ), 3),
            "box_sorted_mm": sorted_mm(row["box_edge_m"] * 1000.0),
            "before_sorted_mm": sorted_mm(before_mm),
            "after_sorted_mm": sorted_mm(after_mm),
            "cad_sorted_mm": sorted_mm(cad_mm),
            "before_err_mm": abs_sum(sorted_mm(before_mm), sorted_mm(cad_mm)),
            "after_err_mm": abs_sum(sorted_mm(after_mm), sorted_mm(cad_mm)),
            "chamfer_before_mm": round(mean_nearest_mm(before_moved, cad), 2),
            "chamfer_after_mm": round(mean_nearest_mm(moved, cad), 2),
        }
        report.append(entry)
        print(row["name"], json.dumps(entry), flush=True)
    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    del estimator
    torch.cuda.empty_cache()
    print("wrote", OUT_DIR, flush=True)
