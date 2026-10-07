# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Place a reconstruction into the real mesh frame with two WAPR poses.

用两次 WAPR 位姿，把重建网格放进真实网格的坐标系。

Both poses are estimated on the same photo and the same mask. One pose belongs
to the reconstruction. The other belongs to the real mesh. Their composition
is a rigid map. It does not change the reconstruction's size.
两个位姿都在同一张照片、同一个 mask 上估计。一个属于重建，一个属于真实网格。
它们合成一个刚体变换。重建的尺寸不变。
"""

import json
import os
import sys

import numpy as np


DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.dirname(os.path.dirname(DEMO_DIR))
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from step01_point_mask import mesh_diameter_m, read_rgb_depth, segment_point  # noqa: E402


DATA_ROOT = os.environ.get("RECON_DATA_ROOT", os.path.join(RELEASE_DIR, "datasets", "YCBInEOAT"))
DATA_ROOT = os.path.abspath(os.path.join(RELEASE_DIR, DATA_ROOT))
PAGE = os.path.join(RELEASE_DIR, "pages", "demo", "reconstruct")
LANG = os.path.join(RELEASE_DIR, "pages", "demo", "language")
# Staging copies. The page bins are replaced only after the previews are checked.
# 先写到这里。看过预览再替换页面上的网格。
OUT_DIR = os.path.join(RELEASE_DIR, "outputs", "reconstruction_stages", "frame_align")
DEVICE = "cuda:0"


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


def cam_k(sequence):
    """
    # Return the 3×3 camera matrix of one YCBInEOAT sequence.

    ## Args

        - sequence: the sequence folder name under DATA_ROOT. It is not None.

    ## Returns

        - The return is (3, 3) float64, pixel intrinsics.
        - It is not None.

    ---

    # 返回一条 YCBInEOAT 序列的 3×3 相机矩阵。

    ## 参数

        - sequence: DATA_ROOT 下的序列文件夹名。不是 None。

    ## 返回

        - 返回值是 (3, 3) float64，像素内参。
        - 不是 None。

"""
    return np.loadtxt(os.path.join(DATA_ROOT, sequence, "cam_K.txt"), dtype=np.float64).reshape(3, 3)


def load_viewer_mesh(stem):
    """
    # Read a page viewer bin and return the trimesh.

    ## Args

        - stem: the path without an extension. stem.json and stem.bin are read. It is not None.

    ## Returns

        - The return is a trimesh.
        - Vertices: float32 meters. It is not None. mode "uv" keeps UV and an RGB texture. Any other mode keeps (N, 4) uint8 vertex colors.

    ---

    # 读入页面上的查看器 bin，并返回 trimesh。

    ## 参数

        - stem: 不带后缀的路径。读取 stem.json 和 stem.bin。不是 None。

    ## 返回

        - 返回值是 trimesh。
        - 顶点是 float32，单位米。
        - 不是 None。
        - mode: "uv" 时保留 UV 和 RGB 贴图。其他 mode 保留 (N, 4) uint8 顶点色。

"""
    import trimesh
    from PIL import Image

    with open(stem + ".json", encoding="utf-8") as stream:
        meta = json.load(stream)
    raw = open(stem + ".bin", "rb").read()
    count = int(meta["vertices"])
    faces = int(meta["faces"])
    vertices = np.frombuffer(raw, dtype=np.float32, count=count * 3).reshape(count, 3).copy()
    triangles = np.frombuffer(raw, dtype=np.uint32, count=faces * 3, offset=count * 12).reshape(faces, 3).copy()
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    offset = count * 12 + faces * 12
    if meta.get("mode") == "uv":
        uv = np.frombuffer(raw, dtype=np.float32, count=count * 2, offset=offset).reshape(count, 2).copy()
        image = Image.open(os.path.join(os.path.dirname(stem), meta["texture"])).convert("RGB")
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=image)
    else:
        colors = np.frombuffer(raw, dtype=np.uint8, count=count * 3, offset=offset).reshape(count, 3).copy()
        alpha = np.full((count, 1), 255, dtype=np.uint8)
        mesh.visual.vertex_colors = np.concatenate([colors, alpha], axis=1)
    return mesh


def load_mask(path, shape):
    """
    # Read a saved mask and return a boolean image.

    ## Args

        - path: the grayscale mask path. It is not None.
        - shape: (height, width) in pixels. It is not None. A missing file or a different height and width raises RuntimeError.

    ## Returns

        - The return is (H, W) bool.
        - True: inside the mask. It is not None.

    ---

    # 读入保存的 mask，并返回布尔图。

    ## 参数

        - path: 灰度 mask 的路径。不是 None。
        - shape: 像素 (height, width)。不是 None。文件缺失，或高宽不一致，会抛出 RuntimeError。

    ## 返回

        - 返回值是 (H, W) bool。
        - True 在 mask 内。
        - 不是 None。

"""
    import cv2

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape[:2] != shape:
        raise RuntimeError("mask " + path)
    return mask > 0


def jobs():
    """
    # Return the four viewer pairs.

    ## Args

        - There are no arguments.

    ## Returns

        - The return is a list of four dicts: cracker, sugar, mustard, language.
        - It is not None.
        - click: (u, v) pixels or None.
        - mask_path: a path string or None. rgb, k, cad, before, and after are set on every row.
        - k: (3, 3) float64 pixels.

    ---

    # 返回页面上的四组对比。

    ## 参数

        - 没有参数。

    ## 返回

        - 返回值是四个字典的列表：饼干盒、糖盒、芥末瓶、语言提示。
        - 不是 None。
        - click: 像素 (u, v) 或 None。
        - mask_path: 路径字符串或 None。每一行都有 rgb、k、cad、before、after。
        - k: (3, 3) float64，像素。

"""
    cracker = first_rgb("cracker_box_reorient")
    sugar = first_rgb("sugar_box1")
    mustard = first_rgb("mustard0")
    return [
        {
            "name": "cracker",
            "rgb": cracker,
            "k": cam_k("cracker_box_reorient"),
            "mask_path": os.path.join(PAGE, "mask.png"),
            "click": None,
            "cad": os.path.join(PAGE, "cad"),
            "before": os.path.join(PAGE, "before"),
            "after": os.path.join(PAGE, "aligned"),
        },
        {
            "name": "sugar",
            "rgb": sugar,
            "k": cam_k("sugar_box1"),
            "mask_path": None,
            "click": (179, 410),
            "cad": os.path.join(PAGE, "sugar_cad"),
            "before": os.path.join(PAGE, "sugar_before"),
            "after": os.path.join(PAGE, "sugar"),
        },
        {
            "name": "mustard",
            "rgb": mustard,
            "k": cam_k("mustard0"),
            "mask_path": None,
            "click": (152, 314),
            "cad": os.path.join(PAGE, "mustard_cad"),
            "before": os.path.join(PAGE, "mustard_before"),
            "after": os.path.join(PAGE, "mustard"),
        },
        {
            "name": "language",
            "rgb": mustard,
            "k": cam_k("mustard0"),
            "mask_path": os.path.join(LANG, "mask.png"),
            "click": None,
            "cad": os.path.join(PAGE, "mustard_cad"),
            "before": os.path.join(LANG, "before"),
            "after": os.path.join(LANG, "after"),
        },
    ]


def collect_masks(rows):
    """
    # Attach a boolean mask and the RGB image to each row.

    ## Args

        - rows: a list of dicts from jobs. It is not None. A row with mask_path uses that file. A row with mask_path None uses click and SAM 2.

    ## Returns

        - Returns None.
        - The return is None.
        - Each row gains mask (H, W) bool and rgb_u8 (H, W, 3) uint8 RGB.
        - The SAM 2 cache is emptied before the return.

    ---

    # 给每一行挂上布尔 mask 和 RGB 图。

    ## 参数

        - rows: jobs 返回的字典列表。不是 None。有 mask_path 的行用那个文件。mask_path 为 None 的行用 click 和 SAM 2。

    ## 返回

        - 返回 None。
        - 返回值是 None。
        - 每一行会多出 mask（(H, W) bool）和 rgb_u8（(H, W, 3) uint8 RGB）。
        - 返回前会清空 SAM 2 占用的缓存。

"""
    import torch

    for row in rows:
        rgb, _depth = read_rgb_depth(row["rgb"])
        if row["mask_path"]:
            row["mask"] = load_mask(row["mask_path"], rgb.shape[:2])
        else:
            row["mask"] = segment_point(rgb, row["click"]) > 0
        row["rgb_u8"] = rgb
        print(row["name"], "mask", int(row["mask"].sum()), flush=True)
    torch.cuda.empty_cache()


def pose_of(estimator, rgb, depth_m, k, mask, mesh):
    """
    # Return the WAPR pose, its score, and the mesh diameter.

    ## Args

        - estimator: a WAPREstimator. It is not None.
        - rgb: (H, W, 3) uint8 RGB. It is not None.
        - depth_m: (H, W) depth in meters. It is not None.
        - k: (3, 3) camera intrinsics in pixels. It is not None.
        - mask: (H, W) bool. It is not None.
        - mesh: a trimesh, vertices in meters. It is not None.

    ## Returns

        - The return is pose (4, 4) float64 with translation in meters, score_6d as one float, and diameter_m as one float in meters.
        - None of the three is None.

    ---

    # 返回 WAPR 位姿、分数，以及网格直径。

    ## 参数

        - estimator: WAPREstimator。不是 None。
        - rgb: (H, W, 3) uint8 RGB。不是 None。
        - depth_m: (H, W) 深度，单位米。不是 None。
        - k: (3, 3) 相机内参，单位像素。不是 None。
        - mask: (H, W) bool。不是 None。
        - mesh: trimesh，顶点单位米。不是 None。

    ## 返回

        - 返回值是 pose（(4, 4) float64，平移单位米）、score_6d（一个浮点）和 diameter_m（一个浮点，米）。
        - 三者都不是 None。

"""
    diameter_m = mesh_diameter_m(mesh.vertices)
    prepared = estimator.prepare_meshes([mesh])[0]
    estimator.warmup_pose(rgb, depth_m, k, [{"mesh": prepared, "diameter_m": diameter_m, "mask": mask}])
    result = estimator.estimate_one_category_one_instance(rgb, depth_m, k, prepared, diameter_m, mask=mask)
    pose = np.asarray(result["pose_4x4"], dtype=np.float64)
    return pose, float(result["score_6d"]), diameter_m


def into_cad_frame(mesh, pose_mesh, pose_cad):
    """
    # Map these vertices into the real mesh frame and return the mesh and the rigid map.

    ## Args

        - mesh: a trimesh, vertices in meters. It is not None. The copy is moved; this mesh stays put.
        - pose_mesh: (4, 4) float64, camera from a reconstruction vertex. Translation is meters. It is not None.
        - pose_cad: (4, 4) float64, camera from a real-mesh vertex. Translation is meters. It is not None.

    ## Returns

        - The return is a trimesh copy, vertices in meters in the real mesh frame, and relative (4, 4) float64.
        - relative translation is meters.
        - Neither: None.

    ---

    # 把这些顶点变到真实网格坐标系，返回网格和刚体变换。

    ## 参数

        - mesh: trimesh，顶点单位米。不是 None。移动的是拷贝；这份网格留在原地。
        - pose_mesh: (4, 4) float64，由重建顶点得到相机坐标。平移单位米。不是 None。
        - pose_cad: (4, 4) float64，由真实网格顶点得到相机坐标。平移单位米。不是 None。

    ## 返回

        - 返回值是一份 trimesh 拷贝（顶点在真实网格坐标系，单位米）和 relative（(4, 4) float64）。
        - relative 的平移单位米。
        - 两者都不是 None。

"""
    relative = np.linalg.inv(pose_cad) @ pose_mesh
    xyz = np.asarray(mesh.vertices, dtype=np.float64)
    hom = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    moved = mesh.copy()
    moved.vertices = (hom @ relative.T)[:, :3]
    return moved, relative


def extents_mm(vertices):
    """
    # Return the axis-aligned side lengths in millimeters, x then y then z.

    ## Args

        - vertices: (N, 3), meters. It is not None.

    ## Returns

        - The return is a list of three floats, millimeters, rounded to 0.1.
        - It is not sorted.
        - It is not None.

    ---

    # 返回轴对齐边长，单位毫米，顺序是 x、y、z。

    ## 参数

        - vertices: (N, 3)，单位米。不是 None。

    ## 返回

        - 返回值是三个浮点数的列表，毫米，四舍五入到 0.1。
        - 不排序。
        - 不是 None。

"""
    span = np.asarray(vertices, dtype=np.float64)
    span = span.max(axis=0) - span.min(axis=0)
    return [round(float(value) * 1000.0, 1) for value in span]


def texture_rgb(mesh):
    """
    # Return the UV coordinates and the RGB texture, or None and None.

    ## Args

        - mesh: a trimesh. It is not None.

    ## Returns

        - The return is uv (N, 2) float64, about 0–1, and image (H, W, 3) uint8 RGB.
        - If UV or the image is missing, both are None.

    ---

    # 返回 UV 坐标和 RGB 贴图；缺任何一个则返回 None 和 None。

    ## 参数

        - mesh: trimesh。不是 None。

    ## 返回

        - 返回值是 uv（(N, 2) float64，大约 0–1）和 image（(H, W, 3) uint8 RGB）。
        - 缺 UV 或缺图像时，两者都是 None。

"""
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None) if visual is not None else None
    material = getattr(visual, "material", None) if visual is not None else None
    image = None
    if material is not None:
        image = getattr(material, "baseColorTexture", None)
        if image is None:
            image = getattr(material, "image", None)
    if uv is None or image is None:
        return None, None
    return np.asarray(uv, dtype=np.float64), np.asarray(image.convert("RGB"))


# Plain yellow body texels are not the printed front. These cuts are for that body.
# 纯黄瓶身纹素不是印着图案的正面。这几个界限是给这块瓶身用的。
BODY_RED_MIN = 150
BODY_GREEN_MIN = 140
BODY_BLUE_MAX = 90
BODY_DARK_SUM = 48
# A UV edge longer than this crosses two charts. The fill would cut through the gutter.
# UV 边长超过这个值就是跨了两块色块。填充会切到色块之间的黑边。
UV_SEAM_EDGE = 0.15
# Saturated yellow is the mustard bottle body, so label_yaw faces the printed label.
# The cracker's printed front is that yellow cracker art, so its pair adds half a turn.
# 饱和黄是芥末瓶身，所以 label_yaw 朝向印刷标签。
# 饼干盒的印刷正面就是这块黄饼干，所以对照图再转半圈。
CRACKER_PAIR_YAW = np.pi


def label_yaw(mesh):
    """
    # Return the yaw that turns the textured front toward the camera.

    ## Args

        - mesh: a trimesh, vertices in meters. It is not None.

    ## Returns

        - The return is one float, radians.
        - It is not None.
        - A mesh with no UV, or fewer than 30 front vertices, returns 35 degrees in radians.

    ---

    # 返回把贴了纹理的正面转到相机前的偏航角。

    ## 参数

        - mesh: trimesh，顶点单位米。不是 None。

    ## 返回

        - 返回值是一个浮点，弧度。
        - 不是 None。
        - 没有 UV，或正面顶点少于 30 个时，返回 35 度对应的弧度。

"""
    uv, image = texture_rgb(mesh)
    if uv is None:
        return np.deg2rad(35.0)
    height, width = image.shape[:2]
    x = np.clip(np.rint(uv[:, 0] * (width - 1)).astype(np.int32), 0, width - 1)
    y = np.clip(np.rint((1.0 - uv[:, 1]) * (height - 1)).astype(np.int32), 0, height - 1)
    color = image[y, x].astype(np.int16)
    yellow = (color[:, 0] > BODY_RED_MIN) & (color[:, 1] > BODY_GREEN_MIN) & (color[:, 2] < BODY_BLUE_MAX)
    dark = color.sum(axis=1) < BODY_DARK_SUM
    front = ~(yellow | dark)
    if int(front.sum()) < 30:
        return np.deg2rad(35.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    centroid = vertices[front].mean(axis=0) - vertices.mean(axis=0)
    return float(np.arctan2(-centroid[0], centroid[2]))


def paint_uv_face(canvas, xy, uv, image):
    """
    # Paste one textured triangle onto the canvas.

    ## Args

        - canvas: (H, W, 3) uint8 BGR. It is not None. Covered pixels are overwritten.
        - xy: (3, 2) screen pixels, x right and y down. It is not None.
        - uv: (3, 2) float, about 0–1, origin at the bottom of the texture. It is not None.
        - image: (H, W, 3) uint8 RGB. It is not None.

    ## Returns

        - Returns None.
        - The return is None.
        - A triangle with no area is left undrawn.

    ---

    # 把一个带纹理的三角形贴到画布上。

    ## 参数

        - canvas: (H, W, 3) uint8 BGR。不是 None。被覆盖的像素会被改写。
        - xy: (3, 2) 屏幕像素，x 向右，y 向下。不是 None。
        - uv: (3, 2) 浮点，大约 0–1，原点在贴图下方。不是 None。
        - image: (H, W, 3) uint8 RGB。不是 None。

    ## 返回

        - 返回 None。
        - 返回值是 None。
        - 没有面积的三角形不画。

"""
    import cv2

    height, width = image.shape[:2]
    src = np.stack([
        uv[:, 0] * float(width - 1),
        (1.0 - uv[:, 1]) * float(height - 1),
    ], axis=1).astype(np.float32)
    dst = np.asarray(xy, dtype=np.float32).reshape(3, 2)
    if abs(float(np.cross(src[1] - src[0], src[2] - src[0]))) < 1.0e-3:
        return
    x0 = max(int(np.floor(dst[:, 0].min())), 0)
    y0 = max(int(np.floor(dst[:, 1].min())), 0)
    x1 = min(int(np.ceil(dst[:, 0].max())) + 1, canvas.shape[1])
    y1 = min(int(np.ceil(dst[:, 1].max())) + 1, canvas.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    local = dst.copy()
    local[:, 0] -= float(x0)
    local[:, 1] -= float(y0)
    if abs(float(np.cross(local[1] - local[0], local[2] - local[0]))) < 1.0e-2:
        return
    warp = cv2.getAffineTransform(src, local)
    patch = cv2.warpAffine(
        image, warp, (x1 - x0, y1 - y0),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
    # Gutter texels stay black. Replace those samples with the triangle's own texels.
    # 色块之间的纹素是黑的。这些采样改用这个三角形自己的纹素。
    dark = patch.sum(axis=2) < BODY_DARK_SUM
    if bool(dark.any()):
        tex_x = np.clip(np.rint(src[:, 0]).astype(np.int32), 0, width - 1)
        tex_y = np.clip(np.rint(src[:, 1]).astype(np.int32), 0, height - 1)
        corner = image[tex_y, tex_x]
        keep = corner.sum(axis=1) >= BODY_DARK_SUM
        if bool(keep.any()):
            patch[dark] = corner[keep].mean(axis=0)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(local).astype(np.int32), 255)
    region = canvas[y0:y1, x0:x1]
    patch_bgr = patch[:, :, ::-1]
    region[mask > 0] = patch_bgr[mask > 0]


def paint_mesh_faces(canvas, xy, faces, depth, mesh):
    """
    # Draw faces from far to near.

    ## Args

        - canvas: (H, W, 3) uint8 BGR. It is not None. Covered pixels are overwritten.
        - xy: (N, 2) screen pixels. It is not None.
        - faces: (F, 3) int vertex indices. It is not None.
        - depth: (F,) float. Smaller depth is drawn first. It is not None. The caller passes the mean view-z of each face, in meters.
        - mesh: the trimesh. It is not None. A UV texture is sampled. Without UV, vertex colors are used.

    ## Returns

        - Returns None.
        - The return is None.

    ---

    # 从远到近画面。

    ## 参数

        - canvas: (H, W, 3) uint8 BGR。不是 None。被覆盖的像素会被改写。
        - xy: (N, 2) 屏幕像素。不是 None。
        - faces: (F, 3) 整数顶点序号。不是 None。
        - depth: (F,) 浮点。较小的深度先画。不是 None。调用方传入每个面的视角 z 均值，单位米。
        - mesh: trimesh。不是 None。有 UV 贴图时采样贴图。没有 UV 时用顶点色。

    ## 返回

        - 返回 None。
        - 返回值是 None。

"""
    import cv2

    uv, image = texture_rgb(mesh)
    colors = None
    if uv is None:
        colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.uint8)
    order = np.argsort(np.asarray(depth, dtype=np.float64))
    for face in np.asarray(faces, dtype=np.int32)[order]:
        poly = np.rint(xy[face]).astype(np.int32)
        if uv is None:
            color = tuple(int(value) for value in colors[face].mean(axis=0)[::-1])
            cv2.fillConvexPoly(canvas, poly, color)
            continue
        face_uv = uv[face]
        edges = np.linalg.norm(np.roll(face_uv, -1, axis=0) - face_uv, axis=1)
        if float(edges.max()) > UV_SEAM_EDGE:
            height, width = image.shape[:2]
            tex_x = np.clip(np.rint(face_uv[:, 0] * (width - 1)).astype(np.int32), 0, width - 1)
            tex_y = np.clip(np.rint((1.0 - face_uv[:, 1]) * (height - 1)).astype(np.int32), 0, height - 1)
            color = image[tex_y, tex_x].mean(axis=0)
            cv2.fillConvexPoly(canvas, poly, (int(color[2]), int(color[1]), int(color[0])))
            continue
        paint_uv_face(canvas, poly, face_uv, image)


def draw_pair(recon, cad, path, yaw_extra=0.0):
    """
    # Draw both meshes in one view and write the picture.

    ## Args

        - recon: the reconstruction trimesh, vertices in meters. It is not None.
        - cad: the real-mesh trimesh, vertices in meters. It is not None. Its outline is drawn in blue.
        - path: the output image path. It is not None.
        - yaw_extra: radians added after label_yaw. The default is 0. It is not None. The cracker pair passes half a turn.

    ## Returns

        - Returns None.
        - The return is None.
        - The picture is 480 by 480 pixels, BGR.

    ---

    # 在同一个视角里画两个网格，并写出图片。

    ## 参数

        - recon: 重建 trimesh，顶点单位米。不是 None。
        - cad: 真实网格 trimesh，顶点单位米。不是 None。它的轮廓画成蓝色。
        - path: 输出图片路径。不是 None。
        - yaw_extra: 加在 label_yaw 之后的弧度。默认 0。不是 None。饼干盒对照图传入半圈。

    ## 返回

        - 返回 None。
        - 返回值是 None。
        - 图片是 480 乘 480 像素，BGR。

"""
    import cv2

    def view_of(vertices):
        """
        # Center vertices and return them in the shared camera view.

        ## Args

            - vertices: (N, 3) meters. It is not None.

        ## Returns

            - The return is (N, 3) float64 meters.
            - x: right, y is up, z points toward the camera. It is not None.

        ---

        # 把顶点移到中心，并返回共同相机视角里的坐标。

        ## 参数

            - vertices: (N, 3)，单位米。不是 None。

        ## 返回

            - 返回值是 (N, 3) float64，米。
            - x 向右，y 向上，z 朝向相机。
            - 不是 None。

"""
        local = np.asarray(vertices, dtype=np.float64)
        local = local - local.mean(axis=0)
        yaw, pitch = label_yaw(recon) + float(yaw_extra), np.deg2rad(18.0)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        rotation = np.array([
            [cy, 0, sy],
            [sy * sp, cp, -cy * sp],
            [-sy * cp, sp, cy * cp],
        ], dtype=np.float64)
        return local @ rotation.T

    recon_v = view_of(recon.vertices)
    cad_v = view_of(cad.vertices)
    both = np.concatenate([recon_v, cad_v], axis=0)
    span = np.ptp(both[:, :2], axis=0).max()
    if span < 1e-8:
        span = 1.0
    width, height = 480, 480
    scale = 0.82 * min(width, height) / span
    center = both[:, :2].mean(axis=0)

    def project(view):
        """
        # Project view coordinates to the 480 pixel canvas.

        ## Args

            - view: (N, 3) float64 meters from view_of. It is not None.

        ## Returns

            - The return is (N, 2) float64 screen pixels, x right and y down.
            - It is not None.

        ---

        # 把视角坐标投到 480 像素的画布上。

        ## 参数

            - view: view_of 返回的 (N, 3) float64，单位米。不是 None。

        ## 返回

            - 返回值是 (N, 2) float64 屏幕像素，x 向右，y 向下。
            - 不是 None。

"""
        return np.stack([
            width / 2 + (view[:, 0] - center[0]) * scale,
            height / 2 - (view[:, 1] - center[1]) * scale,
        ], axis=1)

    recon_xy = project(recon_v)
    cad_xy = project(cad_v)
    canvas = np.full((height, width, 3), 244, dtype=np.uint8)
    recon_faces = np.asarray(recon.faces, dtype=np.int32)
    paint_mesh_faces(
        canvas, recon_xy, recon_faces, recon_v[recon_faces].mean(axis=1)[:, 2], recon,
    )
    cad_faces = np.asarray(cad.faces, dtype=np.int32)
    covered = np.zeros((height, width), dtype=np.uint8)
    for face in cad_faces:
        poly = np.rint(cad_xy[face]).astype(np.int32)
        if poly[:, 0].max() < 0 or poly[:, 1].max() < 0 or poly[:, 0].min() >= width or poly[:, 1].min() >= height:
            continue
        cv2.fillConvexPoly(covered, poly, 255)
    contours, _hier = cv2.findContours(covered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (200, 90, 30), 2, cv2.LINE_AA)
    cv2.imwrite(path, canvas)




if __name__ == "__main__":
    # Run the script stages directly in the entry block.
    # 在入口块中直接执行脚本各阶段。
    import torch
    from wapr.estimator import WAPREstimator
    from step01_point_mask import save_viewer_mesh

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = jobs()
    collect_masks(rows)
    estimator = WAPREstimator(device=DEVICE)
    report = []
    for row in rows:
        rgb = row["rgb_u8"]
        _rgb, depth_m = read_rgb_depth(row["rgb"])
        cad = load_viewer_mesh(row["cad"])
        pose_cad, score_cad, _diameter = pose_of(estimator, rgb, depth_m, row["k"], row["mask"], cad)
        print(row["name"], "cad", round(score_cad, 3), flush=True)
        entry = {"name": row["name"], "cad_score_6d": score_cad, "mask_px": int(row["mask"].sum())}
        for label in ("before", "after"):
            mesh = load_viewer_mesh(row[label])
            pose_mesh, score_mesh, diameter_m = pose_of(
                estimator, rgb, depth_m, row["k"], row["mask"], mesh,
            )
            moved, relative = into_cad_frame(mesh, pose_mesh, pose_cad)
            save_viewer_mesh(moved, os.path.join(OUT_DIR, row["name"] + "_" + label))
            rotation = relative[:3, :3]
            angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))
            entry[label] = {
                "score_6d": score_mesh,
                "diameter_m": diameter_m,
                "angle_deg": round(angle, 2),
                "translation_mm": [round(float(value) * 1000.0, 1) for value in relative[:3, 3]],
                "extents_mm": extents_mm(moved.vertices),
            }
            print(row["name"], label, "score_6d", round(score_mesh, 3), "angle", round(angle, 1), flush=True)
        save_viewer_mesh(cad, os.path.join(OUT_DIR, row["name"] + "_cad"))
        draw_pair(
            load_viewer_mesh(os.path.join(OUT_DIR, row["name"] + "_after")),
            cad,
            os.path.join(OUT_DIR, row["name"] + "_pair.png"),
            CRACKER_PAIR_YAW if row["name"] == "cracker" else 0.0,
        )
        entry["cad_extents_mm"] = extents_mm(cad.vertices)
        report.append(entry)
    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    del estimator
    torch.cuda.empty_cache()
    print("wrote", OUT_DIR, flush=True)
