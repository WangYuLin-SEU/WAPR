# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Stage 01, called by examples/11_reconstruct_object.py; the entry runs the outline path alone.
# 第 01 段，由 examples/11_reconstruct_object.py 调用；入口块单独跑轮廓路径。
# Stage 01 in the combined example; its standalone main does not pass files to stage 02.
# 综合示例的第 01 阶段；单独运行时不会向第 02 阶段传递文件。
# A point on one RGB-D frame. python examples/11_reconstruct_object/step01_point_mask.py
# 一张 RGB-D 上的一个点。运行：python examples/11_reconstruct_object/step01_point_mask.py
"""Build one metric mesh from a point on an RGB-D frame, then estimate pose.

从一个点出发，用一张 RGB-D 得到米制网格，再估计位姿。

The point prompts SAM 2. SAM 3D reconstructs a mesh from that mask and image.
Depth inside the mask gives a starting object extent. Several scales of that
extent are batched through estimate_many_categories_many_instances: WAPR, then SAPR, then
WBPS scores the hypotheses. The higher score_6d is the object model.
Axes that project onto the mask are then scaled, at that pose, until the
outline sits on the mask. The pose is not moved. An axis the first frame cannot
see stays until a later frame shows it. Rendered depth rejects a step that gets
worse. The visible depth cloud is not the object size.
这个点提示 SAM 2。SAM 3D 用该 mask 和图像重建网格。
mask 内的深度给出物体的起始尺寸。尺度候选通过 estimate_many_categories_many_instances 批量计算，
先用 WAPR 修正，再用 SAPR 修正，然后 WBPS 评分。
score_6d 更高的尺度留下，作为物体模型。
在这个位姿下，投影到 mask 上的轴再缩放，直到轮廓贴住 mask。位姿不动。
第一帧看不见的轴留到后面的帧。渲染深度变差的一步丢掉。
可见深度点云不拿来当物体尺寸。
"""
import json
import os
import sys
import time

import numpy as np


DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.dirname(os.path.dirname(DEMO_DIR))
# Use the same source and weight roots as the installed package's downloaders.
# 使用与已安装包下载器一致的源码及权重根目录。
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)
from wapr.resources import resource_root, source_checkout, weights_dir, samples_dir
THIRD_PARTY = os.path.join(resource_root(), "third_party" if source_checkout else "sources")
WEIGHTS_DIR = weights_dir()

# Local checkouts. They are not part of the pose package.
# 本地检出。它们不属于位姿包。
SAM2_ROOT = os.path.join(THIRD_PARTY, "sam2")
SAM3D_ROOT = os.path.join(THIRD_PARTY, "sam-3d-objects")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = os.path.join(WEIGHTS_DIR, "sam2.1_hiera_large.pt")
# pipeline.yaml plus the checkpoints next to it.
# pipeline.yaml 和它旁边的权重。
SAM3D_CONFIG = os.path.join(WEIGHTS_DIR, "sam3d", "checkpoints", "pipeline.yaml")
# Local MoGe weights. The pipeline asks for Ruicheng/moge-vitl.
# 本地 MoGe 权重。pipeline 要的是 Ruicheng/moge-vitl。
MOGE_CHECKPOINT = os.path.join(WEIGHTS_DIR, "moge-vitl", "model.pt")
# One external dataset root. The frame paths below are children of it.
# 一个外部数据根。下面的帧路径都从它推出。
DATA_ROOT = os.environ.get("RECON_DATA_ROOT", os.path.join(samples_dir(), "YCBInEOAT"))
DATA_ROOT = os.path.abspath(os.path.join(RELEASE_DIR, DATA_ROOT))
SEQUENCE = "cracker_box_reorient"
# Frame that is reconstructed, and a later frame that only estimates pose.
# 用来重建的帧，以及之后只做位姿估计的帧。
BUILD_FRAME = 0
LATER_FRAME = 40
# These saved pixel selections act like manual clicks; no annotation is projected at runtime.
# 这些保存的像素位置相当于人工点击；运行时不投影标注数据。
POINTS_PATH = os.path.join(DEMO_DIR, "selected_points.json")
with open(POINTS_PATH, "r", encoding="utf-8") as stream:
    saved_points = json.load(stream)[SEQUENCE]
# Pixel coordinates are (u, v), right and down. Pose checks add the frame index first.
# 像素坐标是 (u, v)，向右和向下；姿态检查项在前面另加帧序号。
CLICK_UV = tuple(saved_points["build"])
POSE_CHECK_FRAMES = tuple(tuple(item) for item in saved_points["pose_checks"])
# Magnified render beside each pose photo. Both panels use this size, in pixels.
# 每张姿态照片旁边的渲染放大。两块都用这个尺寸，像素。
POSE_ZOOM_W = 360
POSE_ZOOM_H = 480
# Callout around the photo crop and the two lines into the magnified panel.
# Thickness, dash, and gap are pixels on the saved figure. The dash keeps this
# frame separate from the object contour, which stays a solid stroke.
# 照片裁切框，以及连到放大图的两条线。
# 粗细、实段和间隔都是保存图上的像素。虚线用来和物体轮廓的实线分开。
CALLOUT_THICKNESS_PX = 6
CALLOUT_DASH_PX = 18
CALLOUT_GAP_PX = 12
# Empirical multipliers on the masked depth extent. Occlusion can reduce that extent;
# a mask that includes background or a gripper can enlarge it. These values are not
# a confidence interval for the object's metric diameter.
# 掩码深度范围上的经验倍数。遮挡会缩小该范围；掩码若包含背景或夹爪则可能放大它。
# 这些倍数不是物体米制直径的置信区间。
SCALE_FACTORS = (0.75, 1.0, 1.4, 2.0)
# Projected extent below this, in pixels, is not a measured side.
# 投影跨度低于这个像素数，就不把它当成测到的一条边。
AXIS_MIN_EXTENT_PX = 12.0
# A side is measured only when its image extent is at least this fraction of the longest side.
# 一条边的图像跨度至少要达到最长边的这个比例，才拿来测量。
AXIS_OBSERVABLE_FRACTION = 0.40
# An outline ratio closer than this to 1 is left unchanged. One percent is about 1.5 px on this mask.
# 轮廓比值比这更接近 1 时不改。百分之一大约是这块 mask 上的 1.5 像素。
AXIS_SCALE_MIN_CHANGE = 0.01
# Cumulative scale on one object axis stays inside this range.
# 一条物体轴上的累积尺度留在这个范围内。
AXIS_SCALE_MIN = 0.70
AXIS_SCALE_MAX = 1.30
# A step that pulls the silhouette onto the mask may worsen depth by up to this much.
# 把剪影收到 mask 上的一步，深度最多允许变差这么多。
OUTLINE_DEPTH_SLACK_M = 0.005
# Outline updates. Each pass rescales the mesh at the same pose.
# 轮廓更新的轮数。每一轮都在同一个位姿下重新缩放网格。
AXIS_FIT_ITERS = 3
# Width of the drawn contour. The filled silhouette is fit this many pixels inside the mask,
# one pixel on each side, so the stroke sits on the mask edge.
# 画出的轮廓宽度。填充剪影收到 mask 内侧这么多像素，每边一个，笔画就落在 mask 边上。
OUTLINE_STROKE_PX = 2.0
# Square depth render. Tile pixel centers map back onto the full frame.
# 正方形深度渲染。块上的像素中心映射回整幅图像。
DEPTH_TILE_PX = 256
# Added to the depth residual for each fraction of the mask the render misses.
# 渲染没有盖到的 mask 比例，按这个深度加进残差。
DEPTH_MISS_M = 0.05
# A silhouette step is rejected when rendered depth gets worse by at least this much.
# 渲染深度至少变差这么多，这一步轮廓更新就丢掉。
DEPTH_LOSS_MIN_GAIN_M = 0.001
# Face count used to measure the outline and the depth. Scales apply to the full mesh.
# 测量轮廓和深度时用的面数。尺度加到完整网格上。
ASPECT_SEARCH_FACES = 8000
device = "cuda:0"
OUT_DIR = os.path.join(RELEASE_DIR, "pages", "demo", "reconstruct")


def frame_list(seq_dir):
    """
    # Return the sorted RGB image paths in one sequence directory.

        `seq_dir` is a directory path.

        The function lists `seq_dir/rgb` and keeps names ending in `.png`, `.jpg`, or `.jpeg`. It is not a tensor and it is not None.

    ---

    # 返回一个序列目录里排好序的 RGB 图像路径。

        `seq_dir` 是目录路径。

        函数列出 `seq_dir/rgb`，保留以 `.png`、`.jpg` 或 `.jpeg` 结尾的名字。

        不是张量，也不是 None。

"""
    rgb_dir = os.path.join(seq_dir, "rgb")
    names = sorted(
        name for name in os.listdir(rgb_dir)
        if name.endswith((".png", ".jpg", ".jpeg"))
    )
    return [os.path.join(rgb_dir, name) for name in names]


def read_rgb_depth(rgb_path):
    """
    # Return RGB uint8 and a float32 depth map.

        A uint16 depth file is divided by 1000, so those values are in meters.

        A three-channel depth file keeps channel 0.

        `rgb_path` is an image path.

        The color image is read as BGR and converted to RGB.

        The depth path replaces `/rgb/` with `/depth/`. It is not a tensor and it is not None.

    ---

    # 返回 RGB uint8 和 float32 深度图。

        uint16 深度文件会除以 1000，因此那些值的单位是米。

        三通道深度文件只保留第 0 通道。

        `rgb_path` 是图像路径。

        彩色图按 BGR 读入再转成 RGB。

        深度路径把 `/rgb/` 换成 `/depth/`。

        不是张量，也不是 None。

"""
    import cv2

    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_path = rgb_path.replace("/rgb/", "/depth/")
    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = np.asarray(raw, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if raw.dtype == np.uint16:
        depth = depth / 1000.0
    return rgb, depth


def segment_point(rgb, click_uv):
    """
    # Return a SAM 2 mask for one foreground point.

        The mask is uint8 with values 0 or 1, the same height and width as `rgb`.

        `rgb` is the image passed to `predictor.set_image`. Callers pass the RGB uint8 array from `read_rgb_depth`. It is not None.

        `click_uv` is reshaped to one float32 point of shape (1, 2).

        The label is foreground.

        The highest SAM score is kept.

        It is not None.

        Pixel x is the first component and y is the second.

    ---

    # 返回一个前景点的 SAM 2 mask。

        mask 是 uint8，取值 0 或 1，高宽与 `rgb` 相同。

        `rgb` 是传给 `predictor.set_image` 的图像。

        调用者传入 `read_rgb_depth` 的 RGB uint8 数组。

        不是 None。

        `click_uv` 会被 reshape 成一个 float32 点，形状 (1, 2)。

        标签是前景。

        保留 SAM 分数最高的那张 mask。

        不是 None。

        第一个分量是像素 x，第二个是 y。

"""
    import torch
    from wapr.sam3d_isolated import SAM3D_ENV_ROOT
    if os.path.realpath(sys.prefix) != os.path.realpath(SAM3D_ENV_ROOT):
        # Keep the pose Torch; run the same large SAM2 point recipe separately.
        # 保留位姿 Torch；在独立进程运行相同 large SAM2 点分割配方。
        from wapr.sam2_isolated import predict_mask
        isolated_checkpoint = SAM2_CHECKPOINT
        if isolated_checkpoint == os.path.join(WEIGHTS_DIR, "sam2.1_hiera_large.pt") and not os.path.isfile(isolated_checkpoint):
            isolated_checkpoint = None
        return predict_mask(rgb, point_uv=click_uv, device=device,
                            checkpoint=isolated_checkpoint, config=SAM2_CONFIG)
    if SAM2_ROOT not in sys.path:
        sys.path.insert(0, SAM2_ROOT)
    # Prepare optional dependencies only when segmentation is requested.
    # 仅在调用分割时准备可选依赖，优先使用上面的用户源码目录。
    from wapr.bootstrap import ensure_optional
    ensure_optional("sam2")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = SAM2_CHECKPOINT
    if not os.path.isfile(checkpoint):
        default_checkpoint = os.path.join(WEIGHTS_DIR, "sam2.1_hiera_large.pt")
        if checkpoint != default_checkpoint:
            raise FileNotFoundError("Custom SAM2 checkpoint missing / 自定义 SAM2 权重缺失: " + checkpoint)
        # Fetch the same large checkpoint on first use; preserve supplied files.
        # 首次使用时获取相同 large 权重；保留用户已提供的文件。
        from wapr.source_setup import prepare_sam2_weights
        prepared_weights = prepare_sam2_weights("large")
        checkpoint = prepared_weights["checkpoint"]
    sam = build_sam2(SAM2_CONFIG, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam)
    predictor.set_image(rgb)
    point = np.asarray(click_uv, dtype=np.float32).reshape(1, 2)
    label = np.ones(1, dtype=np.int32)
    masks, scores, _low = predictor.predict(
        point_coords=point, point_labels=label, multimask_output=True,
    )
    pick = int(np.argmax(scores))
    return (masks[pick] > 0).astype(np.uint8)


def sam3d_utils3d_names():
    """
    # Attach the two utils3d names the SAM 3D bake looks up, when those attributes are missing.

        This function takes no arguments.

        It imports `utils3d.torch` and assigns `intrinsics_from_fov_xy` and `rasterize_triangle_faces` only if they are absent.

    ## Returns

        - Returns None.

    ---

    # 当 SAM 3D 烘焙要找的两个 utils3d 名字还不存在时，把它们挂上。

        这个函数没有参数。

        它导入 `utils3d.torch`，只有在 `intrinsics_from_fov_xy` 和 `rasterize_triangle_faces` 缺失时才赋值。

    ## 返回

        - 返回 None。

"""
    import utils3d.torch as utils3d_torch

    if not hasattr(utils3d_torch, "intrinsics_from_fov_xy"):
        def intrinsics_from_fov_xy(fov_x, fov_y):
            """
            # Return `utils3d.torch.intrinsics_from_fov` called with `fov_x` and `fov_y` as keywords.

                `fov_x` and `fov_y` are forwarded unchanged.

                This function does not state their units or shapes, and it does not accept a None default.

            ---

            # 返回用关键字 `fov_x` 和 `fov_y` 调用的 `utils3d.torch.intrinsics_from_fov`。

                `fov_x` 和 `fov_y` 原样转发。

                这个函数不写它们的单位或形状，也没有 None 默认值。

"""
            return utils3d_torch.intrinsics_from_fov(fov_x=fov_x, fov_y=fov_y)

        utils3d_torch.intrinsics_from_fov_xy = intrinsics_from_fov_xy
    if not hasattr(utils3d_torch, "rasterize_triangle_faces"):
        def rasterize_triangle_faces(ctx, vertices, faces, width, height, **kwargs):
            """
            # Return `utils3d.torch.rasterize_triangles` for the given context, size, vertices, and faces.

                `ctx` is forwarded as the raster context.

                `vertices` and `faces` are forwarded as keywords.

                They are not None.

                `width` and `height` are converted with `int` and passed as the raster size.

                They are not None.

                `kwargs` are forwarded to `rasterize_triangles`. This function does not fix their names.

            ---

            # 返回针对给定上下文、尺寸、顶点和面的 `utils3d.torch.rasterize_triangles`。

                `ctx` 作为光栅上下文转发。

                `vertices` 和 `faces` 按关键字转发。

                不是 None。

                `width` 和 `height` 会用 `int` 转换，并作为光栅尺寸传入。

                不是 None。

                `kwargs` 转给 `rasterize_triangles`。

                这个函数不规定它们的名字。

"""
            return utils3d_torch.rasterize_triangles(
                ctx, int(width), int(height),
                vertices=vertices, faces=faces, **kwargs,
            )

        utils3d_torch.rasterize_triangle_faces = rasterize_triangle_faces


def sam3d_gaussian_backend():
    """
    # Replace the bake's `render_frames` so its options default to the installed gsplat backend.

        This function takes no arguments.

        It reads `render_utils.render_frames` and the marker `_wapr_gsplat`.

    ## Returns

        - Returns None when the replacement is already installed, and otherwise returns None after installing it.

    ---

    # 把烘焙用的 `render_frames` 换成默认使用已安装 gsplat 后端的版本。

        这个函数没有参数。

        它读取 `render_utils.render_frames` 和标记 `_wapr_gsplat`。

    ## 返回

        - 已经装过时返回 None，装完之后也返回 None。

"""
    from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

    if getattr(render_utils.render_frames, "_wapr_gsplat", False):
        return
    original = render_utils.render_frames

    def render_frames(sample, extrinsics, intrinsics, options=None, **kwargs):
        """
        # Call the previous `render_frames` after setting the options backend to `gsplat` when that key is absent.

            `sample`, `extrinsics`, and `intrinsics` are forwarded to the saved function.

            This wrapper does not state their shapes.

            They are not None.

            `options` is a mapping or None.

            None becomes an empty dict.

            `setdefault` writes `backend` as `gsplat` only when the key is missing.

            `kwargs` are forwarded.

            This wrapper does not fix them.

        ## Returns

            - Returns that call's result.

        ---

        # 若 options 里还没有 backend，就先把它设为 `gsplat`，再调用原来的 `render_frames`。

            `sample`、`extrinsics` 和 `intrinsics` 转给保存的函数。

            这个包装不写它们的形状。

            不是 None。

            `options` 是映射或 None。

            None 会变成空字典。

            只有缺少 `backend` 键时，`setdefault` 才把它写成 `gsplat`。

            `kwargs` 会被转发。

            这个包装不规定它们。

        ## 返回

            - 返回那次调用的结果。

"""
        options = dict(options or {})
        options.setdefault("backend", "gsplat")
        return original(sample, extrinsics, intrinsics, options, **kwargs)

    render_frames._wapr_gsplat = True
    render_utils.render_frames = render_frames


def sam3d_local_dino():
    """
    # Point `torch.hub.load` at the DINOv2 checkout already cached on disk.

        This function takes no arguments.

        The checkout comes from the package's pinned DINOv2 source preparation.

    ## Returns

        - Returns None.
        - A loader already marked returns without a change.

    ---

    # 把 `torch.hub.load` 指到磁盘上已经缓存的 DINOv2 检出。

        这个函数没有参数。

        检出来自包内固定版本 DINOv2 源码准备入口。

    ## 返回

        - 返回 None。
        - 加载器已经打过标记时，不做修改。

"""
    import torch

    if getattr(torch.hub.load, "_wapr_local_dino", False):
        return
    from wapr.source_setup import prepare_source, prepare_sam_dino_weights
    cached = prepare_source("dinov2")
    prepare_sam_dino_weights()
    original = torch.hub.load

    def load(repo_or_dir, model, *args, source="github", **kwargs):
        """
        # Load DINOv2 from the local hub cache when the repo argument is the GitHub id.

            Every other repo is forwarded to the original `torch.hub.load`. Returns that result.

            `repo_or_dir` is compared with the string `facebookresearch/dinov2`. A match is replaced by the cached directory and `source` becomes `local`. It is not None.

            `model` is forwarded as the hub model name.

            `args` and `kwargs` are forwarded.

            `source` defaults to `github` and is used only for the non-DINOv2 branch.

            This wrapper does not state their shapes.

        ---

        # 当仓库参数是 GitHub 上的 DINOv2 标识时，从本地 hub 缓存加载。

            其他仓库转给原来的 `torch.hub.load`。

            返回那次结果。

            `repo_or_dir` 会和字符串 `facebookresearch/dinov2` 比较。

            匹配时改成缓存目录，并且 `source` 变成 `local`。

            不是 None。

            `model` 作为 hub 模型名转发。

            `args` 和 `kwargs` 也会转发。

            `source` 默认 `github`，只在非 DINOv2 分支使用。

            这个包装不写它们的形状。

"""
        if repo_or_dir == "facebookresearch/dinov2":
            return original(cached, model, *args, source="local", **kwargs)
        return original(repo_or_dir, model, *args, source=source, **kwargs)

    load._wapr_local_dino = True
    torch.hub.load = load


def reconstruct_mesh(rgb, mask):
    """
    # Run SAM 3D on the masked RGB image and return the `glb` mesh.

        Vertices stay in the frame produced by that pipeline.

        A mask with fewer than 20 positive pixels raises RuntimeError.

        `rgb` is concatenated with the mask as a fourth uint8 channel.

        Callers pass RGB uint8.

        It is not None.

        `mask` is an array broadcast with `[..., None]` and multiplied by 255 to form the alpha channel.

        It is not None.

    ---

    # 对扣好的 RGB 跑 SAM 3D，并返回 `glb` 网格。

        顶点留在该 pipeline 产出的坐标系里。

        mask 中大于 0 的像素少于 20 个时抛出 RuntimeError。

        `rgb` 会和 mask 拼成第四个 uint8 通道。

        调用者传入 RGB uint8。

        不是 None。

        `mask` 是数组，用 `[..., None]` 扩一维后乘 255，成为 alpha 通道。

        不是 None。

"""
    # Preserve the selected checkout before checking reconstruction dependencies.
    # 检查重建依赖前保留用户选择的源码目录，仅调用重建时准备依赖。
    if SAM3D_ROOT not in sys.path:
        sys.path.insert(0, SAM3D_ROOT)
    from wapr.bootstrap import ensure_optional
    ensure_optional("sam3d")
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    os.environ["LIDRA_SKIP_INIT"] = "true"
    sam3d_utils3d_names()
    sam3d_gaussian_backend()
    sam3d_local_dino()
    if int(mask.sum()) < 20:
        raise RuntimeError("mask")
    # Drop the segmentation weights before the reconstruction model.
    # 重建模型加载前先放开分割权重。
    torch.cuda.empty_cache()
    rgba = np.concatenate([rgb, (mask[..., None] * 255).astype(np.uint8)], axis=2)
    config = OmegaConf.load(SAM3D_CONFIG)
    # pytorch3d avoids a second raster context on a busy GPU.
    # 用 pytorch3d，避免在已经有渲染上下文的卡上再开一个。
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.workspace_dir = os.path.dirname(SAM3D_CONFIG)
    config.depth_model.model.pretrained_model_name_or_path = MOGE_CHECKPOINT
    pipeline = instantiate(config)
    with torch.no_grad():
        # Decoder vertex colors. No UV unwrap and no Gaussian-view bake.
        # 用解码器的顶点颜色。不展开 UV，也不做高斯多视角烘焙。
        output = pipeline.run(
            rgba,
            None,
            seed=42,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
        )
    mesh = output["glb"]
    # The pose step needs the same GPU. Keep only the mesh.
    # 位姿估计还要用这张卡。只留下网格。
    del output
    del pipeline
    torch.cuda.empty_cache()
    return mesh


def depth_extent_m(depth_m, mask, K):
    """
    # Return twice the 90th-percentile radius of the back-projected mask points, in meters.

        Fewer than 20 valid pixels raises RuntimeError.

        `depth_m` is a 2D float array.

        Valid pixels are `mask > 0` and `depth_m > 0`. Those values are the camera-frame z in the pinhole back-projection.

        Callers pass meters from `read_rgb_depth`. It is not None.

        `mask` is an array of the same image grid.

        Pixels `> 0` are kept.

        It is not None.

        `K` is a 3×3 camera matrix.

        `K[0, 0]` and `K[0, 2]` scale and shift x, and `K[1, 1]` and `K[1, 2]` do the same for y, in pixels.

        x is right and y is down in that image.

        It is not None.

    ---

    # 返回反投影后 mask 点的 90 分位半径的两倍，单位米。

        有效像素少于 20 个时抛出 RuntimeError。

        `depth_m` 是二维浮点数组。

        有效像素是 `mask > 0` 且 `depth_m > 0`。

        这些值是针孔反投影里的相机系 z。

        调用者传入 `read_rgb_depth` 的米。

        不是 None。

        `mask` 是同一图像网格上的数组。

        `> 0` 的像素被保留。

        不是 None。

        `K` 是 3×3 相机矩阵。

        `K[0, 0]` 和 `K[0, 2]` 缩放并平移 x，`K[1, 1]` 和 `K[1, 2]` 对 y 做同样的事，单位是像素。

        在这张图上 x 向右、y 向下。

        不是 None。

"""
    valid = (mask > 0) & (depth_m > 0)
    ys, xs = np.where(valid)
    if len(xs) < 20:
        raise RuntimeError("depth")
    z = depth_m[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - float(K[0, 2])) / float(K[0, 0]) * z
    y = (ys.astype(np.float64) - float(K[1, 2])) / float(K[1, 1]) * z
    pts = np.stack([x, y, z], axis=1)
    if len(pts) > 4000:
        take = np.linspace(0, len(pts) - 1, 4000).astype(np.int32)
        pts = pts[take]
    center = np.median(pts, axis=0)
    radius = np.linalg.norm(pts - center.reshape(1, 3), axis=1)
    return float(np.percentile(radius, 90) * 2.0)


def mesh_diameter_m(vertices):
    """
    # Return the bounding-box diagonal of the vertices as a float, in the same units as those coordinates.

        `vertices` is an array converted to float64.

        The diagonal is the norm of `max(axis=0) - min(axis=0)`. It is not None.

        Callers pass mesh vertices that they treat as meters after scaling.

    ---

    # 返回顶点包围盒对角线，一个浮点数，单位与这些坐标相同。

        `vertices` 是转成 float64 的数组。

        对角线是 `max(axis=0) - min(axis=0)` 的范数。

        不是 None。

        调用者传入的是缩放之后按米使用的网格顶点。

"""
    lo = np.asarray(vertices, dtype=np.float64).min(axis=0)
    hi = np.asarray(vertices, dtype=np.float64).max(axis=0)
    return float(np.linalg.norm(hi - lo))


def scale_mesh(mesh, factor):
    """
    # Return a copied trimesh whose vertex positions are multiplied by `factor`. A copy that is not a `trimesh.Trimesh` raises TypeError.

        `mesh` is copied.

        Its vertices are read as float64.

        The original mesh is not modified.

        It is not None.

        `factor` is converted with `float` and multiplies every vertex component.

        It is in the same units ratio as the vertex coordinates.

        It is not None.

    ---

    # 返回复制后的 trimesh，其顶点坐标乘以 `factor`。

        复制结果不是 `trimesh.Trimesh` 时抛出 TypeError。

        `mesh` 会被复制。

        顶点按 float64 读取。

        原始网格不会被修改。

        不是 None。

        `factor` 会用 `float` 转换，并乘到每个顶点分量上。

        它是顶点坐标的比例。

        不是 None。

"""
    import trimesh

    scaled = mesh.copy()
    scaled.vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(factor)
    if not isinstance(scaled, trimesh.Trimesh):
        raise TypeError("mesh")
    return scaled


def scale_mesh_axes(mesh, scales):
    """
    # Return a copied trimesh scaled about its bounding-box center by one factor per vertex axis.

        A copy that is not a `trimesh.Trimesh` raises TypeError.

        `mesh` is copied.

        Vertices are float64.

        The center is the midpoint of the per-axis min and max.

        The original mesh is not modified.

        It is not None.

        `scales` is converted to float64 and reshaped to (1, 3).

        It multiplies the centered vertex coordinates on x, y, and z of the mesh.

        It is not None.

    ---

    # 返回绕包围盒中心、按每个顶点轴一个倍数缩放后的 trimesh 副本。

        复制结果不是 `trimesh.Trimesh` 时抛出 TypeError。

        `mesh` 会被复制。

        顶点是 float64。

        中心是各轴最小值和最大值的中点。

        原始网格不会被修改。

        不是 None。

        `scales` 会转成 float64 并 reshape 成 (1, 3)。

        它乘在网格 x、y、z 上已经减去中心的顶点坐标。

        不是 None。

"""
    import trimesh

    scaled = mesh.copy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    vertices = center + (vertices - center) * np.asarray(scales, dtype=np.float64).reshape(1, 3)
    scaled.vertices = vertices
    if not isinstance(scaled, trimesh.Trimesh):
        raise TypeError("mesh")
    return scaled


def scale_axis_about(mesh, axis, factor, fraction):
    """
    # Return a copied trimesh with one vertex axis scaled about a pivot on that axis.

        The function does not change a pose.

        A copy that is not a `trimesh.Trimesh` raises TypeError.

        `mesh` is copied.

        Vertices are float64.

        Only column `axis` is rewritten.

        It is not None.

        `axis` is an integer column, 0, 1, or 2, in the mesh vertex array.

        It is not None.

        `factor` is converted with `float` and multiplies the distance from the pivot along that axis.

        It is not None.

        `fraction` is converted with `float`. The pivot is `min + fraction * (max - min)` on this axis.

        0 is the bounding-box minimum and 1 is the maximum.

        It is not None.

    ---

    # 返回一个 trimesh 副本，其中一条顶点轴绕该轴上的支点缩放。

        这个函数不改位姿。

        复制结果不是 `trimesh.Trimesh` 时抛出 TypeError。

        `mesh` 会被复制。

        顶点是 float64。

        只有第 `axis` 列会被改写。

        不是 None。

        `axis` 是网格顶点数组的整数列，取 0、1 或 2。

        不是 None。

        `factor` 会用 `float` 转换，并乘在该轴上相对支点的距离。

        不是 None。

        `fraction` 会用 `float` 转换。

        支点是该轴上的 `min + fraction * (max - min)`。

        0 是包围盒小端，1 是大端。

        不是 None。

"""
    import trimesh

    scaled = mesh.copy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    lo = float(vertices[:, axis].min())
    hi = float(vertices[:, axis].max())
    pivot = lo + float(fraction) * (hi - lo)
    vertices[:, axis] = pivot + (vertices[:, axis] - pivot) * float(factor)
    scaled.vertices = vertices
    if not isinstance(scaled, trimesh.Trimesh):
        raise TypeError("mesh")
    return scaled


def outline_scale_about(mask_low, mask_high, sil_low, sil_high, axis_low_px, axis_high_px):
    """
    # Return the scale and the pivot fraction that map the silhouette interval onto the mask interval.

        A silhouette span below 1 or a non-positive mask span returns `(1.0, 0.5)`.

        `mask_low` and `mask_high` are pixel coordinates of the mask ends on one image axis.

        Their difference is the target span.

        They are not None.

        `sil_low` and `sil_high` are pixel coordinates of the silhouette ends on that same image axis.

        Their difference is the current span.

        A scale of exactly 1 returns fraction 0.5.

        They are not None.

        `axis_low_px` and `axis_high_px` are the projected ends of the object axis, in pixels.

        The pivot pixel is converted to a fraction of that span and clipped to `[0, 1]`. A span whose absolute value is below 1 returns fraction 0.5.

        They are not None.

    ---

    # 返回把剪影区间映到 mask 区间的缩放和支点比例。

        剪影跨度小于 1，或 mask 跨度不是正数时，返回 `(1.0, 0.5)`。

        `mask_low` 和 `mask_high` 是 mask 在一条图像轴上两端的像素坐标。

        它们的差是目标跨度。

        不是 None。

        `sil_low` 和 `sil_high` 是剪影在同一条图像轴上两端的像素坐标。

        它们的差是当前跨度。

        缩放恰好为 1 时返回支点比例 0.5。

        不是 None。

        `axis_low_px` 和 `axis_high_px` 是物体轴投影后的两端，单位像素。

        支点像素会换成这段跨度上的比例，并截到 `[0, 1]`。

        跨度绝对值小于 1 时返回比例 0.5。

        不是 None。

"""
    span = float(sil_high - sil_low)
    target = float(mask_high - mask_low)
    if span < 1.0 or target <= 0.0:
        return 1.0, 0.5
    factor = target / span
    if abs(factor - 1.0) < 1.0e-8:
        return factor, 0.5
    pivot_px = (float(mask_low) - float(sil_low) * factor) / (1.0 - factor)
    axis_span = float(axis_high_px - axis_low_px)
    if abs(axis_span) < 1.0:
        return factor, 0.5
    fraction = (pivot_px - float(axis_low_px)) / axis_span
    return factor, float(np.clip(fraction, 0.0, 1.0))


def silhouette_iou(silhouette, mask):
    """
    # Return the intersection-over-union of the projected mesh and the prompt mask as a float.

        An empty union uses a denominator of 1.

        `silhouette` and `mask` are arrays.

        Pixels `> 0` are inside.

        Their logical overlap and union are counted.

        They are not None.

        This function does not resize them.

    ---

    # 返回投影网格和提示 mask 的交并比，一个浮点数。

        并集为空时分母用 1。

        `silhouette` 和 `mask` 是数组。

        `> 0` 的像素算在内部。

        函数统计它们的逻辑交集和并集。

        不是 None。

        这个函数不缩放它们。

"""
    overlap = np.logical_and(silhouette > 0, mask > 0).sum()
    union = np.logical_or(silhouette > 0, mask > 0).sum()
    return float(overlap) / float(max(int(union), 1))


def align_scales(estimator, rgb, depth_m, mask, K, mesh, extent_m):
    """
    # Try `SCALE_FACTORS` on the depth extent and return the row with the higher `score_6d`, plus every tried row.

        `estimator` is a pose estimator.

        All scales share one `estimate_many_categories_many_instances` call with the same RGB-D frame and their scaled meshes, diameters and mask.

        It is not None.

        `rgb` and `depth_m` are shared by the frame batch. `rgb` is also passed to `project_silhouette`. Callers pass the uint8 RGB and meter depth from `read_rgb_depth`. They are not None.

        Each batch item carries the same `mask`, also used by `silhouette_iou`. It is not None.

        `K` is the 3×3 camera matrix shared by the frame batch and `project_silhouette`. It is not None.

        `mesh` is the unscaled trimesh.

        A non-positive bounding-box diagonal raises RuntimeError.

        Each trial multiplies vertices so the diagonal equals `extent_m * factor`. It is not None.

        `extent_m` is a float in meters, multiplied by each factor in `SCALE_FACTORS`. It is not None.

    ---

    # 对深度尺寸尝试 `SCALE_FACTORS`，返回 `score_6d` 更高的那一行，以及每一行尝试。

        `estimator` 是位姿估计器。

        所有尺度共用一次 `estimate_many_categories_many_instances` 调用，输入为同帧 RGB-D 及各缩放网格、直径和 mask。

        不是 None。

        `rgb` 和 `depth_m` 由整帧批量计算共用。

        `rgb` 也会传给 `project_silhouette`。

        调用者传入 `read_rgb_depth` 的 uint8 RGB 和米制深度。

        不是 None。

        每个批量计算条目携带相同的 `mask`，它也用于 `silhouette_iou`。

        不是 None。

        `K` 是整帧批量计算与 `project_silhouette` 共用的 3×3 相机矩阵。

        不是 None。

        `mesh` 是尚未按这些倍数缩放的 trimesh。

        包围盒对角线不是正数时抛出 RuntimeError。

        每次尝试都把顶点乘到对角线等于 `extent_m * factor`。

        不是 None。

        `extent_m` 是以米为单位的浮点数，会乘上 `SCALE_FACTORS` 里的每个倍数。

        不是 None。

"""
    base = mesh_diameter_m(mesh.vertices)
    if base <= 0:
        raise RuntimeError("mesh diameter")
    rows = []
    trials = []
    for factor in SCALE_FACTORS:
        target_m = float(extent_m) * float(factor)
        scaled = scale_mesh(mesh, target_m / base)
        diameter_m = mesh_diameter_m(scaled.vertices)
        trials.append({"mesh": scaled, "diameter_m": diameter_m, "mask": mask})
    # Scale trials share the image and are rendered/refined/scored together.
    # 尺度候选共用同帧，渲染、修正和评分批量执行。
    prepared = estimator.prepare_meshes([trial["mesh"] for trial in trials])
    pose_trials = [{**trial, "mesh": mesh} for trial, mesh in zip(trials, prepared)]
    mesh_setup_seconds = estimator.last_mesh_setup_seconds
    warmup_seconds = estimator.warmup_pose(rgb, depth_m, K, pose_trials)
    import torch

    torch.cuda.synchronize(estimator.device)
    pose_started = time.perf_counter()
    outputs = estimator.estimate_many_categories_many_instances(rgb, depth_m, K, pose_trials)
    torch.cuda.synchronize(estimator.device)
    pose_hot_seconds = time.perf_counter() - pose_started
    estimator.last_scale_timing = {"model_setup_seconds": estimator.model_setup_seconds,
          "mesh_setup_seconds": mesh_setup_seconds, "warmup_seconds": warmup_seconds,
          "pose_hot_seconds": pose_hot_seconds, "instances": len(pose_trials),
          "scope": "prepared-mesh scale-trial pose batch; includes H2D/D2H; excludes silhouette evaluation and geometry fitting"}
    print("SCALE_POSE_PREPARATION", estimator.last_scale_timing, flush=True)
    for factor, trial, result in zip(SCALE_FACTORS, trials, outputs):
        scaled = trial["mesh"]
        diameter_m = trial["diameter_m"]
        pose = np.asarray(result["pose_4x4"], dtype=np.float64)
        silhouette, _photo = project_silhouette(rgb, scaled, pose, K)
        row = {
            "factor": float(factor),
            "diameter_m": diameter_m,
            "score_6d": float(result["score_6d"]),
            "mask_iou": silhouette_iou(silhouette, mask),
            "pose_4x4": pose,
            "mesh": scaled,
        }
        rows.append(row)
        print(
            "scale", factor,
            "score_6d", round(row["score_6d"], 4),
            "iou", round(row["mask_iou"], 3),
            "diameter_m", round(diameter_m, 4),
            flush=True,
        )
    best = max(rows, key=lambda row: row["score_6d"])
    return best, rows


def fit_axis_scales(mesh, pose, depth_m, mask, K, locked_axes=()):
    """
    # Scale each observable mesh axis, at the given pose, until the outline agrees with the mask, and return the adjusted mesh plus a report dict.

        Axes in `locked_axes`, and axes whose projected length is below the measurable pixel threshold, stay at scale 1 relative to the mesh passed in.

        `mesh` is a trimesh.

        A mesh with more faces than `ASPECT_SEARCH_FACES` is simplified only for the search copy.

        Accepted scales are applied to the full mesh.

        Vertices stay in the mesh frame.

        It is not None.

        `pose` is copied to a float64 4×4 and then left unchanged.

        Its rotation and translation project the axis ends.

        It is not None.

        `depth_m` is a 2D depth array in meters.

        It is resampled to the `DEPTH_TILE_PX` grid and compared with the rendered depth.

        An empty overlap with the mask raises RuntimeError.

        It is not None.

        `mask` is the image mask.

        Pixels `> 0` define the target outline and the valid depth pixels.

        An empty mask raises RuntimeError.

        It is not None.

        `K` is the 3×3 camera matrix used to project axis ends and to render the depth tile.

        It is not None.

        `locked_axes` is a sequence of axis indices, default empty.

        Those axes are skipped by the outline proposals.

        It is not None.

    ---

    # 在给定位姿下缩放每条看得见的网格轴，直到轮廓和 mask 一致，并返回调整后的网格和一个报告字典。

        `locked_axes` 里的轴，以及投影长度低于可测像素阈值的轴，相对传入网格保持尺度 1。

        `mesh` 是 trimesh。

        面数超过 `ASPECT_SEARCH_FACES` 时，只有搜索用的副本会被简化。

        接受的尺度加到完整网格上。

        顶点留在网格坐标系。

        不是 None。

        `pose` 会被复制成 float64 的 4×4，之后不再修改。

        它的旋转和平移用来投影轴的两端。

        不是 None。

        `depth_m` 是以米为单位的二维深度数组。

        它会被重采样到 `DEPTH_TILE_PX` 的网格上，再和渲染深度比较。

        与 mask 没有重叠时抛出 RuntimeError。

        不是 None。

        `mask` 是图像 mask。

        `> 0` 的像素定义目标轮廓和有效深度像素。

        空 mask 抛出 RuntimeError。

        不是 None。

        `K` 是 3×3 相机矩阵，用来投影轴的两端，并渲染深度块。

        不是 None。

        `locked_axes` 是轴下标序列，默认空。

        轮廓提议会跳过这些轴。

        不是 None。

"""
    from wapr.ogl import runtime_for

    height, width = depth_m.shape[:2]
    tile = int(DEPTH_TILE_PX)
    rows_px = ((np.arange(tile) + 0.5) * float(height) / float(tile)).astype(np.int32)
    cols_px = ((np.arange(tile) + 0.5) * float(width) / float(tile)).astype(np.int32)
    rows_px = np.clip(rows_px, 0, height - 1)
    cols_px = np.clip(cols_px, 0, width - 1)
    measured = np.asarray(depth_m, dtype=np.float64)[rows_px][:, cols_px]
    valid = (np.asarray(mask)[rows_px][:, cols_px] > 0) & (measured > 1.0e-4)
    if int(valid.sum()) == 0:
        raise RuntimeError("depth mask")
    mask_ys, mask_xs = np.nonzero(np.asarray(mask) > 0)
    if len(mask_xs) == 0:
        raise RuntimeError("mask")
    mask_uv = np.stack([mask_xs.astype(np.float64), mask_ys.astype(np.float64)], axis=1)

    search = mesh
    if len(mesh.faces) > int(ASPECT_SEARCH_FACES):
        search = mesh.simplify_quadric_decimation(int(ASPECT_SEARCH_FACES))
    runtime = runtime_for(device)
    bbox = np.array([[0.0, 0.0, float(width), float(height)]], dtype=np.float32)
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    locked = {int(axis) for axis in locked_axes}

    pose_fixed = np.array(pose, dtype=np.float64, copy=True)

    def depth_loss(mesh_in):
        """
        # Render a depth tile of `mesh_in` at the fixed pose and return the robust depth loss in meters, the median absolute deviation in meters, and the covered fraction of the valid mask.

            `mesh_in` is a trimesh loaded into the OpenGL runtime under the name `aspect`. The render uses the pose, `K`, image size, and tile closed over by `fit_axis_scales`. No covered pixel returns `DEPTH_MISS_M`, 0.0, and the cover fraction.

            It is not None.

        ---

        # 在固定位姿下渲染 `mesh_in` 的深度块，返回以米为单位的稳健深度损失、以米为单位的绝对偏差中位数，以及有效 mask 被覆盖的比例。

            `mesh_in` 是 trimesh，会以名字 `aspect` 载入 OpenGL 运行时。

            渲染使用 `fit_axis_scales` 闭包里的位姿、`K`、图像尺寸和块大小。

            没有被覆盖的像素时，返回 `DEPTH_MISS_M`、0.0 和覆盖比例。

            不是 None。

"""
        mesh_id = runtime.load_mesh_trimesh(mesh_in, name="aspect")
        _rgb, rendered = runtime.render_tiles(
            mesh_id, pose_fixed, bbox, K, height, width, tile,
        )
        render_z = rendered[0].detach().float().cpu().numpy()
        covered = valid & (render_z > 1.0e-4)
        cover = float(covered.sum()) / float(valid.sum())
        if int(covered.sum()) == 0:
            return float(DEPTH_MISS_M), 0.0, cover
        residual = render_z[covered] - measured[covered]
        mad = float(np.median(np.abs(residual - np.median(residual))))
        loss = mad + float(DEPTH_MISS_M) * (1.0 - cover)
        return loss, mad, cover

    def axis_direction(scaled, axis):
        """
        # Project both ends of one mesh axis into the image.

            `scaled` is a trimesh.

            The axis ends are the bounding-box min and max on `axis`, with the other coordinates at the box center.

            They are transformed by the closed-over pose into the camera frame and projected with `K`. It is not None.

            `axis` is the integer vertex column 0, 1, or 2.

            It is not None.

        ## Returns

            - Returns the pixel direction, its length in pixels, and the two endpoints.
            - A degenerate axis or a non-positive camera z returns `(None, 0.0, None, None)`.

        ---

        # 把一条网格轴的两端投到图像上。

            `scaled` 是 trimesh。

            轴的两端是 `axis` 上包围盒的最小值和最大值，其余坐标用盒子中心。

            它们由闭包里的位姿变到相机系，再用 `K` 投影。

            不是 None。

            `axis` 是整数顶点列 0、1 或 2。

            不是 None。

        ## 返回

            - 返回像素方向、以像素为单位的长度，以及两个端点。
            - 轴退化或相机 z 不是正数时，返回 `(None, 0.0, None, None)`。

"""
        # Both ends of this object axis, projected into the image. The pose stays fixed.
        # 这条物体轴的两端投到图像上。位姿保持不动。
        vertices = np.asarray(scaled.vertices, dtype=np.float64)
        lo = vertices.min(axis=0)
        hi = vertices.max(axis=0)
        if float(hi[axis] - lo[axis]) < 1.0e-6:
            return None, 0.0, None, None
        center = 0.5 * (lo + hi)
        ends = np.stack([center, center], axis=0)
        ends[0, axis] = lo[axis]
        ends[1, axis] = hi[axis]
        rotation = pose_fixed[:3, :3]
        translation = pose_fixed[:3, 3]
        cam = ends @ rotation.T + translation.reshape(1, 3)
        z = cam[:, 2]
        if np.any(z < 1.0e-4):
            return None, 0.0, None, None
        uv = np.stack([
            K[0, 0] * cam[:, 0] / z + K[0, 2],
            K[1, 1] * cam[:, 1] / z + K[1, 2],
        ], axis=1)
        direction = uv[1] - uv[0]
        return direction, float(np.linalg.norm(direction)), uv[0], uv[1]

    def proposals_for(mesh_in, scales):
        """
        # Return outline scale steps as `(axis, factor, fraction)` triples, and the projected length in pixels of each mesh axis.

            An empty silhouette returns an empty step list and three zeros.

            `mesh_in` is a trimesh rasterized at the fixed pose.

            Its silhouette is compared with the closed-over mask on image x and image y, where x is right and y is down.

            It is not None.

            `scales` is a length-3 array of the scales already accepted.

            A proposed factor is kept only when both the step and the updated cumulative scale stay inside `AXIS_SCALE_MIN` and `AXIS_SCALE_MAX`. It is not None.

        ---

        # 返回轮廓缩放步骤，每步是 `(axis, factor, fraction)`，以及每条网格轴的投影长度，单位像素。

            剪影为空时返回空步骤列表和三个 0。

            `mesh_in` 是在固定位姿下栅格化的 trimesh。

            它的剪影与闭包里的 mask 在图像 x 和 y 上比较，x 向右，y 向下。

            不是 None。

            `scales` 是已经接受的尺度，长度为 3。

            只有这一步和更新后的累积尺度都落在 `AXIS_SCALE_MIN` 和 `AXIS_SCALE_MAX` 内时，提议的倍数才保留。

            不是 None。

"""
        silhouette, _photo = project_silhouette(blank, mesh_in, pose_fixed, K)
        sil_ys, sil_xs = np.nonzero(silhouette > 0)
        if len(sil_xs) == 0:
            return [], [0.0, 0.0, 0.0]
        sil_uv = np.stack([sil_xs.astype(np.float64), sil_ys.astype(np.float64)], axis=1)
        found = []
        extent_px = [0.0, 0.0, 0.0]
        directions = []
        endpoints = []
        for axis in (0, 1, 2):
            direction, projected_px, uv_lo, uv_hi = axis_direction(mesh_in, axis)
            extent_px[axis] = projected_px
            directions.append(direction)
            endpoints.append((uv_lo, uv_hi))
        longest_px = max(extent_px)
        measurable_px = max(
            float(AXIS_MIN_EXTENT_PX),
            float(AXIS_OBSERVABLE_FRACTION) * float(longest_px),
        )
        # Image width and image height, x right and y down. Outer pixels, not an inner band.
        # 图像宽和高，x 向右，y 向下。用外沿像素，不用内侧的一段。
        chosen = {}
        for component in (0, 1):
            inset = float(OUTLINE_STROKE_PX) * 0.5
            mask_low = float(mask_uv[:, component].min()) + inset
            mask_high = float(mask_uv[:, component].max()) - inset
            sil_low = float(sil_uv[:, component].min())
            sil_high = float(sil_uv[:, component].max())
            mesh_px = sil_high - sil_low
            if mesh_px < measurable_px or mask_high <= mask_low:
                continue
            best_axis = None
            best_component = 0.0
            for axis in (0, 1, 2):
                direction = directions[axis]
                if axis in locked or direction is None:
                    continue
                if extent_px[axis] < measurable_px:
                    continue
                component_px = abs(float(direction[component]))
                if component_px > best_component:
                    best_component = component_px
                    best_axis = axis
            if best_axis is None:
                continue
            uv_lo, uv_hi = endpoints[best_axis]
            factor, fraction = outline_scale_about(
                mask_low, mask_high, sil_low, sil_high,
                float(uv_lo[component]), float(uv_hi[component]),
            )
            # A factor outside the range is an unstable measurement, not a scale to saturate.
            # 比值超出范围是不稳定的测量，不把尺度顶到边界上。
            if factor < float(AXIS_SCALE_MIN) or factor > float(AXIS_SCALE_MAX):
                continue
            updated = float(scales[best_axis] * factor)
            if updated < float(AXIS_SCALE_MIN) or updated > float(AXIS_SCALE_MAX):
                continue
            if abs(factor - 1.0) < float(AXIS_SCALE_MIN_CHANGE):
                continue
            previous = chosen.get(best_axis)
            if previous is None or abs(np.log(factor)) > abs(np.log(previous[0])):
                chosen[best_axis] = (factor, fraction)
        found = [(axis, item[0], item[1]) for axis, item in chosen.items()]
        return found, extent_px

    def overflow_px(mesh_in):
        """
        # Return how many pixels the silhouette extends outside the mask, summed over the four image sides.

            An empty silhouette returns 1e6.

            `mesh_in` is a trimesh rasterized at the fixed pose on a blank image.

            The overflow uses the closed-over mask's row and column limits.

            It is not None.

        ---

        # 返回剪影伸出 mask 的像素数，四个图像边相加。

            剪影为空时返回 1e6。

            `mesh_in` 是在固定位姿下画到空白图像上的 trimesh。

            外伸量使用闭包里 mask 的行、列范围。

            不是 None。

"""
        silhouette, _photo = project_silhouette(blank, mesh_in, pose_fixed, K)
        sy, sx = np.nonzero(silhouette > 0)
        if len(sx) == 0:
            return 1.0e6
        return float(
            max(float(mask_ys.min() - sy.min()), 0.0)
            + max(float(sy.max() - mask_ys.max()), 0.0)
            + max(float(mask_xs.min() - sx.min()), 0.0)
            + max(float(sx.max() - mask_xs.max()), 0.0)
        )

    def apply_steps(search_in, full_in, scales_in, steps):
        """
        # Apply the accepted axis steps to the search mesh and the full mesh.

            `search_in` and `full_in` are trimesh copies.

            Each applied step calls `scale_axis_about` on both.

            They are not None.

            `scales_in` is copied to a float64 length-3 vector.

            A step is skipped when the updated component leaves `AXIS_SCALE_MIN` to `AXIS_SCALE_MAX`, or when the factor is within `AXIS_SCALE_MIN_CHANGE` of 1.

            It is not None.

            `steps` is a sequence of `(axis, factor, fraction)` triples.

            `axis` is 0, 1, or 2.

            `factor` and `fraction` are the values from `outline_scale_about`. It is not None.

        ## Returns

            - Returns the two meshes, the updated scale vector, and the steps that were applied.

        ---

        # 把接受的轴步骤加到搜索网格和完整网格上。

            `search_in` 和 `full_in` 是 trimesh 副本。

            每一个实际加上的步骤都会对两者调用 `scale_axis_about`。

            不是 None。

            `scales_in` 会被复制成 float64、长度为 3 的向量。

            更新后的分量超出 `AXIS_SCALE_MIN` 到 `AXIS_SCALE_MAX`，或倍数与 1 的差小于 `AXIS_SCALE_MIN_CHANGE` 时，这一步会被跳过。

            不是 None。

            `steps` 是 `(axis, factor, fraction)` 三元组的序列。

            `axis` 是 0、1 或 2。

            `factor` 和 `fraction` 来自 `outline_scale_about`。

            不是 None。

        ## 返回

            - 返回这两个网格、更新后的尺度向量，以及实际加上的步骤。

"""
        search_out = search_in
        full_out = full_in
        scales_out = np.array(scales_in, dtype=np.float64, copy=True)
        applied = []
        for axis, factor, fraction in steps:
            updated = float(scales_out[axis] * factor)
            if updated < float(AXIS_SCALE_MIN) or updated > float(AXIS_SCALE_MAX):
                continue
            if abs(float(factor) - 1.0) < float(AXIS_SCALE_MIN_CHANGE):
                continue
            search_out = scale_axis_about(search_out, axis, factor, fraction)
            full_out = scale_axis_about(full_out, axis, factor, fraction)
            scales_out[axis] = updated
            applied.append((int(axis), float(factor), float(fraction)))
        return search_out, full_out, scales_out, applied

    def keep_step(trial_loss, trial_overflow, base_loss, base_overflow):
        """
        # Return whether one outline step is kept.

            A trial whose depth loss is within `DEPTH_LOSS_MIN_GAIN_M` of the current loss is kept.

            Otherwise it is kept only when the overflow shrinks by more than half a pixel and the loss stays within `OUTLINE_DEPTH_SLACK_M` of the loss from before the loop.

            `trial_loss` and `base_loss` are depth losses in meters, as returned by `depth_loss`. They are not None.

            `trial_overflow` and `base_overflow` are the pixel overflow values from `overflow_px`. They are not None.

        ---

        # 返回一步轮廓更新是否保留。

            试验的深度损失与当前损失之差不超过 `DEPTH_LOSS_MIN_GAIN_M` 时保留。

            否则，只有外伸量缩小超过半个像素，并且损失相对循环开始前的损失不超过 `OUTLINE_DEPTH_SLACK_M` 时才保留。

            `trial_loss` 和 `base_loss` 是 `depth_loss` 返回的深度损失，单位米。

            不是 None。

            `trial_overflow` 和 `base_overflow` 是 `overflow_px` 返回的像素外伸量。

            不是 None。

"""
        if trial_loss <= base_loss + float(DEPTH_LOSS_MIN_GAIN_M):
            return True
        outline_tighter = trial_overflow + 0.5 < base_overflow
        depth_ok = trial_loss <= float(before_loss) + float(OUTLINE_DEPTH_SLACK_M)
        return bool(outline_tighter and depth_ok)

    search_work = search.copy()
    full_work = mesh.copy()
    accepted = np.ones(3, dtype=np.float64)
    pivots = [0.5, 0.5, 0.5]
    before_loss, before_mad, before_cover = depth_loss(search_work)
    current_loss = before_loss
    current_mad = before_mad
    current_cover = before_cover
    current_overflow = overflow_px(search_work)
    _found, extent_px = proposals_for(search_work, accepted)
    longest_px = max(extent_px)
    measurable_px = max(
        float(AXIS_MIN_EXTENT_PX),
        float(AXIS_OBSERVABLE_FRACTION) * float(longest_px),
    )
    observed_axes = [
        axis for axis, px in enumerate(extent_px) if float(px) >= measurable_px
    ]
    for _iteration in range(int(AXIS_FIT_ITERS)):
        found, _px = proposals_for(search_work, accepted)
        if not found:
            break
        trial_search, trial_full, trial_scales, applied = apply_steps(
            search_work, full_work, accepted, found,
        )
        if not applied:
            break
        trial_loss, trial_mad, trial_cover = depth_loss(trial_search)
        trial_overflow = overflow_px(trial_search)
        if keep_step(trial_loss, trial_overflow, current_loss, current_overflow):
            search_work = trial_search
            full_work = trial_full
            accepted = trial_scales
            for axis, _factor, fraction in applied:
                pivots[axis] = fraction
            current_loss = trial_loss
            current_mad = trial_mad
            current_cover = trial_cover
            current_overflow = trial_overflow
            continue
        changed = False
        found.sort(key=lambda item: abs(np.log(item[1])), reverse=True)
        running_search = search_work
        running_full = full_work
        running_scales = accepted
        running_loss = current_loss
        running_mad = current_mad
        running_cover = current_cover
        running_overflow = current_overflow
        for step in found:
            one_search, one_full, one_scales, one_applied = apply_steps(
                running_search, running_full, running_scales, [step],
            )
            if not one_applied:
                continue
            one_loss, one_mad, one_cover = depth_loss(one_search)
            one_overflow = overflow_px(one_search)
            if keep_step(one_loss, one_overflow, running_loss, running_overflow):
                running_search = one_search
                running_full = one_full
                running_scales = one_scales
                running_loss = one_loss
                running_mad = one_mad
                running_cover = one_cover
                running_overflow = one_overflow
                pivots[step[0]] = step[2]
                changed = True
        if not changed:
            break
        search_work = running_search
        full_work = running_full
        accepted = running_scales
        current_loss = running_loss
        current_mad = running_mad
        current_cover = running_cover
        current_overflow = running_overflow
    adjusted = mesh if np.allclose(accepted, 1.0) else full_work
    if not np.allclose(accepted, 1.0):
        current_loss, current_mad, current_cover = depth_loss(search_work)
        current_overflow = overflow_px(search_work)
    print(
        "aspect", [round(float(value), 3) for value in accepted],
        "pivot", [round(float(value), 3) for value in pivots],
        "extent_px", [round(float(value), 1) for value in extent_px],
        "observed", observed_axes,
        "locked", sorted(locked),
        "overflow_px", round(float(current_overflow), 1),
        "loss_m", round(float(current_loss), 4),
        "before_m", round(float(before_loss), 4),
        "mad_m", round(float(current_mad), 4),
        "cover", round(float(current_cover), 3),
        flush=True,
    )
    return adjusted, {
        "scales": [float(value) for value in accepted],
        "pivots": [float(value) for value in pivots],
        "extent_px": [float(value) for value in extent_px],
        "observed_axes": observed_axes,
        "overflow_px": float(current_overflow),
        "depth_loss_m": float(current_loss),
        "depth_loss_before_m": float(before_loss),
        "depth_mad_m": float(current_mad),
        "depth_cover": float(current_cover),
    }


def vertex_colors_u8(mesh):
    """
    # Return uint8 RGB of shape `(N, 3)`, or None when the mesh has no per-vertex color or the color does not vary.

        `mesh` is a trimesh.

        Colors come from `mesh.visual.vertex_colors`. A missing attribute, a row count other than the vertex count, or a zero standard deviation returns None.

        Floating colors whose maximum is at most 1 are multiplied by 255.

        Only the first three columns are kept.

    ---

    # 返回形状 `(N, 3)` 的 uint8 RGB。

        网格没有逐顶点颜色，或颜色没有变化时，返回 None。

        `mesh` 是 trimesh。

        颜色来自 `mesh.visual.vertex_colors`。

        属性缺失、行数不等于顶点数，或标准差为 0 时返回 None。

        最大值不超过 1 的浮点颜色会乘 255。

        只保留前三列。

"""
    visual = mesh.visual
    if not hasattr(visual, "vertex_colors") or visual.vertex_colors is None:
        return None
    raw = np.asarray(visual.vertex_colors)
    if raw.ndim != 2 or len(raw) != len(mesh.vertices):
        return None
    if np.issubdtype(raw.dtype, np.floating) and float(raw.max()) <= 1.0:
        raw = raw * 255.0
    colors = raw[:, :3].astype(np.uint8)
    if int(colors.std()) == 0:
        return None
    return colors


def transfer_vertex_colors(src_vertices, src_colors, dst_vertices):
    """
    # Return the nearest source color for each destination vertex.

        `src_vertices` and `dst_vertices` are arrays converted to float64.

        A KD-tree is built on the source, and each destination row queries one neighbor.

        They are not None.

        The function does not name a unit beyond the coordinates already stored on the mesh.

        `src_colors` is indexed by those neighbor indices and returned.

        Its row count must match the source vertices.

        It is not None.

    ---

    # 返回每个目标顶点对应的最近源颜色。

        `src_vertices` 和 `dst_vertices` 是转成 float64 的数组。

        源点用来建 KD 树，每个目标行查询一个近邻。

        不是 None。

        除了网格上已经保存的坐标，这个函数不再另写单位。

        `src_colors` 会按这些近邻下标取出并返回。

        它的行数必须和源顶点一致。

        不是 None。

"""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(src_vertices, dtype=np.float64))
    _dist, index = tree.query(np.asarray(dst_vertices, dtype=np.float64), k=1)
    return src_colors[index]


def texture_map(mesh):
    """
    # Return the UV array and the base-color image, or `(None, None)` when either is missing or the UV row count differs from the vertex count.

        `mesh` is a trimesh.

        UV is `visual.uv`, converted to float32.

        The image is `material.baseColorTexture`, or `material.image` when the first is missing.

        It is not None.

    ---

    # 返回 UV 数组和底色图像。

        二者缺一，或 UV 行数与顶点数不同时，返回 `(None, None)`。

        `mesh` 是 trimesh。

        UV 是 `visual.uv`，会转成 float32。

        图像是 `material.baseColorTexture`，若它缺失则用 `material.image`。

        不是 None。

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
    uv = np.asarray(uv, dtype=np.float32)
    if len(uv) != len(mesh.vertices):
        return None, None
    return uv, image


def save_viewer_mesh(mesh, path_stem):
    """
    # Write the page mesh as `path_stem.json` and `path_stem.bin`. A mesh with UV also writes `path_stem.jpg`. Returns None.

        Vertex positions are float32 and face indices are uint32.

        `mesh` is a trimesh.

        With UV, the binary is vertices, faces, then UV.

        Without UV, a mesh above 20000 faces is simplified to 20000, and the binary is vertices, faces, then uint8 RGB.

        Missing colors become 180.

        It is not None.

        `path_stem` is a path without an extension.

        The JSON records `vertices`, `faces`, `mode` as `uv` or `vertex`, and, for UV, the JPEG name.

        It is not None.

    ---

    # 把页面网格写成 `path_stem.json` 和 `path_stem.bin`。

        有 UV 的网格还会写 `path_stem.jpg`。

        返回 None。

        顶点位置是 float32，面下标是 uint32。

        `mesh` 是 trimesh。

        有 UV 时，二进制依次是顶点、面和 UV。

        没有 UV 时，超过 20000 面的网格会简化到 20000，二进制依次是顶点、面和 uint8 RGB。

        没有颜色时用 180。

        不是 None。

        `path_stem` 是不带扩展名的路径。

        JSON 记录 `vertices`、`faces`、`mode`（`uv` 或 `vertex`），UV 模式还会记录 JPEG 文件名。

        不是 None。

"""
    import json

    uv, image = texture_map(mesh)
    if uv is not None:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        blob = vertices.tobytes() + faces.tobytes() + np.ascontiguousarray(uv).tobytes()
        image_name = os.path.basename(path_stem) + ".jpg"
        image.convert("RGB").save(path_stem + ".jpg", quality=85, optimize=True)
        meta = {
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "mode": "uv",
            "texture": image_name,
        }
    else:
        view = mesh.copy()
        source_colors = vertex_colors_u8(mesh)
        if len(view.faces) > 20000:
            view = view.simplify_quadric_decimation(20000)
        vertices = np.asarray(view.vertices, dtype=np.float32)
        faces = np.asarray(view.faces, dtype=np.uint32)
        colors = vertex_colors_u8(view)
        if colors is None and source_colors is not None:
            colors = transfer_vertex_colors(mesh.vertices, source_colors, vertices)
        if colors is None:
            colors = np.full((len(vertices), 3), 180, dtype=np.uint8)
        blob = vertices.tobytes() + faces.tobytes() + colors.tobytes()
        meta = {"vertices": int(len(vertices)), "faces": int(len(faces)), "mode": "vertex"}
    with open(path_stem + ".json", "w") as handle:
        json.dump(meta, handle)
    with open(path_stem + ".bin", "wb") as handle:
        handle.write(blob)


def project_silhouette(rgb, mesh, pose, K):
    """
    # Rasterize the mesh triangles into a silhouette.

        `rgb` sets the image height and width and is converted from RGB to BGR before the contour is drawn.

        It is not None.

        `mesh` supplies float64 vertices and int32 faces.

        A triangle is drawn only when every vertex has camera z above 1e-4 and the projected polygon meets the image.

        It is not None.

        `pose` is a 4×4 matrix.

        `pose[:3, :3]` and `pose[:3, 3]` transform vertices into the camera frame before `K` projects them.

        It is not None.

        `K` is the 3×3 camera matrix.

        `K[0, 0]`, `K[0, 2]`, `K[1, 1]`, and `K[1, 2]` produce integer pixel coordinates.

        It is not None.

    ## Returns

        - Returns the uint8 mask and a BGR photo with the contour drawn.

    ---

    # 把网格三角形栅格化成剪影。

        `rgb` 决定图像的高和宽，并在画轮廓之前从 RGB 转成 BGR。

        不是 None。

        `mesh` 提供 float64 顶点和 int32 面。

        只有每个顶点的相机 z 都大于 1e-4，并且投影多边形碰到图像时，三角形才会被画上。

        不是 None。

        `pose` 是 4×4 矩阵。

        `pose[:3, :3]` 和 `pose[:3, 3]` 先把顶点变到相机系，再由 `K` 投影。

        不是 None。

        `K` 是 3×3 相机矩阵。

        `K[0, 0]`、`K[0, 2]`、`K[1, 1]` 和 `K[1, 2]` 用来得到整数像素坐标。

        不是 None。

    ## 返回

        - 返回 uint8 mask，以及画了轮廓的 BGR 照片。

"""
    import cv2

    height, width = rgb.shape[:2]
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    cam = (pts @ pose[:3, :3].T) + pose[:3, 3].reshape(1, 3)
    z = cam[:, 2]
    u = np.full(len(pts), -1, dtype=np.int32)
    v = np.full(len(pts), -1, dtype=np.int32)
    front = z > 1.0e-4
    u[front] = np.rint(K[0, 0] * cam[front, 0] / z[front] + K[0, 2]).astype(np.int32)
    v[front] = np.rint(K[1, 1] * cam[front, 1] / z[front] + K[1, 2]).astype(np.int32)
    canvas = np.zeros((height, width), dtype=np.uint8)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    for tri in faces:
        if not bool(front[tri].all()):
            continue
        poly = np.stack([u[tri], v[tri]], axis=1)
        if poly[:, 0].max() < 0 or poly[:, 1].max() < 0 or poly[:, 0].min() >= width or poly[:, 1].min() >= height:
            continue
        cv2.fillConvexPoly(canvas, poly, 255)
    contours, _hier = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # Red denotes an estimated pose; green is reserved for ground truth.
    # 红色表示估计位姿，绿色仅用于真值。
    cv2.drawContours(bgr, contours, -1, (40, 60, 210), 2)
    return canvas, bgr


def render_mesh_crop(mesh, pose, K, height, width):
    """
    # Return the crop origin and an OpenGL render of the mesh inside its projected box.

        The color is uint8 BGR, and `hit` marks pixels whose rendered depth is above 1e-4.

        An empty silhouette raises RuntimeError.

        `mesh` and `pose` are forwarded to `project_silhouette` and to `runtime.render_tiles`. `pose` is reshaped to `(1, 4, 4)` float32 for the renderer.

        They are not None.

        `K` is the 3×3 camera matrix passed to the tile renderer.

        It is not None.

        `height` and `width` are the full-image sizes in pixels.

        They size the blank silhouette and the render call.

        The returned color and hit are resized to the padded box.

        They are not None.

    ---

    # 返回裁剪原点，以及网格在其投影框内的 OpenGL 渲染。

        颜色是 uint8 BGR，`hit` 标出渲染深度大于 1e-4 的像素。

        剪影为空时抛出 RuntimeError。

        `mesh` 和 `pose` 会传给 `project_silhouette` 和 `runtime.render_tiles`。

        给渲染器时，`pose` 会被 reshape 成 float32 的 `(1, 4, 4)`。

        不是 None。

        `K` 是传给块渲染器的 3×3 相机矩阵。

        不是 None。

        `height` 和 `width` 是整幅图像的像素尺寸。

        它们决定空白剪影和渲染调用的大小。

        返回的颜色和 hit 会缩放到带边距的框。

        不是 None。

"""
    import cv2
    from wapr.ogl import runtime_for

    blank = np.zeros((height, width, 3), dtype=np.uint8)
    silhouette, _photo = project_silhouette(blank, mesh, pose, K)
    ys, xs = np.nonzero(silhouette > 0)
    if len(xs) == 0:
        raise RuntimeError("pose render")
    pad = 8
    left = max(int(xs.min()) - pad, 0)
    top = max(int(ys.min()) - pad, 0)
    right = min(int(xs.max()) + pad + 1, width)
    bottom = min(int(ys.max()) + pad + 1, height)
    box_w = max(right - left, 1)
    box_h = max(bottom - top, 1)
    tile = int(max(box_w, box_h, 160))
    runtime = runtime_for(device)
    mesh_id = runtime.load_mesh_trimesh(mesh, name="pose-check")
    bbox = np.array([[left, top, right, bottom]], dtype=np.float32)
    pose_batch = np.asarray(pose, dtype=np.float32).reshape(1, 4, 4)
    rendered_rgb, rendered_depth = runtime.render_tiles(
        mesh_id, pose_batch, bbox, K, height, width, tile,
    )
    rgb = rendered_rgb[0].detach().float().cpu().numpy()
    depth = rendered_depth[0].detach().float().cpu().numpy()
    if float(np.nanmax(rgb)) > 1.5:
        rgb = rgb / 255.0
    color = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    hit = depth > 1.0e-4
    color = cv2.resize(color, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
    hit = cv2.resize(hit.astype(np.uint8), (box_w, box_h), interpolation=cv2.INTER_NEAREST) > 0
    return left, top, color, hit


def draw_dashed_segment(image, start, end, color, thickness, dash, gap):
    """
    # Draw one dashed straight segment on `image`. Returns None.

        `image` is the array passed to `cv2.line`. The function draws into it.

        It is not None.

        `start` and `end` are pixel coordinates.

        Index 0 is x and index 1 is y.

        They are not None.

        `color` is the `cv2.line` color.

        Callers pass a BGR triple.

        It is not None.

        `thickness`, `dash`, and `gap` are pixel lengths.

        `thickness` is converted with `int`. `dash` and `gap` are floats.

        They are not None.

    ## Returns

        - A segment shorter than 1 pixel returns without drawing.

    ---

    # 在 `image` 上画一条虚线直线。

        `image` 是传给 `cv2.line` 的数组。

        函数直接在上面画。

        不是 None。

        `start` 和 `end` 是像素坐标。

        下标 0 是 x，下标 1 是 y。

        不是 None。

        `color` 是 `cv2.line` 的颜色。

        调用者传入 BGR 三元组。

        不是 None。

        `thickness`、`dash` 和 `gap` 是像素长度。

        `thickness` 会用 `int` 转换。

        `dash` 和 `gap` 是浮点数。

        不是 None。

    ## 返回

        - 返回 None。
        - 短于 1 像素的线段不会画。

"""
    import cv2

    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length < 1.0:
        return
    step_x = (x1 - x0) / length
    step_y = (y1 - y0) / length
    pos = 0.0
    dash = float(dash)
    gap = float(gap)
    while pos < length:
        stop = min(pos + dash, length)
        p0 = (int(round(x0 + step_x * pos)), int(round(y0 + step_y * pos)))
        p1 = (int(round(x0 + step_x * stop)), int(round(y0 + step_y * stop)))
        cv2.line(image, p0, p1, color, int(thickness), cv2.LINE_AA)
        pos = stop + gap


def draw_zoom_callout(canvas, box, zoom_origin, zoom_size):
    """
    # Draw the dashed crop rectangle and the two dashed lines from its right corners to the magnified panel.

        `canvas` is the BGR image passed to `draw_dashed_segment`. The stroke uses the module callout thickness, dash, and gap, in pixels.

        It is not None.

        `box` is `(left, top, right, bottom)` in pixels on `canvas`. Each value is converted with `int`. It is not None.

        `zoom_origin` is `(x, y)`, the top-left pixel of the magnified panel.

        It is not None.

        `zoom_size` is `(width, height)` of that panel in pixels.

        The lower line ends at `y + height - 1`. It is not None.

    ## Returns

        - Returns None.

    ---

    # 画出虚线裁切矩形，以及从它右侧两角连到放大面板的两条虚线。

        `canvas` 是传给 `draw_dashed_segment` 的 BGR 图像。

        笔画使用模块里的标注粗细、实段和间隔，单位像素。

        不是 None。

        `box` 是 `canvas` 上的 `(left, top, right, bottom)`，单位像素。

        每个值都会用 `int` 转换。

        不是 None。

        `zoom_origin` 是 `(x, y)`，放大面板的左上角像素。

        不是 None。

        `zoom_size` 是该面板的 `(width, height)`，单位像素。

        下面那条线结束在 `y + height - 1`。

        不是 None。

    ## 返回

        - 返回 None。

"""
    left, top, right, bottom = [int(v) for v in box]
    zoom_x, zoom_y = [int(v) for v in zoom_origin]
    zoom_w, zoom_h = [int(v) for v in zoom_size]
    # Cyan callouts identify the magnified view, not a truth contour.
    # 青色引线标记放大区域，不表示真值轮廓。
    color = (190, 140, 40)
    thickness = int(CALLOUT_THICKNESS_PX)
    dash = int(CALLOUT_DASH_PX)
    gap = int(CALLOUT_GAP_PX)
    corners = ((left, top), (right, top), (right, bottom), (left, bottom))
    for index in range(4):
        draw_dashed_segment(canvas, corners[index], corners[(index + 1) % 4], color, thickness, dash, gap)
    draw_dashed_segment(canvas, (right, top), (zoom_x, zoom_y), color, thickness, dash, gap)
    draw_dashed_segment(
        canvas, (right, bottom), (zoom_x, zoom_y + zoom_h - 1), color, thickness, dash, gap,
    )


def draw_pose_render(rgb, mesh, pose, K, path):
    """
    # Draw the projected contour on the photo and the magnified render beside it, then write that canvas.

        `rgb` is the RGB image.

        Its height and width size the photo panel.

        It is not None.

        `mesh`, `pose`, and `K` are forwarded to `project_silhouette` and `render_mesh_crop`. `pose` is the 4×4 estimate and `K` is the 3×3 camera matrix.

        They are not None.

        `path` is the PNG path passed to `cv2.imwrite`. It is not None.

    ## Returns

        - Returns None.

    ---

    # 在照片上画投影轮廓，在旁边放放大后的渲染，然后写出这张画布。

        `rgb` 是 RGB 图像。

        它的高和宽决定照片面板的尺寸。

        不是 None。

        `mesh`、`pose` 和 `K` 会传给 `project_silhouette` 和 `render_mesh_crop`。

        `pose` 是 4×4 估计，`K` 是 3×3 相机矩阵。

        不是 None。

        `path` 是传给 `cv2.imwrite` 的 PNG 路径。

        不是 None。

    ## 返回

        - 返回 None。

"""
    import cv2

    height, width = rgb.shape[:2]
    _silhouette, view = project_silhouette(rgb, mesh, pose, K)
    left, top, color, hit = render_mesh_crop(mesh, pose, K, height, width)
    box_h, box_w = hit.shape[:2]
    render = np.full((box_h, box_w, 3), 244, dtype=np.uint8)
    render[hit] = color[hit]
    zoom_w = int(POSE_ZOOM_W)
    zoom_h = int(POSE_ZOOM_H)
    scale = min(float(zoom_w) / float(box_w), float(zoom_h) / float(box_h))
    fitted_w = max(int(round(box_w * scale)), 1)
    fitted_h = max(int(round(box_h * scale)), 1)
    fitted = cv2.resize(render, (fitted_w, fitted_h), interpolation=cv2.INTER_NEAREST)
    panel = np.full((zoom_h, zoom_w, 3), 244, dtype=np.uint8)
    panel_x = (zoom_w - fitted_w) // 2
    panel_y = (zoom_h - fitted_h) // 2
    panel[panel_y:panel_y + fitted_h, panel_x:panel_x + fitted_w] = fitted
    gap = 16
    canvas_h = max(height, zoom_h)
    canvas_w = width + gap + zoom_w
    canvas = np.full((canvas_h, canvas_w, 3), 244, dtype=np.uint8)
    photo_y = (canvas_h - height) // 2
    canvas[photo_y:photo_y + height, :width] = view
    zoom_top = (canvas_h - zoom_h) // 2
    canvas[zoom_top:zoom_top + zoom_h, width + gap:width + gap + zoom_w] = panel
    box_left, box_top = left, top + photo_y
    box_right, box_bottom = left + box_w - 1, top + photo_y + box_h - 1
    draw_zoom_callout(
        canvas,
        (box_left, box_top, box_right, box_bottom),
        (width + gap, zoom_top),
        (zoom_w, zoom_h),
    )
    cv2.imwrite(path, canvas)


def draw_pose_checks(estimator, mesh, frames, K, out_dir):
    """
    # Estimate a pose on each `POSE_CHECK_FRAMES` view, draw it, and return the per-frame rows.

        The mesh is not scaled again.

        A mask outside 800 to 30000 pixels raises RuntimeError.

        `estimator` receives `estimate_one_category_one_instance` with that frame's RGB, meter depth, `K`, the mesh, its diameter, and the point mask.

        It is not None.

        `mesh` is the trimesh used for every frame.

        Its diameter is `mesh_diameter_m` of the vertices.

        It is not None.

        `frames` is the path list from `frame_list`. Each check item indexes it.

        It is not None.

        `K` is the 3×3 camera matrix forwarded to `estimate_one_category_one_instance` and `draw_pose_render`. It is not None.

        `out_dir` is the directory for `pose_####.png`. It is not None.

    ---

    # 在 `POSE_CHECK_FRAMES` 的每一帧上估计位姿、画出来，并返回每一帧的记录。

        网格不再缩放。

        mask 像素数不在 800 到 30000 之间时抛出 RuntimeError。

        `estimator` 会用该帧的 RGB、米制深度、`K`、网格、它的对角线和点 mask 调用 `estimate_one_category_one_instance`。

        不是 None。

        `mesh` 是每一帧都使用的 trimesh。

        对角线是顶点的 `mesh_diameter_m`。

        不是 None。

        `frames` 是 `frame_list` 返回的路径列表。

        每个检查项用它做下标。

        不是 None。

        `K` 是传给 `estimate_one_category_one_instance` 和 `draw_pose_render` 的 3×3 相机矩阵。

        不是 None。

        `out_dir` 是存放 `pose_####.png` 的目录。

        不是 None。

"""
    diameter_m = mesh_diameter_m(mesh.vertices)
    rows = []
    pose_mesh = estimator.prepare_meshes([mesh])[0]
    for index, click_u, click_v in POSE_CHECK_FRAMES:
        rgb, depth_m = read_rgb_depth(frames[int(index)])
        click = (int(click_u), int(click_v))
        mask = segment_point(rgb, click)
        mask_px = int(mask.sum())
        if mask_px < 800 or mask_px > 30000:
            raise RuntimeError(f"pose check mask {index} {mask_px}")
        estimator.warmup_pose(rgb, depth_m, K, [{"mesh": pose_mesh, "diameter_m": diameter_m, "mask": mask}])
        result = estimator.estimate_one_category_one_instance(rgb, depth_m, K, pose_mesh, diameter_m, mask=mask)
        pose = np.asarray(result["pose_4x4"], dtype=np.float64)
        path = os.path.join(out_dir, f"pose_{int(index):04d}.png")
        draw_pose_render(rgb, mesh, pose, K, path)
        row = {
            "frame": int(index),
            "click_uv": [int(click_u), int(click_v)],
            "mask_px": mask_px,
            "score_6d": float(result["score_6d"]),
        }
        rows.append(row)
        print(
            "pose check", index,
            "mask_px", mask_px,
            "score_6d", round(row["score_6d"], 4),
            flush=True,
        )
    return rows


def draw_overlay(rgb, mesh, pose, K, path):
    """
    # Draw the projected mesh contour on the photo and write that BGR image.

        `rgb`, `mesh`, `pose`, and `K` are forwarded to `project_silhouette`. `rgb` is RGB uint8, `pose` is 4×4, and `K` is 3×3.

        They are not None.

        `path` is the PNG path passed to `cv2.imwrite`. It is not None.

    ## Returns

        - Returns None.

    ---

    # 在照片上画投影后的网格轮廓，并写出这张 BGR 图像。

        `rgb`、`mesh`、`pose` 和 `K` 会传给 `project_silhouette`。

        `rgb` 是 RGB uint8，`pose` 是 4×4，`K` 是 3×3。

        不是 None。

        `path` 是传给 `cv2.imwrite` 的 PNG 路径。

        不是 None。

    ## 返回

        - 返回 None。

"""
    import cv2

    _mask, bgr = project_silhouette(rgb, mesh, pose, K)
    cv2.imwrite(path, bgr)




if __name__ == "__main__":
    # Run the script stages directly in the entry block.
    # 在入口块中直接执行脚本各阶段。
    import json
    import cv2
    import torch
    from wapr.estimator import WAPREstimator

    seq_dir = os.path.join(DATA_ROOT, SEQUENCE)
    frames = frame_list(seq_dir)
    K = np.loadtxt(os.path.join(seq_dir, "cam_K.txt"), dtype=np.float64).reshape(3, 3)
    rgb, depth_m = read_rgb_depth(frames[BUILD_FRAME])
    print("segment", CLICK_UV, flush=True)
    mask = segment_point(rgb, CLICK_UV)
    print("reconstruct", int(mask.sum()), "px", flush=True)
    mesh = reconstruct_mesh(rgb, mask)
    extent_m = depth_extent_m(depth_m, mask, K)
    print("depth extent m", round(extent_m, 4), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUT_DIR, "mask.png"), mask * 255)
    marked = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    tint = marked.copy()
    tint[mask > 0] = (tint[mask > 0] * 0.45 + np.array([40, 180, 60]) * 0.55).astype(np.uint8)
    marked[mask > 0] = tint[mask > 0]
    cv2.circle(marked, (int(CLICK_UV[0]), int(CLICK_UV[1])), 6, (0, 80, 255), -1)
    cv2.imwrite(os.path.join(OUT_DIR, "prompt.png"), marked)

    skip_align = os.environ.get("RECON_SKIP_ALIGN", "") == "1"
    report = {
        "sequence": SEQUENCE,
        "build_frame": BUILD_FRAME,
        "later_frame": LATER_FRAME,
        "click_uv": [int(CLICK_UV[0]), int(CLICK_UV[1])],
        "depth_extent_m": extent_m,
        "mask_px": int(mask.sum()),
    }
    if skip_align:
        save_viewer_mesh(mesh, os.path.join(OUT_DIR, "mesh"))
        report["aligned"] = False
        with open(os.path.join(OUT_DIR, "result.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        print("wrote", OUT_DIR, "without alignment", flush=True)
        raise SystemExit(0)

    estimator = WAPREstimator(device=device)
    best, rows = align_scales(estimator, rgb, depth_m, mask, K, mesh, extent_m)
    # The pose stays this estimate. The mesh is scaled onto the mask, and the pose is not moved.
    # 位姿保持这次估计。网格缩到 mask 上，位姿不动。
    build_pose = np.asarray(best["pose_4x4"], dtype=np.float64)
    adjusted, aspect = fit_axis_scales(
        best["mesh"], build_pose, depth_m, mask, K,
    )
    build_score = float(best["score_6d"])
    diameter_m = mesh_diameter_m(adjusted.vertices)
    later_rgb, later_depth = read_rgb_depth(frames[LATER_FRAME])
    later_mask = segment_point(later_rgb, CLICK_UV)
    later_mesh = estimator.prepare_meshes([adjusted])[0]
    estimator.warmup_pose(later_rgb, later_depth, K,
                          [{"mesh": later_mesh, "diameter_m": diameter_m, "mask": later_mask}])
    later = estimator.estimate_one_category_one_instance(
        later_rgb, later_depth, K, later_mesh, diameter_m, mask=later_mask,
    )
    # Axes the first frame could measure stay fixed. The later frame may set the rest.
    # 第一帧能量到的轴保持不动。后面一帧可以补上其余的轴。
    seen = list(aspect["observed_axes"])
    later_pose = np.asarray(later["pose_4x4"], dtype=np.float64)
    adjusted, aspect_later = fit_axis_scales(
        adjusted, later_pose, later_depth, later_mask, K, locked_axes=tuple(seen),
    )
    later_changed = not np.allclose(aspect_later["scales"], 1.0)
    if later_changed:
        # The later frame changed an axis the first frame did not see. Scale the build
        # outline again at the same build pose, and leave that new axis locked.
        # 后面一帧改了第一帧没看到的轴。在原来的位姿下再贴一次轮廓，这条新轴锁住。
        later_axes = [
            axis for axis, scale in enumerate(aspect_later["scales"])
            if abs(float(scale) - 1.0) > 1.0e-4
        ]
        adjusted, snap = fit_axis_scales(
            adjusted, build_pose, depth_m, mask, K, locked_axes=tuple(later_axes),
        )
        frame_scales = (
            np.asarray(aspect["scales"], dtype=np.float64)
            * np.asarray(snap["scales"], dtype=np.float64)
        )
        aspect = snap
        aspect["scales"] = [float(value) for value in frame_scales]
    diameter_m = mesh_diameter_m(adjusted.vertices)
    final_scales = (
        np.asarray(aspect["scales"], dtype=np.float64)
        * np.asarray(aspect_later["scales"], dtype=np.float64)
    )
    draw_overlay(rgb, adjusted, build_pose, K, os.path.join(OUT_DIR, "overlay_build.png"))
    save_viewer_mesh(adjusted, os.path.join(OUT_DIR, "mesh"))
    draw_overlay(
        later_rgb, adjusted, later_pose, K, os.path.join(OUT_DIR, "overlay_later.png"),
    )
    pose_checks = draw_pose_checks(estimator, adjusted, frames, K, OUT_DIR)
    report["aligned"] = True
    report["chosen_factor"] = best["factor"]
    report["chosen_score_6d"] = best["score_6d"]
    report["aspect_scales"] = [float(value) for value in final_scales]
    report["frame_axis_scales"] = aspect["scales"]
    report["later_axis_scales"] = aspect_later["scales"]
    report["extent_px"] = aspect["extent_px"]
    report["later_extent_px"] = aspect_later["extent_px"]
    report["depth_loss_m"] = aspect["depth_loss_m"]
    report["depth_loss_before_m"] = aspect["depth_loss_before_m"]
    report["depth_mad_m"] = aspect["depth_mad_m"]
    report["depth_cover"] = aspect["depth_cover"]
    report["later_depth_loss_m"] = aspect_later["depth_loss_m"]
    report["later_depth_loss_before_m"] = aspect_later["depth_loss_before_m"]
    report["diameter_m"] = diameter_m
    report["build_score_6d"] = build_score
    report["scales"] = [
        {
            "factor": row["factor"],
            "score_6d": row["score_6d"],
            "mask_iou": row["mask_iou"],
        }
        for row in rows
    ]
    report["later_score_6d"] = float(later["score_6d"])
    report["pose_checks"] = pose_checks
    with open(os.path.join(OUT_DIR, "result.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    torch.cuda.empty_cache()
    print("wrote", OUT_DIR, flush=True)
