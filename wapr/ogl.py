# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

from __future__ import annotations

# One mesh, many pose tiles, then the network crop pair.
# 一个网格、多块位姿图，再组成网络的裁剪对。
# Vertices, depth, and poses are meters. The camera is OpenCV, +Z forward, y down.
# 顶点、深度和位姿是米。相机是 OpenCV，+Z 朝前，y 向下。
# UV V is flipped once: image-top V becomes the renderer V. Tiles do not occlude each other.
# UV 的 V 只翻转一次：图像上方的 V 变成渲染器的 V。各块之间不互相遮挡。
# The first inference compiles OGL when its library is missing, then loads it.
# 第一次推理时，如果 OGL 库还没编出来，就先编译再载入。
# Rendering stays on OGL. EGL/GL failing does not switch rasterizers.
# 渲染固定用 OGL。EGL/GL 起不来时不换光栅器。
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np


DEFAULT_MAX_TEXTURE_SIZE = 1024
# Gray used when a mesh has no usable vertex color, in 0-1.
# 网格没有可用顶点色时用的灰，范围 0-1。
DEFAULT_UNTEXTURED_VERTEX_COLOR = 0.5
# trimesh writes this constant when a mesh has no real vertex color.
# trimesh 在网格没有真实顶点色时会写成这个常数。
TRIMESH_PLACEHOLDER_VERTEX_COLOR = np.float32(102.0 / 255.0)


def _downscale_texture_rgb(img: np.ndarray, max_size: int = DEFAULT_MAX_TEXTURE_SIZE) -> np.ndarray:
    """
    # Shrink one RGB image.

        The long side is at most max_size pixels.

        uint8 stays uint8.

        Any other dtype is resized as float32 and cast back to the input dtype.

        Channels after the third are dropped on the resize path.

    ## Args

        - img: the image. A 3-D array whose last dimension is at least 3 can be resized. Any other shape is returned as an array without resizing.
        - max_size: the long-side limit in pixels. The default is 1024. A value at or below 0 returns img unchanged. The resize uses cv2 INTER_AREA.

    ---

    # 缩小一张 RGB 图像。

        长边不超过 max_size 像素。

        uint8 保持 uint8。

        其他 dtype 先按 float32 缩放，再转回输入 dtype。

        缩放那一支会丢掉第三通道之后的通道。

    ## 参数

        - img: 图像。最后一维至少为 3 的三维数组才可以缩放。其他形状按数组返回，不缩放。
        - max_size: 长边上限，单位像素。默认是 1024。小于或等于 0 时，img 原样返回。缩放使用 cv2 的 INTER_AREA。

"""
    if max_size <= 0:
        return img
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if max(h, w) <= max_size:
        return arr
    import cv2

    scale = float(max_size) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if arr.dtype == np.uint8:
        return cv2.resize(arr[..., :3], (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(arr[..., :3].astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized.astype(arr.dtype, copy=False)


def resize_texture_rgb_exact(tex_rgb, tex_w: int, tex_h: int, tw: int, th: int):
    """
    # Reshape a flat RGB buffer to tex_h by tex_w, then resize it to tw by th.

    ## Args

        - tex_rgb: a flat buffer, or None. None returns None. A usable buffer is cast to uint8 and must contain tex_h * tex_w * 3 values.
        - tex_w: the stored width in pixels. A value at or below 0 returns None.
        - tex_h: the stored height in pixels. A value at or below 0 returns None.
        - tw: the target width in pixels. A value at or below 0 returns None.
        - th: the target height in pixels. A value at or below 0 returns None. A target size other than tex_w by tex_h is resized with cv2 INTER_AREA.

    ## Returns

        - The return is a flat uint8 buffer of length tw * th * 3, or None.

    ---

    # 把扁平 RGB 还原成 tex_h 乘 tex_w，再缩放到 tw 乘 th。

    ## 参数

        - tex_rgb: 扁平缓冲，或 None。None 返回 None。可用的缓冲会转成 uint8，并且必须含有 tex_h * tex_w * 3 个值。
        - tex_w: 存下来的宽度，单位像素。小于或等于 0 时返回 None。
        - tex_h: 存下来的高度，单位像素。小于或等于 0 时返回 None。
        - tw: 目标宽度，单位像素。小于或等于 0 时返回 None。
        - th: 目标高度，单位像素。小于或等于 0 时返回 None。目标尺寸不是 tex_w 乘 tex_h 时，用 cv2 的 INTER_AREA 缩放。

    ## 返回

        - 返回长度为 tw * th * 3 的扁平 uint8 缓冲，或 None。

"""
    import cv2

    tw_i, th_i = int(tw), int(th)
    if tex_rgb is None or int(tex_w) <= 0 or int(tex_h) <= 0 or tw_i <= 0 or th_i <= 0:
        return None
    img = np.asarray(tex_rgb, dtype=np.uint8).reshape(int(tex_h), int(tex_w), 3)
    if int(tex_w) != tw_i or int(tex_h) != th_i:
        img = cv2.resize(img, (tw_i, th_i), interpolation=cv2.INTER_AREA)
    return img.reshape(-1).astype(np.uint8)


class UvLayoutError(ValueError):
    """
    # The UV count is neither the vertex count nor three times the face count.

        This is a ValueError.

        The count is what classify_uv_layout and resolve_uv_layout compare with the mesh.

        Raised by classify_uv_layout and resolve_uv_layout.

    ---

    # UV 数量既不是顶点数，也不是面数的三倍。

        它是 ValueError。

        classify_uv_layout 和 resolve_uv_layout 用这个数量和网格比较。

        由 classify_uv_layout 和 resolve_uv_layout 抛出。
    """
    pass


@dataclass
class MeshUploadArrays:
    """
    # Arrays the GL loader receives for one mesh.

        pack_trimesh_upload and pack_array_upload build it.

        GpuRenderRuntime._upload_item reshapes the arrays and uploads them.

    ## Returns

        - Vertices: meters.
        - vertices: float32 and reshapes to (V, 3). The pack functions store meters.
        - faces: uint32, three indices per triangle. The pack functions store it flat. Upload reshapes it to (F, 3).
        - normals: a flat float32 buffer, three components per vertex, or None.
        - vertex_colors: a flat float32 buffer in 0-1, three components per vertex, or None.
        - uvs: float32 (N, 2), or None.
        - N: the vertex count when uv_per_corner is false, and three times the face count when it is true.
        - texture_rgb: flat uint8 RGB of length tex_w * tex_h * 3, or None.
        - tex_w: the texture width in pixels. 0 means there is no texture.
        - tex_h: the texture height in pixels. 0 means there is no texture.
        - uv_per_corner: true for wedge UV, one coordinate on each triangle corner.
        - force_flat_normal: forwarded to the native loader. The default is False.

    ---

    # GL 载入器收到的一个网格的数组。

        pack_trimesh_upload 和 pack_array_upload 构建它。

        GpuRenderRuntime._upload_item 把这些数组 reshape 之后上传。

    ## 返回

        - 顶点单位是米。
        - vertices: float32，可 reshape 成 (V, 3)。打包函数按米存下它们。
        - faces: uint32，每个三角形三个下标。打包函数把它存成一维。上传时 reshape 成 (F, 3)。
        - normals: 扁平 float32 缓冲，每个顶点三个分量，或 None。
        - vertex_colors: 0-1 的扁平 float32 缓冲，每个顶点三个分量，或 None。
        - uvs: float32 (N, 2)，或 None。
        - uv_per_corner: 假时 N 是顶点数，为真时 N 是面数的三倍。
        - texture_rgb: 长度为 tex_w * tex_h * 3 的扁平 uint8 RGB，或 None。
        - tex_w: 纹理宽度，单位像素。0 表示没有纹理。
        - tex_h: 纹理高度，单位像素。0 表示没有纹理。
        - uv_per_corner: 真表示 wedge UV，三角形每个角一个坐标。
        - force_flat_normal 会传给渲染模块的网格加载器。
        - 默认是 False。

"""
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray]
    vertex_colors: Optional[np.ndarray]
    uvs: Optional[np.ndarray]
    texture_rgb: Optional[np.ndarray]
    tex_w: int
    tex_h: int
    uv_per_corner: bool
    force_flat_normal: bool = False


def _normals_usable(normals: np.ndarray, num_vertices: int) -> bool:
    """
    # Return True when every vertex has one finite, non-zero normal.

        normals reshapes to (N, 3) float32.

        num_vertices is the required row count.

    ## Returns

        - The return is a Python bool.
        - It is True only when the row count matches, every value is finite, and the shortest vector is longer than 1e-6.

    ---

    # 每个顶点都有一条有限且非零的法线时返回 True。

        normals 会 reshape 成 (N, 3) float32。

        num_vertices 是要求的行数。

    ## 返回

        - 返回 Python bool。
        - 只有行数一致、每个值都有限，且最短向量长于 1e-6 时才是 True。

"""
    n = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    if n.shape[0] != num_vertices:
        return False
    if not np.isfinite(n).all():
        return False
    return bool(np.linalg.norm(n, axis=1).min() > 1e-6)


def ensure_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    # Keep usable normals, otherwise recompute them.

        A failed recompute becomes +Y.

    ## Args

        - vertices: the mesh positions, read as float64.
        - faces: triangle indices, reshaped to (F, 3) int64.
        - normals: an existing normal array, or None. None, a bad count, a non-finite value, or a near-zero vector is replaced.

    ## Returns

        - The return is flat float32, three components per vertex.
        - The fallback direction is (0, 1, 0).

    ---

    # 能用的法线就保留，否则重新计算。

        算不出来就用 +Y。

    ## 参数

        - vertices: 网格位置，按 float64 读取。
        - faces: 三角形下标，reshape 成 (F, 3) int64。
        - normals: 已有的法线数组，或 None。None、数量不对、非有限值，或接近零的向量都会被替换。

    ## 返回

        - 返回扁平 float32，每个顶点三个分量。
        - 兜底方向是 (0, 1, 0)。

"""
    verts = np.asarray(vertices, dtype=np.float64)
    faces_i = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    nv = int(verts.shape[0])
    if normals is not None and _normals_usable(normals, nv):
        return np.asarray(normals, dtype=np.float32).reshape(-1, 3).reshape(-1)

    import trimesh

    tmp = trimesh.Trimesh(vertices=verts, faces=faces_i, process=False)
    tmp.fix_normals()
    out = np.asarray(tmp.vertex_normals, dtype=np.float32)
    if not _normals_usable(out, nv):
        out = np.zeros((nv, 3), dtype=np.float32)
        out[:, 1] = 1.0
    return out.reshape(-1)


def _vertex_colors_usable(vertex_colors: np.ndarray, num_vertices: int) -> bool:
    """
    # Return True when the color array has one finite RGB triple per vertex.

        vertex_colors reshapes to (N, 3) float32.

        num_vertices is the required row count.

    ## Returns

        - The return is a Python bool.
        - The row count must match, and every value must be finite.

    ---

    # 颜色数组为每个顶点提供一个有限 RGB 三元组时返回 True。

        vertex_colors 会 reshape 成 (N, 3) float32。

        num_vertices 是要求的行数。

    ## 返回

        - 返回 Python bool。
        - 行数必须一致，并且每个值都必须有限。

"""
    c = np.asarray(vertex_colors, dtype=np.float32).reshape(-1, 3)
    # ndarray.all() is numpy.bool_, not a Python bool.
    # ndarray.all() 得到的是 numpy.bool_，不是 Python bool。
    return bool(c.shape[0] == num_vertices and np.isfinite(c).all())


def _is_placeholder_vertex_color(vertex_colors: np.ndarray) -> bool:
    """
    # Return True for empty, uniform black, or the constant trimesh writes when no vertex color exists.

        vertex_colors reshapes to (N, 3) float32.

        An empty array is True.

        A color that is not uniform within 1e-3 is False.

        Uniform black means the first channel is at or below 1e-3.

        The trimesh constant is 102/255, accepted within 0.02.

    ---

    # 空数组、整片纯黑，或 trimesh 在没有顶点色时写下的那个常数，返回 True。

        vertex_colors 会 reshape 成 (N, 3) float32。

        空数组是 True。

        在 1e-3 内不均匀的颜色是 False。

        均匀纯黑是指第一通道小于或等于 1e-3。

        trimesh 的常数是 102/255，容差 0.02。

"""
    c = np.asarray(vertex_colors, dtype=np.float32).reshape(-1, 3)
    if c.size == 0:
        return True
    if not np.allclose(c, c[0], atol=1e-3):
        return False
    v = float(c[0, 0])
    if v <= 1e-3:
        return True
    return abs(v - float(TRIMESH_PLACEHOLDER_VERTEX_COLOR)) < 0.02


def uv_is_constant_placeholder(
    uv: Optional[np.ndarray],
    value: float = 0.5,
    atol: float = 1e-3,
) -> bool:
    """
    # Return True when every UV equals the same value.

        The default value is 0.5.

    ## Args

        - uv: (N, 2), a flat buffer, or None. None returns False. An empty array returns True.
        - value: the constant being tested. The default is 0.5.
        - atol: the absolute tolerance. The default is 1e-3.

    ## Returns

        - The return is a Python bool from allclose.

    ---

    # 每个 UV 都等于同一个值时返回 True。

        默认这个值是 0.5。

    ## 参数

        - uv: (N, 2)、扁平缓冲，或 None。None 返回 False。空数组返回 True。
        - value: 被测试的那个常数。默认是 0.5。
        - atol: 绝对容差。默认是 1e-3。

    ## 返回

        - 返回值是 allclose 得到的 Python bool。

"""
    if uv is None:
        return False
    arr = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    if arr.size == 0:
        return True
    return bool(np.allclose(arr, float(value), atol=float(atol)))


def discard_placeholder_uv(
    uv: Optional[np.ndarray],
    uv_per_corner: bool,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    # Drop UV that is missing or constant, and clear the per-corner flag with it.

    ## Args

        - uv: the UV array, or None. A constant array uses uv_is_constant_placeholder.
        - uv_per_corner: the wedge flag kept when the UV stays. It becomes False when the UV is dropped.

    ## Returns

        - The return is (uv, True or False) when the UV stays, and (None, False) when it is dropped.

    ---

    # 丢掉缺失或常数的 UV，并一并清掉逐角标志。

    ## 参数

        - uv: UV 数组，或 None。常数数组通过 uv_is_constant_placeholder 判断。
        - uv_per_corner: UV 保留时留下的 wedge 标志。UV 被丢掉时它变成 False。

    ## 返回

        - UV 留下时返回 (uv, True 或 False)。
        - 丢掉时返回 (None, False)。

"""
    if uv is None or uv_is_constant_placeholder(uv):
        return None, False
    return uv, bool(uv_per_corner)


def texture_solid_rgb(
    texture_rgb: Optional[np.ndarray],
    tex_w: int = 0,
    tex_h: int = 0,
) -> Optional[np.ndarray]:
    """
    # Return the mean texture color in 0-1.

        Values above 1 are treated as 0-255.

    ## Args

        - texture_rgb: an array, a flat buffer, a PIL image with convert, or None. None, an empty array, and an unrecognized shape return None.
        - tex_w: the width used to reshape a flat buffer. The default is 0. Together with a positive tex_h, the first tex_w * tex_h * 3 values are used.
        - tex_h: the height used to reshape a flat buffer. The default is 0. When either size is not positive, a length divisible by 3 is reshaped to (N, 3).

    ## Returns

        - The return is (3,) float32 in 0-1, the mean over every RGB triple, or None.

    ---

    # 返回纹理平均色，范围 0-1。

        大于 1 的值按 0-255 处理。

    ## 参数

        - texture_rgb: 数组、扁平缓冲、带 convert 的 PIL 图像，或 None。None、空数组和认不出的形状返回 None。
        - tex_w: 把扁平缓冲还原成图像时用的宽度。默认是 0。它和大于 0 的 tex_h 一起决定取前 tex_w * tex_h * 3 个值。
        - tex_h: 把扁平缓冲还原成图像时用的高度。默认是 0。任一尺寸不是正数时，长度能被 3 整除就 reshape 成 (N, 3)。

    ## 返回

        - 返回 (3,) float32，范围 0-1，是所有 RGB 三元组的平均，或 None。

"""
    if texture_rgb is None:
        return None
    img = texture_rgb
    # A PIL image can arrive even though the annotation is an array. Look up convert by name.
    # 注解是数组，但这里也可能收到 PIL 图像。按名字取出 convert。
    convert = getattr(img, "convert", None)
    if callable(convert) and hasattr(img, "size"):
        img = np.asarray(convert("RGB"))
    arr = np.asarray(img)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        n = int(arr.size)
        w, h = int(tex_w), int(tex_h)
        if w > 0 and h > 0 and n >= w * h * 3:
            arr = arr[: w * h * 3].reshape(h, w, 3)
        elif n % 3 == 0:
            arr = arr.reshape(-1, 3)
        else:
            return None
    if arr.ndim == 3:
        rgb = arr[..., :3]
    elif arr.ndim == 2 and int(arr.shape[-1]) >= 3:
        rgb = arr[:, :3]
    else:
        return None
    rgb = np.asarray(rgb, dtype=np.float32)
    if float(np.max(rgb)) > 1.0 + 1e-3:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb.reshape(-1, 3).mean(axis=0).astype(np.float32)


def vertex_colors_from_texture_solid(
    num_vertices: int,
    texture_rgb: Optional[np.ndarray],
    tex_w: int = 0,
    tex_h: int = 0,
) -> Optional[np.ndarray]:
    """
    # Paint every vertex with the mean texture color and return a flat 0-1 buffer.

    ## Args

        - num_vertices: the vertex count. Zero or a negative count returns None.
        - texture_rgb: forwarded to texture_solid_rgb. A missing mean returns None.
        - tex_w: the texture width in pixels, forwarded to the mean. The default is 0.
        - tex_h: the texture height in pixels, forwarded to the mean. The default is 0.

    ## Returns

        - The return is flat float32 of length 3 * num_vertices, or None.

    ---

    # 每个顶点都涂成纹理平均色，返回扁平的 0-1 缓冲。

    ## 参数

        - num_vertices: 顶点数。0 或负数返回 None。
        - texture_rgb 传给 texture_solid_rgb。
        - 没有平均色时返回 None。
        - tex_w: 纹理宽度，单位像素，传给平均色计算。默认是 0。
        - tex_h: 纹理高度，单位像素，传给平均色计算。默认是 0。

    ## 返回

        - 返回长度为 3 * num_vertices 的扁平 float32，或 None。

"""
    solid = texture_solid_rgb(texture_rgb, tex_w, tex_h)
    if solid is None or int(num_vertices) <= 0:
        return None
    return np.broadcast_to(solid, (int(num_vertices), 3)).copy().reshape(-1)


def upload_is_textured(upload: Optional[MeshUploadArrays]) -> bool:
    """
    # Return True when the upload has non-constant UV and a texture with positive width and height.

        texture_rgb must not be None, and both tex_w and tex_h must be positive.

        The return is a Python bool.

    ## Args

        - upload: a MeshUploadArrays, or None. None returns False. Missing or empty UV returns False. Constant UV returns False.

    ---

    # 上传对象有非常数 UV，且纹理的宽和高都大于 0 时，返回 True。

        texture_rgb 不能是 None，并且 tex_w 和 tex_h 都必须大于 0。

        返回 Python bool。

    ## 参数

        - upload: MeshUploadArrays，或 None。None 返回 False。UV 缺失或为空时返回 False。常数 UV 返回 False。

"""
    if upload is None:
        return False
    uvs = getattr(upload, "uvs", None)
    if uvs is None or np.asarray(uvs).size == 0:
        return False
    if uv_is_constant_placeholder(uvs):
        return False
    tex = getattr(upload, "texture_rgb", None)
    w = int(getattr(upload, "tex_w", 0) or 0)
    h = int(getattr(upload, "tex_h", 0) or 0)
    return tex is not None and w > 0 and h > 0


def apply_atlas_to_upload(upload: MeshUploadArrays, tw: int, th: int) -> MeshUploadArrays:
    """
    # Resize the texture to tw by th, or bake a constant UV into vertex color.

        Constant UV is replaced by the mean texture color on the vertices, in 0-1, and the texture fields are cleared.

        An upload that is not textured also clears those fields.

        No caller in WAPR.

    ## Args

        - upload: a MeshUploadArrays. The result is a shallow copy, so the caller's object stays unchanged until a field is replaced on the copy.
        - tw: the target width in pixels. A non-positive tw or th keeps a textured upload that still has usable UV.
        - th: the target height in pixels.

    ## Returns

        - The return is the MeshUploadArrays copy.

    ---

    # 把纹理缩到 tw 乘 th，或把常数 UV 烘成顶点色。

        常数 UV 会改成顶点上的纹理平均色，范围 0-1，纹理字段被清空。

        不是带纹理的上传也会清空这些字段。

        WAPR 里没有调用它。

    ## 参数

        - upload: MeshUploadArrays。结果是浅拷贝，因此调用方的对象保持不变，直到副本上的字段被替换。
        - tw: 目标宽度，单位像素。tw 或 th 不是正数时，仍有可用 UV 的带纹理上传会保持原样。
        - th: 目标高度，单位像素。

    ## 返回

        - 返回这份 MeshUploadArrays 副本。
    """
    import copy

    out = copy.copy(upload)
    verts = np.asarray(out.vertices)
    nv = int(verts.shape[0]) if verts.ndim == 2 else int(verts.size // 3)
    uvs = out.uvs
    if uvs is not None and uv_is_constant_placeholder(uvs):
        baked = vertex_colors_from_texture_solid(nv, out.texture_rgb, out.tex_w, out.tex_h)
        if baked is not None:
            out.vertex_colors = baked
        out.uvs = None
        out.uv_per_corner = False
        out.texture_rgb = None
        out.tex_w = 0
        out.tex_h = 0
        return out
    if not upload_is_textured(out):
        out.texture_rgb = None
        out.tex_w = 0
        out.tex_h = 0
        return out
    tw_i, th_i = int(tw), int(th)
    if tw_i <= 0 or th_i <= 0:
        return out
    tex = resize_texture_rgb_exact(out.texture_rgb, out.tex_w, out.tex_h, tw_i, th_i)
    if tex is None:
        out.texture_rgb = None
        out.tex_w = 0
        out.tex_h = 0
        return out
    out.texture_rgb = tex
    out.tex_w = tw_i
    out.tex_h = th_i
    return out


def should_apply_untextured_gray(
    *,
    has_texture: bool,
    has_uv: bool,
    vertex_colors: Optional[np.ndarray],
    num_vertices: int,
) -> bool:
    """
    # Return True when gray should replace a missing or placeholder vertex color.

    ## Args

        - has_texture: a bool. Together with has_uv True, it returns False.
        - has_uv: a bool. The gray replacement is for the case that does not have both a texture and UV.
        - vertex_colors: an array, or None. A usable color that is not a placeholder returns False. None, a bad count, a non-finite value, black, and the 102/255 constant continue to gray.
        - num_vertices: the vertex count passed to the color checks.

    ## Returns

        - The return is a Python bool.

    ---

    # 顶点色缺失或是占位色、需要改成灰色时，返回 True。

    ## 参数

        - has_texture: bool。它和 has_uv 同时为 True 时返回 False。
        - has_uv: bool。灰色替换用于不同时具备纹理和 UV 的情况。
        - vertex_colors: 数组，或 None。可用且不是占位色时返回 False。None、数量不对、非有限值、纯黑和 102/255 常数都会继续走到灰色。
        - num_vertices: 传给颜色检查的顶点数。

    ## 返回

        - 返回 Python bool。

"""
    if has_texture and has_uv:
        return False
    if has_texture and not has_uv:
        pass
    if vertex_colors is not None and _vertex_colors_usable(vertex_colors, num_vertices):
        if not _is_placeholder_vertex_color(vertex_colors):
            return False
    return True


def ensure_untextured_vertex_color(
    num_vertices: int,
    vertex_colors: Optional[np.ndarray],
    *,
    has_texture: bool,
    has_uv: bool,
) -> Optional[np.ndarray]:
    """
    # Return flat vertex color in 0-1.

        The gray return is flat float32 of length 3 * num_vertices, filled with DEFAULT_UNTEXTURED_VERTEX_COLOR.

    ## Args

        - num_vertices: the vertex count used for the gray buffer.
        - vertex_colors: an existing color array, or None. When gray is not applied, None stays None and an array is returned as flat float32.
        - has_texture: a bool forwarded to should_apply_untextured_gray.
        - has_uv: a bool forwarded to should_apply_untextured_gray.

    ## Returns

        - Gray: 0.5 when the untextured rule asks for it.

    ---

    # 返回 0-1 的扁平顶点色。

        灰色返回值是长度为 3 * num_vertices 的扁平 float32，填的是 DEFAULT_UNTEXTURED_VERTEX_COLOR。

    ## 参数

        - num_vertices: 灰色缓冲使用的顶点数。
        - vertex_colors: 已有的颜色数组，或 None。不涂灰时，None 仍是 None，数组则按扁平 float32 返回。
        - has_texture: bool，传给 should_apply_untextured_gray。
        - has_uv: bool，传给 should_apply_untextured_gray。

    ## 返回

        - 无纹理规则要求灰色时，灰色是 0.5。

"""
    if not should_apply_untextured_gray(
        has_texture=has_texture,
        has_uv=has_uv,
        vertex_colors=vertex_colors,
        num_vertices=num_vertices,
    ):
        if vertex_colors is None:
            return None
        return np.asarray(vertex_colors, dtype=np.float32).reshape(-1, 3).reshape(-1)
    gray = np.float32(DEFAULT_UNTEXTURED_VERTEX_COLOR)
    return np.full((num_vertices, 3), gray, dtype=np.float32).reshape(-1)


def extract_trimesh_vertex_colors(mesh: Any) -> Optional[np.ndarray]:
    """
    # Return flat 0-1 vertex color from a color visual.

        A texture visual returns None.

        Values whose maximum is above 1 + 1e-3 are divided by 255.

        An empty color returns None.

    ## Args

        - mesh: an object with visual. Missing visual returns None. TextureVisuals returns None. vertex_colors uses the first three channels.

    ## Returns

        - The return is flat float32, three components per vertex, or None.

    ---

    # 从颜色可视对象返回扁平的 0-1 顶点色。

        纹理可视对象返回 None。

        最大值大于 1 + 1e-3 的数值会除以 255。

        空颜色返回 None。

    ## 参数

        - mesh: 带 visual 的对象。没有 visual 时返回 None。TextureVisuals 返回 None。vertex_colors 使用前三个通道。

    ## 返回

        - 返回扁平 float32，每个顶点三个分量，或 None。

"""
    # visual is a submodule, not an attribute exported by trimesh/__init__.py.
    # visual 是子模块，trimesh/__init__.py 没有把它作为属性导出。
    from trimesh.visual.texture import TextureVisuals

    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None
    if isinstance(visual, TextureVisuals):
        return None
    raw = getattr(visual, "vertex_colors", None)
    if raw is None:
        return None
    colors = np.asarray(raw[..., :3], dtype=np.float32)
    if colors.size == 0:
        return None
    if colors.max() > 1.0 + 1e-3:
        colors = colors / 255.0
    return colors.reshape(-1, 3).reshape(-1)


def flip_uv_v_opengl(uv: np.ndarray) -> np.ndarray:
    """
    # Flip image V to renderer V.

        The second column becomes 1 - v.

        uv reshapes to (N, 2).

        The return is a float32 copy of that shape.

    ---

    # 把图像的 V 换成渲染器的 V。

        第二列变成 1 - v。

        uv 会 reshape 成 (N, 2)。

        返回这个形状的 float32 副本。

"""
    out = np.asarray(uv, dtype=np.float32).reshape(-1, 2).copy()
    out[:, 1] = 1.0 - out[:, 1]
    return out


def expand_vertex_uv_to_wedge(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    # Repeat per-vertex UV onto the three corners of each face.

        uv reshapes to (V, 2) float32.

        faces reshapes to (F, 3) int64 and indexes that UV.

    ## Returns

        - The return is float32 (3 * F, 2), in face-corner order.

    ---

    # 把逐顶点 UV 铺到每个面的三个角上。

        uv 会 reshape 成 (V, 2) float32。

        faces 会 reshape 成 (F, 3) int64，并用它索引 UV。

    ## 返回

        - 返回 float32 (3 * F, 2)，顺序是每个面的三个角。

"""
    uv2 = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    faces_i = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    return np.asarray(uv2[faces_i.reshape(-1)], dtype=np.float32)


def _count_vertices_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[int, int]:
    """
    # Return the vertex count and the triangle count from arrays stored as rows of 3.

        vertices reshapes to (V, 3).

        faces reshapes to (F, 3).

    ## Returns

        - The return is two ints, V and F.

    ---

    # 从每行 3 个数的数组返回顶点数和三角形数。

        vertices 会 reshape 成 (V, 3)。

        faces 会 reshape 成 (F, 3)。

    ## 返回

        - 返回两个 int，即 V 和 F。

"""
    nv = int(np.asarray(vertices).reshape(-1, 3).shape[0])
    nf = int(np.asarray(faces).reshape(-1, 3).shape[0])
    return nv, nf


def classify_uv_layout(
    uv: np.ndarray,
    num_vertices: int,
    num_faces: int,
) -> str:
    """
    # Return vertex when the UV count equals the vertex count, and wedge when it equals three times the faces.

        uv reshapes to (N, 2).

        No caller in WAPR.

    ## Args

        - num_vertices: the vertex count compared with N.
        - num_faces: the triangle count. Three times this count is the wedge count.

    ## Returns

        - The return is the string vertex or the string wedge.
        - Any other count raises UvLayoutError.

    ---

    # UV 数量等于顶点数时返回 vertex，等于面数的三倍时返回 wedge。

        uv 会 reshape 成 (N, 2)。

        WAPR 里没有调用它。

    ## 参数

        - num_vertices: 用来和 N 比较的顶点数。
        - num_faces: 三角形数。它的三倍是 wedge 的数量。

    ## 返回

        - 返回字符串 vertex 或字符串 wedge。
        - 其他数量抛出 UvLayoutError。
    """
    nu = int(np.asarray(uv).reshape(-1, 2).shape[0])
    if nu == num_vertices:
        return "vertex"
    if nu == num_faces * 3:
        return "wedge"
    raise UvLayoutError(
        f"UV count {nu} does not match vertex ({num_vertices}) or wedge ({num_faces * 3}) layout"
    )


def resolve_uv_layout(
    uv: Optional[np.ndarray],
    *,
    num_vertices: int,
    num_faces: int,
    faces: Optional[np.ndarray] = None,
    uv_per_corner: Optional[bool] = None,
    prefer_vertex_on_arrays: bool = True,
    flip_uv_v: bool = False,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    # Return UV and whether it is wedge.

        flip_uv_v applies v = 1 - v first.

        uv_per_corner True forces wedge.

        A vertex-count array is expanded with faces.

        False forces vertex UV and requires num_vertices rows.

        None chooses vertex or wedge from the count.

        flip_uv_v False copies the UV.

        True replaces V with 1 - V before the count check.

        The default is False.

        The UV return is float32 (N, 2), or None.

        The bool is True for wedge UV.

    ## Args

        - uv: (N, 2) or a flat buffer, or None. None returns (None, False).
        - num_vertices: the vertex count that selects per-vertex UV.
        - num_faces: the triangle count. Three times this count selects wedge UV.
        - faces: (F, 3) indices, or None. It is required only when per-vertex UV must be promoted to wedge.
        - prefer_vertex_on_arrays: accepted and ignored. The count chooses the layout.

    ---

    # 返回 UV，以及它是不是 wedge。

        flip_uv_v 会先做 v = 1 - v。

        uv_per_corner 为 True 时强制 wedge。

        数量等于顶点数的数组会用 faces 展开。

        False 强制逐顶点 UV，并且行数必须是 num_vertices。

        None 按数量选择 vertex 或 wedge。

        flip_uv_v 为 False 时复制 UV。

        True 在检查数量之前把 V 换成 1 - V。

        默认是 False。

        UV 返回值是 float32 (N, 2)，或 None。

        那个 bool 在 wedge UV 时为 True。

    ## 参数

        - uv: (N, 2) 或扁平缓冲，或 None。None 返回 (None, False)。
        - num_vertices: 用来选择逐顶点 UV 的顶点数。
        - num_faces: 三角形数。它的三倍用来选择 wedge UV。
        - faces: (F, 3) 下标，或 None。只有把逐顶点 UV 提升成 wedge 时才需要它。
        - prefer_vertex_on_arrays 会被接受并忽略。
        - 布局由数量决定。

"""
    # Count decides the layout. The flag stays so existing callers can still pass it.
    # 布局由数量决定。这个参数仍保留，已有调用处可以继续传入。
    del prefer_vertex_on_arrays
    if uv is None:
        return None, False

    uv_arr = flip_uv_v_opengl(uv) if flip_uv_v else np.asarray(uv, dtype=np.float32).reshape(-1, 2).copy()
    nu = int(uv_arr.shape[0])

    if uv_per_corner is True:
        if nu == num_faces * 3:
            return uv_arr, True
        if nu == num_vertices:
            if faces is None:
                raise UvLayoutError(
                    "per-vertex UV cannot be promoted to wedge without face indices"
                )
            return expand_vertex_uv_to_wedge(uv_arr, faces), True
        raise UvLayoutError(
            f"forced wedge UV: expected {num_faces * 3} or {num_vertices} coords, got {nu}"
        )

    if uv_per_corner is False:
        if nu != num_vertices:
            raise UvLayoutError(
                f"forced vertex UV: expected {num_vertices} coords, got {nu}"
            )
        return uv_arr, False

    if nu == num_vertices:
        return uv_arr, False
    if nu == num_faces * 3:
        return uv_arr, True
    raise UvLayoutError(
        f"auto UV layout failed: {nu} coords vs vertex={num_vertices}, wedge={num_faces * 3}"
    )


def _tex_wedge_uv_from_mesh_fields(mesh: Any) -> Optional[np.ndarray]:
    """
    # Return wedge UV already stored on the mesh metadata, or None.

        mesh.metadata may contain wapr_wedge_uv.

        Missing metadata or a missing value returns None.

        The array is returned as float32 when wapr_uv_layout is wedge, pbr_shard_id starts with tex_, or pbr_version is objaverse_tex.

        Any other metadata returns None.

    ---

    # 返回网格 metadata 里已经存好的 wedge UV，或 None。

        mesh.metadata 里可以有 wapr_wedge_uv。

        没有 metadata 或没有这个值时返回 None。

        当 wapr_uv_layout 是 wedge、pbr_shard_id 以 tex_ 开头，或 pbr_version 是 objaverse_tex 时，数组按 float32 返回。

        其他 metadata 返回 None。

"""
    meta = getattr(mesh, "metadata", None) or {}
    raw = meta.get("wapr_wedge_uv")
    if raw is None:
        return None
    layout = str(meta.get("wapr_uv_layout") or "").strip().lower()
    shard_id = str(meta.get("pbr_shard_id") or "").strip()
    version = str(meta.get("pbr_version") or "").strip()
    if layout == "wedge" or shard_id.startswith("tex_") or version == "objaverse_tex":
        return np.asarray(raw, dtype=np.float32)
    return None


def infer_trimesh_uv_layout(
    mesh: Any,
    *,
    uv_per_corner: Optional[bool] = None,
    flip_uv_v: bool = False,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    # Return UV from metadata wedge data when present, otherwise from the mesh visual.

        mesh supplies vertices, faces, metadata, and visual.uv.

        A visual without uv returns (None, False) when metadata is not used.

        uv_per_corner False skips the metadata wedge and asks resolve_uv_layout for vertex UV.

        True asks for wedge.

        None lets metadata request wedge, and otherwise lets the visual count decide.

    ## Args

        - flip_uv_v: forwarded. The default is False, so V is unchanged unless the caller sets it.

    ## Returns

        - The return is the resolve_uv_layout pair: float32 UV or None, and a wedge bool.

    ---

    # 有 metadata 里的 wedge UV 就用它，否则用网格 visual 上的 UV。

        mesh 提供 vertices、faces、metadata 和 visual.uv。

        不用 metadata 时，没有 uv 的 visual 返回 (None, False)。

        uv_per_corner 为 False 时跳过 metadata 里的 wedge，并向 resolve_uv_layout 要求逐顶点 UV。

        True 要求 wedge。

        None 允许 metadata 要求 wedge，否则由 visual 的数量决定。

    ## 参数

        - flip_uv_v 会原样传下去。
        - 默认是 False，因此除非调用方打开它，V 保持不变。

    ## 返回

        - 返回值是 resolve_uv_layout 的那一对：float32 UV 或 None，以及一个 wedge bool。

"""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    nv = int(len(mesh.vertices))
    nf = int(len(faces))
    stamped = _tex_wedge_uv_from_mesh_fields(mesh)
    if stamped is not None and uv_per_corner is not False:
        return resolve_uv_layout(
            stamped,
            num_vertices=nv,
            num_faces=nf,
            faces=faces,
            uv_per_corner=True,
            flip_uv_v=flip_uv_v,
        )
    visual = getattr(mesh, "visual", None)
    raw_uv = getattr(visual, "uv", None) if visual is not None else None
    if raw_uv is None:
        return None, False
    return resolve_uv_layout(
        raw_uv,
        num_vertices=nv,
        num_faces=nf,
        faces=faces,
        uv_per_corner=uv_per_corner,
        flip_uv_v=flip_uv_v,
    )


def extract_texture_rgb(
    mesh: Any,
    *,
    max_texture_size: int = DEFAULT_MAX_TEXTURE_SIZE,
    override_max_texture_size: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    # Return the material image as flat RGB bytes, plus width and height.

        override_max_texture_size None keeps max_texture_size.

        An int replaces it, and 0 leaves the converted image at its own size.

        The buffer return is flat uint8 of length width * height * 3, or None.

        The width and height returns are ints in pixels.

        They are 0 when the buffer is None.

    ## Args

        - mesh: read through visual.material. SimpleMaterial uses image. A missing image then uses baseColorTexture. No image returns None, 0, 0. The image must provide convert RGB.
        - max_texture_size: the long-side limit in pixels. The default is 1024.

    ---

    # 把材质图返回成扁平 RGB 字节，并给出宽和高。

        override_max_texture_size 为 None 时保留 max_texture_size。

        整数会替换它，0 则让转换后的图像保持自身尺寸。

        缓冲返回值是长度为 width * height * 3 的扁平 uint8，或 None。

        宽度和高度返回值是像素整数。

        缓冲为 None 时它们是 0。

    ## 参数

        - mesh 通过 visual.material 读取。
        - SimpleMaterial 用 image。
        - 没有 image 时再取 baseColorTexture。
        - 没有图像时返回 None、0、0。
        - 图像必须提供 convert RGB。
        - max_texture_size: 长边上限，单位像素。默认是 1024。

"""
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None) if visual is not None else None
    image = getattr(material, "image", None) if material is not None else None
    if image is None and material is not None:
        # glTF and the reconstructed mesh keep the color image here.
        # SimpleMaterial uses image. Missing this sends the renderer back to vertex color.
        # glTF 和重建网格把颜色图放在这里。SimpleMaterial 用的是 image。
        # 漏掉这一路时，渲染会退回顶点色。
        image = getattr(material, "baseColorTexture", None)
    if image is None:
        return None, 0, 0
    tex_limit = max_texture_size if override_max_texture_size is None else int(override_max_texture_size)
    img = np.asarray(image.convert("RGB"))
    img = _downscale_texture_rgb(img, tex_limit)
    tex_h, tex_w = img.shape[:2]
    return img.reshape(-1).astype(np.uint8), int(tex_w), int(tex_h)


def pack_trimesh_upload(
    mesh: Any,
    *,
    uv_per_corner: Optional[bool] = None,
    max_texture_size: int = DEFAULT_MAX_TEXTURE_SIZE,
    override_max_texture_size: Optional[int] = None,
    force_flat_normal: bool = False,
    flip_uv_v: bool = True,
) -> MeshUploadArrays:
    """
    # Pack vertex color, one RGB texture, and per-vertex or wedge UV from a mesh.

        Vertices from the mesh are stored as float32 and are meters.

        Faces from the mesh are stored as flat uint32, three indices per triangle.

        uv_per_corner None lets the UV count decide, while metadata wedge UV is still requested as wedge.

        True or False is forwarded to infer_trimesh_uv_layout.

        override_max_texture_size None uses max_texture_size.

        An int replaces it.

        A texture without usable UV is replaced by its mean color on the vertices, in 0-1.

        Placeholder black and 102/255 become gray 0.5 when the untextured rule applies.

    ## Args

        - mesh: a trimesh.Trimesh or a path trimesh.load_mesh can read.
        - max_texture_size: the long-side limit in pixels. The default is 1024.
        - force_flat_normal: stored on the result and forwarded by the uploader. The default is False.
        - flip_uv_v: to True, so renderer V is 1 - v. False leaves V unchanged.

    ## Returns

        - The return is a MeshUploadArrays.

    ---

    # 从网格打包顶点色、一张 RGB 纹理，以及逐顶点或 wedge UV。

        网格顶点按 float32 存下，单位是米。

        网格的面按扁平 uint32 存下，每个三角形三个下标。

        uv_per_corner 为 None 时由 UV 数量决定布局，但 metadata 里的 wedge UV 仍按 wedge 请求。

        True 或 False 会传给 infer_trimesh_uv_layout。

        override_max_texture_size 为 None 时使用 max_texture_size。

        整数会替换它。

        纹理没有可用 UV 时，用它的平均色涂到顶点上，范围 0-1。

        无纹理规则生效时，占位纯黑和 102/255 变成 0.5 灰。

    ## 参数

        - mesh: trimesh.Trimesh，或 trimesh.load_mesh 能读取的路径。
        - max_texture_size: 长边上限，单位像素。默认是 1024。
        - force_flat_normal 存在结果上，并由上传器继续传递。
        - 默认是 False。
        - flip_uv_v 默认是 True，因此渲染器的 V 是 1 - v。
        - False 保持 V 不变。

    ## 返回

        - 返回 MeshUploadArrays。

"""
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.load_mesh(mesh, process=False)

    verts = np.asarray(mesh.vertices, np.float32)
    faces = np.asarray(mesh.faces, np.uint32).reshape(-1)
    nv = int(len(verts))

    colors = extract_trimesh_vertex_colors(mesh)
    tex_rgb, tex_w, tex_h = extract_texture_rgb(
        mesh,
        max_texture_size=max_texture_size,
        override_max_texture_size=override_max_texture_size,
    )
    uvs, wedge = infer_trimesh_uv_layout(
        mesh, uv_per_corner=uv_per_corner, flip_uv_v=bool(flip_uv_v)
    )
    uvs, wedge = discard_placeholder_uv(uvs, wedge)
    has_texture = tex_rgb is not None and tex_w > 0 and tex_h > 0
    has_uv = uvs is not None
    if has_texture and not has_uv:
        baked = vertex_colors_from_texture_solid(nv, tex_rgb, tex_w, tex_h)
        if baked is not None:
            colors = baked
        tex_rgb = None
        tex_w = 0
        tex_h = 0
        has_texture = False
    colors = ensure_untextured_vertex_color(
        nv,
        colors,
        has_texture=has_texture,
        has_uv=has_uv,
    )
    normals = ensure_vertex_normals(verts, faces, np.asarray(mesh.vertex_normals, np.float32))

    return MeshUploadArrays(
        vertices=verts,
        faces=faces,
        normals=normals,
        vertex_colors=colors,
        uvs=uvs,
        texture_rgb=tex_rgb,
        tex_w=tex_w,
        tex_h=tex_h,
        uv_per_corner=wedge,
        force_flat_normal=force_flat_normal,
    )


def pack_array_upload(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    normals: Optional[np.ndarray] = None,
    vertex_colors: Optional[np.ndarray] = None,
    uvs: Optional[np.ndarray] = None,
    texture_rgb: Optional[Union[np.ndarray, Tuple[int, int]]] = None,
    tex_w: int = 0,
    tex_h: int = 0,
    uv_per_corner: Optional[bool] = None,
    force_flat_normal: bool = False,
    max_texture_size: int = DEFAULT_MAX_TEXTURE_SIZE,
    override_max_texture_size: Optional[int] = None,
    flip_uv_v: bool = True,
) -> MeshUploadArrays:
    """
    # Pack the same color, texture, and UV choice as pack_trimesh_upload, starting from arrays.

        uv_per_corner None chooses vertex or wedge from the count.

        True forces wedge.

        False forces vertex UV.

        max_texture_size limits a 3-D image.

        The default is 1024.

        override_max_texture_size None uses max_texture_size.

        An int replaces it.

        No caller in WAPR.

    ## Args

        - vertices: stored as float32 and are meters.
        - faces: stored as flat uint32, three indices per triangle.
        - normals: an existing normal array, or None. None recomputes normals.
        - vertex_colors: float32, or None. A texture without usable UV replaces it with the mean texture color.
        - uvs: the UV array, or None. The count must match the vertices, or three times the faces when wedge is selected.
        - texture_rgb: an HxWx3 image, a flat RGB buffer, a (width, height) tuple with no pixels, or None. A 3-D image is downscaled. A flat buffer requires positive tex_w and tex_h.
        - tex_w: the flat-buffer width in pixels. The default is 0. A 3-D image replaces it with that image's width.
        - tex_h: the flat-buffer height in pixels. The default is 0. A 3-D image replaces it with that image's height.
        - force_flat_normal: stored on the result. The default is False.
        - flip_uv_v: to True, so V becomes 1 - v before the layout check.

    ## Returns

        - The return is a MeshUploadArrays.
        - Placeholder vertex colors become gray 0.5 when the untextured rule applies.

    ---

    # 和 pack_trimesh_upload 做同一套颜色、纹理和 UV 选择，只是从数组开始。

        uv_per_corner 为 None 时按数量选择 vertex 或 wedge。

        True 强制 wedge。

        False 强制逐顶点 UV。

        max_texture_size 用来限制三维图像。

        默认是 1024。

        override_max_texture_size 为 None 时使用 max_texture_size。

        整数会替换它。

        WAPR 里没有调用它。

    ## 参数

        - vertices 按 float32 存下，单位是米。
        - faces 按扁平 uint32 存下，每个三角形三个下标。
        - normals: 已有的法线数组，或 None。None 会重新计算法线。
        - vertex_colors: float32，或 None。纹理没有可用 UV 时，它会被纹理平均色替换。
        - uvs: UV 数组，或 None。数量必须对上顶点数；选择 wedge 时则必须是面数的三倍。
        - texture_rgb: HxWx3 图像、扁平 RGB 缓冲、没有像素的 (width, height) 元组，或 None。三维图像会缩小。扁平缓冲要求 tex_w 和 tex_h 都大于 0。
        - tex_w: 扁平缓冲的宽度，单位像素。默认是 0。三维图像会用该图像的宽度替换它。
        - tex_h: 扁平缓冲的高度，单位像素。默认是 0。三维图像会用该图像的高度替换它。
        - force_flat_normal 存在结果上。
        - 默认是 False。
        - flip_uv_v 默认是 True，因此在检查布局之前 V 变成 1 - v。

    ## 返回

        - 返回 MeshUploadArrays。
        - 无纹理规则生效时，占位顶点色变成 0.5 灰。
    """
    nv, nf = _count_vertices_faces(vertices, faces)
    faces_u32 = np.asarray(faces, dtype=np.uint32).reshape(-1)
    faces_i64 = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    tex_flat: Optional[np.ndarray] = None
    out_w, out_h = int(tex_w), int(tex_h)
    if texture_rgb is not None:
        if isinstance(texture_rgb, tuple):
            out_w, out_h = int(texture_rgb[0]), int(texture_rgb[1])
            tex_flat = None
        else:
            arr = np.asarray(texture_rgb)
            if arr.ndim == 3:
                tex_limit = max_texture_size if override_max_texture_size is None else int(override_max_texture_size)
                arr = _downscale_texture_rgb(arr, tex_limit)
                out_h, out_w = arr.shape[:2]
                tex_flat = arr.reshape(-1).astype(np.uint8)
            else:
                tex_flat = arr.astype(np.uint8, copy=False).reshape(-1)
                if out_w <= 0 or out_h <= 0:
                    raise ValueError("tex_w/tex_h required when texture_rgb is a flat buffer")

    verts = np.asarray(vertices, np.float32)
    resolved_uvs, wedge = resolve_uv_layout(
        uvs,
        num_vertices=nv,
        num_faces=nf,
        faces=faces_i64,
        uv_per_corner=uv_per_corner,
        flip_uv_v=bool(flip_uv_v),
    )
    resolved_uvs, wedge = discard_placeholder_uv(resolved_uvs, wedge)

    has_texture = tex_flat is not None and out_w > 0 and out_h > 0
    has_uv = resolved_uvs is not None
    resolved_colors = None if vertex_colors is None else np.asarray(vertex_colors, np.float32)
    if has_texture and not has_uv:
        baked = vertex_colors_from_texture_solid(nv, tex_flat, out_w, out_h)
        if baked is not None:
            resolved_colors = baked
        tex_flat = None
        out_w = 0
        out_h = 0
        has_texture = False
    resolved_colors = ensure_untextured_vertex_color(
        nv,
        resolved_colors,
        has_texture=has_texture,
        has_uv=has_uv,
    )
    resolved_normals = ensure_vertex_normals(verts, faces_u32, normals)

    return MeshUploadArrays(
        vertices=verts,
        faces=faces_u32,
        normals=resolved_normals,
        vertex_colors=resolved_colors,
        uvs=resolved_uvs,
        texture_rgb=tex_flat,
        tex_w=out_w,
        tex_h=out_h,
        uv_per_corner=wedge,
        force_flat_normal=force_flat_normal,
    )

import importlib.util
import hashlib
import os
import shutil
import subprocess
import sys
import weakref
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from wapr import recipe


_NATIVE = None
_RUNTIMES = {}


def parse_cuda_device(device):
    """
    # Return a CUDA ordinal from an int, a cuda:N string, or a device with index.

        None means 0.

        device None returns 0.

        An int returns that ordinal.

        A cuda:N string, or an object with index such as torch.device, returns that number.

        A bool raises ValueError.

        Any other text is parsed with int.

    ## Returns

        - The return is an int.

    ---

    # 从整数、cuda:N 字符串，或带 index 的 device 返回 CUDA 序号。

        None 表示 0。

        device 为 None 时返回 0。

        整数返回这个序号。

        cuda:N 字符串，或带 index 的对象（例如 torch.device），返回那个数字。

        bool 抛出 ValueError。

        其他文本用 int 解析。

    ## 返回

        - 返回 int。

"""
    if device is None:
        return 0
    if isinstance(device, bool):
        raise ValueError("device")
    if isinstance(device, int):
        return int(device)
    idx = getattr(device, "index", None)
    if idx is not None and not isinstance(device, (str, bytes)) and not callable(idx):
        return int(idx)
    text = str(device).strip()
    if text.startswith("cuda:"):
        return int(text.split(":")[-1])
    return int(text)


def _native_dir():
    """
    # Return the directory that holds the GL renderer sources and the compiled library.

    ## Args

        - There are no arguments.
        - The return is a Path, this file's directory joined with ogl_native.

    ---

    # 返回放着 GL 渲染器源码和编译库的目录。

    ## 参数

        - 没有参数。
        - 返回值是 Path，即本文件所在目录拼上 ogl_native。

"""
    return Path(__file__).resolve().parent / "ogl_native"


def _find_so(native_dir):
    """
    # Return the compiled renderer library, or None when it is absent.

    ## Args

        - native_dir: the Path from _native_dir. The search name is _gpu_render*.so, first beside the sources, then under build/.

    ## Returns

        - The return is a Path, or None.

    ---

    # 返回编译好的渲染库；不存在时返回 None。

    ## 参数

        - native_dir: _native_dir 给出的 Path。查找的名字是 _gpu_render*.so，先在源码旁边，再在 build/ 下面。

    ## 返回

        - 返回 Path，或 None。

"""
    found = list(native_dir.glob("_gpu_render*.so"))
    if found:
        return found[0]
    build = native_dir / "build"
    if build.is_dir():
        found = list(build.rglob("_gpu_render*.so"))
        if found:
            return found[0]
    return None


def ensure_ogl():
    """
    # Load the renderer module.

        A missing library is compiled here on the first call.

    ## Args

        - There are no arguments.
        - A module already stored in _NATIVE is returned as it is.
        - The build uses cmake in ogl_native/build when no _gpu_render library is found.

    ## Returns

        - The return is the imported _gpu_render module.
        - A missing library after the build, or a loader that cannot be created, raises RuntimeError.

    ---

    # 载入渲染器模块。

        库不存在时，在第一次调用这里编译。

    ## 参数

        - 没有参数。
        - _NATIVE 里已经有模块时，原样返回它。
        - 找不到 _gpu_render 库时，用 cmake 在 ogl_native/build 里编译。

    ## 返回

        - 返回导入的 _gpu_render 模块。
        - 编译之后仍然没有库，或无法创建 loader 时，抛出 RuntimeError。

"""
    global _NATIVE
    if _NATIVE is not None:
        return _NATIVE
    native_dir = _native_dir()
    so = _find_so(native_dir)
    build_stamp = native_dir / "build-source.sha256"
    source_digest = None
    if (native_dir / "CMakeLists.txt").is_file():
        # An installed source wheel can update CMake or C++ without pip removing
        # generated libraries. Rebuild stale outputs, keeping the source files.
        # 安装源码 wheel 时 pip 不清理生成库；源码或编译配置变化后重建生成文件。
        digest = hashlib.sha256(sys.implementation.cache_tag.encode("ascii"))
        digest.update(str(torch.version.cuda).encode("ascii"))
        for source_path in sorted(native_dir.rglob("*")):
            if source_path.is_file() and "build" not in source_path.relative_to(native_dir).parts:
                if source_path.suffix in (".cpp", ".h", ".hpp", ".cu") or source_path.name == "CMakeLists.txt":
                    digest.update(str(source_path.relative_to(native_dir)).encode("utf-8"))
                    digest.update(source_path.read_bytes())
        source_digest = digest.hexdigest()
        if not build_stamp.is_file() or build_stamp.read_text().strip() != source_digest:
            if so is not None:
                so.unlink()
            build = native_dir / "build"
            if build.is_dir():
                shutil.rmtree(build)
            so = None
    if so is None:
        # Derived from the missing library. There is no separate compile switch.
        # 由库文件是否存在决定。没有单独的编译开关。
        print("OGL library missing, compiling now. / 没有 OGL 库，现在编译。", flush=True)
        build = native_dir / "build"
        from wapr.bootstrap import native_build_options
        compiler_options = native_build_options()
        subprocess.check_call(
            [
                "cmake", "-S", str(native_dir), "-B", str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DPython3_EXECUTABLE=%s" % sys.executable,
            ] + compiler_options
        )
        subprocess.check_call(["cmake", "--build", str(build), "-j", "4"])
        so = _find_so(native_dir)
    if so is None:
        raise RuntimeError("ogl")
    # The init symbol is PyInit__gpu_render, the pybind module name.
    # 初始化符号是 PyInit__gpu_render，也就是 pybind 的模块名。
    spec = importlib.util.spec_from_file_location("_gpu_render", so)
    # spec_from_file_location can return None, and its loader can also be None.
    # spec_from_file_location 可能返回 None，它的 loader 也可能是 None。
    if spec is None or spec.loader is None:
        raise RuntimeError("ogl")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if source_digest is not None:
        build_stamp.write_text(source_digest + "\n")
    _NATIVE = module
    return module


def clip_affine4_from_bbox2d(bbox2d, frame_h, frame_w):
    """
    # Map a pixel box into the clip frame.

        A y-down pixel box becomes the y-up clip frame, as 16 float32 values.

    ## Args

        - bbox2d: left, top, right, bottom in full-frame pixels, y down. It reshapes to four numbers.
        - frame_h: the full frame height in pixels. Y is flipped with frame_h - y before the scale is formed.
        - frame_w: the full frame width in pixels. It scales the horizontal bbox into the clip frame.

    ## Returns

        - The return is (16,) float32, row-major.
        - Scale sits on the diagonal, and translation sits in the last row.

    ---

    # 像素框转到裁剪坐标。

        y 向下的像素包围盒变到 y 向上的裁剪坐标系，返回 16 个 float32。

    ## 参数

        - bbox2d: 整幅图像素的 left、top、right、bottom，y 向下。它会 reshape 成四个数。
        - frame_h: 整幅图的高度，单位像素。形成缩放之前，Y 用 frame_h - y 翻转。
        - frame_w: 整幅图的宽度，单位像素。它把水平包围盒缩放到裁剪坐标系。

    ## 返回

        - 返回 (16,) float32，按行展开。
        - 缩放在对角线上，平移在最后一行。

"""
    bb = np.asarray(bbox2d, dtype=np.float64).reshape(-1)
    left, vmin, right, vmax = bb[0], bb[1], bb[2], bb[3]
    top = frame_h - vmin
    bottom = frame_h - vmax
    sx = frame_w / (right - left)
    sy = frame_h / (top - bottom)
    tx = (frame_w - right - left) / (right - left)
    ty = (frame_h - top - bottom) / (top - bottom)
    return np.array(
        [[sx, 0.0, 0.0, 0.0], [0.0, sy, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [tx, ty, 0.0, 1.0]],
        dtype=np.float32,
    ).reshape(-1)


class GpuRenderRuntime:
    """
    # GL tile renderer with a mesh cache.

        runtime_for and onboard_meshes construct it.

        make_crop_pair calls ensure_mesh_slots, load_mesh_trimesh, and render_tiles.

        onboard_meshes, fit_axis_scales.depth_loss, and render_mesh_crop call load_mesh_trimesh and render_tiles.

    ## Returns

        - Poses and depth: meters, and the camera K is OpenCV.

    ---

    # 带网格缓存的 GL 分块渲染器。

        runtime_for 和 onboard_meshes 构造它。

        make_crop_pair 调用 ensure_mesh_slots、load_mesh_trimesh 和 render_tiles。

        onboard_meshes、fit_axis_scales.depth_loss 和 render_mesh_crop 调用 load_mesh_trimesh 和 render_tiles。

    ## 返回

        - 位姿和深度是米，相机 K 是 OpenCV。

"""
    def __init__(self, native, device, shader_dir):
        """
        # Keep one native GL runtime on this CUDA ordinal.

            The mesh cache limit starts at 8.

            The stored texture long side is 1024 pixels.

        ## Args

            - native: the module from ensure_ogl.
            - device: the CUDA ordinal.
            - shader_dir: the shader directory passed to the native runtime.

        ## Returns

            - Returns None.

        ---

        # 在这个 CUDA 序号上保留一个本地编译的 GL 运行模块。

            网格缓存上限从 8 开始。

            存下的纹理长边是 1024 像素。

        ## 参数

            - native: ensure_ogl 给出的模块。
            - device: CUDA 序号。
            - shader_dir: 传给渲染模块的着色器目录。

        ## 返回

            - 返回 None。

"""
        self._device = int(device)
        self._native = native
        self._rt = native.GpuRenderRuntime(self._device, shader_dir)
        self._mesh_cache = OrderedDict()
        self._mesh_cache_size = 8
        self._mesh_by_obj = {}
        self._max_texture_size = 1024
        # Resource counters make accidental cold work inside timing observable.
        # 资源计数便于检查预热后的计时中是否误做了冷准备。
        self.mesh_pack_count = 0
        self.mesh_upload_count = 0

    def ensure_mesh_slots(self, count):
        """
        # Raise the mesh-cache limit to at least count.

        ## Args

            - count: the number of rows in one multi-object batch. The limit never shrinks.

        ## Returns

            - Returns None.

        ---

        # 把网格缓存上限提到至少 count。

        ## 参数

            - count: 一批多物体的行数。上限不会缩小。

        ## 返回

            - 返回 None。

"""
        self._mesh_cache_size = max(int(self._mesh_cache_size), int(count))

    def load_mesh_trimesh(self, mesh, name="mesh"):
        """
        # Pack one mesh, upload it, and return the cached mesh id.

        ## Args

            - mesh: a trimesh.Trimesh or a path. A path is loaded with trimesh.load, force mesh, process False. A live Trimesh that is already cached and still valid returns its old id.
            - name: to mesh. It is the upload name. The pack uses texture long side 1024 and force_flat_normal False.

        ## Returns

            - The return is an int mesh id.

        ---

        # 打包一个网格并上传，返回缓存里的网格编号。

        ## 参数

            - mesh: trimesh.Trimesh 或路径。路径用 trimesh.load 载入，force 为 mesh，process 为 False。已经缓存且仍然有效的 Trimesh 会返回原来的编号。
            - name 默认是 mesh。
            - 它是上传名字。
            - 打包使用的纹理长边是 1024，force_flat_normal 是 False。

        ## 返回

            - 返回 int 网格编号。

"""
        import trimesh

        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.load(mesh, force="mesh", process=False)
        else:
            found = self._mesh_by_obj.get(id(mesh))
            if found is not None:
                ref, mid = found
                if ref() is mesh and self._rt.mesh_valid(int(mid)):
                    return int(mid)
                self._mesh_by_obj.pop(id(mesh), None)
        self.mesh_pack_count += 1
        upload = pack_trimesh_upload(
            mesh,
            max_texture_size=self._max_texture_size,
            override_max_texture_size=self._max_texture_size,
            force_flat_normal=False,
        )
        mid = self._upload(name or "mesh", upload)
        try:
            self._mesh_by_obj[id(mesh)] = (weakref.ref(mesh), int(mid))
        except TypeError:
            pass
        return int(mid)

    def cached_mesh_id(self, mesh):
        """Resolve an uploaded mesh without packing or uploading in a hot call.

        预热后调用只查已上传网格的编号，不打包或上传；缓存失效时要求重新准备。
        """
        found = self._mesh_by_obj.get(id(mesh))
        if found is None or found[0]() is not mesh or not self._rt.mesh_valid(found[1]):
            raise RuntimeError("Mesh is not resident; call estimator.prepare_meshes before timing")
        return int(found[1])

    def _cache_name(self, name, upload):
        """
        # Return an identity string for the uploaded arrays so the same mesh is not uploaded twice.

        ## Args

            - name: the string prefix.
            - upload: a MeshUploadArrays. vertices, faces, normals, vertex_colors, uvs, and texture_rgb are hashed. None contributes one zero byte.

        ## Returns

            - The return is name, a hash mark, and a blake2b hex digest of 12 bytes.

        ---

        # 返回上传数组的标识字符串，同一个网格不重复上传。

        ## 参数

            - name: 字符串前缀。
            - upload: MeshUploadArrays。vertices、faces、normals、vertex_colors、uvs 和 texture_rgb 会进入哈希。None 贡献一个 0 字节。

        ## 返回

            - 返回 name、一个井号，以及 12 字节 blake2b 的十六进制摘要。

"""
        import hashlib

        h = hashlib.blake2b(digest_size=12)
        for value in (upload.vertices, upload.faces, upload.normals, upload.vertex_colors, upload.uvs, upload.texture_rgb):
            if value is None:
                h.update(b"\x00")
                continue
            arr = np.ascontiguousarray(value)
            h.update(str(arr.dtype).encode("ascii"))
            h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
            h.update(arr.tobytes())
        return "%s#%s" % (name, h.hexdigest())

    def _evict(self, cache_name):
        """
        # Drop one cached mesh from the GL runtime and return whether it was present.

        ## Args

            - cache_name: the key from _cache_name. A missing key returns False. A valid native mesh is unloaded before the key is removed.

        ## Returns

            - The return is True after the cache entry is removed.

        ---

        # 从 GL 运行时丢掉一份缓存网格，并返回它原先是否存在。

        ## 参数

            - cache_name: _cache_name 给出的键。没有这个键时返回 False。有效的渲染模块中的网格会先卸载，再删掉这个键。

        ## 返回

            - 缓存项被删除后返回 True。

"""
        mid = self._mesh_cache.get(cache_name)
        if mid is None:
            return False
        if self._rt.mesh_valid(int(mid)):
            self._rt.unload_mesh(int(mid))
        self._mesh_cache.pop(cache_name, None)
        return True

    def _upload_item(self, name, upload):
        """
        # Return contiguous arrays the native loader reads.

            vertices in the dict is (V, 3) float32, meters, or None when upload.vertices is None.

            faces in the dict is (F, 3) uint32, or None.

            normals in the dict is (N, 3) float32, or None.

            vertex_colors in the dict is (N, 3) float32, or None.

            uvs in the dict is (N, 2) float32, or None.

            texture_rgb in the dict is (tex_h, tex_w, 3) uint8, or None.

            tex_w in the dict is the int copied from the upload.

            tex_h in the dict is the int copied from the upload.

            uv_per_corner in the dict is a bool.

            force_flat_normal in the dict is a bool.

        ## Args

            - name: stored under the name field.
            - upload: a MeshUploadArrays. Its arrays are reshaped below.

        ---

        # 返回渲染模块的网格加载器读取的连续数组。

            字典里的 vertices 是 (V, 3) float32，单位米。

            upload.vertices 为 None 时，这个字段是 None。

            字典里的 faces 是 (F, 3) uint32，或 None。

            字典里的 normals 是 (N, 3) float32，或 None。

            字典里的 vertex_colors 是 (N, 3) float32，或 None。

            字典里的 uvs 是 (N, 2) float32，或 None。

            字典里的 texture_rgb 是 (tex_h, tex_w, 3) uint8，或 None。

            字典里的 tex_w 是从上传对象复制的 int。

            字典里的 tex_h 是从上传对象复制的 int。

            字典里的 uv_per_corner 是 bool。

            字典里的 force_flat_normal 是 bool。

        ## 参数

            - name 存在 name 字段里。
            - upload: MeshUploadArrays。它的数组按下面的形状 reshape。

"""
        def shaped(value, cols, dtype):
            """
            # Return one contiguous numeric array with the requested row width, or None.

                value None returns None.

                Any other value is made contiguous, cast to dtype, and reshaped to (N, cols).

            ## Args

                - cols: the row width. This loader uses 3 for positions, faces, normals, and colors, and 2 for UV.
                - dtype: the numpy dtype of the returned array.

            ---

            # 返回指定行宽的连续数值数组，或 None。

                value 为 None 时返回 None。

                其他值会变成 contiguous，转成 dtype，并 reshape 成 (N, cols)。

            ## 参数

                - cols: 行宽。这个载入器用 3 表示位置、面、法线和颜色，用 2 表示 UV。
                - dtype: 返回数组的 numpy dtype。

"""
            if value is None:
                return None
            return np.ascontiguousarray(value, dtype=dtype).reshape(-1, cols)

        texture = None
        if upload.texture_rgb is not None:
            texture = np.ascontiguousarray(upload.texture_rgb, dtype=np.uint8).reshape(
                int(upload.tex_h), int(upload.tex_w), 3
            )
        return {
            "name": name,
            "vertices": shaped(upload.vertices, 3, np.float32),
            "faces": shaped(upload.faces, 3, np.uint32),
            "normals": shaped(upload.normals, 3, np.float32),
            "vertex_colors": shaped(upload.vertex_colors, 3, np.float32),
            "uvs": shaped(upload.uvs, 2, np.float32),
            "texture_rgb": texture,
            "tex_w": int(upload.tex_w),
            "tex_h": int(upload.tex_h),
            "uv_per_corner": bool(upload.uv_per_corner),
            "force_flat_normal": bool(upload.force_flat_normal),
        }

    def _upload(self, name, upload):
        """
        # Upload a mesh unless this cache key is already valid, and return the mesh id.

            The cache limit starts at 8 and can be raised by ensure_mesh_slots.

            The oldest entry is evicted while the cache is over that limit.

            A native texture size mismatch clears every mesh and loads this one again.

        ## Args

            - name: passed to _cache_name and _upload_item.
            - upload: the MeshUploadArrays passed to _cache_name and _upload_item. A valid cached id is moved to the newest end and returned.

        ## Returns

            - The return is an int mesh id.

        ---

        # 这个缓存键已经有效就不再上传，返回网格编号。

            缓存上限从 8 开始，ensure_mesh_slots 可以把它提高。

            缓存超过这个上限时，最旧的一项会被逐出。

            渲染模块报告纹理尺寸不一致时，会清空全部网格再载入这一份。

        ## 参数

            - name 传给 _cache_name 和 _upload_item。
            - upload: 传给 _cache_name 和 _upload_item 的 MeshUploadArrays。有效的缓存编号会被移到最新一端并返回。

        ## 返回

            - 返回 int 网格编号。

"""
        cache_name = self._cache_name(name, upload)
        if cache_name in self._mesh_cache:
            cached = self._mesh_cache[cache_name]
            if self._rt.mesh_valid(cached):
                self._mesh_cache.move_to_end(cache_name)
                return int(cached)
            self._evict(cache_name)
        item = self._upload_item(cache_name, upload)
        self.mesh_upload_count += 1
        try:
            mid = self._rt.load_mesh_from_arrays(
                item["name"], item["vertices"], item["faces"],
                item["normals"], item["vertex_colors"], item["uvs"],
                item["texture_rgb"], item["tex_w"], item["tex_h"],
                item["uv_per_corner"], item["force_flat_normal"],
            )
        except RuntimeError as exc:
            if "texture size mismatch" not in str(exc):
                raise
            self._rt.clear_all_meshes()
            self._mesh_cache.clear()
            mid = self._rt.load_mesh_from_arrays(
                item["name"], item["vertices"], item["faces"],
                item["normals"], item["vertex_colors"], item["uvs"],
                item["texture_rgb"], item["tex_w"], item["tex_h"],
                item["uv_per_corner"], item["force_flat_normal"],
            )
        self._mesh_cache[cache_name] = int(mid)
        while len(self._mesh_cache) > self._mesh_cache_size:
            self._evict(next(iter(self._mesh_cache)))
        return int(mid)

    def _tile_spec(self, tile, indexed):
        """
        # Return the native tile-output spec for one device or host draw.

            indexed True uses a viewport array and scissor, with an atlas limit of 1280 by 1280.

            False uses layered tiles, with atlas width 128 and atlas height 0.

            The spec asks for RGB, metric depth, a Y flip, and NHWC.

            The return is a native RenderOutputSpec.

        ## Args

            - tile: the square tile side in pixels. It becomes tile_w and tile_h.

        ---

        # 返回一次设备绘制或主机绘制使用的渲染模块的分块输出规格。

            indexed 为 True 时使用视口数组和裁剪，图集上限是 1280 乘 1280。

            False 使用分层分块，图集宽 128、高 0。

            这个规格要求 RGB、米制深度、Y 翻转和 NHWC。

            返回渲染模块的 RenderOutputSpec。

        ## 参数

            - tile: 正方形分块的边长，单位像素。它会写成 tile_w 和 tile_h。

"""
        native = self._native
        spec = native.RenderOutputSpec()
        spec.mode = native.RenderMode.Tile
        spec.projection = native.ProjectionMode.ClipAffine
        spec.tile_w = int(tile)
        spec.tile_h = int(tile)
        spec.rgb = True
        spec.metric_depth = True
        spec.y_flip = True
        spec.layout = native.DataLayout.NHWC
        spec.persistent_buffers = True
        if indexed:
            spec.use_scissor = True
            spec.indexed_viewport_scissor = True
            spec.viewport_per_tile = False
            spec.max_atlas_w = 1280
            spec.max_atlas_h = 1280
        else:
            spec.layered_tiles = True
            spec.use_scissor = False
            spec.max_atlas_w = 128
            spec.max_atlas_h = 0
        return spec

    def _draw_device_tiles(self, mesh_id, poses_t, bboxes_t, camera, frame_h, frame_w, tile, mesh_ids=None):
        """
        # Draw one mesh on the device.

            The result is rgb and metric depth.

            One indexed draw is used when the GL caps allow it and N is within max_viewports and 256.

            Otherwise the rows are drawn in chunks of 128 with layered tiles.

        ## Args

            - mesh_id: the native mesh int.
            - poses_t: (N, 4, 4) CUDA float32, object-to-camera, meters.
            - bboxes_t: (N, 4) CUDA float32, left, top, right, bottom, y down, pixels.
            - camera: a 3x3 OpenCV K. fx, fy, cx, and cy are the entries passed to the native draw.
            - frame_h: the full frame height in pixels.
            - frame_w: the full frame width in pixels.
            - tile: the square output side in pixels.

        ## Returns

            - The return is rgb and depth on CUDA.
            - A single chunk returns that chunk.
            - Several chunks are concatenated on dimension 0.

        ---

        # 在设备上绘制一个网格。

            返回 rgb 和米制深度。

            GL 能力允许，并且 N 不超过 max_viewports 和 256 时，使用一次索引绘制。

            否则各行按 128 一块、用分层分块来画。

        ## 参数

            - mesh_id: 渲染模块中的网格的 int。
            - poses_t: (N, 4, 4) CUDA float32，物体到相机，单位米。
            - bboxes_t: (N, 4) CUDA float32，即 left、top、right、bottom，y 向下，单位像素。
            - camera: 3x3 的 OpenCV K。传给渲染模块绘制的是 fx、fy、cx、cy。
            - frame_h: 整幅图的高度，单位像素。
            - frame_w: 整幅图的宽度，单位像素。
            - tile: 正方形输出边长，单位像素。

        ## 返回

            - 返回 CUDA 上的 rgb 和 depth。
            - 只有一块时返回这一块。
            - 多块沿第 0 维拼接。

"""
        n = int(poses_t.shape[0])
        caps = self._rt.gl_caps()
        can_indexed = bool(caps.viewport_array) and bool(caps.shader_viewport_layer_array) and bool(caps.multi_draw_indirect)
        max_viewports = int(caps.max_viewports)
        indexed = bool(can_indexed and n <= max_viewports and n <= 256)
        chunk = n if indexed else 128
        light = self._native.LightParams(float(recipe.w_ambient), float(recipe.w_diffuse), tuple(recipe.light_dir))
        stream = torch.cuda.current_stream(device=poses_t.device)
        rgb_parts = []
        depth_parts = []
        for start in range(0, n, chunk):
            poses_c = poses_t[start:start + chunk].contiguous()
            boxes_c = bboxes_t[start:start + chunk].contiguous()
            count = int(poses_c.shape[0])
            # Keep the layered atlas for every capacity chunk of a large batch.
            # 大批量放不进视口时，各容量块均使用分层图集。
            spec = self._tile_spec(tile, indexed=indexed and start == 0 and count == n)
            draw = self._rt.render_device_tiles if mesh_ids is None else self._rt.render_device_mesh_tiles
            mesh_arg = int(mesh_id) if mesh_ids is None else mesh_ids[start:start + count]
            count_arg = (count,) if mesh_ids is None else ()
            draw(
                mesh_arg,
                int(poses_c.data_ptr()),
                int(boxes_c.data_ptr()),
                *count_arg,
                float(camera[0, 0]),
                float(camera[1, 1]),
                float(camera[0, 2]),
                float(camera[1, 2]),
                int(frame_w),
                int(frame_h),
                int(stream.cuda_stream),
                spec,
                light,
            )
            rgb_c, depth_c = self.pack_to_torch()
            rgb_parts.append(rgb_c)
            depth_parts.append(depth_c)
        if len(rgb_parts) == 1:
            return rgb_parts[0], depth_parts[0]
        return torch.cat(rgb_parts, dim=0), torch.cat(depth_parts, dim=0)

    def _render_mixed_meshes(self, row_mesh_ids, poses_t, bboxes_t, camera, frame_h, frame_w, tile):
        """
        # Draw all row meshes together through the device path, preserving row order.

        ## Args

            - row_mesh_ids: one mesh id per row.
            - poses_t: the full-batch pose tensor passed to _draw_device_tiles, (N, 4, 4) CUDA float32, object-to-camera, meters.
            - bboxes_t: the full-batch box tensor, (N, 4) CUDA float32, left, top, right, bottom, pixels, y down.
            - camera: the 3x3 OpenCV K used for every mesh group.
            - frame_h: the full frame height in pixels.
            - frame_w: the full frame width in pixels.
            - tile: the square output side in pixels.

        ## Returns

            - rgb: (N, tile, tile, 3) on the device of poses_t. No mesh groups leaves it None.
            - depth: (N, tile, tile) meters, on the same device. No mesh groups leaves it None.

        ---

        # 所有行的网格一起走设备路径，输出保持原行序。

        ## 参数

            - row_mesh_ids: 每行一个网格编号。
            - poses_t: 传给 _draw_device_tiles 的整批位姿，(N, 4, 4) CUDA float32，物体到相机，单位米。
            - bboxes_t: 整批包围盒，(N, 4) CUDA float32，即 left、top、right、bottom，单位像素，y 向下。
            - camera: 每一组网格共用的 3x3 OpenCV K。
            - frame_h: 整幅图的高度，单位像素。
            - frame_w: 整幅图的宽度，单位像素。
            - tile: 正方形输出边长，单位像素。

        ## 返回

            - rgb: (N, tile, tile, 3)，在 poses_t 的设备上。没有网格组时它是 None。
            - depth: (N, tile, tile)，单位米，在同一设备上。没有网格组时它是 None。

"""
        if not hasattr(self._rt, "render_device_mesh_tiles"):
            raise RuntimeError("Rebuild wapr/ogl_native: mixed-mesh device batching is required")
        # One mixed-mesh submission per capacity chunk, with original row order.
        # 每个容量块只提交一次混合网格绘制，保持原行序。
        return self._draw_device_tiles(row_mesh_ids[0], poses_t, bboxes_t, camera,
                                       frame_h, frame_w, tile, mesh_ids=row_mesh_ids)

    def render_tiles(self, mesh_id, poses, bboxes, K, frame_h, frame_w, tile, mesh_ids=None):
        """
        # Render one tile per pose.

            RGB and metric depth come back on CUDA.

            mesh_ids None uses mesh_id for every tile.

            A sequence must have length N.

            Distinct ids take one device draw per mesh and are copied back by row.

        ## Args

            - mesh_id: the mesh used when mesh_ids is None or every listed id is the same.
            - poses: (N, 4, 4) or one (4, 4), object-to-camera, meters. CUDA float32 poses stay on the device when bboxes are CUDA float32, every K is the same, and WAPR_OGL_CPU_POSES is not 1. Any other layout is copied to the host instance list.
            - bboxes: (N, 4) left, top, right, bottom, full-frame pixels, y down.
            - K: OpenCV intrinsics in pixels, (3, 3) or (N, 3, 3). One matrix is repeated for every tile. A CUDA tensor is copied to NumPy before use.
            - frame_h: the full frame height in pixels.
            - frame_w: the full frame width in pixels.
            - tile: the square output side in pixels.

        ## Returns

            - rgb: (N, tile, tile, 3) float32.
            - depth: (N, tile, tile) float32 meters. Both are on this runtime's CUDA device.

        ---

        # 每个位姿渲染一块。

            返回 CUDA 上的 RGB 和米制深度。

            mesh_ids 为 None 时，每一块都用 mesh_id。

            序列的长度必须是 N。

            不同的编号按网格各做一次设备绘制，再按行拷回。

        ## 参数

            - mesh_ids: None，或列出的编号全都相同时，使用 mesh_id 这个网格。
            - poses: (N, 4, 4)，或单独一个 (4, 4)，物体到相机，单位米。bboxes 也是 CUDA float32、所有 K 相同，且 WAPR_OGL_CPU_POSES 不是 1 时，CUDA float32 位姿留在设备上。其他布局会拷到主机实例列表。
            - bboxes: (N, 4)，即 left、top、right、bottom，整幅图像素，y 向下。
            - K: OpenCV 内参，单位像素，形状 (3, 3) 或 (N, 3, 3)。一个矩阵会重复到每一块。CUDA 张量在使用前会拷成 NumPy。
            - frame_h: 整幅图的高度，单位像素。
            - frame_w: 整幅图的宽度，单位像素。
            - tile: 正方形输出边长，单位像素。

        ## 返回

            - rgb: (N, tile, tile, 3) float32。
            - depth: (N, tile, tile) float32，单位米。两者都在这个运行时的 CUDA 设备上。

"""
        native = self._native
        poses_t = None
        bboxes_t = None
        # WAPR_OGL_CPU_POSES=1 keeps the host instance list for a numeric check.
        # WAPR_OGL_CPU_POSES=1 时仍走主机实例列表，用来对数值。
        if os.environ.get("WAPR_OGL_CPU_POSES") != "1" and torch.is_tensor(poses) and torch.is_tensor(bboxes):
            if poses.is_cuda and bboxes.is_cuda and poses.dtype == torch.float32 and bboxes.dtype == torch.float32:
                poses_t = poses.detach()
                if poses_t.ndim == 2:
                    poses_t = poses_t.reshape(1, 4, 4)
                poses_t = poses_t.contiguous()
                bboxes_t = bboxes.detach().reshape(-1, 4).contiguous()
        if poses_t is None:
            if torch.is_tensor(poses):
                poses = poses.detach().cpu()
            if torch.is_tensor(bboxes):
                bboxes = bboxes.detach().cpu()
            poses_np = np.asarray(poses, dtype=np.float32)
            if poses_np.ndim == 2:
                poses_np = poses_np[None]
            bboxes_np = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
            n = int(poses_np.shape[0])
        else:
            n = int(poses_t.shape[0])
            poses_np = None
            bboxes_np = None
        row_mesh_ids = None
        if mesh_ids is not None:
            row_mesh_ids = [int(value) for value in mesh_ids]
            if len(row_mesh_ids) != n:
                raise ValueError("mesh_ids %d != tiles %d" % (len(row_mesh_ids), n))
            if len(set(row_mesh_ids)) <= 1:
                mesh_id = row_mesh_ids[0]
                row_mesh_ids = None
        one_mesh = row_mesh_ids is None
        if torch.is_tensor(K) and K.device.type != "cpu":
            K_np = np.asarray(K.detach().cpu().numpy(), dtype=np.float64)
        else:
            K_np = np.asarray(K.detach().numpy() if torch.is_tensor(K) else K, dtype=np.float64)
        if K_np.ndim == 2:
            K_rows = np.repeat(K_np.reshape(1, 3, 3), n, axis=0)
        else:
            K_rows = K_np.reshape(-1, 3, 3)
            if K_rows.shape[0] == 1 and n > 1:
                K_rows = np.repeat(K_rows, n, axis=0)
        one_camera = bool(np.allclose(K_rows, K_rows[:1]))
        if poses_t is not None and one_camera and n > 0:
            camera = K_rows[0]
            if one_mesh:
                return self._draw_device_tiles(mesh_id, poses_t, bboxes_t, camera, frame_h, frame_w, tile)
            return self._render_mixed_meshes(row_mesh_ids, poses_t, bboxes_t, camera, frame_h, frame_w, tile)
        caps = self._rt.gl_caps()
        spec = self._tile_spec(
            tile,
            indexed=bool(caps.viewport_array) and bool(caps.shader_viewport_layer_array) and bool(caps.multi_draw_indirect) and n <= int(caps.max_viewports),
        )
        light = native.LightParams(float(recipe.w_ambient), float(recipe.w_diffuse), tuple(recipe.light_dir))
        if poses_np is None:
            poses_np = poses_t.detach().float().cpu().numpy().reshape(n, 4, 4)
            bboxes_np = bboxes_t.detach().float().cpu().numpy().reshape(n, 4)
        instances = []
        for i in range(n):
            instances.append(
                {
                    "mesh_id": int(mesh_id) if row_mesh_ids is None else int(row_mesh_ids[i]),
                    "tile_slot": int(i),
                    "T_cam_obj": np.ascontiguousarray(poses_np[i].reshape(-1), dtype=np.float32),
                    "clip_affine4": np.ascontiguousarray(
                        clip_affine4_from_bbox2d(bboxes_np[i], int(frame_h), int(frame_w)),
                        dtype=np.float32,
                    ),
                    "K": np.ascontiguousarray(K_rows[i], dtype=np.float64),
                }
            )
        self._rt.render(
            instances,
            np.ascontiguousarray(K_rows[0], dtype=np.float64),
            int(frame_w),
            int(frame_h),
            spec,
            light,
        )
        return self.pack_to_torch()

    def pack_to_torch(self):
        """
        # Copy the packed GL buffers into torch and return rgb and depth.

            The copy is device-to-device.

            A failed pack or a failed copy raises RuntimeError.

        ## Args

            - There are no arguments.
            - The native pack must report ok.

        ## Returns

            - rgb: (N, H, W, 3) float32, allocated on cuda of this runtime's ordinal.
            - depth: (N, H, W) float32 meters, on that same device.

        ---

        # 把已经打包的 GL 缓冲拷进 torch，返回 rgb 和 depth。

            这次拷贝是设备到设备。

            pack 失败或拷贝失败时抛出 RuntimeError。

        ## 参数

            - 没有参数。
            - 渲染模块的 pack 必须报告 ok。

        ## 返回

            - rgb: (N, H, W, 3) float32，分配在这个运行时序号对应的 cuda 上。
            - depth: (N, H, W) float32，单位米，在同一个设备上。

"""
        # Copy the packed device buffers into torch. cudaMemcpy kind 3 is device-to-device.
        # rgb is (N, H, W, 3), depth is (N, H, W), meters.
        # 把已经打包的设备缓冲拷进 torch。cudaMemcpy 的 kind 3 是设备到设备。
        # rgb 是 (N, H, W, 3)，depth 是 (N, H, W)，米。
        import ctypes

        info = self._rt.pack_device_ptrs()
        if not info.get("ok"):
            raise RuntimeError("pack")
        n, h, w = int(info["batch_size"]), int(info["height"]), int(info["width"])
        dev = "cuda:%d" % self._device
        rgb_ptr = int(info.get("rgb_ptr") or 0)
        depth_ptr = int(info.get("depth_ptr") or 0)
        # A new pair each draw, copied before the next draw reuses the GL buffer.
        # 每次绘制一对新缓冲，下一次绘制复用 GL 缓冲之前先拷完。
        depth = torch.empty((n, h, w), device=dev, dtype=torch.float32)
        rgb = torch.empty((n, h, w, 3), device=dev, dtype=torch.float32)
        cudart = ctypes.CDLL("libcudart.so")
        cuda_memcpy = cudart.cudaMemcpy
        cuda_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        cuda_memcpy.restype = ctypes.c_int
        pix = n * h * w
        err = cuda_memcpy(ctypes.c_void_p(depth.data_ptr()), ctypes.c_void_p(depth_ptr), pix * 4, 3)
        if err != 0:
            raise RuntimeError("pack depth %s" % err)
        err = cuda_memcpy(ctypes.c_void_p(rgb.data_ptr()), ctypes.c_void_p(rgb_ptr), pix * 12, 3)
        if err != 0:
            raise RuntimeError("pack rgb %s" % err)
        return rgb, depth


def runtime_for(device):
    """
    # Return the OGL renderer for one CUDA ordinal.

        The first call compiles OGL when its library is missing.

        The shader directory is ogl_native/shaders.

        The runtime is stored in _RUNTIMES.

    ## Args

        - device: anything parse_cuda_device accepts. None means ordinal 0. The same ordinal returns the same GpuRenderRuntime.

    ---

    # 返回一个 CUDA 序号对应的 OGL 渲染器。

        第一次调用时，如果 OGL 库不存在就先编译。

        着色器目录是 ogl_native/shaders。

        运行时存在 _RUNTIMES 里。

    ## 参数

        - device: parse_cuda_device 能接受的值。None 表示 0 号。同一个序号返回同一个 GpuRenderRuntime。

"""
    ordinal = parse_cuda_device(device)
    if ordinal not in _RUNTIMES:
        native = ensure_ogl()
        torch.cuda.set_device(ordinal)
        shader = str(_native_dir() / "shaders")
        _RUNTIMES[ordinal] = GpuRenderRuntime(native, ordinal, shader)
        print("renderer: ogl / 渲染器：ogl", flush=True)
    return _RUNTIMES[ordinal]


def depth2xyzmap_batch(depths, Ks):
    """
    # Convert depth to camera coordinates.

        Depth in meters becomes camera-frame xyz in OpenCV, shaped (B, H, W, 3).

    ## Args

        - depths: (B, H, W) meters, on the torch device used for the grid. Values below 0.001, one millimeter, become 0 in the output.
        - Ks: OpenCV intrinsics in pixels, (3, 3) or (B, 3, 3). A single matrix is expanded to B. x uses column index u, and y uses row index v.

    ## Returns

        - The return is (B, H, W, 3), meters, in camera coordinates.
        - z: the input depth.

    ---

    # 把深度转到相机坐标。

        米制深度变成 OpenCV 相机坐标系的 xyz，形状是 (B, H, W, 3)。

    ## 参数

        - depths: (B, H, W)，单位米，网格使用它所在的 torch 设备。小于 0.001（1 毫米）的值在输出里变成 0。
        - Ks: OpenCV 内参，单位像素，形状 (3, 3) 或 (B, 3, 3)。单个矩阵会扩到 B。x 使用列下标 u，y 使用行下标 v。

    ## 返回

        - 返回 (B, H, W, 3)，单位米，相机坐标。
        - z 就是输入深度。

"""
    bs = depths.shape[0]
    if Ks.ndim == 2:
        Ks = Ks[None].expand(bs, -1, -1)
    invalid = depths < 0.001
    H, W = depths.shape[-2:]
    vs, us = torch.meshgrid(torch.arange(0, H, device=depths.device), torch.arange(0, W, device=depths.device), indexing="ij")
    vs = vs.reshape(-1).float()[None].expand(bs, -1)
    us = us.reshape(-1).float()[None].expand(bs, -1)
    zs = depths.reshape(bs, -1)
    Ks_e = Ks[:, None].expand(bs, zs.shape[-1], 3, 3)
    xs = (us - Ks_e[..., 0, 2]) * zs / Ks_e[..., 0, 0]
    ys = (vs - Ks_e[..., 1, 2]) * zs / Ks_e[..., 1, 1]
    xyz = torch.stack([xs, ys, zs], dim=-1).reshape(bs, H, W, 3)
    xyz = xyz.clone()
    xyz[invalid] = 0
    return xyz


def normalize_xyz_maps(xyz_map, poseA, mesh_diameter):
    """
    # Subtract the camera translation and divide by the object radius, diameter over 2.

    ## Args

        - xyz_map: (B, 3, H, W) meters. Its third channel is camera z.
        - poseA: (B, 4, 4). Only the translation poseA[:, :3, 3] is subtracted. The rotation is not applied.
        - mesh_diameter: meters, one value or (B,). The radius is that diameter divided by 2.

    ## Returns

        - The return has the same shape as xyz_map.
        - Input z below 0.001, and any absolute value at or above 2 after the division, becomes 0.

    ---

    # 减去相机坐标系里的平移，再除以物体半径，也就是直径的一半。

    ## 参数

        - xyz_map: (B, 3, H, W)，单位米。第三通道是相机 z。
        - poseA: (B, 4, 4)。只减去平移 poseA[:, :3, 3]。旋转不会被用到。
        - mesh_diameter: 米，一个数值或 (B,)。半径是这个直径除以 2。

    ## 返回

        - 返回值与 xyz_map 形状相同。
        - 输入 z 小于 0.001，以及除法之后绝对值大于或等于 2 的位置，变成 0。

"""
    bs = xyz_map.shape[0]
    mesh_radius = torch.as_tensor(mesh_diameter, device=xyz_map.device, dtype=xyz_map.dtype).reshape(-1) / 2.0
    if mesh_radius.numel() == 1:
        mesh_radius = mesh_radius.expand(bs)
    invalid = xyz_map[:, 2:3] < 0.001
    xyz = xyz_map - poseA[:, :3, 3].reshape(bs, 3, 1, 1)
    xyz = xyz * (1.0 / mesh_radius.reshape(bs, 1, 1, 1))
    invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(xyz) >= 2)
    xyz = xyz.clone()
    xyz[invalid] = 0
    return xyz


def _diameter_radius(mesh_diameter, B, crop_ratio, device):
    """
    # Compute the crop radius.

        Half of crop_ratio times the diameter, in meters, for B poses.

        crop_ratio multiplies the diameter before the division by 2.

    ## Args

        - mesh_diameter: meters, one value or a tensor. One value is expanded to B.
        - B: the pose count used when one diameter is expanded.
        - device: the torch device of the returned tensor.

    ## Returns

        - The return is float32 (B,).

    ---

    # 计算裁剪半径。

        crop_ratio 乘直径的一半，单位米，对应 B 个位姿。

        crop_ratio 先乘直径，再除以 2。

    ## 参数

        - mesh_diameter: 米，一个数值或一个张量。一个数值会扩到 B。
        - B: 单个直径被扩展时使用的位姿数。
        - device: 返回张量的 torch 设备。

    ## 返回

        - 返回 float32 (B,)。

"""
    dia = torch.as_tensor(mesh_diameter, device=device, dtype=torch.float32).reshape(-1)
    if dia.numel() == 1:
        dia = dia.expand(int(B))
    return dia * float(crop_ratio) / 2.0


def _axis_offsets(radius):
    """
    # Return the origin and the plus and minus X and Y offsets.

        Z stays 0.

    ## Args

        - radius: (B,) meters. The return is (B, 5, 3) on the same device and dtype: origin, +X, -X, +Y, -Y.

    ---

    # 返回原点，以及 +X、-X、+Y、-Y 偏移。

        Z 保持 0。

    ## 参数

        - radius: (B,)，单位米。返回 (B, 5, 3)，设备和 dtype 与 radius 相同：原点、+X、-X、+Y、-Y。

"""
    z = torch.zeros_like(radius)
    return torch.stack(
        [
            torch.stack([z, z, z], dim=-1),
            torch.stack([radius, z, z], dim=-1),
            torch.stack([-radius, z, z], dim=-1),
            torch.stack([z, radius, z], dim=-1),
            torch.stack([z, -radius, z], dim=-1),
        ],
        dim=1,
    )


def compute_crop_window(poses, Ks, crop_ratio, out_size, mesh_diameter):
    """
    # Compute the crop window.

        A square window in pixels, around the projected object origin.

        crop_ratio multiplies the diameter before the half-extent.

        out_size is (width, height) in pixels: index 0 is width and index 1 is height.

        The map return is (B, 3, 3).

        It sends full-frame pixels into the crop of size out_size.

        The bbox return is (B, 4) left, top, right, bottom, full-frame pixels, y down.

        The pixel half-size is at least 1.

    ## Args

        - poses: (B, 4, 4) torch, object-to-camera, meters. The five sample points are the camera-frame translation plus the X and Y radius offsets. The rotation is not applied.
        - Ks: OpenCV intrinsics in pixels, (3, 3) or (B, 3, 3), moved to poses.device as float32.
        - mesh_diameter: meters, one value or (B,).

    ---

    # 计算裁剪窗口。

        以物体原点的投影为中心的正方形窗口，单位是像素。

        crop_ratio 在取半边长之前乘直径。

        out_size 是像素 (width, height)：下标 0 是宽，下标 1 是高。

        映射返回值是 (B, 3, 3)。

        它把整幅图像素送进 out_size 那么大的裁剪块。

        bbox 返回值是 (B, 4)，即 left、top、right、bottom，整幅图像素，y 向下。

        像素半边长至少是 1。

    ## 参数

        - poses: (B, 4, 4) torch，物体到相机，单位米。五个采样点是相机坐标系平移再加上 X 和 Y 的半径偏移。旋转不会被用到。
        - Ks: OpenCV 内参，单位像素，形状 (3, 3) 或 (B, 3, 3)，会转到 poses.device 并成为 float32。
        - mesh_diameter: 米，一个数值或 (B,)。

"""
    device = poses.device
    B = len(poses)
    out_w, out_h = int(out_size[0]), int(out_size[1])
    Ks = torch.as_tensor(Ks, device=device, dtype=torch.float32).reshape(-1, 3, 3)
    if Ks.shape[0] == 1 and B > 1:
        Ks = Ks.expand(B, -1, -1).contiguous()
    radius = _diameter_radius(mesh_diameter, B, crop_ratio, device)
    pts = poses[:, :3, 3].reshape(-1, 1, 3) + _axis_offsets(radius)
    projected = torch.einsum("bij,bkj->bki", Ks, pts)
    z = projected[:, :, 2:3].clone()
    z[z == 0] = 1
    uvs = projected[:, :, :2] / z
    center = uvs[:, 0]
    radius_b = torch.abs(uvs - center.reshape(-1, 1, 2)).reshape(B, -1).max(dim=-1)[0]
    radius_b = torch.clamp(radius_b, min=1.0)
    left = center[:, 0] - radius_b
    right = center[:, 0] + radius_b
    top = center[:, 1] - radius_b
    bottom = center[:, 1] + radius_b
    tf = torch.eye(3, device=device)[None].expand(B, -1, -1).contiguous().clone()
    tf[:, 0, 2] = -left
    tf[:, 1, 2] = -top
    new_tf = torch.eye(3, device=device)[None].expand(B, -1, -1).contiguous().clone()
    new_tf[:, 0, 0] = out_w / (right - left).clamp(min=1e-3)
    new_tf[:, 1, 1] = out_h / (bottom - top).clamp(min=1e-3)
    bboxes = torch.stack([left, top, right, bottom], dim=1)
    return new_tf @ tf, bboxes


def _crop_adjusted_K(Ks, bboxes, out_h, out_w):
    """
    # Adjust the camera matrix into the crop.

        K is in pixels, inside the crop window.

    ## Args

        - Ks: (B, 3, 3), OpenCV, pixels.
        - bboxes: (B, 4) left, top, right, bottom, pixels, y down.
        - out_h: the crop height in pixels. fy and cy scale with out_h divided by the bbox height.
        - out_w: the crop width in pixels. fx and cx scale with out_w divided by the bbox width. The principal point moves to the window origin.

    ## Returns

        - The return is a cloned (B, 3, 3).

    ---

    # 把相机矩阵变到裁剪窗口里。

        K 的单位是像素。

    ## 参数

        - Ks: (B, 3, 3)，OpenCV，单位像素。
        - bboxes: (B, 4)，即 left、top、right、bottom，单位像素，y 向下。
        - out_h: 裁剪高度，单位像素。fy 和 cy 按 out_h 除以包围盒高度来缩放。
        - out_w: 裁剪宽度，单位像素。fx 和 cx 按 out_w 除以包围盒宽度来缩放。主点移到窗口原点。

    ## 返回

        - 返回克隆出来的 (B, 3, 3)。

"""
    left, top, right, bottom = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    a1 = float(out_w) / (right - left).clamp(min=1e-3)
    a2 = float(out_h) / (bottom - top).clamp(min=1e-3)
    Ks_c = Ks.clone()
    Ks_c[:, 0, 0] = Ks[:, 0, 0] * a1
    Ks_c[:, 1, 1] = Ks[:, 1, 1] * a2
    Ks_c[:, 0, 2] = Ks[:, 0, 2] * a1 - left * a1
    Ks_c[:, 1, 2] = Ks[:, 1, 2] * a2 - top * a2
    return Ks_c


def make_crop_pair(rgb, depth, mask, poses, K, mesh, diameter_m, crop_ratio, use_mask, device, meshes=None, observed_xyz=None, mesh_ids=None):
    """
    # Build the rendered view A and the observed view B, and return the poses and diameters with them.

        crop_ratio multiplies the diameter.

        The window half-extent in meters is crop_ratio times diameter over 2.

        use_mask True stacks a mask channel.

        False leaves RGB and xyz only.

        meshes None uses mesh for every pose.

        A sequence must have one mesh per pose.

        A is the rendered view, float32 and contiguous.

        use_mask makes it (N, 7, S, S): RGB, mask, xyz.

        Otherwise it is (N, 6, S, S): RGB, xyz.

        S is recipe.crop_px.

        RGB is 0-1.

        xyz is the camera-frame point with the pose translation removed and divided by half the diameter.

        Depth below 1 mm, or an absolute value at or above 2, is 0.

    ## Args

        - rgb: channel-last RGB, (H, W, 3) or (N, H, W, 3), cast to float32. A leading size of 1 is expanded to the pose count. It is later divided by 255, so the stored numbers are on the 0-255 scale.
        - depth: (H, W) or (N, H, W), cast to float32, meters. A leading size of 1 is expanded. Values below 0.001 m become 0 in the xyz map.
        - mask: (H, W) or (N, H, W), cast to float32. A leading size of 1 is expanded. After the warp, values above 0 become 1.
        - poses: (4, 4) or (N, 4, 4), object-to-camera, meters. A single matrix gains a batch axis. The stored tensor is float32 on device.
        - K: OpenCV intrinsics in pixels, (3, 3) or (N, 3, 3). One matrix is expanded to N.
        - mesh: a trimesh.Trimesh or a path. It is used for every pose when meshes is None.
        - diameter_m: meters, one value or (N,), expanded to one float32 per pose.
        - device: passed to torch.device. rgb, depth, mask, poses, and K are moved there.

    ## Returns

        - B: the observed view, with the same shape and channel order as A.
        - RGB: a bilinear warp of the input divided by 255. Depth, mask, and xyz are nearest warps, and xyz uses the same normalization as A.
        - poseA: the input poses as float32 (N, 4, 4) on device, object-to-camera, meters.
        - dia: (N,) float32 meters, the expanded diameter.

    ---

    # 构造渲染视图 A 和观测视图 B，并一起返回位姿和直径。

        crop_ratio 乘直径。

        窗口的半边长是 crop_ratio 乘直径再除以 2，单位米。

        use_mask 为 True 时叠上一个 mask 通道。

        False 时只留下 RGB 和 xyz。

        meshes 为 None 时，每个位姿都用 mesh。

        给出序列时，每个位姿一个网格。

        A 是渲染视图，float32 且 contiguous。

        use_mask 时它是 (N, 7, S, S)：RGB、mask、xyz。

        否则是 (N, 6, S, S)：RGB、xyz。

        S 是 recipe.crop_px。

        RGB 是 0-1。

        xyz 是去掉位姿平移后再除以直径一半的相机坐标点。

        深度小于 1 毫米，或绝对值大于或等于 2 的位置，是 0。

    ## 参数

        - rgb: 通道在最后一维的 RGB，(H, W, 3) 或 (N, H, W, 3)，转成 float32。第 0 维为 1 时会扩到位姿数。之后会除以 255，因此存下的数在 0-255 尺度上。
        - depth: (H, W) 或 (N, H, W)，转成 float32，单位米。第 0 维为 1 时会扩展。小于 0.001 米的值在 xyz 图里变成 0。
        - mask: (H, W) 或 (N, H, W)，转成 float32。第 0 维为 1 时会扩展。变换之后，大于 0 的值变成 1。
        - poses: (4, 4) 或 (N, 4, 4)，物体到相机，单位米。单个矩阵会增加 batch 维。存下的张量是 device 上的 float32。
        - K: OpenCV 内参，单位像素，形状 (3, 3) 或 (N, 3, 3)。一个矩阵会扩到 N。
        - mesh: trimesh.Trimesh 或路径。
        - meshes: None 时，每个位姿都用它。
        - diameter_m: 米，一个数值或 (N,)，会扩成每个位姿一个 float32。
        - device 传给 torch.device。
        - rgb、depth、mask、poses 和 K 都会转到那里。

    ## 返回

        - B: 观测视图，形状和通道顺序与 A 相同。
        - RGB: 输入图的双线性变换再除以 255。深度、mask 和 xyz 是最近邻变换，xyz 使用与 A 相同的归一化。
        - poseA: 输入位姿，device 上的 float32 (N, 4, 4)，物体到相机，单位米。
        - dia: (N,) float32，单位米，即扩展后的直径。

"""
    import kornia

    device = torch.device(device)
    rgb_t = torch.as_tensor(rgb, device=device, dtype=torch.float32)
    depth_t = torch.as_tensor(depth, device=device, dtype=torch.float32)
    mask_t = torch.as_tensor(mask, device=device, dtype=torch.float32) if use_mask else None
    poseA = torch.as_tensor(poses, device=device, dtype=torch.float32)
    if poseA.ndim == 2:
        poseA = poseA[None]
    if rgb_t.ndim == 3:
        rgb_t = rgb_t[None]
    if depth_t.ndim == 2:
        depth_t = depth_t[None]
    if use_mask and mask_t.ndim == 2:
        mask_t = mask_t[None]
    B = int(poseA.shape[0])
    if int(rgb_t.shape[0]) == 1 and B > 1:
        rgb_t = rgb_t.expand(B, -1, -1, -1)
    if int(depth_t.shape[0]) == 1 and B > 1:
        depth_t = depth_t.expand(B, -1, -1)
    if use_mask and int(mask_t.shape[0]) == 1 and B > 1:
        mask_t = mask_t.expand(B, -1, -1)
    H, W = int(depth_t.shape[-2]), int(depth_t.shape[-1])
    Ks = torch.as_tensor(K, device=device, dtype=torch.float32).reshape(-1, 3, 3)
    if Ks.shape[0] == 1 and B > 1:
        Ks = Ks.expand(B, -1, -1)
    if torch.is_tensor(K) and K.device.type != "cpu":
        K_np = np.asarray(K.detach().cpu().numpy(), dtype=np.float64)
    else:
        K_np = np.asarray(K.detach().numpy() if torch.is_tensor(K) else K, dtype=np.float64)
    crop = int(recipe.crop_px)
    dia = torch.as_tensor(diameter_m, device=device, dtype=torch.float32).reshape(-1)
    if dia.numel() == 1:
        dia = dia.expand(B)
    tf, bboxes = compute_crop_window(poseA, Ks, crop_ratio, (crop, crop), dia)
    # B is the observed image, warped into the crop. RGB is bilinear; depth and mask are nearest.
    # B 是观测图，变到裁剪块里。RGB 用双线性，深度和 mask 用最近邻。
    rgbBs = kornia.geometry.transform.warp_perspective(
        rgb_t.permute(0, 3, 1, 2), tf, dsize=(crop, crop), mode="bilinear", align_corners=False
    )
    depthBs = kornia.geometry.transform.warp_perspective(
        depth_t[:, None], tf, dsize=(crop, crop), mode="nearest", align_corners=False
    )
    # SAPR and WBPS omit the mask channel; skip its full batched warp.
    # SAPR 和 WBPS 不读取 mask 通道，因此跳过整批 mask 透视变换。
    if use_mask:
        maskBs = kornia.geometry.transform.warp_perspective(
            mask_t[:, None], tf, dsize=(crop, crop), mode="nearest", align_corners=False
        )
        maskBs = (maskBs > 0).float()
    # A caller may share one observed xyz map across every update and score crop.
    # 调用方可让所有修正及评分裁剪共享一份观测 xyz 图，形状为 (1/N, 3, H, W)。
    if observed_xyz is not None:
        xyz_full = torch.as_tensor(observed_xyz, device=device, dtype=torch.float32)
        if xyz_full.ndim != 4 or tuple(xyz_full.shape[1:]) != (3, H, W):
            raise ValueError("observed_xyz must have shape (1/N, 3, H, W)")
        if int(xyz_full.shape[0]) == 1:
            xyz_full = xyz_full.expand(B, -1, -1, -1)
        elif int(xyz_full.shape[0]) != B:
            raise ValueError("observed_xyz rows differ from pose rows")
    elif B > 1 and int(depth_t.stride(0)) == 0:
        xyz_1 = depth2xyzmap_batch(depth_t[:1].contiguous(), Ks[:1]).permute(0, 3, 1, 2)
        xyz_full = xyz_1.expand(B, -1, -1, -1)
    else:
        xyz_full = depth2xyzmap_batch(depth_t, Ks).permute(0, 3, 1, 2)
    xyzBs = kornia.geometry.transform.warp_perspective(
        xyz_full, tf, dsize=(crop, crop), mode="nearest", align_corners=False
    )
    rt = runtime_for(device)
    # A is one tile per pose, meters. meshes gives each pose its own mesh.
    # A 是每个位姿一块，单位米。meshes 让每个位姿用自己的网格。
    if mesh_ids is not None:
        if len(mesh_ids) != B:
            raise ValueError("mesh_ids rows differ from pose rows")
        rgb_r, depth_r = rt.render_tiles(
            mesh_ids[0], poseA, bboxes, K_np, H, W, crop, mesh_ids=mesh_ids,
        )
    elif meshes is None:
        mesh_id = rt.load_mesh_trimesh(mesh, name="mesh")
        rgb_r, depth_r = rt.render_tiles(mesh_id, poseA, bboxes, K_np, H, W, crop)
    else:
        if len(meshes) != B:
            raise ValueError("meshes %d != poses %d" % (len(meshes), B))
        rt.ensure_mesh_slots(len(meshes))
        # Hypotheses often repeat the same object mesh. Resolve each mesh only
        # once per crop step, then preserve the original per-pose row order.
        # 同一物体的多个候选姿态复用物体网格模型；每步只查一次编号，再保持位姿行序。
        mesh_id_by_object = {}
        mesh_ids = []
        for item in meshes:
            object_key = id(item)
            if object_key not in mesh_id_by_object:
                mesh_id_by_object[object_key] = rt.load_mesh_trimesh(item, name="mesh")
            mesh_ids.append(mesh_id_by_object[object_key])
        rgb_r, depth_r = rt.render_tiles(
            mesh_ids[0], poseA, bboxes, K_np, H, W, crop, mesh_ids=mesh_ids,
        )
    if float(rgb_r.detach().max()) <= 1.5:
        rgbAs = (rgb_r * 255.0).permute(0, 3, 1, 2).contiguous()
    else:
        rgbAs = rgb_r.permute(0, 3, 1, 2).contiguous()
    depthAs = depth_r[:, None].float()
    maskAs = (depthAs > 0).float()
    Ks_crop = _crop_adjusted_K(Ks, bboxes.float(), crop, crop)
    xyzAs = depth2xyzmap_batch(depthAs[:, 0], Ks_crop).permute(0, 3, 1, 2)
    rgbAs_n = rgbAs.float() / 255.0
    rgbBs_n = rgbBs.float() / 255.0
    xyzAs_n = normalize_xyz_maps(xyzAs, poseA, dia)
    xyzBs_n = normalize_xyz_maps(xyzBs, poseA, dia)
    if use_mask:
        A = torch.cat([rgbAs_n, maskAs, xyzAs_n], dim=1)
        B = torch.cat([rgbBs_n, maskBs, xyzBs_n], dim=1)
    else:
        A = torch.cat([rgbAs_n, xyzAs_n], dim=1)
        B = torch.cat([rgbBs_n, xyzBs_n], dim=1)
    return A.contiguous().float(), B.contiguous().float(), poseA, dia


__all__ = ["make_crop_pair"]
