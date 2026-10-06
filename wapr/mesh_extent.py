# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Mesh and depth size notices. The call continues either way.
# 网格和深度的尺度提示。无论是否异常，调用都继续。
import warnings

import numpy as np

# Below this, the file is about a millimeter. A real object here is rarely that small.
# 低于这个值，文件大约只有一毫米。这里的真实物体很少这么小。
extent_small_m = 0.002
# Above this, the file is past 10 m. That is beyond a typical depth camera.
# 高于这个值，文件超过 10 米，一般已经超出深度相机的范围。
# A couple of meters still prints the size and does not warn.
# 一两米仍会打印尺寸，但不警告。
extent_large_m = 10.0

_reported = set()


def mesh_extent_m(vertices):
    """
    # Axis-aligned diagonal, in the same unit as the vertices.

    ## Args

        - vertices: (V, 3) or wider. An empty array, or fewer than three columns, raises ValueError.

    ## Returns

        - Returns one float.

    ---

    # 轴对齐包围盒的对角线，单位与顶点相同。

    ## 参数

        - vertices: (V, 3) 或更宽。没有顶点，或列数少于 3，抛出 ValueError。

    ## 返回

        - 返回一个浮点数。

"""
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
        raise ValueError("mesh has no vertices")
    span = points[:, :3].max(axis=0) - points[:, :3].min(axis=0)
    return float(np.linalg.norm(span))


def report_mesh_extent(vertices, name):
    """
    # Report the mesh size.

        The diagonal is printed in meters and millimeters.

        Warn when that size is unlikely.

        Returns the diagonal in meters.

        A diagonal below extent_small_m, 0.002 m, or above extent_large_m, 10 m, warns.

        The call still returns.

    ## Args

        - vertices: the vertex array, treated as meters.
        - name: the label on the MESH_EXTENT line. The same name and rounded size is printed once. The vertices are not changed.

    ---

    # 报告网格尺寸。

        对角线以米和毫米打印。

        尺度不像常见物体时警告。

        返回对角线，米。

        对角线低于 extent_small_m（0.002 米）或高于 extent_large_m（10 米）时警告。

        调用仍然返回。

    ## 参数

        - vertices: 顶点，按米理解。
        - name: MESH_EXTENT 那一行的名字。同一个名字和取整后的尺寸只打印一次。不改顶点。

"""
    extent_m = mesh_extent_m(vertices)
    key = ("mesh", str(name), round(extent_m, 6))
    if key in _reported:
        return extent_m
    _reported.add(key)
    extent_mm = extent_m * 1000.0
    print(
        "MESH_EXTENT",
        {"name": str(name), "extent_m": round(extent_m, 6), "extent_mm": round(extent_mm, 3)},
        flush=True,
    )
    if extent_m < extent_small_m or extent_m > extent_large_m:
        warnings.warn(
            "mesh %s extent is %.4f m (%.1f mm). "
            "Objects here are usually a few centimeters to a couple of meters. "
            "About a millimeter, or more than 10 m, often means millimeters and meters were mixed. "
            "The call continues. "
            "网格 %s 的尺度是 %.4f 米（%.1f 毫米）。"
            "这里的物体通常是几厘米到一两米。"
            "大约一毫米，或超过 10 米，常常是毫米和米混用了。"
            "调用会继续。"
            % (name, extent_m, extent_mm, name, extent_m, extent_mm),
            stacklevel=3,
        )
    return extent_m


def report_depth_m(depth_m, name):
    """
    # Report the depth.

        The median positive depth is printed in meters.

        Warn when that median is unlikely.

        Returns the median, or None when no pixel is valid.

        depth.npy in a custom folder is already meters.

        A raw millimeter image belongs in depth.png, and load_scene divides it by 1000 before this call.

    ## Args

        - depth_m: HxW meters.
        - name: the label on the DEPTH_MEDIAN line. The same name and rounded median is printed once.

    ---

    # 报告深度。

        打印有效深度的中位数，单位是米。

        中位数不像一帧深度时警告。

        返回中位数。

        没有有效像素时返回 None。

        自定义目录里的 depth.npy 已经是米。

        原始毫米图放在 depth.png，load_scene 先除以 1000 再调用这里。

    ## 参数

        - depth_m: HxW，米。
        - name: DEPTH_MEDIAN 那一行的名字。同一个名字和取整后的中位数只打印一次。

"""
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = depth[np.isfinite(depth) & (depth > 1e-6)]
    if valid.size == 0:
        warnings.warn(
            "depth %s has no positive finite values. 深度 %s 没有正的有限值。 The call continues."
            % (name, name),
            stacklevel=2,
        )
        return None
    median_m = float(np.median(valid))
    key = ("depth", str(name), round(median_m, 6))
    if key in _reported:
        return median_m
    _reported.add(key)
    print(
        "DEPTH_MEDIAN",
        {"name": str(name), "median_m": round(median_m, 6), "median_mm": round(median_m * 1000.0, 3)},
        flush=True,
    )
    if median_m < extent_small_m or median_m > extent_large_m:
        warnings.warn(
            "depth %s median is %.4f m (%.1f mm). "
            "A frame is usually tens of centimeters to a few meters. "
            "depth.png is raw times depth_scale, in millimeters, then divided by 1000. "
            "depth.npy is already meters. The call continues. "
            "深度 %s 的中位数是 %.4f 米（%.1f 毫米）。"
            "一帧通常是几十厘米到几米。"
            "depth.png 是原始值乘 depth_scale，单位毫米，再除以 1000。"
            "depth.npy 已经是米。调用会继续。"
            % (name, median_m, median_m * 1000.0, name, median_m, median_m * 1000.0),
            stacklevel=2,
        )
    return median_m
