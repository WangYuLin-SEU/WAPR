# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# One custom RGB-D folder. Bundled BOP sets do not use this layout.
# 一个自定义 RGB-D 目录。随包的 BOP 数据集不走这个目录。
import json
import os

import cv2
import numpy as np
import trimesh

from wapr.mesh_extent import report_depth_m

# Mesh files trimesh reads here. The unit is mesh_unit in objects.json, not the suffix.
# 这里 trimesh 读取的网格。单位看 objects.json 的 mesh_unit，不看后缀。
mesh_suffixes = (".ply", ".obj", ".stl", ".glb", ".gltf")


def _first_file(scene_dir, names):
    """
    # The first existing file among names, or None.

    ## Args

        - scene_dir: the folder.
        - names: a sequence of file names, not paths. The join is scene_dir plus that name.

    ---

    # 返回 names 中第一个存在的文件。

        如果均不存在，则返回 None。

    ## 参数

        - scene_dir: 目录。
        - names: 文件名序列，不是路径。逐个与 scene_dir 拼接后检查是否存在。

"""
    for name in names:
        path = os.path.join(scene_dir, name)
        if os.path.isfile(path):
            return path
    return None


def _read_rgb(path):
    """
    # Read one RGB image.

        A BGR file comes back as RGB uint8, HxWx3.

    ## Args

        - path: rgb.png or rgb.jpg. A missing or unreadable file raises FileNotFoundError.

    ---

    # 读入一张 RGB 图像。

        BGR 文件返回为 RGB uint8，HxWx3。

    ## 参数

        - path: rgb.png 或 rgb.jpg。文件不存在或读不出来时抛出 FileNotFoundError。

"""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_depth_image(path, depth_scale):
    """
    # Read one depth image.

        Raw image times depth_scale is millimeters.

        Divide by 1000 and return meters, float32.

    ## Args

        - path: depth.png, depth.tif, or depth.tiff. depth_scale comes from camera.json and defaults to 1. A 3-channel image keeps the first channel. An unreadable file raises FileNotFoundError.

    ---

    # 读入一张深度图。

        原始图像乘 depth_scale 是毫米。

        再除以 1000，返回米，float32。

    ## 参数

        - path: depth.png、depth.tif 或 depth.tiff。depth_scale 来自 camera.json，默认 1。三通道图只留第一通道。读不出来时抛出 FileNotFoundError。

"""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    depth = np.asarray(image, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth * float(depth_scale) / 1000.0


def _load_one_mesh(path, mesh_unit, name):
    """
    # Load one mesh.

        The mesh comes back in meters, and name is stored on metadata.

    ## Args

        - path: ply, obj, stl, glb, or gltf.
        - mesh_unit: mm or m. mm is scaled by 0.001. Any other unit raises ValueError. A file that is not one Trimesh raises TypeError.
        - name: the string later drawn on the image.

    ---

    # 载入一个网格。

        网格换成米，并把 name 写入 metadata。

    ## 参数

        - path: ply、obj、stl、glb 或 gltf。
        - mesh_unit: mm 或 m。mm 会乘 0.001。其他单位抛出 ValueError。不是单个 Trimesh 时抛出 TypeError。
        - name: 之后画在图上的字符串。

    ## 返回

        - 返回这个网格。

"""
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError("expected one mesh: %s" % path)
    mesh = loaded.copy()
    if mesh_unit == "mm":
        mesh.apply_scale(0.001)
    elif mesh_unit != "m":
        raise ValueError("mesh_unit must be mm or m, got %s" % mesh_unit)
    metadata = dict(mesh.metadata or {})
    metadata["name"] = str(name)
    metadata["mesh_unit"] = mesh_unit
    mesh.metadata = metadata
    return mesh


def _diameter_m(record, mesh):
    """
    # Read the object diameter in meters.

        The value is in meters.

    ## Args

        - record: one objects.json entry. diameter_m is used as given. diameter_mm is divided by 1000. Both fields together raise ValueError. Neither field uses the mesh bounds diagonal, already in meters.

    ## Returns

        - Returns one float.

    ---

    # 读取物体直径，单位米。

        数值的单位是米。

    ## 参数

        - record: objects.json 中的一条记录。若有 diameter_m，则使用该值。若有 diameter_mm，则除以 1000。两个字段同时存在时抛出 ValueError。两者都没有时，使用网格包围盒对角线；此时顶点单位已经是米。

    ## 返回

        - 返回一个浮点数。

"""
    if "diameter_m" in record and "diameter_mm" in record:
        raise ValueError("object %s has both diameter_m and diameter_mm" % record.get("id"))
    if "diameter_m" in record:
        return float(record["diameter_m"])
    if "diameter_mm" in record:
        return float(record["diameter_mm"]) / 1000.0
    span = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
    return float(np.linalg.norm(span))


def load_scene(scene_dir):
    """
    # Load one scene of many category meshes.

        One folder. Returns RGB, depth, K, and meshes. No masks and no poses.

    ## Args

        - scene_dir: rgb.png or rgb.jpg, depth.png or depth.tif or depth.npy, camera.json, and objects.json. camera.json holds fx, fy, cx, cy in pixels, and depth_scale. depth_scale defaults to 1. A depth image times depth_scale is millimeters, then divided by 1000. depth.npy is already meters, and depth_scale is not applied to it.

    ## Returns

        - Vertices and depth come back in meters.
        - objects.json: mesh_unit, mm or m, and objects. Each object has id, name, and file.
        - file: relative to scene_dir.
        - id: the integer obj_id.
        - name: the string drawn on the image.
        - Returns rgb, depth_m, K, meshes, names.
        - meshes: obj_id to (mesh, diameter_m).
        - names: obj_id to name.
        - K: 3×3 float32.

    ---

    # 载入一个多类别网格的场景。

        一个目录。返回 RGB、深度、K 和网格。没有 mask，也没有位姿。

    ## 参数

        - scene_dir 里要有 rgb.png 或 rgb.jpg，depth.png、depth.tif 或 depth.npy，camera.json，以及 objects.json。
        - camera.json 里是像素单位的 fx、fy、cx、cy，以及 depth_scale。
        - depth_scale 默认 1。
        - 深度图乘 depth_scale 是毫米，再除以 1000。
        - depth.npy 已经是米，不再乘 depth_scale。

    ## 返回

        - 返回的顶点和深度都是米。
        - objects.json 里有 mesh_unit，取 mm 或 m，以及 objects。
        - 每个物体有 id、name、file。
        - file 相对 scene_dir。
        - id: 整数 obj_id。
        - name: 画在图上的字符串。
        - 返回 rgb、depth_m、K、meshes、names。
        - meshes 把 obj_id 映射到 (mesh, diameter_m)。
        - names 把 obj_id 映射到 name。
        - K: 3×3 float32。

"""
    scene_dir = os.path.abspath(scene_dir)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(scene_dir)
    rgb_path = _first_file(scene_dir, ("rgb.png", "rgb.jpg", "rgb.jpeg"))
    if rgb_path is None:
        raise FileNotFoundError("rgb.png or rgb.jpg in %s" % scene_dir)
    camera_path = os.path.join(scene_dir, "camera.json")
    objects_path = os.path.join(scene_dir, "objects.json")
    with open(camera_path, "r") as stream:
        camera = json.load(stream)
    with open(objects_path, "r") as stream:
        listing = json.load(stream)
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    depth_scale = float(camera.get("depth_scale", 1.0))
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    npy_path = os.path.join(scene_dir, "depth.npy")
    if os.path.isfile(npy_path):
        depth_m = np.load(npy_path).astype(np.float32)
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]
    else:
        depth_path = _first_file(scene_dir, ("depth.png", "depth.tif", "depth.tiff"))
        if depth_path is None:
            raise FileNotFoundError("depth.png, depth.tif, or depth.npy in %s" % scene_dir)
        depth_m = _read_depth_image(depth_path, depth_scale)
    rgb = _read_rgb(rgb_path)
    if depth_m.shape[:2] != rgb.shape[:2]:
        raise ValueError("rgb and depth sizes differ")
    report_depth_m(depth_m, os.path.basename(scene_dir))
    mesh_unit = str(listing["mesh_unit"])
    records = listing["objects"]
    if not records:
        raise ValueError("objects.json has no objects")
    meshes = {}
    names = {}
    for record in records:
        obj_id = int(record["id"])
        if obj_id in meshes:
            raise ValueError("repeated object id %d" % obj_id)
        name = str(record["name"])
        file_field = str(record["file"])
        mesh_path = file_field if os.path.isabs(file_field) else os.path.join(scene_dir, file_field)
        if os.path.splitext(mesh_path)[1].lower() not in mesh_suffixes:
            raise ValueError("mesh suffix must be one of %s: %s" % (", ".join(mesh_suffixes), mesh_path))
        mesh = _load_one_mesh(mesh_path, mesh_unit, name)
        meshes[obj_id] = (mesh, _diameter_m(record, mesh))
        names[obj_id] = name
    return rgb, depth_m, K, meshes, names
