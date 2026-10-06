# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Two pose calls. One category and one instance, or many categories and many instances.
# 两种位姿调用。单类别单实例，或者多类别多实例。
# Poses are object-to-camera, meters, OpenCV +Z forward.
# 位姿是物体到相机，单位米，OpenCV，+Z 朝前。
import os
import time
import weakref

import numpy as np
import torch

from wapr import recipe
from wapr.nets import load_net
from wapr.pose_groups import wbps_group_sizes
from wapr.ogl import depth2xyzmap_batch, make_crop_pair, runtime_for
from wapr.resources import source_checkout


def _rots_from_z(directions):
    """
    # Rotations that map +Z onto every direction at once.

    ## Args

        - directions: (N, 3). Each row is normalized. A direction already on +Z stays identity. A direction opposite +Z is a 180 degree turn about X.

    ## Returns

        - Returns (N, 3, 3) float32.

    ---

    # 一次把 +Z 转到每一个方向。

    ## 参数

        - directions: (N, 3)。每一行会归一化。已经在 +Z 上的方向保持单位阵。与 +Z 相反的方向绕 X 转 180 度。

    ## 返回

        - 返回 (N, 3, 3) float32。

"""
    target = np.asarray(directions, dtype=np.float64)
    target = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cross = np.cross(z_axis, target)
    cosine = target @ z_axis
    sine = np.linalg.norm(cross, axis=1)
    vx, vy, vz = cross[:, 0], cross[:, 1], cross[:, 2]
    skew = np.zeros((target.shape[0], 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -vz
    skew[:, 0, 2] = vy
    skew[:, 1, 0] = vz
    skew[:, 1, 2] = -vx
    skew[:, 2, 0] = -vy
    skew[:, 2, 1] = vx
    eye = np.eye(3, dtype=np.float64)
    scale = ((1.0 - cosine) / (sine * sine + 1e-12))[:, None, None]
    rotation = eye + skew + np.matmul(skew, skew) * scale
    rotation[sine < 1e-8] = eye
    opposite = cosine < -0.999999
    if np.any(opposite):
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        flip = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
            dtype=np.float64,
        )
        rotation[opposite] = eye + 2.0 * flip @ flip
    return rotation.astype(np.float32)


def _unit_dirs(pts, subtract_mean):
    """
    # Normalize points to unit directions.

    ## Args

        - pts: (N, 3). subtract_mean True recenters the solid before normalizing. make_view_dirs passes True for every solid and for the Fibonacci lattice.

    ## Returns

        - Returns (N, 3) float64.

    ---

    # 把点归一化成单位方向。

    ## 参数

        - pts: (N, 3)。
        - subtract_mean: True 时先把点移到质心，再归一化。make_view_dirs 对每种多面体和斐波那契格点都传入 True。

    ## 返回

        - 返回 (N, 3) float64。

"""
    pts = np.asarray(pts, dtype=np.float64)
    if subtract_mean:
        pts = pts - pts.mean(axis=0, keepdims=True)
    length = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / (length + 1e-8)


def make_view_dirs(n_view):
    """
    # Unit view directions in the OpenCV camera frame, +Z forward.

    ## Args

        - n_view: an integer. 4, 6, 8, 12, and 20 are the vertices of the tetrahedron, octahedron, cube, icosahedron, and dodecahedron. Any other count samples the sphere with a Fibonacci lattice.

    ## Returns

        - Returns (n_view, 3) float64.

    ---

    # 相机坐标系下的单位视线方向，OpenCV，+Z 朝前。

    ## 参数

        - n_view: 整数。4、6、8、12、20 分别是正四面体、正八面体、立方体、正二十面体、正十二面体的顶点。其他数量在球面上做斐波那契采样。

    ## 返回

        - 返回 (n_view, 3) float64。

"""
    n = int(n_view)
    if n == 4:
        # One vertex on +Z. The other three sit at z=-1/3, 120 degrees apart.
        # 一个顶点在 +Z。另外三个在 z=-1/3，彼此间隔 120°。
        height = -1.0 / 3.0
        radius = np.sqrt(8.0 / 9.0)
        pts = np.array(
            [
                [0.0, radius, height],
                [-np.sqrt(2.0 / 3.0), -np.sqrt(2.0 / 9.0), height],
                [np.sqrt(2.0 / 3.0), -np.sqrt(2.0 / 9.0), height],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return _unit_dirs(pts, subtract_mean=True)
    if n == 6:
        pts = np.array(
            [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]],
            dtype=np.float64,
        )
        return _unit_dirs(pts, subtract_mean=True)
    if n == 8:
        pts = np.array(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )
        return _unit_dirs(pts, subtract_mean=True)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    if n == 12:
        # Cyclic (0, ±1, ±φ). Vertices of a regular icosahedron.
        # 循环排列 (0, ±1, ±φ)，正二十面体的顶点。
        pts = np.array(
            [
                [0, 1, phi],
                [0, 1, -phi],
                [0, -1, phi],
                [0, -1, -phi],
                [1, phi, 0],
                [1, -phi, 0],
                [-1, phi, 0],
                [-1, -phi, 0],
                [phi, 0, 1],
                [phi, 0, -1],
                [-phi, 0, 1],
                [-phi, 0, -1],
            ],
            dtype=np.float64,
        )
        return _unit_dirs(pts, subtract_mean=True)
    if n == 20:
        pts = np.array(
            [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [1, -1, -1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, 1],
                [-1, -1, -1],
                [0, 1.0 / phi, phi],
                [0, 1.0 / phi, -phi],
                [0, -1.0 / phi, phi],
                [0, -1.0 / phi, -phi],
                [1.0 / phi, phi, 0],
                [1.0 / phi, -phi, 0],
                [-1.0 / phi, phi, 0],
                [-1.0 / phi, -phi, 0],
                [phi, 0, 1.0 / phi],
                [phi, 0, -1.0 / phi],
                [-phi, 0, 1.0 / phi],
                [-phi, 0, -1.0 / phi],
            ],
            dtype=np.float64,
        )
        return _unit_dirs(pts, subtract_mean=True)
    index = np.arange(n, dtype=np.float64)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (index / max(n - 1, 1)) * 2.0
    radius = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = golden * index
    pts = np.stack([np.cos(theta) * radius, y, np.sin(theta) * radius], axis=1)
    return _unit_dirs(pts, subtract_mean=True)


def make_view_rots(n_view, n_inplane):
    """
    # Rotate object +Z onto each view direction, then spin about camera +Z.

    ## Args

        - n_view: the direction count.
        - n_inplane: the number of spins. The angles are 2π k / n_inplane. Direction is the outer axis.

    ## Returns

        - Returns (n_view * n_inplane, 3, 3) float32.

    ---

    # 把物体 +Z 转到每个视线方向，再绕相机 +Z 做面内旋转。

    ## 参数

        - n_view: 视线个数。
        - n_inplane: 面内旋转次数。角度是 2π k / n_inplane。视线在外层。

    ## 返回

        - 返回 (n_view * n_inplane, 3, 3) float32。

"""
    align = _rots_from_z(make_view_dirs(n_view))
    count = int(n_inplane)
    angle = 2.0 * np.pi * np.arange(count, dtype=np.float64) / float(count)
    cosine = np.cos(angle).astype(np.float32)
    sine = np.sin(angle).astype(np.float32)
    spin = np.zeros((count, 3, 3), dtype=np.float32)
    spin[:, 0, 0] = cosine
    spin[:, 0, 1] = -sine
    spin[:, 1, 0] = sine
    spin[:, 1, 1] = cosine
    spin[:, 2, 2] = 1.0
    rots = np.matmul(spin[None], align[:, None])
    return np.ascontiguousarray(rots.reshape(-1, 3, 3))


def resolve_view_counts(n_view, n_inplane):
    """
    # Resolve the two view counts.

        n_view None reads recipe.n_view.

        n_inplane None reads recipe.n_inplane.

        The caller passes the count, not a list of angles.

        A count below 1 raises ValueError.

    ## Returns

        - Returns (n_view, n_inplane), both integers at least 1.

    ---

    # 解析两个视角数量。

        n_view 为 None 时读 recipe.n_view。

        n_inplane 为 None 时读 recipe.n_inplane。

        调用方传入的是次数，不是角度列表。

        小于 1 抛出 ValueError。

    ## 返回

        - 返回 (n_view, n_inplane)，都是不小于 1 的整数。

"""
    if n_view is None:
        n_view = recipe.n_view
    if n_inplane is None:
        n_inplane = recipe.n_inplane
    view_count = int(n_view)
    inplane_count = int(n_inplane)
    if view_count < 1 or inplane_count < 1:
        raise ValueError("n_view and n_inplane must be >= 1, got %s, %s" % (n_view, n_inplane))
    return view_count, inplane_count


def bbox_model_center(vertices):
    """
    # Axis-aligned bbox center, in the same unit as the vertices.

    ## Args

        - vertices: (V, 3) or wider. The center is the midpoint of the min and max corners.

    ## Returns

        - Returns (3,) float32.

    ---

    # 轴对齐包围盒中心，单位与顶点相同。

    ## 参数

        - vertices: (V, 3) 或更宽。中心是最小角和最大角的中点。

    ## 返回

        - 返回 (3,) float32。

"""
    v = np.asarray(vertices, dtype=np.float32)
    return (0.5 * (v.min(axis=0) + v.max(axis=0))).astype(np.float32)


def prepare_mesh(mesh):
    """
    # Center one mesh.

        The subtracted offset is stored in metadata["model_center"], in meters.

    ## Args

        - mesh: a trimesh.Trimesh or a path. A path is loaded with process False. A file that is not one mesh raises TypeError. A mesh that already stores model_center is not moved again. The extent notice uses the vertices as given. Centering does not change the diagonal.

    ## Returns

        - Returns the centered Trimesh.

    ---

    # 把网格移到中心。

        减去的偏移写入 metadata["model_center"]，单位是米。

    ## 参数

        - mesh: trimesh.Trimesh 或路径。路径按 process False 载入。不是单个网格时抛出 TypeError。已经有 model_center 的网格不再平移。尺度提示用的是读入时的顶点。居中不改变对角线。

    ## 返回

        - 返回居中后的 Trimesh。

"""
    import trimesh

    if isinstance(mesh, trimesh.Trimesh):
        loaded = mesh
        given_name = None
    else:
        given_name = os.path.splitext(os.path.basename(str(mesh)))[0]
        loaded = trimesh.load(mesh, force="mesh", process=False)
        # force="mesh" is still declared to also return a list, which has no metadata.
        # force="mesh" 的声明返回值里仍包含 list，list 没有 metadata。
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError("expected one mesh")
    centered = loaded.copy()
    md = dict(centered.metadata or {})
    # The notice uses the vertices as given. Centering does not change the diagonal.
    # 提示用的是读入时的顶点。居中不改变对角线。
    from wapr.mesh_extent import report_mesh_extent
    report_mesh_extent(centered.vertices, md.get("name") or given_name or "mesh")
    if "model_center" not in md:
        center = bbox_model_center(centered.vertices)
        centered.vertices = np.asarray(centered.vertices, dtype=np.float32) - center.reshape(1, 3)
        md["model_center"] = center
    centered.metadata = md
    return centered


def center_from_mesh(mesh):
    """
    # Read the mesh-center offset.

        The offset stored by prepare_mesh, in meters.

    ## Args

        - mesh: the centered mesh. Missing metadata means the origin, a zero vector.

    ## Returns

        - Returns (3,) float32.

    ---

    # 读取网格中心的偏移。

        prepare_mesh 存下的偏移，单位是米。

    ## 参数

        - mesh: 居中后的网格。没有这项 metadata 就当作原点，返回零向量。

    ## 返回

        - 返回 (3,) float32。

"""
    md = getattr(mesh, "metadata", None) or {}
    center = md.get("model_center", None)
    if center is None:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(center, dtype=np.float32).reshape(3)


def _right_multiply_center(poses, center, sign):
    """
    # Right-multiply poses by a translation of sign times center.

    ## Args

        - poses: (4, 4) or (N, 4, 4).
        - center: (3,) or (N, 3), meters.
        - sign: +1 toward the centered mesh and -1 back to the original mesh. A row count that matches neither 1 nor N raises ValueError.

    ## Returns

        - Returns (N, 4, 4).

    ---

    # 把位姿右乘 sign 乘 center 的平移。

    ## 参数

        - poses: (4, 4) 或 (N, 4, 4)。
        - center: (3,) 或 (N, 3)，米。
        - sign: +1 时走向居中网格，为 -1 时回到原始网格。行数既不是 1 也不是 N 时抛出 ValueError。

    ## 返回

        - 返回 (N, 4, 4)。

"""
    if poses.ndim == 2:
        poses = poses[None]
    center_t = torch.as_tensor(center, device=poses.device, dtype=poses.dtype)
    if int(center_t.numel()) == 3:
        shift = torch.eye(4, device=poses.device, dtype=poses.dtype)
        shift[:3, 3] = float(sign) * center_t.reshape(3)
        # Preserve FP32 pose geometry when old PyTorch enables TF32 matmul.
        # 旧 PyTorch 默认启用 TF32 矩阵乘时，位姿几何仍使用原有 FP32 精度。
        return (poses[:, :, :, None] * shift[None, None, :, :]).sum(dim=2)
    center_t = center_t.reshape(-1, 3)
    if int(center_t.shape[0]) == 1 and int(poses.shape[0]) > 1:
        center_t = center_t.expand(int(poses.shape[0]), 3)
    if int(center_t.shape[0]) != int(poses.shape[0]):
        raise ValueError("center rows %d != poses %d" % (int(center_t.shape[0]), int(poses.shape[0])))
    eye = torch.eye(4, device=poses.device, dtype=poses.dtype)
    shift = eye.reshape(1, 4, 4).expand(int(poses.shape[0]), 4, 4).contiguous().clone()
    shift[:, :3, 3] = float(sign) * center_t
    return (poses[:, :, :, None] * shift[:, None, :, :]).sum(dim=2)


def poses_original_to_centered(poses, center):
    """
    # Right-multiply so the pose refers to the centered mesh.

    ## Args

        - poses: (N, 4, 4) or one (4, 4), object-to-camera.
        - center: (3,) for one mesh, or (N, 3) when each pose has its own mesh.

    ## Returns

        - Returns (N, 4, 4), meters.

    ---

    # 右乘，使位姿对应居中后的网格。

    ## 参数

        - poses: (N, 4, 4) 或单个 (4, 4)，物体到相机。
        - center: (3,) 表示同一网格，或 (N, 3) 表示每个位姿各自的网格。

    ## 返回

        - 返回 (N, 4, 4)，米。

"""
    return _right_multiply_center(poses, center, 1.0)


def poses_centered_to_original(poses, center):
    """
    # Convert centered poses back to the original mesh.

        The output is still object-to-camera, meters, (N, 4, 4).

        poses and center use the same shapes as poses_original_to_centered.

    ---

    # 把居中后的位姿变回原始网格。

        输出仍是物体到相机，米，(N, 4, 4)。

        poses 和 center 的形状与 poses_original_to_centered 相同。

"""
    return _right_multiply_center(poses, center, -1.0)


def so3_exp(rotvec):
    """
    # Rotation vector in radians to a matrix that left-multiplies the current rotation.

    ## Args

        - rotvec: (N, 3). Near-zero angles stay identity.

    ## Returns

        - Returns (N, 3, 3).

    ---

    # 旋转向量，弧度，变成左乘当前旋转的矩阵。

    ## 参数

        - rotvec: (N, 3)。接近零的角度保持为单位阵。

    ## 返回

        - 返回 (N, 3, 3)。

"""
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(rotvec.shape[0], 3, 3).clone()
    theta = torch.linalg.norm(rotvec, dim=-1)
    near_zero = theta < 1e-8
    theta_c = theta.clamp(min=1e-8).unsqueeze(-1)
    k = rotvec / theta_c
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zeros = torch.zeros_like(kx)
    K = torch.stack(
        [zeros, -kz, ky, kz, zeros, -kx, -ky, kx, zeros],
        dim=-1,
    ).reshape(-1, 3, 3)
    th = theta_c.unsqueeze(-1)
    # Small geometric products must not inherit TF32 rounding from network matmul.
    # 小型几何矩阵乘积不应继承网络矩阵乘的 TF32 舍入；网络精度设置保持原样。
    K_squared = (K[:, :, :, None] * K[:, None, :, :]).sum(dim=2)
    R = eye + torch.sin(th) * K + (1.0 - torch.cos(th)) * K_squared
    # Preserve the zero-angle rule without synchronizing a CUDA Boolean on CPU.
    # 保留零角度规则，用 GPU 选择避免将 CUDA 布尔值同步到 CPU。
    R = torch.where(near_zero[:, None, None], eye, R)
    return R.transpose(1, 2)


def egocentric_delta_pose_to_pose(A_in_cam, trans_delta, rot_mat_delta):
    """
    # Add a camera-frame translation and left-multiply the rotation.

    ## Args

        - A_in_cam: (N, 4, 4), object-to-camera, meters.
        - trans_delta: (N, 3), meters.
        - rot_mat_delta: (N, 3, 3), dimensionless.

    ## Returns

        - Returns (N, 4, 4).

    ---

    # 在相机系加上平移，并左乘旋转。

    ## 参数

        - A_in_cam: (N, 4, 4)，物体到相机，米。
        - trans_delta: (N, 3)，米。
        - rot_mat_delta: (N, 3, 3)，无量纲。

    ## 返回

        - 返回 (N, 4, 4)。

"""
    trans = A_in_cam[:, :3, 3] + trans_delta
    current_rot = A_in_cam[:, :3, :3]
    rot = (rot_mat_delta[:, :, :, None] * current_rot[:, None, :, :]).sum(dim=2)
    RT_34 = torch.cat([rot, trans[..., None]], dim=2)
    RT_14 = torch.zeros_like(RT_34[:, 0:1, :])
    RT_14[:, 0, 3] = 1
    return torch.cat([RT_34, RT_14], dim=1)


def guess_translation(depth_m, mask, K):
    """
    # Back-project the mask bbox center at the mean valid depth.

    ## Args

        - depth_m: HxW meters.
        - mask: HxW. Foreground is value > 0, and valid depth must be finite and > 0.
        - K: 3×3, pixels. No valid pixel returns zeros.

    ## Returns

        - Returns xyz meters, OpenCV, shape (3,).

    ---

    # 用有效深度的均值，把 mask 包围盒中心反投影。

    ## 参数

        - depth_m: HxW，米。
        - mask: HxW。前景是大于 0 的值，有效深度必须有限且大于 0。
        - K: 3×3，像素。没有有效像素时返回零。

    ## 返回

        - 返回 xyz，米，OpenCV，形状 (3,)。

"""
    return guess_translations(depth_m, [mask], K)[0]


def guess_translations(depth_m, masks, K, device="cpu"):
    """
    # One depth image and N masks.

    ## Args

        - depth_m: HxW meters. Non-finite and non-positive values are invalid.
        - masks: a list of HxW masks.
        - K: 3×3, pixels.
        - device: where the box centers and the mean valid depths are reduced. The returned array is on CPU. A mask with no valid pixel is a zero row.

    ## Returns

        - Returns (N, 3) meters, OpenCV.

    ---

    # 一张深度、N 个 mask。

    ## 参数

        - depth_m: HxW，米。非有限值和非正值均无效。
        - masks: HxW mask 的列表。
        - K: 3×3，像素。
        - device: 计算包围盒中心和有效深度均值的设备。返回的数组在 CPU 上。没有有效像素的 mask 对应一行零。

    ## 返回

        - 返回 (N, 3)，米，OpenCV。

"""
    depth = torch.as_tensor(np.asarray(depth_m, dtype=np.float32), device=device)
    if depth.ndim == 3:
        depth = depth[..., 0]
    mask_t = torch.as_tensor(np.stack([np.asarray(item) for item in masks]), device=device)
    if mask_t.ndim == 4:
        mask_t = mask_t[..., 0]
    valid = (mask_t > 0) & torch.isfinite(depth) & (depth > 0)
    height, width = int(valid.shape[-2]), int(valid.shape[-1])
    xs = torch.arange(width, device=device)
    ys = torch.arange(height, device=device)
    x_plane = xs.view(1, 1, width).expand_as(valid)
    y_plane = ys.view(1, height, 1).expand_as(valid)
    # Pixel indices are int64; old torch.where requires the sentinel to match.
    # 像素索引为 int64；旧版 torch.where 要求哨兵值同为整数，数值与边界不变。
    far = width + height
    x_min = torch.where(valid, x_plane, far).amin(dim=(1, 2))
    x_max = torch.where(valid, x_plane, -far).amax(dim=(1, 2))
    y_min = torch.where(valid, y_plane, far).amin(dim=(1, 2))
    y_max = torch.where(valid, y_plane, -far).amax(dim=(1, 2))
    count = valid.sum(dim=(1, 2))
    # Mask invalid pixels before summation: NaN multiplied by False remains NaN.
    # 求和前将无效像素置零：NaN 乘以 False 仍会得到 NaN。
    depth_sum = torch.where(valid, depth.to(torch.float64), 0.0).sum(dim=(1, 2))
    z_mid = torch.where(count > 0, depth_sum / count.clamp(min=1), torch.zeros_like(depth_sum))
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    camera = torch.as_tensor(np.asarray(K, dtype=np.float32), device=device).reshape(3, 3)
    z32 = z_mid.to(torch.float32)
    x_c = (x_mid - camera[0, 2]) / camera[0, 0] * z32
    y_c = (y_mid - camera[1, 2]) / camera[1, 1] * z32
    out = torch.stack([x_c, y_c, z32], dim=1)
    out = torch.where(count[:, None] > 0, out, torch.zeros_like(out))
    return out.detach().float().cpu().numpy()


def mask_from_bbox(bbox_xywh, height, width):
    """
    # Fill a bbox as a mask.

    ## Args

        - bbox_xywh: x, y, w, h in pixels, origin at the top-left, y down.
        - height and width: the image size. The box is clipped to the image. An empty intersection stays all zeros.

    ## Returns

        - Returns uint8 HxW, with 1 inside the box.

    ---

    # 把包围盒填成 mask。

    ## 参数

        - bbox_xywh: x、y、w、h，像素，原点在左上，y 向下。
        - height 和 width: 图像尺寸。盒子会被裁到图像内。没有交集时全是 0。

    ## 返回

        - 返回 uint8 的 HxW，盒子里是 1。

"""
    x, y, w, h = [float(v) for v in bbox_xywh[:4]]
    x0 = max(int(np.floor(x)), 0)
    y0 = max(int(np.floor(y)), 0)
    x1 = min(int(np.ceil(x + w)), int(width))
    y1 = min(int(np.ceil(y + h)), int(height))
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 1
    return mask


def _rgb_u8(rgb):
    """
    # Convert RGB to float values.

        Stored as float32 in 0–255.

    ## Args

        - rgb: HxWx3, or a gray HxW which is repeated to three channels. uint8 is already 0–255. Any other numeric dtype whose maximum is at most 1.5 is treated as 0–1 and scaled. A larger value is left as an already-scaled 0–255 image.

    ## Returns

        - Returns HxWx3 float32.

    ---

    # 把 RGB 转成浮点。

        存成 0–255 的 float32。

    ## 参数

        - rgb: HxWx3。灰度 HxW 会复制成三个通道。uint8 已经是 0–255。其他数值类型的最大值不超过 1.5 时按 0–1 放大。更大的值当作已经是 0–255。

    ## 返回

        - 返回 HxWx3 float32。

"""
    # Float RGB in 0-1 is scaled to 0-255. The returned array stays float32.
    # 0-1 的浮点 RGB 放大到 0-255。返回值仍是 float32。
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    arr = arr[..., :3]
    if arr.dtype != np.uint8 and float(np.nanmax(arr)) <= 1.5:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    return np.asarray(arr, dtype=np.float32)


class WAPREstimator:
    """
    # 6D pose estimator.

        Load WAPR, SAPR, and WBPS. The two pose calls are different modes.

        estimate_one_category_one_instance is one category and one instance: one mesh, one mask or bbox, one pose.

        estimate_many_categories_many_instances is many categories and many instances. Each item already has its own mesh and mask. That call does not detect.

        estimate_frame_many_categories_many_instances in wapr/frame.py detects first. One obj_id restricts the search to that category; the call may return multiple instances.

        Construct it as WAPREstimator(device). device defaults to cuda:0.

        from wapr import WAPREstimator is the public entry.

    ---

    # 6D 位姿估计器。

        载入 WAPR、SAPR、WBPS。两次位姿调用不是同一种模式。

        estimate_one_category_one_instance 处理单类别、单实例：输入网格和 mask 或包围盒，输出一个位姿。

        estimate_many_categories_many_instances 是多类别、多实例。每一项已经自带网格和 mask。那次调用不检测。

        wapr/frame.py 的 estimate_frame_many_categories_many_instances 先检测。obj_ids 只放一个 id 时，仅搜索该类别，仍可返回多个实例。

        构造方式是 WAPREstimator(device)。device 默认 cuda:0。

        公开入口是 from wapr import WAPREstimator。
    """

    def __init__(self, device="cuda:0"):
        """
        # 6D pose estimator.

            WAPR with a mask, WAPR without a mask, SAPR, and WBPS.

        ## Args

            - device: a torch device string. The default is cuda:0. An installed wheel fetches missing pose weights and builds the FP16 engine set on the first visible GPU when all four engines are absent. A partial or incompatible engine set requires an explicit rebuild. A source checkout retains its explicit installation step. The trt backend never silently switches to PyTorch. The torch backend loads checkpoints without engines. The OGL renderer creates its context, shaders, and warmup tile here; each object mesh is uploaded on its first render.

        ## Returns

            - Returns None.

        ---

        # 6D 位姿估计器。

            带 mask 的 WAPR、不带 mask 的 WAPR、SAPR、WBPS。

        ## 参数

            - device: torch 设备字符串，默认 cuda:0。安装后的 wheel 在权重缺失时下载；四个 FP16 引擎均不存在时，于第一个可见 GPU 上构建整套引擎。部分缺失或不兼容的引擎需要显式重建。源码目录仍使用显式安装步骤。trt 后端不会静默切换到 PyTorch。torch 后端只载入权重。OGL 渲染器在此建立上下文、shader 并预热；每份物体网格在首次渲染时上传。

        ## 返回

            - 返回 None。

"""
        setup_started = time.perf_counter()
        self.device = device
        use_engine = str(recipe.backend) == "trt"
        if not source_checkout:
            # A wheel keeps large assets in the user cache. Only a missing pack or
            # engine starts setup; ordinary construction performs local file checks.
            # wheel 的大资源位于用户缓存。仅缺文件时下载或构建；正常启动只检查本地文件。
            from wapr.download_assets import check_and_fetch_pack

            check_and_fetch_pack("wapr_sapr_wbps")
            if use_engine:
                engine_names = ("wapr_w_mask", "wapr_wo_mask", "sapr", "wbps")
                missing_engines = [name for name in engine_names if not os.path.isfile(recipe.engine_file(name))]
                if missing_engines:
                    if len(missing_engines) != len(engine_names):
                        raise FileNotFoundError(
                            "Only some pose engines are present. Run python -m wapr.export_engines "
                            "to review and rebuild the full set. / 位姿引擎仅部分存在；请显式重建并核对整套引擎。"
                        )
                    if torch.device(device).index not in (None, 0):
                        raise RuntimeError(
                            "Automatic engine build uses the first visible CUDA device. "
                            "Select the target GPU with CUDA_VISIBLE_DEVICES and construct WAPREstimator('cuda:0'). / "
                            "自动构建使用第一个可见 GPU；请先用 CUDA_VISIBLE_DEVICES 指定目标显卡，再使用 cuda:0。"
                        )
                    from wapr.export_engines import main as build_pose_engines

                    print("WAPR_ENGINE_BUILD", {"missing": missing_engines, "device": device}, flush=True)
                    build_pose_engines()
        self.nets = {}
        for name in ("wapr_w_mask", "wapr_wo_mask", "sapr", "wbps"):
            engine = recipe.engine_file(name) if use_engine else None
            self.nets[name] = load_net(name, recipe.weight_file(name), engine=engine, device=device)
            loaded_engine = self.nets[name].engine
            if loaded_engine is not None:
                expected = (min(wbps_group_sizes), int(recipe.n_view * recipe.n_inplane), max(wbps_group_sizes))
                actual = (loaded_engine.min_batch, loaded_engine.opt_batch, loaded_engine.max_batch)
                if actual != expected:
                    raise RuntimeError(
                        "%s engine profile %s is obsolete; expected %s. Rebuild locally with "
                        "python -m wapr.export_engines" % (name, actual, expected)
                    )
        loaded = {name: self.nets[name].engine is not None for name in self.nets}
        batch_score_path = os.path.join(os.path.dirname(os.fspath(recipe.engine_file("wbps"))), "wbps_batch.engine")
        print("WAPR_POSE", {"backend": str(recipe.backend), "fp16_engine": loaded,
                            "batch_score_engine_present": use_engine and os.path.isfile(batch_score_path)}, flush=True)
        # Context, shaders, and one warmup tile. Object meshes arrive later, with the first render of each mesh.
        # 上下文、着色器和一次预热绘制。物体网格要到每份网格第一次渲染时才上传。
        self.renderer = runtime_for(device)
        self._prepared_meshes = weakref.WeakValueDictionary()
        torch.cuda.synchronize(self.device)
        self.model_setup_seconds = time.perf_counter() - setup_started
        self.mesh_prepare_count = 0
        self.last_mesh_setup_seconds = 0.0

    def prepare_meshes(self, meshes):
        """Center and upload distinct meshes before warmup or timing.

        在预热和计时之前，将不同网格居中并上传。返回值与输入逐项对应。
        Reuse the returned snapshots; prepare again after editing a source mesh.
        后续复用返回的网格快照；修改源网格后重新准备，不缓存可变源网格。
        Treat returned geometry as immutable; edit the source, not the snapshot.
        返回快照的几何不可原地修改；需要变形时编辑源网格，再创建快照。
        """
        torch.cuda.synchronize(self.device)
        setup_started = time.perf_counter()
        unique = {}
        prepared = []
        for mesh in meshes:
            key = id(mesh) if not isinstance(mesh, (str, os.PathLike)) else os.fspath(mesh)
            if key not in unique:
                existing = self._prepared_meshes.get(id(mesh))
                unique[key] = mesh if existing is mesh else prepare_mesh(mesh)
                self.mesh_prepare_count += int(existing is not mesh)
            prepared.append(unique[key])
        self.renderer.ensure_mesh_slots(len(self._prepared_meshes) + len(unique))
        for mesh in unique.values():
            self.renderer.load_mesh_trimesh(mesh, name="mesh")
            self._prepared_meshes[id(mesh)] = mesh
        torch.cuda.synchronize(self.device)
        self.last_mesh_setup_seconds = time.perf_counter() - setup_started
        return prepared

    def _pose_meshes(self, meshes):
        """Accept prepared snapshots; legacy raw inputs retain automatic setup.

        复用准备好的网格快照；旧调用传原网格时仍自动准备，不能当作预热后的计时。
        """
        if all(self._prepared_meshes.get(id(mesh)) is mesh for mesh in meshes):
            self.last_mesh_setup_seconds = 0.0
            return list(meshes)
        return self.prepare_meshes(meshes)

    def warmup_pose(self, rgb, depth_m, K, instances, n_view=None, n_inplane=None, calls=3):
        """Warm the actual complete pose workload; discard outputs before hot timing.

        在预热后的计时前预热实际完整位姿负载，丢弃输出；instances 须使用准备好的网格。
        Returns elapsed warmup seconds separately. No scientific recipe is changed.
        单独返回预热秒数，不修改候选数、修正次数或其他实验配置。
        """
        if int(calls) < 1:
            raise ValueError("warmup calls must be positive")
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for _ in range(int(calls)):
            self.estimate_many_categories_many_instances(
                rgb, depth_m, K, instances, n_view=n_view, n_inplane=n_inplane,
            )
        torch.cuda.synchronize(self.device)
        return time.perf_counter() - started

    def estimate_one_category_one_instance(self, rgb, depth_m, K, mesh, diameter_m, bbox_xywh=None, mask=None, n_view=None, n_inplane=None):
        """
        # 6D pose of one category and one instance.

            One mesh, one mask or bbox, one pose. It does not look for other instances or other categories.

            A mask selects wapr_w_mask.

            No mask selects wapr_wo_mask and fills the bbox.

            The hypothesis count must belong to the shared supported set.

            Non-positive depth at the mask center raises RuntimeError.

        ## Args

            - rgb: (H, W, 3) RGB. uint8 is 0–255. Any other numeric dtype is 0–1 when its max is at most 1.5, otherwise already 0–255. The image is stored as float32. (3, H, W) is not this call.
            - depth_m: (H, W) meters, cast to float32, with no unit change. A uint16 millimeter image stays unscaled.
            - K: (3, 3) pixels, OpenCV. Nine row-major numbers also reshape. Any other count raises ValueError.
            - mesh: a trimesh.Trimesh or a path. Vertices are meters.
            - diameter_m: one number in meters.
            - bbox_xywh: x, y, w, h in pixels, y down, and is used only when mask is None.
            - mask: (H, W). Foreground is value > 0. A missing mask and a missing bbox raises ValueError.
            - n_view and n_inplane: integers of at least 1. None uses recipe.n_view and recipe.n_inplane. Those defaults are 4 and 3. 4, 6, 8, 12, and 20 are polyhedron vertices. Any other n_view is Fibonacci.
            - n_inplane: the count of spins about camera +Z.

        ## Returns

            - Returns a dict.
            - pose_4x4: float32 (4, 4).
            - R: float32 (3, 3).
            - t_m: float32 (3,) meters.
            - score_6d: a Python float, (100 - max between_group) / 200. A larger between_group lowers it.

        ---

        # 单类别、单实例的 6D 位姿。

            输入一个网格和一块 mask 或包围盒，输出一个位姿。不搜索其他实例或类别。

            有 mask 用 wapr_w_mask。

            没有 mask 用 wapr_wo_mask，并用包围盒填区域。

            候选数量必须属于共享的允许集合。

            mask 中心的深度不是正数时抛出 RuntimeError。

        ## 参数

            - rgb: (H, W, 3)，RGB。uint8 是 0–255。其他数值类型的最大值不超过 1.5 时是 0–1，否则已经是 0–255。存成 float32。(3, H, W) 不是这次调用。
            - depth_m: (H, W)，米，转成 float32，不换单位。uint16 的毫米图不会被缩放。
            - K: (3, 3)，像素，OpenCV。按行排开的 9 个数也可以 reshape。其他个数抛出 ValueError。
            - mesh: trimesh.Trimesh 或路径。顶点是米。
            - diameter_m: 一个数，米。
            - bbox_xywh: x、y、w、h，像素，y 向下，只在 mask 为 None 时使用。
            - mask: (H, W)。前景是大于 0 的值。mask 和 bbox 都没有时抛出 ValueError。
            - n_view 和 n_inplane: 不小于 1 的整数。None 用 recipe.n_view 和 recipe.n_inplane。默认是 4 和 3。4、6、8、12、20 是多面体顶点。其他 n_view 是斐波那契。
            - n_inplane: 绕相机 +Z 的次数。

        ## 返回

            - 返回一个 dict。
            - pose_4x4: float32 (4, 4)。
            - R: float32 (3, 3)。
            - t_m: float32 (3,)，米。
            - score_6d: Python float，即 (100 - between_group 的最大值) / 200。between_group 越大，这个值越小。

"""
        # One object is one independent group in the same GPU batch implementation.
        # 单物体在相同 GPU 批量计算实现中占一个独立评分组，不再维护第二套估计路径。
        if mask is None and bbox_xywh is None:
            raise ValueError("bbox_xywh or mask")
        instances = [{"mesh": mesh, "diameter_m": diameter_m,
                      "bbox_xywh": bbox_xywh, "mask": mask}]
        return self.estimate_many_categories_many_instances(
            rgb, depth_m, K, instances, n_view=n_view, n_inplane=n_inplane,
        )[0]

    def estimate_many_categories_many_instances(self, rgb, depth_m, K, instances, n_view=None, n_inplane=None):
        """
        # 6D pose of many categories and many instances.

            Regions already known. WAPR, SAPR, rendering and WBPS batch across instances on both backends; WBPS attention stays independent per object. This call does not detect.

            rgb, depth_m, and K are the same frame for every instance, with the same forms as estimate_one_category_one_instance.

            instances is a list of dicts.

            Each dict has mesh, diameter_m in meters, and mask or bbox_xywh. A missing mask selects unmasked WAPR; a supplied mask selects masked WAPR.

            An empty list returns an empty list.

            A mask of a different size raises ValueError.

            The mask selects wapr_w_mask.

            Hypotheses of one instance stay one WBPS group.

            n_view and n_inplane follow estimate_one_category_one_instance.

            None uses recipe.py.

            A non-positive translation depth raises RuntimeError.

            Each result has pose_4x4, R, t_m, and score_6d, in the same order as instances.

            The shapes match estimate_one_category_one_instance.

        ## Returns

            - Returns a list of dicts.

        ---

        # 多类别、多实例的 6D 位姿。

            区域必须已经有。两种后端都对 WAPR、SAPR、渲染和 WBPS 批量计算；WBPS 注意力保持每个物体独立。这次调用不检测。

            rgb、depth_m、K 是所有实例共用的这一帧，形式与 estimate_one_category_one_instance 相同。

            instances 是字典列表。

            每个字典含 mesh、米制 diameter_m，以及 mask 或 bbox_xywh。未提供 mask 时选择无掩码 WAPR；提供 mask 时选择有掩码 WAPR。

            空列表返回空列表。

            mask 尺寸不同则抛出 ValueError。

            mask 选择 wapr_w_mask。

            一个实例的候选姿态仍是一组 WBPS。

            n_view 和 n_inplane 与 estimate_one_category_one_instance 相同。

            None 用 recipe.py。

            平移的深度不是正数时抛出 RuntimeError。

            每项含 pose_4x4、R、t_m、score_6d，顺序与 instances 相同。

            形状与 estimate_one_category_one_instance 一致。

        ## 返回

            - 返回字典列表。

"""
        if len(instances) == 0:
            return []
        rgb = _rgb_u8(rgb)
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        K = np.asarray(K, dtype=np.float32).reshape(3, 3)
        height, width = int(depth.shape[0]), int(depth.shape[1])
        view_count, inplane_count = resolve_view_counts(n_view, n_inplane)
        self._require_engine_group(view_count, inplane_count)
        rots = make_view_rots(view_count, inplane_count)
        group = int(rots.shape[0])
        refine_engine = self.nets["wapr_w_mask"].engine
        if refine_engine is None:
            row_capacity = max(wbps_group_sizes)
        else:
            row_capacity = min(int(self.nets[name].engine.max_batch)
                               for name in ("wapr_w_mask", "wapr_wo_mask", "sapr", "wbps"))
        objects_per_batch = max(1, row_capacity // group)
        # Explicitly prepared inputs bypass mesh copying and GL loading altogether.
        # 显式准备过的输入完全跳过网格复制及 GL 加载；旧接口仍可自动准备。
        prepared_meshes = self._pose_meshes([item["mesh"] for item in instances])
        prepared_ids = [self.renderer.cached_mesh_id(mesh) for mesh in prepared_meshes]
        # The same frame is uploaded once; chunks preserve complete object groups.
        # 同一帧只上传一次；容量分块保持每个物体的完整候选组。
        rgb_pose = torch.as_tensor(rgb, device=self.device, dtype=torch.float32)
        depth_pose = torch.as_tensor(depth, device=self.device, dtype=torch.float32)
        camera_pose = torch.as_tensor(K, device=self.device, dtype=torch.float32)[None]
        observed_xyz = depth2xyzmap_batch(depth_pose[None], camera_pose).permute(0, 3, 1, 2)
        results = []
        for batch_start in range(0, len(instances), objects_per_batch):
            batch = instances[batch_start:batch_start + objects_per_batch]
            masks = []
            meshes = []
            centers = []
            diameters = []
            masked = []
            for index, item in enumerate(batch):
                use_mask = item.get("mask") is not None
                mask = (np.asarray(item["mask"]) if use_mask else
                        mask_from_bbox(item["bbox_xywh"], height, width))
                if mask.ndim == 3:
                    mask = mask[..., 0]
                if tuple(mask.shape[:2]) != (height, width):
                    raise ValueError("mask size differs from depth")
                mesh = prepared_meshes[batch_start + index]
                masks.append(mask)
                meshes.append(mesh)
                centers.append(center_from_mesh(mesh))
                diameters.append(float(item["diameter_m"]))
                masked.append(use_mask)
            translations = guess_translations(depth, masks, K, device=self.device)
            if np.any(translations[:, 2] <= 0.0):
                raise RuntimeError("depth")
            count = len(batch)
            # Rows of one object stay adjacent: (objects, hypotheses, 4, 4).
            # 同一物体的候选姿态连续排列：(物体数, 候选数量, 4, 4)。
            poses_np = np.zeros((count, group, 4, 4), dtype=np.float32)
            poses_np[:, :, 3, 3] = 1
            poses_np[:, :, :3, :3] = rots[None]
            poses_np[:, :, :3, 3] = translations[:, None, :]
            poses = torch.as_tensor(poses_np.reshape(count * group, 4, 4), device=self.device)
            mask_batch = np.stack(masks, axis=0)
            mask_pose = torch.as_tensor(mask_batch, device=self.device, dtype=torch.float32)
            mask_pose = mask_pose.repeat_interleave(group, dim=0)
            mesh_rows = [mesh for mesh in meshes for _ in range(group)]
            mesh_ids = [mid for mid in prepared_ids[batch_start:batch_start + count] for _ in range(group)]
            center_rows = np.repeat(np.stack(centers, axis=0).astype(np.float32), group, axis=0)
            diameter_rows = np.repeat(np.asarray(diameters, dtype=np.float32), group)
            # Two checkpoint batches when masks and boxes coexist, never one call per box.
            # 掩码与框混合时按所用的两组权重分别批量处理，不按检测框逐个调用。
            masked_rows = np.repeat(np.asarray(masked, dtype=bool), group)
            for use_mask, net_name in ((True, "wapr_w_mask"), (False, "wapr_wo_mask")):
                row_indices = np.flatnonzero(masked_rows == use_mask)
                if not len(row_indices):
                    continue
                indices = torch.as_tensor(row_indices, device=self.device)
                selected_meshes = [mesh_rows[index] for index in row_indices]
                refined = self._refine(
                    self.nets[net_name], poses[indices], rgb_pose, depth_pose, mask_pose[indices],
                    K, selected_meshes[0], center_rows[row_indices], diameter_rows[row_indices],
                    iters=recipe.wapr_iters, max_rot=recipe.wapr_max_rot_rad,
                    crop_ratio=recipe.refine_crop_ratio, use_mask=use_mask, meshes=selected_meshes,
                    group_size=group,
                    observed_xyz=observed_xyz, mesh_ids=[mesh_ids[index] for index in row_indices],
                )
                poses[indices] = refined
            poses = self._refine(
                self.nets["sapr"], poses, rgb_pose, depth_pose, mask_pose, K, mesh_rows[0],
                center_rows, diameter_rows, iters=recipe.sapr_iters,
                max_rot=recipe.sapr_max_rot_rad, crop_ratio=recipe.refine_crop_ratio,
                use_mask=False, meshes=mesh_rows, group_size=group,
                observed_xyz=observed_xyz, mesh_ids=mesh_ids,
            )
            # Build all WBPS crops in one GPU batch. The observed RGB-D frame
            # and its xyz map are shared across the objects in this chunk.
            # WBPS 裁剪在 GPU 上批量计算；这批物体共用同一帧 RGB-D 及其 xyz 图。
            centered_poses = poses_original_to_centered(poses, center_rows)
            score_A, score_B, _score_poses, _score_diameters = make_crop_pair(
                rgb_pose, depth_pose, mask_pose, centered_poses, K, mesh_rows[0], diameter_rows,
                crop_ratio=recipe.wbps_crop_ratio, use_mask=False, device=self.device,
                meshes=mesh_rows,
                observed_xyz=observed_xyz, mesh_ids=mesh_ids,
            )
            with torch.no_grad():
                score_output = self.nets["wbps"](score_A, score_B, L=group)
            within_group = score_output["within_group"].reshape(count, group)
            between_group = score_output["between_group"].reshape(count, group)
            best = within_group.argmax(dim=1)
            selected = poses.reshape(count, group, 4, 4)[torch.arange(count, device=self.device), best]
            selected = selected.detach().float().cpu().numpy()
            scores = ((100.0 - between_group.max(dim=1).values) / 200.0).detach().float().cpu().numpy()
            # CPU record assembly only; all rendering and inference have finished.
            # 此处只整理 CPU 记录；所有渲染与网络推理已完成。
            for pose, score_6d in zip(selected, scores):
                results.append({
                    "pose_4x4": pose,
                    "score_6d": float(score_6d),
                    "R": pose[:3, :3].copy(),
                    "t_m": pose[:3, 3].copy(),
                })
        return results

    def _require_engine_group(self, view_count, inplane_count):
        """
        # Reject a pose count outside the shared one-object group set.

        ## Args

            - view_count and inplane_count: their product must belong to wbps_group_sizes for either backend. An unsupported count raises ValueError.

        ## Returns

            - Returns None.

        ---

        # 拒绝不在共享单物体组长度集合内的位姿数。

        ## 参数

            - view_count 和 inplane_count: 两者的乘积在两种后端都必须属于 wbps_group_sizes，否则抛出 ValueError。

        ## 返回

            - 返回 None。

"""
        asked = int(view_count) * int(inplane_count)
        if asked not in wbps_group_sizes:
            raise ValueError(
                "n_view * n_inplane is %d; one-object pose count must belong to %s"
                % (asked, sorted(wbps_group_sizes))
            )

    def _refine(self, net, poses, rgb, depth, mask, K, mesh, center, diameter, iters, max_rot, crop_ratio, use_mask, meshes=None, group_size=None, observed_xyz=None, mesh_ids=None):
        """
        # Run one refinement.

            One refine net runs for iters steps. Poses come back in the original mesh frame, shaped (N, 4, 4).

        ## Args

            - net: wapr_w_mask, wapr_wo_mask, or sapr.
            - poses: object-to-camera, meters. rgb, depth, mask, and K are the frame.
            - mesh: one mesh, or meshes overrides it with one mesh per row.
            - center: (3,) or (N, 3), meters.
            - diameter: meters, one value or one per row.
            - iters: the update count.
            - max_rot: the per-step rotation cap in radians, applied after tanh.
            - crop_ratio: the crop window as a multiple of the diameter. use_mask True adds the mask channel. The work is done on the centered mesh, then converted back.

        ---

        # 做一次修正。

            一个修正网络运行 iters 步。位姿回到原始网格坐标系，形状是 (N, 4, 4)。

        ## 参数

            - net: wapr_w_mask、wapr_wo_mask 或 sapr。
            - poses: 物体到相机，米。
            - rgb、depth、mask、K: 这一帧。
            - mesh: 一个网格。meshes 给出时，每一行一个网格。
            - center: (3,) 或 (N, 3)，米。
            - diameter: 米，一个数或每行一个。
            - iters: 更新次数。
            - max_rot: 每步旋转上限，弧度，加在 tanh 之后。
            - crop_ratio: 裁剪窗口相对直径的倍数。
            - use_mask: True 时加上 mask 通道。更新在居中网格上做，再变回原始网格。

"""
        poses_c = poses_original_to_centered(poses, center)
        for _ in range(int(iters)):
            A, B, poseA, dia = make_crop_pair(
                rgb, depth, mask, poses_c, K, mesh, diameter,
                crop_ratio=crop_ratio, use_mask=use_mask, device=self.device,
                meshes=meshes,
                observed_xyz=observed_xyz, mesh_ids=mesh_ids,
            )
            with torch.no_grad():
                pred = net(A, B, L=group_size)
            # rot is tanh-limited radians. trans is scaled by half the diameter, meters.
            # rot 是 tanh 限幅后的弧度。trans 乘直径的一半，单位米。
            rot_vec = torch.tanh(pred["rot"]) * float(max_rot)
            rot_mat = so3_exp(rot_vec)
            trans = pred["trans"] * (dia.reshape(-1, 1) / 2.0)
            poses_c = egocentric_delta_pose_to_pose(poseA, trans, rot_mat)
        return poses_centered_to_original(poses_c, center)

    def _wbps(self, poses, rgb, depth, mask, K, mesh, center, diameter, L, meshes=None, observed_xyz=None, mesh_ids=None):
        """
        # Score the hypotheses.

            poses, rgb, depth, mask, K, mesh, center, diameter, and meshes match _refine.

            L is the number of hypotheses for one object and must belong to wbps_group_sizes.

            Each L adjacent rows form one object's group; outputs are (N / L, L).

            WBPS does not use the mask channel.

            within_group picks the hypothesis.

            A larger between_group is worse.

        ## Returns

            - Returns within_group and between_group.

        ---

        # 给这些候选姿态打分。

            poses、rgb、depth、mask、K、mesh、center、diameter、meshes 与 _refine 相同。

            L 是一个物体的候选数量，必须属于 wbps_group_sizes。

            每 L 个相邻行组成一个物体的候选组，输出形状是 (N / L, L)。

            WBPS 不使用 mask 通道。

            within_group 用来选候选姿态。

            between_group 越大越差。

        ## 返回

            - 返回 within_group 和 between_group。

"""
        group_size = int(L)
        if group_size not in wbps_group_sizes or int(poses.shape[0]) % group_size:
            raise ValueError(
                "WBPS requires complete supported pose groups; got %d poses with L=%d"
                % (int(poses.shape[0]), group_size)
            )
        poses_c = poses_original_to_centered(poses, center)
        A, B, _poseA, _dia = make_crop_pair(
            rgb, depth, mask, poses_c, K, mesh, diameter,
            crop_ratio=recipe.wbps_crop_ratio, use_mask=False, device=self.device,
            meshes=meshes,
            observed_xyz=observed_xyz, mesh_ids=mesh_ids,
        )
        with torch.no_grad():
            pred = self.nets["wbps"](A, B, L=group_size)
        return pred["within_group"], pred["between_group"]
