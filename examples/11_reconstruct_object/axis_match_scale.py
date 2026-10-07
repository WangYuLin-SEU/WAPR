# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""RoMa render-to-photo axis fitting at a fixed object rotation.

固定物体旋转，用 RoMa 渲染到照片的对应拟合三轴缩放。
Published computational helpers from the local axis-scale experiment.
公开本地轴缩放实验的计算函数；不包含参考 CAD 或保存位姿的加载入口。
"""

import cv2
import numpy as np

# Preserve the matching crop recipe. / 保留匹配裁剪配置。
TILE = 448


def square_box(mask, pad):
    """Square crop in this image's pixel coordinates. / 当前图像的像素正方形裁剪。"""
    height, width = mask.shape
    ys, xs = np.nonzero(mask > 0)
    if not len(xs):
        raise ValueError("Cannot crop an empty mask / 空掩码无法裁剪")
    side = min(int(max(xs.max() - xs.min(), ys.max() - ys.min()) + 2 * pad), height, width)
    cx = int(round(0.5 * (xs.min() + xs.max())))
    cy = int(round(0.5 * (ys.min() + ys.max())))
    left = max(0, min(width - side, cx - side // 2))
    top = max(0, min(height - side, cy - side // 2))
    return (left, top, left + side, top + side)


def extents_mm(mesh):
    v = np.asarray(mesh.vertices, dtype=np.float64)
    return (v.max(axis=0) - v.min(axis=0)) * 1000.0


def scale_axes(mesh, scales):
    """Scale the original mesh about its own box center. The pose is not changed.

    绕原始网格自己的盒中心缩放。位姿不动。
    """
    import trimesh

    scaled = mesh.copy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    vertices = center + (vertices - center) * np.asarray(scales, dtype=np.float64).reshape(1, 3)
    scaled.vertices = vertices
    return scaled


def render_box(runtime, mesh, pose, K, box, image_shape):
    """Render this input's crop using its original frame size.

    使用本次输入的原图尺寸渲染裁剪。
    """
    height, width = image_shape[:2]
    left, top, right, bottom = box
    mesh_id = runtime.load_mesh_trimesh(mesh, name="axis-match")
    bbox = np.array([[left, top, right, bottom]], dtype=np.float32)
    pose_batch = np.asarray(pose, dtype=np.float32).reshape(1, 4, 4)
    rgb, depth = runtime.render_tiles(
        mesh_id, pose_batch, bbox, K, height, width, TILE,
    )
    color = rgb[0].detach().float().cpu().numpy()
    z = depth[0].detach().float().cpu().numpy()
    if float(np.nanmax(color)) > 1.5:
        color = color / 255.0
    color = np.clip(color * 255.0, 0.0, 255.0).astype(np.uint8)
    hit = z > 1.0e-4
    return color, z, hit


def gray_render(color, hit):
    """Object pixels keep the texture. Empty pixels are dark gray, not the photo.

    物体像素保留贴图。空像素是深灰，不拿照片去填。
    """
    out = np.full_like(color, 40)
    out[hit] = color[hit]
    return out


def match_points(model, src_rgb, dst_rgb, depth, hit, box, pose, K, min_cert=0.25):
    """Object points on the render, and the photo pixels RoMa pairs them with.

    渲染上的物体点，以及 RoMa 配到的照片像素。
    """
    from PIL import Image

    if model is None:
        # Preserve the original matching resolution and sample count across the worker.
        # 独立进程保持原匹配分辨率和采样数；下方几何过滤保持不变。
        from wapr.roma_isolated import match_pixels
        src_xy, dst_xy, weight = match_pixels(src_rgb, dst_rgb, device="cuda",
                                              coarse_res=560, upsample_res=560, num=6000)
    else:
        warp, certainty = model.match(Image.fromarray(src_rgb), Image.fromarray(dst_rgb), device="cuda")
        matches, cert = model.sample(warp, certainty, num=6000)
        kpts_src, kpts_dst = model.to_pixel_coordinates(matches, TILE, TILE, TILE, TILE)
        src_xy = kpts_src.detach().float().cpu().numpy()
        dst_xy = kpts_dst.detach().float().cpu().numpy()
        weight = cert.detach().float().cpu().numpy()
    left, top, right, bottom = box
    side = float(right - left)
    R = pose[:3, :3]
    t = pose[:3, 3]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    obj = []
    target = []
    kept_w = []
    for i in range(len(src_xy)):
        if weight[i] < min_cert:
            continue
        x, y = float(src_xy[i, 0]), float(src_xy[i, 1])
        xi, yi = int(round(x)), int(round(y))
        if xi < 2 or yi < 2 or xi >= TILE - 2 or yi >= TILE - 2:
            continue
        if not hit[yi, xi]:
            continue
        flow = float(np.linalg.norm(dst_xy[i] - src_xy[i]))
        if flow > 48.0:
            continue
        z = float(depth[yi, xi])
        if z < 1.0e-3:
            continue
        u = left + (x + 0.5) / TILE * side
        v = top + (y + 0.5) / TILE * side
        cam = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)
        point = R.T @ (cam - t)
        xd, yd = float(dst_xy[i, 0]), float(dst_xy[i, 1])
        tu = left + (xd + 0.5) / TILE * side
        tv = top + (yd + 0.5) / TILE * side
        obj.append(point)
        target.append((tu, tv))
        kept_w.append(float(weight[i]))
    if len(obj) < 40:
        return None
    return (
        np.asarray(obj, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        np.asarray(kept_w, dtype=np.float64),
    )


def project(cam, K):
    z = cam[:, 2]
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def pack_jacobian(cam, q, R, K):
    """Pixel rows against three axis scales and a camera translation. q is the point minus the box center.

    像素对三个轴缩放和相机平移的导数。q 是点相对盒中心的坐标。
    """
    n = cam.shape[0]
    X, Y, Z = cam[:, 0], cam[:, 1], cam[:, 2]
    inv_z = 1.0 / Z
    fx, fy = float(K[0, 0]), float(K[1, 1])
    du = np.stack([fx * inv_z, np.zeros(n), -fx * X * inv_z * inv_z], axis=1)
    dv = np.stack([np.zeros(n), fy * inv_z, -fy * Y * inv_z * inv_z], axis=1)
    jac = np.zeros((n, 2, 6), dtype=np.float64)
    for axis in range(3):
        dcam = q[:, axis:axis + 1] * R[:, axis].reshape(1, 3)
        jac[:, 0, axis] = np.sum(du * dcam, axis=1)
        jac[:, 1, axis] = np.sum(dv * dcam, axis=1)
    jac[:, 0, 3:6] = du
    jac[:, 1, 3:6] = dv
    return jac


def solve_scales(points, target, pose, K):
    """Scales about the box center, plus a small camera shift. Rotation is fixed.

    绕盒中心的三轴缩放，再加一个小的相机平移。旋转固定。
    """
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    # The box center of the visible points is not the mesh center. Use the
    # centroid of the matched points only as a fallback when the mesh center
    # is passed in by overwriting `center` below.
    # 可见点的盒心不是网格中心。网格中心由调用方覆盖 center。
    return _solve_about(points, target, pose, K, center)


def solve_about(points, target, pose, K, center):
    """Fit about the mesh center, in meters. / 绕米制网格中心拟合。"""
    return _solve_about(points, target, pose, K, np.asarray(center, dtype=np.float64))


def _solve_about(points, target, pose, K, center):
    """Robust fit with the original scale and shift priors.

    保留原缩放和平移先验的稳健拟合。
    """
    R = pose[:3, :3]
    t = pose[:3, 3]
    q = points - center.reshape(1, 3)
    scales = np.ones(3, dtype=np.float64)
    delta_t = np.zeros(3, dtype=np.float64)
    lam = 1.0e-2
    rms = None
    n_in = int(len(points))
    for _step in range(8):
        placed = center.reshape(1, 3) + q * scales.reshape(1, 3)
        cam = placed @ R.T + (t + delta_t).reshape(1, 3)
        pred = project(cam, K)
        err = pred - target
        resid = np.linalg.norm(err, axis=1)
        med = float(np.median(resid))
        keep = resid < max(4.0, 3.0 * med)
        if int(keep.sum()) < 40:
            keep = np.ones(len(resid), dtype=bool)
        err_k = err[keep]
        jac = pack_jacobian(cam[keep], q[keep], R, K)
        r = err_k.reshape(-1)
        jf = jac.reshape(-1, 6)
        huber = np.minimum(1.0, 2.0 / np.maximum(np.abs(r), 1.0e-6))
        sw = np.sqrt(huber)
        weighted = jf * sw[:, None]
        normal = weighted.T @ weighted
        grad = weighted.T @ (r * sw)
        # Weak pull toward the current scale and zero shift, in pixels-per-parameter units.
        # 很弱地拉回当前缩放和零平移，单位按像素对参数。
        prior = np.concatenate([scales - 1.0, delta_t])
        sigma = np.array([0.35, 0.35, 0.35, 0.02, 0.02, 0.03], dtype=np.float64)
        normal = normal + np.diag(1.0 / sigma ** 2)
        grad = grad + prior / sigma ** 2
        damp = lam * np.diag(np.diag(normal) + 1.0e-6)
        try:
            step = np.linalg.solve(normal + damp, grad)
        except np.linalg.LinAlgError:
            break
        scales = np.clip(scales - step[:3], 0.65, 1.45)
        delta_t = delta_t - step[3:]
        rms = float(np.sqrt(np.mean(np.sum(err_k * err_k, axis=1))))
        n_in = int(keep.sum())
        if float(np.max(np.abs(step[:3]))) < 2.0e-3 and float(np.max(np.abs(step[3:]))) < 5.0e-4:
            break
    info = axis_strength(q, R, K, points, center, scales, delta_t, pose)
    return {
        "scales": [float(v) for v in scales],
        "delta_t_mm": [float(v * 1000.0) for v in delta_t],
        "rms_px": None if rms is None else float(rms),
        "n": n_in,
        "strength": info,
    }


def axis_strength(q, R, K, points, center, scales, delta_t, pose):
    """RMS image motion, in pixels, of a 10 percent stretch on each axis.

    每个轴拉长 10% 时，图像上的均方根位移，单位像素。
    """
    placed = center.reshape(1, 3) + q * scales.reshape(1, 3)
    cam = placed @ pose[:3, :3].T + (pose[:3, 3] + delta_t).reshape(1, 3)
    base = project(cam, K)
    out = []
    for axis in range(3):
        stretched = scales.copy()
        stretched[axis] *= 1.10
        moved = center.reshape(1, 3) + q * stretched.reshape(1, 3)
        cam_s = moved @ pose[:3, :3].T + (pose[:3, 3] + delta_t).reshape(1, 3)
        delta = project(cam_s, K) - base
        out.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return out


def mesh_center(mesh):
    """Object-coordinate bounding-box center. / 物体坐标下的包围盒中心。"""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    return 0.5 * (v.min(axis=0) + v.max(axis=0))


def evaluate(model, runtime, mesh, pose, K, box, photo, label, image_shape, min_cert=0.25):
    """Render and fit once, returning diagnostics. / 单次渲染并拟合，返回诊断。"""
    color, depth, hit = render_box(runtime, mesh, pose, K, box, image_shape)
    touched = bool(hit[0].any() or hit[-1].any() or hit[:, 0].any() or hit[:, -1].any())
    src = gray_render(color, hit)
    matched = match_points(model, src, photo, depth, hit, box, pose, K, min_cert=min_cert)
    if matched is None:
        print(label, "NO_MATCH", "clipped", touched, flush=True)
        return None, color, hit
    points, target, weights = matched
    fit = solve_about(points, target, pose, K, mesh_center(mesh))
    fit["clipped"] = touched
    fit["match_n"] = int(len(points))
    fit["cert"] = float(np.median(weights))
    print(
        label,
        "n", fit["match_n"],
        "cert", round(fit["cert"], 3),
        "scales", [round(v, 3) for v in fit["scales"]],
        "dt_mm", [round(v, 1) for v in fit["delta_t_mm"]],
        "rms", None if fit["rms_px"] is None else round(fit["rms_px"], 2),
        "px_per_10pct", [round(v, 2) for v in fit["strength"]],
        "clip", touched,
        flush=True,
    )
    return fit, color, hit


def save_pair(photo, color, hit, path):
    """Save photo/render comparison. / 保存照片与渲染对照。"""
    render = gray_render(color, hit)
    pair = np.concatenate([photo, render], axis=1)
    cv2.imwrite(path, cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
