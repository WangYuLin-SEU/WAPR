# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Observation-only shape updates at a fixed pose. / 固定位姿下仅用观测更新形状。"""
import os

import cv2
import numpy as np

import axis_match_scale as match


def fit_fixed_pose(points, target, pose, k, center):
    """Fit three scales in object coordinates; translation stays in meters.

    只拟合物体坐标系的三轴缩放；位姿平移保持不变，单位米。
    """
    q = points - center
    scales = np.ones(3, dtype=np.float64)
    rotation = pose[:3, :3]
    translation_m = pose[:3, 3]
    # Preserve the eight-step robust fitting recipe and scale prior.
    # 保留八步稳健拟合方案与缩放先验。
    for iteration in range(8):
        camera = (center + q * scales) @ rotation.T + translation_m
        error = match.project(camera, k) - target
        residual = np.linalg.norm(error, axis=1)
        keep = residual < max(4.0, 3.0 * float(np.median(residual)))
        if int(keep.sum()) < 40:
            keep = np.ones(len(points), dtype=bool)
        jacobian = match.pack_jacobian(camera[keep], q[keep], rotation, k)[:, :, :3]
        jacobian = jacobian.reshape(-1, 3)
        vector = error[keep].reshape(-1)
        weight = np.sqrt(np.minimum(1.0, 2.0 / np.maximum(np.abs(vector), 1.0e-6)))
        weighted = jacobian * weight[:, None]
        prior_precision = 1.0 / 0.35 ** 2
        normal = weighted.T @ weighted + np.eye(3) * prior_precision
        gradient = weighted.T @ (vector * weight) + (scales - 1.0) * prior_precision
        damping = 1.0e-2 * np.diag(np.diag(normal) + 1.0e-6)
        step = np.linalg.solve(normal + damping, gradient)
        scales = np.clip(scales - step, 0.65, 1.45)
        if float(np.max(np.abs(step))) < 2.0e-3:
            break
    return scales


def correspondence_rms(points, target, pose, k, center, scales):
    """Evaluate the applied shape at the unchanged pose. / 在不变位姿下评估实际应用的形状。"""
    camera = (center + (points - center) * scales) @ pose[:3, :3].T + pose[:3, 3]
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 1.0e-4):
        return float("inf")
    error = match.project(camera, k) - target
    return float(np.sqrt(np.mean(np.sum(error * error, axis=1))))


def run_rounds(mesh, pose, k, rgb, mask, device, output_dir, max_rounds,
               stop_delta, min_matches, delta_bounds, total_bounds, pad_px, min_cert):
    """Fresh correspondences per round; failed updates retain the previous mesh.

    每轮重新建立对应关系；失败或退化的更新保留上一轮网格。
    """
    from romatch import roma_outdoor
    from wapr.ogl import runtime_for

    runtime = runtime_for(device)
    model = roma_outdoor(device=device, coarse_res=560, upsample_res=560,
                         symmetric=False, use_custom_corr=False)
    box = match.square_box(mask, pad=pad_px)
    photo = cv2.resize(rgb[box[1]:box[3], box[0]:box[2]],
                       (match.TILE, match.TILE), interpolation=cv2.INTER_AREA)
    current = mesh
    scales = np.ones(3, dtype=np.float64)
    rows = []
    stop_reason = "max_rounds"
    # Rounds depend on the preceding shape; this is one object in one frame.
    # 后一轮依赖前一轮形状；这里处理的是同帧的一个物体。
    for index in range(max_rounds):
        color, depth, hit = match.render_box(runtime, current, pose, k, box, rgb.shape)
        clipped = bool(hit[0].any() or hit[-1].any() or hit[:, 0].any() or hit[:, -1].any())
        matched = match.match_points(model, match.gray_render(color, hit), photo,
                                     depth, hit, box, pose, k, min_cert=min_cert)
        row = {"round": index, "accepted": False, "clipped": clipped}
        rows.append(row)
        if clipped:
            stop_reason = "render_clipped"
            break
        if matched is None or len(matched[0]) < min_matches:
            stop_reason = "insufficient_matches"
            row["match_n"] = 0 if matched is None else len(matched[0])
            break
        points, target, weights = matched
        center = match.mesh_center(current)
        try:
            delta = fit_fixed_pose(points, target, pose, k, center)
        except np.linalg.LinAlgError:
            stop_reason = "singular_fit"
            break
        row.update({"match_n": len(points), "median_certainty": float(np.median(weights)),
                    "proposed_delta": delta.tolist()})
        if not np.isfinite(delta).all() or delta.min() <= delta_bounds[0] or delta.max() >= delta_bounds[1]:
            stop_reason = "delta_scale_guard"
            break
        proposed_scales = np.clip(scales * delta, *total_bounds)
        applied_delta = proposed_scales / scales
        before = correspondence_rms(points, target, pose, k, center, np.ones(3))
        after = correspondence_rms(points, target, pose, k, center, applied_delta)
        row.update({"rms_before_px": before if np.isfinite(before) else None,
                    "rms_after_px": after if np.isfinite(after) else None,
                    "applied_delta": applied_delta.tolist()})
        if not np.isfinite(after) or after >= before:
            stop_reason = "correspondence_error_not_improved"
            break
        candidate = match.scale_axes(mesh, proposed_scales)
        # Verify fresh matches on the candidate render before keeping this update.
        # 保留更新前，在候选网格的新渲染上核验重新建立的对应关系。
        candidate_color, candidate_depth, candidate_hit = match.render_box(
            runtime, candidate, pose, k, box, rgb.shape,
        )
        candidate_clipped = bool(candidate_hit[0].any() or candidate_hit[-1].any()
                                 or candidate_hit[:, 0].any() or candidate_hit[:, -1].any())
        if candidate_clipped:
            stop_reason = "candidate_render_clipped"
            break
        fresh = match.match_points(model, match.gray_render(candidate_color, candidate_hit), photo,
                                   candidate_depth, candidate_hit, box, pose, k, min_cert=min_cert)
        if fresh is None or len(fresh[0]) < min_matches:
            stop_reason = "candidate_insufficient_matches"
            break
        fresh_rms = correspondence_rms(fresh[0], fresh[1], pose, k,
                                       match.mesh_center(candidate), np.ones(3))
        row["fresh_rms_px"] = fresh_rms if np.isfinite(fresh_rms) else None
        if not np.isfinite(fresh_rms) or fresh_rms > before:
            stop_reason = "candidate_error_not_improved"
            break
        scales = proposed_scales
        current = candidate
        row.update({"accepted": True, "scales": scales.tolist(),
                    "extents_mm": match.extents_mm(current).tolist()})
        match.save_pair(photo, candidate_color, candidate_hit,
                        os.path.join(output_dir, "roma_%02d.png" % index))
        print("ROMA", index, "scales", scales.tolist(), "rms_px", fresh_rms, flush=True)
        if float(np.max(np.abs(applied_delta - 1.0))) < stop_delta:
            stop_reason = "converged"
            break
    del model
    return current, {"enabled": True, "pose_fixed": True, "reference_cad_used": False,
                     "scales": scales.tolist(), "rounds": rows, "stop_reason": stop_reason}
