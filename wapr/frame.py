# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Estimate 6D pose from 2D detection on one RGB-D frame.
# 从二维检测估计 6D 位姿。一张 RGB-D 上先做 2D 检测。
# One obj_id restricts the search to that category; multiple instances may be returned.
# obj_ids 只放一个 id 时，仅搜索该类别，但仍可返回多个实例。
# That narrowed call is not estimate_one_category_one_instance.
# 收窄之后的调用仍然不是 estimate_one_category_one_instance。
# n_view and n_inplane may be passed. None uses wapr/recipe.py.
# n_view 和 n_inplane 可以传入。None 用 wapr/recipe.py。
# score_2d_min None uses wapr/det2d.py score_threshold. score_6d_min None keeps every pose.
# score_2d_min 为 None 时用 wapr/det2d.py 的 score_threshold。score_6d_min 为 None 时每条位姿都留。
# Update counts stay in wapr/recipe.py.
# 更新次数仍在 wapr/recipe.py。
import time

import numpy as np
import torch

from wapr.estimator import resolve_view_counts


def _rgb_uint8(rgb):
    """
    # Convert RGB to uint8.

        The array is HxWx3 uint8.

    ## Args

        - rgb: HxWx3. uint8 is copied as is. A float whose maximum is at most 1.5 is treated as 0–1 and scaled to 0–255. A larger float is clipped to 0–255. A different shape raises ValueError.

    ## Returns

        - Returns a contiguous array.

    ---

    # 把 RGB 转成 uint8。

        数组是 HxWx3 的 uint8。

    ## 参数

        - rgb: HxWx3。uint8 原样拷贝。最大值不超过 1.5 的浮点按 0–1 放大到 0–255。更大的浮点截到 0–255。形状不对时抛出 ValueError。

    ## 返回

        - 返回连续数组。

"""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or int(arr.shape[-1]) != 3:
        raise ValueError("rgb must be HxWx3")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    values = arr.astype(np.float32)
    if float(np.nanmax(values)) <= 1.5:
        values = np.clip(values, 0.0, 1.0) * 255.0
    else:
        values = np.clip(values, 0.0, 255.0)
    return np.ascontiguousarray(values.astype(np.uint8))


def select_poses(poses, score_6d_min, inst_count):
    """
    # Keep the top poses in each category.

        Drop low WBPS scores, then keep at most inst_count rows of each category. Other categories do not fill that count.

        inst_count None does not cap.

        It maps obj_id to a count.

        A missing class is not capped.

        Count 0 keeps none of that class.

        A negative count raises ValueError.

        Within a class the higher score_6d comes first.

        The returned order follows the input.

    ## Args

        - poses: the list estimate_frame_many_categories_many_instances built. Each row has obj_id and score_6d. score_6d_min None keeps every row. Otherwise a row stays when score_6d is at least that value.

    ## Returns

        - Returns a list.

    ---

    # 按类别留下分数最高的位姿。

        先丢掉 WBPS 分数不够的，再按类别留下不超过 inst_count 的条数。其他类别不拿来补这个数量。

        inst_count 为 None 时不截断。

        它把 obj_id 映射到数量。

        字典里没有的类别不截断。

        数量 0 则该类一条不留。

        负数抛出 ValueError。

        同一类里 score_6d 高的在前。

        返回顺序仍按输入里的先后。

    ## 参数

        - poses: estimate_frame_many_categories_many_instances 组好的列表。每条有 obj_id 和 score_6d。
        - score_6d_min: None 时每条都留。否则 score_6d 不低于这个值才留下。

    ## 返回

        - 返回列表。

"""
    kept = list(poses)
    if score_6d_min is not None:
        gate = float(score_6d_min)
        kept = [pose for pose in kept if float(pose["score_6d"]) >= gate]
    if inst_count is None:
        return kept
    caps = {}
    for key, value in inst_count.items():
        count = int(value)
        if count < 0:
            raise ValueError("inst_count for obj_id %s is %s" % (key, value))
        caps[int(key)] = count
    grouped = {}
    for index, pose in enumerate(kept):
        grouped.setdefault(int(pose["obj_id"]), []).append((index, pose))

    def rank(item):
        """
        # Sort key for one class: higher score_6d first, then the original index.

        ## Args

            - item: (index, pose). Returns a tuple. select_poses passes it to list.sort. Nothing else calls it.

        ---

        # 同一类的排序键：score_6d 高的在前，然后是原来的下标。

        ## 参数

            - item: (index, pose)。返回一个元组。select_poses 把它交给 list.sort。
        """
        index, pose = item
        return (-float(pose["score_6d"]), index)

    chosen = []
    for obj_id, rows in grouped.items():
        rows.sort(key=rank)
        cap = caps.get(obj_id)
        if cap is None:
            chosen.extend(rows)
        else:
            chosen.extend(rows[:cap])
    chosen.sort(key=lambda item: item[0])
    return [pose for _index, pose in chosen]


def estimate_frame_many_categories_many_instances(
    estimator, detector, rgb, depth_m, K, meshes,
    n_view=None, n_inplane=None,
    obj_ids=None, inst_count=None, score_2d_min=None, score_6d_min=None,
):
    """
    # Estimate 6D pose from 2D detection.

        From 2D detection. One obj_id restricts the search to that category; multiple instances may be returned. This is not estimate_one_category_one_instance.

        n_view and n_inplane follow WAPREstimator.estimate_one_category_one_instance.

        None uses wapr/recipe.py.

        estimator is a WAPREstimator.

        detector is a WAPRDet2D.

        score_2d_min None uses score_threshold in wapr/det2d.py, which is 0.1.

        A box stays when its best allowed class is at least that gate.

        score_6d_min None keeps every pose.

        Otherwise a pose stays when score_6d is at least score_6d_min.

        inst_count then caps each class, higher score_6d first.

        A class missing from inst_count is not capped.

        The order is the class list, then the 2D score, then same-class mask NMS inside the detector, then the WBPS score, then the per-class count.

        A detection without finite, positive depth under its mask stays in detections but cannot enter pose estimation.

        WAPR, SAPR, and WBPS each run once over the remaining instances.

        Each pose has obj_id, score_2d, bbox_xywh, pose_4x4, R, t_m, score_6d, and time_s.

        time_s is the pose batch divided by the number of instances that entered that batch, in seconds.

        Rows dropped after the batch are absent, so the kept rows sum to the batch only when nothing was dropped.

        score_2d is the detector score, not score_6d.

        name is present when the mesh metadata has one.

    ## Args

        - rgb: HxWx3 RGB, uint8 or 0–1.
        - depth_m: HxW meters.
        - K: 3×3, pixels. A 3-channel depth keeps the first channel. Different rgb and depth sizes raise ValueError.
        - meshes: obj_id to (mesh, diameter_m). diameter_m is meters. The visible mask selects wapr_w_mask. The detector bbox is not the pose region.
        - obj_ids: the class list. None uses the mesh ids that are also in the template bank. 2D matching and 6D estimation both stay inside that list. An allowed id missing from meshes raises KeyError.

    ## Returns

        - Returns (poses, timing, detections).
        - poses: a list, in detector order, after those gates. An empty detection list returns an empty list.
        - timing: the detector's timing dict.
        - detections: the detector rows that passed the 2D gate, including rows without usable depth, before score_6d_min and inst_count. Each row has obj_id, score_2d, bbox, and mask. visualize_2d_detection draws the blue 2D box and the tinted mask. visualize_6d_pose draws the red 6D box and the rendered contour.

    ---

    # 从二维检测估计 6D 位姿。

        从 2D 检测开始。obj_ids 只放一个 id 时，仅搜索该类别，但仍可返回多个实例。这不是 estimate_one_category_one_instance。

        n_view 和 n_inplane 与 WAPREstimator.estimate_one_category_one_instance 相同。

        None 用 wapr/recipe.py。

        estimator 是 WAPREstimator。

        detector 是 WAPRDet2D。

        score_2d_min 为 None 时用 wapr/det2d.py 的 score_threshold，也就是 0.1。

        允许类别里的最高分不低于这个阈值，包围盒才留下。

        score_6d_min 为 None 时每条位姿都留。

        否则 score_6d 不低于 score_6d_min 才留下。

        然后 inst_count 按类别截断，score_6d 高的优先。

        inst_count 里没有的类别不截断。

        顺序是先类别，再 2D 分数，再检测器里的同类 mask NMS，再 WBPS 分数，再每个类别的数量。

        mask 内没有有限正深度的检测仍保留在 detections 中，但不进入位姿估计。

        WAPR、SAPR、WBPS 对其余实例各执行一次。

        每条位姿含 obj_id、score_2d、bbox_xywh、pose_4x4、R、t_m、score_6d、time_s。

        time_s 是这一次位姿 batch 除以进入这次 batch 的实例数，单位秒。

        batch 之后删掉的行不在列表里，所以只有没有再删行时，留下的行才加回这次 batch。

        score_2d 是检测器分数，不是 score_6d。

        网格 metadata 里有 name 时，位姿里也有 name。

    ## 参数

        - rgb: HxWx3，RGB，uint8 或 0–1。
        - depth_m: HxW，米。
        - K: 3×3，像素。三通道深度只留第一通道。rgb 和深度尺寸不同则抛出 ValueError。
        - meshes 把 obj_id 映射到 (mesh, diameter_m)。
        - diameter_m 是米。
        - 可见 mask 选择 wapr_w_mask。
        - 检测包围盒不用作位姿区域。
        - obj_ids: 类别名单。None 时用既在 meshes 里、也在模板库里的 id。2D 匹配和 6D 估计都只在这份名单里。名单里的 id 不在 meshes 中则抛出 KeyError。

    ## 返回

        - 返回 (poses, timing, detections)。
        - poses: 列表，顺序与检测器一致，已经过这些阈值。没有检测时返回空列表。
        - timing: 检测器的计时字典。
        - detections: 通过 2D 阈值的检测行，包括没有可用深度的行，还没经过 score_6d_min 和 inst_count。每行有 obj_id、score_2d、bbox、mask。visualize_2d_detection 画出蓝色二维框和带颜色的 mask。visualize_6d_pose 画出红色 6D 框和渲染轮廓。

"""
    from pycocotools import mask as mask_utils

    rgb_u8 = _rgb_uint8(rgb)
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.shape[:2] != rgb_u8.shape[:2]:
        raise ValueError("rgb and depth sizes differ")
    # Classes outside this list are not detected and are not estimated.
    # 名单以外的类别不检测，也不估计。
    if obj_ids is None:
        # A mesh that is not in the template bank cannot be detected. It is left out.
        # 不在模板库里的网格检测不到，这里不放进名单。
        bank = getattr(getattr(detector, "matcher", None), "bank", None)
        bank_ids = None
        if bank is not None and "obj_ids" in bank:
            bank_ids = {int(item) for item in bank["obj_ids"].detach().cpu().tolist()}
        allowed = []
        for key in meshes.keys():
            obj_id = int(key)
            if bank_ids is None or obj_id in bank_ids:
                allowed.append(obj_id)
    else:
        allowed = [int(key) for key in obj_ids]
    missing = [obj_id for obj_id in allowed if obj_id not in meshes]
    if missing:
        raise KeyError("obj_id %s is not in meshes" % missing)
    detector_shape = (tuple(rgb_u8.shape), tuple(allowed))
    detector_warmed = getattr(detector, "_frame_warmed_shapes", set())
    detector_warmup_seconds = 0.0
    if detector_shape not in detector_warmed:
        torch.cuda.synchronize(estimator.device)
        warm_started = time.perf_counter()
        for _ in range(3):
            detector.detect_many_categories_many_instances(rgb_u8, obj_ids=allowed, score_min=score_2d_min)
        torch.cuda.synchronize(estimator.device)
        detector_warmup_seconds = time.perf_counter() - warm_started
        detector_warmed.add(detector_shape)
        detector._frame_warmed_shapes = detector_warmed
    torch.cuda.synchronize(estimator.device)
    frame_started = time.perf_counter()
    instances, timing = detector.detect_many_categories_many_instances(rgb_u8, obj_ids=allowed, score_min=score_2d_min)
    # A pose requires at least one valid depth pixel inside its visible mask.
    # Preserve the 2D rows so the detection and pose stages remain inspectable.
    # 位姿至少需要一个落在可见 mask 内的有效深度像素；原始 2D 行仍保留供检查。
    valid_depth = np.isfinite(depth) & (depth > 0)
    skipped_no_depth = 0
    items = []
    fields = []
    for instance in instances:
        obj_id = int(instance["obj_id"])
        if obj_id not in meshes:
            raise KeyError("obj_id %d is not in meshes" % obj_id)
        mesh, diameter_m = meshes[obj_id]
        visible = np.asarray(mask_utils.decode(instance["mask"]))
        if visible.ndim == 3:
            visible = visible[..., 0]
        if visible.shape[:2] != rgb_u8.shape[:2]:
            raise ValueError("mask size differs from rgb")
        if not np.any((visible > 0) & valid_depth):
            skipped_no_depth += 1
            continue
        # The visible mask is the pose region. bbox stays the detector bbox.
        # 可见 mask 是位姿区域。bbox 仍是检测包围盒。
        items.append({"mesh": mesh, "diameter_m": float(diameter_m), "mask": visible})
        field = {
            "obj_id": obj_id,
            "score_2d": float(instance["score_2d"]),
            "bbox_xywh": [float(value) for value in instance["bbox"]],
        }
        # Keep the 2D detector score separate from the later WBPS 6D score.
        # 2D 检测分数与后续 WBPS 的 6D 分数分别保存。
        mesh_name = (getattr(mesh, "metadata", None) or {}).get("name")
        if mesh_name:
            field["name"] = str(mesh_name)
        fields.append(field)
    if skipped_no_depth:
        print("POSE_SKIP_NO_DEPTH", skipped_no_depth, flush=True)
    # Setup and warmup are separate from the returned pose time. New GPU shapes
    # are warmed once; no pose from these discarded calls updates a track.
    # 资源准备和预热与返回的位姿时间分开。新 GPU 形状只预热一次，丢弃的
    # 预热位姿不会用于更新跟踪状态。
    torch.cuda.synchronize(estimator.device)
    detector_and_filter_seconds = time.perf_counter() - frame_started
    prepared = estimator._pose_meshes([item["mesh"] for item in items])
    mesh_setup_seconds = estimator.last_mesh_setup_seconds
    for item, mesh in zip(items, prepared):
        item["mesh"] = mesh
    view_count, inplane_count = resolve_view_counts(n_view, n_inplane)
    warm_shape = (tuple(depth.shape), view_count * inplane_count,
                  tuple((estimator.renderer.cached_mesh_id(item["mesh"]), item.get("mask") is not None) for item in items))
    warmed = getattr(estimator, "_frame_warmed_pose_shapes", set())
    warmup_seconds = 0.0
    if items and warm_shape not in warmed:
        warmup_seconds = estimator.warmup_pose(rgb_u8, depth, K, items, n_view=n_view, n_inplane=n_inplane)
        warmed.add(warm_shape)
        estimator._frame_warmed_pose_shapes = warmed
    torch.cuda.synchronize(estimator.device)
    started = time.perf_counter()
    estimated = estimator.estimate_many_categories_many_instances(rgb_u8, depth, K, items, n_view=n_view, n_inplane=n_inplane)
    torch.cuda.synchronize(estimator.device)
    elapsed_s = time.perf_counter() - started
    timing["pose_hot_seconds"] = elapsed_s
    timing["pose_warmup_seconds"] = warmup_seconds
    timing["mesh_setup_seconds"] = mesh_setup_seconds
    timing["detector_warmup_seconds"] = detector_warmup_seconds
    timing["hot_frame_seconds"] = detector_and_filter_seconds + elapsed_s
    timing["hot_frame_time_scope"] = "sum of measured detection/filter and pose segments; excludes intervening setup/warmup and output assembly"
    timing["pose_time_scope"] = "prepared mesh, warm GPU batch; excludes model/GL setup and warmup"
    share_s = elapsed_s / float(len(estimated)) if estimated else 0.0
    poses = []
    for field, out in zip(fields, estimated):
        field.update(
            {
                "pose_4x4": out["pose_4x4"],
                "R": out["R"],
                "t_m": out["t_m"],
                "score_6d": out["score_6d"],
                "time_s": share_s,
            }
        )
        poses.append(field)
    # Score gate first, then the per-class count. time_s stays the share of the batch that ran.
    # 先按分数阈值，再按每个类别的数量。time_s 仍是已经执行过的那次 batch 的份额。
    return select_poses(poses, score_6d_min, inst_count), timing, instances
