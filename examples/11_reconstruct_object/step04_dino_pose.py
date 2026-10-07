# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Stage 04, called by examples/11_reconstruct_object.py; the entry ranks the cross-scene mustard.
# 第 04 段，由 examples/11_reconstruct_object.py 调用；入口块给跨场景芥末瓶排序。
# Stage 04 in the combined example; its standalone main checks a cross-scene ranking.
# 综合示例的第 04 阶段；单独运行时检查跨场景位姿排序。
# DINOv2 picks the drawn pose among the 12. python examples/11_reconstruct_object/step04_dino_pose.py
# DINOv2 在 12 个位姿里留下画出来的那一个。运行：python examples/11_reconstruct_object/step04_dino_pose.py
"""Keep WAPR poses above a threshold, then pick the DINOv2 match.

先用 WAPR 的阈值留下位姿，再用 DINOv2 选和照片最像的一张。

WAPR and SAPR refine 12 hypotheses. within_group above WITHIN_MIN stays.
DINOv2-L compares each rendered crop with the photo crop, and the highest
cosine is the pose. A larger within_group is better. A larger cosine is closer
to the photo.
WAPR 和 SAPR 修正 12 个候选姿态。within_group 高于 WITHIN_MIN 的留下。DINOv2-L
比较每个渲染裁剪和照片裁剪，余弦最高的是选中的位姿。within_group 越大越好。
余弦越大越接近照片。
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.dirname(os.path.dirname(DEMO_DIR))
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from step01_point_mask import mesh_diameter_m  # noqa: E402
from wapr.estimator import (  # noqa: E402
    WAPREstimator,
    center_from_mesh,
    guess_translation,
    make_view_rots,
    poses_original_to_centered,
    prepare_mesh,
)
from wapr.ogl import depth2xyzmap_batch, make_crop_pair  # noqa: E402
import wapr.recipe as recipe  # noqa: E402


DEVICE = "cuda:0"
OUT_DIR = os.path.join(RELEASE_DIR, "outputs", "reconstruction_stages", "step04_dino_pose")
DINO_REPO = os.path.join(RELEASE_DIR, "third_party", "dinov2")
DINO_WEIGHT = os.path.join(RELEASE_DIR, "assets", "weights", "det2d", "dinov2_vitl14_pretrain.pth")
# Poses at or below this within_group are dropped before DINOv2 ranks the rest.
# within_group 小于或等于这个值的位姿先去掉，剩下的再由 DINOv2 排序。
WITHIN_MIN = 0.0
# How many geometric hypotheses DINOv2 is allowed to see in the shortlist mix below.
# 下面的短名单混合里，DINOv2 能看到的几何候选姿态数量。
TOP_K = 3
# Weight of the DINOv2 cosine after both scores are scaled to 0–1. The rest is within_group.
# 两路都缩到 0–1 之后，DINOv2 余弦的权重。剩下的是 within_group。
DINO_MIX = 0.5
DINO_PX = 224


def unit_interval(values):
    """
    # Map a vector onto 0–1 and return that vector.

    ## Args

        - values: a numeric array with a min and a max. It is not None.

    ## Returns

        - The return has the same shape as values, dtype float64, unitless, in 0–1.
        - A span below 1e-6 returns ones.
        - It is not None.

    ---

    # 把一组数缩到 0–1，并返回这组数。

    ## 参数

        - values: 带最小值和最大值的数值数组。不是 None。

    ## 返回

        - 返回值和 values 同形，dtype 为 float64，无量纲，范围 0–1。
        - 极差小于 1e-6 时返回全 1。
        - 不是 None。

"""
    low = float(values.min())
    high = float(values.max())
    if high - low < 1.0e-6:
        return np.ones_like(values, dtype=np.float64)
    return (values - low) / (high - low)


def refine_candidates(estimator, rgb, depth_m, mask, k, mesh):
    """
    # Refine the view hypotheses and return poses in the original mesh frame.

    ## Args

        - estimator: a WAPREstimator. It is not None. Its wapr_w_mask and sapr nets refine the poses, then WBPS scores them.
        - rgb: (H, W, 3) uint8 RGB. It is not None.
        - depth_m: (H, W) depth in meters. It is not None.
        - mask: (H, W). Nonzero pixels are the object. It is not None.
        - k: (3, 3) camera intrinsics in pixels. It is not None.
        - mesh: the trimesh to pose. Vertices are meters. It is not None.

    ## Returns

        - The return has six fields, and none of them is None.
        - prepared: the centered trimesh from prepare_mesh, meters.
        - center: (3,) float32 meters, the stored model center.
        - poses: a torch float32 tensor of shape (recipe.n_view * recipe.n_inplane, 4, 4) in the original mesh frame; translation is meters.
        - within_group: a 1D numpy float vector, one score per pose, larger is better.
        - diameter: one float, meters.
        - between: one numpy WBPS between-group score per hypothesis; max is used for score_6d.

    ---

    # 修正各个视角的候选姿态，并返回原始网格坐标系中的位姿。

    ## 参数

        - estimator: WAPREstimator。不是 None。它的 wapr_w_mask 和 sapr 网络修正位姿，然后由 WBPS 评分。
        - rgb: (H, W, 3) uint8 RGB。不是 None。
        - depth_m: (H, W) 深度，单位米。不是 None。
        - mask: (H, W)。非零像素是物体。不是 None。
        - k: (3, 3) 相机内参，单位像素。不是 None。
        - mesh: 要估计位姿的 trimesh。顶点单位米。不是 None。

    ## 返回

        - 返回值有六项，都不是 None。
        - prepared: prepare_mesh 居中后的 trimesh，单位米。
        - center: (3,) float32，米，存下的模型中心。
        - poses: torch float32，形状 (recipe.n_view * recipe.n_inplane, 4, 4)，原始网格坐标系，平移单位米。
        - within_group: 一维 numpy 浮点向量，每个位姿一个分数，越大越好。
        - diameter: 一个浮点，单位米。
        - between: 每个候选姿态的 WBPS 组间分数；最大值用于计算 score_6d。

"""
    prepared = estimator.prepare_meshes([mesh])[0]
    mesh_setup_seconds = estimator.last_mesh_setup_seconds
    center = center_from_mesh(prepared)
    translation = guess_translation(depth_m, mask, k)
    rots = make_view_rots(recipe.n_view, recipe.n_inplane)
    count = int(rots.shape[0])
    poses = np.zeros((count, 4, 4), dtype=np.float32)
    poses[:, 3, 3] = 1
    poses[:, :3, :3] = rots
    poses[:, :3, 3] = translation.reshape(1, 3)
    poses = torch.as_tensor(poses, device=estimator.device, dtype=torch.float32)
    diameter = mesh_diameter_m(prepared.vertices)
    warmup_seconds = estimator.warmup_pose(
        rgb, depth_m, k, [{"mesh": prepared, "diameter_m": diameter, "mask": mask}],
    )
    rgb_device = torch.as_tensor(rgb, device=estimator.device, dtype=torch.float32)
    depth_device = torch.as_tensor(depth_m, device=estimator.device, dtype=torch.float32)
    mask_device = torch.as_tensor(mask, device=estimator.device, dtype=torch.float32)
    k_device = torch.as_tensor(k, device=estimator.device, dtype=torch.float32)[None]
    observed_xyz = depth2xyzmap_batch(depth_device[None], k_device).permute(0, 3, 1, 2)
    mesh_ids = [estimator.renderer.cached_mesh_id(prepared)] * count
    torch.cuda.synchronize(estimator.device)
    pose_started = time.perf_counter()
    poses = estimator._refine(
        estimator.nets["wapr_w_mask"], poses, rgb_device, depth_device, mask_device, k, prepared, center, diameter,
        iters=recipe.wapr_iters, max_rot=recipe.wapr_max_rot_rad,
        crop_ratio=recipe.refine_crop_ratio, use_mask=True,
        group_size=count, observed_xyz=observed_xyz, mesh_ids=mesh_ids,
    )
    poses = estimator._refine(
        estimator.nets["sapr"], poses, rgb_device, depth_device, mask_device, k, prepared, center, diameter,
        iters=recipe.sapr_iters, max_rot=recipe.sapr_max_rot_rad,
        crop_ratio=recipe.refine_crop_ratio, use_mask=False,
        group_size=count, observed_xyz=observed_xyz, mesh_ids=mesh_ids,
    )
    within_group, between = estimator._wbps(
        poses, rgb_device, depth_device, mask_device, k, prepared, center, diameter, count,
        observed_xyz=observed_xyz, mesh_ids=mesh_ids,
    )
    within_group = within_group.reshape(-1).detach().float().cpu().numpy()
    between = between.reshape(-1).detach().float().cpu().numpy()
    torch.cuda.synchronize(estimator.device)
    estimator.last_candidate_timing = {"model_setup_seconds": estimator.model_setup_seconds,
        "mesh_setup_seconds": mesh_setup_seconds, "warmup_seconds": warmup_seconds,
        "refine_score_hot_seconds": time.perf_counter() - pose_started,
        "backend": str(recipe.backend), "renderer": "ogl",
        "refine_score_time_scope": "prepared GPU tensors; H2D preparation separate"}
    return prepared, center, poses, within_group, diameter, between


def refine_group(estimator, rgb, depth_m, mask, k, mesh):
    """Keep the gallery helper's return format. / 保留历史绘图辅助函数的返回格式。"""
    prepared, center, poses, within, diameter, between = refine_candidates(
        estimator, rgb, depth_m, mask, k, mesh,
    )
    return prepared, center, poses, within, diameter


def dino_model():
    """
    # Return the DINOv2-L network, loading the local weights on the first call.

    ## Args

        - There are no arguments.

    ## Returns

        - The return is the eval-mode network on DEVICE.
        - It is not None.
        - Later calls return the same object stored on dino_model.net.

    ---

    # 返回 DINOv2-L 网络。

        第一次调用时加载本地权重。

    ## 参数

        - 没有参数。

    ## 返回

        - 返回值是 DEVICE 上处于 eval 的网络。
        - 不是 None。
        - 之后的调用返回存在 dino_model.net 上的同一个对象。

"""
    if getattr(dino_model, "net", None) is None:
        dino_repo = DINO_REPO
        dino_weight = DINO_WEIGHT
        # Preserve provided resources; fetch the same ViT-L/14 only if absent.
        # 保留已有资源；缺失时才获取相同 ViT-L/14 源码与权重。
        from wapr.bootstrap import ensure_optional
        ensure_optional("dinov2")
        if not os.path.isfile(os.path.join(dino_repo, "hubconf.py")):
            if dino_repo != os.path.join(RELEASE_DIR, "third_party", "dinov2"):
                raise FileNotFoundError(dino_repo)
            from wapr.source_setup import prepare_source
            dino_repo = prepare_source("dinov2")
        if not os.path.isfile(dino_weight):
            if dino_weight != os.path.join(RELEASE_DIR, "assets", "weights", "det2d", "dinov2_vitl14_pretrain.pth"):
                raise FileNotFoundError(dino_weight)
            from wapr.det2d import default_weights_dir, prepare_dino_weight
            prepare_dino_weight(default_weights_dir, dino="vitl14")
            dino_weight = os.path.join(default_weights_dir, "dinov2_vitl14_pretrain.pth")
        # UniPose may already own this package namespace; use the original backbone entry only.
        # UniPose 可能已载入此包命名空间；只调用原始骨干入口，不导入无关的新版 cell 模型。
        if dino_repo not in sys.path:
            sys.path.insert(0, dino_repo)
        from dinov2.hub import backbones
        print("DINO_BACKBONE_SOURCE", backbones.__file__, flush=True)
        net = backbones.dinov2_vitl14(pretrained=False)
        state = torch.load(dino_weight, map_location="cpu", weights_only=True)
        net.load_state_dict(state, strict=True)
        dino_model.net = net.eval().to(DEVICE)
    return dino_model.net


def dino_cosine(rgb_a, rgb_b, hit):
    """
    # Return the CLS cosine between each rendered crop and its photo crop.

    ## Args

        - rgb_a: a torch tensor (N, 3, H, W), the rendered RGB crop, roughly 0–1. It is not None.
        - rgb_b: a torch tensor (N, 3, H, W), the photo RGB crop, roughly 0–1. It is not None.
        - hit: a torch tensor (N, 1, H, W). Nonzero marks the rendered silhouette. It is not None. Both crops are multiplied by this mask before DINOv2.

    ## Returns

        - The return is a numpy float vector of shape (N,), unitless cosine.
        - Larger: closer. It is not None.

    ---

    # 返回每张渲染裁剪和对应照片裁剪的 CLS 余弦。

    ## 参数

        - rgb_a: torch 张量 (N, 3, H, W)，渲染出的 RGB 裁剪，大约 0–1。不是 None。
        - rgb_b: torch 张量 (N, 3, H, W)，照片 RGB 裁剪，大约 0–1。不是 None。
        - hit: torch 张量 (N, 1, H, W)。非零处是渲染轮廓。不是 None。两路裁剪在进入 DINOv2 之前都乘上这个 mask。

    ## 返回

        - 返回值是形状 (N,) 的 numpy 浮点向量，无量纲余弦。
        - 越大越接近。
        - 不是 None。

"""
    model = dino_model()
    covered = hit.float()
    images = torch.cat([rgb_a * covered, rgb_b * covered], dim=0)
    images = F.interpolate(images, size=(DINO_PX, DINO_PX), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    images = (images - mean) / std
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        tokens = model.forward_features(images)
    cls = tokens["x_norm_clstoken"].float()
    count = rgb_a.shape[0]
    return F.cosine_similarity(cls[:count], cls[count:], dim=-1).detach().float().cpu().numpy()


def select_pose_result(estimator, rgb, depth_m, mask, k, mesh):
    """
    # Return the pose whose render is closest to the photo among the kept hypotheses.

    ## Args

        - estimator: a WAPREstimator. It is not None.
        - rgb: (H, W, 3) uint8 RGB. It is not None.
        - depth_m: (H, W) depth in meters. It is not None.
        - mask: (H, W). Nonzero pixels are the object. It is not None.
        - k: (3, 3) camera intrinsics in pixels. It is not None.
        - mesh: the trimesh. Vertices are meters. It is not None.

    ## Returns

        - Returns a dictionary containing the selected pose and scores.
        - pose_4x4: (4, 4) list in the original mesh frame; translation is meters.
        - within_group: one float; larger is better.
        - score_6d: (100 - max between_group) / 200 for this hypothesis group.
        - dino_cosine: one float; larger is closer to the photo.
        - hypothesis: int index; within_group at or below WITHIN_MIN is ignored unless none remain.
        - candidate_count: the number of jointly refined and scored hypotheses.

    ---

    # 在保留的候选姿态中，返回渲染结果与照片最接近的位姿。

    ## 参数

        - estimator: WAPREstimator。不是 None。
        - rgb: (H, W, 3) uint8 RGB。不是 None。
        - depth_m: (H, W) 深度，单位米。不是 None。
        - mask: (H, W)。非零像素是物体。不是 None。
        - k: (3, 3) 相机内参，单位像素。不是 None。
        - mesh: trimesh。顶点单位米。不是 None。

    ## 返回

        - 返回包含所选位姿与评分的字典。
        - pose_4x4: （4, 4）列表，原始网格坐标系，平移单位米。
        - within_group: 一个浮点，越大越好。
        - score_6d: 此候选组的 (100 - max between_group) / 200。
        - dino_cosine: 浮点余弦，越大越接近照片。
        - hypothesis: 整数序号；忽略 within_group 不高于 WITHIN_MIN 的候选姿态，除非没有剩余候选姿态。
        - candidate_count: 批量计算修正与评分的候选数。

"""
    # Load DINOv2 before refinement timing, then warm its actual batched crops.
    # 修正计时前载入 DINOv2，并在其实际批量计算裁剪上完成预热。
    dino_setup_started = time.perf_counter()
    dino_model()
    torch.cuda.synchronize(estimator.device)
    dino_setup_seconds = time.perf_counter() - dino_setup_started
    prepared, center, poses, within_group, diameter, between = refine_candidates(
        estimator, rgb, depth_m, mask, k, mesh,
    )
    poses_c = poses_original_to_centered(poses, center)
    packed_a, packed_b, _pose_a, _dia = make_crop_pair(
        rgb, depth_m, mask, poses_c, k, prepared, diameter,
        crop_ratio=recipe.wbps_crop_ratio, use_mask=True, device=DEVICE,
    )
    dino_warm_started = time.perf_counter()
    for _ in range(3):
        dino_cosine(packed_a[:, 0:3], packed_b[:, 0:3], packed_a[:, 3:4])
    torch.cuda.synchronize(estimator.device)
    dino_warmup_seconds = time.perf_counter() - dino_warm_started
    dino_started = time.perf_counter()
    cosine = dino_cosine(packed_a[:, 0:3], packed_b[:, 0:3], packed_a[:, 3:4])
    torch.cuda.synchronize(estimator.device)
    dino_hot_seconds = time.perf_counter() - dino_started
    keep = within_group > WITHIN_MIN
    if not bool(np.any(keep)):
        keep = np.ones(len(within_group), dtype=bool)
    index = int(np.argmax(np.where(keep, cosine, -1.0)))
    pose = poses[index].detach().float().cpu().numpy()
    return {
        "pose_4x4": pose.tolist(), "within_group": float(within_group[index]),
        "score_6d": (100.0 - float(between.max())) / 200.0,
        "dino_cosine": float(cosine[index]),
        "hypothesis": index, "candidate_count": int(len(poses)),
        "timing": {**estimator.last_candidate_timing, "dino_setup_seconds": dino_setup_seconds,
                   "dino_warmup_seconds": dino_warmup_seconds,
                   "dino_hot_seconds": dino_hot_seconds,
                   "dino_time_scope": "prepared rendered crops; excludes crop generation, model loading and warmup"},
    }


def select_pose(estimator, rgb, depth_m, mask, k, mesh):
    """Select a refined pose using image features. / 用图像特征选择修正后的位姿。"""
    result = select_pose_result(estimator, rgb, depth_m, mask, k, mesh)
    return (np.asarray(result["pose_4x4"]), result["within_group"],
            result["dino_cosine"], result["hypothesis"])




if __name__ == "__main__":
    # Run the script stages directly in the entry block.
    # 在入口块中直接执行脚本各阶段。
    from cross_scene_mustard import load_reconstructed_mesh, load_scene_b

    rgb, depth_m, k, masks = load_scene_b()
    mask = next(item["mask"] for item in masks if item["obj_id"] == 5)
    mesh = load_reconstructed_mesh()
    estimator = WAPREstimator(device=DEVICE)
    prepared, center, poses, within_group, diameter = refine_group(
        estimator, rgb, depth_m, mask, k, mesh,
    )
    poses_c = poses_original_to_centered(poses, center)
    packed_a, packed_b, _pose_a, _dia = make_crop_pair(
        rgb, depth_m, mask, poses_c, k, prepared, diameter,
        crop_ratio=recipe.wbps_crop_ratio, use_mask=True, device=DEVICE,
    )
    rgb_a = packed_a[:, 0:3]
    rgb_b = packed_b[:, 0:3]
    hit = packed_a[:, 3:4]
    cosine = dino_cosine(rgb_a, rgb_b, hit)
    os.makedirs(OUT_DIR, exist_ok=True)
    for index in range(rgb_a.shape[0]):
        left = (rgb_a[index].permute(1, 2, 0).detach().float().cpu().numpy() * 255.0).astype(np.uint8)
        right = (rgb_b[index].permute(1, 2, 0).detach().float().cpu().numpy() * 255.0).astype(np.uint8)
        pair = np.concatenate([cv2.cvtColor(left, cv2.COLOR_RGB2BGR), cv2.cvtColor(right, cv2.COLOR_RGB2BGR)], axis=1)
        cv2.imwrite(os.path.join(OUT_DIR, "hyp_%02d.png" % index), pair)
    order = np.argsort(-within_group)
    print("ALL within_group dino_cosine", flush=True)
    for index in order:
        print(int(index), round(float(within_group[index]), 3), round(float(cosine[index]), 4), flush=True)
    short = order[:TOP_K]
    wapr_unit = unit_interval(within_group[short])
    dino_unit = unit_interval(cosine[short])
    joint = (1.0 - DINO_MIX) * wapr_unit + DINO_MIX * dino_unit
    joint_order = np.argsort(-joint)
    print("TOP", TOP_K, "mix", DINO_MIX, flush=True)
    for place in joint_order:
        index = int(short[place])
        print(
            "hyp", index,
            "within", round(float(within_group[index]), 3),
            "dino", round(float(cosine[index]), 4),
            "joint", round(float(joint[place]), 3),
            flush=True,
        )
