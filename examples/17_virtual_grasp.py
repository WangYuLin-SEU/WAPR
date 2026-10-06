# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 17. Previous: examples/16_follow_saved.py. Next: none
# 示例 17。上一例：examples/16_follow_saved.py。下一例：无
# Draw the top grasp on the estimated pose, in the third-person camera.
# 把从上往下的抓取画到估计位姿上，画在第三视角相机里。
# python examples/17_virtual_grasp.py
# 运行：python examples/17_virtual_grasp.py
"""One still per clip. The camera is the one that perceived the object.

每个片段一张静帧。这台相机就是做感知的那一台。

The cyan gripper and the arm both use the first-frame estimated pose.
青色夹爪和手臂都用第 0 帧估计出的位姿。
"""

import importlib
import json
import os
import sys

import numpy as np


# Geometry and saved-pose helpers live in the public recipes 12 and 13.
# 几何与已保存位姿读取函数位于公开示例 13 和 13。
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.dirname(EXAMPLES_DIR)
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)

bridge = importlib.import_module("14_bridge_tasks")
known = importlib.import_module("13_known_mesh_place")


PAGE_DIR = os.path.join(RELEASE_DIR, "pages", "demo", "robot")
# Local gripper drawing, meters. +X is the approach, down. +Y is the finger gap.
# 画出来的夹爪，米。+X 是接近方向，朝下。+Y 是指缝。
# Local +X points down, at the object. The palm and the approach sit on -X, above it.
# 局部 +X 朝下，指向物体。手掌和接近段在 -X 上，在物体上方。
APPROACH_ABOVE_M = 0.08
PALM_ABOVE_M = 0.028
FINGER_GAP_M = 0.036


def grasp_axes(rotation, closing_mode):
    """Approach is world down. Closing is the object's thin axis, or world +Y.

    接近方向是世界的下方。合拢方向是物体较薄的轴，或者世界的 +Y。
    """
    approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    if closing_mode == "world_y":
        closing = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        closing = np.asarray(rotation, dtype=np.float64)[:3, 1].copy()
        closing[2] = 0.0
        closing = closing / max(np.linalg.norm(closing), 1e-8)
    side = np.cross(approach, closing)
    return approach, closing, side


def gripper_segments(center, rotation, closing_mode):
    """Line segments of an open gripper at the object center, world meters.

    张开的夹爪在物体中心处的线段，世界坐标，米。
    """
    approach, closing, _side = grasp_axes(rotation, closing_mode)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = approach
    pose[:3, 1] = closing
    pose[:3, 2] = np.cross(approach, closing)
    pose[:3, 3] = np.asarray(center, dtype=np.float64)

    def world(local):
        point = np.asarray(local, dtype=np.float64)
        return pose[:3, :3] @ point + pose[:3, 3]

    gap = FINGER_GAP_M * 0.5
    above = np.array([-APPROACH_ABOVE_M, 0.0, 0.0], dtype=np.float64)
    palm = np.array([-PALM_ABOVE_M, 0.0, 0.0], dtype=np.float64)
    tip = np.zeros(3, dtype=np.float64)
    left = np.array([0.0, gap, 0.0], dtype=np.float64)
    right = np.array([0.0, -gap, 0.0], dtype=np.float64)
    return (
        (world(above), world(palm)),
        (world(palm + left), world(palm + right)),
        (world(palm + left), world(tip + left)),
        (world(palm + right), world(tip + right)),
    )


def draw_segments(canvas, segments, extrinsic, camera_k, color):
    """Project world segments into the image and draw them.

    把世界线段投到图像上再画出来。
    """
    import cv2

    world_to_cam = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
    intrinsic = np.asarray(camera_k, dtype=np.float64).reshape(3, 3)
    height, width = canvas.shape[:2]
    for start, end in segments:
        points = np.stack([start, end], axis=0)
        camera = (world_to_cam[:3, :3] @ points.T).T + world_to_cam[:3, 3]
        if np.any(camera[:, 2] <= 1e-4):
            continue
        pixels = (intrinsic @ camera.T).T
        pixels = pixels[:, :2] / camera[:, 2:3]
        corners = np.round(pixels).astype(np.int32)
        if np.any(corners[:, 0] < -20) or np.any(corners[:, 0] >= width + 20):
            continue
        if np.any(corners[:, 1] < -20) or np.any(corners[:, 1] >= height + 20):
            continue
        cv2.line(canvas, tuple(corners[0]), tuple(corners[1]), color, 3, cv2.LINE_AA)


if __name__ == "__main__":
    import cv2
    import trimesh
    # This case visualizes grasps from saved WAPR poses; physical execution is in 12–14.
    # 本例基于保存的 WAPR 位姿绘制虚拟抓取，实际机械臂执行流程在 12–14 中。
    root = os.path.join(RELEASE_DIR, "outputs")
    for folder, closing_mode, stem in (
        (os.path.join(root, "sim_bridge_tasks", "carrot"), "thin", "carrot"),
        (os.path.join(root, "sim_bridge_tasks", "eggplant"), "thin", "eggplant"),
        (os.path.join(root, "sim_xarm_cube", "cube"), "world_y", "xarm_cube"),
    ):
        # 1. Load the saved camera frame and estimated object-to-camera pose from 13/14.
        # 1. 读取 13/14 保存的相机帧与估计的物体到相机位姿。
        with open(os.path.join(folder, "rows.json"), "r", encoding="utf-8") as stream:
            rows = json.load(stream)
        poses = bridge.load_poses(folder)
        mesh = trimesh.load(os.path.join(folder, "mesh.ply"), force="mesh", process=False)
        outline = mesh.convex_hull
        row = next((item for item in rows if item.get("job") in poses), None)
        if row is None:
            raise RuntimeError("no pose %s" % folder)
        bgr = cv2.imread(row["view"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(row["view"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        extrinsic = np.asarray(row["render_extrinsic"], dtype=np.float64)
        camera_k = np.asarray(row["render_K"], dtype=np.float64)
        # 2. Convert the estimate to world meters; truth below is for overlay comparison only.
        # 2. 将估计转换为世界坐标（米）；下方真值仅供叠图对照。
        pred = known.world_pose(extrinsic, poses[row["job"]])
        gt = np.asarray(row["gt"], dtype=np.float64).reshape(4, 4)
        mask = cv2.imread(os.path.join(folder, "frames", "f%05d_mask.png" % row["i"]), cv2.IMREAD_GRAYSCALE)
        visible_px = 0 if mask is None else int((mask > 0).sum())
        titles = {"zh": "黄：可见掩码 %d 像素 | 红：估计 | 绿：真值 | 青：抓取" % visible_px,
                  "en": "Visible mask: %d px | red: estimate | green: truth | cyan: grasp" % visible_px}
        # 3. Construct the approach/finger axes from the estimated pose, then project into RGB.
        # The top approach is world-down; closing follows the thin object axis or world +Y.
        # 3. 根据估计位姿构造接近轴与夹指轴，再投影到 RGB 上。
        # 顶部接近方向为世界向下，合拢方向采用物体薄轴或世界 +Y。
        segments = gripper_segments(pred[:3, 3], pred[:3, :3], closing_mode)
        os.makedirs(PAGE_DIR, exist_ok=True)
        for lang, title in titles.items():
            frame = np.array(bridge.draw_frame(rgb, camera_k, extrinsic, outline, pred, gt, title)).copy()
            if mask is not None:
                frame[mask > 0] = (255, 210, 0)
            draw_segments(frame, segments, extrinsic, camera_k, (0, 190, 255))
            name = "%s_grasp.jpg" % stem if lang == "zh" else "%s_grasp_en.jpg" % stem
            path = os.path.join(PAGE_DIR, name)
            if not cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                raise OSError("Cannot write grasp image: " + path)
            print("WROTE", path, "frame", row["i"], flush=True)
