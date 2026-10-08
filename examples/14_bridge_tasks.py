# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 14. Previous: examples/13_known_mesh_place.py. Next: examples/15_xarm_cube.py
# 示例 14。上一例：examples/13_known_mesh_place.py。下一例：examples/15_xarm_cube.py
# WidowX on the Bridge table and the Bridge sink. The objects are that task's meshes.
# WidowX，Bridge 的桌子和洗手台。物体是那个任务自己的网格。
# python examples/14_bridge_tasks.py
# 运行：python examples/14_bridge_tasks.py
# python examples/14_bridge_tasks.py sequence <jobs.json>
# 子进程：python examples/14_bridge_tasks.py sequence <jobs.json>
"""Two BridgeData tasks on a WidowX250S. No YCB mesh.

两个 BridgeData 任务，手臂是 WidowX250S。不用 YCB 网格。

Carrot onto a plate, eggplant into the sink basket.
The arm grasps from the first-frame estimate. The mask is the object's color.
胡萝卜放到盘子上，茄子放进水槽的篮子。
手臂按第 0 帧的估计去抓。mask 按物体的颜色来切。
"""

import importlib
import json
import os
import subprocess
import sys
import time

import cv2
import trimesh
import numpy as np
import torch


# All shared helpers are in the numbered public examples, not a private case library.
# 共用函数均在编号公开示例中，不依赖隐藏的案例库。
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.dirname(EXAMPLES_DIR)
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)

known = importlib.import_module("13_known_mesh_place")

DEVICE = "cuda:0"
from wapr.resources import outputs_dir, samples_dir
from wapr import recipe
OUT_DIR = outputs_dir(__file__)
PAGE_DIR = OUT_DIR
# Fallback overview camera, meters, used when a task does not name its own.
# 任务没有写自己的视点时，用这台总览相机，米。
OVERVIEW_EYE_M = np.array([0.55, -0.55, 1.45], dtype=np.float64)
OVERVIEW_TARGET_M = np.array([-0.05, 0.00, 0.88], dtype=np.float64)
OVERVIEW_WIDTH = 960
OVERVIEW_HEIGHT = 720
OVERVIEW_FOVY = 0.95
# How far above the object the open gripper stops, then how far it lifts, meters.
# 张开的夹爪停在物体上方多高，再提起多高，米。
PREGRASP_M = 0.08
LIFT_M = 0.10
# Extra height above the target center when the object is set down, meters.
# 放到目标中心之上再留的高度，米。
PLACE_CLEAR_M = 0.03
# The eggplant body in this camera is about 170 px. 200 drops frame 0.
# 这台相机里茄子身体大约 170 像素。200 会丢掉第 0 帧。
MIN_MASK_PX = 150
VIDEO_FPS = 20
# Use the public font shipped with WAPR on every machine.
# 所有机器都读取 WAPR 随包分发的公共字体。
import wapr
# The licensed font ships with the wheel, even when examples live elsewhere.
# 字体及许可随 wheel 分发；示例位于其他目录时仍从已安装包定位。
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(wapr.__file__)), "fonts", "wqy-microhei.ttc")

# env_id is a ManiSkill Bridge task. source is the moving mesh.
# env_id 是 ManiSkill 的 Bridge 任务。source 是被搬走的那份网格。
TASKS = (
    {
        "key": "carrot",
        "env_id": "PutCarrotOnPlateInScene-v1",
        "source": "bridge_carrot_generated_modified",
        "en": "Put the carrot on the plate",
        "zh": "把胡萝卜放到盘子上",
        "closing": "thin",
        "eye": [0.55, -0.55, 1.45],
        "target": [-0.05, 0.00, 0.88],
    },
    {
        "key": "eggplant",
        "env_id": "PutEggplantInBasketScene-v1",
        "source": "eggplant",
        "en": "Put the eggplant in the basket",
        "zh": "把茄子放进水槽的篮子",
        "closing": "thin",
        "eye": [0.62, -0.20, 1.55],
        "target": [-0.11, 0.10, 0.95],
    },
)


def install_joint_controller():
    """Give the Bridge WidowX a joint-position arm so the planner can drive it.

    给 Bridge 的 WidowX 一个关节位置手臂，规划器才能直接驱动。
    """
    from mani_skill.agents.controllers.pd_joint_pos import (
        PDJointPosControllerConfig,
        PDJointPosMimicControllerConfig,
    )
    from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import (
        WidowX250SBridgeDatasetFlatTable,
    )

    def configs(self):
        arm = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        gripper = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            0.014,
            0.037,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=True,
            drive_mode="force",
            mimic={"right_finger": {"joint": "left_finger"}},
        )
        return dict(pd_joint_pos=dict(arm=arm, gripper=gripper))

    WidowX250SBridgeDatasetFlatTable._controller_configs = property(configs)


def make_env(env_id):
    """One Bridge task, WidowX, joint position control.

    一个 Bridge 任务，WidowX，关节位置控制。
    """
    # Prepare simulation dependencies only when creating the environment.
    # 仅在创建仿真环境时准备机器人可选依赖。
    from wapr.bootstrap import ensure_optional
    ensure_optional("robot")
    import gymnasium as gym

    import mani_skill.envs  # noqa: F401

    # Prepare both the robot and the task before gym starts its own downloader.
    # 在 gym 启动内置下载器前准备机械臂及任务资源。
    from wapr.source_setup import prepare_robot_assets
    prepare_robot_assets("widowx250s")
    prepare_robot_assets("bridge_v2_real2sim")

    install_joint_controller()
    # This task only boots with rgb+segmentation. Greenscreening uses that channel.
    # The pose mask is color_mask. capture_overview does not read segmentation.
    # 这个任务只能用 rgb+segmentation 启动。贴背景用那一路。
    # 位姿的 mask 是 color_mask。capture_overview 不读分割。
    env = gym.make(
        env_id,
        obs_mode="rgb+segmentation",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="cpu",
    )
    # The registered limit is 60. A pick and a place take more steps than that.
    # 注册时的上限是 60。一次抓取再加一次放下会超过这个步数。
    env._max_episode_steps = 2000
    return env


def make_planner(env):
    """mplib on ee_gripper_link. The base pose is the WidowX mount, not the origin.

    mplib 规划 ee_gripper_link。底座位姿是 WidowX 的安装位置，不是原点。
    """
    import sapien
    import xml.etree.ElementTree as ET

    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

    # mplib's convex=True loader requires derived .convex.stl collision meshes.
    # mplib 的 convex=True 加载器要求派生 .convex.stl；官方资源只含原始 STL。
    urdf_path = env.unwrapped.agent.urdf_path
    urdf_root = ET.parse(urdf_path).getroot()
    for mesh_node in urdf_root.findall("./link/collision/geometry/mesh"):
        mesh_file = mesh_node.attrib["filename"]
        mesh_path = os.path.join(os.path.dirname(urdf_path), mesh_file)
        convex_path = mesh_path + ".convex.stl"
        if not os.path.isfile(convex_path):
            collision_mesh = trimesh.load(mesh_path, force="mesh", process=False)
            convex_mesh = collision_mesh.convex_hull
            pending_path = convex_path + ".pending"
            convex_mesh.export(pending_path, file_type="stl")
            os.replace(pending_path, convex_path)
            print("ROBOT_CONVEX_MESH", convex_path, flush=True)

    class WidowPlanner(PandaArmMotionPlanningSolver):
        OPEN = 1
        CLOSED = -1
        MOVE_GROUP = "ee_gripper_link"

    base = env.unwrapped
    robot_pose = base.agent.robot.pose
    mount = sapien.Pose(robot_pose.p[0].cpu().numpy(), robot_pose.q[0].cpu().numpy())
    return WidowPlanner(
        env,
        debug=False,
        vis=False,
        base_pose=mount,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )


def top_grasp(center, closing):
    """Gripper pointing down. Finger closing follows `closing`, meters.

    夹爪朝下。手指的合拢方向跟着 closing，米。

    WidowX ee_gripper_link +X is the approach, +Y is the finger gap.
    WidowX 的 ee_gripper_link，+X 是接近方向，+Y 是指缝。
    """
    import sapien

    approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    closing = np.asarray(closing, dtype=np.float64)
    closing[2] = 0.0
    closing = closing / max(np.linalg.norm(closing), 1e-8)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = approach
    matrix[:3, 1] = closing
    matrix[:3, 2] = np.cross(approach, closing)
    matrix[:3, 3] = np.asarray(center, dtype=np.float64)
    return sapien.Pose(matrix)


def load_source_mesh(source_name):
    """Collision mesh of the moving object, meters, actor frame.

    运动物体的碰撞网格，米，物体坐标系。
    """

    # The installed catalog resolves its data subdirectory and user asset override.
    # 按已安装资源清单解析 data 子目录与用户覆盖；CAD 先于 make_env 读取。
    from wapr.bootstrap import ensure_optional
    from wapr.source_setup import prepare_robot_assets
    ensure_optional("robot")
    bridge_root = prepare_robot_assets("bridge_v2_real2sim")
    path = os.path.join(bridge_root, "custom", "models", source_name, "collision.obj")
    if not os.path.isfile(path):
        raise FileNotFoundError("Bridge collision mesh missing / Bridge 碰撞网格缺失: " + path)
    mesh = trimesh.load(path, force="mesh", process=False)
    return mesh


def add_overview(env, eye, target):
    """A fixed camera that sees the arm and the table. Its shader stores position.

    一台固定相机，能看到手臂和桌面。这个着色器带位置缓冲。
    """
    from mani_skill.sensors.camera import Camera, CameraConfig
    from mani_skill.utils import sapien_utils

    pose = sapien_utils.look_at(
        np.asarray(eye, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
    )
    config = CameraConfig(
        "overview",
        pose,
        OVERVIEW_WIDTH,
        OVERVIEW_HEIGHT,
        OVERVIEW_FOVY,
        0.01,
        8.0,
        shader_pack="default",
    )
    return Camera(config, env.unwrapped.scene)


def capture_overview(camera):
    """RGB uint8, depth meters, K, world-to-camera. OpenCV, y down.

    RGB uint8，深度米，K，世界到相机。OpenCV，y 向下。

    The default shader stores position as int16 millimeters in the OpenGL camera.
    默认着色器把位置存成 OpenGL 相机里的 int16 毫米。
    """
    camera.capture()
    obs = camera.get_obs(rgb=True, depth=False, position=True, segmentation=False)
    rgb = obs["rgb"][0].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb[..., :3] * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb = np.ascontiguousarray(rgb[..., :3])
    position_m = obs["position"][0].detach().cpu().numpy()[..., :3].astype(np.float64) / 1000.0
    depth_m = (-position_m[..., 2]).astype(np.float32)
    params = camera.get_params()
    extrinsic = params["extrinsic_cv"]
    if hasattr(extrinsic, "detach"):
        extrinsic = extrinsic.detach().cpu().numpy()
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.ndim == 3:
        extrinsic = extrinsic[0]
    if extrinsic.shape == (3, 4):
        full = np.eye(4, dtype=np.float64)
        full[:3] = extrinsic
        extrinsic = full
    intrinsic = params["intrinsic_cv"]
    if hasattr(intrinsic, "detach"):
        intrinsic = intrinsic.detach().cpu().numpy()
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.ndim == 3:
        intrinsic = intrinsic[0]
    return rgb, depth_m, intrinsic[:3, :3], extrinsic


def actor_matrix(actor):
    """Actor pose, 4x4, meters.

    物体位姿，4x4，米。
    """
    return actor.pose.to_transformation_matrix()[0].detach().cpu().numpy().astype(np.float64)


def color_mask(rgb, key):
    """Pixels of this object by its visible color. The largest blob is kept.

    按看得见的颜色取这个物体的像素。留下最大的一块。

    These cuts are the lit appearance of each task object. They are not actor ids.
    这些阈值是每个任务物体在光照下的样子。不是物体编号。
    """

    image = np.asarray(rgb)
    if image.ndim == 4:
        image = image[0]
    red = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    blue = image[..., 2].astype(np.int16)
    if key == "carrot":
        raw = (red > 140) & (green > 40) & (green < 180) & (blue < 90) & (red > green + 25)
    elif key == "eggplant":
        raw = (red > 40) & (blue > 40) & (red > green + 10) & (blue > green + 10) & (green < 100)
    elif key == "cube":
        # The wooden table also passes a looser red cut. This keeps the cube, about 400 px.
        # 木桌也会通过更松的红色阈值。这一档留下方块，大约 400 像素。
        raw = (red > 160) & (green < 80) & (blue < 80) & (red > green + 80) & (red > blue + 80)
    else:
        raise KeyError(key)
    labels_n, labels, stats, _centroid = cv2.connectedComponentsWithStats(raw.astype(np.uint8), 8)
    if labels_n <= 1:
        return np.zeros(raw.shape, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    chosen = 1 + int(np.argmax(areas))
    kept = labels == chosen
    # A blob this large is the table or the background, not the object.
    # 大到这个程度的一块是桌子或背景，不是物体。
    if int(kept.sum()) > int(0.4 * kept.size):
        return np.zeros(kept.shape, dtype=bool)
    return kept


def flat_closing(rotation):
    """Horizontal thin axis. Local Y, with world Z removed.

    水平方向上较薄的一侧。局部 Y，去掉世界 Z。
    """
    closing = np.asarray(rotation, dtype=np.float64).reshape(3, 3)[:, 1].copy()
    closing[2] = 0.0
    norm = float(np.linalg.norm(closing))
    if norm < 1e-8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return closing / norm


def record_frame(camera, folder, index, source, target, mesh_path, diameter_m, jobs, mask_key):
    """Save the overview, and queue one pose when the color mask is large enough.

    存下总览。颜色 mask 够大时，排一次位姿。
    """

    camera.camera.scene.update_render()
    rgb, depth_m, intrinsic, extrinsic = capture_overview(camera)
    frame_dir = os.path.join(folder, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    view_path = os.path.join(frame_dir, "f%05d_view.png" % index)
    cv2.imwrite(view_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    mask = color_mask(rgb, mask_key)
    job_id = None
    if int(mask.sum()) >= MIN_MASK_PX:
        job_id = "f%05d" % index
        depth_path = os.path.join(frame_dir, "f%05d_depth.npy" % index)
        mask_path = os.path.join(frame_dir, "f%05d_mask.png" % index)
        np.save(depth_path, depth_m)
        cv2.imwrite(mask_path, (mask.astype(np.uint8) * 255))
        jobs.append({
            "id": job_id,
            "rgb": view_path,
            "depth": depth_path,
            "mask": mask_path,
            "K": intrinsic.tolist(),
            "mesh": mesh_path,
            "diameter_m": float(diameter_m),
        })
    return {
        "i": int(index),
        "view": view_path,
        "job": job_id,
        "gt": actor_matrix(source).reshape(4, 4).tolist(),
        "target": actor_matrix(target)[:3, 3].tolist(),
        "render_K": intrinsic.tolist(),
        "render_extrinsic": extrinsic.tolist(),
    }


def carry(env, camera, folder, source, target, mesh_path, diameter_m, grasp_center, grasp_closing, mask_key):
    """Open, grasp from above, lift, set the object on the target, open, back up.

    张开，从上往下抓，提起，放到目标上，张开，再抬起来。

    grasp_center and grasp_closing come from the first-frame estimate, meters.
    grasp_center 和 grasp_closing 来自第 0 帧的估计，米。
    """
    planner = make_planner(env)
    jobs = []
    rows = []
    state = {"i": 0}
    original_step = env.step

    def remember(action):
        outcome = original_step(action)
        rows.append(record_frame(
            camera, folder, state["i"], source, target, mesh_path, diameter_m, jobs, mask_key,
        ))
        state["i"] += 1
        if state["i"] % 20 == 0:
            print("FRAME", state["i"], "jobs", len(jobs), flush=True)
        return outcome

    env.step = remember
    planner.open_gripper()
    center = np.asarray(grasp_center, dtype=np.float64).reshape(3).copy()
    closing = np.asarray(grasp_closing, dtype=np.float64).reshape(3).copy()
    above = top_grasp(center + np.array([0.0, 0.0, PREGRASP_M]), closing)
    known.move_arm(planner, above)
    grasp = top_grasp(center + np.array([0.0, 0.0, 0.0]), closing)
    known.move_arm(planner, grasp)
    planner.close_gripper(t=12)
    lifted = top_grasp(center + np.array([0.0, 0.0, LIFT_M]), closing)
    known.move_arm(planner, lifted)
    goal = target.pose.p[0].detach().cpu().numpy().copy()
    goal[2] = float(goal[2]) + PLACE_CLEAR_M + LIFT_M
    known.move_arm(planner, top_grasp(goal, closing))
    goal[2] = float(target.pose.p[0, 2].detach().cpu().numpy()) + PLACE_CLEAR_M
    known.move_arm(planner, top_grasp(goal, closing))
    planner.open_gripper(t=8)
    goal[2] = goal[2] + LIFT_M
    known.move_arm(planner, top_grasp(goal, closing))
    planner.close()
    return rows, jobs


def estimate_sequence(path):
    """Child entry. One pose network, then every queued frame.

    子进程入口。位姿网络载入一次，然后估计排进来的每一帧。
    """

    from wapr import WAPREstimator

    with open(path, "r", encoding="utf-8") as stream:
        jobs = json.load(stream)
    estimator = WAPREstimator(device=DEVICE)
    meshes = {}
    warmed_shapes = set()
    written = []
    frame_jobs = {}
    for job in jobs:
        key = (job["rgb"], job["depth"], tuple(np.asarray(job["K"]).reshape(-1)))
        frame_jobs.setdefault(key, []).append(job)
    for same_frame in frame_jobs.values():
        first = same_frame[0]
        rgb = cv2.cvtColor(cv2.imread(first["rgb"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        depth_m = np.load(first["depth"]).astype(np.float32)
        camera_k = np.array(first["K"], dtype=np.float32).reshape(3, 3)
        instances, valid_jobs = [], []
        for job in same_frame:
            mesh_path = job["mesh"]
            if mesh_path not in meshes:
                source = trimesh.load(mesh_path, force="mesh", process=False)
                meshes[mesh_path] = estimator.prepare_meshes([source])[0]
            mask = cv2.imread(job["mask"], cv2.IMREAD_GRAYSCALE)
            if (mask is None or mask.shape != depth_m.shape or
                    not np.any((mask > 0) & np.isfinite(depth_m) & (depth_m > 0))):
                written.append({"id": job["id"], "error": "Missing mask or valid masked depth"})
                continue
            instances.append({"mesh": meshes[mesh_path], "diameter_m": float(job["diameter_m"]), "mask": mask})
            valid_jobs.append(job)
        try:
            shape = (tuple(depth_m.shape), tuple(estimator.renderer.cached_mesh_id(item["mesh"]) for item in instances))
            if instances and shape not in warmed_shapes:
                warmup_seconds = estimator.warmup_pose(rgb, depth_m, camera_k, instances)
                warmed_shapes.add(shape)
                print("POSE_WARMUP_SECONDS", warmup_seconds, flush=True)
            torch.cuda.synchronize(DEVICE)
            pose_started = time.perf_counter()
            outputs = estimator.estimate_many_categories_many_instances(rgb, depth_m, camera_k, instances)
            torch.cuda.synchronize(DEVICE)
            pose_hot_seconds = time.perf_counter() - pose_started
        except Exception as error:
            for job in valid_jobs:
                print("ESTIMATE_MISS", job["id"], type(error).__name__, error, flush=True)
                written.append({"id": job["id"], "error": str(error)})
            continue
        for job, out in zip(valid_jobs, outputs):
            pose = np.asarray(out["pose_4x4"], dtype=np.float64)
            written.append({"id": job["id"], "pose": pose.tolist(), "score_6d": float(out["score_6d"]),
                            "pose_hot_seconds": pose_hot_seconds, "pose_time_scope": "warm prepared-mesh GPU frame batch"})
            print("ESTIMATE", job["id"], "t_m", np.round(pose[:3, 3], 4).tolist(),
                  "score_6d", round(float(out["score_6d"]), 4), flush=True)
    out_path = os.path.splitext(path)[0] + "_poses.json"
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(written, stream)
    print("ESTIMATE_WROTE", out_path, flush=True)


def run_sequence(path):
    """Parent side. Returns {frame id: object-to-camera}.

    父进程一侧。返回 {帧编号: 物体到相机}。
    """
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "sequence", path],
        cwd=RELEASE_DIR,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("sequence failed %s" % proc.returncode)
    pose_path = os.path.splitext(path)[0] + "_poses.json"
    with open(pose_path, "r", encoding="utf-8") as stream:
        rows = json.load(stream)
    poses = {}
    for row in rows:
        if "pose" not in row:
            continue
        poses[row["id"]] = np.asarray(row["pose"], dtype=np.float64)
    return poses


def draw_frame(rgb, camera_k, extrinsic, mesh, pred_world, gt_world, title):
    """Green ground truth, then red prediction.

    先画绿色真值，再画红色预测。
    """
    from PIL import Image, ImageDraw, ImageFont

    canvas = np.ascontiguousarray(rgb[..., :3].copy())
    height, width = canvas.shape[:2]
    world_to_cam = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
    gt_cam = world_to_cam @ np.asarray(gt_world, dtype=np.float64).reshape(4, 4)
    gt_mask = known.silhouette_mask(mesh, gt_cam, camera_k, height, width)
    known.draw_largest_contour(canvas, gt_mask, (0, 170, 0), 3)
    if pred_world is not None:
        pred_cam = world_to_cam @ np.asarray(pred_world, dtype=np.float64).reshape(4, 4)
        pred_mask = known.silhouette_mask(mesh, pred_cam, camera_k, height, width)
        known.draw_largest_contour(canvas, pred_mask, (220, 30, 30), 2)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, 28)
    draw.rectangle([0, 0, width, 42], fill=(18, 18, 18))
    draw.text((12, 6), title, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def compose_video(task, rows, poses, mesh, lang):
    """Write the clip and a poster. The outline uses the convex hull.

    写出视频和海报。轮廓用凸包。

    lang is "zh" or "en". The page shows the matching file.
    lang 是 "zh" 或 "en"。页面显示对应的那一份。
    """
    import imageio.v2 as imageio

    if lang == "en":
        title = "%s  red estimate  green truth  cyan grasp" % task["en"]
        stem = "%s_en" % task["key"]
    else:
        title = "%s    红：预测    绿：真值    青：抓取" % task["zh"]
        stem = task["key"]
    # Bridge tasks close along the thin axis. The cube task sets closing to world +Y.
    # Bridge 任务沿着较薄的轴合拢。方块任务把 closing 设成世界 +Y。
    closing_mode = task["closing"]
    grasp_draw = importlib.import_module("17_virtual_grasp")

    outline = mesh.convex_hull
    frames = []
    centers = []
    axes = []
    for row in rows:
        pred = None
        job = row.get("job")
        if job in poses:
            pred = known.world_pose(np.asarray(row["render_extrinsic"], dtype=np.float64), poses[job])
            err_mm, err_deg = known.pose_error(pred, np.asarray(row["gt"], dtype=np.float64))
            centers.append(err_mm)
            axes.append(err_deg)
        frame = imageio.imread(row["view"])
        painted = np.array(draw_frame(
            frame,
            np.asarray(row["render_K"]),
            np.asarray(row["render_extrinsic"]),
            outline,
            pred,
            np.asarray(row["gt"]),
            title,
        )).copy()
        if pred is not None:
            segments = grasp_draw.gripper_segments(pred[:3, 3], pred[:3, :3], closing_mode)
            grasp_draw.draw_segments(
                painted,
                segments,
                np.asarray(row["render_extrinsic"]),
                np.asarray(row["render_K"]),
                (0, 190, 255),
            )
        frames.append(painted)
    os.makedirs(PAGE_DIR, exist_ok=True)
    video_path = os.path.join(PAGE_DIR, "%s.mp4" % stem)
    writer = imageio.get_writer(video_path, fps=VIDEO_FPS, codec="libx264", quality=8, macro_block_size=1)
    for frame in frames:
        writer.append_data(np.ascontiguousarray(frame))
    writer.close()
    poster = frames[min(len(frames) // 3, len(frames) - 1)]
    imageio.imwrite(os.path.join(PAGE_DIR, "%s.jpg" % stem), poster)
    report = {
        "key": task["key"],
        "env_id": task["env_id"],
        "frames": len(frames),
        "estimates": len(centers),
        "fps": VIDEO_FPS,
        "video": video_path,
    }
    if centers:
        report["center_mm"] = float(np.median(centers))
        report["axis_deg"] = float(np.median(axes))
    final_gt = np.asarray(rows[-1]["gt"], dtype=np.float64)
    final_target = np.asarray(rows[-1]["target"], dtype=np.float64)
    report["final_xy_error_mm"] = float(np.linalg.norm(final_gt[:2, 3] - final_target[:2]) * 1000.0)
    print(
        "VIDEO", video_path,
        "frames", len(frames),
        "center_mm", round(report.get("center_mm", -1), 2),
        "axis_deg", round(report.get("axis_deg", -1), 2),
        "xy_mm", round(report["final_xy_error_mm"], 1),
        flush=True,
    )
    return report


def run_one(task):
    """Simulate one task, close the simulator, then estimate and draw.

    仿真一个任务，关掉仿真，再估计并画出来。
    """
    folder = os.path.join(OUT_DIR, task["key"])
    os.makedirs(folder, exist_ok=True)
    mesh = load_source_mesh(task["source"])
    mesh_path = os.path.join(folder, "mesh.ply")
    mesh.export(mesh_path)
    diameter_m = known.mesh_diameter_m(mesh)
    print("TASK", task["key"], "diameter_m", round(diameter_m, 3), flush=True)
    env = make_env(task["env_id"])
    try:
        env.reset(seed=0, options={"episode_id": 0})
        base = env.unwrapped
        source = base.objs[base.source_obj_name]
        target = base.objs[base.target_obj_name]
        camera = add_overview(
            env,
            task.get("eye", OVERVIEW_EYE_M),
            task.get("target", OVERVIEW_TARGET_M),
        )
        # Frame 0 is one color mask and one estimate. The arm then uses that pose.
        # 第 0 帧是一块颜色 mask 和一次估计。手臂接着用这个位姿。
        base.scene.update_render()
        init_jobs = []
        init_row = record_frame(
            camera, folder, 0, source, target, mesh_path, diameter_m, init_jobs, task["key"],
        )
        if not init_jobs:
            raise RuntimeError("frame 0 mask")
        init_path = os.path.join(folder, "init_jobs.json")
        with open(init_path, "w", encoding="utf-8") as stream:
            json.dump(init_jobs, stream)
        init_poses = run_sequence(init_path)
        pose_cam = init_poses.get("f00000")
        if pose_cam is None:
            raise RuntimeError("frame 0 estimate")
        world = known.world_pose(np.asarray(init_row["render_extrinsic"], dtype=np.float64), pose_cam)
        grasp_center = world[:3, 3].copy()
        grasp_closing = flat_closing(world[:3, :3])
        print(
            "GRASP", task["key"],
            "t_m", np.round(grasp_center, 4).tolist(),
            flush=True,
        )
        rows, jobs = carry(
            env, camera, folder, source, target, mesh_path, diameter_m,
            grasp_center, grasp_closing, task["key"],
        )
    finally:
        env.close()
    with open(os.path.join(folder, "rows.json"), "w", encoding="utf-8") as stream:
        json.dump(rows, stream)
    jobs_path = os.path.join(folder, "jobs.json")
    with open(jobs_path, "w", encoding="utf-8") as stream:
        json.dump(jobs, stream)
    print("JOBS", len(jobs), "frames", len(rows), flush=True)
    poses = run_sequence(jobs_path) if jobs else {}
    report = compose_video(task, rows, poses, mesh, "zh")
    compose_video(task, rows, poses, mesh, "en")
    report_path = os.path.join(OUT_DIR, "report.json")
    previous = []
    if os.path.isfile(report_path):
        with open(report_path, "r", encoding="utf-8") as stream:
            previous = json.load(stream)
    previous = [item for item in previous if item.get("key") != task["key"]]
    previous.append(report)
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(previous, stream, indent=2)
    print("WROTE", report_path, flush=True)
    return report


def load_poses(folder):
    """Read the saved object-to-camera poses. No new estimate.

    读已经存下的物体到相机位姿。不再估计。
    """
    pose_path = os.path.join(folder, "jobs_poses.json")
    with open(pose_path, "r", encoding="utf-8") as stream:
        rows = json.load(stream)
    poses = {}
    for row in rows:
        if "pose" not in row:
            continue
        poses[row["id"]] = np.asarray(row["pose"], dtype=np.float64)
    return poses


def redraw_saved(task):
    """Write the Chinese and English title bars from saved frames. No simulator.

    用已经存下的画面重写中文和英文标题。不开仿真。
    """

    folder = os.path.join(OUT_DIR, task["key"])
    with open(os.path.join(folder, "rows.json"), "r", encoding="utf-8") as stream:
        rows = json.load(stream)
    poses = load_poses(folder)
    mesh = trimesh.load(os.path.join(folder, "mesh.ply"), force="mesh", process=False)
    compose_video(task, rows, poses, mesh, "zh")
    report = compose_video(task, rows, poses, mesh, "en")
    print("REDREW", task["key"], report["frames"], flush=True)


if __name__ == "__main__":
    recipe.visualize_path = OUT_DIR
    # The worker above loads WAPREstimator once and batches objects per RGB-D frame.
    # 上方子进程入口仅载入一次 WAPREstimator，按 RGB-D 帧批量处理物体。
    if len(sys.argv) >= 3 and sys.argv[1] == "sequence":
        estimate_sequence(sys.argv[2])
        sys.exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == "redraw":
        for task in TASKS:
            redraw_saved(task)
        sys.exit(0)
    selected = TASKS
    if len(sys.argv) >= 2:
        wanted = sys.argv[1]
        selected = tuple(task for task in TASKS if task["key"] == wanted)
        if not selected:
            raise RuntimeError("unknown task %s" % wanted)
    for task in selected:
        # 1. Select the task, load its CAD in meters, and start WidowX joint control.
        # 1. 选择任务，载入以米为单位的 CAD，并启动 WidowX 关节位置控制。
        folder = os.path.join(OUT_DIR, task["key"])
        os.makedirs(folder, exist_ok=True)
        mesh = load_source_mesh(task["source"])
        mesh_path = os.path.join(folder, "mesh.ply")
        mesh.export(mesh_path)
        diameter_m = known.mesh_diameter_m(mesh)
        env = make_env(task["env_id"])
        planner = None
        try:
            env.reset(seed=0, options={"episode_id": 0})
            base = env.unwrapped
            source = base.objs[base.source_obj_name]
            target = base.objs[base.target_obj_name]
            camera = add_overview(env, task.get("eye", OVERVIEW_EYE_M), task.get("target", OVERVIEW_TARGET_M))
            # 2. Obtain RGB-D/K and an observed color mask, then estimate before moving.
            # record_frame saves simulator truth only as report metadata.
            # 2. 获取 RGB-D、内参 K 与观测颜色掩码，在运动前估计位姿。
            # record_frame 仅将仿真真值另存为报告元数据。
            base.scene.update_render()
            init_jobs = []
            init_row = record_frame(camera, folder, 0, source, target, mesh_path, diameter_m, init_jobs, task["key"])
            if not init_jobs:
                raise RuntimeError("frame 0 mask")
            init_path = os.path.join(folder, "init_jobs.json")
            with open(init_path, "w", encoding="utf-8") as stream:
                json.dump(init_jobs, stream)
            # run_sequence launches this file's worker: estimate_many_categories_many_instances.
            # Vulkan simulation and OpenGL pose rendering run in separate processes.
            # run_sequence 启动本文件的子进程，调用 estimate_many_categories_many_instances。
            # Vulkan 仿真与 OpenGL 位姿渲染分别运行在独立进程中。
            init_poses = run_sequence(init_path)
            pose_cam = init_poses.get("f00000")
            if pose_cam is None:
                raise RuntimeError("frame 0 estimate")
            # 3. T_world_object = inverse(T_camera_world) @ T_camera_object, meters.
            # The object's estimated thin axis determines the gripper closing direction.
            # 3. 世界物体位姿 = 世界到相机外参的逆 × 相机物体位姿，单位米。
            # 估计出的物体薄轴决定夹爪合拢方向。
            world = known.world_pose(np.asarray(init_row["render_extrinsic"], dtype=np.float64), pose_cam)
            grasp_center = world[:3, 3].copy()
            grasp_closing = flat_closing(world[:3, :3])
            print("GRASP", task["key"], "t_m", grasp_center.tolist(), flush=True)
            planner = make_planner(env)
            rows, jobs = [], []
            frame_index = 0
            original_step = env.step

            def step_and_record(action):
                """Advance the arm/physics and save one RGB-D observation per step.

                每个手臂/物理步保存一帧 RGB-D；录像期间不逐物体运行推理。
                """
                global frame_index
                outcome = original_step(action)
                rows.append(record_frame(camera, folder, frame_index, source, target,
                                         mesh_path, diameter_m, jobs, task["key"]))
                frame_index += 1
                return outcome

            env.step = step_and_record
            # 4. Open, approach from above, descend to the estimated center, and close.
            # 4. 张开夹爪，从上方接近，下降到估计中心后合拢。
            planner.open_gripper()
            center = np.asarray(grasp_center, dtype=np.float64).reshape(3).copy()
            closing = np.asarray(grasp_closing, dtype=np.float64).reshape(3).copy()
            known.move_arm(planner, top_grasp(center + [0.0, 0.0, PREGRASP_M], closing))
            known.move_arm(planner, top_grasp(center, closing))
            planner.close_gripper(t=12)
            # Lift first. The task supplies the destination; its pose is not the source grasp.
            # 先抬升。放置目标由任务给定，不用源物体真值作为抓取输入。
            known.move_arm(planner, top_grasp(center + [0.0, 0.0, LIFT_M], closing))
            goal = target.pose.p[0].detach().cpu().numpy().copy()
            goal[2] = float(goal[2]) + PLACE_CLEAR_M + LIFT_M
            known.move_arm(planner, top_grasp(goal, closing))
            goal[2] = float(target.pose.p[0, 2].detach().cpu().numpy()) + PLACE_CLEAR_M
            known.move_arm(planner, top_grasp(goal, closing))
            # Release at the target and retreat vertically with the fingers open.
            # 在目标处松开物体，保持夹爪张开并竖直撤回。
            planner.open_gripper(t=8)
            goal[2] += LIFT_M
            known.move_arm(planner, top_grasp(goal, closing))
        finally:
            if planner is not None:
                planner.close()
            env.close()
        # 5. Save observations, estimate each recorded frame in batches, then render reports.
        # Only this post-hoc report compares estimates with simulator annotations.
        # 5. 保存观测，按帧批量估计录像，然后绘制报告。
        # 仅此事后报告将估计与仿真标注对照。
        with open(os.path.join(folder, "rows.json"), "w", encoding="utf-8") as stream:
            json.dump(rows, stream)
        jobs_path = os.path.join(folder, "jobs.json")
        with open(jobs_path, "w", encoding="utf-8") as stream:
            json.dump(jobs, stream)
        poses = run_sequence(jobs_path) if jobs else {}
        report = compose_video(task, rows, poses, mesh, "zh")
        compose_video(task, rows, poses, mesh, "en")
        report_path = os.path.join(OUT_DIR, "report.json")
        previous = []
        if os.path.isfile(report_path):
            with open(report_path, "r", encoding="utf-8") as stream:
                previous = json.load(stream)
        previous = [item for item in previous if item.get("key") != task["key"]]
        previous.append(report)
        with open(report_path, "w", encoding="utf-8") as stream:
            json.dump(previous, stream, indent=2)
        print("WROTE", report_path, flush=True)
