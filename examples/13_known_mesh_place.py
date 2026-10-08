# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 13. Previous: examples/12_cross_scene_pose.py. Next: examples/14_bridge_tasks.py
# 示例 13。上一例：examples/12_cross_scene_pose.py。下一例：examples/14_bridge_tasks.py
# Known-CAD RGB-D estimation, Panda grasping, and placement with measured object-to-tool correction.
# 已知 CAD 的 RGB-D 位姿估计、Panda 抓取，以及按测得物体到工具变换修正放置。
# The hand camera is mounted on the gripper, so it follows the object while the arm moves.
# 手上的相机装在夹爪上，手臂带着物体走的时候，相机跟着走。
# Pose estimation runs in a child process so it does not open a second OpenGL context here.
# 位姿估计放在子进程里，避免这个进程再开一个 OpenGL 上下文。
# python examples/13_known_mesh_place.py
# 运行：python examples/13_known_mesh_place.py
# python examples/13_known_mesh_place.py track
# 跟踪录像：python examples/13_known_mesh_place.py track
# A child call is: python examples/13_known_mesh_place.py estimate <snapshot.json>
# 子进程：python examples/13_known_mesh_place.py estimate <snapshot.json>
"""One Panda pick-and-place. The bottle and the box meshes are given.

一次 Panda 抓取和摆放。瓶子和盒子的网格是给定的。

The table camera estimates both poses before the arm moves. After the fingers
close, the hand camera estimates the bottle again. That pose, expressed in the
tool frame, is the grasp the arm actually got. The place command aims the tool
so the bottle, not the fingers, stands upright at the target. A second run
places with the planned grasp and ignores that measurement.
桌上的相机在手臂动之前估计两个位姿。手指合上之后，手上的相机再估计一次瓶子。
这个位姿变到工具坐标系里，就是这次真正抓住的偏差。摆放时让瓶子直立落到目标上，
而不是只让手指走到计划的位置。另一次运行按计划的抓取去放，不用这次测量。
"""

import json
import os
import subprocess
import sys
import time

import cv2
import trimesh
import numpy as np
import torch


# Resolve the package relative to this public recipe, independently of cwd.
# 从本公开示例定位包目录，不依赖启动目录。
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.dirname(EXAMPLES_DIR)
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)

# Panda is the arm that already finishes this carry. The meshes are meters.
# Panda 是已经能把这次搬运做完的那只手臂。网格单位是米。
ROBOT_UID = "panda"
DEVICE = "cuda:0"
from wapr.resources import outputs_dir
from wapr import recipe
OUT_DIR = outputs_dir(__file__)
BOTTLE_RADIUS_M = 0.025
BOTTLE_HEIGHT_M = 0.14
BOX_EXTENTS_M = np.array([0.090, 0.060, 0.060], dtype=np.float64)
# Same table layout as the language demo. Centers are meters, world frame.
# 和语言那次相同的桌面摆法。中心是米，世界坐标系。
BOTTLE_CENTER_M = np.array([0.02, -0.14, 0.07], dtype=np.float64)
BOX_CENTER_M = np.array([0.00, 0.04, 0.03], dtype=np.float64)
TABLE_CAMERA_WIDTH = 640
TABLE_CAMERA_HEIGHT = 480
HAND_CAMERA_WIDTH = 640
HAND_CAMERA_HEIGHT = 480
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 720
# Local pose of the camera on panda_hand. It looks into the fingers.
# 相机在 panda_hand 上的局部位姿。镜头对着指缝。
HAND_CAM_P = (0.0464982, -0.0200011, 0.0360011)
HAND_CAM_Q = (0.0, 0.70710678, 0.0, 0.70710678)
PREGRASP_M = 0.08
# Drawn side gripper, meters. The palm sits this far back from the bottle center.
# The finger gap is just wider than the bottle diameter.
# 画出来的侧面夹爪，米。手掌在瓶子中心后方这么远。指缝比瓶子直径略宽。
GRASP_PALM_BACK_M = 0.032
GRASP_FINGER_GAP_M = 0.056
GRASP_RGB = (30, 120, 255)
LIFT_M = 0.12
PLACE_GAP_M = 0.03
# How many held joint targets to step after the gripper opens, so the bottle can fall.
# 松手之后再保持关节目标走这么多步，瓶子如果会倒，就倒在这几步里。
SETTLE_STEPS = 40
# Every recorded frame is estimated again. 1 means the outline is redrawn from a new measurement each frame.
# 每一帧录像都重新估计。1 表示轮廓每一帧都来自一次新的测量。
TRACK_EVERY = 1
# A frame is sent to pose estimation only when its color mask is at least this large.
# 颜色 mask 至少这么多像素，这一帧才送去估计位姿。
TRACK_MIN_MASK_PX = 200
# How long the drawn view volume is, meters. The opening angle comes from K.
# The simulator far clip is 10 m, and that face would cover the third-person picture.
# 画出来的视野有多长，米。张角来自 K。
# 仿真的远裁剪是 10 米，那个面会铺满第三人称画面。
TABLE_VIEW_NEAR_M = 0.12
TABLE_VIEW_FAR_M = 0.90
WRIST_VIEW_NEAR_M = 0.05
WRIST_VIEW_FAR_M = 0.45
# Dash and gap of the view-volume edges, pixels on the third-person picture.
# 视野棱的虚线实段和空段，第三人称画面上的像素。
VIEW_DASH_PX = 12
VIEW_GAP_PX = 8
# The wrist camera's home pose looks straight down at the box top, so the box
# reads as a flat card. The video starts once the camera has tilted this far
# from straight down, degrees, and a side face is visible.
# 腕部相机的初始位姿正俯视盒子顶面，盒子在图像中近似为一个平面。
# 相机偏离正下方达到这个角度、侧面露出来之后，视频才开始，度。
WRIST_OPEN_MIN_TILT_DEG = 30.0
# The box stays on the table. A center jump larger than this, meters, is a bad
# estimate, and the outline keeps the previous pose.
# 盒子停在桌上。中心位移若超过该距离（单位：米），则将该次估计视为无效，轮廓沿用上一帧。
BOX_HOLD_JUMP_M = 0.03


def mesh_diameter_m(mesh):
    """
    # Compute the bounding-sphere diameter in meters.

    ---

    # 计算包围球直径，单位米。

"""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    center = vertices.mean(axis=0)
    return float(2.0 * np.linalg.norm(vertices - center, axis=1).max())


def write_known_meshes(out_dir):
    """
    # Write the bottle and the box that the arm is given.

        Both are centered.

        The bottle axis is +Z.

        The box matches the simulated box, XYZ.

    ## Args

        - out_dir: the directory created for bottle.ply and box.ply. The signature has no default.

    ## Returns

        - The return has keys bottle and box.
        - Each value has path, mesh, and diameter_m.
        - diameter_m: the bounding-sphere diameter in meters.

    ---

    # 写出交给手臂的瓶子和盒子。

        两个网格都放在自己的中心。

        瓶子的轴是 +Z。

        盒子和仿真里的盒子一致，XYZ。

    ## 参数

        - out_dir: 创建出来存放 bottle.ply 和 box.ply 的目录。签名里没有默认值。

    ## 返回

        - 返回值的键是 bottle 和 box。
        - 每个值含有 path、mesh 和 diameter_m。
        - diameter_m: 包围球直径，单位米。

"""

    os.makedirs(out_dir, exist_ok=True)
    bottle = trimesh.creation.cylinder(
        radius=BOTTLE_RADIUS_M,
        height=BOTTLE_HEIGHT_M,
        sections=48,
    )
    box = trimesh.creation.box(extents=BOX_EXTENTS_M)
    bottle_path = os.path.join(out_dir, "bottle.ply")
    box_path = os.path.join(out_dir, "box.ply")
    bottle.export(bottle_path)
    box.export(box_path)
    return {
        "bottle": {"path": bottle_path, "mesh": bottle, "diameter_m": mesh_diameter_m(bottle)},
        "box": {"path": box_path, "mesh": box, "diameter_m": mesh_diameter_m(box)},
    }


def make_env():
    """
    # ManiSkill table, one Panda, a yellow bottle, a red box, and a hand camera.

    ---

    # ManiSkill 桌子、一只 Panda、黄色瓶子、红色盒子，以及手上的相机。

"""
    # Prepare simulation dependencies only when creating the environment.
    # 仅在创建仿真环境时准备机器人可选依赖。
    from wapr.bootstrap import ensure_optional
    ensure_optional("robot")
    import gymnasium as gym
    import sapien
    from transforms3d.euler import euler2quat

    import mani_skill.envs  # noqa: F401
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.building import actors
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.scene_builder.table import TableSceneBuilder

    bottle_q = euler2quat(0.0, np.pi / 2.0, 0.0)

    class KnownMeshPlaceEnv(BaseEnv):
        """
        # ManiSkill table environment with one Panda, a yellow bottle, a red box, and a hand camera.

        ## Returns

            - SUPPORTED_ROBOTS: the list ["panda"].

        ---

        # ManiSkill 桌子环境，一只 Panda、黄色瓶子、红色盒子，以及手上的相机。

        ## 返回

            - SUPPORTED_ROBOTS: 列表 ["panda"]。

"""
        SUPPORTED_ROBOTS = ["panda"]

        def __init__(self, *args, robot_uids="panda", **kwargs):
            """
            # Set robot_init_qpos_noise to 0.0 and construct the ManiSkill base environment.

            ## Args

                - args and kwargs: forwarded to BaseEnv.__init__.
                - robot_uids: to the string panda and is forwarded with that keyword.

            ## Returns

                - Returns None.

            ---

            # 把 robot_init_qpos_noise 设为 0.0，并构造 ManiSkill 基类环境。

            ## 参数

                - args 和 kwargs 转给 BaseEnv.__init__。
                - robot_uids 默认字符串 panda，并按这个关键字转发。

            ## 返回

                - 返回 None。

"""
            self.robot_init_qpos_noise = 0.0
            super().__init__(*args, robot_uids=robot_uids, **kwargs)

        @property
        def _default_sensor_configs(self):
            """
            # Return a list of two CameraConfig values: table_camera and hand_camera.

                table_camera looks from [0.42, 0.0, 0.38] m toward [0.0, 0.0, 0.05] m, at TABLE_CAMERA_WIDTH by TABLE_CAMERA_HEIGHT.

                It passes 1.0, 0.01, 10, and shader_pack minimal.

                hand_camera uses HAND_CAM_P and HAND_CAM_Q, mounted on panda_hand, at HAND_CAMERA_WIDTH by HAND_CAMERA_HEIGHT.

                It passes 1.0, 0.01, 10, and shader_pack minimal.

            ## Args

                - There is no argument.

            ---

            # 返回含两个 CameraConfig 的列表：table_camera 和 hand_camera。

                table_camera 从 [0.42, 0.0, 0.38] 米看向 [0.0, 0.0, 0.05] 米，宽高是 TABLE_CAMERA_WIDTH 和 TABLE_CAMERA_HEIGHT。

                它传入 1.0、0.01、10，shader_pack 是 minimal。

                hand_camera 用 HAND_CAM_P 和 HAND_CAM_Q，装在 panda_hand 上，宽高是 HAND_CAMERA_WIDTH 和 HAND_CAMERA_HEIGHT。

                它传入 1.0、0.01、10，shader_pack 是 minimal。

            ## 参数

                - 没有参数。

"""
            table_pose = sapien_utils.look_at(eye=[0.42, 0.0, 0.38], target=[0.0, 0.0, 0.05])
            hand_pose = sapien.Pose(p=list(HAND_CAM_P), q=list(HAND_CAM_Q))
            return [
                CameraConfig(
                    "table_camera", table_pose,
                    TABLE_CAMERA_WIDTH, TABLE_CAMERA_HEIGHT, 1.0, 0.01, 10,
                    shader_pack="minimal",
                ),
                CameraConfig(
                    "hand_camera", hand_pose,
                    HAND_CAMERA_WIDTH, HAND_CAMERA_HEIGHT, 1.0, 0.01, 10,
                    mount=self.agent.robot.links_map["panda_hand"],
                    shader_pack="minimal",
                ),
            ]

        @property
        def _default_human_render_camera_configs(self):
            """
            # Return one CameraConfig named render_camera for the third-person frames.

                The camera looks from [0.85, 0.75, 0.55] m toward [0.0, 0.0, 0.12] m, at VIDEO_WIDTH by VIDEO_HEIGHT, and passes 1.0, 0.01, and 10.

            ## Args

                - There is no argument.

            ---

            # 返回名为 render_camera 的一份 CameraConfig，给第三人称画面用。

                相机从 [0.85, 0.75, 0.55] 米看向 [0.0, 0.0, 0.12] 米，宽高是 VIDEO_WIDTH 和 VIDEO_HEIGHT，并传入 1.0、0.01 和 10。

            ## 参数

                - 没有参数。

"""
            pose = sapien_utils.look_at(eye=[0.85, 0.75, 0.55], target=[0.0, 0.0, 0.12])
            return CameraConfig("render_camera", pose, VIDEO_WIDTH, VIDEO_HEIGHT, 1.0, 0.01, 10)

        def _load_agent(self, options):
            """
            # Place the Panda base at [-0.615, 0, 0] m and forward options.

            ## Args

                - options: forwarded to BaseEnv._load_agent. This body does not read it. The signature has no default.

            ## Returns

                - Returns None.

            ---

            # 把 Panda 底座放在 [-0.615, 0, 0] 米，并转发 options。

            ## 参数

                - options 转给 BaseEnv._load_agent。
                - 这里不读它。
                - 签名里没有默认值。

            ## 返回

                - 返回 None。

"""
            super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

        def _load_scene(self, options):
            """
            # Build the table, the yellow bottle, and the red box.

                The table uses TableSceneBuilder with robot_init_qpos_noise 0.0.

                The bottle is a cylinder of radius BOTTLE_RADIUS_M and half length 0.5 * BOTTLE_HEIGHT_M, color [1.0, 0.85, 0.05, 1.0], at BOTTLE_CENTER_M with bottle_q.

                bottle_q is a Y rotation of pi/2, which stands the SAPIEN cylinder up.

                Lengths are meters.

                The box half sizes are 0.5 * BOX_EXTENTS_M, color [0.8, 0.12, 0.08, 1.0], at BOX_CENTER_M.

                Lengths are meters.

            ## Args

                - options: accepted and this body does not read it. The signature has no default.

            ## Returns

                - Returns None.

            ---

            # 建立桌子、黄色瓶子和红色盒子。

                桌子用 TableSceneBuilder，robot_init_qpos_noise 为 0.0。

                瓶子是圆柱，半径 BOTTLE_RADIUS_M，半长 0.5 * BOTTLE_HEIGHT_M，颜色 [1.0, 0.85, 0.05, 1.0]，位姿是 BOTTLE_CENTER_M 和 bottle_q。

                bottle_q 是绕 Y 转 pi/2，把 SAPIEN 圆柱立起来。

                长度单位是米。

                盒子半尺寸是 0.5 * BOX_EXTENTS_M，颜色 [0.8, 0.12, 0.08, 1.0]，位姿是 BOX_CENTER_M。

                长度单位是米。

            ## 参数

                - options 会被接收，这里不读它。
                - 签名里没有默认值。

            ## 返回

                - 返回 None。

"""
            self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.0)
            self.table_scene.build()
            # SAPIEN cylinders lie along X. This Y rotation stands that axis up.
            # SAPIEN 的圆柱沿 X。绕 Y 这一转把轴立起来。
            self.bottle = actors.build_cylinder(
                self.scene,
                radius=BOTTLE_RADIUS_M,
                half_length=0.5 * BOTTLE_HEIGHT_M,
                color=[1.0, 0.85, 0.05, 1.0],
                name="bottle",
                initial_pose=sapien.Pose(p=BOTTLE_CENTER_M.tolist(), q=bottle_q),
            )
            self.box = actors.build_box(
                self.scene,
                half_sizes=(0.5 * BOX_EXTENTS_M).tolist(),
                color=[0.8, 0.12, 0.08, 1.0],
                name="box",
                initial_pose=sapien.Pose(p=BOX_CENTER_M.tolist()),
            )

        def _initialize_episode(self, env_idx, options):
            """
            # Initialize the table scene for one env_idx.

            ## Args

                - env_idx: passed to table_scene.initialize. The signature has no default.
                - options: accepted and this body does not read it. The signature has no default.

            ## Returns

                - Returns None.

            ---

            # 按一个 env_idx 初始化桌面场景。

            ## 参数

                - env_idx 传给 table_scene.initialize。
                - 签名里没有默认值。
                - options 会被接收，这里不读它。
                - 签名里没有默认值。

            ## 返回

                - 返回 None。

"""
            self.table_scene.initialize(env_idx)

        def evaluate(self):
            """
            # Report success without a task score.

                The place error is measured after the motion.

            ---

            # 报告成功，不打任务分。

                摆放误差在动作结束之后量。

"""
            return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

        def compute_dense_reward(self, obs, action, info):
            """
            # Return a zero dense reward.

                This script does not train.

            ---

            # 返回零稠密奖励。

                这个脚本不训练。

"""
            return torch.zeros(self.num_envs, device=self.device)

        def compute_normalized_dense_reward(self, obs, action, info):
            """
            # Return the same zero reward.

            ---

            # 返回同样的零奖励。

"""
            return self.compute_dense_reward(obs, action, info)

    if "KnownMeshPlace-v1" not in gym.registry:
        register_env("KnownMeshPlace-v1", max_episode_steps=2000)(KnownMeshPlaceEnv)
    return gym.make(
        "KnownMeshPlace-v1",
        robot_uids=ROBOT_UID,
        obs_mode="rgb+position",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="cpu",
    )


def as_numpy_matrix(value):
    """
    # Convert one pose to a 4×4 float64 matrix.

        A batch keeps the first item.

    ---

    # 把一个位姿转成 4×4 float64 矩阵。

        如果带 batch，取第一个。

"""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim == 3:
        matrix = matrix[0]
    return matrix


def actor_matrix(actor):
    """
    # Read the actor pose as a 4×4 matrix.

    ---

    # 读取 actor 的 4×4 位姿。

"""
    return as_numpy_matrix(actor.pose.to_transformation_matrix())


def bottle_mesh_matrix(actor_pose):
    """
    # Read the world pose of one bottle mesh.

        The mesh +Z lies on the actor +X.

    ---

    # 读取一个瓶子网格的世界位姿。

        网格的 +Z 落在 actor 的 +X 上。

"""
    align = np.eye(4, dtype=np.float64)
    align[:3, :3] = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)
    return actor_pose @ align


def color_mask(rgb, name):
    """
    # Select pixels of one solid color.

        The simulated bottle is yellow and the box is red.

    ---

    # 选出一种纯色的像素。

        仿真瓶子是黄的，盒子是红的。

"""
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    # Lit bottle is yellow, the body is olive. The table is orange, so green stays near red.
    # 受光的瓶子是黄的，瓶身偏橄榄。桌面是橙色，所以绿色要接近红色才算瓶子。
    if name == "bottle":
        return (blue < 80) & (green > 100) & (green + 30 > red) & (red > 120)
    # The box is redder than the table: red is about twice the green.
    # 盒子比桌面更红：红色大约是绿色的两倍。
    if name == "box":
        return (red > 100) & (green < 110) & (blue < 90) & (red > (1.9 * green))
    raise RuntimeError("color " + name)


def extrinsic_and_K(params):
    """
    # Read the world-to-camera matrix and K.

        A 3x4 extrinsic is padded.

    ---

    # 读取世界到相机的矩阵和 K。

        3x4 的外参补成 4x4。

"""
    extrinsic = as_numpy_matrix(params["extrinsic_cv"])
    if extrinsic.shape == (3, 4):
        full = np.eye(4, dtype=np.float64)
        full[:3] = extrinsic
        extrinsic = full
    K = as_numpy_matrix(params["intrinsic_cv"])[:3, :3].astype(np.float32)
    return extrinsic, K


def camera_frame(env, camera_name, sensor_data):
    """
    # One sensor already read: RGB, depth meters, K, world-to-camera.

    ---

    # 传感器已经读过：RGB、米为单位的深度、K、世界到相机。

"""
    camera = sensor_data[camera_name]
    rgb = camera["rgb"][0].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb[..., :3] * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb = np.ascontiguousarray(rgb[..., :3])
    # minimal shader stores position as int16 millimeters in the OpenGL camera.
    # minimal 着色器把位置存成 OpenGL 相机里的 int16 毫米。
    position_m = camera["position"][0].detach().cpu().numpy()[..., :3].astype(np.float64) / 1000.0
    depth_m = (-position_m[..., 2]).astype(np.float32)
    extrinsic, K = extrinsic_and_K(env.unwrapped._sensors[camera_name].get_params())
    return rgb, depth_m, K, extrinsic


def capture_camera(env, camera_name):
    """
    # Capture RGB, depth, K, and the camera pose.

        OpenCV camera, y down.

    ---

    # 采集 RGB、深度、K 和相机位姿。

        OpenCV 相机，y 向下。

"""
    sensor_data = env.unwrapped._get_obs_sensor_data()
    return camera_frame(env, camera_name, sensor_data)


def mask_center(mask):
    """
    # Compute the mean pixel of a mask.

        No caller in this file.

    ---

    # 计算 mask 像素的平均位置。

    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


def project_point(pose_4x4, K, point_m):
    """
    # Project one object point to a pixel.

    ## Args

        - pose_4x4: object-to-camera.

    ---

    # 把物体上的一个点投影成像素。

    ## 参数

        - pose_4x4: 物体到相机。

"""
    point = np.asarray(point_m, dtype=np.float64)
    camera = pose_4x4[:3, :3] @ point + pose_4x4[:3, 3]
    if camera[2] <= 1e-6:
        return None
    pixel = K @ camera
    return pixel[:2] / pixel[2]


def save_snapshot(path, rgb, depth_m, K, objects):
    """
    # Save one camera frame and its meshes.

        The child process reads this.

    ---

    # 保存一帧相机和对应网格。

        子进程读这份文件。

"""
    import imageio.v2 as imageio

    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    stem = os.path.splitext(path)[0]
    imageio.imwrite(stem + "_rgb.png", rgb)
    np.save(stem + "_depth.npy", depth_m)
    payload = {
        "rgb": stem + "_rgb.png",
        "depth": stem + "_depth.npy",
        "K": np.asarray(K, dtype=np.float64).tolist(),
        "objects": [],
    }
    for item in objects:
        mask_path = stem + "_mask_%s.png" % item["name"]
        imageio.imwrite(mask_path, (item["mask"].astype(np.uint8) * 255))
        payload["objects"].append({
            "name": item["name"],
            "mask": mask_path,
            "mesh": item["mesh_path"],
            "diameter_m": float(item["diameter_m"]),
        })
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return path


def estimate_snapshot(path):
    """
    # Estimate one snapshot in a child process.

        Load the pose networks, write object-to-camera poses, exit.

    ---

    # 在子进程里估计一张快照。

        载入位姿网络，写出物体到相机的位姿，然后退出。

"""

    from wapr import WAPREstimator

    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    rgb = cv2.cvtColor(cv2.imread(payload["rgb"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depth_m = np.load(payload["depth"]).astype(np.float32)
    K = np.array(payload["K"], dtype=np.float32)
    estimator = WAPREstimator(device=DEVICE)
    poses = {}
    scores = {}
    instances = []
    for item in payload["objects"]:
        mask = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE)
        mesh = trimesh.load(item["mesh"], force="mesh", process=False)
        instances.append({"mesh": mesh, "diameter_m": float(item["diameter_m"]), "mask": mask})
    prepared = estimator.prepare_meshes([item["mesh"] for item in instances])
    for item, mesh in zip(instances, prepared):
        item["mesh"] = mesh
    mesh_setup_seconds = estimator.last_mesh_setup_seconds
    warmup_seconds = estimator.warmup_pose(rgb, depth_m, K, instances)
    torch.cuda.synchronize(DEVICE)
    pose_started = time.perf_counter()
    outputs = estimator.estimate_many_categories_many_instances(rgb, depth_m, K, instances)
    torch.cuda.synchronize(DEVICE)
    pose_hot_seconds = time.perf_counter() - pose_started
    print("POSE_TIMING", {"model_setup_seconds": estimator.model_setup_seconds,
          "mesh_setup_seconds": mesh_setup_seconds, "warmup_seconds": warmup_seconds,
          "pose_hot_seconds": pose_hot_seconds}, flush=True)
    for item, out in zip(payload["objects"], outputs):
        poses[item["name"]] = np.asarray(out["pose_4x4"], dtype=np.float32)
        scores[item["name"]] = float(out["score_6d"])
        print(
            "ESTIMATE", item["name"],
            "t_m", np.round(poses[item["name"]][:3, 3], 4).tolist(),
            "score_6d", round(scores[item["name"]], 4),
            flush=True,
        )
    out_path = os.path.splitext(path)[0] + "_poses.npz"
    np.savez(
        out_path,
        **{name + "_pose": poses[name] for name in poses},
        **{name + "_score": np.array(scores[name]) for name in scores},
    )
    print("ESTIMATE_WROTE", out_path, flush=True)


def estimate_sequence(path):
    """
    # Estimate a frame list in a child process.

        One pose network load, then every frame in the list.

    ---

    # 在子进程里估计一串帧。

        位姿网络只载入一次，然后估计列表里的每一帧。

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
        K = np.array(first["K"], dtype=np.float32).reshape(3, 3)
        instances = []
        valid_jobs = []
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
        # One bad frame stays in the list as a miss. The rest of the video still runs.
        # 某一帧失败就记成 miss，后面的帧继续估计。
        try:
            shape = (tuple(depth_m.shape), tuple(estimator.renderer.cached_mesh_id(item["mesh"]) for item in instances))
            if instances and shape not in warmed_shapes:
                warmup_seconds = estimator.warmup_pose(rgb, depth_m, K, instances)
                warmed_shapes.add(shape)
                print("POSE_WARMUP_SECONDS", warmup_seconds, flush=True)
            torch.cuda.synchronize(DEVICE)
            pose_started = time.perf_counter()
            outputs = estimator.estimate_many_categories_many_instances(rgb, depth_m, K, instances)
            torch.cuda.synchronize(DEVICE)
            pose_hot_seconds = time.perf_counter() - pose_started
        except Exception as error:
            for job in valid_jobs:
                print("ESTIMATE_MISS", job["id"], type(error).__name__, error, flush=True)
                written.append({"id": job["id"], "error": str(error)})
            continue
        for job, out in zip(valid_jobs, outputs):
            pose = np.asarray(out["pose_4x4"], dtype=np.float64)
            written.append({"id": job["id"], "pose": pose.tolist(), "pose_hot_seconds": pose_hot_seconds,
                            "pose_time_scope": "warm prepared-mesh GPU frame batch"})
            print("ESTIMATE", job["id"], "t_m", np.round(pose[:3, 3], 4).tolist(),
                  "score_6d", round(float(out["score_6d"]), 4), flush=True)
    out_path = os.path.splitext(path)[0] + "_poses.json"
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(written, stream)
    print("ESTIMATE_WROTE", out_path, flush=True)


def run_estimate(snapshot_path):
    """
    # Run the child and return {name: pose_4x4} in the camera frame.

    ---

    # 跑子进程，返回 {名字: pose_4x4}，相机坐标系。

"""
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "estimate", snapshot_path],
        cwd=RELEASE_DIR,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("estimate failed %s" % proc.returncode)
    pose_path = os.path.splitext(snapshot_path)[0] + "_poses.npz"
    data = np.load(pose_path)
    names = []
    for key in data.files:
        if key.endswith("_pose"):
            names.append(key[: -len("_pose")])
    return {name: np.asarray(data[name + "_pose"], dtype=np.float64) for name in names}


def run_sequence(path):
    """
    # Run the child over a list of frames.

    ## Returns

        - Keys: job ids, values are object-to-camera.

    ---

    # 让子进程估计一整段帧。

    ## 返回

        - 键是任务编号，值是物体到相机。

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


def world_pose(extrinsic_cv, pose_cam):
    """
    # Compose the object-to-world pose.

    ---

    # 合成物体到世界的位姿。

"""
    return np.linalg.inv(extrinsic_cv) @ pose_cam


def axis_angle_deg(a, b):
    """
    # Measure the angle between two directions.

        A flip counts as the same axis.

    ---

    # 计算两个方向的夹角。

        反向算同一根轴。

"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-8)
    b = b / max(np.linalg.norm(b), 1e-8)
    cosine = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def rotation_angle_deg(a, b):
    """
    # Measure the angle between two rotations.

    ---

    # 计算两个旋转之间的夹角。

"""
    cosine = float(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def pose_error(estimated_world, true_world):
    """
    # Center error in millimeters, and the mesh-+Z axis error in degrees.

    ---

    # 中心误差，毫米；网格 +Z 轴的误差，度。

"""
    center_mm = float(np.linalg.norm(estimated_world[:3, 3] - true_world[:3, 3]) * 1000.0)
    axis_deg = axis_angle_deg(estimated_world[:3, 2], true_world[:3, 2])
    return center_mm, axis_deg


def tcp_matrix(env):
    """
    # Read the Panda tool pose.

    ---

    # 读取 Panda 工具位姿。

"""
    return as_numpy_matrix(env.unwrapped.agent.tcp.pose.to_transformation_matrix())


def side_grasp_matrix(bottle_world):
    """Side grasp on the bottle center. Tool +Z approaches. +Y is the finger gap.

    侧面抓取，抓在瓶子中心。工具的 +Z 是接近方向。+Y 是指缝。

    The approach is horizontal, from the Panda mount toward the center. Meters.
    接近方向在水平面内，从 Panda 的安装位置指向中心。米。
    """
    center = np.asarray(bottle_world, dtype=np.float64).reshape(4, 4)[:3, 3].copy()
    base_xy_m = np.array([-0.615, 0.0], dtype=np.float64)
    approaching = np.array([center[0] - base_xy_m[0], center[1] - base_xy_m[1], 0.0], dtype=np.float64)
    approaching = approaching / max(np.linalg.norm(approaching), 1e-8)
    closing = np.cross(np.array([0.0, 0.0, 1.0]), approaching)
    closing = closing / max(np.linalg.norm(closing), 1e-8)
    ortho = np.cross(closing, approaching)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
    matrix[:3, 3] = center
    return matrix


def side_grasp_from_world(bottle_world):
    """
    # Build a side grasp at the bottle center.

        The base is the Panda table mount, meters.

    ---

    # 在瓶子中心生成侧面抓取。

        底座是 Panda 在这张桌子上的安装位置，米。

"""
    import sapien

    return sapien.Pose(side_grasp_matrix(bottle_world))


def planned_place(grasp, bottle_world, place_center):
    """
    # Stand the cylinder upright by the smallest axis correction, preserving its held end.

    ---

    # 用最小轴向修正立起圆柱，保持当前夹持端；对称位姿轴负号不触发翻腕。

"""
    import sapien

    grasp_matrix = grasp.to_transformation_matrix()
    relative = np.linalg.inv(bottle_world) @ grasp_matrix
    upright = np.eye(4, dtype=np.float64)
    # A cylinder's +Z and -Z predictions represent the same physical axial line.
    # 圆柱的 +Z 与 -Z 预测表示同一条物理轴线；目标端符号跟随当前轴，不强制 +Z。
    cylinder_axis = np.asarray(bottle_world[:3, 2], dtype=np.float64)
    axis_length = float(np.linalg.norm(cylinder_axis))
    assert axis_length > 0.0, "Cylinder axis is invalid / 圆柱轴无效"
    cylinder_axis = cylinder_axis / axis_length
    standing_axis = np.array([0.0, 0.0, 1.0 if cylinder_axis[2] >= 0.0 else -1.0])
    cross_axis = np.cross(cylinder_axis, standing_axis)
    cosine = float(np.clip(np.dot(cylinder_axis, standing_axis), -1.0, 1.0))
    cross_matrix = np.array([[0.0, -cross_axis[2], cross_axis[1]],
                             [cross_axis[2], 0.0, -cross_axis[0]],
                             [-cross_axis[1], cross_axis[0], 0.0]])
    # The chosen sign gives cosine >= 0, so Rodrigues has no antiparallel singularity.
    # 同向端选择保证 cosine >= 0，Rodrigues 公式不存在反平行奇点，也不引入经验阈值。
    axis_correction = np.eye(3) + cross_matrix + (cross_matrix @ cross_matrix) / (1.0 + cosine)
    upright[:3, :3] = axis_correction @ bottle_world[:3, :3]
    upright[:3, 3] = place_center
    return sapien.Pose(upright @ relative)


def corrected_place(tcp_object, place_center, up_sign):
    """
    # Tool pose that stands the measured bottle up at place_center.

        The place keeps that end up and takes out the tilt.

        Yaw stays the measured yaw.

    ## Args

        - tcp_object: object-to-tool, from the hand-camera estimate. It is read as a 4x4. The signature has no default.
        - up_sign: +1 or -1, the end of the mesh axis that was already up in the hand. It becomes the world-z sign of the upright axis. The signature has no default.
        - place_center: the 3-vector written as the upright translation, in meters. The signature has no default.

    ## Returns

        - Returns a sapien.Pose.

    ---

    # 让测量到的瓶子立在 place_center 的工具位姿。

        放下时保留这一端朝上，只把倾斜转正。

        绕竖直轴的转角保留测量值。

    ## 参数

        - tcp_object: 物体到工具，来自手上相机的估计。这里按 4x4 读取。签名里没有默认值。
        - up_sign: +1 或 -1，表示手里已经朝上的那一端。它成为直立轴的世界 z 符号。签名里没有默认值。
        - place_center: 写成立起来之后平移的三维向量，单位米。签名里没有默认值。

    ## 返回

        - 返回 sapien.Pose。

"""
    import sapien

    heading = tcp_object[:3, 0].copy()
    heading[2] = 0.0
    if float(np.linalg.norm(heading)) < 1e-6:
        heading = np.array([1.0, 0.0, 0.0])
    heading = heading / np.linalg.norm(heading)
    z_axis = np.array([0.0, 0.0, float(up_sign)], dtype=np.float64)
    y_axis = np.cross(z_axis, heading)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-8)
    x_axis = np.cross(y_axis, z_axis)
    desired = np.eye(4, dtype=np.float64)
    desired[:3, 0] = x_axis
    desired[:3, 1] = y_axis
    desired[:3, 2] = z_axis
    desired[:3, 3] = place_center
    return sapien.Pose(desired @ np.linalg.inv(tcp_object))


def place_center_for(box_world, right):
    """
    # Bottle center to the right of the estimated box, meters, world frame.

    ---

    # 瓶子中心放在估计出的盒子右侧，米，世界坐标系。

"""
    box_span = 0.5 * float(np.max(BOX_EXTENTS_M[:2]))
    bottle_span = BOTTLE_RADIUS_M
    shift = right * (box_span + bottle_span + PLACE_GAP_M)
    center = box_world[:3, 3] + shift
    center = center.copy()
    center[2] = 0.5 * BOTTLE_HEIGHT_M
    return center


def table_right(extrinsic_cv):
    """
    # Image-right on the table, from the OpenCV camera, unit length.

    ---

    # 图像向右在桌面上的方向，来自 OpenCV 相机，单位长度。

"""
    cam2world = np.linalg.inv(extrinsic_cv)
    right = cam2world[:3, 0].copy()
    right[2] = 0.0
    return right / max(np.linalg.norm(right), 1e-8)


def move_arm(planner, pose):
    """
    # Move the arm, screw motion first.

        If that fails, RRT-Connect.

    ---

    # 先用螺旋运动移动手臂。

        失败再用 RRT-Connect。

"""
    result = planner.move_to_pose_with_screw(pose)
    if result == -1:
        result = planner.move_to_pose_with_RRTConnect(pose)
    if result == -1:
        # Keep the failed target and actual joint state visible; do not relax IK limits.
        # 显式记录失败目标与实际关节状态，不放宽 IK 门槛或修改目标掩盖失败。
        print("PLAN_FAILED", {"target_pose_world": pose.to_transformation_matrix().tolist(),
                              "base_pose_world": planner.base_pose.to_transformation_matrix().tolist(),
                              "qpos": planner.robot.get_qpos().detach().cpu().numpy().tolist()}, flush=True)
        raise RuntimeError("plan failed")
    return result


def render_frame(env):
    """
    # Render one third-person RGB frame.

    ---

    # 渲染一帧第三人称 RGB。

"""
    frame = env.render()
    if frame is None:
        return None
    image = frame
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    return image[..., :3].astype(np.uint8)


def draw_axes(image, extrinsic_cv, K, world_pose_4x4, length_m):
    """
    # RGB axes of one pose on a third-person frame.

        X red, Y green, Z blue.

    ---

    # 在第三人称画面上画一个位姿的 RGB 轴。

        X 红，Y 绿，Z 蓝。

"""

    canvas = image.copy()
    origin = project_point(
        extrinsic_cv @ world_pose_4x4, K, np.zeros(3),
    )
    if origin is None:
        return canvas
    colors = ((220, 30, 30), (30, 170, 30), (30, 90, 220))
    for axis, color in enumerate(colors):
        point = np.zeros(3, dtype=np.float64)
        point[axis] = length_m
        tip = project_point(extrinsic_cv @ world_pose_4x4, K, point)
        if tip is None:
            continue
        cv2.line(
            canvas,
            (int(round(origin[0])), int(round(origin[1]))),
            (int(round(tip[0])), int(round(tip[1]))),
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def paste_inset(image, inset):
    """
    # Hand-camera view in the top-left of the third-person frame.

    ---

    # 第三人称画面左上角贴上手上相机的画面。

"""

    canvas = image.copy()
    thumb = cv2.resize(inset, (240, 180), interpolation=cv2.INTER_AREA)
    canvas[12:192, 12:252] = thumb
    cv2.rectangle(canvas, (12, 12), (251, 191), (255, 255, 255), 2)
    return canvas


def bottle_tilt_deg(env):
    """
    # Angle between the simulated bottle axis and world up, degrees.

    ---

    # 仿真瓶子的轴和世界上方的夹角，度。

"""
    pose = actor_matrix(env.unwrapped.bottle)
    axis = pose[:3, 0]
    return axis_angle_deg(axis, np.array([0.0, 0.0, 1.0]))


def bottle_center_m(env):
    """
    # Read the simulated bottle center.

    ---

    # 读取仿真瓶子的中心。

"""
    return actor_matrix(env.unwrapped.bottle)[:3, 3].copy()


def carry(env, grasp, place_pose, frames, snapshot_dir, meshes, draw_tcp_object, track_dir=None):
    """
    # Grasp from the side, photograph the hand, place, open, and let it settle.

        Fewer than TRACK_MIN_MASK_PX bottle pixels in the lift photo raises RuntimeError after writing lift_rgb.png.

        frames collects the third-person video.

        It is a list.

        The signature has no default.

        snapshot_dir receives the lift photo and the hover photo.

        It is the directory for those snapshots.

        The signature has no default.

        hover_snapshot in that dict is None when the hover bottle mask has fewer than TRACK_MIN_MASK_PX pixels.

        Otherwise it is the hover snapshot path.

        meshes supplies the bottle path and diameter_m at meshes["files"]["bottle"]. diameter_m is meters.

        The signature has no default.

        track_dir stores the table camera and the wrist camera on every frame.

        The third-person frames are then written to disk instead of kept in frames.

    ## Args

        - env: the gym environment whose step is wrapped for the motion. The signature has no default.
        - grasp: the side-grasp sapien.Pose. Its translation is meters. The signature has no default.
        - place_pose: the sapien.Pose for the place and the retreat. Its translation is meters. The signature has no default.
        - draw_tcp_object: stored as the object-to-tool pose drawn on the frames. None means those axes are skipped.

    ## Returns

        - The returned dict holds the tool pose at those photos.
        - The returned dict also holds track_rows, the list remember fills.
        - None: the default. None leaves track_rows empty and keeps the third-person frames in frames.

    ---

    # 从侧面抓住，拍手上的相机，摆放，松开，再等它稳下来。

        抬起照片里瓶子像素少于 TRACK_MIN_MASK_PX 时，先写出 lift_rgb.png，再抛出 RuntimeError。

        frames 收集第三人称录像。

        它是一个列表。

        签名里没有默认值。

        snapshot_dir 收下抬起时和悬停时的照片。

        它是这些快照的目录。

        签名里没有默认值。

        这个 dict 里的 hover_snapshot，在悬停时瓶子 mask 少于 TRACK_MIN_MASK_PX 个像素时是 None。

        否则它是悬停快照的路径。

        meshes 在 meshes["files"]["bottle"] 提供瓶子路径和 diameter_m。

        diameter_m 单位是米。

        签名里没有默认值。

        传入 track_dir 时，每一帧都存桌上相机和腕部相机。

        第三人称画面写到磁盘，不再留在 frames 里。

    ## 参数

        - env: 这次运动里被包住 step 的 gym 环境。签名里没有默认值。
        - grasp: 侧面抓取的 sapien.Pose。平移单位是米。签名里没有默认值。
        - place_pose: 摆放和撤回用的 sapien.Pose。平移单位是米。签名里没有默认值。
        - draw_tcp_object 存成画在画面上的物体到工具位姿。
        - None 表示跳过这些轴。

    ## 返回

        - 返回的 dict 里是这两次拍照时的工具位姿。
        - 返回的 dict 里还有 track_rows，即 remember 填进去的列表。
        - None: 默认值。None 让 track_rows 保持为空，第三人称画面留在 frames 里。

"""
    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )

    render_camera = env.unwrapped._human_render_cameras["render_camera"]
    held = {"active": False, "attached": False, "tcp_object": None}
    last_action = {"value": None}
    track_rows = []

    def remember(view):
        """
        # Store one tracking row for this view.

            save_track_row writes it, and the row is appended to track_rows.

            track_dir None or view None returns immediately.

        ---

        # 存下这一帧的跟踪记录。

            save_track_row 写出这一行，并追加到 track_rows。

            track_dir 为 None 或 view 为 None 时立刻返回。

"""
        if track_dir is None or view is None:
            return
        index = len(track_rows)
        row = save_track_row(
            track_dir, index, env, view, held["attached"], meshes,
        )
        track_rows.append(row)

    def step_and_record(action):
        """
        # Step the saved environment, record one third-person frame, and return that step's output.

            render_frame returns None when the human camera has no frame.

            None means this call neither appends a frame nor calls remember.

            held["tcp_object"] None means the object axes are skipped.

            Another value is drawn at length 0.06 m.

            The hand camera is pasted when track_dir is None, held["active"] is true, and the frame count is a multiple of 4.

            A frame is passed to remember.

            When track_dir is None it is also appended to frames.

        ## Args

            - action: stored on last_action and forwarded to the saved env.step. The signature has no default.

        ---

        # 走保存下来的环境一步，记录一帧第三人称画面，并返回这次 step 的输出。

            render_frame 在第三人称相机没有画面时返回 None。

            None 表示这次既不追加帧，也不调用 remember。

            held["tcp_object"] 为 None 时跳过物体轴。

            其他值按 0.06 米的长度画出。

            track_dir 为 None、held["active"] 为真、且帧数是 4 的倍数时，贴上手上相机。

            这一帧会交给 remember。

            track_dir 为 None 时，它也追加到 frames。

        ## 参数

            - action 存到 last_action，并转给保存下来的 env.step。
            - 签名里没有默认值。

"""
        last_action["value"] = action
        out = raw_step(action)
        frame = render_frame(env)
        if frame is not None and held["tcp_object"] is not None:
            params = render_camera.get_params()
            extrinsic = as_numpy_matrix(params["extrinsic_cv"])
            if extrinsic.shape == (3, 4):
                full = np.eye(4, dtype=np.float64)
                full[:3] = extrinsic
                extrinsic = full
            K = as_numpy_matrix(params["intrinsic_cv"])[:3, :3]
            object_world = tcp_matrix(env) @ held["tcp_object"]
            frame = draw_axes(frame, extrinsic, K, object_world, 0.06)
        if frame is not None and track_dir is None and held["active"] and len(frames) % 4 == 0:
            hand_rgb, _depth, _K, _extrinsic = capture_camera(env, "hand_camera")
            frame = paste_inset(frame, hand_rgb)
        if frame is not None:
            if track_dir is None:
                frames.append(frame)
            remember(frame)
        return out

    raw_step = env.step
    env.step = step_and_record
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    first = render_frame(env)
    if first is not None:
        if track_dir is None:
            frames.append(first)
        remember(first)
    reach = grasp * sapien.Pose([0, 0, -PREGRASP_M])
    move_arm(planner, reach)
    move_arm(planner, grasp)
    planner.close_gripper()
    held["active"] = True
    held["attached"] = True
    held["tcp_object"] = draw_tcp_object
    lift = sapien.Pose(p=np.array(grasp.p) + np.array([0.0, 0.0, LIFT_M]), q=grasp.q)
    move_arm(planner, lift)
    lift_tcp = tcp_matrix(env)
    true_bottle_at_lift = bottle_mesh_matrix(actor_matrix(env.unwrapped.bottle))
    lift_rgb, lift_depth, lift_K, lift_extrinsic = capture_camera(env, "hand_camera")
    lift_mask = color_mask(lift_rgb, "bottle")
    print("HAND_MASK lift", int(lift_mask.sum()), flush=True)
    if int(lift_mask.sum()) < TRACK_MIN_MASK_PX:
        import imageio.v2 as imageio
        os.makedirs(snapshot_dir, exist_ok=True)
        imageio.imwrite(os.path.join(snapshot_dir, "lift_rgb.png"), lift_rgb)
        raise RuntimeError("hand mask")
    lift_snapshot = save_snapshot(
        os.path.join(snapshot_dir, "lift.json"),
        lift_rgb, lift_depth, lift_K,
        [{
            "name": "bottle",
            "mask": lift_mask,
            "mesh_path": meshes["files"]["bottle"]["path"],
            "diameter_m": meshes["files"]["bottle"]["diameter_m"],
        }],
    )
    above = sapien.Pose(p=np.array(place_pose.p) + np.array([0.0, 0.0, LIFT_M]), q=place_pose.q)
    move_arm(planner, above)
    hover_tcp = tcp_matrix(env)
    hover_rgb, hover_depth, hover_K, hover_extrinsic = capture_camera(env, "hand_camera")
    hover_mask = color_mask(hover_rgb, "bottle")
    print("HAND_MASK hover", int(hover_mask.sum()), flush=True)
    hover_snapshot = None
    if int(hover_mask.sum()) >= TRACK_MIN_MASK_PX:
        hover_snapshot = save_snapshot(
            os.path.join(snapshot_dir, "hover.json"),
            hover_rgb, hover_depth, hover_K,
            [{
                "name": "bottle",
                "mask": hover_mask,
                "mesh_path": meshes["files"]["bottle"]["path"],
                "diameter_m": meshes["files"]["bottle"]["diameter_m"],
            }],
        )
    move_arm(planner, place_pose)
    planner.open_gripper()
    held["attached"] = False
    retreat = sapien.Pose(p=np.array(place_pose.p) + np.array([0.0, 0.0, LIFT_M]), q=place_pose.q)
    move_arm(planner, retreat)
    for _index in range(SETTLE_STEPS):
        step_and_record(last_action["value"])
    planner.close()
    return {
        "grasp": grasp,
        "lift_tcp": lift_tcp,
        "lift_snapshot": lift_snapshot,
        "lift_rgb": lift_rgb,
        "lift_K": lift_K,
        "lift_extrinsic": lift_extrinsic,
        "hover_tcp": hover_tcp,
        "hover_snapshot": hover_snapshot,
        "hover_rgb": hover_rgb,
        "hover_K": hover_K,
        "hover_extrinsic": hover_extrinsic,
        "true_bottle_at_lift": true_bottle_at_lift,
        "held": held,
        "track_rows": track_rows,
    }


def save_pose_images(folder, table, hand, meshes, true_world):
    """
    # Save the pose images.

    ## Returns

        - Green: the simulated pose. Both are object-to-camera.

    ---

    # 保存位姿图。

    ## 返回

        - 绿色是仿真位姿。
        - 两个都是物体到相机。

"""
    from wapr.view import save_pose_view

    records = []
    for name in ("bottle", "box"):
        true_cam = table["extrinsic"] @ true_world[name]
        records.append({
            "name": name,
            "pose_4x4": table["poses"][name],
            "gt_pose_4x4": true_cam,
            "mesh": meshes[name]["mesh"],
        })
    save_pose_view(
        table["rgb"], records, os.path.join(folder, "table_pose.png"), K=table["K"],
    )
    true_hand = hand["extrinsic"] @ true_world["bottle_at_lift"]
    save_pose_view(
        hand["rgb"],
        [{
            "name": "bottle",
            "pose_4x4": hand["pose"],
            "gt_pose_4x4": true_hand,
            "mesh": meshes["bottle"]["mesh"],
        }],
        os.path.join(folder, "hand_pose.png"),
        K=hand["K"],
    )


def compose_flow(folder):
    """
    # Table pose, hand pose, and the two finished placements on one board.

    ---

    # 桌上的位姿、手里的位姿，以及两种放完的结果，拼成一张。

"""

    labels = (
        ("table_pose.png", "Table camera: initial pose"),
        ("hand_pose.png", "Wrist camera: grasped object"),
        ("uncorrected_last.png", "Placement: planned grasp"),
        ("corrected_last.png", "Placement: measured grasp"),
    )
    tile_w, tile_h = 480, 360
    caption_h, gap, margin = 44, 18, 18
    board_w = 2 * tile_w + gap + 2 * margin
    board_h = 2 * (caption_h + tile_h) + gap + 2 * margin
    board = np.full((board_h, board_w, 3), (246, 248, 250), dtype=np.uint8)
    # Camera frames already contain pose legends; put panel titles above them.
    # 相机画面自带位姿图例，分图标题放在画面之外以免遮挡。
    for index, (name, label) in enumerate(labels):
        source = cv2.imread(os.path.join(folder, name), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(os.path.join(folder, name))
        image = cv2.resize(source, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        left = margin + (index % 2) * (tile_w + gap)
        top = margin + (index // 2) * (caption_h + tile_h + gap)
        cv2.putText(
            board, label, (left + 12, top + 29),
            cv2.FONT_HERSHEY_SIMPLEX, 0.67, (43, 37, 29), 2, cv2.LINE_AA,
        )
        board[top + caption_h:top + caption_h + tile_h, left:left + tile_w] = image
    cv2.imwrite(os.path.join(folder, "flow.png"), board)


def run_from_open(grasp, place_pose, tcp_object, meshes, label):
    """
    # Open the table, grasp, photograph, place, and close.

    ---

    # 打开桌子，抓住，拍照，摆放，然后关掉。

"""
    import imageio.v2 as imageio

    env = make_env()
    env.reset(seed=0)
    frames = []
    folder = os.path.join(OUT_DIR, label)
    outcome = carry(env, grasp, place_pose, frames, folder, meshes, tcp_object)
    tilt_deg = bottle_tilt_deg(env)
    center_m = bottle_center_m(env)
    if frames:
        imageio.mimwrite(os.path.join(folder, "place.mp4"), frames, fps=20)
        imageio.imwrite(os.path.join(OUT_DIR, label + "_last.png"), frames[-1])
    env.close()
    outcome["tilt_deg"] = tilt_deg
    outcome["center_m"] = center_m
    outcome["true_lift"] = outcome["true_bottle_at_lift"]
    print(
        "PLACE", label,
        "tilt_deg", round(tilt_deg, 2),
        "center_mm", np.round(center_m * 1000.0, 1).tolist(),
        "frames", len(frames),
        flush=True,
    )
    return outcome


def write_rgb(path, rgb):
    """
    # Write one RGB uint8 image.

    ---

    # 写出一张 RGB uint8 图。

"""

    bgr = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, bgr):
        raise RuntimeError("could not write %s" % path)


def read_rgb(path):
    """
    # Read one image as RGB uint8.

    ---

    # 读一张图，得到 RGB uint8。

"""

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("could not read %s" % path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_track_row(track_dir, index, env, view, attached, meshes):
    """
    # One step of the carry: table camera, wrist camera, and the overview.

        Both pose cameras are stored every frame.

        Pose estimation is queued every TRACK_EVERY frames, and only when that camera's color mask is large enough.

        The wrist queue is the bottle.

        The table queue is the bottle and the box.

    ---

    # 搬运中的一步：桌上相机、腕部相机，以及第三人称。

        两个位姿相机每一帧都存。

        每隔 TRACK_EVERY 帧排一次估计，而且该相机的颜色 mask 要够大。

        腕部只排瓶子。

        桌上排瓶子和盒子。

"""

    frame_dir = os.path.join(track_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    sensor_data = env.unwrapped._get_obs_sensor_data()
    estimate_this = (index % TRACK_EVERY) == 0
    cameras = {}
    for camera_name, short in (("table_camera", "table"), ("hand_camera", "hand")):
        rgb, depth_m, K, extrinsic = camera_frame(env, camera_name, sensor_data)
        rgb_path = os.path.join(frame_dir, "f%05d_%s.png" % (index, short))
        write_rgb(rgb_path, rgb)
        jobs = []
        if estimate_this:
            if camera_name == "table_camera":
                object_names = ("bottle", "box")
            else:
                object_names = ("bottle",)
            kept = []
            for object_name in object_names:
                mask = color_mask(rgb, object_name)
                if int(mask.sum()) < TRACK_MIN_MASK_PX:
                    continue
                mask_path = os.path.join(
                    frame_dir, "f%05d_%s_%s_mask.png" % (index, short, object_name),
                )
                mask_u8 = (mask.astype(np.uint8) * 255)
                if not cv2.imwrite(mask_path, mask_u8):
                    raise RuntimeError("could not write %s" % mask_path)
                kept.append((object_name, mask_path))
            if kept:
                depth_path = os.path.join(frame_dir, "f%05d_%s_depth.npy" % (index, short))
                np.save(depth_path, depth_m)
                for object_name, mask_path in kept:
                    jobs.append({
                        "id": "f%05d_%s_%s" % (index, short, object_name),
                        "name": object_name,
                        "rgb": rgb_path,
                        "depth": depth_path,
                        "mask": mask_path,
                        "K": np.asarray(K, dtype=np.float64).reshape(3, 3).tolist(),
                        "mesh": meshes["files"][object_name]["path"],
                        "diameter_m": float(meshes["files"][object_name]["diameter_m"]),
                    })
        cameras[camera_name] = {
            "rgb": rgb_path,
            "K": np.asarray(K, dtype=np.float64).reshape(3, 3).tolist(),
            "extrinsic": np.asarray(extrinsic, dtype=np.float64).reshape(4, 4).tolist(),
            "jobs": jobs,
        }
    view_path = os.path.join(frame_dir, "f%05d_view.png" % index)
    write_rgb(view_path, view)
    render_extrinsic, render_K = extrinsic_and_K(
        env.unwrapped._human_render_cameras["render_camera"].get_params()
    )
    if index % 30 == 0:
        print("TRACK_FRAME", index, "attached", int(bool(attached)), flush=True)
    return {
        "i": int(index),
        "attached": bool(attached),
        "tcp": np.asarray(tcp_matrix(env), dtype=np.float64).reshape(4, 4).tolist(),
        "gt_bottle": bottle_mesh_matrix(actor_matrix(env.unwrapped.bottle)).reshape(4, 4).tolist(),
        "gt_box": actor_matrix(env.unwrapped.box).reshape(4, 4).tolist(),
        "view": view_path,
        "render_K": np.asarray(render_K, dtype=np.float64).reshape(3, 3).tolist(),
        "render_extrinsic": np.asarray(render_extrinsic, dtype=np.float64).reshape(4, 4).tolist(),
        "cameras": cameras,
    }


def perceive_table(files, snapshot_path):
    """
    # Estimate both meshes on the fixed table camera, then close that session.

        The grasp aims at the estimated bottle.

        The place stands that estimate upright beside the box.

    ## Returns

        - The returned poses are object-to-world, meters.

    ---

    # 用固定的桌上相机估计两个网格，然后关掉这次仿真。

        抓取对准估计出的瓶子。

        摆放时把这个估计立在盒子旁边。

    ## 返回

        - 返回的位姿是物体到世界，米。

"""
    env = make_env()
    try:
        env.reset(seed=0)
        true_bottle = bottle_mesh_matrix(actor_matrix(env.unwrapped.bottle))
        true_box = actor_matrix(env.unwrapped.box)
        rgb, depth_m, K, extrinsic = capture_camera(env, "table_camera")
        bottle_mask = color_mask(rgb, "bottle")
        box_mask = color_mask(rgb, "box")
        print(
            "MASK bottle", int(bottle_mask.sum()),
            "box", int(box_mask.sum()),
            flush=True,
        )
        if int(bottle_mask.sum()) < TRACK_MIN_MASK_PX or int(box_mask.sum()) < TRACK_MIN_MASK_PX:
            raise RuntimeError("table mask")
        # Input validity depends on the observed masks, never the true object pose.
        # 输入有效性只由观测掩码判断，不依据物体真值位姿筛选运行。
        snapshot = save_snapshot(
            snapshot_path, rgb, depth_m, K,
            [
                {
                    "name": "bottle",
                    "mask": bottle_mask,
                    "mesh_path": files["bottle"]["path"],
                    "diameter_m": files["bottle"]["diameter_m"],
                },
                {
                    "name": "box",
                    "mask": box_mask,
                    "mesh_path": files["box"]["path"],
                    "diameter_m": files["box"]["diameter_m"],
                },
            ],
        )
    finally:
        env.close()
    poses = run_estimate(snapshot)
    bottle_world = world_pose(extrinsic, poses["bottle"])
    box_world = world_pose(extrinsic, poses["box"])
    bottle_mm, bottle_axis = pose_error(bottle_world, true_bottle)
    box_mm, box_axis = pose_error(box_world, true_box)
    print(
        "TABLE bottle_mm", round(bottle_mm, 1),
        "bottle_axis_deg", round(bottle_axis, 1),
        "box_mm", round(box_mm, 1),
        "box_axis_deg", round(box_axis, 1),
        flush=True,
    )
    right = table_right(extrinsic)
    target_m = place_center_for(box_world, right)
    grasp = side_grasp_from_world(bottle_world)
    planned = planned_place(grasp, bottle_world, target_m)
    return {
        "bottle_world": bottle_world,
        "box_world": box_world,
        "grasp": grasp,
        "planned": planned,
        "table_bottle_center_mm": bottle_mm,
        "table_bottle_axis_deg": bottle_axis,
        "table_box_center_mm": box_mm,
        "table_box_axis_deg": box_axis,
    }


def samples_from_rows(rows, camera_name, object_name, pose_by_id):
    """
    # Collect successful estimates as object-to-world.

    ---

    # 收集成功的估计，写成物体到世界。

"""
    samples = {}
    for row in rows:
        camera = row["cameras"][camera_name]
        extrinsic = np.asarray(camera["extrinsic"], dtype=np.float64).reshape(4, 4)
        for job in camera["jobs"]:
            if job["name"] != object_name or job["id"] not in pose_by_id:
                continue
            samples[int(row["i"])] = world_pose(extrinsic, pose_by_id[job["id"]])
    return samples


def hold_track(count, samples, seed):
    """
    # Keep the last world pose until this camera measures again.

        The table camera uses this.

        It does not follow the gripper between photos.

    ---

    # 保持上一次的世界位姿，直到这个相机再次测到。

        桌上相机用这个。

        两次拍照之间，它不跟着夹爪走。

"""
    pred = []
    how = []
    last = seed
    for index in range(count):
        if index in samples:
            last = samples[index]
            pred.append(last)
            how.append("本帧估计")
        elif last is not None:
            pred.append(last)
            how.append("保持上次")
        else:
            pred.append(None)
            how.append("还没估计")
    return pred, how


def wrist_track(rows, samples):
    """
    # Track the bottle from the wrist camera.

        Before the fingers close, the last measurement stays in the world.

        After they close, the measurement is carried by the tool pose until the wrist measures again.

        After the fingers open, it stays where it was released.

    ---

    # 用腕部相机跟踪瓶子。

        手指合上之前，上一次测量留在世界里。

        合上之后，这次测量由工具位姿带着走，直到腕部再次测量。

        手指松开之后，停在松开时的位置。

"""
    pred = []
    how = []
    last = None
    object_in_tcp = None
    frozen = None
    seen_attached = False
    for index, row in enumerate(rows):
        tcp = np.asarray(row["tcp"], dtype=np.float64).reshape(4, 4)
        attached = bool(row["attached"])
        if attached:
            seen_attached = True
        fresh = index in samples
        if fresh:
            last = samples[index]
            if attached:
                object_in_tcp = np.linalg.inv(tcp) @ last
            else:
                object_in_tcp = None
                frozen = last
        if attached and object_in_tcp is not None:
            pred.append(tcp @ object_in_tcp)
            frozen = pred[-1]
            how.append("本帧估计" if fresh else "沿夹爪跟上")
        elif (not attached) and seen_attached and frozen is not None and not fresh:
            pred.append(frozen)
            how.append("松开后停住")
        elif last is not None:
            pred.append(last)
            how.append("本帧估计" if fresh else "保持上次")
        else:
            pred.append(None)
            how.append("还没看到")
    return pred, how


def scale_rgb_K(rgb, K, width, height):
    """
    # Resize an image and scale K with it.

        K stays in pixels, y down.

    ---

    # 缩放图像，并按同样比例缩放 K。

        K 仍是像素，y 向下。

"""

    src_h, src_w = rgb.shape[:2]
    scaled = cv2.resize(rgb, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    scaled_K = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    scaled_K[0, :] *= float(width) / float(src_w)
    scaled_K[1, :] *= float(height) / float(src_h)
    return scaled, scaled_K


def text_bar(text, width, height, size):
    """
    # Draw one text bar.

    ---

    # 画一条文字横条。

"""
    from PIL import Image, ImageDraw

    from wapr.view import _font

    image = Image.new("RGB", (int(width), int(height)), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    draw.text((12, 6), text, fill=(255, 255, 255), font=_font(size))
    return np.asarray(image)


def silhouette_mask(mesh, pose_cam, K, height, width):
    """
    # Union of the mesh triangles in front of the camera.

    ## Args

        - pose_cam: object-to-camera, meters.

    ---

    # 相机前方网格三角形的并集。

    ## 参数

        - pose_cam: 物体到相机，米。

"""

    pose = np.asarray(pose_cam, dtype=np.float64).reshape(4, 4)
    intrinsic = np.asarray(K, dtype=np.float64).reshape(3, 3)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    camera = (pose[:3, :3] @ vertices.T).T + pose[:3, 3]
    depth = camera[:, 2]
    projected = (intrinsic @ camera.T).T
    pixels = projected[:, :2] / np.maximum(depth[:, None], 1e-4)
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    for face in faces:
        if np.any(depth[face] <= 1e-3):
            continue
        corners = np.round(pixels[face]).astype(np.int32)
        if np.any(np.abs(corners[:, 0]) > width * 8) or np.any(np.abs(corners[:, 1]) > height * 8):
            continue
        cv2.fillConvexPoly(mask, corners, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def draw_largest_contour(canvas, mask, color, thickness):
    """The outer silhouette only. Internal facets of a symmetric mesh are not drawn.

    只画外轮廓。对称网格内部的小面不画。


    """

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(canvas, [contour], -1, color, thickness, cv2.LINE_AA)


def draw_pose_outlines(rgb, K, extrinsic, items):
    """
    # Draw the ground-truth silhouette, then the prediction.

    ---

    # 先画真值轮廓，再画预测轮廓。

        A cylinder can spin about its axis without moving this outline. The square bounding box of that same cylinder turns, so the outline is what shows alignment. 圆柱绕自己的轴转时，这条轮廓不动。

        同一个圆柱的方形包围盒会转，所以对齐看的是轮廓。

        items are (name, mesh, pred_world, gt_world). World poses are meters. items 是 (名字, 网格, 预测的世界位姿, 真值世界位姿)。

        世界位姿的单位是米。

"""
    from PIL import Image, ImageDraw

    from wapr.view import _font

    canvas = np.ascontiguousarray(rgb[..., :3].copy())
    height, width = canvas.shape[:2]
    world_to_cam = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
    drawn = []
    for name, mesh, pred_world, gt_world in items:
        gt_cam = world_to_cam @ np.asarray(gt_world, dtype=np.float64).reshape(4, 4)
        pred_cam = None
        if pred_world is not None:
            pred_cam = world_to_cam @ np.asarray(pred_world, dtype=np.float64).reshape(4, 4)
        drawn.append((name, mesh, pred_cam, gt_cam))
    for _name, mesh, _pred_cam, gt_cam in drawn:
        mask = silhouette_mask(mesh, gt_cam, K, height, width)
        draw_largest_contour(canvas, mask, (0, 170, 0), 3)
    anchors = []
    for name, mesh, pred_cam, gt_cam in drawn:
        pose = gt_cam
        if pred_cam is not None:
            mask = silhouette_mask(mesh, pred_cam, K, height, width)
            draw_largest_contour(canvas, mask, (220, 30, 30), 2)
        else:
            mask = silhouette_mask(mesh, pose, K, height, width)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        anchors.append((name, int(xs.min()), int(max(0, ys.min() - 4))))
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = _font(16)
    for name, x, y in anchors:
        box = font.getbbox(name)
        tw = box[2] - box[0]
        th = box[3] - box[1]
        x = min(max(0, x), max(0, width - tw - 6))
        y = min(max(0, y - th - 4), max(0, height - th - 6))
        draw.rectangle([x, y, x + tw + 6, y + th + 4], fill=(20, 20, 20))
        draw.text((x + 3, y + 2 - box[1]), name, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def median_error(preds, rows, gt_key, only_attached):
    """
    # Median center error in mm and axis error in degrees.

        None when empty.

    ---

    # 中心误差的中位数，毫米；轴误差的中位数，度。

        没有样本时是 None。

"""
    centers = []
    axes = []
    for index, pred in enumerate(preds):
        if pred is None:
            continue
        if only_attached and not bool(rows[index]["attached"]):
            continue
        gt = np.asarray(rows[index][gt_key], dtype=np.float64).reshape(4, 4)
        center_mm, axis_deg = pose_error(pred, gt)
        centers.append(center_mm)
        axes.append(axis_deg)
    if not centers:
        return None, None
    return float(np.median(centers)), float(np.median(axes))


def frustum_corners_world(extrinsic_cv, K, width, height, near_m, far_m):
    """Eight corners of the view volume, meters, world frame. Near four, then far four.

    视野的八个角，米，世界坐标系。先是近处四个，再是远处四个。

    extrinsic_cv is world-to-camera. K is the original image, y down.
    The corners are the rays through the image corners at near_m and far_m.
    extrinsic_cv 是世界到相机。K 对应该相机的原始图像，y 向下。
    这八个角是图像四个角在 near_m 和 far_m 上的射线。
    """
    intrinsic = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    camera_to_world = np.linalg.inv(np.asarray(extrinsic_cv, dtype=np.float64).reshape(4, 4))
    pixels = (
        (0.0, 0.0),
        (float(width - 1), 0.0),
        (float(width - 1), float(height - 1)),
        (0.0, float(height - 1)),
    )
    corners = []
    for depth_m in (float(near_m), float(far_m)):
        for u, v in pixels:
            point_cam = np.array([
                (u - cx) * depth_m / fx,
                (v - cy) * depth_m / fy,
                depth_m,
                1.0,
            ], dtype=np.float64)
            corners.append((camera_to_world @ point_cam)[:3])
    return np.stack(corners, axis=0)


def draw_dashed_segment(canvas, start_xy, end_xy, color, thickness, dash_px, gap_px):
    """One dashed line in pixels. The gap is left as the original picture.

    一条虚线，像素。空段留下原来的画面。
    """

    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1.0:
        return
    direction = delta / length
    cursor = 0.0
    drawing = True
    while cursor < length:
        step = float(dash_px if drawing else gap_px)
        nxt = min(length, cursor + step)
        if drawing:
            point_a = start + direction * cursor
            point_b = start + direction * nxt
            cv2.line(
                canvas,
                (int(round(point_a[0])), int(round(point_a[1]))),
                (int(round(point_b[0])), int(round(point_b[1]))),
                color,
                thickness,
                cv2.LINE_AA,
            )
        drawing = not drawing
        cursor = nxt


def draw_view_volume(image, render_ext, render_K, cam_ext, cam_K, width, height, near_m, far_m):
    """Cyan trapezoid of one camera, drawn on the third-person picture.

    在第三人称画面上画一个相机的青色梯形视野。

    The far face is the window that camera is shooting. The four sides are the volume.
    远处那个面是它正在拍的窗口。四条棱是这段视野的空间。
    """

    canvas = np.ascontiguousarray(image[..., :3].copy())
    corners_w = frustum_corners_world(cam_ext, cam_K, width, height, near_m, far_m)
    render = np.asarray(render_ext, dtype=np.float64).reshape(4, 4)
    intrinsic = np.asarray(render_K, dtype=np.float64).reshape(3, 3)
    camera = (render[:3, :3] @ corners_w.T).T + render[:3, 3]

    def project(point_cam):
        if float(point_cam[2]) <= 1e-4:
            return None
        pixel = intrinsic @ point_cam
        return pixel[:2] / pixel[2]

    def clip_segment(a, b):
        zmin = 0.02
        if a[2] >= zmin and b[2] >= zmin:
            return a, b
        if a[2] < zmin and b[2] < zmin:
            return None
        hit = a + (zmin - a[2]) / (b[2] - a[2]) * (b - a)
        if a[2] < zmin:
            return hit, b
        return a, hit

    color = (40, 210, 255)
    far_pixels = []
    far_ok = True
    for index in range(4, 8):
        pixel = project(camera[index])
        if pixel is None:
            far_ok = False
            break
        far_pixels.append(pixel)
    if far_ok:
        polygon = np.round(np.stack(far_pixels)).astype(np.int32)
        tint = canvas.copy()
        cv2.fillConvexPoly(tint, polygon, color)
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255)
        mixed = (0.88 * canvas + 0.12 * tint).astype(np.uint8)
        canvas[mask > 0] = mixed[mask > 0]
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for start, end in edges:
        clipped = clip_segment(camera[start], camera[end])
        if clipped is None:
            continue
        pix_a = project(clipped[0])
        pix_b = project(clipped[1])
        if pix_a is None or pix_b is None:
            continue
        draw_dashed_segment(
            canvas, pix_a, pix_b, color, 2, VIEW_DASH_PX, VIEW_GAP_PX,
        )
    return canvas


def wrist_camera_tilt_deg(row):
    """
    # Angle between the wrist camera's forward axis and straight down, degrees.

    ---

    # 腕部相机前向和正下方的夹角，度。

"""
    extrinsic = np.asarray(row["cameras"]["hand_camera"]["extrinsic"], dtype=np.float64).reshape(4, 4)
    forward = np.linalg.inv(extrinsic)[:3, 2]
    down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    cosine = float(np.dot(forward, down))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def wrist_opening_index(rows):
    """
    # First frame whose wrist camera has tilted far enough to show a box side.

    ---

    # 腕部相机侧到能看见盒子侧面的第一帧。

"""
    for index, row in enumerate(rows):
        if wrist_camera_tilt_deg(row) >= WRIST_OPEN_MIN_TILT_DEG:
            return index
    return 0


def hold_jumped_box(box_pred):
    """
    # Keep the previous box pose when the center jumps.

        The box does not move. A jumped center is not drawn.

    ---

    # 盒子中心跳开时，沿用上一帧的位姿。

        盒子不动。跳开的中心不画。

"""
    held = []
    last = None
    for pose in box_pred:
        if pose is None:
            held.append(last)
            continue
        pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        if last is not None:
            jump_m = float(np.linalg.norm(pose[:3, 3] - last[:3, 3]))
            if jump_m > BOX_HOLD_JUMP_M:
                held.append(last)
                continue
        last = pose
        held.append(pose)
    return held


# Words burned into the tracking video. The page has an English mode and a
# Chinese mode, so the same frames are written twice. The status tokens stored
# on each row stay Chinese. Only the drawn words change.
# 烧进跟踪视频的字。页面有英文和中文两套，所以同一批帧写两遍。每一行里存的状态词仍是中文，变的只是画上去的字。
VIDEO_CAPTION = {
    "zh": {
        "banner_cameras": "位姿相机有 2 个。红轮廓是预测，绿轮廓是真值。后面每一帧跟着上一帧的位姿走。",
        "banner_views": "青色虚线是视野。蓝色实线是侧面抓取，跟着这一路估计的瓶子走。左：桌上。右：腕部。",
        "top_table": "上面左 · 桌上相机自己看到的",
        "top_wrist": "上面右 · 腕部相机自己看到的",
        "bottom_table": "下面左 · 桌上相机的视野",
        "bottom_wrist": "下面右 · 腕部相机的视野",
        "bottle": "瓶子",
        "box": "盒子",
        "table": "桌上",
        "wrist": "腕部",
        "axis": "轴偏",
        "how": {
            "本帧估计": "本帧跟踪",
            "沿夹爪跟上": "沿夹爪跟上",
            "松开后停住": "松开后停住",
            "保持上次": "保持上次",
            "还没估计": "还没估计",
            "还没看到": "还没看到",
        },
    },
    "en": {
        "banner_cameras": "Two pose cameras. Red is the estimate. Green is ground truth. Later frames track the previous pose.",
        "banner_views": "Cyan dashes: camera view. Blue lines: side grasp on that camera's estimate. Left: table. Right: wrist.",
        "top_table": "Top left · what the table camera sees",
        "top_wrist": "Top right · what the wrist camera sees",
        "bottom_table": "Bottom left · table camera view",
        "bottom_wrist": "Bottom right · wrist camera view",
        "bottle": "bottle",
        "box": "box",
        "table": "Table",
        "wrist": "Wrist",
        "axis": "axis",
        "how": {
            "本帧估计": "tracked",
            "沿夹爪跟上": "follows the gripper",
            "松开后停住": "held after release",
            "保持上次": "held",
            "还没估计": "not yet",
            "还没看到": "not seen",
        },
    },
}


def grasp_fork_segments(bottle_world):
    """Open side gripper at the perceived bottle center, world meters.

    张开的侧面夹爪，放在感知到的瓶子中心，世界坐标，米。

    Local +Z approaches the bottle. The stick starts PREGRASP_M behind the center.
    局部 +Z 朝瓶子接近。接近段从中心后方 PREGRASP_M 处开始。
    """
    pose = side_grasp_matrix(bottle_world)
    gap = GRASP_FINGER_GAP_M * 0.5
    local = (
        (np.array([0.0, 0.0, -PREGRASP_M]), np.array([0.0, 0.0, -GRASP_PALM_BACK_M])),
        (np.array([0.0, gap, -GRASP_PALM_BACK_M]), np.array([0.0, -gap, -GRASP_PALM_BACK_M])),
        (np.array([0.0, gap, -GRASP_PALM_BACK_M]), np.array([0.0, gap, 0.0])),
        (np.array([0.0, -gap, -GRASP_PALM_BACK_M]), np.array([0.0, -gap, 0.0])),
    )

    def world(point):
        return pose[:3, :3] @ point + pose[:3, 3]

    return tuple((world(start), world(end)) for start, end in local)


def draw_grasp_fork(canvas, extrinsic, camera_k, bottle_world):
    """Solid blue side grasp on this camera's perceived bottle pose.

    蓝色实线，这一路相机感知到的瓶子上的侧面抓取。
    """

    if bottle_world is None:
        return
    world_to_cam = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
    intrinsic = np.asarray(camera_k, dtype=np.float64).reshape(3, 3)
    for start, end in grasp_fork_segments(bottle_world):
        points = np.stack([start, end], axis=0)
        camera = (world_to_cam[:3, :3] @ points.T).T + world_to_cam[:3, 3]
        if np.any(camera[:, 2] <= 1e-4):
            continue
        pixels = (intrinsic @ camera.T).T
        pixels = pixels[:, :2] / camera[:, 2:3]
        cv2.line(
            canvas,
            (int(round(pixels[0, 0])), int(round(pixels[0, 1]))),
            (int(round(pixels[1, 0])), int(round(pixels[1, 1]))),
            GRASP_RGB,
            3,
            cv2.LINE_AA,
        )


def compose_tracking_video(path, rows, wrist_pred, wrist_how, table_pred, table_how, box_pred, files, lang):
    """
    # Write the two pose cameras and the overview into one video.

        Top left is the fixed table camera.

        Top right is the wrist camera on the gripper.

        The bottom view does not estimate; its outlines are the wrist track drawn in the world.

    ## Returns

        - Red: the predicted silhouette, green is the simulated silhouette.

    ---

    # 把两个位姿相机和第三人称写成一段视频。

        左上是固定的桌上相机。

        右上是装在夹爪上的腕部相机。

        下面的画面不估计，它的轮廓是腕部跟踪在世界里的投影。

    ## 返回

        - 红轮廓是预测，绿轮廓是仿真位姿。

"""
    import imageio.v2 as imageio

    # Two 640-wide cameras side by side. The overview is scaled to that width.
    # 两个 640 宽的相机并排。第三人称缩放到同样的总宽。
    panel_w = 640
    panel_h = 480
    overview_w = panel_w * 2
    title_h = 32
    status_h = 32
    gap_h = 16
    bottle_mesh = files["bottle"]["mesh"]
    words = VIDEO_CAPTION[lang]
    box_mesh = files["box"]["mesh"]
    preview_dir = os.path.join(os.path.dirname(path), "preview_%s" % lang)
    os.makedirs(preview_dir, exist_ok=True)
    marks = {"start": 0, "end": len(rows) - 1}
    lifted_z = []
    for index, row in enumerate(rows):
        gt_bottle = np.asarray(row["gt_bottle"], dtype=np.float64).reshape(4, 4)
        lifted_z.append(float(gt_bottle[2, 3]))
        if wrist_how[index] == "本帧估计" and "hand_est" not in marks:
            marks["hand_est"] = index
        if wrist_how[index] == "沿夹爪跟上" and "follow" not in marks:
            marks["follow"] = index
        if wrist_how[index] == "松开后停住" and "release" not in marks:
            marks["release"] = index
    marks["lift"] = int(np.argmax(np.asarray(lifted_z)))
    banner = np.concatenate([
        text_bar(words["banner_cameras"], overview_w, 40, 20),
        text_bar(words["banner_views"], overview_w, 40, 20),
    ], axis=0)
    writer = imageio.get_writer(path, fps=20, codec="libx264", quality=8, macro_block_size=1)
    try:
        for index, row in enumerate(rows):
            gt_bottle = np.asarray(row["gt_bottle"], dtype=np.float64).reshape(4, 4)
            gt_box = np.asarray(row["gt_box"], dtype=np.float64).reshape(4, 4)
            table = row["cameras"]["table_camera"]
            hand = row["cameras"]["hand_camera"]
            table_ext = np.asarray(table["extrinsic"], dtype=np.float64).reshape(4, 4)
            hand_ext = np.asarray(hand["extrinsic"], dtype=np.float64).reshape(4, 4)
            table_rgb, table_K = scale_rgb_K(read_rgb(table["rgb"]), table["K"], panel_w, panel_h)
            hand_rgb, hand_K = scale_rgb_K(read_rgb(hand["rgb"]), hand["K"], panel_w, panel_h)
            table_view = np.array(draw_pose_outlines(
                table_rgb, table_K, table_ext,
                [
                    (words["bottle"], bottle_mesh, table_pred[index], gt_bottle),
                    (words["box"], box_mesh, box_pred[index], gt_box),
                ],
            )).copy()
            draw_grasp_fork(table_view, table_ext, table_K, table_pred[index])
            hand_view = np.array(draw_pose_outlines(
                hand_rgb, hand_K, hand_ext,
                [
                    (words["bottle"], bottle_mesh, wrist_pred[index], gt_bottle),
                    (words["box"], box_mesh, box_pred[index], gt_box),
                ],
            )).copy()
            draw_grasp_fork(hand_view, hand_ext, hand_K, wrist_pred[index])
            render_ext = np.asarray(row["render_extrinsic"], dtype=np.float64).reshape(4, 4)
            overview_rgb, overview_K = scale_rgb_K(
                read_rgb(row["view"]), row["render_K"], panel_w, panel_h,
            )
            # Same outside picture twice. Cyan trapezoid is that camera's view volume.
            # 同一张外面的画面画两遍。青色梯形是这一路相机的视野。
            table_base = draw_view_volume(
                overview_rgb, render_ext, overview_K,
                table_ext, table["K"], TABLE_CAMERA_WIDTH, TABLE_CAMERA_HEIGHT,
                TABLE_VIEW_NEAR_M, TABLE_VIEW_FAR_M,
            )
            wrist_base = draw_view_volume(
                overview_rgb, render_ext, overview_K,
                hand_ext, hand["K"], HAND_CAMERA_WIDTH, HAND_CAMERA_HEIGHT,
                WRIST_VIEW_NEAR_M, WRIST_VIEW_FAR_M,
            )
            table_scene = np.array(draw_pose_outlines(
                table_base, overview_K, render_ext,
                [
                    (words["bottle"], bottle_mesh, table_pred[index], gt_bottle),
                    (words["box"], box_mesh, box_pred[index], gt_box),
                ],
            )).copy()
            draw_grasp_fork(table_scene, render_ext, overview_K, table_pred[index])
            wrist_scene = np.array(draw_pose_outlines(
                wrist_base, overview_K, render_ext,
                [(words["bottle"], bottle_mesh, wrist_pred[index], gt_bottle)],
            )).copy()
            draw_grasp_fork(wrist_scene, render_ext, overview_K, wrist_pred[index])
            table_status = status_line(words["table"], table_pred[index], gt_bottle, table_how[index], lang)
            hand_status = status_line(words["wrist"], wrist_pred[index], gt_bottle, wrist_how[index], lang)
            frame = np.concatenate([
                banner,
                np.concatenate([
                    text_bar(words["top_table"], panel_w, title_h, 18),
                    text_bar(words["top_wrist"], panel_w, title_h, 18),
                ], axis=1),
                np.concatenate([table_view, hand_view], axis=1),
                np.concatenate([
                    text_bar(table_status, panel_w, status_h, 18),
                    text_bar(hand_status, panel_w, status_h, 18),
                ], axis=1),
                text_bar("", overview_w, gap_h, 12),
                np.concatenate([
                    text_bar(words["bottom_table"], panel_w, title_h, 18),
                    text_bar(words["bottom_wrist"], panel_w, title_h, 18),
                ], axis=1),
                np.concatenate([table_scene, wrist_scene], axis=1),
                np.concatenate([
                    text_bar(table_status, panel_w, status_h, 18),
                    text_bar(hand_status, panel_w, status_h, 18),
                ], axis=1),
            ], axis=0)
            writer.append_data(np.ascontiguousarray(frame))
            for mark, mark_index in marks.items():
                if index == mark_index:
                    write_rgb(os.path.join(preview_dir, mark + ".png"), frame)
            if index % 40 == 0:
                print("TRACK_DRAW", index, "/", len(rows), flush=True)
    finally:
        writer.close()
    print("TRACK_VIDEO", path, "frames", len(rows), flush=True)
    return preview_dir


def status_line(name, pred_world, gt_world, how, lang):
    """
    # Center error in mm and mesh-axis error in degrees, plus where the pose came from.

    ---

    # 中心误差，毫米；网格轴误差，度；再加这一帧的位姿从哪来。

"""
    words = VIDEO_CAPTION[lang]
    shown = words["how"].get(how, how)
    if pred_world is None:
        return "%s  %s" % (name, shown)
    center_mm, axis_deg = pose_error(pred_world, gt_world)
    return "%s %.1f mm  %s %.1f°  %s" % (name, center_mm, words["axis"], axis_deg, shown)


def write_tracking_video():
    """
    # Record one carry and write both cameras' pose tracks.

        The table camera is fixed outside the arm.

        The wrist camera is mounted on the gripper.

        Each one estimates the bottle from its own RGB-D.

        The wrist pose is carried by the tool between its own measurements.

    ---

    # 录一次搬运，并写出两个相机上的位姿跟踪。

        桌上相机固定在手臂外面。

        腕部相机装在夹爪上。

        各自用自己的 RGB-D 估计瓶子。

        腕部两次测量之间，位姿由工具带着走。

"""
    track_dir = os.path.join(OUT_DIR, "tracking")
    os.makedirs(track_dir, exist_ok=True)
    files = write_known_meshes(os.path.join(track_dir, "meshes"))
    plan = perceive_table(files, os.path.join(track_dir, "table.json"))
    env = make_env()
    try:
        env.reset(seed=0)
        meshes = {"files": files, "bottle_world": plan["bottle_world"]}
        outcome = carry(
            env,
            plan["grasp"],
            plan["planned"],
            [],
            os.path.join(track_dir, "snapshots"),
            meshes,
            None,
            track_dir=track_dir,
        )
    finally:
        env.close()
    rows = outcome["track_rows"]
    if not rows:
        raise RuntimeError("track empty")
    for index, row in enumerate(rows):
        if int(row["i"]) != index:
            raise RuntimeError("track index")
    with open(os.path.join(track_dir, "rows.json"), "w", encoding="utf-8") as stream:
        json.dump(rows, stream)
    jobs = []
    for row in rows:
        for camera in row["cameras"].values():
            jobs.extend(camera["jobs"])
    print("TRACK_JOBS", len(jobs), "frames", len(rows), flush=True)
    with open(os.path.join(track_dir, "plan.json"), "w", encoding="utf-8") as stream:
        json.dump({
            "bottle_world": np.asarray(plan["bottle_world"], dtype=np.float64).reshape(4, 4).tolist(),
            "box_world": np.asarray(plan["box_world"], dtype=np.float64).reshape(4, 4).tolist(),
            "table_bottle_center_mm": plan["table_bottle_center_mm"],
            "table_bottle_axis_deg": plan["table_bottle_axis_deg"],
        }, stream)
    pose_by_id = {}
    if jobs:
        sequence_path = os.path.join(track_dir, "sequence.json")
        with open(sequence_path, "w", encoding="utf-8") as stream:
            json.dump(jobs, stream)
        pose_by_id = run_sequence(sequence_path)
    print("TRACK_POSES", len(pose_by_id), flush=True)
    finish_tracking(
        track_dir, rows, files, plan["bottle_world"], plan["box_world"], pose_by_id,
        plan["table_bottle_center_mm"], plan["table_bottle_axis_deg"], len(jobs),
    )


def finish_tracking(track_dir, rows, files, bottle_world, box_world, pose_by_id, table_mm, table_axis, n_jobs):
    """
    # Turn stored frames and estimates into the tracking video.

    ---

    # 用已经存下的画面和估计结果写成跟踪视频。

"""
    table_samples = samples_from_rows(rows, "table_camera", "bottle", pose_by_id)
    box_samples = samples_from_rows(rows, "table_camera", "box", pose_by_id)
    wrist_samples = samples_from_rows(rows, "hand_camera", "bottle", pose_by_id)
    table_pred, table_how = hold_track(len(rows), table_samples, bottle_world)
    box_pred, _box_how = hold_track(len(rows), box_samples, box_world)
    box_pred = hold_jumped_box(box_pred)
    wrist_pred, wrist_how = wrist_track(rows, wrist_samples)
    opening = wrist_opening_index(rows)
    print(
        "TRACK_OPEN", opening,
        "tilt_deg", round(wrist_camera_tilt_deg(rows[opening]), 1),
        flush=True,
    )
    # zh keeps the old filename. en is the same frames with English bars.
    # 中文沿用原来的文件名。英文是同一批帧，横条换成英文。
    videos = {}
    previews = {}
    for lang in ("zh", "en"):
        suffix = "" if lang == "zh" else "_en"
        video_path = os.path.join(track_dir, "pose_track%s.mp4" % suffix)
        previews[lang] = compose_tracking_video(
            video_path,
            rows[opening:],
            wrist_pred[opening:],
            wrist_how[opening:],
            table_pred[opening:],
            table_how[opening:],
            box_pred[opening:],
            files,
            lang,
        )
        videos[lang] = video_path
    video_path = videos["zh"]
    preview_dir = previews["zh"]
    hand_mm, hand_axis = median_error(wrist_pred, rows, "gt_bottle", True)
    track_mm, track_axis = median_error(table_pred, rows, "gt_bottle", False)
    report = {
        "frames": len(rows),
        "jobs": int(n_jobs),
        "poses": len(pose_by_id),
        "table_estimates": len(table_samples),
        "wrist_estimates": len(wrist_samples),
        "box_estimates": len(box_samples),
        "table_bottle_center_mm": table_mm,
        "table_bottle_axis_deg": table_axis,
        "hand_attached_center_mm": hand_mm,
        "hand_attached_axis_deg": hand_axis,
        "table_track_center_mm": track_mm,
        "table_track_axis_deg": track_axis,
        "video": videos["zh"],
        "video_en": videos["en"],
        "preview_dir": preview_dir,
    }
    with open(os.path.join(track_dir, "report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("TRACK_REPORT", json.dumps(report), flush=True)
    print("WROTE", video_path, flush=True)


def recompose_tracking():
    """
    # Draw the video again from the saved frames and estimates.

    ---

    # 用已经存下的画面和估计结果把视频再画一遍。

"""

    track_dir = os.path.join(OUT_DIR, "tracking")
    with open(os.path.join(track_dir, "rows.json"), "r", encoding="utf-8") as stream:
        rows = json.load(stream)
    with open(os.path.join(track_dir, "plan.json"), "r", encoding="utf-8") as stream:
        plan = json.load(stream)
    pose_path = os.path.join(track_dir, "sequence_poses.json")
    pose_by_id = {}
    if os.path.isfile(pose_path):
        with open(pose_path, "r", encoding="utf-8") as stream:
            for row in json.load(stream):
                if "pose" in row:
                    pose_by_id[row["id"]] = np.asarray(row["pose"], dtype=np.float64)
    files = {}
    for name in ("bottle", "box"):
        mesh_path = os.path.join(track_dir, "meshes", name + ".ply")
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        files[name] = {"path": mesh_path, "mesh": mesh, "diameter_m": mesh_diameter_m(mesh)}
    n_jobs = 0
    for row in rows:
        for camera in row["cameras"].values():
            n_jobs += len(camera["jobs"])
    finish_tracking(
        track_dir, rows, files,
        np.asarray(plan["bottle_world"], dtype=np.float64),
        np.asarray(plan["box_world"], dtype=np.float64),
        pose_by_id,
        plan.get("table_bottle_center_mm"),
        plan.get("table_bottle_axis_deg"),
        n_jobs,
    )


if __name__ == "__main__":
    recipe.visualize_path = OUT_DIR
    # These workers isolate WAPR's OpenGL renderer from SAPIEN's Vulkan context.
    # These functions above batch every object of one RGB-D in one estimator call.
    # 子进程隔离 WAPR 的 OpenGL 渲染器与 SAPIEN 的 Vulkan 上下文。
    # 上方的估计函数将同一 RGB-D 的所有物体收集后，一次调用估计器。
    if len(sys.argv) >= 3 and sys.argv[1] == "estimate":
        estimate_snapshot(sys.argv[2])
        sys.exit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "sequence":
        estimate_sequence(sys.argv[2])
        sys.exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == "track":
        write_tracking_video()
        sys.exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == "compose":
        recompose_tracking()
        sys.exit(0)
    # Prepare optional simulation libraries before their first direct import.
    # 首次直接导入仿真库前准备可选依赖；估计与重绘子入口不触发安装。
    from wapr.bootstrap import ensure_optional
    ensure_optional("robot")
    import imageio.v2 as imageio
    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    # 1. Prepare known CADs in meters and observe the bottle and box before moving.
    # 1. 准备以米为单位的已知 CAD，在手臂运动前观测瓶子与盒子。
    os.makedirs(OUT_DIR, exist_ok=True)
    files = write_known_meshes(os.path.join(OUT_DIR, "meshes"))
    env = make_env()
    env.reset(seed=0)
    true_bottle = bottle_mesh_matrix(actor_matrix(env.unwrapped.bottle))
    true_box = actor_matrix(env.unwrapped.box)
    rgb, depth_m, K, extrinsic = capture_camera(env, "table_camera")
    imageio.imwrite(os.path.join(OUT_DIR, "table_rgb.png"), rgb)
    bottle_mask = color_mask(rgb, "bottle")
    box_mask = color_mask(rgb, "box")
    print("MASK bottle", int(bottle_mask.sum()), "box", int(box_mask.sum()), flush=True)
    if int(bottle_mask.sum()) < TRACK_MIN_MASK_PX or int(box_mask.sum()) < TRACK_MIN_MASK_PX:
        raise RuntimeError("table mask")
    # True poses below are retained for evaluation; they do not gate inference.
    # 真值位姿保留给后续评价，不作为推理执行条件。
    table_snapshot = save_snapshot(
        os.path.join(OUT_DIR, "table.json"),
        rgb, depth_m, K,
        [
            {
                "name": "bottle",
                "mask": bottle_mask,
                "mesh_path": files["bottle"]["path"],
                "diameter_m": files["bottle"]["diameter_m"],
            },
            {
                "name": "box",
                "mask": box_mask,
                "mesh_path": files["box"]["path"],
                "diameter_m": files["box"]["diameter_m"],
            },
        ],
    )
    # The arm has not moved. Close this camera session before the child opens OpenGL.
    # 手臂还没动。子进程打开 OpenGL 之前，先关掉这次相机。
    env.close()
    # 2. The worker calls WAPREstimator.estimate_many_categories_many_instances
    # once for bottle + box. No annotated pose is an inference input.
    # 2. 子进程对瓶子与盒子共同调用一次 WAPREstimator.estimate_many_categories_many_instances。
    # 标注位姿不参与推理。
    table_poses = run_estimate(table_snapshot)
    # Extrinsic is world-to-camera; inverse(extrinsic) @ pose_cam gives world pose.
    # 外参为世界到相机；外参的逆乘相机中的物体位姿，得到世界位姿。
    bottle_world = world_pose(extrinsic, table_poses["bottle"])
    box_world = world_pose(extrinsic, table_poses["box"])
    bottle_center_mm, bottle_axis_deg = pose_error(bottle_world, true_bottle)
    box_center_mm, box_axis_deg = pose_error(box_world, true_box)
    print(
        "TABLE bottle_mm", round(bottle_center_mm, 1),
        "bottle_axis_deg", round(bottle_axis_deg, 1),
        "box_mm", round(box_center_mm, 1),
        "box_axis_deg", round(box_axis_deg, 1),
        flush=True,
    )
    right = table_right(extrinsic)
    target_m = place_center_for(box_world, right)
    grasp = side_grasp_from_world(bottle_world)
    planned = planned_place(grasp, bottle_world, target_m)

    # 3. Execute the same scene twice, preserving the original comparison recipe.
    # The second trial uses the object-to-tool transform measured during the first.
    # 3. 同一场景运行两次，保留原有对照方案。
    # 第二次使用第一次腕部测得的物体到工具变换，修正放置工具位姿。
    meshes = {"files": files, "bottle_world": bottle_world}
    tcp_object = None
    for label in ("uncorrected", "corrected"):
        place_pose = planned if label == "uncorrected" else corrected_pose
        folder = os.path.join(OUT_DIR, label)
        os.makedirs(folder, exist_ok=True)
        env = make_env()
        planner = None
        frames = []
        last_action = None
        raw_step = env.step
        held_active = False
        render_camera = None

        def step_and_record(action):
            """Advance physics and record the camera after every planner step.

            规划器每执行一步，就推进物理并记录相机；此循环不执行位姿推理。
            """
            global last_action
            last_action = action
            outcome = raw_step(action)
            frame = render_frame(env)
            if frame is not None and tcp_object is not None:
                params = render_camera.get_params()
                ext = as_numpy_matrix(params["extrinsic_cv"])
                if ext.shape == (3, 4):
                    full = np.eye(4, dtype=np.float64)
                    full[:3] = ext
                    ext = full
                intrinsic = as_numpy_matrix(params["intrinsic_cv"])[:3, :3]
                frame = draw_axes(frame, ext, intrinsic, tcp_matrix(env) @ tcp_object, 0.06)
            if frame is not None:
                if held_active and len(frames) % 4 == 0:
                    hand_rgb, _depth, _K, _ext = capture_camera(env, "hand_camera")
                    frame = paste_inset(frame, hand_rgb)
                frames.append(frame)
            return outcome

        try:
            env.reset(seed=0)
            render_camera = env.unwrapped._human_render_cameras["render_camera"]
            env.step = step_and_record
            planner = PandaArmMotionPlanningSolver(
                env, debug=False, vis=False, base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False, print_env_info=False,
            )
            first = render_frame(env)
            if first is not None:
                frames.append(first)
            # Approach from the side, close at the estimated bottle pose, then lift.
            # 从侧面接近，在估计瓶子位姿处合爪，再竖直抬升。
            reach = grasp * sapien.Pose([0, 0, -PREGRASP_M])
            move_arm(planner, reach)
            move_arm(planner, grasp)
            planner.close_gripper()
            held_active = True
            lift = sapien.Pose(p=np.array(grasp.p) + [0.0, 0.0, LIFT_M], q=grasp.q)
            move_arm(planner, lift)
            lift_tcp = tcp_matrix(env)
            true_lift_trial = bottle_mesh_matrix(actor_matrix(env.unwrapped.bottle))
            # Save wrist RGB-D for measuring the actual object-to-tool transform.
            # 保存腕部 RGB-D，用于测量实际夹持后的物体到工具变换。
            lift_rgb, lift_depth, lift_K, lift_extrinsic = capture_camera(env, "hand_camera")
            lift_mask = color_mask(lift_rgb, "bottle")
            if int(lift_mask.sum()) < TRACK_MIN_MASK_PX:
                imageio.imwrite(os.path.join(folder, "lift_rgb.png"), lift_rgb)
                raise RuntimeError("hand mask")
            lift_snapshot = save_snapshot(os.path.join(folder, "lift.json"), lift_rgb, lift_depth, lift_K,
                [{"name": "bottle", "mask": lift_mask, "mesh_path": files["bottle"]["path"],
                  "diameter_m": files["bottle"]["diameter_m"]}])
            # Move above the destination first; this avoids crossing the table at grasp height.
            # 先移动到目标上方，再下降，避免在抓取高度横穿桌面。
            above = sapien.Pose(p=np.array(place_pose.p) + [0.0, 0.0, LIFT_M], q=place_pose.q)
            move_arm(planner, above)
            hover_tcp = tcp_matrix(env)
            hover_rgb, hover_depth, hover_K, hover_extrinsic = capture_camera(env, "hand_camera")
            hover_mask = color_mask(hover_rgb, "bottle")
            hover_snapshot = None
            if int(hover_mask.sum()) >= TRACK_MIN_MASK_PX:
                hover_snapshot = save_snapshot(os.path.join(folder, "hover.json"), hover_rgb, hover_depth, hover_K,
                    [{"name": "bottle", "mask": hover_mask, "mesh_path": files["bottle"]["path"],
                      "diameter_m": files["bottle"]["diameter_m"]}])
            # Lower, open the fingers, retreat, and simulate the settling period.
            # 下降、张开夹爪、撤回，再推进等待物体稳定的仿真步。
            move_arm(planner, place_pose)
            planner.open_gripper()
            retreat = sapien.Pose(p=np.array(place_pose.p) + [0.0, 0.0, LIFT_M], q=place_pose.q)
            move_arm(planner, retreat)
            for _index in range(SETTLE_STEPS):
                step_and_record(last_action)
            trial = {"lift_tcp": lift_tcp, "lift_snapshot": lift_snapshot, "lift_rgb": lift_rgb,
                     "lift_K": lift_K, "lift_extrinsic": lift_extrinsic, "hover_tcp": hover_tcp,
                     "hover_snapshot": hover_snapshot, "hover_extrinsic": hover_extrinsic,
                     "true_lift": true_lift_trial, "tilt_deg": bottle_tilt_deg(env),
                     "center_m": bottle_center_m(env)}
            if frames:
                imageio.mimwrite(os.path.join(folder, "place.mp4"), frames, fps=20)
                imageio.imwrite(os.path.join(OUT_DIR, label + "_last.png"), frames[-1])
        finally:
            if planner is not None:
                planner.close()
            env.close()
        print("PLACE", label, "tilt_deg", trial["tilt_deg"], "center_m", trial["center_m"].tolist(), flush=True)
        if label == "uncorrected":
            uncorrected = trial
            # 4. Estimate wrist object pose, then T_tool_object = inverse(T_world_tool) @ T_world_object.
            # 4. 估计腕部物体位姿，再由世界工具位姿的逆求物体到工具变换。
            lift_poses = run_estimate(uncorrected["lift_snapshot"])
            measured_world = world_pose(uncorrected["lift_extrinsic"], lift_poses["bottle"])
            tcp_object = np.linalg.inv(uncorrected["lift_tcp"]) @ measured_world
            true_lift = uncorrected["true_lift"]
            lift_center_mm, lift_axis_deg = pose_error(measured_world, true_lift)
            offset_mm = float(np.linalg.norm(tcp_object[:3, 3]) * 1000.0)
            follow_mm = None
            follow_deg = None
            if uncorrected["hover_snapshot"] is not None:
                hover_poses = run_estimate(uncorrected["hover_snapshot"])
                hover_world = world_pose(uncorrected["hover_extrinsic"], hover_poses["bottle"])
                hover_tcp_object = np.linalg.inv(uncorrected["hover_tcp"]) @ hover_world
                follow_mm = float(np.linalg.norm(hover_tcp_object[:3, 3] - tcp_object[:3, 3]) * 1000.0)
                follow_deg = rotation_angle_deg(hover_tcp_object, tcp_object)
            # 5. Solve the corrected tool pose so the held bottle stands upright at target_m.
            # 5. 求修正后的工具位姿，使实际夹持的瓶子直立落在 target_m。
            up_sign = 1.0 if float(measured_world[2, 2]) >= 0.0 else -1.0
            corrected_pose = corrected_place(tcp_object, target_m, up_sign)
        else:
            corrected = trial
    target_xy = target_m[:2]
    report = {
        "table_bottle_center_mm": bottle_center_mm,
        "table_bottle_axis_deg": bottle_axis_deg,
        "table_box_center_mm": box_center_mm,
        "table_box_axis_deg": box_axis_deg,
        "hand_center_mm": lift_center_mm,
        "hand_axis_deg": lift_axis_deg,
        "tcp_offset_mm": offset_mm,
        "follow_delta_mm": follow_mm,
        "follow_delta_deg": follow_deg,
        "target_m": target_m.tolist(),
        "uncorrected_tilt_deg": uncorrected["tilt_deg"],
        "corrected_tilt_deg": corrected["tilt_deg"],
        "uncorrected_xy_mm": float(np.linalg.norm(uncorrected["center_m"][:2] - target_xy) * 1000.0),
        "corrected_xy_mm": float(np.linalg.norm(corrected["center_m"][:2] - target_xy) * 1000.0),
    }
    save_pose_images(
        OUT_DIR,
        {
            "rgb": rgb,
            "K": K,
            "extrinsic": extrinsic,
            "poses": table_poses,
        },
        {
            "rgb": uncorrected["lift_rgb"],
            "K": uncorrected["lift_K"],
            "extrinsic": uncorrected["lift_extrinsic"],
            "pose": lift_poses["bottle"],
        },
        files,
        {
            "bottle": true_bottle,
            "box": true_box,
            "bottle_at_lift": true_lift,
        },
    )
    compose_flow(OUT_DIR)
    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("REPORT", json.dumps(report), flush=True)
    print("WROTE", OUT_DIR, flush=True)
