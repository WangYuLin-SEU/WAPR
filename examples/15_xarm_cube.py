# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 15. Previous: examples/14_bridge_tasks.py. Next: examples/16_follow_saved.py
# 示例 15。上一例：examples/14_bridge_tasks.py。下一例：examples/16_follow_saved.py
# xArm6 on the ManiSkill table. The object is that task's cube, not a YCB mesh.
# xArm6，ManiSkill 的桌子。物体是这个任务自己的方块，不是 YCB 网格。
# python examples/15_xarm_cube.py
# 运行：python examples/15_xarm_cube.py
"""Pick the red cube with xArm6 and set it down at the green marker.

xArm6 抓起红色方块，放到绿色标记的位置。

The arm grasps the first-frame estimate. The mask is the red cube, not a simulator id.
手臂抓第 0 帧估计到的位置。mask 是红色方块，不是仿真器编号。
"""

import importlib
import os
import sys

import numpy as np


# Reuse camera/geometry helpers from the adjacent public recipes 12 and 13.
# 从相邻的公开示例 13 与 13 复用相机和几何函数。
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.dirname(EXAMPLES_DIR)
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)

bridge = importlib.import_module("14_bridge_tasks")
known = importlib.import_module("13_known_mesh_place")

OUT_DIR = os.path.join(RELEASE_DIR, "outputs", "sim_xarm_cube")
# Overview of the ManiSkill table, meters. The xArm6 base sits behind the table.
# ManiSkill 桌子的总览，米。xArm6 的底座在桌子后面。
EYE_M = np.array([0.60, 0.70, 0.60], dtype=np.float64)
TARGET_M = np.array([0.00, 0.00, 0.20], dtype=np.float64)
# How far above the cube the open gripper stops, then how far it lifts, meters.
# 张开的夹爪停在方块上方多高，再提起多高，米。
PREGRASP_M = 0.08
LIFT_M = 0.10


def make_env():
    """PickCube with xArm6, joint position control.

    PickCube，手臂是 xArm6，关节位置控制。
    """
    # Prepare simulation dependencies only when creating the environment.
    # 仅在创建仿真环境时准备机器人可选依赖。
    from wapr.bootstrap import ensure_optional
    ensure_optional("robot")
    import gymnasium as gym

    import mani_skill.envs  # noqa: F401

    # Repair interrupted asset downloads before constructing the robot.
    # 构造机械臂前补齐被中断的资源下载。
    from wapr.source_setup import prepare_robot_assets
    prepare_robot_assets("xarm6")

    env = gym.make(
        "PickCube-v1",
        robot_uids="xarm6_robotiq",
        obs_mode="rgb",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="cpu",
    )
    env._max_episode_steps = 2000
    return env


def make_planner(env):
    """mplib on the xArm6 tool link. The base pose is the table mount.

    mplib 规划 xArm6 的工具连杆。底座位姿是桌子上的安装位置。
    """
    import sapien

    from mani_skill.examples.motionplanning.xarm6.motionplanner import (
        XArm6RobotiqMotionPlanningSolver,
    )

    robot_pose = env.unwrapped.agent.robot.pose
    mount = sapien.Pose(robot_pose.p[0].cpu().numpy(), robot_pose.q[0].cpu().numpy())
    return XArm6RobotiqMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=mount,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )


def top_grasp(env, center):
    """Gripper pointing down. Tool +Z is the approach.

    夹爪朝下。工具的 +Z 是接近方向。
    """
    approaching = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    closing = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return env.unwrapped.agent.build_grasp_pose(approaching, closing, center)


def redraw():
    """Redraw saved cube frames without another simulation or estimate.

    使用保存的方块帧重绘，不重新仿真或估计。
    """
    import json
    import trimesh

    folder = os.path.join(OUT_DIR, "cube")
    with open(os.path.join(folder, "rows.json"), "r", encoding="utf-8") as stream:
        rows = json.load(stream)
    poses = bridge.load_poses(folder)
    mesh = trimesh.load(os.path.join(folder, "mesh.ply"), force="mesh", process=False)
    task = {"key": "xarm_cube", "env_id": "PickCube-v1", "zh": "xArm6 把方块放到标记处",
            "en": "xArm6 sets the cube on the marker", "closing": "world_y"}
    bridge.compose_video(task, rows, poses, mesh, "zh")
    report = bridge.compose_video(task, rows, poses, mesh, "en")
    print("REDREW", report["video"], report["frames"], flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "redraw":
        redraw()
        sys.exit(0)
    import json
    import trimesh

    # 1. Create a known cube CAD and xArm6/Robotiq joint-control simulation.
    # 1. 创建已知方块 CAD 与 xArm6/Robotiq 关节控制仿真。
    folder = os.path.join(OUT_DIR, "cube")
    os.makedirs(folder, exist_ok=True)
    env = make_env()
    planner = None
    try:
        env.reset(seed=0)
        base = env.unwrapped
        half_m = float(base.cube_half_size)
        mesh = trimesh.creation.box(extents=[2 * half_m, 2 * half_m, 2 * half_m])
        mesh_path = os.path.join(folder, "mesh.ply")
        mesh.export(mesh_path)
        diameter_m = known.mesh_diameter_m(mesh)
        print("TASK xarm_cube diameter_m", round(diameter_m, 3), flush=True)
        camera = bridge.add_overview(env, EYE_M, TARGET_M)
        base.scene.update_render()
        init_jobs = []
        init_row = bridge.record_frame(
            camera, folder, 0, base.cube, base.cube, mesh_path, diameter_m, init_jobs, "cube",
        )
        if not init_jobs:
            raise RuntimeError("frame 0 mask")
        init_path = os.path.join(folder, "init_jobs.json")
        with open(init_path, "w", encoding="utf-8") as stream:
            json.dump(init_jobs, stream)
        # 2. The worker defined in public example 14 calls WAPREstimator once per RGB-D.
        # It estimates every object in that image together, in a separate OpenGL process.
        # 2. 公开示例 14 的子进程逐 RGB-D 调用 WAPREstimator，将同图所有物体批量估计。
        # 位姿 OpenGL 渲染在独立进程运行，避免与仿真 Vulkan 上下文冲突。
        init_poses = bridge.run_sequence(init_path)
        pose_cam = init_poses.get("f00000")
        if pose_cam is None:
            raise RuntimeError("frame 0 estimate")
        # 3. Convert the estimate to world coordinates; grasp the estimated center.
        # 3. 将估计转换到世界坐标，抓取估计的物体中心；平移单位米。
        world = known.world_pose(np.asarray(init_row["render_extrinsic"], dtype=np.float64), pose_cam)
        grasp_center = world[:3, 3].copy()
        print("GRASP xarm_cube", np.round(grasp_center, 4).tolist(), flush=True)
        goal = base.goal_site.pose.p[0].detach().cpu().numpy().copy()
        # The marker can float. The cube is set down on the table at that xy.
        # 标记可以悬在空中；方块采用标记的水平坐标，中心高度设为半边长以落在桌面上。
        goal[2] = half_m
        planner = make_planner(env)
        rows, jobs = [], []
        frame_index = 0
        original_step = env.step

        def step_and_record(action):
            """Save one RGB-D frame after each planned joint action.

            每次执行规划的关节动作后保存一帧 RGB-D，不在该循环内逐物体推理。
            """
            global frame_index
            outcome = original_step(action)
            rows.append(bridge.record_frame(camera, folder, frame_index, base.cube, base.cube,
                                             mesh_path, diameter_m, jobs, "cube"))
            frame_index += 1
            return outcome

        env.step = step_and_record
        # 4. Open, move above the estimated center, descend, close, and lift vertically.
        # 4. 张开夹爪，移动到估计中心上方，下降、合拢，然后竖直抬升。
        planner.open_gripper()
        center = np.asarray(grasp_center, dtype=np.float64).reshape(3).copy()
        known.move_arm(planner, top_grasp(env, center + [0.0, 0.0, PREGRASP_M]))
        known.move_arm(planner, top_grasp(env, center))
        planner.close_gripper(t=12)
        known.move_arm(planner, top_grasp(env, center + [0.0, 0.0, LIFT_M]))
        # Preserve the original task recipe: marker xy, and the estimated grasp height.
        # 保留原任务设置：标记的水平坐标，加估计出的抓取高度。
        place = np.asarray(goal, dtype=np.float64).copy()
        place[2] = float(center[2])
        known.move_arm(planner, top_grasp(env, place + [0.0, 0.0, LIFT_M]))
        known.move_arm(planner, top_grasp(env, place))
        # Release on the table, then retreat; physics decides whether the cube stays put.
        # 在桌面处松爪，然后撤回；方块是否稳定由物理仿真决定。
        planner.open_gripper(t=8)
        known.move_arm(planner, top_grasp(env, place + [0.0, 0.0, LIFT_M]))
    finally:
        if planner is not None:
            planner.close()
        env.close()
    task = {
        "key": "xarm_cube",
        "env_id": "PickCube-v1",
        "zh": "xArm6 把方块放到标记处",
        "en": "xArm6 sets the cube on the marker",
        "closing": "world_y",
    }
    # record_frame stores the cube pose in gt and also copies it into target.
    # The place target is the marker, written over the last row's comparison below.
    # record_frame 把方块位姿写进 gt，target 里先放了同一份。
    # 放下的目标是标记，比较时用下面这份。
    for row in rows:
        row["target"] = goal.tolist()
    with open(os.path.join(folder, "rows.json"), "w", encoding="utf-8") as stream:
        json.dump(rows, stream)
    jobs_path = os.path.join(folder, "jobs.json")
    with open(jobs_path, "w", encoding="utf-8") as stream:
        json.dump(jobs, stream)
    # 5. Estimate the saved motion frames only for post-hoc overlays and error reports.
    # 5. 估计保存的运动帧，仅用于事后叠图与误差报告；真值不反馈给抓取控制。
    poses = bridge.run_sequence(jobs_path) if jobs else {}
    bridge.compose_video(task, rows, poses, mesh, "zh")
    report = bridge.compose_video(task, rows, poses, mesh, "en")
    print("WROTE", report["video"], flush=True)
