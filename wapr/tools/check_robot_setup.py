# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Check the local prerequisites for robot examples 13–17 without changing them.

只读检查机器人示例 13–17 的本地依赖和输入资源。
Run from the release root: python wapr/tools/check_robot_setup.py
从发布根目录运行：python wapr/tools/check_robot_setup.py
"""

import ctypes.util
import importlib.metadata
import importlib.util
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Source-only tools live under wapr/tools, two levels below the release root.
# 仅供源码使用的工具位于 wapr/tools，发布根目录在其上两级。
RELEASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
EXAMPLES_DIR = os.path.join(RELEASE_DIR, "examples")
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)


def main():
    """Report prerequisites and saved-input readiness for each robot lesson.

    报告各机器人案例的前置条件和已保存输入；不下载、不启动仿真。
    """
    print("ROBOT_PYTHON", sys.executable, sys.version.split()[0], flush=True)
    core_problems = []
    pose_problems = []
    if sys.version_info[:2] != (3, 10):
        core_problems.append("Python 3.10 / Python 3.10")
    if sys.platform != "linux":
        core_problems.append("Linux renderer / Linux 渲染环境")

    # These versions reproduce the environment used for the recorded episodes.
    # 这些版本是已记录仿真回合所用的环境，不是推断出的通用兼容区间。
    tested_versions = {
        "mani-skill": "3.0.1",
        "sapien": "3.0.3",
        "mplib": "0.1.1",
    }
    for package_name in (
        "mani-skill", "sapien", "mplib", "gymnasium", "transforms3d",
        "imageio", "imageio-ffmpeg", "torch", "numpy",
    ):
        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            version = "MISSING"
            core_problems.append(package_name)
        expected = tested_versions.get(package_name)
        print("ROBOT_PACKAGE", package_name, version, flush=True)
        if expected and version != "MISSING" and version != expected:
            print("ROBOT_VERSION_NOTE", package_name, "tested / 已测试", expected, flush=True)

    for module_name in ("cv2", "trimesh", "PIL"):
        found = importlib.util.find_spec(module_name) is not None
        print("ROBOT_MODULE", module_name, "OK" if found else "MISSING", flush=True)
        if not found:
            core_problems.append(module_name)

    vulkan = ctypes.util.find_library("vulkan")
    print("ROBOT_VULKAN", vulkan or "SYSTEM_LIBRARY_MISSING", flush=True)
    if vulkan is None:
        # SAPIEN may use its bundled loader; a rendered frame decides availability.
        # SAPIEN 可能使用内置加载器，是否可渲染应以实际一帧为准。
        print("ROBOT_VULKAN_NOTE", "SAPIEN may provide a loader; run the render smoke check / SAPIEN 可能提供内置加载器，请运行最小渲染检查", flush=True)

    try:
        import torch

        gpu_ready = torch.cuda.is_available()
        print("ROBOT_CUDA", torch.cuda.get_device_name(0) if gpu_ready else "MISSING", flush=True)
        if not gpu_ready:
            core_problems.append("CUDA GPU / CUDA 显卡")
    except ImportError:
        pass

    # Read constants from the public recipes without entering their __main__ guards.
    # 从公开示例读取常量，不进入 __main__ 主流程，避免预检启动仿真。
    try:
        known = importlib.import_module("13_known_mesh_place")
        bridge = importlib.import_module("14_bridge_tasks")
        xarm = importlib.import_module("15_xarm_cube")
    except ImportError as error:
        print("ROBOT_EXAMPLE_IMPORT", "MISSING", str(error), flush=True)
        return 1

    # Pose TensorRT plans are GPU-specific; presence is a static check only.
    # 位姿 TensorRT 引擎与 GPU 相关；这里只检查文件存在，不能判定可反序列化。
    for name in ("wapr_w_mask.engine", "wapr_wo_mask.engine", "sapr.engine", "wbps.engine", "wbps_batch.engine"):
        path = os.path.join(RELEASE_DIR, "assets", "weights", name)
        present = os.path.isfile(path) and os.path.getsize(path) > 0
        print("ROBOT_POSE_ENGINE", name, "OK" if present else "MISSING", flush=True)
        if not present:
            pose_problems.append(name)
    region_path = os.path.join(EXAMPLES_DIR, "16_follow_saved", "region_tracking.py")
    if not os.path.isfile(region_path):
        core_problems.append("published region tracker / 公开区域跟踪模块")

    try:
        mani_skill_root = importlib.metadata.distribution("mani-skill").locate_file("mani_skill")
    except importlib.metadata.PackageNotFoundError:
        mani_skill_root = ""
    motion_files = (
        "examples/motionplanning/panda/motionplanner.py",
        "examples/motionplanning/xarm6/motionplanner.py",
        "examples/motionplanning/so100/motionplanner.py",
        "envs/tasks/digital_twins/bridge_dataset_eval/base_env.py",
    )
    for relative_path in motion_files:
        path = os.path.join(str(mani_skill_root), relative_path)
        present = bool(mani_skill_root) and os.path.isfile(path)
        print("ROBOT_MANISKILL_SOURCE", relative_path, "OK" if present else "MISSING", flush=True)
        if not present:
            core_problems.append(relative_path)

    bridge_problems = []
    for task in bridge.TASKS:
        path = os.path.join(bridge.BRIDGE_ROOT, "custom", "models", task["source"], "collision.obj")
        present = os.path.isfile(path) and os.path.getsize(path) > 0
        print("ROBOT_BRIDGE_ASSET", task["key"], "OK" if present else "MISSING", path, flush=True)
        if not present:
            bridge_problems.append(task["key"])
    asset_override = os.environ.get("MS_ASSET_DIR")
    if asset_override:
        expected_root = os.path.join(os.path.abspath(os.path.expanduser(asset_override)), "tasks", "bridge_v2_real2sim_dataset")
        if os.path.abspath(bridge.BRIDGE_ROOT) != expected_root:
            print("ROBOT_BRIDGE_PATH_NOTE", "Example 14 currently uses / 示例 14 当前使用", bridge.BRIDGE_ROOT, flush=True)
            print("ROBOT_BRIDGE_PATH_NOTE", "ManiSkill asset root would be / 官方资源文件根为", expected_root, flush=True)

    # Example 16 consumes saved rows and poses from 12–14; example 17 uses the same rows.
    # 示例 16 读取 12–14 保存的帧和位姿；示例 17 复用这些帧。
    saved_problems = []
    saved_cases = (
        ("12", os.path.join(known.OUT_DIR, "tracking"), ("rows.json", "sequence_poses.json", "meshes/bottle.ply", "meshes/box.ply")),
        ("13 carrot", os.path.join(bridge.OUT_DIR, "carrot"), ("rows.json", "jobs.json", "jobs_poses.json", "mesh.ply")),
        ("13 eggplant", os.path.join(bridge.OUT_DIR, "eggplant"), ("rows.json", "jobs.json", "jobs_poses.json", "mesh.ply")),
        ("14", os.path.join(xarm.OUT_DIR, "cube"), ("rows.json", "jobs.json", "jobs_poses.json", "mesh.ply")),
    )
    for label, folder, names in saved_cases:
        missing = [name for name in names if not os.path.isfile(os.path.join(folder, name))]
        print("ROBOT_SAVED_FRAMES", label, "OK" if not missing else "MISSING", ", ".join(missing) if missing else folder, flush=True)
        if missing:
            saved_problems.append(label)
    # Check the encoder actually selected by the shared tracking recipe.
    # 检查共享跟踪配置实际选用的编码器，避免把其他型号的权重误报为已就绪。
    tracker_recipe = importlib.import_module("08_ycbineoat_one_instance")
    from wapr.det2d import DINO_CHOICES, default_weights_dir
    dino_spec = DINO_CHOICES[tracker_recipe.dino_name]
    dino_path = os.path.join(default_weights_dir, dino_spec["file"])
    dino_ready = os.path.isfile(dino_path)
    print("ROBOT_TRACKER_WEIGHT", "OK" if dino_ready else "MISSING", dino_path, flush=True)

    pose_ready = not core_problems and not pose_problems
    print("ROBOT_SETUP", "READY" if pose_ready else "INCOMPLETE", "Static checks only / 仅静态检查", flush=True)
    print("ROBOT_EXAMPLE_13", "READY" if pose_ready and not bridge_problems else "INCOMPLETE", flush=True)
    print("ROBOT_EXAMPLE_15", "READY" if pose_ready and not saved_problems and dino_ready else "INCOMPLETE", flush=True)
    for problem in core_problems + pose_problems:
        print("ROBOT_MISSING", problem, flush=True)
    return 0 if pose_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
