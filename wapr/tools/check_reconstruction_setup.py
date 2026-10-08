# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Check the local dependencies for the single-image reconstruction examples.

检查单图重建示例的本地依赖。只读取文件和环境，不下载或运行推理。
Run from the release root: python wapr/tools/check_reconstruction_setup.py
从发布根目录运行：python wapr/tools/check_reconstruction_setup.py
"""

import importlib.util
import os
import shutil
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Source-only tools live under wapr/tools, two levels below the release root.
# 仅供源码使用的工具位于 wapr/tools，发布根目录在其上两级。
RELEASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CASE_DIR = os.path.join(RELEASE_DIR, "examples", "11_reconstruct_object")
if CASE_DIR not in sys.path:
    sys.path.insert(0, CASE_DIR)
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)


def main():
    """Report missing sources, checkpoints, packages, and runtime requirements.

    报告缺少的源码、权重、Python 包和运行条件；失败时以状态码 1 退出。
    """
    print("RECON_PYTHON", sys.executable, sys.version.split()[0], flush=True)
    problems = []
    # Public algorithm and font resources must exist before costly reconstruction.
    # 耗时重建之前，必须备齐公开算法和字体资源。
    for relative_path in ("examples/11_reconstruct_object/axis_match_scale.py",
                          "examples/11_reconstruct_object/roma_shape.py",
                          "examples/11_reconstruct_object/language_prompt.py",
                          "examples/12_cross_scene_pose.py", "wapr/fonts/wqy-microhei.ttc"):
        path = os.path.join(RELEASE_DIR, relative_path)
        found = os.path.isfile(path) and os.path.getsize(path) > 0
        print("RECON_PUBLIC_RESOURCE", relative_path, "OK" if found else "MISSING", flush=True)
        if not found:
            problems.append(relative_path)
    if sys.version_info[:2] != (3, 10):
        problems.append("Python 3.10 is required / 需要 Python 3.10")
    if sys.platform != "linux":
        problems.append("SAM3D reconstruction requires Linux / SAM3D 重建需要 Linux")

    # Import the paths used by the examples, so this check follows their recipe.
    # 从示例导入实际使用的路径，避免预检与运行脚本各维护一套路径。
    try:
        from step01_point_mask import (
            DATA_ROOT,
            MOGE_CHECKPOINT,
            SAM3D_CONFIG,
            SAM3D_ROOT,
        )
        from step03_box_lengths import UNIPOSE_CFG, UNIPOSE_CKPT, UNIPOSE_INFER
    except ImportError as error:
        print("RECON_MISSING_BASE", str(error), flush=True)
        print("Run the base installation first / 请先完成基础安装：https://wangyulin-seu.github.io/WAPR/docs/install-guide.html", flush=True)
        return 1

    from wapr.resources import weights_dir
    SAM2_CHECKPOINT = os.path.join(weights_dir(), "det2d", "sam2.1_l.pt")
    source_files = (
        ("SAM 3D Objects", os.path.join(SAM3D_ROOT, "sam3d_objects", "pipeline", "inference_pipeline.py")),
        ("UniPose9D", os.path.join(UNIPOSE_INFER, "unipose9d_inference.py")),
    )
    for label, path in source_files:
        found = os.path.isfile(path)
        print("RECON_SOURCE", label, "OK" if found else "MISSING", path, flush=True)
        if not found:
            problems.append(label + " source / 源码")

    checkpoint_files = (
        ("SAM 2", SAM2_CHECKPOINT),
        ("SAM 3D pipeline", SAM3D_CONFIG),
        ("MoGe v1", MOGE_CHECKPOINT),
        ("UniPose9D weights", UNIPOSE_CKPT),
        ("UniPose9D config", UNIPOSE_CFG),
        ("WAPR pose engine", os.path.join(weights_dir(), "wapr_w_mask.engine")),
        ("SAPR pose engine", os.path.join(weights_dir(), "sapr.engine")),
        ("WBPS score engine", os.path.join(weights_dir(), "wbps.engine")),
        ("DINOv2 pose selector", os.path.join(weights_dir(), "det2d", "dinov2_vitl14_pretrain.pth")),
    )
    for label, path in checkpoint_files:
        found = os.path.isfile(path) and os.path.getsize(path) > 0
        if found:
            with open(path, "rb") as stream:
                found = not stream.read(32).startswith(b"version https://git-lfs")
        print("RECON_WEIGHT", label, "OK" if found else "MISSING", path, flush=True)
        if not found:
            problems.append(label + " checkpoint / 权重")

    # Each of these files is named by the tested pipeline.yaml.
    # 这些文件逐一由当前 pipeline.yaml 引用。
    sam3d_files = (
        "ss_generator.yaml", "ss_generator.ckpt",
        "ss_decoder.yaml", "ss_decoder.ckpt",
        "slat_generator.yaml", "slat_generator.ckpt",
        "slat_decoder_mesh.yaml", "slat_decoder_mesh.ckpt",
        "slat_decoder_gs.yaml", "slat_decoder_gs.ckpt",
        "slat_decoder_gs_4.yaml", "slat_decoder_gs_4.ckpt",
    )
    for name in sam3d_files:
        path = os.path.join(os.path.dirname(SAM3D_CONFIG), name)
        found = os.path.isfile(path) and os.path.getsize(path) > 0
        if found:
            with open(path, "rb") as stream:
                found = not stream.read(32).startswith(b"version https://git-lfs")
        print("RECON_SAM3D_FILE", name, "OK" if found else "MISSING", path, flush=True)
        if not found:
            problems.append("SAM 3D " + name)

    # The local SAM 3D changes are needed by the documented UV bake path.
    # 文档中的 UV 烘焙路径需要这份本地 SAM 3D 兼容补丁。
    patch_path = os.path.join(SCRIPT_DIR, "patches", "sam3d_wapr_compat.patch")
    if not os.path.isfile(patch_path):
        print("RECON_SAM3D_PATCH", "MISSING", patch_path, flush=True)
        problems.append("SAM 3D compatibility patch file / SAM 3D 兼容补丁文件")
    elif not os.path.isdir(SAM3D_ROOT):
        print("RECON_SAM3D_PATCH", "SOURCE_MISSING", SAM3D_ROOT, flush=True)
    elif shutil.which("git") is None:
        problems.append("git for patch verification / git 用于检查补丁")
    else:
        result = subprocess.run(
            ["git", "-C", SAM3D_ROOT, "apply", "--reverse", "--check", patch_path],
            capture_output=True, text=True, check=False,
        )
        patched = result.returncode == 0
        print("RECON_SAM3D_PATCH", "OK" if patched else "MISSING", patch_path, flush=True)
        if not patched:
            problems.append("SAM 3D compatibility patch / SAM 3D 兼容补丁")

    for module_name in (
        "torch", "torchvision", "pytorch3d", "xatlas", "omegaconf", "hydra",
        "iopath", "moge", "trimesh",
    ):
        found = importlib.util.find_spec(module_name) is not None
        print("RECON_PACKAGE", module_name, "OK" if found else "MISSING", flush=True)
        if not found:
            problems.append(module_name + " Python package / Python 包")

    try:
        import numpy as np

        numpy_ok = np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion("1.24.4")
        print("RECON_NUMPY", np.__version__, "OK" if numpy_ok else "TOO_OLD", flush=True)
        if not numpy_ok:
            problems.append("NumPy >= 1.24.4 for native SAM 2 / 官方 SAM 2 实现 需要 NumPy >= 1.24.4")
    except ImportError:
        problems.append("NumPy Python package / NumPy Python 包")

    try:
        import torch

        gpu_ok = torch.cuda.is_available()
        print("RECON_GPU", torch.cuda.get_device_name(0) if gpu_ok else "MISSING", flush=True)
        if gpu_ok:
            # The upstream requirement is 32 decimal GB, not 32 GiB.
            # 上游写的是十进制 32 GB，不能按 32 GiB 判定。
            gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1_000_000_000
            print("RECON_GPU_GB", round(gpu_gb, 1), flush=True)
            if gpu_gb < 32.0:
                print("RECON_GPU_NOTE", "Upstream SAM 3D setup specifies at least 32 GB VRAM / 上游 SAM 3D 安装说明要求至少 32 GB 显存", flush=True)
                problems.append("SAM 3D GPU memory >= 32 GB / SAM 3D 显存至少 32 GB")
        else:
            problems.append("CUDA GPU / CUDA 显卡")
    except ImportError:
        pass

    if "RECON_DATA_ROOT" not in os.environ:
        print("RECON_DATA_NOTE", "Set RECON_DATA_ROOT explicitly for example 11 / 示例 11 请显式设置 RECON_DATA_ROOT", flush=True)
    data_ok = os.path.isdir(DATA_ROOT)
    print("RECON_DATA", "OK" if data_ok else "MISSING", DATA_ROOT, flush=True)
    if not data_ok:
        problems.append("YCBInEOAT data root / YCBInEOAT 数据根")

    if problems:
        for problem in problems:
            print("RECON_MISSING", problem, flush=True)
        print("RECON_SETUP", "INCOMPLETE", "See pages/docs/reconstruct.html#setup / 参见重建安装指南", flush=True)
        return 1
    print("RECON_SETUP", "READY", "Inputs and model loading still require a runtime check / 输入与模型加载仍需实际运行检查", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
