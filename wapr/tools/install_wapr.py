# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Prepare the local WAPR inference environment for the initial pose and 2D examples.

Run from the release root: python wapr/tools/install_wapr.py.
为初始位姿与 2D 检测示例准备本机 WAPR 推理环境。从发布根目录运行：python wapr/tools/install_wapr.py。
"""

import ast
import ctypes.util
import os
import shutil
import subprocess
import sys


# Select the installation stages here. The detector needs the compatible source
# trees under third_party/; set this to False for pose-only examples 02 and 02.
# 在这里选择安装阶段。2D 检测需要 third_party/ 下的兼容源码；仅运行位姿示例
# 01、02 时可设为 False。
prepare_2d_detector = True

# Build the DINOv2 TensorRT engine during installation, before the first 2D call.
# 安装时提前构建 DINOv2 TensorRT 引擎，避免首次 2D 调用再等待编译。
build_dino_engine_now = True

script_dir = os.path.dirname(os.path.abspath(__file__))
# This source-only tool is two directory levels below the release root.
# 这个仅供源码使用的工具位于发布根目录下两级。
release_dir = os.path.dirname(os.path.dirname(script_dir))
requirements_path = os.path.join(release_dir, "requirements.txt")
native_dir = os.path.join(release_dir, "wapr", "ogl_native")
weights_dir = os.path.join(release_dir, "assets", "weights")
detector_weights_dir = os.path.join(weights_dir, "det2d")


def check_host():
    """Return a build environment after checking the renderer's host tools.

    检查渲染器所需的解释器和系统工具，并返回含 nvcc 路径的构建环境。
    """
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("WAPR requires Python 3.10 / WAPR 需要 Python 3.10: " + sys.executable)
    if sys.platform != "linux":
        raise RuntimeError("The EGL/CUDA renderer currently requires Linux / EGL/CUDA 渲染器目前需要 Linux")
    for program in ("cmake", "c++"):
        if shutil.which(program) is None:
            raise RuntimeError("Missing system tool / 缺少系统工具: " + program)
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        # CUDA is often installed here without adding its bin directory to PATH.
        # CUDA 常安装在这里，但其 bin 目录未加入 PATH。
        cuda_home = os.environ.get("CUDA_HOME", "")
        candidates = (
            os.path.join(cuda_home, "bin", "nvcc") if cuda_home else "",
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-12.8/bin/nvcc",
        )
        for candidate in candidates:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                nvcc_path = candidate
                break
    if nvcc_path is None:
        raise RuntimeError("Missing CUDA compiler / 缺少 CUDA 编译器 nvcc；安装 CUDA toolkit 12.8")
    nvcc_version = subprocess.check_output([nvcc_path, "--version"], text=True)
    if "release 12.8" not in nvcc_version:
        raise RuntimeError("CUDA toolkit 12.8 is required by this installation recipe / 本安装脚本需要 CUDA toolkit 12.8: " + nvcc_path)
    for library in ("EGL", "GL"):
        if ctypes.util.find_library(library) is None:
            raise RuntimeError("Missing system library / 缺少系统库: lib" + library)
    if not os.path.isfile(requirements_path):
        raise FileNotFoundError(requirements_path)
    build_env = os.environ.copy()
    nvcc_bin_dir = os.path.dirname(nvcc_path)
    build_env["PATH"] = nvcc_bin_dir + os.pathsep + build_env.get("PATH", "")
    print("INSTALL_HOST", {"python": sys.executable, "nvcc": nvcc_path, "release": release_dir}, flush=True)
    return build_env


def check_detector_sources():
    """Require the project-compatible local detector checkouts.

    检查项目兼容的检测器源码。上游 GroundingDINO 的 CUDA 扩展路径不能直接代替。
    """
    grounding_attention = os.path.join(
        release_dir, "third_party", "GroundingDINO", "groundingdino",
        "models", "GroundingDINO", "ms_deform_attn.py",
    )
    ultralytics_init = os.path.join(
        release_dir, "third_party", "ultralytics", "ultralytics", "__init__.py",
    )
    source_files = (
        grounding_attention,
        os.path.join(release_dir, "third_party", "GroundingDINO", "groundingdino", "config", "groundingdino_swinb.json"),
        os.path.join(release_dir, "third_party", "GroundingDINO", "groundingdino", "config", "groundingdino_swint.json"),
        ultralytics_init,
        os.path.join(release_dir, "third_party", "dinov2", "hubconf.py"),
    )
    missing = [path for path in source_files if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "2D source missing / 2D 源码缺失: " + ", ".join(missing)
            + ". See https://wangyulin-seu.github.io/WAPR/docs/install-core.html#third-party / 参见安装文档第三方源码一节。"
        )
    with open(grounding_attention, encoding="utf-8") as stream:
        attention_source = stream.read()
    if "MultiScaleDeformableAttnFunction.apply(" in attention_source:
        raise RuntimeError(
            "GroundingDINO still calls its CUDA extension. Apply the attention"
            " adaptation at https://wangyulin-seu.github.io/WAPR/docs/install-core.html#groundingdino before installation. / "
            "GroundingDINO 仍调用 CUDA 扩展；请先按安装文档修改注意力调用。"
        )
    if "multi_scale_deformable_attn_pytorch(" not in attention_source:
        raise RuntimeError("GroundingDINO PyTorch attention implementation is missing / 缺少 PyTorch 注意力实现")
    with open(ultralytics_init, encoding="utf-8") as stream:
        ultralytics_tree = ast.parse(stream.read(), filename=ultralytics_init)
    ultralytics_version = None
    for statement in ultralytics_tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id == "__version__":
            if isinstance(statement.value, ast.Constant):
                ultralytics_version = statement.value.value
            break
    if ultralytics_version != "8.3.70":
        raise RuntimeError("Expected Ultralytics v8.3.70 source / 需要 Ultralytics v8.3.70 源码")
    print("INSTALL_2D_SOURCES", "required files, attention path, and Ultralytics 8.3.70 present / 必需文件、注意力路径及 Ultralytics 8.3.70 已就绪", flush=True)


def main():
    """Install packages, samples, and renderer; leave pose engine export explicit.

    按顺序安装 Python 包、样例和本地编译的渲染器；位姿引擎由用户显式导出。
    """
    build_env = check_host()
    print(
        "INSTALL_CHOICES",
        {"prepare_2d_detector": prepare_2d_detector, "build_dino_engine_now": build_dino_engine_now},
        flush=True,
    )
    if prepare_2d_detector:
        check_detector_sources()

    # Use the current interpreter so venv/conda and the compiled module agree.
    # 使用当前解释器，确保虚拟环境与编译出的 Python 模块一致。
    print("INSTALL_STAGE", "Python packages / Python 依赖", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])

    import numpy as np
    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access a CUDA GPU / PyTorch 无法访问 CUDA GPU；检查驱动与容器 GPU 映射")
    gpu_capability = torch.cuda.get_device_capability(0)
    if gpu_capability < (7, 5):
        raise RuntimeError("TensorRT 10 requires NVIDIA SM 7.5+ / TensorRT 10 需要 NVIDIA SM 7.5 及以上")
    print(
        "INSTALL_PACKAGES",
        {"numpy": np.__version__, "torch": torch.__version__, "tensorrt": trt.__version__},
        flush=True,
    )
    print(
        "INSTALL_GPU",
        {"name": torch.cuda.get_device_name(0), "compute_capability": gpu_capability, "torch_cuda": torch.version.cuda},
        flush=True,
    )

    # Prepare the pose weights and LM-O samples used by the basic workflows.
    # 准备基础流程使用的位姿权重和 LM-O 小样。
    sys.path.insert(0, release_dir)
    from wapr.download_assets import check_and_fetch_pack

    print("INSTALL_STAGE", "Pose weights and LM-O samples / 位姿权重与 LM-O 小样", flush=True)
    for pack_name in ("wapr_sapr_wbps", "pose_lmo", "lmo"):
        check_and_fetch_pack(pack_name)

    print("INSTALL_STAGE", "EGL/CUDA renderer / EGL/CUDA 渲染器", flush=True)
    subprocess.check_call([
        "cmake", "-S", native_dir, "-B", os.path.join(native_dir, "build"),
        "-DCMAKE_BUILD_TYPE=Release", "-DPython3_EXECUTABLE=" + sys.executable,
    ], env=build_env)
    subprocess.check_call(["cmake", "--build", os.path.join(native_dir, "build"), "-j"], env=build_env)

    # The source checkout never silently accepts or replaces old fixed-group
    # plans. Explicit export builds four model engines plus the grouped WBPS plan.
    # 源码目录不静默接受或替换旧引擎；显式导出构建四个模型引擎及 WBPS 批量推理引擎。
    print("INSTALL_POSE_ENGINE_NEXT", "python -m wapr.export_engines", flush=True)

    if prepare_2d_detector:
        from wapr.det2d import ensure_dino_engine, prepare_det2d_weights

        print("INSTALL_STAGE", "2D model weights / 2D 模型权重", flush=True)
        prepare_det2d_weights(detector_weights_dir, dino="vitl14", grounding="swinb")
        if build_dino_engine_now:
            print("INSTALL_STAGE", "DINOv2 TensorRT engine / DINOv2 TensorRT 引擎", flush=True)
            ensure_dino_engine(detector_weights_dir, device="cuda:0", dino="vitl14")

    print("INSTALL_DONE", "Run python examples/02_one_category_one_instance.py / 可以运行示例 02", flush=True)


if __name__ == "__main__":
    main()
