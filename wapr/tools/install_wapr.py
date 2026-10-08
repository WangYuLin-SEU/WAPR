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

import ctypes.util
import os
import shutil
import subprocess
import sys


# Select the installation stages; bootstrap prepares the shared dependency/cache layout.
# 选择安装阶段；bootstrap 统一准备依赖与缓存目录。仅位姿使用时可关闭检测阶段。
prepare_2d_detector = True

# Build the DINOv2 TensorRT engine during installation, before the first 2D call.
# 安装时提前构建 DINOv2 TensorRT 引擎，避免首次 2D 调用再等待编译。
build_dino_engine_now = True

script_dir = os.path.dirname(os.path.abspath(__file__))
# This source-only tool is two directory levels below the release root.
# 这个仅供源码使用的工具位于发布根目录下两级。
release_dir = os.path.dirname(os.path.dirname(script_dir))
requirements_path = os.path.join(release_dir, "requirements.txt")
detector_requirements_path = os.path.join(release_dir, "requirements-detector.txt")
native_dir = os.path.join(release_dir, "wapr", "ogl_native")
if release_dir not in sys.path:
    sys.path.insert(0, release_dir)
from wapr.resources import weights_dir as cached_weights_dir, cache_dir

weights_dir = cached_weights_dir()
detector_weights_dir = os.path.join(weights_dir, "det2d")


def check_host():
    """Return a build environment after checking the renderer's host tools.

    检查渲染器所需的解释器和系统工具，并返回含 nvcc 路径的构建环境。
    """
    if sys.version_info[:2] < (3, 10):
        raise RuntimeError("Use Python 3.10 or newer for source setup / 源码环境使用 Python 3.10 或更新版本: " + sys.executable)
    if sys.version_info[:2] != (3, 10):
        print("INSTALL_NOTE", "Source build with an unverified Python version; cp310 wheels require Python 3.10 / 当前 Python 版本尚未验证；cp310 wheel 仍需要 Python 3.10", flush=True)
    if sys.platform != "linux":
        from wapr.ogl import selected_renderer
        print("INSTALL_HOST", {"python": sys.executable, "renderer": selected_renderer(), "release": release_dir}, flush=True)
        return os.environ.copy()
    for program in ("cmake", "c++"):
        if shutil.which(program) is None:
            raise RuntimeError("Missing system tool / 缺少系统工具: " + program)
    from wapr.bootstrap import _find_nvcc
    nvcc_path = _find_nvcc()
    if nvcc_path is None:
        raise RuntimeError("Missing CUDA compiler / 缺少 CUDA 编译器 nvcc；安装适合当前 GPU 的 CUDA toolkit")
    nvcc_version = subprocess.check_output([nvcc_path, "--version"], text=True)
    for library in ("EGL", "GL"):
        if ctypes.util.find_library(library) is None:
            raise RuntimeError("Missing system library / 缺少系统库: lib" + library)
    if not os.path.isfile(requirements_path):
        raise FileNotFoundError(requirements_path)
    build_env = os.environ.copy()
    nvcc_bin_dir = os.path.dirname(nvcc_path)
    build_env["PATH"] = nvcc_bin_dir + os.pathsep + build_env.get("PATH", "")
    print("INSTALL_HOST", {"python": sys.executable, "nvcc": nvcc_path, "nvcc_version": nvcc_version.strip(), "release": release_dir}, flush=True)
    return build_env


def check_detector_sources():
    """Require the project-compatible local detector checkouts.

    检查项目兼容的检测器源码。上游 GroundingDINO 的 CUDA 扩展路径不能直接代替。
    """
    grounding_attention = os.path.join(
        cache_dir(), "sources", "GroundingDINO", "groundingdino",
        "models", "GroundingDINO", "ms_deform_attn.py",
    )
    source_files = (
        grounding_attention,
        os.path.join(cache_dir(), "sources", "GroundingDINO", "groundingdino", "config", "groundingdino_swinb.json"),
        os.path.join(cache_dir(), "sources", "GroundingDINO", "groundingdino", "config", "groundingdino_swint.json"),
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
    from importlib import metadata
    print("INSTALL_2D_SOURCES", {"ultralytics": metadata.version("ultralytics"),
                                "dinov2": "official torch.hub"}, flush=True)


def main():
    """Install packages, samples, and renderer; leave pose engine export explicit.

    按顺序安装 Python 包、样例和本地编译的渲染器；位姿引擎由用户显式导出。
    """
    from wapr.bootstrap import prepare_feature, prepare_runtime
    prepare_runtime()
    if prepare_2d_detector:
        prepared = prepare_feature("det2d")
        if prepared.get("status") != "ready":
            raise RuntimeError(str(prepared))
    check_host()
    print(
        "INSTALL_CHOICES",
        {"prepare_2d_detector": prepare_2d_detector, "build_dino_engine_now": build_dino_engine_now},
        flush=True,
    )
    if prepare_2d_detector:
        check_detector_sources()

    # Check user-selected GPU packages before pip can resolve transitive torch dependencies.
    # 在 pip 解析间接 torch 依赖前，先检查用户选择的 GPU 包，不自动替换 GPU 软件栈。
    try:
        import cv2
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Prepare CUDA-enabled PyTorch and one OpenCV distribution first. / "
            "请先准备带 CUDA 的 PyTorch 及一种 OpenCV 发行包。"
        ) from exc
    trt = None
    from wapr.recipe import resolve_backend
    if resolve_backend() == "trt":
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "Prepare TensorRT 10.x before the trt backend. / 使用 trt 后端前请先准备 TensorRT 10.x。"
            ) from exc
        if not trt.__version__.startswith("10."):
            raise RuntimeError("This engine API uses TensorRT 10.x / 当前引擎 API 使用 TensorRT 10.x: " + trt.__version__)
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access a CUDA GPU / PyTorch 无法访问 CUDA GPU；检查驱动与容器 GPU 映射")
    if prepare_2d_detector:
        import torchvision
        # Check the compiled operator before installing additional detector packages.
        # 安装检测器附加依赖前，检查 torchvision 编译算子是否可用。
        torchvision.ops.nms(torch.empty((0, 4)), torch.empty(0), 0.5)
        if not os.path.isfile(detector_requirements_path):
            raise FileNotFoundError(detector_requirements_path)

    # Shared bootstrap has resolved dependencies and approved every replacement above.
    # 上面的共用 bootstrap 已解析依赖并确认更换计划，不再重复直接调用 pip。
    import numpy as np
    from wapr.bootstrap import cuda_device_index
    ordinal = cuda_device_index()
    gpu_capability = torch.cuda.get_device_capability(ordinal)
    if trt is not None and gpu_capability < (7, 5):
        raise RuntimeError("TensorRT 10 requires NVIDIA SM 7.5+ / TensorRT 10 需要 NVIDIA SM 7.5 及以上")
    print(
        "INSTALL_PACKAGES",
        {"numpy": np.__version__, "torch": torch.__version__,
         "tensorrt": None if trt is None else trt.__version__, "opencv": cv2.__version__},
        flush=True,
    )
    print(
        "INSTALL_GPU",
        {"name": torch.cuda.get_device_name(ordinal), "compute_capability": gpu_capability, "torch_cuda": torch.version.cuda},
        flush=True,
    )

    # Prepare the pose weights and LM-O samples used by the basic workflows.
    # 准备基础流程使用的位姿权重和 LM-O 小样。
    from wapr.download_assets import check_and_fetch_pack

    print("INSTALL_STAGE", "Pose weights and LM-O samples / 位姿权重与 LM-O 小样", flush=True)
    for pack_name in ("wapr_sapr_wbps", "pose_lmo", "lmo"):
        check_and_fetch_pack(pack_name)

    if os.environ.get("WAPR_RENDERER", "").strip().lower() != "nvdiffrast":
        print("INSTALL_STAGE", "OpenGL/CUDA renderer / OpenGL/CUDA 渲染器", flush=True)
        from wapr.ogl import ensure_ogl
        ensure_ogl()
    else:
        print("INSTALL_STAGE", "CUDA rasterizer builds on the first pose call / CUDA 光栅器在首次位姿调用时编译", flush=True)

    # Explicit export can prepare engines ahead of inference; partial sets are not overwritten.
    # 显式导出可提前准备引擎；不自动覆盖不完整的已有引擎组。
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
