# Author: Yulin Wang (yulinwang@seu.edu.cn)
# SPDX-License-Identifier: LGPL-2.1-only
"""Prepare an installed WAPR source wheel in the existing Python environment.

在当前 Python 环境准备 WAPR 源码 wheel；不创建环境，更换已有包须用户明确同意。
Run / 运行: python -m wapr.bootstrap
"""
import ctypes.util
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


_prepared_optional = set()


def check_runtime():
    """Inspect the existing core environment without installing anything.

    只检查当前核心环境，不安装包、不编译、不下载权重。
    """
    from importlib import metadata
    versions = {}
    for name in ("torch", "torchvision", "torchaudio", "numpy", "trimesh", "kornia", "onnx"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    result = {"feature": "core", "python": sys.version.split()[0], "executable": sys.executable,
              "platform": sys.platform, "versions": versions, "status": "needs_preparation"}
    try:
        import torch
        result.update(torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            result.update(gpu=torch.cuda.get_device_name(0), gpu_memory_gb=round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2))
        result["nvcc"] = shutil.which("nvcc") or (
            "/usr/local/cuda/bin/nvcc" if os.path.isfile("/usr/local/cuda/bin/nvcc") else None)
        if sys.platform != "linux" or not torch.cuda.is_available():
            result.update(status="blocked", reason="OGL requires Linux and working CUDA / OGL 需要 Linux 与可用 CUDA")
        elif all(versions[name] is not None for name in ("numpy", "trimesh", "kornia")) and result["nvcc"]:
            result["status"] = "dependencies_present"
            result["note"] = "Presence is not inference validation / 依赖存在不等于推理验证通过"
    except (ImportError, OSError) as error:
        result.update(status="blocked", reason="Existing PyTorch is unavailable / 当前 PyTorch 不可用: " + type(error).__name__)
    return result


def prepare_feature(feature, allow_replacement=None, check_only=False):
    """Prepare one requested feature, preserving unrelated optional stacks.

    只准备被请求的功能；更换已有库须明确同意，不安装无关可选功能。
    """
    from wapr.installation import install_requirements, prepare_optional
    if feature == "core":
        if check_only:
            return check_runtime()
        prepare_runtime(allow_replacement=allow_replacement)
        return {"feature": "core", "status": "ready"}
    if feature == "sam3d":
        from wapr.reconstruction_setup import prepare_reconstruction
        return prepare_reconstruction(allow_replacement=allow_replacement, check_only=check_only)
    if feature == "compatible":
        # Resolve first; replacing an existing package still requires approval.
        # 先逐项解析；更换已有包仍须明确同意，无法准备的功能单独报告。
        results = []
        for name in ("dinov2", "det2d", "robot", "sam2", "roma", "qwen", "sam3d", "unipose9d"):
            plan = prepare_feature(name, allow_replacement=False, check_only=True)
            if not check_only and plan.get("status") in ("ready", "installable", "dependencies_present", "approval_required", "needs_source"):
                try:
                    plan = prepare_feature(name, allow_replacement=allow_replacement)
                except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                    plan = {"feature": name, "status": "blocked", "reason": str(error)}
            results.append(plan)
            if plan.get("status") == "restart_required":
                return {"feature": feature, "status": "restart_required", "features": results,
                        "reason": "Restart Python before preparing further features / 请重启 Python 后继续准备其他功能"}
        incomplete = any(item.get("status") not in ("ready", "installed", "prepared") for item in results)
        return {"feature": feature, "status": "checked" if check_only else ("partial" if incomplete else "prepared"), "features": results}
    if not check_only and feature in _prepared_optional:
        return {"feature": feature, "status": "ready"}
    source = None
    if feature in ("sam2", "roma"):
        from wapr.source_setup import prepare_source
        if check_only:
            from wapr.resources import resource_root, source_checkout
            source_name = "sam2" if feature == "sam2" else "RoMa"
            source_parent = os.path.join(resource_root(), "third_party" if source_checkout else "sources")
            candidate = os.path.join(source_parent, source_name)
            if os.path.isdir(candidate):
                source = prepare_source(feature)
        else:
            source = prepare_source(feature)
    result = prepare_optional(feature, allow_replacement=allow_replacement, check_only=check_only,
                              source_requirement=source)
    if check_only:
        if result.get("status") == "ready" and result.get("install"):
            result["status"] = "installable"
        return result
    if result.get("status") not in ("ready", "installed"):
        return result
    if feature == "unipose9d" and result.get("existing_source_api"):
        _prepared_optional.add(feature)
        return result
    # Source-based projects stay outside site-packages and outside the wheel.
    # 源码项目放在资源目录，不打进 wheel；使用当前解释器解析其实际依赖。
    if feature in ("dinov2", "det2d", "sam2", "roma", "unipose9d"):
        from wapr.source_setup import prepare_source
        if source is None:
            source = prepare_source(feature)
        result["source"] = source
        if feature == "det2d":
            parent = os.path.dirname(source)
            source_paths = [source, os.path.join(parent, "ultralytics"), os.path.join(parent, "dinov2")]
        elif feature == "unipose9d":
            source_paths = [os.path.join(source, "infer")]
        else:
            source_paths = [source]
        for source_path in source_paths:
            if source_path not in sys.path:
                sys.path.insert(0, source_path)
        if feature == "unipose9d":
            import importlib
            module = importlib.import_module("unipose9d_inference")
            for name in ("estimate_pose", "load_pose_model", "set_seed"):
                if not callable(getattr(module, name, None)):
                    raise RuntimeError("UniPose9D API missing / UniPose9D API 缺失: " + name)
    # This marker means installation finished, not that model inference was tested.
    # 此标记只表示安装阶段完成，不表示模型推理已经验证。
    _prepared_optional.add(feature)
    result["status"] = "ready"
    return result


def ensure_optional(feature):
    """First-call preparation; decline or conflict stops this feature.

    首次调用时准备；拒绝更换或依赖冲突时退出该功能。
    """
    result = prepare_feature(feature)
    if result.get("status") != "ready":
        raise RuntimeError("Optional feature preparation stopped / 可选功能准备已停止: " + json.dumps(result, ensure_ascii=False))
    return result


def prepare_runtime(allow_replacement=None):
    """Install missing pose dependencies and compile for the existing CUDA.

    安装缺失的位姿依赖，按当前 CUDA 编译；默认保留已有库，更换须明确同意。
    """
    if sys.platform != "linux":
        raise RuntimeError("This native renderer requires Linux / 本地渲染器需要 Linux")
    import torch
    from importlib import metadata
    if not torch.cuda.is_available():
        raise RuntimeError("Existing PyTorch cannot use CUDA / 已有 PyTorch 无法使用 CUDA")
    original_torch = torch.__version__
    original_cuda = torch.version.cuda
    missing = []
    # OpenCV 4.12+ requires NumPy 2; retain preinstalled NumPy 1 environments.
    # OpenCV 4.12 之后要求 NumPy 2；保留镜像自带 NumPy 1 时选择兼容版本范围。
    opencv_package = "opencv-python-headless"
    try:
        if int(metadata.version("numpy").split(".")[0]) < 2:
            opencv_package += "<4.12"
    except metadata.PackageNotFoundError:
        pass
    for module, package in [("numpy", "numpy"), ("scipy", "scipy"), ("trimesh", "trimesh"), ("PIL", "Pillow"),
                            ("huggingface_hub", "huggingface-hub"), ("kornia", "kornia"),
                            ("cv2", opencv_package), ("pybind11", "pybind11>=2.10")]:
        if importlib.util.find_spec(module) is None:
            missing.append(package)
    # TensorRT provides separate CUDA families. Select from the existing torch
    # CUDA runtime; its own packages are downloaded rather than bundled here.
    # TensorRT 分 CUDA 家族发行；按已有 torch 的 CUDA 运行时选包，不打进本 wheel。
    cuda_major = int(original_cuda.split(".")[0])
    if cuda_major not in (11, 12, 13):
        raise RuntimeError("Unverified TensorRT CUDA family / 未验证的 TensorRT CUDA 家族")
    if importlib.util.find_spec("tensorrt") is None:
        missing.append("tensorrt-cu%d>=10,<11" % cuda_major)
    if importlib.util.find_spec("onnx") is None:
        missing.append("onnx")
    cmake_path = shutil.which("cmake")
    cmake_version = (0, 0)
    if cmake_path is not None:
        cmake_output = subprocess.check_output([cmake_path, "--version"], text=True)
        cmake_match = re.search(r"version (\d+)\.(\d+)", cmake_output)
        if cmake_match is not None:
            cmake_version = tuple(int(value) for value in cmake_match.groups())
    if cmake_version < (3, 18):
        missing.append("cmake>=3.18")
    print("WAPR_TARGET", {"python": sys.version.split()[0], "executable": sys.executable,
                          "torch": original_torch, "torch_cuda": original_cuda,
                          "gpu": torch.cuda.get_device_name(0), "missing": missing}, flush=True)
    if missing:
        from wapr.installation import install_requirements
        dependency_result = install_requirements(missing, allow_replacement=allow_replacement)
        if dependency_result.get("status") not in ("ready", "installed"):
            raise RuntimeError("Core preparation stopped / 核心环境准备已停止: " + json.dumps(dependency_result, ensure_ascii=False))
    # Missing EGL/GL development headers are system prerequisites, not wheel contents.
    # EGL/GL 开发头文件是系统前置条件，不打进 wheel；缺项才安装。
    system_packages = []
    for header, package in [("/usr/include/EGL/egl.h", "libegl1-mesa-dev"),
                             ("/usr/include/GL/gl.h", "libgl1-mesa-dev")]:
        if not os.path.isfile(header):
            system_packages.append(package)
    if shutil.which("g++") is None:
        system_packages.append("g++")
    if system_packages:
        if os.geteuid() != 0 or shutil.which("apt-get") is None:
            raise RuntimeError("Install system prerequisites / 请安装系统前置依赖: " + " ".join(system_packages))
        subprocess.check_call(["apt-get", "update"])
        subprocess.check_call(["apt-get", "install", "-y"] + system_packages)
    if metadata.version("torch") != original_torch:
        raise RuntimeError("Restart Python after an approved PyTorch replacement / 同意更换 PyTorch 后，请重启 Python 再准备运行环境")
    from wapr.ogl import ensure_ogl
    ensure_ogl()
    print("WAPR_RUNTIME_READY", {"torch": original_torch, "torch_cuda": original_cuda}, flush=True)


def native_build_options():
    """Derive the compiler and supported SM target without replacing CUDA.

    推导编译器与支持的 SM 目标，不替换 CUDA；旧编译器为新显卡保留 PTX。
    """
    import torch
    from importlib import metadata
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not os.path.isfile(nvcc):
        raise RuntimeError("CUDA compiler nvcc is missing / 缺少 CUDA 编译器 nvcc")
    version_output = subprocess.check_output([nvcc, "--version"], text=True)
    match = re.search(r"release (\d+)\.(\d+)", version_output)
    if match is None:
        raise RuntimeError("Cannot determine CUDA compiler version / 无法确定 CUDA 编译器版本")
    compiler_version = tuple(int(value) for value in match.groups())
    targets_output = subprocess.check_output([nvcc, "--list-gpu-code"], text=True)
    supported = sorted({int(value) for value in re.findall(r"sm_(\d+)", targets_output)})
    major, minor = torch.cuda.get_device_capability(0)
    target_sm = major * 10 + minor
    supported_sm = max(value for value in supported if value <= target_sm)
    architecture = str(target_sm) if target_sm in supported else str(supported_sm) + "-virtual"
    # Pybind11 resolves its CMake package using the active interpreter.
    # pybind11 的 CMake 包使用当前解释器定位；CUDA 不依赖 torch 扩展的 ABI。
    print("WAPR_NATIVE_BUILD", {"nvcc": nvcc, "compiler_cuda": compiler_version,
                                "torch_cuda": torch.version.cuda, "gpu_sm": target_sm,
                                "cmake_architecture": architecture,
                                "pybind11": metadata.version("pybind11")}, flush=True)
    return ["-DCMAKE_CUDA_COMPILER=" + nvcc, "-DCMAKE_CUDA_ARCHITECTURES=" + architecture]


def fetch_example():
    """Fetch the public usage example from the package's GitHub source URL.

    从包内指定的 GitHub 地址下载公开用法示例，不执行下载的代码。
    """
    from wapr.resources import resource_root
    urls = ["https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/main/examples/02_one_category_one_instance.py",
            "https://api.github.com/repos/WangYuLin-SEU/WAPR/contents/examples/02_one_category_one_instance.py?ref=main"]
    destination = Path(resource_root()) / "examples" / "02_one_category_one_instance.py"
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = None
        for url in urls:
            # Direct GitHub often works when an acceleration proxy returns 503.
            # 加速代理返回 503 时，GitHub 直连常仍可用；只在包内执行回退。
            for direct in (True, False):
                opener = build_opener(ProxyHandler({})) if direct else build_opener()
                request = Request(url, headers={"Accept": "application/vnd.github.raw+json",
                                                "User-Agent": "WAPR-resource-loader"})
                try:
                    with opener.open(request, timeout=20) as response:
                        content = response.read()
                    if b"WAPREstimator" not in content:
                        content = None
                        continue
                    break
                except (HTTPError, URLError, TimeoutError, OSError) as error:
                    print("WAPR_GITHUB_RETRY", {"direct": direct, "error": type(error).__name__}, flush=True)
            if content is not None:
                break
        if content is None:
            raise RuntimeError("Cannot download the GitHub usage example / 无法下载 GitHub 用法示例")
        if b"WAPREstimator" not in content:
            raise RuntimeError("Unexpected GitHub example content / GitHub 示例内容异常")
        destination.write_bytes(content)
    print("WAPR_GITHUB_EXAMPLE", str(destination), flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare WAPR in the existing Python / 在当前 Python 准备 WAPR")
    parser.add_argument("--feature", nargs="+", default=["core"], choices=[
        "core", "dinov2", "det2d", "sam2", "sam3d", "roma", "qwen", "robot", "unipose9d", "compatible"])
    parser.add_argument("--check", action="store_true", help="Inspect without installing / 只检查，不安装")
    parser.add_argument("--yes", action="store_true", help="Explicitly approve shown package replacements / 明确同意包更换")
    arguments = parser.parse_args()
    approved = True if arguments.yes else None
    for requested_feature in arguments.feature:
        preparation = prepare_feature(requested_feature, allow_replacement=approved, check_only=arguments.check)
        print("WAPR_PREPARATION", json.dumps(preparation, ensure_ascii=False), flush=True)
        if preparation.get("status") in ("blocked", "declined", "failed", "approval_required", "restart_required", "partial", "needs_source", "dependencies_ready") and not arguments.check:
            raise SystemExit(1)
