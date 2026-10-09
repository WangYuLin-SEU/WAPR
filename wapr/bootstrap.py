# Author: Yulin Wang (yulinwang@seu.edu.cn)
# SPDX-License-Identifier: LGPL-2.1-only
"""Prepare an installed WAPR source wheel in the existing Python environment.

核心及 SAM2 使用当前 Python；SAM3D 按需准备兼容独立前缀，更换已有包须用户明确同意。
Run / 运行: python -m wapr.bootstrap
"""
import ctypes.util
import argparse
import glob
import hashlib
import html
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import ProxyHandler, Request, build_opener


_prepared_optional = set()


def _find_nvcc():
    """Locate nvcc on PATH, a CUDA toolkit, or the pip CUDA package.

    在 PATH、CUDA 工具包或 pip 的 CUDA 包里找 nvcc。
    """
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    candidates = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        toolkit = os.environ.get(variable, "").strip()
        if toolkit:
            candidates.extend(os.path.join(toolkit, "bin", name) for name in ("nvcc.exe", "nvcc"))
    # Discover versioned system toolkits and packages in the current interpreter.
    # 搜索系统多版本工具包与当前解释器中的包，不绑定某个机器路径。
    import sysconfig
    roots = [sys.prefix, os.path.join(sys.prefix, "Library")]
    if sys.platform == "win32":
        roots.extend(glob.glob(os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"),
                                           "NVIDIA GPU Computing Toolkit", "CUDA", "v*")))
    else:
        roots.extend(glob.glob("/usr/local/cuda*"))
        roots.extend(glob.glob("/opt/cuda*"))
    for package_root in set(sysconfig.get_paths().get(key, "") for key in ("purelib", "platlib")):
        if package_root:
            roots.extend(glob.glob(os.path.join(package_root, "nvidia", "*")))
    for root in roots:
        candidates.extend(os.path.join(root, "bin", name) for name in ("nvcc", "nvcc.exe"))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def check_runtime():
    """Inspect the existing core environment without installing anything.

    只检查当前核心环境，不安装包、不编译、不下载权重。
    """
    if sys.platform == "darwin":
        return {"feature": "core", "python": sys.version.split()[0], "executable": sys.executable,
                "platform": sys.platform, "machine": platform.machine(), "status": "blocked",
                "reason": "macOS cannot run this pose runtime: no NVIDIA CUDA, OGL is WGL on Windows and EGL on Linux"
                          " / macOS 无法运行此位姿运行时：没有 NVIDIA CUDA，OGL 在 Windows 上是 WGL、在 Linux 上是 EGL"}
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
            ordinal = cuda_device_index()
            result.update(cuda_device=ordinal, gpu=torch.cuda.get_device_name(ordinal), gpu_memory_gb=round(
                torch.cuda.get_device_properties(ordinal).total_memory / 1024 ** 3, 2))
        result["nvcc"] = _find_nvcc()
        from wapr.ogl import selected_renderer
        result["renderer"] = selected_renderer()
        if not torch.cuda.is_available():
            result.update(status="blocked", reason="Working CUDA is required / 需要可用的 CUDA")
        elif result["renderer"] == "ogl" and not result["nvcc"]:
            result.update(status="blocked", reason="OGL requires nvcc / OGL 需要 nvcc")
        elif all(versions[name] is not None for name in ("numpy", "trimesh", "kornia")) and (
                result["renderer"] == "nvdiffrast" or result["nvcc"]):
            result["status"] = "dependencies_present"
            result["note"] = "Presence is not inference validation / 依赖存在不等于推理验证通过"
    except (ImportError, OSError) as error:
        result.update(status="blocked", reason="Existing PyTorch is unavailable / 当前 PyTorch 不可用: " + type(error).__name__)
    return result


def prepare_feature(feature, allow_replacement=None, check_only=False):
    """Prepare one requested feature, preserving unrelated optional stacks.

    只准备被请求的功能；更换已有库须明确同意，不安装无关可选功能。
    """
    # Block unsupported recipes before importing Torch or starting installations.
    # 不支持的方案在导入 Torch、启动安装前停止。
    if sys.platform == "win32" and feature in ("sam3d", "robot", "roma"):
        return {"feature": feature, "status": "blocked",
                "reason": "This WAPR feature setup requires Linux; Windows is unsupported"
                          " / 此 WAPR 功能安装方案需要 Linux，不支持 Windows"}
    if feature == "det2d":
        from wapr.resources import cache_dir
        source_path = os.path.join(cache_dir(), "sources", "GroundingDINO")
        if not os.path.isdir(source_path) and shutil.which("git") is None:
            return {"feature": feature, "status": "blocked",
                    "reason": "Git command-line tool is required. Install Git, add it to PATH, reopen the terminal and verify git --version"
                              " / 需要 Git 命令行工具。请安装 Git 并加入 PATH，重开终端后用 git --version 确认"}
    from wapr.installation import install_requirements, prepare_optional
    if feature == "core":
        if check_only:
            return check_runtime()
        prepare_runtime(allow_replacement=allow_replacement)
        return {"feature": "core", "status": "ready"}
    if feature == "sam3d":
        import torch
        supported_pair = (torch.__version__.split("+", 1)[0] == "2.5.1"
                          and torch.version.cuda in ("11.8", "12.1", "12.4"))
        if sys.version_info[:2] >= (3, 12) or sys.version_info[:2] < (3, 9) or not supported_pair:
            # Keep the pose interpreter; prepare SAM3D in the compatible interpreter.
            # 保留位姿解释器；仅在兼容的独立解释器准备 SAM3D。
            from wapr.sam3d_isolated import prepare_reconstruction_environment
            return prepare_reconstruction_environment(allow_replacement=allow_replacement,
                                                       check_only=check_only)
        from wapr.reconstruction_setup import prepare_reconstruction
        return prepare_reconstruction(allow_replacement=allow_replacement, check_only=check_only)
    if feature == "roma":
        from wapr.sam3d_isolated import SAM3D_ENV_ROOT
        if os.path.realpath(sys.prefix) != os.path.realpath(SAM3D_ENV_ROOT):
            # RoMa's headless OpenCV dependency conflicts with the robot distribution.
            # RoMa 的无界面 OpenCV 依赖与机器人发行冲突，神经匹配独立运行。
            from wapr.roma_isolated import prepare_environment
            return prepare_environment(allow_replacement=allow_replacement, check_only=check_only)
    if feature == "compatible":
        # Resolve first; replacing an existing package still requires approval.
        # 先逐项解析；更换已有包仍须明确同意，无法准备的功能单独报告。
        results = []
        for name in ("dinov2", "det2d", "robot", "sam2", "roma", "qwen", "sam3d", "unipose9d"):
            plan = prepare_feature(name, allow_replacement=False, check_only=True)
            if not check_only and plan.get("status") in ("ready", "installable", "dependencies_present", "approval_required", "needs_source", "needs_interpreter", "native_build_required"):
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
    if feature == "roma":
        from wapr.source_setup import prepare_source
        if check_only:
            from wapr.resources import cache_dir
            source_name = "RoMa"
            source_parent = os.path.join(cache_dir(), "sources")
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
        if feature == "roma" and source is not None and result.get("status") == "ready":
            # Check the actual inference API without installing dependencies or weights.
            # 不安装依赖或权重；检查现有推理源码的真实 API。
            if source not in sys.path:
                sys.path.insert(0, source)
            import importlib
            try:
                module = importlib.import_module("romatch")
                if not callable(getattr(module, "roma_outdoor", None)):
                    raise RuntimeError("RoMa API missing / RoMa API 缺失: roma_outdoor")
            except (ImportError, RuntimeError, OSError) as error:
                result.update(status="blocked", reason=str(error))
        return result
    if result.get("status") not in ("ready", "installed"):
        return result
    # Source-based projects stay outside site-packages and outside the wheel.
    # 源码项目放在资源目录，不打进 wheel；使用当前解释器解析其实际依赖。
    if feature in ("det2d", "roma", "unipose9d") and not result.get("existing_source_api"):
        from wapr.source_setup import prepare_source
        if source is None:
            source = prepare_source(feature)
        result["source"] = source
        if feature == "det2d":
            source_paths = [source]
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
        elif feature == "roma":
            import importlib
            module = importlib.import_module("romatch")
            if not callable(getattr(module, "roma_outdoor", None)):
                raise RuntimeError("RoMa API missing / RoMa API 缺失: roma_outdoor")
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


def cuda_device_index():
    """Return the CUDA device index selected for this process.

    返回本进程选用的 CUDA 设备编号。
    设置了 WAPR_CUDA_DEVICE 就用该整数，超出 device_count 则拒绝。
    未设置时用 torch.cuda.current_device()，该值遵守 CUDA_VISIBLE_DEVICES。
    """
    import torch
    chosen = os.environ.get("WAPR_CUDA_DEVICE", "").strip()
    if not chosen:
        return torch.cuda.current_device()
    try:
        index = int(chosen)
    except ValueError:
        raise RuntimeError("WAPR_CUDA_DEVICE must be an integer / WAPR_CUDA_DEVICE 必须是整数: " + chosen) from None
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise RuntimeError("WAPR_CUDA_DEVICE is out of range / WAPR_CUDA_DEVICE 超出设备范围: "
                           + "%s (device_count=%d)" % (chosen, count))
    return index


def _tensorrt_requirements(cuda_major):
    """Reuse verified NVIDIA wheels instead of downloading them again for metadata.

    复用已校验的 NVIDIA wheel，避免解析元数据与安装时重复下载大型库。
    Match the existing version bounds, Python ABI and OS; keep the upstream fallback.
    按已有版本范围、Python ABI 和系统匹配；目录不可用时保留上游安装方式。
    """
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename, canonicalize_name
    from packaging.version import Version
    from wapr.resources import cache_dir
    import urllib.request

    family = "tensorrt-cu%d" % cuda_major
    fallback = [family + ">=10,<11"]
    if os.environ.get("WHEEL_STUB_PIP_INDEX_URL"):
        return fallback
    compatible_tags = set(sys_tags())
    catalogs = []
    try:
        for suffix in ("libs", "bindings"):
            name = family + "-" + suffix
            index = "https://pypi.nvidia.com/" + name + "/"
            with urllib.request.urlopen(index, timeout=20) as response:
                page = response.read().decode("utf-8")
            candidates = {}
            for href in re.findall(r'href=[\"\x27]([^\"\x27]+)', page):
                url = urljoin(index, html.unescape(href))
                parts = urlsplit(url)
                filename = unquote(os.path.basename(parts.path))
                if not filename.endswith(".whl") or parts.hostname != "pypi.nvidia.com":
                    continue
                wheel_name, version, _, tags = parse_wheel_filename(filename)
                digest = re.fullmatch(r"sha256=([0-9a-f]{64})", parts.fragment)
                if (canonicalize_name(wheel_name) == name and digest is not None
                        and Version("10") <= version < Version("11")
                        and not version.is_prerelease and compatible_tags.intersection(tags)):
                    candidates[version] = (name, filename, url, digest.group(1))
            catalogs.append(candidates)
    except (OSError, ValueError, URLError) as error:
        print("WAPR_TENSORRT_ROUTE", {"route": "upstream", "reason": type(error).__name__}, flush=True)
        return fallback
    common_versions = set(catalogs[0]).intersection(catalogs[1])
    if not common_versions:
        return fallback
    version = max(common_versions)
    requirements = [family + "==" + str(version)]
    directory = os.path.join(cache_dir(), "package_wheels", "tensorrt")
    os.makedirs(directory, exist_ok=True)
    from wapr.det2d import _download_file
    for catalog in catalogs:
        name, filename, url, expected_digest = catalog[version]
        destination = os.path.join(directory, filename)
        if not os.path.isfile(destination):
            _download_file(url.split("#", 1)[0], destination)
        digest = hashlib.sha256()
        with open(destination, "rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_digest:
            raise RuntimeError("NVIDIA TensorRT wheel checksum mismatch; retain and review the file"
                               " / NVIDIA TensorRT wheel 校验失败，保留文件并检查: " + destination)
        requirements.append(name + " @ " + Path(destination).as_uri() + "#sha256=" + expected_digest)
    print("WAPR_TENSORRT_ROUTE", {"route": "verified_wheels", "version": str(version),
                                  "python": sys.version.split()[0], "platform": sys.platform}, flush=True)
    return requirements


def prepare_runtime(allow_replacement=None):
    """Install missing pose dependencies and compile for the existing CUDA.

    安装缺失的位姿依赖，按当前 CUDA 编译；默认保留已有库，更换须明确同意。
    默认编译 OGL。Linux 用 EGL，Windows 用 WGL。只有 WAPR_RENDERER=nvdiffrast 才准备 CUDA 光栅器。
    """
    if sys.platform == "darwin":
        raise RuntimeError(
            "macOS cannot run this pose runtime: no NVIDIA CUDA, OGL is WGL on Windows and EGL on Linux, "
            "Docker cannot provide Darwin"
            " / macOS 无法运行此位姿运行时：没有 NVIDIA CUDA，OGL 在 Windows 上是 WGL、在 Linux 上是 EGL，Docker 不能提供 Darwin")
    from wapr.ogl import selected_renderer
    renderer = selected_renderer()
    import torch
    from importlib import metadata
    if not torch.cuda.is_available():
        raise RuntimeError("Existing PyTorch cannot use CUDA / 已有 PyTorch 无法使用 CUDA"
                           " (platform=%s, machine=%s)" % (sys.platform, platform.machine()))
    original_torch = torch.__version__
    original_cuda = torch.version.cuda
    missing = []
    # OpenCV 4.12+ requires NumPy 2; retain preinstalled NumPy 1 environments.
    # OpenCV 4.12 之后要求 NumPy 2；保留镜像自带 NumPy 1 时选择兼容版本范围。
    # The default setup also prepares Ultralytics, which requires this cv2 provider.
    # 默认准备包含 Ultralytics，使用其要求的 cv2 发行，避免首次准备再次切换发行。
    opencv_package = "opencv-python"
    try:
        if int(metadata.version("numpy").split(".")[0]) < 2:
            opencv_package += "<4.12"
    except metadata.PackageNotFoundError:
        pass
    for module, package in [("numpy", "numpy"), ("scipy", "scipy"), ("trimesh", "trimesh"), ("PIL", "Pillow"),
                            ("huggingface_hub", "huggingface-hub>=0.34,<1"), ("kornia", "kornia"),
                            ("cv2", opencv_package)]:
        if importlib.util.find_spec(module) is None:
            missing.append(package)
    # TensorRT provides separate CUDA families. Select from the existing torch
    # CUDA runtime; its own packages are downloaded rather than bundled here.
    # TensorRT 分 CUDA 家族发行；按已有 torch 的 CUDA 运行时选包，不打进本 wheel。
    use_trt = True
    cuda_major = int(original_cuda.split(".")[0])
    if renderer == "ogl":
        if use_trt and cuda_major not in (11, 12, 13):
            raise RuntimeError("Unverified TensorRT CUDA family / 未验证的 TensorRT CUDA 家族")
        if use_trt and importlib.util.find_spec("tensorrt") is None:
            missing.extend(_tensorrt_requirements(cuda_major))
        if importlib.util.find_spec("pybind11") is None:
            missing.append("pybind11>=2.10")
        cmake_path = shutil.which("cmake") or os.path.join(sys.prefix, "Scripts", "cmake.exe")
        cmake_version = (0, 0)
        if os.path.isfile(cmake_path):
            cmake_output = subprocess.check_output([cmake_path, "--version"], encoding="utf-8", errors="replace")
            cmake_match = re.search(r"version (\d+)\.(\d+)", cmake_output)
            if cmake_match is not None:
                cmake_version = tuple(int(value) for value in cmake_match.groups())
        if cmake_version < (3, 18):
            missing.append("cmake>=3.18")
        ninja_candidates = [shutil.which("ninja") or "",
                            os.path.join(sys.prefix, "Scripts", "ninja.exe"),
                            os.path.join(sys.prefix, "bin", "ninja")]
        if not any(path and os.path.isfile(path) for path in ninja_candidates):
            # Linux can use make; install Ninja only when neither builder is present.
            # Linux 可以用 make；两种构建工具都没有时才安装 Ninja。Windows 使用 Ninja。
            if sys.platform == "win32" or shutil.which("make") is None:
                missing.append("ninja")
    if importlib.util.find_spec("onnx") is None and use_trt:
        missing.append("onnx")
    print("WAPR_TARGET", {"python": sys.version.split()[0], "executable": sys.executable,
                          "torch": original_torch, "torch_cuda": original_cuda,
                          "gpu": torch.cuda.get_device_name(cuda_device_index()), "missing": missing}, flush=True)
    if missing:
        from wapr.installation import install_requirements
        dependency_result = install_requirements(missing, allow_replacement=allow_replacement)
        if dependency_result.get("status") not in ("ready", "installed"):
            raise RuntimeError("Core preparation stopped / 核心环境准备已停止: " + json.dumps(dependency_result, ensure_ascii=False))
    if renderer == "ogl" and sys.platform == "linux":
        # Missing EGL/GL development headers are system prerequisites, not wheel contents.
        # EGL/GL 开发头文件是系统前置条件，不打进 wheel；缺项才安装。Windows 用系统自带的 opengl32。
        system_packages = []
        include_roots = [os.path.join(sys.prefix, "include"), "/usr/include", "/usr/local/include"]
        for variable in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
            include_roots.extend(path for path in os.environ.get(variable, "").split(os.pathsep) if path)
        for prefix in os.environ.get("CMAKE_PREFIX_PATH", "").split(os.pathsep):
            if prefix:
                include_roots.append(os.path.join(prefix, "include"))
        for header, package in [(os.path.join("EGL", "egl.h"), "libegl1-mesa-dev"),
                                 (os.path.join("GL", "gl.h"), "libgl1-mesa-dev")]:
            if not any(os.path.isfile(os.path.join(root, header)) for root in include_roots):
                system_packages.append(package)
        if shutil.which("g++") is None:
            system_packages.append("g++")
        if system_packages:
            if not hasattr(os, "geteuid") or os.geteuid() != 0 or shutil.which("apt-get") is None:
                raise RuntimeError("Install system prerequisites / 请安装系统前置依赖: " + " ".join(system_packages))
            subprocess.check_call(["apt-get", "update"])
            subprocess.check_call(["apt-get", "install", "-y"] + system_packages)
    if metadata.version("torch") != original_torch:
        raise RuntimeError("Restart Python after an approved PyTorch replacement / 同意更换 PyTorch 后，请重启 Python 再准备运行环境")
    if renderer == "ogl":
        from wapr.ogl import ensure_ogl
        ensure_ogl()
    else:
        from wapr.source_setup import prepare_raster_source
        prepare_raster_source(allow_replacement=allow_replacement)
    print("WAPR_RUNTIME_READY", {"torch": original_torch, "torch_cuda": original_cuda, "renderer": renderer}, flush=True)


def _expose_pip_cuda_sonames(cuda_root):
    """Supply the unversioned linker names missing from pip CUDA runtime wheels.

    为 pip CUDA 运行库补齐 CMake 查找的无版本链接名称；不修改系统工具包。
    """
    if sys.platform != "linux" or "site-packages" not in cuda_root.replace("\\", "/").lower():
        return
    libdir = os.path.join(cuda_root, "lib")
    if not os.path.isdir(libdir):
        return
    for filename in sorted(os.listdir(libdir)):
        stem, separator, version = filename.partition(".so.")
        if not separator or not version[:1].isdigit() or not stem.startswith("lib"):
            continue
        link_path = os.path.join(libdir, stem + ".so")
        if os.path.lexists(link_path):
            continue
        try:
            os.symlink(filename, link_path)
        except OSError as error:
            raise RuntimeError("Cannot expose CUDA linker name / 无法创建 CUDA 链接名称: " + link_path) from error


def ensure_cuda_build_stack():
    """Install the pip CUDA compiler pieces that match this PyTorch, when they are missing.

    缺 nvcc 或 pip 工具包缺 CCCL 头时，按当前 PyTorch 的 CUDA 主版本补齐。
    系统里的完整 CUDA 不重装；更换已有包仍通过安装器征得同意。
    """
    nvcc = _find_nvcc()
    import torch
    cuda_major = int(torch.version.cuda.split(".")[0])
    if cuda_major not in (11, 12, 13):
        raise RuntimeError("Unsupported CUDA compiler package family / 不支持的 CUDA 编译器包家族: " + str(cuda_major))
    suffix = "-cu%d" % cuda_major if cuda_major < 13 else ""
    version_range = ">=%d,<%d" % (cuda_major, cuda_major + 1)
    if nvcc is None:
        package = "nvidia-cuda-nvcc" + suffix + version_range
        print("WAPR_CUDA_COMPILER", {"action": "install", "package": package}, flush=True)
        from wapr.installation import install_requirements
        result = install_requirements([package])
        if result.get("status") not in ("ready", "installed"):
            raise RuntimeError("CUDA compiler preparation stopped / CUDA 编译器准备已停止: " + json.dumps(result, ensure_ascii=False))
        nvcc = _find_nvcc()
    if not nvcc or not os.path.isfile(nvcc):
        raise RuntimeError("CUDA compiler nvcc is missing / 缺少 CUDA 编译器 nvcc")
    cuda_root = os.path.dirname(os.path.dirname(nvcc))
    # The pip NVIDIA layout tells CMake to include this directory. A system toolkit already has it.
    # pip 的 NVIDIA 目录会让 CMake 包含这个路径。系统 CUDA 工具包本身已经带齐。
    if "site-packages" in cuda_root.replace("\\", "/").lower() and not os.path.isdir(os.path.join(cuda_root, "include", "cccl")):
        package = "nvidia-cuda-cccl" + suffix + version_range
        print("WAPR_CUDA_COMPILER", {"action": "install", "package": package}, flush=True)
        from wapr.installation import install_requirements
        result = install_requirements([package])
        if result.get("status") not in ("ready", "installed"):
            raise RuntimeError("CUDA headers preparation stopped / CUDA 头文件准备已停止: " + json.dumps(result, ensure_ascii=False))
    _expose_pip_cuda_sonames(cuda_root)
    return nvcc


def native_build_options():
    """Derive the compiler and supported SM target without replacing CUDA.

    推导编译器与支持的 SM 目标，不替换 CUDA；旧编译器为新显卡保留 PTX。
    """
    import torch
    from importlib import metadata
    nvcc = ensure_cuda_build_stack()
    version_output = subprocess.check_output([nvcc, "--version"], encoding="utf-8", errors="replace")
    match = re.search(r"release (\d+)\.(\d+)", version_output)
    if match is None:
        raise RuntimeError("Cannot determine CUDA compiler version / 无法确定 CUDA 编译器版本")
    compiler_version = tuple(int(value) for value in match.groups())
    targets_output = subprocess.check_output([nvcc, "--list-gpu-code"], encoding="utf-8", errors="replace")
    supported = sorted({int(value) for value in re.findall(r"sm_(\d+)", targets_output)})
    major, minor = torch.cuda.get_device_capability(cuda_device_index())
    target_sm = major * 10 + minor
    usable = [value for value in supported if value <= target_sm]
    if not usable:
        raise RuntimeError("CUDA compiler has no target for this GPU / CUDA 编译器没有适合当前 GPU 的目标: " + str(target_sm))
    supported_sm = max(usable)
    architecture = str(target_sm) if target_sm in supported else str(supported_sm) + "-virtual"
    # Pybind11 resolves its CMake package using the active interpreter.
    # pybind11 的 CMake 包使用当前解释器定位；CUDA 不依赖 torch 扩展的 ABI。
    print("WAPR_NATIVE_BUILD", {"nvcc": nvcc, "compiler_cuda": compiler_version,
                                "torch_cuda": torch.version.cuda, "gpu_sm": target_sm,
                                "cmake_architecture": architecture,
                                "pybind11": metadata.version("pybind11")}, flush=True)
    options = ["-DCMAKE_CUDA_COMPILER=" + nvcc, "-DCMAKE_CUDA_ARCHITECTURES=" + architecture,
               "-DCUDAToolkit_ROOT=" + os.path.dirname(os.path.dirname(nvcc))]
    return options


def export_examples(destination):
    """Export bundled recipes without overwriting user edits; return their root.

    导出包内示例并保留用户修改，返回示例目录；目标目录就是 examples 根目录。
    """
    source = Path(__file__).resolve().parent / "runtime_examples"
    target = Path(destination).expanduser().resolve()
    if not source.is_dir():
        raise RuntimeError("Wheel has no bundled examples / wheel 未包含示例源码")
    target.mkdir(parents=True, exist_ok=True)
    copied, preserved = [], []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in (".py", ".md", ".json") and path.name != "LICENSE":
            continue
        relative = path.relative_to(source)
        output = target / relative
        content = path.read_bytes()
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation preserves edits even if another exporter wins the race.
        # 排他创建保证并发导出时也不会覆盖用户修改。
        try:
            with output.open("xb") as stream:
                stream.write(content)
            copied.append(str(relative))
        except FileExistsError:
            if not output.is_file() or output.read_bytes() != content:
                preserved.append(str(relative))
                print("WAPR_EXAMPLE_PRESERVED", str(output),
                      "Existing content differs / 已有内容不同，保留原文件", flush=True)
    print("WAPR_EXAMPLES_EXPORTED", {"destination": str(target),
                                     "copied": len(copied), "preserved": preserved}, flush=True)
    return target


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
    parser.add_argument("--feature", nargs="+", default=None, choices=[
        "core", "dinov2", "det2d", "sam2", "sam3d", "roma", "qwen", "robot", "unipose9d", "compatible"])
    parser.add_argument("--check", action="store_true", help="Inspect without installing / 只检查，不安装")
    parser.add_argument("--yes", action="store_true", help="Explicitly approve shown package replacements / 明确同意包更换")
    arguments = parser.parse_args()
    if arguments.feature is None:
        arguments.feature = ["core", "dinov2", "det2d", "sam2", "roma", "qwen", "unipose9d"]
        if sys.platform == "win32":
            # Default preparation continues with supported features; explicit requests fail.
            # 默认准备继续处理兼容功能；显式请求不支持的功能时仍报错。
            arguments.feature.remove("roma")
            print("WAPR_FEATURE_SKIPPED", "roma: current setup requires Linux / 当前安装方案需要 Linux", flush=True)
    approved = True if arguments.yes else None
    for requested_feature in arguments.feature:
        preparation = prepare_feature(requested_feature, allow_replacement=approved, check_only=arguments.check)
        print("WAPR_PREPARATION", json.dumps(preparation, ensure_ascii=False), flush=True)
        if preparation.get("status") in ("blocked", "declined", "failed", "approval_required", "restart_required", "partial", "needs_source", "dependencies_ready") and not arguments.check:
            raise SystemExit(1)
        if not arguments.check:
            # Explicit setup fetches assets; first-use checks retain caller-selected paths.
            # 显式准备时获取资源；首次调用检查保留调用者选定的路径。
            if requested_feature == "dinov2":
                from wapr.det2d import default_weights_dir, prepare_dino_weight
                prepare_dino_weight(default_weights_dir)
            elif requested_feature == "det2d":
                from wapr.det2d import default_weights_dir, prepare_det2d_weights
                prepare_det2d_weights(default_weights_dir)
            elif requested_feature == "sam2":
                from wapr.source_setup import prepare_sam2_weights
                prepare_sam2_weights()
            elif requested_feature == "roma":
                # Explicit setup fetches matching weights without running the matcher.
                # 显式准备时获取匹配权重，不执行匹配；复用已校验的检测器 DINOv2。
                from wapr.roma_isolated import _matching_weights
                _matching_weights("cpu")
            elif requested_feature == "qwen":
                from wapr.source_setup import prepare_qwen_weights
                prepare_qwen_weights()
            elif requested_feature == "unipose9d":
                from wapr.source_setup import prepare_unipose_weights
                prepare_unipose_weights()
    if "core" in arguments.feature and not arguments.check:
        # Download the shared demo packs after environment preparation succeeds.
        # 环境准备成功后下载共用小样，不在 pip 安装钩子中执行。
        from wapr.download_assets import main as download_demo_assets
        download_demo_assets()
