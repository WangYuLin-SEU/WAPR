# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Build a WAPR wheel from a fresh staging directory without uploading it.

从干净的暂存目录构建 WAPR wheel，不上传。
Run from any directory: python wheel_build/build_wheel.py
可从任意目录运行：python wheel_build/build_wheel.py
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.dirname(SCRIPT_DIR)
DIST_DIR = os.path.join(SCRIPT_DIR, "dist")
# CUDA 12.8 accepts these SM targets. Keep the release matrix explicit until
# each GPU family has been tested; the local check below covers this host only.
# CUDA 12.8 支持这些 SM 目标。各显卡家族验证之前显式保留这份发行矩阵；
# 下面的本地检查只覆盖当前机器。
CUDA_ARCHITECTURES = "75;80;86;89;90;100;120"


def source_ignore(directory, names):
    """Exclude generated files while copying first-party package sources.

    复制第一方包源码时排除生成文件。
    """
    ignored = set()
    for name in names:
        if name in {"__pycache__", "build", ".pytest_cache", ".git", "reports", "benchmarks", "samples", "datasets", "third_party", "outputs", "cache", ".cache"} or name.endswith((".pyc", ".so", ".pth", ".pt", ".engine", ".onnx", ".o", ".a", ".whl")):
            ignored.add(name)
        if name == "paths.json":
            ignored.add(name)
        if os.path.abspath(directory) == os.path.join(RELEASE_DIR, "wapr") and name == "tools":
            # Source-checkout diagnostics and patches are not wheel modules.
            # 源码检出用的检查脚本与补丁不属于 wheel 模块。
            ignored.add(name)
    return ignored


def main():
    """Compile the renderer, build a tagged wheel, and inspect its contents.

    编译渲染器、构建带 ABI 标签的 wheel，并检查其内容。
    """
    # Explicit build choice: source wheel preserves the target Python/CUDA stack.
    # 显式构建选择：源码 wheel 保留目标机器已有的 Python/CUDA 软件栈。
    if os.environ.get("WAPR_WHEEL_SOURCE_ONLY") == "1":
        build_source_wheel()
        return
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Build with Python 3.10 / 请使用 Python 3.10 构建")
    if sys.platform != "linux":
        raise RuntimeError("Build on Linux / 请在 Linux 上构建")
    if shutil.which("cmake") is None:
        raise RuntimeError("Missing build tool / 缺少构建工具: cmake")
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    nvcc_path = shutil.which("nvcc") or os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc_path):
        raise RuntimeError("Missing CUDA compiler / 缺少 CUDA 编译器: " + nvcc_path)
    nvcc_version = subprocess.check_output([nvcc_path, "--version"], text=True)
    if "release 12.8," not in nvcc_version:
        raise RuntimeError("Build with CUDA toolkit 12.8 / 请使用 CUDA toolkit 12.8 构建")
    build_env = os.environ.copy()
    build_env["CUDACXX"] = nvcc_path
    build_env["PATH"] = os.path.dirname(nvcc_path) + os.pathsep + build_env.get("PATH", "")
    os.makedirs(DIST_DIR, exist_ok=True)
    print("WHEEL_CUDA_ARCHITECTURES", CUDA_ARCHITECTURES, flush=True)

    with tempfile.TemporaryDirectory(prefix="wapr-wheel-") as temporary_dir:
        stage_dir = os.path.join(temporary_dir, "stage")
        os.makedirs(stage_dir)
        for name in ("pyproject.toml", "setup.py", "README.md"):
            shutil.copy2(os.path.join(SCRIPT_DIR, name), os.path.join(stage_dir, name))
        shutil.copy2(os.path.join(RELEASE_DIR, "LICENSE"), os.path.join(stage_dir, "LICENSE"))
        # Distribute the code author's notice with the license metadata.
        # 代码作者与权益声明随许可元数据分发。
        shutil.copy2(os.path.join(RELEASE_DIR, "AUTHORS.md"), os.path.join(stage_dir, "AUTHORS.md"))
        shutil.copy2(os.path.join(RELEASE_DIR, "WEIGHTS_LICENSE.txt"), os.path.join(stage_dir, "WEIGHTS_LICENSE.txt"))
        # Keep upstream license copies in the wheel metadata, not as executable code.
        # 上游许可副本随 wheel 元数据分发，不作为可执行代码打包。
        shutil.copy2(os.path.join(RELEASE_DIR, "THIRD_PARTY_NOTICES.txt"),
                     os.path.join(stage_dir, "THIRD_PARTY_NOTICES.txt"))
        shutil.copytree(
            os.path.join(RELEASE_DIR, "wapr"),
            os.path.join(stage_dir, "wapr"),
            ignore=source_ignore,
        )
        # The detector catalog is copied with wapr/ and included as package data.
        # 检测器校验清单随 wapr/ 一起复制，并作为包数据收录。

        native_dir = os.path.join(stage_dir, "wapr", "ogl_native")
        build_dir = os.path.join(temporary_dir, "native-build")
        wheel_output_dir = os.path.join(temporary_dir, "wheel-output")
        os.makedirs(wheel_output_dir)
        subprocess.check_call([
            "cmake", "-S", native_dir, "-B", build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CUDA_ARCHITECTURES=" + CUDA_ARCHITECTURES,
            "-DPython3_EXECUTABLE=" + sys.executable,
        ], env=build_env)
        subprocess.check_call(["cmake", "--build", build_dir, "-j", "4"], env=build_env)

        native_files = [name for name in os.listdir(native_dir) if name.startswith("_gpu_render") and name.endswith(".so")]
        if len(native_files) != 1:
            raise RuntimeError("Expected one native renderer / 预期一份本地编译的渲染器: " + str(native_files))
        subprocess.check_call([
            sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
            "--wheel-dir", wheel_output_dir, stage_dir,
        ])
        wheel_files = [name for name in os.listdir(wheel_output_dir) if name.endswith(".whl")]
        if len(wheel_files) != 1:
            raise RuntimeError("Expected one wheel / 预期恰好一个 wheel: " + str(wheel_files))
        wheel_path = os.path.join(DIST_DIR, wheel_files[0])
        shutil.copy2(os.path.join(wheel_output_dir, wheel_files[0]), wheel_path)

    with zipfile.ZipFile(wheel_path) as archive:
        members = archive.namelist()
        native_members = [name for name in members if name.startswith("wapr/ogl_native/_gpu_render") and name.endswith(".so")]
        shader_members = [name for name in members if name.startswith("wapr/ogl_native/shaders/")]
        detector_catalog = "wapr/det2d_assets.json" in members
        # Only package runtime files and distribution metadata belong in a wheel.
        # wheel 仅收录运行包与发行元数据，许可文件逐一核对。
        leaked = []
        for name in members:
            parts = name.split("/")
            if not (parts[0] == "wapr" or parts[0].endswith(".dist-info")):
                leaked.append(name)
            if any(part in {"assets", "weights", "samples", "datasets", "reports", "benchmarks", "third_party", "outputs", "cache", ".cache", "__pycache__", "tools", "pages"} for part in parts):
                leaked.append(name)
            if parts[-1] in {"paths.json", "PATHS.txt", "AGENTS.md", "check_list.md"} or name.endswith((".pth", ".pt", ".engine", ".onnx", ".pyc", ".log")):
                leaked.append(name)
        required_licenses = ["LICENSE", "AUTHORS.md", "THIRD_PARTY_NOTICES.txt", "WEIGHTS_LICENSE.txt"]
        required_licenses += ["wapr/fonts/LICENSE_DEJAVU", "wapr/fonts/wqy-microhei-copyright.txt", "wapr/fonts/Apache-2.0.txt"]
        missing_licenses = [name for name in required_licenses if not any(member.endswith(".dist-info/licenses/" + name) for member in members)]
        if len(native_members) != 1 or len(shader_members) != 2 or not detector_catalog or leaked or missing_licenses:
            raise RuntimeError("Wheel content check failed / wheel 内容检查失败: " + str((native_members, shader_members, detector_catalog, leaked, missing_licenses)))
        print("WHEEL_LICENSE_FILES", len(required_licenses), flush=True)
    digest = hashlib.sha256()
    with open(wheel_path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    print("WHEEL_BUILT", wheel_path, flush=True)
    print("WHEEL_SHA256", digest.hexdigest(), flush=True)
    print("WHEEL_CONTENT", {"native": native_members[0], "shaders": shader_members}, flush=True)


def build_source_wheel():
    """Bundle runtime and native sources, without prebuilt GPU binaries.

    打包运行库及本地模块源码，不包含预编译 GPU 二进制。
    WAPR_WHEEL_SOURCE_ONLY=1 python wheel_build/build_wheel.py
    """
    os.makedirs(DIST_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wapr-source-wheel-") as temporary_dir:
        stage_dir = os.path.join(temporary_dir, "stage")
        os.makedirs(stage_dir)
        for name in ("setup.py", "README.md"):
            shutil.copy2(os.path.join(SCRIPT_DIR, name), stage_dir)
        metadata_path = os.path.join(SCRIPT_DIR, "pyproject.toml")
        with open(metadata_path, encoding="utf-8") as stream:
            metadata = stream.read()
        metadata = metadata.replace('requires-python = "==3.10.*"', 'requires-python = ">=3.8"')
        metadata += '\n# Native sources compile on the target. / 本地源码在目标机器编译。\n'
        metadata += '[tool.setuptools.exclude-package-data]\nwapr = ["**/*.so", "**/*.pyc"]\n'
        metadata = metadata.replace('"ogl_native/_gpu_render*.so",',
                                    '"ogl_native/CMakeLists.txt", "ogl_native/cpp/*", "ogl_native/cuda/*", "ogl_native/python/*",')
        with open(os.path.join(stage_dir, "pyproject.toml"), "w", encoding="utf-8") as stream:
            stream.write(metadata)
        for name in ("LICENSE", "AUTHORS.md", "WEIGHTS_LICENSE.txt", "THIRD_PARTY_NOTICES.txt"):
            shutil.copy2(os.path.join(RELEASE_DIR, name), stage_dir)
        shutil.copytree(os.path.join(RELEASE_DIR, "wapr"), os.path.join(stage_dir, "wapr"), ignore=source_ignore)
        output_dir = os.path.join(temporary_dir, "dist")
        subprocess.check_call([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                               "--wheel-dir", output_dir, stage_dir])
        wheels = [name for name in os.listdir(output_dir) if name.endswith(".whl")]
        if len(wheels) != 1 or not wheels[0].endswith("-py3-none-any.whl"):
            raise RuntimeError("Expected one source-only Python wheel / 应产生一个仅含源码的 wheel")
        wheel_path = os.path.join(DIST_DIR, wheels[0])
        shutil.copy2(os.path.join(output_dir, wheels[0]), wheel_path)
    with zipfile.ZipFile(wheel_path) as archive:
        members = archive.namelist()
        required = ["wapr/bootstrap.py", "wapr/ogl_native/CMakeLists.txt",
                    "wapr/ogl_native/python/gpu_render_pybind.cpp",
                    "wapr/ogl_native/cuda/pack_outputs.cu"]
        if any(name not in members for name in required):
            raise RuntimeError("Native source or setup entry missing / 缺少本地源码或安装入口")
        prohibited = {"reports", "benchmarks", "samples", "third_party", "outputs", "__pycache__", "tools", "pages"}
        if any(prohibited.intersection(name.split("/")) or name.endswith((".so", ".pth", ".engine", ".onnx", ".pyc")) for name in members):
            raise RuntimeError("Non-runtime content in wheel / wheel 含非运行内容")
        for name in ("LICENSE", "AUTHORS.md", "WEIGHTS_LICENSE.txt", "THIRD_PARTY_NOTICES.txt"):
            if not any(member.endswith(".dist-info/licenses/" + name) for member in members):
                raise RuntimeError("Missing license / 缺少许可: " + name)
    with open(wheel_path, "rb") as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    print("WHEEL_BUILT", wheel_path, flush=True)
    print("WHEEL_SHA256", digest, flush=True)
    print("WHEEL_CONTENT", members, flush=True)


if __name__ == "__main__":
    main()
