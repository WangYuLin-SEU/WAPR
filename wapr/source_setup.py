# Author: Yulin Wang (yulinwang@seu.edu.cn)
# Copyright (c) 2026 Yulin Wang. See LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
"""Fetch optional source without replacing checkouts; build the rasterizer on first use.

获取可选上游源码，不覆盖用户已有检出；光栅器首次调用可按已有 PyTorch 编译安装。
"""

import os
import time
import posixpath
import re
import importlib
import hashlib
import json
import http.client
import shutil
import subprocess
import sys
import tempfile
import tarfile
import urllib.request
import urllib.parse
import zipfile

from wapr.resources import resource_root, source_checkout


# These public snapshots are pinned, not a claim of inference validation.
# 固定公开快照，不代表已验证其推理；SAM commits follow the reconstruction recipe.
# SAM commit 与重建 recipe 一致。
DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
GROUNDING_REVISION = "856dde20aee659246248e20734ef9ba5214f5e44"
ULTRALYTICS_REVISION = "f38fa61ffc47a749be34e9317e9be9ea01d1ad9b"  # v8.3.70
SAM2_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM3D_REVISION = "f91db411c50efee93d8db7aeb323885650f6f722"
ROMA_REVISION = "77f8d68803526dcddfd9b7a46bc76125bdc25f15"
UNIPOSE9D_REVISION = "bfb2afd1d06f9ebe0c5edee46625b71a0cb2ecbf"
NVDIFFRAST_REVISION = "253ac4fcea7de5f396371124af597e6cc957bfae"  # Official v0.4.0 / 官方 v0.4.0
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _env_value(env, name):
    """Read an environment value on Windows, where key case is not significant.

    Windows 环境变量名不区分大小写。
    """
    for key, value in env.items():
        if key.lower() == name.lower():
            return value
    return ""


def _set_env_value(env, name, value):
    """Update an environment value without creating a second key of another case.

    按已有键的大小写写回，避免同时留下 Path 和 PATH。
    """
    for key in list(env):
        if key.lower() == name.lower():
            env[key] = value
            return
    env[name] = value


def _windows_compiler_env(base):
    """Put MSVC and nvcc on PATH when building native code on Windows.

    Windows 上编译 OGL 或可选光栅器时，把 MSVC 和 nvcc 放进 PATH。Linux 原样返回。
    """
    env = dict(base)
    if sys.platform != "win32":
        return env
    if shutil.which("cl", path=_env_value(env, "PATH")) is None:
        vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                               "Microsoft Visual Studio", "Installer", "vswhere.exe")
        install = ""
        if os.path.isfile(vswhere):
            try:
                install = subprocess.check_output(
                    [vswhere, "-latest", "-products", "*", "-requires",
                     "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                     "-property", "installationPath"], encoding="utf-8", errors="replace").strip()
            except (OSError, subprocess.CalledProcessError):
                install = ""
        vcvars = os.path.join(install, "VC", "Auxiliary", "Build", "vcvars64.bat") if install else ""
        if os.path.isfile(vcvars):
            # cmd.exe strips one pair of quotes from /c. Pass a raw command line so
            # list2cmdline does not escape the path that contains spaces.
            # cmd.exe 会剥掉 /c 的一层引号。这里直接传命令行，避免带空格的路径被转义。
            # cmd.exe prints `set` in the ANSI code page, not UTF-8.
            # cmd.exe 的 set 输出是 ANSI 代码页，不是 UTF-8。
            output = subprocess.check_output('cmd /s /c "call "%s" >nul && set"' % vcvars, encoding="mbcs", errors="replace")
            for line in output.splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                _set_env_value(env, key, value)
    if shutil.which("nvcc", path=_env_value(env, "PATH")) is None:
        cuda_path = _env_value(env, "CUDA_PATH")
        candidates = []
        if cuda_path:
            candidates.append(os.path.join(cuda_path, "bin", "nvcc.exe"))
        nvidia_root = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
        if os.path.isdir(nvidia_root):
            for entry in os.listdir(nvidia_root):
                candidates.append(os.path.join(nvidia_root, entry, "bin", "nvcc.exe"))
        for candidate in candidates:
            if os.path.isfile(candidate):
                cuda_home = os.path.dirname(os.path.dirname(candidate))
                _set_env_value(env, "CUDA_PATH", cuda_home)
                _set_env_value(env, "CUDA_HOME", cuda_home)
                _set_env_value(env, "PATH", os.path.dirname(candidate) + os.pathsep + _env_value(env, "PATH"))
                break
    scripts = os.path.join(sys.prefix, "Scripts")
    if os.path.isdir(scripts):
        _set_env_value(env, "PATH", scripts + os.pathsep + _env_value(env, "PATH"))
    missing = []
    if shutil.which("cl", path=_env_value(env, "PATH")) is None:
        missing.append("Visual Studio C++ (cl.exe)")
    if shutil.which("nvcc", path=_env_value(env, "PATH")) is None:
        missing.append("CUDA toolkit nvcc")
    if missing:
        hint = ""
        if "Visual Studio C++ (cl.exe)" in missing:
            hint = (" Install Visual Studio 2022 Build Tools with the C++ workload. "
                    "安装 Visual Studio 2022 生成工具并勾选 C++ 工作负载："
                    "winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "
                    "\"--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended\"")
        raise RuntimeError("Windows native build needs " + " and ".join(missing)
                           + " / Windows 上编译本地渲染器需要：" + "、".join(missing) + hint)
    # Torch cpp_extension refuses an already-activated VC prompt when these are unset.
    # 已激活的 VC 环境里，Torch 的 cpp_extension 缺少这两个变量就会拒绝编译。
    if shutil.which("cl", path=_env_value(env, "PATH")) is not None:
        if not _env_value(env, "DISTUTILS_USE_SDK"):
            _set_env_value(env, "DISTUTILS_USE_SDK", "1")
        if not _env_value(env, "MSSdk"):
            _set_env_value(env, "MSSdk", "1")
    return env


def prepare_raster_source(allow_replacement=None):
    """Build the pinned CUDA rasterizer against the current PyTorch, on first use.

    首次调用时按当前 PyTorch 编译固定版本 CUDA 光栅器；不替换 PyTorch。
    """
    import torch
    from wapr.installation import install_requirements
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA rasterizer requires working existing PyTorch CUDA / CUDA 光栅器要求已有 PyTorch 可使用 CUDA")
    _windows_compiler_env(os.environ.copy())
    preparation = install_requirements(["setuptools>=64", "wheel", "ninja"],
                                       allow_replacement=allow_replacement)
    if preparation.get("status") not in ("ready", "installed"):
        raise RuntimeError("Raster build tools not prepared / 光栅编译工具未准备: " + json.dumps(preparation, ensure_ascii=False))
    source = _checkout("nvdiffrast", "https://github.com/NVlabs/nvdiffrast.git", NVDIFFRAST_REVISION)
    build_env = _windows_compiler_env(os.environ.copy())
    build_env.setdefault("MAX_JOBS", "2")
    # The current GPU uniquely determines the default native architecture.
    # 当前显卡唯一确定默认本地编译架构；保留用户显式设置的架构列表。
    major, minor = torch.cuda.get_device_capability()
    build_env.setdefault("TORCH_CUDA_ARCH_LIST", str(major) + "." + str(minor))
    # Fingerprint source inputs, not generated objects or a guessed upstream version.
    # 对源码输入做指纹，不把构建产物或猜测的上游版本当作来源证明。
    source_digest = hashlib.sha256()
    for parent, directories, files in os.walk(source):
        directories[:] = sorted(name for name in directories
                                if name not in (".git", "build", "dist", "__pycache__") and not name.endswith(".egg-info"))
        for name in sorted(files):
            if not name.endswith((".py", ".cpp", ".cu", ".h", ".hpp", ".cuh", ".toml", ".txt", ".cmake")):
                continue
            path = os.path.join(parent, name)
            file_digest = hashlib.sha256()
            with open(path, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    file_digest.update(block)
            source_digest.update(os.path.relpath(path, source).encode("utf-8") + b"\0" + file_digest.digest())
    identity = {"python": os.path.realpath(sys.executable), "python_abi": sys.implementation.cache_tag,
                "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                "architectures": build_env["TORCH_CUDA_ARCH_LIST"], "gpu_capability": [major, minor],
                "source_sha256": source_digest.hexdigest(), "requested_revision": NVDIFFRAST_REVISION}
    stamp_directory = os.path.join(resource_root(), "native_wheels", "nvdiffrast")
    os.makedirs(stamp_directory, exist_ok=True)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    stamp_path = os.path.join(stamp_directory, key + ".json")

    def probe_raster():
        # Verify the CUDA operator, rather than accepting an import-only result.
        # 验证实际 CUDA 光栅算子，不能仅以 import 成功判断兼容。
        raster = importlib.import_module("nvdiffrast.torch")
        device = torch.device("cuda", torch.cuda.current_device())
        context = raster.RasterizeCudaContext(device=device)
        positions = torch.tensor([[[-0.5, -0.5, 0.0, 1.0], [0.5, -0.5, 0.0, 1.0],
                                   [0.0, 0.5, 0.0, 1.0]]], dtype=torch.float32, device=device)
        triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
        rendered, _ = raster.rasterize(context, positions, triangles, resolution=[8, 8])
        torch.cuda.synchronize(device)
        if tuple(rendered.shape) != (1, 8, 8, 4) or not bool((rendered[..., 3] > 0).any()):
            raise RuntimeError("Raster CUDA triangle probe failed / 光栅 CUDA 三角形验证失败")
        return os.path.realpath(raster.__file__)

    if os.path.isfile(stamp_path):
        try:
            with open(stamp_path, encoding="utf-8") as stream:
                stored = json.load(stream)
            if stored.get("identity") == identity and stored.get("module") == probe_raster():
                print("WAPR_RASTER_REUSE", identity, flush=True)
                return source
        except (ImportError, OSError, RuntimeError, ValueError):
            print("WAPR_RASTER_REBUILD Unhealthy cached operator / 缓存算子健康验证失败，重新构建", flush=True)
    print("WAPR_RASTER_BUILD", {"source": source, "requested_revision": NVDIFFRAST_REVISION,
                                "python": sys.executable, "torch": torch.__version__,
                                "torch_cuda": torch.version.cuda,
                                "architectures": build_env["TORCH_CUDA_ARCH_LIST"]}, flush=True)
    if sys.platform == "win32":
        # ATen includes these math headers. Pip nvcc does not ship them; they are not wheel dependencies.
        # ATen 会包含这些数学库头。pip 的 nvcc 不自带，也不写入 wheel 依赖。只在编译可选光栅器时补齐。
        nvcc_path = shutil.which("nvcc", path=_env_value(build_env, "PATH"))
        cuda_include = os.path.join(os.path.dirname(os.path.dirname(nvcc_path)), "include") if nvcc_path else ""
        header_names = ("cusparse.h", "cublas_v2.h", "cusolverDn.h")

        def missing_headers(directory):
            if not directory:
                return list(header_names)
            return [name for name in header_names if not os.path.isfile(os.path.join(directory, name))]

        if missing_headers(cuda_include):
            cuda_major = int(torch.version.cuda.split(".")[0])
            if cuda_major not in (11, 12, 13):
                raise RuntimeError("Unsupported CUDA header package family / 不支持的 CUDA 头文件包家族: " + str(cuda_major))
            suffix = "-cu%d" % cuda_major if cuda_major < 13 else ""
            requirements = ["nvidia-" + name + suffix for name in ("cusparse", "cublas", "cusolver")]
            print("WAPR_RASTER_CUDA_HEADERS", {"action": "install",
                                              "packages": requirements}, flush=True)
            result = install_requirements(requirements, allow_replacement=allow_replacement)
            if result.get("status") not in ("ready", "installed"):
                raise RuntimeError("CUDA math headers preparation stopped / CUDA 数学头文件准备已停止: " + json.dumps(result, ensure_ascii=False))
            nvidia_root = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
            include_dirs = [os.path.join(nvidia_root, "cu%d" % cuda_major, "include")]
            if os.path.isdir(nvidia_root):
                include_dirs.extend(os.path.join(nvidia_root, name, "include") for name in os.listdir(nvidia_root))
            if missing_headers(cuda_include):
                missing = [name for name in header_names if not any(os.path.isfile(os.path.join(directory, name)) for directory in include_dirs)]
                if missing:
                    raise RuntimeError(
                        "CUDA math headers still missing / CUDA 数学头文件仍然缺失: "
                        + ", ".join(missing) + " (nvcc include: " + (cuda_include or "missing") + ")")
                # MSVC reads INCLUDE. CPATH covers the other include search.
                # MSVC 读取 INCLUDE。CPATH 覆盖另一条头文件搜索路径。
                for variable in ("INCLUDE", "CPATH"):
                    current = _env_value(build_env, variable)
                    paths = [item for item in current.split(os.pathsep) if item]
                    for directory in include_dirs:
                        if os.path.isdir(directory) and directory not in paths:
                            paths.insert(0, directory)
                    _set_env_value(build_env, variable, os.pathsep.join(paths))
    with tempfile.TemporaryDirectory(prefix="wapr-raster-wheel-") as output:
        # Build without isolation so torch headers/ABI come from this interpreter.
        # 不隔离构建，确保 torch 头文件与 ABI 来自当前解释器；不解析或下载 torch。
        subprocess.check_call([sys.executable, "-m", "pip", "wheel", "--no-deps",
                               "--no-build-isolation", "--wheel-dir", output, source], env=build_env)
        wheels = [os.path.join(output, name) for name in os.listdir(output)
                  if name.startswith("nvdiffrast-") and name.endswith(".whl")]
        if len(wheels) != 1:
            raise RuntimeError("Expected one raster wheel / 预期一个光栅器 wheel")
        result = install_requirements(wheels, allow_replacement=allow_replacement)
        if result.get("status") not in ("ready", "installed"):
            raise RuntimeError("Raster installation stopped / 光栅器安装已停止: " + json.dumps(result, ensure_ascii=False))
    importlib.invalidate_caches()
    module = probe_raster()
    temporary_stamp = stamp_path + ".pending"
    with open(temporary_stamp, "w", encoding="utf-8") as stream:
        json.dump({"identity": identity, "module": module}, stream, sort_keys=True)
    os.replace(temporary_stamp, stamp_path)
    return source


def prepare_ycbineoat(sequence):
    """Fetch one original RGB-D sequence, excluding annotations from placement.

    获取一段原始 RGB-D 序列，放置时不包含标注；保留原始帧名和深度单位。
    """
    from pathlib import Path
    from wapr.resources import samples_dir
    from wapr.det2d import _download_file
    if sequence not in ("mustard_easy_00_02", "cracker_box_reorient", "sugar_box1"):
        raise ValueError("Unsupported tutorial sequence / 不支持的教程序列: " + sequence)
    target = Path(samples_dir()) / "YCBInEOAT" / sequence
    marker = target / ".wapr-sequence.json"
    if marker.is_file():
        stored = json.loads(marker.read_text())
        if stored.get("sequence") == sequence and stored.get("files") and all((target / name).is_file() and (target / name).stat().st_size == size for name, size in stored["files"].items()):
            return str(target)
    cache = Path(resource_root()) / ".downloads" / "ycbineoat"
    cache.mkdir(parents=True, exist_ok=True)
    archive_path = cache / (sequence + ".tar.gz")
    url = "https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCBInEOAT/" + sequence + ".tar.gz"
    if not archive_path.is_file():
        _download_file(url, str(archive_path))
    with tempfile.TemporaryDirectory(dir=cache) as staging:
        files = {}
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                parts = member.name.strip("/").split("/")
                if ".." in parts or member.name.startswith("/") or "\\" in member.name:
                    raise RuntimeError("Unsafe sequence archive path / 序列归档路径不安全")
                if sequence in parts:
                    parts = parts[parts.index(sequence) + 1:]
                if not parts or not (parts == ["cam_K.txt"] or len(parts) == 2 and parts[0] in ("rgb", "depth") and parts[1].endswith(".png")):
                    continue
                if not member.isfile():
                    raise RuntimeError("Sequence link/special file rejected / 拒绝序列链接及特殊文件")
                # Store archive-relative names with forward slashes. Path() on Windows
                # would write rgb\frame.png, and the rgb/ depth/ pairing check would see nothing.
                # 归档内相对路径用正斜杠。Windows 上 Path() 会写成反斜杠，rgb/ 与 depth/ 的配对检查会落空。
                relative = "/".join(parts)
                destination = Path(staging) / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                files[relative] = member.size
        rgb_stems = {Path(name).stem for name in files if name.startswith("rgb/")}
        depth_stems = {Path(name).stem for name in files if name.startswith("depth/")}
        if "cam_K.txt" not in files or len(rgb_stems & depth_stems) < 2 or rgb_stems != depth_stems:
            raise RuntimeError("Sequence lacks paired RGB-D/camera files / 序列缺少配对 RGB-D 或相机文件")
        target.mkdir(parents=True, exist_ok=True)
        for relative in files:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if not destination.is_file() or hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256((Path(staging) / relative).read_bytes()).digest():
                    raise RuntimeError("Existing sequence differs; retained / 已有序列不同，保留原件: " + relative)
            else:
                # Move verified missing files; avoid a second full sequence on disk.
                # 移动已审核的缺失文件，避免磁盘同时保存两份完整序列。
                shutil.move(str(Path(staging) / relative), str(destination))
        pending = marker.with_suffix(".pending")
        pending.write_text(json.dumps({"sequence": sequence, "source": url, "files": files}, indent=2))
        os.replace(pending, marker)
    return str(target)


def prepare_robot_assets(name):
    """Fetch one installed ManiSkill catalog entry with resumable HTTP.

    按当前 ManiSkill 的资源目录获取一项，支持续传且不删除已有资源。
    Called after robot dependency preparation, never during pip installation.
    仅在机器人依赖准备后调用，pip 安装时不触发。
    """
    from pathlib import Path
    from mani_skill.utils.assets.data import DATA_SOURCES
    from wapr.det2d import _download_file
    spec = DATA_SOURCES[name]
    target = Path(spec.output_dir) / spec.target_path
    marker = target / ".wapr-assets.json"
    if marker.is_file():
        record = json.loads(marker.read_text())
        if record.get("url") == spec.url and all((target / relative).is_file()
                                                for relative in record.get("files", [])) and record.get("files"):
            return str(target)
    if not spec.url or not spec.url.endswith(".zip"):
        raise RuntimeError("Unsupported ManiSkill asset catalog entry / 不支持的 ManiSkill 资源条目: " + name)
    cached = Path(resource_root()) / ".downloads" / "robot"
    cached.mkdir(parents=True, exist_ok=True)
    archive_path = cached / (name + ".zip")
    if not archive_path.is_file():
        _download_file(spec.url, str(archive_path))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if spec.checksum is not None and digest != spec.checksum:
        archive_path.unlink()
        raise RuntimeError("Robot archive checksum mismatch / 机器人资源归档校验失败: " + name)
    # Extract outside the destination; reject links and traversal before writing.
    # 先在目标目录外解压，写入前拒绝链接及越界路径。
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            archive_path.unlink()
            raise RuntimeError("Robot archive CRC failure / 机器人资源归档 CRC 校验失败")
        files = [member for member in archive.infolist() if not member.is_dir()]
        if not files:
            raise RuntimeError("Empty robot archive / 机器人资源归档为空")
        roots = {member.filename.split("/")[0] for member in files}
        strip_root = len(roots) == 1 and all("/" in member.filename for member in files)
        with tempfile.TemporaryDirectory(dir=cached) as staging:
            placed = []
            for member in files:
                parts = member.filename.split("/")
                if member.filename.startswith("/") or ".." in parts or "" in parts or "\\" in member.filename or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError("Unsafe robot archive member / 机器人归档成员路径不安全")
                relative = Path(*parts[1:] if strip_root else parts)
                output = Path(staging) / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, output.open("wb") as stream:
                    shutil.copyfileobj(source, stream)
                placed.append(str(relative))
            # Do not replace user-edited files; fill missing files only.
            # 不覆盖用户修改文件，仅补齐缺失文件；冲突时停止而非混用版本。
            for relative in placed:
                destination = target / relative
                source = Path(staging) / relative
                if destination.exists() and (not destination.is_file() or hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest()):
                    raise RuntimeError("Existing robot asset differs; retained / 已有机器人资源不同，保留原件: " + str(destination))
            for relative in placed:
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(Path(staging) / relative, destination)
            temporary = marker.with_suffix(".pending")
            temporary.write_text(json.dumps({"url": spec.url, "sha256": digest, "files": placed}, indent=2))
            os.replace(temporary, marker)
    return str(target)


def _github_archive(url, revision, candidate):
    """Fetch the same official revision when Git transport fails.

    Git 传输失败时获取同一官方版本的源码归档；安全解引用根目录内的相对链接。
    """
    parsed = urllib.parse.urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if parts[-1].endswith(".git"):
        parts[-1] = parts[-1][:-4]
    if parsed.hostname != "github.com" or len(parts) != 2 or not all(parts):
        raise RuntimeError("Archive fallback requires an official GitHub repository / 归档回退仅支持官方 GitHub 仓库")
    archive_url = "https://codeload.github.com/" + "/".join(parts) + "/tar.gz/" + revision
    archive_path = candidate + ".tar.gz"
    try:
        for direct in (False, True):
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
            try:
                with opener.open(archive_url, timeout=120) as response, open(archive_path, "wb") as stream:
                    headers = getattr(response, "headers", {})
                    declared_length = headers.get("Content-Length")
                    expected_bytes = int(declared_length) if declared_length and declared_length.isdigit() else None
                    shutil.copyfileobj(response, stream)
                actual_bytes = os.path.getsize(archive_path)
                print("WAPR_SOURCE_ARCHIVE_BYTES", {"revision": revision, "bytes": actual_bytes,
                                                   "expected_bytes": expected_bytes, "direct": direct}, flush=True)
                if expected_bytes is not None and actual_bytes != expected_bytes:
                    raise EOFError("Official archive response byte count differs / 官方归档响应字节数不符")
                # A proxy may terminate a successful HTTP response before the gzip stream ends.
                # 代理可能以成功 HTTP 状态返回截断归档；完整解析后才接受，避免跳过直连重试。
                with tarfile.open(archive_path, "r:gz") as validation_archive:
                    validation_archive.getmembers()
                break
            except (OSError, http.client.HTTPException, EOFError, tarfile.TarError):
                if direct:
                    raise RuntimeError("Official archive fetch failed / 官方归档获取失败") from None
        expected_root = parts[1] + "-" + revision
        os.makedirs(candidate, exist_ok=True)
        candidate_root = os.path.realpath(candidate)
        with tarfile.open(archive_path, "r:gz") as archive:
            deferred_links = []
            seen_paths = set()
            for member in archive.getmembers():
                names = member.name.rstrip("/").split("/")
                if names[0] != expected_root or "\\" in member.name or any(name in ("..", ".", ".git", "") for name in names[1:]):
                    raise RuntimeError("Unsafe archive member / 归档成员路径不安全")
                relative_path = "/".join(names[1:])
                if relative_path in seen_paths:
                    raise RuntimeError("Duplicate archive member / 归档成员重复")
                seen_paths.add(relative_path)
                if member.isdir():
                    continue
                destination = os.path.join(candidate, *names[1:])
                if os.path.commonpath([candidate_root, os.path.realpath(destination)]) != candidate_root:
                    raise RuntimeError("Archive destination escapes root / 归档写入路径越界")
                if member.issym() or member.islnk():
                    if not member.linkname or member.linkname.startswith("/") or "\\" in member.linkname or re.match(r"^[A-Za-z]:", member.linkname):
                        raise RuntimeError("Unsafe absolute archive link / 归档链接不是安全相对路径")
                    # Symlinks are relative to their parent; tar hardlinks name an archive member.
                    # 符号链接相对其父目录，tar 硬链接则使用归档成员路径。
                    link_base = posixpath.dirname(member.name) if member.issym() else ""
                    target_name = posixpath.normpath(posixpath.join(link_base, member.linkname))
                    target_parts = target_name.split("/")
                    if target_parts[0] != expected_root or any(name in ("..", ".git", "") for name in target_parts[1:]):
                        raise RuntimeError("Archive link escapes root / 归档链接目标越界")
                    target = os.path.join(candidate, *target_parts[1:])
                    deferred_links.append((destination, target, member.islnk()))
                    continue
                if not member.isfile():
                    raise RuntimeError("Unsupported archive link/member / 不支持归档链接或特殊成员")
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                with archive.extractfile(member) as source, open(destination, "wb") as output:
                    shutil.copyfileobj(source, output)
                os.chmod(destination, 0o755 if member.mode & 0o111 else 0o644)
            # Materialize links only after regular files, so no archive write follows a link.
            # 常规文件写完后才解引用链接，避免解包写入经过归档中的链接。
            while deferred_links:
                pending = []
                for destination, target, hardlink in deferred_links:
                    target_real = os.path.realpath(target)
                    if os.path.commonpath([candidate_root, target_real]) != candidate_root:
                        raise RuntimeError("Archive link escapes root / 归档链接目标越界")
                    if not os.path.exists(target):
                        pending.append((destination, target, hardlink))
                        continue
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    if os.path.isfile(target):
                        shutil.copy2(target, destination)
                    elif os.path.isdir(target) and not hardlink:
                        if os.path.commonpath([target_real, os.path.realpath(destination)]) == target_real:
                            raise RuntimeError("Recursive archive directory link / 归档目录链接递归")
                        # Directory copies must include their own deferred children.
                        # 目录解引用必须等待其内部链接完成，避免漏掉目录内文件。
                        if any(os.path.commonpath([target_real, os.path.realpath(other)]) == target_real
                               and not os.path.exists(other) for other, _, _ in deferred_links):
                            pending.append((destination, target, hardlink))
                            continue
                        shutil.copytree(target, destination)
                    else:
                        raise RuntimeError("Unsupported archive link target / 不支持归档链接目标")
                if len(pending) == len(deferred_links):
                    raise RuntimeError("Missing or cyclic archive link / 归档链接目标缺失或循环")
                deferred_links = pending
    finally:
        if os.path.exists(archive_path):
            os.unlink(archive_path)


def _checkout(name, url, revision, compatibility=""):
    """Create a verified checkout or return the user's existing directory.

    创建并核验检出，或直接返回用户已有目录；既有源码不打补丁。
    """
    parent = os.path.join(resource_root(), "third_party" if source_checkout else "sources")
    path = os.path.join(parent, name)
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise RuntimeError("Source path is not a directory / 源码路径不是目录: " + path)
        # Never let git discover the parent WAPR repository as this source's HEAD.
        # 不允许 git 把 WAPR 父仓库 HEAD 当成此源码版本。
        git_marker = os.path.join(path, ".git")
        if os.path.exists(git_marker) and shutil.which("git") is not None:
            current = subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD"],
                capture_output=True, encoding="utf-8", errors="replace", check=False,
            )
            print("WAPR_SOURCE_EXISTING", name, current.stdout.strip() or "unknown", path, flush=True)
        else:
            print("WAPR_SOURCE_EXISTING", name, "revision unknown / 版本未知", path, flush=True)
        print("WAPR_SOURCE_UNCHANGED", "Existing source retained; compatibility not verified / 保留已有源码，兼容性未验证", flush=True)
        return os.path.abspath(path)

    if shutil.which("git") is None:
        raise RuntimeError("git is required to fetch source / 获取源码需要 git")
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix="." + name + "-", dir=parent)
    candidate = os.path.join(staging, "checkout")
    archive_fetched = False
    try:
        # Retry the same official repository directly if configured proxies fail.
        # 代理失败时直连同一官方仓库，不更换未验证镜像。
        for direct in (False, True):
            environment = os.environ.copy()
            # Download checkpoints explicitly, not as a side effect of source checkout.
            # 权重由明确的下载入口获取，不在检出源码时隐式下载。
            environment["GIT_LFS_SKIP_SMUDGE"] = "1"
            git = ["git"]
            if direct:
                for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
                    environment.pop(key, None)
                git += ["-c", "http.proxy="]
            try:
                # Fetch only the requested revision; cloning HEAD first doubles work.
                # 仅获取指定版本，避免先 clone HEAD 再拉固定版本的重复传输。
                subprocess.run(
                    git + ["init", candidate],
                    env=environment, check=True, capture_output=True, encoding="utf-8", errors="replace", timeout=60,
                )
                subprocess.run(
                    git + ["-C", candidate, "remote", "add", "origin", url],
                    env=environment, check=True, capture_output=True, encoding="utf-8", errors="replace", timeout=60,
                )
                subprocess.run(
                    git + ["-C", candidate, "fetch", "--depth", "1", "origin", revision],
                    env=environment, check=True, capture_output=True, encoding="utf-8", errors="replace", timeout=180,
                )
                subprocess.run(
                    ["git", "-C", candidate, "checkout", "--detach", revision],
                    env=environment, check=True, capture_output=True, encoding="utf-8", errors="replace", timeout=60,
                )
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                shutil.rmtree(candidate, ignore_errors=True)
                if direct:
                    # Do not print credentials that may occur in proxy diagnostics.
                    # 不输出可能包含代理凭据的诊断信息。
                    print("WAPR_SOURCE_RETRY_ARCHIVE", name, revision, flush=True)
                    _github_archive(url, revision, candidate)
                    archive_fetched = True
                    break
                print("WAPR_SOURCE_RETRY_DIRECT", name, flush=True)
        current = revision if archive_fetched else subprocess.check_output(["git", "-C", candidate, "rev-parse", "HEAD"], encoding="utf-8", errors="replace").strip()
        if current != revision:
            raise RuntimeError("Source commit mismatch / 源码 commit 不匹配: " + name)

        if compatibility == "groundingdino":
            assets = os.path.join(MODULE_DIR, "runtime_assets", "groundingdino")
            copies = (
                ("ms_deform_attn.py", os.path.join("groundingdino", "models", "GroundingDINO", "ms_deform_attn.py")),
                ("groundingdino_swinb.json", os.path.join("groundingdino", "config", "groundingdino_swinb.json")),
                ("groundingdino_swint.json", os.path.join("groundingdino", "config", "groundingdino_swint.json")),
                ("LICENSE", "WAPR_COMPAT_LICENSE"),
            )
            for source_name, destination in copies:
                shutil.copyfile(os.path.join(assets, source_name), os.path.join(candidate, destination))
        elif compatibility in ("sam3d", "dinov2"):
            patch_name = "sam3d_wapr_compat.patch" if compatibility == "sam3d" else "dinov2_python38_annotations.patch"
            patch = os.path.join(MODULE_DIR, "runtime_assets", "patches", patch_name)
            patch_environment = os.environ.copy()
            # An archive has no .git; never discover an unrelated parent repository.
            # 归档没有 .git，禁止发现无关的上级仓库。
            patch_environment["GIT_CEILING_DIRECTORIES"] = staging
            subprocess.run(["git", "-C", candidate, "apply", "--check", patch], env=patch_environment, check=True, capture_output=True, encoding="utf-8", errors="replace")
            subprocess.run(["git", "-C", candidate, "apply", patch], env=patch_environment, check=True, capture_output=True, encoding="utf-8", errors="replace")
        # Rename only an absent destination; concurrent/existing source stays intact.
        # Windows can deny the rename while a scanner still holds a file. Copy after that.
        # 仅移动到缺失位置；保留并发生成或已有源码。
        # Windows 上扫描程序占用文件时 rename 会拒绝访问，这时改为复制。
        last_error = None
        moved = False
        for _ in range(8):
            if os.path.isdir(path):
                print("WAPR_SOURCE_CONCURRENT", name, "Existing directory retained / 保留已有目录", flush=True)
                return os.path.abspath(path)
            try:
                os.rename(candidate, path)
                moved = True
                break
            except OSError as error:
                last_error = error
                time.sleep(0.5)
        if not moved and sys.platform == "win32":
            try:
                for directory, _, filenames in os.walk(candidate):
                    os.chmod(directory, 0o700)
                    for filename in filenames:
                        os.chmod(os.path.join(directory, filename), 0o700)
                shutil.copytree(candidate, path)
                moved = True
            except OSError as error:
                last_error = error
        if not moved:
            raise last_error
        print("WAPR_SOURCE_PIN", name, revision, "Runtime compatibility unverified / 运行兼容性未验证", flush=True)
        return os.path.abspath(path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def prepare_source(feature, allow_replacement=None, check_only=False):
    """Return optional source without installing its training/build dependencies.

    返回可选功能源码，不安装训练或构建依赖；det2d 同时准备三项源码。
    """
    if feature == "dinov2":
        return _checkout("dinov2", "https://github.com/facebookresearch/dinov2.git", DINOV2_REVISION, "dinov2")
    if feature == "det2d":
        prepare_source("dinov2")
        _checkout("ultralytics", "https://github.com/ultralytics/ultralytics.git", ULTRALYTICS_REVISION)
        return _checkout("GroundingDINO", "https://github.com/IDEA-Research/GroundingDINO.git", GROUNDING_REVISION, "groundingdino")
    if feature == "sam2":
        return _checkout("sam2", "https://github.com/facebookresearch/sam2.git", SAM2_REVISION)
    if feature == "sam3d":
        prepare_source("sam2")
        return _checkout("sam-3d-objects", "https://github.com/facebookresearch/sam-3d-objects.git", SAM3D_REVISION, "sam3d")
    if feature == "roma":
        # WAPR imports the pinned inference source directly, not the training package.
        # WAPR 直接导入固定版本推理源码，不构建或安装上游训练包。
        return _checkout("RoMa", "https://github.com/Parskatt/RoMa.git", ROMA_REVISION)
    if feature == "unipose9d":
        return _checkout("UniPose9D", "https://github.com/qq456cvb/UniPose9D.git", UNIPOSE9D_REVISION)
    raise ValueError("Unknown source feature / 未知源码功能: " + str(feature))


def prepare_sam_dino_weights():
    """Cache the official DINOv2 ViT-L/14 four-register checkpoint used by SAM3D.

    缓存 SAM3D 使用的官方 DINOv2 ViT-L/14 四寄存器权重；指纹来自审核缓存。
    """
    import torch
    from wapr.det2d import _download_file
    hub_directory = os.path.join(os.environ.get("TORCH_HOME", os.path.join(resource_root(), "torchhub")), "hub")
    # Use one cache for the resumable download and upstream torch.hub state loading.
    # 续传下载与上游 torch.hub 权重加载共用同一个缓存目录。
    torch.hub.set_dir(hub_directory)
    directory = os.path.join(hub_directory, "checkpoints")
    os.makedirs(directory, exist_ok=True)
    name = "dinov2_vitl14_reg4_pretrain.pth"
    path = os.path.join(directory, name)
    # The pinned official hub constructs this URL from vit_large, patch 14, registers 4.
    # 固定官方 hub 按 vit_large、patch 14、寄存器 4 构造此地址；并非普通 ViT-L 权重。
    url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/" + name
    expected_size = 1217607321
    expected_sha256 = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
    if not os.path.isfile(path):
        _download_file(url, path)
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if os.path.getsize(path) != expected_size or digest.hexdigest() != expected_sha256:
        raise RuntimeError("SAM DINO checkpoint fingerprint differs; retained / SAM DINO 权重指纹不同，保留文件等待检查: " + path)
    return path


def prepare_unipose_weights():
    """Fetch the pinned public checkpoint separately from the source checkout.

    独立下载固定公开权重并核验 Git LFS SHA256，不覆盖使用者源码中的权重。
    """
    from wapr.resources import weights_dir
    from filelock import FileLock
    directory = os.path.join(weights_dir(), "unipose9d")
    os.makedirs(directory, exist_ok=True)
    destination = os.path.join(directory, "last.ckpt")
    # Serialize download, checksum and rename as one cache transaction.
    # 将下载、校验与正式文件重命名放在同一锁内，避免并发首次调用竞态。
    with FileLock(destination + ".lock"):
        return _prepare_unipose_weights_body(destination)


def _prepare_unipose_weights_body(destination):
    source = prepare_source("unipose9d")
    expected_sha256 = "02c91a3df7d37740e552c421c9bd697189155c7e3936fdb1e7798a5e16dd3e73"
    expected_size = 371248494
    if not os.path.isfile(destination):
        url = "https://media.githubusercontent.com/media/qq456cvb/UniPose9D/" + UNIPOSE9D_REVISION + "/checkpoints/last.ckpt"
        from wapr.det2d import _download_file
        # Both URLs address the same pinned Git LFS object; reject pointer-size responses.
        # 两个官方地址指向同一固定 Git LFS 对象；测速拒绝只有指针大小的响应。
        from wapr.download_route import probe_prefix, public_url
        raw_url = "https://github.com/qq456cvb/UniPose9D/raw/" + UNIPOSE9D_REVISION + "/checkpoints/last.ckpt"
        measured = []
        for candidate in (url, raw_url):
            sample = probe_prefix(candidate, expected_total=expected_size)
            if sample is not None and sample["bytes"] > 0:
                measured.append((sample["seconds"] / sample["bytes"], candidate))
        if measured:
            url = min(measured)[1]
            print("WAPR_UNIPOSE_IDENTICAL_WEIGHT_ROUTE", {"url": public_url(url), "sha256": expected_sha256}, flush=True)
        # Keep resumable bytes separate from the checksum-verified checkpoint.
        # 续传缓存与通过校验的正式权重分开，网络中断后不丢失已下载内容。
        temporary = destination + ".download"
        _download_file(url, temporary)
        try:
            digest = hashlib.sha256()
            size = 0
            with open(temporary, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if size != expected_size or digest.hexdigest() != expected_sha256:
                raise RuntimeError("UniPose9D checkpoint checksum mismatch / UniPose9D 权重校验失败")
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    digest = hashlib.sha256()
    with open(destination, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if os.path.getsize(destination) != expected_size or digest.hexdigest() != expected_sha256:
        raise RuntimeError("Existing UniPose9D checkpoint checksum mismatch / 已有 UniPose9D 权重校验失败")
    return {"checkpoint": destination, "config": os.path.join(source, "checkpoints", "config.yaml"),
            "sha256": expected_sha256, "source": source}


def _qwen_weights_ready(directory):
    """Validate the referenced shards, not merely the snapshot directory.

    检查索引引用的所有分片，不能把已有目录或配置文件当成下载完成。
    """
    required = ["config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"]
    if not all(os.path.isfile(os.path.join(directory, name)) for name in required):
        return False
    try:
        index_path = os.path.join(directory, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as stream:
                index = json.load(stream)
            shards = sorted(set(index["weight_map"].values()))
        else:
            shards = ["model.safetensors"]
        if not shards:
            return False
        from safetensors import safe_open
        for name in shards:
            if os.path.basename(name) != name:
                return False
            with safe_open(os.path.join(directory, name), framework="pt", device="cpu") as shard:
                if not list(shard.keys()):
                    return False
        return True
    except Exception:
        # Safetensors errors also cover a truncated final shard.
        # safetensors 的检查也能识别最后分片被截断的情况。
        return False


def prepare_qwen_weights():
    """Download the optional public language model only when requested.

    仅在请求时下载可选公开语言模型；权重不进入 wheel，不使用维护者凭据。
    """
    from huggingface_hub import HfApi, hf_hub_url
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, LocalEntryNotFoundError, HfHubHTTPError
    from wapr.download_assets import hub_official, hub_mirror
    from wapr.resources import weights_dir
    directory = os.path.join(weights_dir(), "Qwen2.5-VL-3B-Instruct")
    if _qwen_weights_ready(directory):
        return directory
    from wapr.download_route import hub_endpoints
    from wapr.det2d import _download_file
    endpoints = hub_endpoints(hub_official, hub_mirror)
    for endpoint in endpoints:
        try:
            # Pin one public snapshot before downloading shards; do not mix moving branches.
            # 下载分片前固定一次公开快照；不能混用会移动的分支内容。
            repository = "Qwen/Qwen2.5-VL-3B-Instruct"
            info = HfApi(endpoint=endpoint, token=False).model_info(repository, files_metadata=True, timeout=30)
            revision = info.sha
            if not revision or not info.siblings:
                raise RuntimeError("Qwen snapshot identity missing / Qwen 快照身份缺失")
            os.makedirs(directory, exist_ok=True)
            identities = []
            for item in info.siblings:
                name = item.rfilename
                if not (name.endswith((".json", ".safetensors")) or name in ("merges.txt", "vocab.txt", "tokenizer.model")):
                    continue
                if os.path.basename(name) != name:
                    raise RuntimeError("Unexpected Qwen artifact path / Qwen 资源路径异常")
                size = item.size
                lfs = item.lfs
                fingerprint = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else item.blob_id
                if not isinstance(size, int) or size < 1 or not fingerprint:
                    raise RuntimeError("Qwen artifact identity missing / Qwen 资源身份缺失: " + name)
                path = os.path.join(directory, name)
                if not os.path.isfile(path):
                    _download_file(hf_hub_url(repository, name, revision=revision, endpoint=endpoint), path)
                if os.path.getsize(path) != size:
                    raise RuntimeError("Qwen artifact byte count differs; retained / Qwen 资源字节数不符，保留: " + name)
                digest = hashlib.sha256() if lfs else hashlib.sha1()
                if not lfs:
                    digest.update(("blob " + str(size) + "\0").encode("ascii"))
                with open(path, "rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != fingerprint:
                    raise RuntimeError("Qwen artifact checksum differs; retained / Qwen 资源摘要不符，保留: " + name)
                identities.append({"name": name, "bytes": size, "fingerprint": fingerprint})
            stamp = os.path.join(directory, ".wapr-snapshot.json")
            with open(stamp + ".pending", "w", encoding="utf-8") as stream:
                json.dump({"repository": repository, "revision": revision, "files": identities}, stream, indent=2)
            os.replace(stamp + ".pending", stamp)
            if _qwen_weights_ready(directory):
                return directory
            print("WAPR_QWEN_RETRY", "Incomplete snapshot / 快照文件不完整", flush=True)
        except LocalEntryNotFoundError:
            print("WAPR_QWEN_RETRY", "Network/cache miss / 网络或缓存未命中", flush=True)
        except GatedRepoError:
            raise
        except RepositoryNotFoundError:
            # A public mirror may not have this repository yet; try the original.
            # 公开镜像可能尚未同步仓库；继续尝试原始官方源。
            if endpoint == hub_official.rstrip("/"):
                raise
            print("WAPR_QWEN_RETRY", "Repository absent on mirror / 镜像未同步仓库", flush=True)
        except (OSError, HfHubHTTPError) as error:
            # Public weights need no token; retry only transport/server failures.
            # 公开权重不使用 token，仅对传输或服务端故障尝试下一源。
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status is not None and status < 500 and status not in (408, 429):
                raise
            print("WAPR_QWEN_RETRY", type(error).__name__, flush=True)
    raise RuntimeError("Public Qwen checkpoint download failed / 公开 Qwen 权重下载失败")


def prepare_sam2_weights(model="large"):
    """Fetch an official SAM2.1 checkpoint and its matching configuration.

    获取官方 SAM2.1 权重及匹配配置；默认沿用 large，测试可显式选择 tiny。
    These candidates are not a claim of compatibility with the current environment.
    提供这些候选不表示当前环境已经通过兼容验证；导入模块不下载资源。
    """
    import zipfile
    from wapr.det2d import _download_file
    from wapr.resources import weights_dir

    candidates = {
        "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
        "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    }
    if model not in candidates:
        raise ValueError("SAM2 model must be large or tiny / SAM2 型号必须为 large 或 tiny: " + str(model))
    filename, config = candidates[model]
    source = prepare_source("sam2")
    config_path = os.path.join(source, "sam2", config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError("SAM2 configuration is missing / SAM2 配置缺失: " + config_path)
    directory = weights_dir()
    os.makedirs(directory, exist_ok=True)
    checkpoint = os.path.join(directory, filename)
    if not os.path.isfile(checkpoint):
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/" + filename
        _download_file(url, checkpoint)
    # The public download catalog has no checksum; verify ZIP indexes and data CRCs.
    # 公开下载目录未提供校验值，核验 ZIP 索引和数据 CRC，损坏缓存不能表示准备成功。
    try:
        with zipfile.ZipFile(checkpoint) as archive:
            broken_record = archive.testzip()
            if broken_record is not None:
                raise RuntimeError("Invalid SAM2 checkpoint record / SAM2 权重数据记录损坏: " + broken_record)
    except (zipfile.BadZipFile, EOFError, OSError) as error:
        raise RuntimeError("Incomplete SAM2 checkpoint / SAM2 权重不完整: " + checkpoint) from error
    return {"checkpoint": checkpoint, "config": config, "config_path": config_path, "source": source}
