# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Call SAM3D in a separate interpreter. The pose process does not import it.

用另一个解释器调用 SAM3D。位姿进程不导入 SAM3D，也不替换当前 PyTorch。
"""

import json
import importlib.util
import os
import subprocess
import sys
import tempfile
import platform
import re
import urllib.parse
import urllib.request
import hashlib
import shutil
import tarfile
import threading
import signal
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from wapr.resources import resource_root


# Python 3.11 has the official Open3D 0.18.0 wheel. Python 3.12 does not.
# Python 3.11 有官方 Open3D 0.18.0 wheel。Python 3.12 没有。
SAM3D_PYTHON_VERSION = "3.11"
OPEN3D_VERSION = "0.18.0"
# Keep the independent prefix with user-selected resources; an explicit override wins.
# 独立环境随用户选择的资源根目录保存；显式环境变量优先。
SAM3D_ENV_ROOT = os.environ.get(
    "WAPR_SAM3D_ENV",
    os.path.join(resource_root(), "environments", "sam3d"),
)
SAM3D_PYTHON = os.path.join(SAM3D_ENV_ROOT, "bin", "python")


def _worker_cache_environment(environment):
    """Keep large temporary downloads and native caches on the resource volume.

    大下载临时文件及原生缓存留在资源盘；保留用户显式的缓存路径。
    """
    # CUDA JIT reads UTF-8 source files even when the SSH login locale is ASCII.
    # SSH 登录区域设置可能为 ASCII；CUDA JIT 仍需读取 UTF-8 原生源码。
    environment['PYTHONUTF8'] = '1'
    from wapr.resources import resource_root
    cache_root = resource_root()
    # pip's Ninja executable lives in the selected prefix; child JIT builds need it.
    # pip 安装的 Ninja 可执行文件位于所选前缀；子进程 JIT 编译需要该目录。
    environment["PATH"] = os.path.dirname(SAM3D_PYTHON) + os.pathsep + environment.get("PATH", "")
    paths = {"TMPDIR": os.path.join(cache_root, ".tmp"),
             "PIP_CACHE_DIR": os.path.join(cache_root, ".pip-cache"),
             "TORCH_HOME": os.path.join(cache_root, "torchhub"),
             "TORCH_EXTENSIONS_DIR": os.path.join(cache_root, "native_extensions")}
    for key, directory in paths.items():
        environment.setdefault(key, directory)
        os.makedirs(environment[key], exist_ok=True)
    return environment


def _select_python_mirror(environment):
    """Measure equivalent managed Python releases, preserving user mirror choices.

    测速同一独立 Python 发行的镜像；保留用户镜像，完整文件由 UV 内置 SHA256 清单核验。
    """
    user_keys = ("UV_PYTHON_INSTALL_MIRROR", "UV_ASTRAL_MIRROR_URL", "UV_PYTHON_DOWNLOADS_JSON_URL")
    if any(environment.get(key) for key in user_keys):
        return
    command = [sys.executable, "-m", "uv", "python", "list", SAM3D_PYTHON_VERSION,
               "--only-downloads", "--output-format", "json"]
    try:
        listed = subprocess.run(command, env=environment, capture_output=True, text=True,
                                check=False, timeout=30)
        entries = json.loads(listed.stdout)
        entry = next(item for item in entries if item["implementation"] == "cpython"
                     and item["arch"] == "x86_64" and item["os"] == "linux"
                     and item["libc"] == "gnu" and item.get("variant", "default") == "default"
                     and item["version"].startswith(SAM3D_PYTHON_VERSION + "."))
        official_url = entry["url"]
        suffix = official_url.split("/download/", 1)[1]
        from wapr.download_route import probe_prefix
        github = "https://github.com/astral-sh/python-build-standalone/releases/download"
        nju = "https://mirrors.nju.edu.cn/github-release/astral-sh/python-build-standalone"
        candidates = [("official", None, official_url), ("github", github, github + "/" + suffix),
                      ("nju", nju, nju + "/" + suffix)]
        with ThreadPoolExecutor(max_workers=3) as workers:
            samples = list(workers.map(lambda candidate: probe_prefix(candidate[2]), candidates))
        measured = []
        for candidate, sample in zip(candidates, samples):
            print("WAPR_PYTHON_DOWNLOAD_ROUTE", {"name": candidate[0], "ok": sample is not None,
                                               "seconds": None if sample is None else round(sample["seconds"], 3)}, flush=True)
            if sample is not None and sample["bytes"] > 0:
                measured.append((sample["bytes"] / max(sample["seconds"], 0.001), candidate))
        if measured:
            selected = max(measured, key=lambda item: item[0])[1]
            if selected[1] is not None:
                environment["UV_PYTHON_INSTALL_MIRROR"] = selected[1]
            print("WAPR_PYTHON_DOWNLOAD_SELECTED", selected[0], "build=" + entry["version"], flush=True)
    except (OSError, ValueError, KeyError, IndexError, StopIteration, subprocess.SubprocessError):
        # Catalog/probe failure keeps the official route and bounded retry behavior.
        # 清单或测速失败时仍保留官方路线及有限重试，不猜下载日期或版本。
        print("WAPR_PYTHON_DOWNLOAD_ROUTE", {"reason": "probe_failed", "route": "official"}, flush=True)


def _run_preparation_worker(mode, path, environment, worker_file=None):
    """Forward preparation progress while keeping the final JSON protocol private.

    转发准备过程的进度，保留最终 JSON 返回协议；打印前隐藏完整 URL，避免签名地址泄漏。
    """
    environment = _worker_cache_environment(environment)
    worker_file = os.path.abspath(__file__) if worker_file is None else worker_file
    completed = subprocess.Popen([SAM3D_PYTHON, worker_file, mode, path],
                                 env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    recent = []
    final_line = ""
    previous_printed = None
    for line in completed.stdout:
        final_line = line.strip()
        safe_line = re.sub(r"https?://\S+", "[URL omitted / 地址已隐藏]", line).rstrip()
        recent.append(safe_line)
        recent = recent[-12:]
        if not final_line.startswith("{") and safe_line != previous_printed:
            print("WAPR_SAM3D_PREPARE", safe_line, flush=True)
            previous_printed = safe_line
    exit_code = completed.wait()
    try:
        result = json.loads(final_line)
    except ValueError:
        result = {"status": "failed", "exit_code": exit_code,
                  "reason": "Isolated preparation worker did not return JSON / 独立准备进程未返回 JSON",
                  "diagnostic": "\n".join(recent)[-2000:]}
    print("WAPR_SAM3D_PREPARE_STATUS", result["status"], flush=True)
    return result


def prepare_environment(allow_replacement=None, check_only=False):
    """Prepare a separate interpreter with a hardware compatible Torch pair.

    按硬件兼容的 Torch/torchvision 组合准备独立解释器；不修改基础环境。
    """
    if sys.platform != "linux" or platform.machine() != "x86_64":
        return {"status": "blocked", "reason": "This verified isolated recipe requires Linux x86_64 / 此已核验独立方案需要 Linux x86_64"}
    if os.path.realpath(SAM3D_PYTHON) == os.path.realpath(sys.executable):
        return {"status": "blocked", "reason": "The selected interpreter is the caller / 所选解释器是调用者本身"}
    try:
        torch_version = metadata.version("torch")
        vision_version = metadata.version("torchvision")
    except metadata.PackageNotFoundError as error:
        return {"status": "blocked", "reason": "The caller needs an existing Torch/torchvision pair / 调用者需要已有 Torch/torchvision 组合: " + str(error)}
    cuda_match = re.fullmatch(r"[0-9.]+\+cu([0-9]+)", torch_version)
    if cuda_match is None or not vision_version.endswith("+cu" + cuda_match.group(1)):
        return {"status": "blocked", "reason": "The existing pair has no matching explicit CUDA build / 已有组合缺少相同的明确 CUDA 构建标识", "torch": torch_version, "torchvision": vision_version}
    base_torch_version = torch_version
    base_vision_version = vision_version
    import torch
    if not torch.cuda.is_available():
        return {"status": "blocked", "reason": "This isolated CUDA recipe needs a visible GPU / 此独立 CUDA 方案需要可见 GPU"}
    gpu_capability = torch.cuda.get_device_capability()
    # Ada and earlier GPUs use the tested SAM stack; newer GPUs need verification.
    # Ada 及更早 GPU 使用已验证 SAM 软件栈；更新架构必须先验证。
    if gpu_capability[0] >= 10:
        return {"status": "blocked", "reason": "The isolated SAM native stack is not verified for this GPU architecture / 此 GPU 架构的独立 SAM 原生软件栈尚未验证",
                "base_torch": base_torch_version, "gpu_capability": list(gpu_capability),
                "base_environment_unchanged": True}
    if gpu_capability[0] < 10:
        base_cuda = int(cuda_match.group(1))
        if base_cuda >= 124:
            isolated_cuda = "124"
        elif base_cuda >= 121:
            isolated_cuda = "121"
        elif base_cuda == 118:
            isolated_cuda = "118"
        else:
            return {"status": "blocked", "reason": "No verified isolated recipe for this CUDA build / 此 CUDA 构建没有已验证独立方案", "base_torch": base_torch_version}
        torch_version = "2.5.1+cu" + isolated_cuda
        vision_version = "0.20.1+cu" + isolated_cuda
        cuda_match = re.fullmatch(r"[0-9.]+\+cu([0-9]+)", torch_version)
    print("WAPR_SAM3D_STACK", "base=" + base_torch_version, "isolated=" + torch_version,
          "GPU=" + torch.cuda.get_device_name(), flush=True)
    if not os.path.isfile(SAM3D_PYTHON):
        interpreter_plan = {"status": "needs_interpreter", "python_version": SAM3D_PYTHON_VERSION,
                            "prefix": SAM3D_ENV_ROOT, "manager": "uv", "torch": torch_version,
                            "torchvision": vision_version}
        if check_only:
            return interpreter_plan
        if os.path.exists(SAM3D_ENV_ROOT) and (not os.path.isdir(SAM3D_ENV_ROOT) or os.listdir(SAM3D_ENV_ROOT)):
            interpreter_plan.update(status="blocked", reason="The requested prefix is nonempty but has no interpreter; retain its files / 目标前缀非空却没有解释器；保留其中文件")
            return interpreter_plan
        from wapr.installation import install_requirements
        uv_prepared = install_requirements(["uv"], allow_replacement=allow_replacement)
        if uv_prepared["status"] not in ("ready", "installed"):
            return uv_prepared
        from wapr.download_route import choose_pypi_route
        route = choose_pypi_route()
        # uv downloads managed Python and creates a venv; it never switches the caller.
        # uv 下载独立管理的 Python 并创建 venv；从不切换调用者的解释器。
        command = [sys.executable, "-m", "uv", "venv", "--python", SAM3D_PYTHON_VERSION,
                   "--managed-python", "--seed", "--no-config", SAM3D_ENV_ROOT]
        environment = os.environ.copy()
        if not environment.get("UV_DEFAULT_INDEX") and not environment.get("UV_INDEX_URL"):
            environment["UV_DEFAULT_INDEX"] = route["index"]
        # Keep managed interpreter/cache downloads on the user data volume.
        # 独立解释器和下载缓存放在用户数据盘；尊重用户已有 UV_* 配置。
        from wapr.resources import resource_root
        environment.setdefault("UV_PYTHON_INSTALL_DIR", os.path.join(resource_root(), "interpreters"))
        environment.setdefault("UV_CACHE_DIR", os.path.join(resource_root(), ".uv-cache"))
        environment.setdefault("UV_HTTP_TIMEOUT", "30")
        environment.setdefault("UV_HTTP_RETRIES", "1")
        user_python_mirror = any(environment.get(key) for key in
                                ("UV_PYTHON_INSTALL_MIRROR", "UV_ASTRAL_MIRROR_URL", "UV_PYTHON_DOWNLOADS_JSON_URL"))
        _select_python_mirror(environment)
        created_code = 1
        recent = []
        for direct in (False, True):
            if direct:
                for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                            "UV_HTTP_PROXY", "UV_HTTPS_PROXY", "UV_PROXY"):
                    environment.pop(key, None)
                # Retry the same official release via GitHub when its CDN stalls.
                # 官方 CDN 停滞时，改从 GitHub 获取相同官方发行；保留用户镜像。
                if not user_python_mirror:
                    environment["UV_PYTHON_INSTALL_MIRROR"] = "https://github.com/astral-sh/python-build-standalone/releases/download"
            print("WAPR_SAM_INTERPRETER", {"python": SAM3D_PYTHON_VERSION, "direct": direct,
                                          "custom_mirror": bool(environment.get("UV_PYTHON_INSTALL_MIRROR")),
                                          "timeout_seconds": 180}, flush=True)
            created = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            # A hard stage bound prevents silent multi-hour GitHub retries.
            # 整个阶段有超时上限，避免 GitHub 重试静默持续数小时。
            timer = threading.Timer(180, os.killpg, args=(created.pid, signal.SIGKILL))
            timer.start()
            try:
                for line in created.stdout:
                    safe_line = re.sub(r"https?://\S+", "[URL omitted / 地址已隐藏]", line).rstrip()
                    recent.append(safe_line)
                    recent = recent[-12:]
                    print("WAPR_SAM_INTERPRETER", safe_line, flush=True)
                created_code = created.wait()
            finally:
                timer.cancel()
            if created_code == 0 and os.path.isfile(SAM3D_PYTHON):
                break
        if created_code != 0 or not os.path.isfile(SAM3D_PYTHON):
            interpreter_plan.update(status="failed", exit_code=created_code,
                                    reason="Managed Python/venv preparation failed / 独立 Python/venv 准备失败",
                                    stderr="\n".join(recent)[-1500:],
                                    guidance="Set UV_PYTHON_INSTALL_MIRROR to an accessible trusted mirror, or supply WAPR_SAM3D_ENV with a Python 3.11 prefix / 可设置可信且可达的 UV_PYTHON_INSTALL_MIRROR，或通过 WAPR_SAM3D_ENV 提供 Python 3.11 前缀")
            return interpreter_plan
    index = "https://download.pytorch.org/whl/cu" + cuda_match.group(1)
    # Inspect only the official index, never download a CUDA wheel just to plan.
    # 只读取官方索引；不为生成计划而下载完整 CUDA wheel。
    requirements = ["torch==" + torch_version, "torchvision==" + vision_version]
    request = {"requirements": requirements, "allow_replacement": allow_replacement,
               "check_only": check_only, "index": index, "probe_only": True}
    with tempfile.TemporaryDirectory(prefix="wapr-sam3d-prepare-") as directory:
        path = os.path.join(directory, "request.json")
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(request, stream)
        environment = os.environ.copy()
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
        # Exact CUDA requirements resolve on this official supplemental index.
        # 精确 CUDA 版本通过这个官方补充索引解析，普通依赖仍可使用测速源。
        environment["PIP_EXTRA_INDEX_URL"] = index
        result = _run_preparation_worker("prepare-worker", path, environment)
        # A healthy prepared prefix is usable offline; only missing stacks consult indexes.
        # 健康的已有前缀可以离线使用；只有缺失软件栈才查询索引。
        if result["status"] == "needs_preparation":
            try:
                for name, version in (("torch", torch_version), ("torchvision", vision_version)):
                    with urllib.request.urlopen(index + "/" + name + "/", timeout=30) as response:
                        listing = response.read().decode("utf-8")
                    prefix = name + "-" + version + "-cp311-cp311-"
                    available = [urllib.parse.unquote(link).split("#", 1)[0].rsplit("/", 1)[-1]
                                 for link in re.findall(r'href="([^"]+)"', listing)]
                    # New official Torch wheels use manylinux tags; pip validates host glibc.
                    # 新版官方 Torch wheel 使用 manylinux 标签；实际主机 glibc 由独立 pip 校验。
                    candidates = [filename for filename in available if filename.startswith(prefix)
                                  and re.fullmatch(r"(?:linux|manylinux[^/]+)_x86_64\.whl", filename[len(prefix):])]
                    if not candidates:
                        return {"status": "blocked", "reason": "No official Python 3.11 CUDA wheel for the existing pair / 已有组合没有官方 Python 3.11 CUDA wheel", "missing": prefix + "Linux x86_64"}
            except Exception as error:
                return {"status": "blocked", "reason": "Official Torch index check failed / 官方 Torch 索引检查失败: " + type(error).__name__}
            request["probe_only"] = False
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(request, stream)
            result = _run_preparation_worker("prepare-worker", path, environment)
    result.update(caller_python=sys.executable, isolated_python=SAM3D_PYTHON,
                  torch=torch_version, torchvision=vision_version,
                  base_torch=base_torch_version, base_torchvision=base_vision_version,
                  gpu_capability=list(gpu_capability))
    return result


def _prepare_worker(path):
    """Install only inside this existing prefix; replacement needs approval.

    只在这个已有前缀内安装；替换已有依赖仍需明确批准。
    """
    if sys.version_info[:2] != (3, 11) or sys.platform != "linux" or platform.machine() != "x86_64":
        print(json.dumps({"status": "blocked", "reason": "Expected Python 3.11 Linux x86_64 / 需要 Python 3.11 Linux x86_64"}), flush=True)
        return 1
    with open(path, encoding="utf-8") as stream:
        request = json.load(stream)
    from wapr.installation import install_requirements, _installed_requirements_healthy
    # A separate ABI prefix cannot borrow the caller's installed dependencies.
    # 独立 ABI 前缀不能借用调用者已有依赖；包含 wheel 核心及 SAM3D 准备入口依赖。
    requirements = request["requirements"] + ["numpy", "trimesh", "Pillow", "huggingface-hub",
                                              "packaging", "requests", "PyYAML"]
    if request.get("probe_only"):
        status = "ready" if _installed_requirements_healthy(requirements) else "needs_preparation"
        print(json.dumps({"status": status, "requirements": requirements}), flush=True)
        return 0
    result = install_requirements(requirements, allow_replacement=request["allow_replacement"],
                                  check_only=request["check_only"])
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] in ("ready", "installed", "approval_required") else 1


def prepare_reconstruction_environment(allow_replacement=None, check_only=False, checkpoint_directory=None):
    """Prepare the complete SAM3D stack inside the separate interpreter.

    在独立解释器内准备完整 SAM3D 软件栈；基础环境仅负责启动子进程。
    """
    bootstrap = prepare_environment(allow_replacement=allow_replacement, check_only=check_only)
    if bootstrap["status"] not in ("ready", "installed") or (check_only and bootstrap.get("install")):
        return bootstrap
    request = {"allow_replacement": allow_replacement, "check_only": check_only,
               "checkpoint_directory": os.fspath(checkpoint_directory) if checkpoint_directory is not None else None}
    with tempfile.TemporaryDirectory(prefix="wapr-sam3d-full-prepare-") as directory:
        path = os.path.join(directory, "request.json")
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(request, stream)
        environment = os.environ.copy()
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
        result = _run_preparation_worker("prepare-reconstruction-worker", path, environment)
        # Only the child imported the replaced native dependency; restart it once.
        # 只有子进程导入了被更换的原生依赖；仅重启该子进程一次，不重启基础环境。
        if not check_only and result["status"] == "restart_required":
            print("WAPR_SAM3D_PREPARE_RESTART", result.get("restart_packages", []), flush=True)
            result = _run_preparation_worker("prepare-reconstruction-worker", path, environment)
    result["bootstrap"] = bootstrap
    return result


def _reconstruction_prepare_worker(path):
    """Run native preparation in the selected prefix, not the pose interpreter.

    在所选前缀执行原生依赖准备，不在位姿解释器内安装。
    """
    with open(path, encoding="utf-8") as stream:
        request = json.load(stream)
    toolchain = _prepare_cuda_toolchain(check_only=request["check_only"])
    if toolchain["status"] not in ("ready", "needs_toolchain"):
        print(json.dumps(toolchain, ensure_ascii=False), flush=True)
        return 1
    from wapr.reconstruction_setup import prepare_reconstruction
    result = prepare_reconstruction(allow_replacement=request["allow_replacement"],
                                    check_only=request["check_only"],
                                    checkpoint_directory=request["checkpoint_directory"])
    result["toolchain"] = toolchain
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] in ("ready", "native_build_required", "approval_required") else 1


def _prepare_cuda_toolchain(check_only=False):
    """Select an exact compiler or fetch NVIDIA components into the private cache.

    选择匹配的编译器，或将 NVIDIA 组件下载到私有缓存；不改系统 CUDA 或驱动。
    """
    import torch
    from wapr.resources import resource_root
    cuda_version = torch.version.cuda
    releases = {"12.4": "12.4.1", "12.1": "12.1.1", "11.8": "11.8.0"}
    candidates = [os.environ.get("CUDA_HOME", ""), "/usr/local/cuda-" + str(cuda_version), "/usr/local/cuda"]
    cached_root = os.path.join(resource_root(), "toolchains", "cuda-" + str(cuda_version))
    candidates.append(cached_root)
    selected = None
    for directory in candidates:
        compiler = os.path.join(directory, "bin", "nvcc")
        if not os.path.isfile(compiler):
            continue
        version = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=False)
        if version.returncode == 0 and "release " + str(cuda_version) + "," in version.stdout:
            selected = directory
            break
    if selected is None:
        if cuda_version not in releases:
            return {"status": "blocked", "reason": "No verified CUDA component manifest / 没有已核验 CUDA 组件清单", "cuda": cuda_version}
        component_names = ["cuda_nvcc", "cuda_cudart", "cuda_cccl"]
        if check_only:
            return {"status": "needs_toolchain", "cuda": cuda_version, "components": component_names,
                    "destination": cached_root, "system_cuda_unchanged": True}
        from wapr.det2d import _download_file
        archive_root = os.path.join(resource_root(), "toolchains", "archives")
        os.makedirs(archive_root, exist_ok=True)
        base_url = "https://developer.download.nvidia.com/compute/cuda/redist/"
        manifest_path = os.path.join(archive_root, "redistrib_" + releases[cuda_version] + ".json")
        _download_file(base_url + os.path.basename(manifest_path), manifest_path)
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        os.makedirs(cached_root, exist_ok=True)
        for name in component_names:
            component = manifest[name]["linux-x86_64"]
            relative_path = component["relative_path"]
            if relative_path.startswith("/") or ".." in relative_path.split("/"):
                raise ValueError("Unsafe CUDA manifest path / CUDA 清单路径不安全")
            archive_path = os.path.join(archive_root, os.path.basename(relative_path))
            _download_file(base_url + relative_path, archive_path)
            digest = hashlib.sha256()
            with open(archive_path, "rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            if os.path.getsize(archive_path) != int(component["size"]) or digest.hexdigest() != component["sha256"]:
                os.remove(archive_path)
                raise RuntimeError("CUDA component integrity failed / CUDA 组件完整性校验失败: " + name)
            with tempfile.TemporaryDirectory(prefix="cuda-components-", dir=archive_root) as staging:
                # The data filter rejects absolute paths, traversal and external links.
                # data 过滤器拒绝绝对路径、目录穿越及指向外部的链接。
                with tarfile.open(archive_path, "r:xz") as archive:
                    archive.extractall(staging, filter="data")
                roots = os.listdir(staging)
                if len(roots) != 1 or not os.path.isdir(os.path.join(staging, roots[0])):
                    raise RuntimeError("Unexpected CUDA component layout / CUDA 组件目录结构异常")
                shutil.copytree(os.path.join(staging, roots[0]), cached_root, dirs_exist_ok=True, symlinks=True)
        lib64 = os.path.join(cached_root, "lib64")
        if not os.path.lexists(lib64) and os.path.isdir(os.path.join(cached_root, "lib")):
            os.symlink("lib", lib64)
        compiler = os.path.join(cached_root, "bin", "nvcc")
        version = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=False)
        if version.returncode != 0 or "release " + str(cuda_version) + "," not in version.stdout:
            raise RuntimeError("Downloaded CUDA compiler failed its probe / 下载的 CUDA 编译器探针失败")
        selected = cached_root
    if os.path.realpath(selected) == os.path.realpath(cached_root):
        # Torch's exact NVIDIA runtime wheels also provide the matching ATen headers.
        # Torch 的精确 NVIDIA 运行时 wheel 也提供 ATen 所需的配套头文件。
        # Link only into our private compiler cache; never alter a system toolkit.
        # 只在私有编译器缓存建立链接；不修改系统工具链或独立环境中的原文件。
        nvidia_root = os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "nvidia")
        if not os.path.isdir(nvidia_root):
            return {"status": "blocked", "reason": "Matched Torch NVIDIA runtime headers are absent / 缺少配套 Torch NVIDIA 运行时头文件",
                    "cuda": cuda_version, "cuda_home": selected}
        if check_only and not os.path.isfile(os.path.join(selected, "include", "cusparse.h")):
            return {"status": "needs_toolchain", "cuda": cuda_version, "cuda_home": selected,
                    "components": ["existing Torch NVIDIA headers and libraries / 已有 Torch NVIDIA 头文件与库链接"]}
        linked_components = []
        if not check_only:
            for component_name in sorted(os.listdir(nvidia_root)):
                component_root = os.path.join(nvidia_root, component_name)
                header_root = os.path.join(component_root, "include")
                if os.path.isdir(header_root):
                    for directory, _, filenames in os.walk(header_root):
                        relative = os.path.relpath(directory, header_root)
                        destination_directory = os.path.join(selected, "include", relative)
                        os.makedirs(destination_directory, exist_ok=True)
                        for filename in filenames:
                            destination = os.path.join(destination_directory, filename)
                            if not os.path.lexists(destination):
                                os.symlink(os.path.join(directory, filename), destination)
                    linked_components.append(component_name)
                library_root = os.path.join(component_root, "lib")
                if not os.path.isdir(library_root):
                    continue
                library_directory = os.path.join(selected, "lib64")
                os.makedirs(library_directory, exist_ok=True)
                for filename in sorted(os.listdir(library_root)):
                    if ".so" not in filename:
                        continue
                    origin = os.path.join(library_root, filename)
                    if not os.path.isfile(origin):
                        continue
                    for link_name in (filename, filename.split(".so", 1)[0] + ".so"):
                        destination = os.path.join(library_directory, link_name)
                        if not os.path.lexists(destination):
                            os.symlink(origin, destination)
            print("WAPR_SAM_CUDA_RUNTIME_HEADERS", linked_components, flush=True)
    os.environ["CUDA_HOME"] = selected
    os.environ["PATH"] = os.path.join(selected, "bin") + os.pathsep + os.environ.get("PATH", "")
    print("WAPR_SAM_CUDA_TOOLCHAIN", selected, "cuda=" + str(cuda_version), flush=True)
    return {"status": "ready", "cuda": cuda_version, "cuda_home": selected, "system_cuda_unchanged": True}


def glibcxx_max(libstdcpp_path):
    """Return the newest GLIBCXX version symbol in one libstdc++.

    返回一份 libstdc++ 里最高的 GLIBCXX 版本符号。
    """
    # strings(1) is the same check used on the eight servers.
    # strings(1) 与八台服务器上的核查方式相同。
    try:
        completed = subprocess.run(["strings", libstdcpp_path], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return ""
    versions = []
    for line in completed.stdout.splitlines():
        if not line.startswith("GLIBCXX_3.4."):
            continue
        suffix = line[len("GLIBCXX_3.4."):]
        if suffix.isdigit():
            versions.append(int(suffix))
    if not versions:
        return ""
    return "GLIBCXX_3.4.%d" % max(versions)


def probe_sam3d_environment():
    """Run the isolated interpreter and return its Open3D probe.

    运行单独解释器并返回它的 Open3D 探针结果。位姿进程不加载 libOpen3D。
    """
    if not os.path.isfile(SAM3D_PYTHON):
        return {"status": "missing", "python": SAM3D_PYTHON, "reason": "SAM3D interpreter is absent / 缺少 SAM3D 解释器"}
    completed = subprocess.run(
        [SAM3D_PYTHON, os.path.abspath(__file__), "check"],
        capture_output=True, text=True, check=False,
    )
    if not completed.stdout.strip():
        return {
            "status": "failed",
            "python": SAM3D_PYTHON,
            "exit_code": completed.returncode,
            "stderr": completed.stderr[-500:],
        }
    try:
        report = json.loads(completed.stdout.splitlines()[-1])
    except (ValueError, IndexError):
        return {"status": "failed", "python": SAM3D_PYTHON,
                "exit_code": completed.returncode, "stderr": completed.stderr[-500:]}
    report["caller_python"] = sys.executable
    return report


def _check_worker():
    """Import official Open3D inside this interpreter and report the C++ runtime.

    在当前解释器里导入官方 Open3D，并报告 C++ 运行库。
    """
    import open3d
    # Report the runtime actually loaded by Open3D, including system libraries in a venv.
    # 报告 Open3D 实际载入的运行库，包括 venv 使用的系统库；不套用其他版本的最低符号要求。
    loaded = []
    try:
        with open("/proc/self/maps", encoding="utf-8") as stream:
            loaded = sorted({line.split()[-1] for line in stream if "libstdc++.so.6" in line})
    except OSError:
        pass
    libstdcpp = loaded[0] if loaded else ""
    newest = glibcxx_max(libstdcpp) if libstdcpp and os.path.isfile(libstdcpp) else ""
    status = "ready" if open3d.__version__ == OPEN3D_VERSION else "open3d_version_mismatch"
    report = {
        "status": status,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "open3d": open3d.__version__,
        "expected_open3d": OPEN3D_VERSION,
        "glibcxx_max": newest,
        "libstdcpp": libstdcpp,
        "loaded_libstdcpp": loaded,
    }
    # Open3D alone is not the full inference environment; report missing stacks.
    # 只有 Open3D 不代表完整推理环境；明确报告缺少的组件。
    missing = []
    for name in ("torch", "pytorch3d", "spconv", "sam3d_objects"):
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    report["missing_inference_modules"] = missing
    report["inference_verified"] = False
    if status == "ready" and missing:
        report["status"] = "dependencies_missing"
    print(json.dumps(report), flush=True)
    if report["status"] != "ready" or open3d.__version__ != OPEN3D_VERSION:
        return 1
    return 0


def reconstruct_example(rgb, mask, case_dir, device, size_path, object_name):
    """Run the unchanged example recipe in the selected SAM3D interpreter.

    在指定 SAM3D 解释器执行原示例配方；RGB/mask 与完整网格通过私有文件传输。
    NumPy arrays use no pickle; geometry is not simplified or rescaled in transit.
    NumPy 数组不使用 pickle；传输时不简化或缩放几何。
    """
    if size_path not in ("outline", "unipose"):
        raise ValueError(size_path)
    request = {"case_dir": os.path.abspath(case_dir), "device": device,
               "size_path": size_path, "object_name": object_name}
    return _transfer_reconstruction(rgb, mask, request)


def reconstruct(rgb, mask, checkpoint_directory=None, allow_replacement=None):
    """Run the package's vertex-color recipe in the separate interpreter.

    在独立解释器执行包内顶点颜色配方；权限与依赖确认仍由包内准备入口处理。
    """
    request = {"size_path": "package", "checkpoint_directory": os.fspath(checkpoint_directory) if checkpoint_directory is not None else None,
               "allow_replacement": allow_replacement}
    return _transfer_reconstruction(rgb, mask, request)


def _transfer_reconstruction(rgb, mask, request):
    """Transfer validated arrays through a private subprocess boundary.

    通过私有子进程边界传输已校验数组；request 是内部 JSON 协议，不是实验配置框架。
    """
    import numpy as np
    import trimesh
    from PIL import Image
    if os.path.realpath(SAM3D_PYTHON) == os.path.realpath(sys.executable):
        raise RuntimeError("SAM3D interpreter must differ from the caller / SAM3D 解释器必须与调用者不同")
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError("Expected uint8 RGB and matching mask / 需要 uint8 RGB 与同尺寸 mask")
    # First use prepares only the independent prefix; replacements retain their approval gate.
    # 首次调用只准备独立前缀；已有依赖替换仍保留确认流程。
    prepared = prepare_reconstruction_environment(allow_replacement=request.get("allow_replacement"),
                                                   checkpoint_directory=request.get("checkpoint_directory"))
    if prepared["status"] not in ("ready", "installed"):
        raise RuntimeError("SAM3D isolated preparation did not complete / SAM3D 独立准备未完成: " + json.dumps(prepared, ensure_ascii=False))
    with tempfile.TemporaryDirectory(prefix="wapr-sam3d-inference-") as directory:
        np.savez(os.path.join(directory, "input.npz"), rgb=rgb, mask=mask)
        with open(os.path.join(directory, "request.json"), "w", encoding="utf-8") as stream:
            json.dump(request, stream)
        environment = os.environ.copy()
        # The child imports this installed wheel rather than another WAPR version.
        # 子进程导入当前已安装 wheel，避免误用另一份 WAPR。
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
        environment = _worker_cache_environment(environment)
        completed = subprocess.run([SAM3D_PYTHON, os.path.abspath(__file__), "infer-example", directory],
                                   env=environment, check=False)
        if completed.returncode != 0:
            raise RuntimeError("SAM3D isolated inference failed; worker return code / SAM3D 独立推理失败，子进程返回码: "
                               + str(completed.returncode)
                               + "; inspect the worker diagnostics / 请查看子进程诊断")
        with np.load(os.path.join(directory, "mesh.npz"), allow_pickle=False) as result:
            mesh = trimesh.Trimesh(vertices=result["vertices"], faces=result["faces"], process=False)
            if "uv" in result:
                with Image.open(os.path.join(directory, "texture.png")) as image:
                    texture = image.copy()
                mesh.visual = trimesh.visual.TextureVisuals(uv=result["uv"], image=texture)
            else:
                mesh.visual.vertex_colors = result["colors"]
        if not len(mesh.vertices) or not np.isfinite(mesh.vertices).all():
            raise RuntimeError("Invalid worker mesh / 子进程返回无效网格")
        return mesh


def _example_worker(directory):
    """Execute only the SAM3D stage and serialize its geometry losslessly.

    仅执行 SAM3D 阶段，无损保存几何；WAPR 位姿阶段留在原解释器。
    """
    import numpy as np
    # Bind the installed package before importing the example's local modules.
    # 导入示例本地模块之前固定当前已安装包，避免源码目录覆盖 wheel。
    import wapr.resources
    with open(os.path.join(directory, "request.json"), encoding="utf-8") as stream:
        request = json.load(stream)
    with np.load(os.path.join(directory, "input.npz"), allow_pickle=False) as inputs:
        rgb, mask = inputs["rgb"], inputs["mask"]
    toolchain = _prepare_cuda_toolchain()
    if toolchain["status"] != "ready":
        raise RuntimeError("SAM inference CUDA toolchain / SAM 推理 CUDA 工具链: " + json.dumps(toolchain))
    if request["size_path"] == "package":
        from wapr.reconstruction_setup import reconstruct
        mesh = reconstruct(rgb, mask, checkpoint_directory=request["checkpoint_directory"],
                           allow_replacement=request["allow_replacement"])
    elif request["size_path"] in ("outline", "unipose"):
        # Preparation must bind all upstream source paths in this worker process.
        # 准备阶段须在当前子进程绑定全部上游源码路径，再导入烘焙工具。
        from wapr.reconstruction_setup import prepare_reconstruction
        prepared = prepare_reconstruction(allow_replacement=request.get("allow_replacement"),
                                          checkpoint_directory=request.get("checkpoint_directory"))
        if prepared["status"] not in ("ready", "installed"):
            raise RuntimeError("SAM example preparation / SAM 示例准备: " + json.dumps(prepared))
        sys.path.insert(0, request["case_dir"])
        from step01_point_mask import reconstruct_mesh
    if request["size_path"] == "outline":
        mesh = reconstruct_mesh(rgb, mask)
    elif request["size_path"] == "unipose":
        # Baking requests Gaussian views and a CUDA rasterizer; plain meshes do not.
        # 烘焙才需要高斯视图和 CUDA 光栅器；普通顶点色网格不安装这些组件。
        from wapr.installation import install_requirements
        baking_dependencies = install_requirements(["gsplat"], allow_replacement=request.get("allow_replacement"))
        if baking_dependencies["status"] not in ("ready", "installed"):
            raise RuntimeError("SAM baking dependencies / SAM 烘焙依赖: " + json.dumps(baking_dependencies))
        from wapr.source_setup import prepare_raster_source
        prepare_raster_source(allow_replacement=request.get("allow_replacement"))
        from step02_bake_mesh import apply_bake_budget, texture_config
        from fast_sam3d.session import Sam3dSession
        apply_bake_budget()
        session = Sam3dSession(device=request["device"], compile_model=False).load()
        mesh = session.reconstruct(rgb, mask, cfg=texture_config(request["object_name"]),
                                   allow_fallback=False)["mesh"]
    elif request["size_path"] != "package":
        raise ValueError(request["size_path"])
    arrays = {"vertices": np.asarray(mesh.vertices), "faces": np.asarray(mesh.faces)}
    uv, image = None, None
    if mesh.visual.kind == "texture":
        uv = mesh.visual.uv
        image = getattr(mesh.visual.material, "image", None)
        if image is None:
            image = getattr(mesh.visual.material, "baseColorTexture", None)
        if uv is None or image is None:
            raise RuntimeError("Incomplete worker texture / 子进程网格贴图不完整")
    if uv is not None:
        arrays["uv"] = np.asarray(uv)
        image.save(os.path.join(directory, "texture.png"))
    else:
        arrays["colors"] = np.asarray(mesh.visual.vertex_colors)
    np.savez(os.path.join(directory, "mesh.npz"), **arrays)
    print("WAPR_SAM3D_ISOLATED_COMPLETE", len(mesh.vertices), len(mesh.faces), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "prepare-reconstruction-worker":
        sys.exit(_reconstruction_prepare_worker(sys.argv[2]))
    if len(sys.argv) == 3 and sys.argv[1] == "prepare-worker":
        sys.exit(_prepare_worker(sys.argv[2]))
    if len(sys.argv) == 2 and sys.argv[1] == "check":
        sys.exit(_check_worker())
    if len(sys.argv) == 3 and sys.argv[1] == "infer-example":
        sys.exit(_example_worker(sys.argv[2]))
    sys.exit("usage: python sam3d_isolated.py check | infer-example PRIVATE_DIRECTORY")
