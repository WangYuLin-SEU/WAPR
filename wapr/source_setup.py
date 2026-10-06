# Author: Yulin Wang (yulinwang@seu.edu.cn)
# Copyright (c) 2026 Yulin Wang. See LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
"""Fetch optional upstream source without replacing user checkouts.

获取可选上游源码，不覆盖用户已有检出；不下载权重、不安装依赖。
"""

import os
import hashlib
import http.client
import shutil
import subprocess
import tempfile
import tarfile
import urllib.request
import urllib.parse

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
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _github_archive(url, revision, candidate):
    """Fetch the same official revision when Git transport fails.

    Git 传输失败时获取同一官方版本的源码归档；拒绝越界路径及链接成员。
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
                    shutil.copyfileobj(response, stream)
                break
            except (OSError, http.client.HTTPException):
                if direct:
                    raise RuntimeError("Official archive fetch failed / 官方归档获取失败") from None
        expected_root = parts[1] + "-" + revision
        os.makedirs(candidate, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                names = member.name.rstrip("/").split("/")
                if names[0] != expected_root or any(name in ("..", ".git", "") for name in names[1:]):
                    raise RuntimeError("Unsafe archive member / 归档成员路径不安全")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise RuntimeError("Unsupported archive link/member / 不支持归档链接或特殊成员")
                destination = os.path.join(candidate, *names[1:])
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                with archive.extractfile(member) as source, open(destination, "wb") as output:
                    shutil.copyfileobj(source, output)
                os.chmod(destination, 0o755 if member.mode & 0o111 else 0o644)
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
                capture_output=True, text=True, check=False,
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
                    env=environment, check=True, capture_output=True, text=True, timeout=60,
                )
                subprocess.run(
                    git + ["-C", candidate, "remote", "add", "origin", url],
                    env=environment, check=True, capture_output=True, text=True, timeout=60,
                )
                subprocess.run(
                    git + ["-C", candidate, "fetch", "--depth", "1", "origin", revision],
                    env=environment, check=True, capture_output=True, text=True, timeout=180,
                )
                subprocess.run(
                    ["git", "-C", candidate, "checkout", "--detach", revision],
                    env=environment, check=True, capture_output=True, text=True, timeout=60,
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
        current = revision if archive_fetched else subprocess.check_output(["git", "-C", candidate, "rev-parse", "HEAD"], text=True).strip()
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
            subprocess.run(["git", "-C", candidate, "apply", "--check", patch], env=patch_environment, check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", candidate, "apply", patch], env=patch_environment, check=True, capture_output=True, text=True)
        # Rename only an absent destination; concurrent/existing source stays intact.
        # 仅移动到缺失位置；保留并发生成或已有源码。
        try:
            os.rename(candidate, path)
        except OSError:
            if not os.path.isdir(path):
                raise
            print("WAPR_SOURCE_CONCURRENT", name, "Existing directory retained / 保留已有目录", flush=True)
            return os.path.abspath(path)
        print("WAPR_SOURCE_PIN", name, revision, "Runtime compatibility unverified / 运行兼容性未验证", flush=True)
        return os.path.abspath(path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def prepare_source(feature):
    """Return the source root for one optional feature without installing packages.

    返回单项可选功能源码根目录，不安装包；det2d 同时准备三项源码。
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
        return _checkout("RoMa", "https://github.com/Parskatt/RoMa.git", ROMA_REVISION)
    if feature == "unipose9d":
        return _checkout("UniPose9D", "https://github.com/qq456cvb/UniPose9D.git", UNIPOSE9D_REVISION)
    raise ValueError("Unknown source feature / 未知源码功能: " + str(feature))


def prepare_unipose_weights():
    """Fetch the pinned public checkpoint separately from the source checkout.

    独立下载固定公开权重并核验 Git LFS SHA256，不覆盖使用者源码中的权重。
    """
    from wapr.resources import weights_dir
    source = prepare_source("unipose9d")
    directory = os.path.join(weights_dir(), "unipose9d")
    os.makedirs(directory, exist_ok=True)
    destination = os.path.join(directory, "last.ckpt")
    expected_sha256 = "02c91a3df7d37740e552c421c9bd697189155c7e3936fdb1e7798a5e16dd3e73"
    expected_size = 371248494
    if not os.path.isfile(destination):
        url = "https://media.githubusercontent.com/media/qq456cvb/UniPose9D/" + UNIPOSE9D_REVISION + "/checkpoints/last.ckpt"
        from wapr.det2d import _download_file
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


def prepare_qwen_weights():
    """Download the optional public language model only when requested.

    仅在请求时下载可选公开语言模型；权重不进入 wheel，不使用维护者凭据。
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, LocalEntryNotFoundError, HfHubHTTPError
    from wapr.download_assets import hub_official, hub_mirror
    from wapr.resources import weights_dir
    directory = os.path.join(weights_dir(), "Qwen2.5-VL-3B-Instruct")
    endpoints = []
    for endpoint in (os.environ.get("HF_ENDPOINT", ""), hub_official, hub_mirror):
        endpoint = endpoint.strip().rstrip("/")
        if endpoint and endpoint not in endpoints:
            endpoints.append(endpoint)
    for endpoint in endpoints:
        try:
            snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", local_dir=directory, token=False,
                              endpoint=endpoint, etag_timeout=30,
                              allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.txt", "tokenizer.model"],
                              max_workers=2)
            return directory
        except LocalEntryNotFoundError:
            print("WAPR_QWEN_RETRY", "Network/cache miss / 网络或缓存未命中", flush=True)
        except (GatedRepoError, RepositoryNotFoundError):
            raise
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
