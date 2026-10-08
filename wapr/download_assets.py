# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Fetch the four checkpoints and small dataset excerpts from SEU-WYL/WAPR.
# 从 SEU-WYL/WAPR 获取四份权重和各数据集的小样本摘录。
# python -m wapr.download_assets downloads missing packs to the shared resource cache.
# python -m wapr.download_assets 将缺少的数据包下载到共用资源缓存。
# A run that needs one pack calls check_and_fetch_pack(name).
# 某次运行仅需一份时，调用 check_and_fetch_pack(name)。
# Full BOP test sets are not in this repo and are not downloaded here.
# 完整 BOP 测试集不在这个仓库里，此处亦不下载。
import os
import hashlib
import json
import shutil
import socket
import sys
import tempfile
from pathlib import Path
from urllib.request import getproxies, proxy_bypass as is_http_proxy_bypassed

# Use resumable HTTP for public weights; acceleration proxies often do not
# support the Xet transfer service. Respect an explicit user setting.
# 公开权重使用可续传 HTTP；加速代理常不支持 Xet 传输服务。保留用户显式设置。
if os.path.isfile("/etc/network_turbo"):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
# The public mirror and signed CDN transfers must bypass an academic proxy
# that only forwards GitHub/Hugging Face front-end hosts.
# 公开镜像与签名 CDN 下载绕开只支持 GitHub/Hugging Face 前端域名的学术代理。
if os.path.isfile("/etc/network_turbo") and getproxies().get("https"):
    proxy_bypass = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
    proxy_bypass_hosts = [host.strip() for host in proxy_bypass.split(",") if host.strip()]
    for proxy_bypass_host in ("hf-mirror.com", "cas-bridge.xethub.hf.co", "cas-server.xethub.hf.co", "transfer.xethub.hf.co"):
        if proxy_bypass_host not in proxy_bypass_hosts:
            proxy_bypass_hosts.append(proxy_bypass_host)
    os.environ["NO_PROXY"] = ",".join(proxy_bypass_hosts)
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

from wapr.resources import resource_root, cache_dir, samples_dir, weights_dir

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
RESOURCE_ROOT = Path(resource_root())
WEIGHTS_DIR = Path(weights_dir())
SAMPLES_DIR = Path(samples_dir())
# This file lives in wapr/, so the release root is its parent.
# 本文件在 wapr/ 下，release 根目录是它的上一级。
sys.path.insert(0, str(ROOT))

# Model repository; assets/hf/ can supply an optional local resource bundle.
# 模型仓库；assets/hf/ 可提供本地资源包。
repo_id = "SEU-WYL/WAPR"
# Official hub. Mainland China often cannot open this host.
# 官方站点。中国大陆常无法访问该地址。
hub_official = "https://huggingface.co"
# Used automatically when hub_official does not connect. This public mirror can change.
# 无法连接官方站点时自动改用。该公开镜像的地址以后可能变更。
hub_mirror = "https://hf-mirror.com"
# How long to wait, in seconds, before treating one hub host as unreachable.
# 将某一站点判定为无法连接之前的等待时间，单位为秒。
hub_connect_timeout_s = 5.0
repo_type = "model"
weight_names = (
    "wapr_w_mask.pth",
    "wapr_wo_mask.pth",
    "sapr.pth",
    "wbps.pth",
)
# One frame each. pose_lmo is that LM-O frame in meters for examples/02_one_category_one_instance.py.
# 每个数据集一帧。pose_lmo 是同一帧 LM-O，单位已换成米，给 examples/02_one_category_one_instance.py。
sample_names = ("pose_lmo", "lmo", "ycbv", "tless", "tudl")
# These files are all needed by the first pose lesson; meta.json alone does not
# prove that its RGB, depth, mesh, and mask were placed successfully.
# 第一节位姿课会读取全部这些文件；只有 meta.json 不足以证明 RGB、深度、网格和 mask 已放齐。
pose_lmo_required_files = ("meta.json", "rgb.png", "depth.npy", "model.ply", "mask.png")
# The LM-O lessons read both frames, all eight CADs, and the published 2D excerpt.
# LM-O 教程要读两帧、八份 CAD，以及已公布的 2D 检测摘录。
lmo_required_files = (
    "PROVENANCE.txt",
    "cnos-fastsam_scene2_im307.json",
    "models/models_info.json",
    "models/obj_000001.ply",
    "models/obj_000005.ply",
    "models/obj_000006.ply",
    "models/obj_000008.ply",
    "models/obj_000009.ply",
    "models/obj_000010.ply",
    "models/obj_000011.ply",
    "models/obj_000012.ply",
    "test/000002/scene_camera.json",
    "test/000002/rgb/000001.png",
    "test/000002/rgb/000307.png",
    "test/000002/depth/000001.png",
    "test/000002/depth/000307.png",
)
# Check the files a one-frame BOP lesson actually opens, not only its provenance marker.
# 检查单帧 BOP 教程实际读取的文件，而不只检查溯源标记。
bop_required_files = {
    "ycbv": (
        "PROVENANCE.txt", "models/models_info.json", "models/obj_000005.ply",
        "test/000050/scene_camera.json", "test/000050/scene_gt.json",
        "test/000050/rgb/001130.png", "test/000050/depth/001130.png",
    ),
    "tless": (
        "PROVENANCE.txt", "models/models_info.json", "models/obj_000030.ply",
        "test_primesense/000001/scene_camera.json", "test_primesense/000001/scene_gt.json",
        "test_primesense/000001/rgb/000001.png", "test_primesense/000001/depth/000001.png",
    ),
    "tudl": (
        "PROVENANCE.txt", "models/models_info.json", "models/obj_000001.ply",
        "test/000001/scene_camera.json", "test/000001/scene_gt.json",
        "test/000001/rgb/000000.png", "test/000001/depth/000000.png",
    ),
}
# wapr_sapr_wbps is the four pose checkpoints, written under weights_dir():
# wapr_w_mask.pth, wapr_wo_mask.pth, sapr.pth, wbps.pth.
# wapr_sapr_wbps 是四份位姿权重，写到 weights_dir() 返回的目录：
# wapr_w_mask.pth、wapr_wo_mask.pth、sapr.pth、wbps.pth。
pack_names = ("wapr_sapr_wbps",) + sample_names
# Empty downloads every pack above. A tuple downloads those names only.
# 空元组下载上面的每一个示例包。若写入名称，则仅下载所列数据包。
only = ()


def _fetch_example_sample(name):
    """Fetch only requested tutorial files, verifying their pinned SHA256.

    仅获取明确请求的教程文件，并核验固定 SHA256；不下载完整数据集。
    """
    catalog_path = Path(__file__).with_name("example_assets.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if name not in catalog:
        raise KeyError(name)
    spec = catalog[name]
    destination_root = SAMPLES_DIR / spec["folder"]
    from huggingface_hub import hf_hub_download
    from wapr.download_route import hub_endpoints
    endpoints = None
    for relative, identity in spec["files"].items():
        destination = destination_root / relative
        if destination.is_file() and destination.stat().st_size == identity["bytes"]:
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest == identity["sha256"]:
                continue
        local = ASSETS_DIR / "hf" / "samples" / spec["folder"] / relative
        source = None
        if local.is_file() and local.stat().st_size == identity["bytes"] and hashlib.sha256(local.read_bytes()).hexdigest() == identity["sha256"]:
            source = local
        else:
            if endpoints is None:
                endpoints = hub_endpoints(hub_official, hub_mirror)
            errors = []
            for endpoint in endpoints:
                try:
                    downloaded = Path(hf_hub_download(repo_id, "samples/" + spec["folder"] + "/" + relative,
                                                     endpoint=endpoint, token=False))
                    if downloaded.stat().st_size != identity["bytes"] or hashlib.sha256(downloaded.read_bytes()).hexdigest() != identity["sha256"]:
                        raise RuntimeError("Tutorial file checksum mismatch / 教程文件校验失败")
                    source = downloaded
                    break
                except Exception as error:
                    errors.append(type(error).__name__)
            if source is None:
                raise RuntimeError("Tutorial sample unavailable / 教程小样暂不可获取: %s; %s; original source / 原始来源: %s" % (name, errors, spec["source"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_name(destination.name + ".pending")
        shutil.copy2(source, pending)
        os.replace(pending, destination)
    print("WAPR_SAMPLE_READY", name, str(destination_root), flush=True)
    return str(destination_root)


def _place_weight_license():
    """Keep the first-party weight terms beside downloaded weights.

    将第一方权重条款放在下载后的权重旁边；不改动权重文件或第三方许可。
    """
    notice_dir = Path(__file__).resolve().parent / "weight_license"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ("WEIGHTS_LICENSE.txt", "CC-BY-ND-4.0.txt"):
        destination = WEIGHTS_DIR / filename
        if not destination.is_file():
            shutil.copy2(notice_dir / filename, destination)


def _marker(name):
    """
    # Marker path in the shared resource cache; pack_ready checks required files too.

    ## Args

        - name: wapr_sapr_wbps, pose_lmo, or a small BOP pack: lmo, ycbv, tless, tudl. Markers live under cache/weights/ and cache/samples/.

    ## Returns

        - Returns a Path.

    ---

    # 共用资源缓存中的标记路径；pack_ready 还检查实际所需文件。

    ## 参数

        - name: wapr_sapr_wbps、pose_lmo，或 BOP 小样本包：lmo、ycbv、tless、tudl。标记位于 cache/weights/ 与 cache/samples/。

    ## 返回

        - 返回 Path。

"""
    if name == "wapr_sapr_wbps":
        return WEIGHTS_DIR / weight_names[0]
    if name == "pose_lmo":
        return SAMPLES_DIR / "pose_lmo" / "meta.json"
    return SAMPLES_DIR / "bop" / name / "PROVENANCE.txt"


def pack_ready(name):
    """
    # True when the placed files for this pack are on disk.

    ## Args

        - name: _marker. Every pack checks the files needed by its lesson.

    ---

    # 该示例包的目标文件已在磁盘上时返回 True。

    ## 参数

        - name 与 _marker 相同。
        - wapr_sapr_wbps 在 cache/weights/ 下四份位姿权重均存在时视为就绪。
        - pose_lmo 需要第一节课使用的五个文件。
        - lmo 需要两帧、CAD 及已公布的 2D 检测摘录。
        - 其他 BOP 小样也检查相机、RGB、深度和物体网格文件。

"""
    if name == "wapr_sapr_wbps":
        return all((WEIGHTS_DIR / item).is_file() for item in weight_names)
    if name == "pose_lmo":
        sample_dir = SAMPLES_DIR / "pose_lmo"
        return all((sample_dir / item).is_file() for item in pose_lmo_required_files)
    if name == "lmo":
        sample_dir = SAMPLES_DIR / "bop" / "lmo"
        return all((sample_dir / item).is_file() for item in lmo_required_files)
    if name in bop_required_files:
        sample_dir = SAMPLES_DIR / "bop" / name
        return all((sample_dir / item).is_file() for item in bop_required_files[name])
    return _marker(name).is_file()


def _local_source(name):
    """
    # The same bytes under assets/hf/ when a local resource bundle is available.

    ## Args

        - name: pack_ready. Return a local source only when its required files are present.

    ## Returns

        - Returns paths, a directory, or None.

    ---

    # 如果本机已有 assets/hf/ 本地资源包，则返回其中对应的文件。

    ## 参数

        - name 与 pack_ready 相同。
        - wapr_sapr_wbps 返回四份位姿权重。
        - pose_lmo 五个课件文件齐全时返回源目录；其他 BOP 小样本的来源以溯源标记判断。
        - BOP 小样需要各自的相机、RGB、深度和物体网格文件齐全。

    ## 返回

        - 若不存在则返回 None。

"""
    if name == "wapr_sapr_wbps":
        paths = [ASSETS_DIR / "hf" / item for item in weight_names]
        if all(path.is_file() for path in paths):
            return paths
        return None
    if name == "pose_lmo":
        src = ASSETS_DIR / "hf" / "samples" / "pose_lmo"
        if all((src / item).is_file() for item in pose_lmo_required_files):
            return src
        return None
    if name == "lmo":
        src = ASSETS_DIR / "hf" / "samples" / "bop" / "lmo"
        if all((src / item).is_file() for item in lmo_required_files):
            return src
        return None
    src = ASSETS_DIR / "hf" / "samples" / "bop" / name
    required = bop_required_files.get(name)
    if required is not None and all((src / item).is_file() for item in required):
        return src
    return None


def _place_local(name, source):
    """
    # Copy one local pack into cache/weights/ or cache/samples/. Returns None.

        name selects the destination.

        source is the list or directory from _local_source.

    ---

    # 将一份本地示例包复制到 cache/weights/ 或 cache/samples/。

        name 决定目标位置。

        source 是 _local_source 返回的列表或目录。

    ## 返回

        - 返回 None。

"""
    if name == "wapr_sapr_wbps":
        dest_dir = WEIGHTS_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in source:
            shutil.copy2(src, dest_dir / src.name)
        return
    if name == "pose_lmo":
        dest = SAMPLES_DIR / "pose_lmo"
    else:
        dest = SAMPLES_DIR / "bop" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, dirs_exist_ok=True)


def _patterns(names):
    """
    # Hugging Face allow-patterns for the packs still to fetch.

    ## Args

        - names: an iterable of pack names. wapr_sapr_wbps expands to wapr_w_mask.pth, wapr_wo_mask.pth, sapr.pth, and wbps.pth. pose_lmo uses samples/pose_lmo/**. A BOP sample uses samples/bop/<name>/**.

    ## Returns

        - Returns a list of strings.

    ---

    # 尚待获取的示例包在 Hugging Face 上的匹配模式。

    ## 参数

        - names: 示例包名字。wapr_sapr_wbps 展开成 wapr_w_mask.pth、wapr_wo_mask.pth、sapr.pth、wbps.pth。pose_lmo 用 samples/pose_lmo/**。BOP 小样本用 samples/bop/<name>/**。

    ## 返回

        - 返回字符串列表。

"""
    patterns = []
    for name in names:
        if name == "wapr_sapr_wbps":
            patterns.extend(weight_names)
        elif name == "pose_lmo":
            patterns.append("samples/pose_lmo/**")
        else:
            patterns.append("samples/bop/%s/**" % name)
    return patterns


def _place_download(staging, names):
    """
    # Copy fetched packs from the snapshot into cache/weights/ or cache/samples/. Returns None.

    ## Args

        - staging: the snapshot directory.
        - names: the packs that were fetched. A missing file or directory raises FileNotFoundError.

    ---

    # 将已下载的示例包从快照目录复制到 cache/weights/ 或 cache/samples/。

    ## 参数

        - staging: 快照目录。
        - names: 本次下载的示例包。缺文件或缺目录时抛出 FileNotFoundError。

    ## 返回

        - 返回 None。

"""
    staging = Path(staging)
    if "wapr_sapr_wbps" in names:
        dest_dir = WEIGHTS_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        for item in weight_names:
            src = staging / item
            if not src.is_file():
                raise FileNotFoundError("%s is missing in %s" % (item, repo_id))
            # Staging and destination share a filesystem, so a completed weight
            # can be placed without keeping another large copy.
            # 暂存区与目标目录位于同一文件系统，避免为大权重再保留一份副本。
            os.replace(src, dest_dir / item)
    for name in names:
        if name == "wapr_sapr_wbps":
            continue
        if name == "pose_lmo":
            src = staging / "samples" / "pose_lmo"
            dest = SAMPLES_DIR / "pose_lmo"
        else:
            src = staging / "samples" / "bop" / name
            dest = SAMPLES_DIR / "bop" / name
        if not src.is_dir():
            raise FileNotFoundError("samples pack %s is missing in %s" % (name, repo_id))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest, dirs_exist_ok=True)


def _fetch_snapshot(staging_dir, names):
    """
    # Fetch from the official hub, then the mirror.

        Try each address in order. HF_ENDPOINT, when set, is first. Then hub_official, then hub_mirror. A host that does not accept a connection within hub_connect_timeout_s is skipped. A download error that is not a missing repo is also skipped, and the next address is tried. A missing repo raises immediately.

        Returns None. The files are left in staging_dir.

    ---

    # 先从官方源获取，再换镜像。

        按所列顺序尝试每个地址。若已设置 HF_ENDPOINT，则优先尝试该地址，随后为 hub_official，再为 hub_mirror。在 hub_connect_timeout_s 秒内无法连接的地址予以跳过。下载出错但并非仓库不存在时，同样跳过并尝试下一地址。仓库不存在则立即报错。

        返回 None。文件保留在 staging_dir。
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        BadRequestError,
        DisabledRepoError,
        EntryNotFoundError,
        GatedRepoError,
        HFValidationError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    official = hub_official.rstrip("/")
    mirror = hub_mirror.rstrip("/")
    from wapr.download_route import hub_endpoints
    # User HF_ENDPOINT stays first. The other addresses are ordered by a measured prefix.
    # 用户设置的 HF_ENDPOINT 仍排在最前。其余地址按实测前缀排序。
    order = hub_endpoints(official, mirror)
    repo_missing = (
        RepositoryNotFoundError,
        GatedRepoError,
        DisabledRepoError,
        RevisionNotFoundError,
        EntryNotFoundError,
        BadRequestError,
        HFValidationError,
    )
    last_error = None
    staging = Path(staging_dir)
    for endpoint in order:
        host = endpoint.split("://", 1)[-1].split("/", 1)[0]
        # A configured HTTPS proxy can reach the hub even when direct TCP cannot.
        # 已配置 HTTPS 代理时，官方源可能经代理可达，不能以直连 TCP 失败跳过它。
        if getproxies().get("https") and not is_http_proxy_bypassed(host):
            connected = True
        else:
            try:
                with socket.create_connection((host, 443), timeout=hub_connect_timeout_s):
                    connected = True
            except OSError:
                connected = False
        if not connected:
            print("WAPR_HUB", {"endpoint": endpoint, "connect": False}, flush=True)
            continue
        attempt = staging / "_attempt"
        # Reuse validated Hub cache metadata and incomplete HTTP downloads.
        # 复用 Hub 校验元数据及未完成的 HTTP 下载，避免网络重试重新传全部权重。
        attempt.mkdir(exist_ok=True)
        print("WAPR_HUB", {"endpoint": endpoint, "connect": True}, flush=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                allow_patterns=_patterns(names),
                local_dir=str(attempt),
                endpoint=endpoint,
                etag_timeout=30.0,
                # Avoid optional Brotli decoder ABI mismatches in base images.
                # 避免基础镜像中可选 Brotli 解码器 ABI 不匹配；二进制下载无需 HTTP 压缩。
                headers={"Accept-Encoding": "identity"},
                # These public assets need no account token, including on a mirror.
                # 公开资源无需账号令牌；切换镜像时也不向第三方站点发送本机令牌。
                token=False,
            )
        except LocalEntryNotFoundError as exc:
            # A failed network lookup with no local cache is not a missing repo.
            # 网络查询失败且无本地缓存不表示仓库不存在，应继续尝试下一源。
            last_error = exc
            print("WAPR_HUB", {"endpoint": endpoint, "error": type(exc).__name__}, flush=True)
            continue
        except repo_missing as exc:
            # A mirror's missing/stale listing does not establish an upstream 404.
            # 镜像缺失或未同步不能证明官方仓库不存在，仍尝试后续站点。
            if endpoint == official:
                raise
            last_error = exc
            print("WAPR_HUB", {"endpoint": endpoint, "error": type(exc).__name__}, flush=True)
            continue
        except Exception as exc:
            last_error = exc
            print("WAPR_HUB", {"endpoint": endpoint, "error": type(exc).__name__}, flush=True)
            continue
        for child in attempt.iterdir():
            dest = staging / child.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(child), str(dest))
        shutil.rmtree(attempt, ignore_errors=True)
        return
    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not connect to %s" % ", ".join(order))


def download(names):
    """
    # Fetch the named asset packs.

        Use assets/hf/ when it is on disk. Otherwise download from SEU-WYL/WAPR.

        The download tries hub_official first. When that host does not connect, the same download continues from hub_mirror. The switch is automatic. An HF_ENDPOINT environment variable is tried first; if that address also fails to connect, the other address is still tried.

        Returns None.

    ## Args

        - names: an iterable of pack names. A name outside pack_names raises KeyError. Packs found under assets/hf/ are copied. The rest are downloaded from hub_official, then from hub_mirror if that host does not connect.

    ---

    # 获取这些资源包。

        若磁盘上已有 assets/hf/，则使用其中的文件。否则从 SEU-WYL/WAPR 下载。

        下载首先访问 hub_official。该地址无法连接时，同一次下载自动改用 hub_mirror，无需手动修改。若已设置环境变量 HF_ENDPOINT，则优先尝试该地址；该地址也无法连接时，仍会尝试其余地址。

        返回 None。

    ## 参数

        - names: 待放置的名称。不在 pack_names 中的名称抛出 KeyError。assets/hf/ 中已有的文件直接复制。其余文件先从 hub_official 下载，无法连接时再从 hub_mirror 下载。

"""
    names = tuple(names)
    pending = []
    for name in names:
        if name not in pack_names:
            raise KeyError(name)
        source = _local_source(name)
        if source is not None:
            _place_local(name, source)
            print("copied", name, flush=True)
        else:
            pending.append(name)
    if not pending:
        if "wapr_sapr_wbps" in names:
            _place_weight_license()
        return
    RESOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    # Stable download staging survives disconnects; files are placed only after
    # the Hub finishes the requested snapshot. Keep the existing resource root.
    # 固定下载暂存目录保留断线续传；Hub 完成请求后再放置文件，不改变现有资源根目录。
    staging_dir = Path(cache_dir()) / "downloads" / "-".join(sorted(pending))
    staging_dir.mkdir(parents=True, exist_ok=True)
    with (staging_dir / ".lock").open("a") as download_lock:
        if os.name == "posix":
            import fcntl
            fcntl.flock(download_lock.fileno(), fcntl.LOCK_EX)
        _fetch_snapshot(staging_dir, pending)
        _place_download(staging_dir, pending)
    for name in pending:
        print("downloaded", name, flush=True)
    if "wapr_sapr_wbps" in names:
        _place_weight_license()


def check_and_fetch_pack(name):
    """
    # Check, then fetch.

        Look at the destination first. When the check passes, model and sample files stay untouched and no download runs. For weights, missing license sidecars are copied from the installed package.

        When the check fails, two ways remain. If this machine already has the same files under assets/hf/, copy them into place and do not use the network. Otherwise download this one pack from the Hugging Face model repo SEU-WYL/WAPR, then copy it into place.

        That download tries https://huggingface.co first. Mainland China often cannot open that host. When the connection fails, the same download continues from https://hf-mirror.com. The switch is automatic. Set the environment variable HF_ENDPOINT to one address to try that address first. If it also fails to connect, the other address is still tried.

        A failed check fetches the whole pack again. Files from that pack that were already on disk are overwritten. A passed check never overwrites.

        The full BOP test set is not fetched. Detector weights are not fetched. Any other name raises KeyError.

    ## Args

        - name: which pack. The check and the files are different for each name.
        - wapr_sapr_wbps: the four pose checkpoints in cache/weights/: wapr_w_mask.pth, wapr_wo_mask.pth, sapr.pth, and wbps.pth. These are WAPR with a mask, WAPR without a mask, SAPR, and WBPS. One missing file fetches all four again.
        - pose_lmo: one LM-O frame in meters, for examples/02_one_category_one_instance.py. Required files live under cache/samples/pose_lmo/.
        - lmo, ycbv, tless, tudl: small BOP packs under cache/samples/bop/<name>/, checked with their required files. These are not full test sets.

    ## Returns

        - Returns None.

    ---

    # 先检查本地文件，缺失时再获取。

        先检查目标路径。检查通过时不复制模型或样本，也不下载；对于权重，仅从安装包补齐缺失的许可附带文件。

        检查未通过时有两种途径。若本机 assets/hf/ 中已有相同文件，则复制到目标位置，不访问网络。否则从 Hugging Face 模型仓库 SEU-WYL/WAPR 下载该数据包，再复制到目标位置。

        此次下载首先访问 https://huggingface.co。中国大陆常无法访问该地址。无法连接时，同一次下载自动改用 https://hf-mirror.com，无需手动切换。若将环境变量 HF_ENDPOINT 设为某一地址，则优先尝试该地址；该地址也无法连接时，仍会尝试另一地址。

        检查未通过时将重新获取整份数据。该数据包中已在磁盘上的文件将被覆盖。检查通过时不予覆盖。

        不下载完整的 BOP 测试集，也不下载检测器权重。其他名称抛出 KeyError。

    ## 参数

        - name: 所获取的数据包。每个名字检查的文件和写入的位置均不同。
        - wapr_sapr_wbps: cache/weights/ 下四份位姿权重 wapr_w_mask.pth、wapr_wo_mask.pth、sapr.pth、wbps.pth。它们是带 mask 的 WAPR、不带 mask 的 WAPR、SAPR、WBPS。任一文件缺失时，四份均重新获取。
        - pose_lmo: 一帧 LM-O，单位为米，供示例 02 使用；所需文件位于 cache/samples/pose_lmo/。
        - lmo、ycbv、tless、tudl: cache/samples/bop/<名称>/ 下的 BOP 小样，检查实际所需文件，不下载完整测试集。

    ## 返回

        - 返回 None。

"""
    # Optional sequences never join the default core download list.
    # 可选序列不加入默认核心下载清单，只在对应示例请求时获取。
    if name in ("taco", "robi"):
        _fetch_example_sample(name)
        return
    if pack_ready(name):
        if name == "wapr_sapr_wbps":
            _place_weight_license()
        return
    print("WAPR_RESOURCES", str(RESOURCE_ROOT), flush=True)
    download((name,))


def main():
    """
    # Download every small pack, or the names in only.

        Packs already on disk stay put.

        Returns None.

        only is the module tuple.

        An empty only means pack_names: wapr_sapr_wbps plus the small sample packs.

        Run it as python -m wapr.download_assets.

        The if __name__ block is the caller.

    ---

    # 下载每一个示例包，或 only 中列出的名称。

        已在磁盘上的示例包保持不变。

        返回 None。

        only 是模块里的元组。

        空元组表示 pack_names，即四份位姿权重和各数据集的小样本包。

        运行方式是 python -m wapr.download_assets。

        由 if __name__ 块调用。
    """
    names = pack_names if len(only) == 0 else only
    print("WAPR_RESOURCES", str(RESOURCE_ROOT), flush=True)
    missing = [name for name in names if not pack_ready(name)]
    if not missing:
        print("present", list(names), flush=True)
        return
    download(missing)


if __name__ == "__main__":
    main()
