# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。

"""Choose one measured download host and keep the same file.

按实测吞吐选择一个下载主机，并保持文件版本不变。
A region hint only orders the candidates. A probe decides which host is used.
地区信息只排列候选。真正使用哪一台由测速决定。
"""

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request


# Same safetensors wheel on every host. The total size is part of the identity check.
# 各主机上的同一份 safetensors wheel。总大小参与同一性检查。
PYPI_PROBE_RELATIVE = (
    "/28/50/f203ff3a3ddfe19308efc83c5a3a29ed02bf786732ec35e68bf9162f3365/"
    "safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
)
PYPI_PROBE_BYTES = 516040
# Public DINOv2 config is tiny, so the hub probe reads the weight prefix instead.
# 公开 DINOv2 配置太小，站点测速读取权重文件的开头。
HUB_PROBE_PATH = "/facebook/dinov2-small/resolve/main/pytorch_model.bin"
PROBE_BYTES = 262144
PROBE_TIMEOUT_S = 12
ROUTE_CACHE_TTL_S = 6 * 3600
# Mainland mirrors are tried first only as an ordering hint on AutoDL.
# 大陆镜像只在 AutoDL 上作为排序提示排在前面。
PYPI_ROUTES_MIRROR_FIRST = (
    ("tuna", "https://pypi.tuna.tsinghua.edu.cn/simple", "https://pypi.tuna.tsinghua.edu.cn/packages"),
    ("aliyun", "https://mirrors.aliyun.com/pypi/simple", "https://mirrors.aliyun.com/pypi/packages"),
    ("official", "https://pypi.org/simple", "https://files.pythonhosted.org/packages"),
)
PYPI_ROUTES_OFFICIAL_FIRST = (
    ("official", "https://pypi.org/simple", "https://files.pythonhosted.org/packages"),
    ("tuna", "https://pypi.tuna.tsinghua.edu.cn/simple", "https://pypi.tuna.tsinghua.edu.cn/packages"),
    ("aliyun", "https://mirrors.aliyun.com/pypi/simple", "https://mirrors.aliyun.com/pypi/packages"),
)


def route_cache_path():
    """Return the speed-test cache outside the source tree.

    返回源码目录之外的测速缓存路径。
    """
    return os.path.join(os.path.expanduser("~"), ".cache", "wapr", "download-route.json")


def public_url(url):
    """Drop user, query, and fragment so logs cannot keep a signed link.

    去掉用户信息、查询和片段，避免日志留下签名链接。
    """
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", ""))


def _candidate_order():
    """Order PyPI candidates. AutoDL prefers mirrors before the probe.

    排列 PyPI 候选。AutoDL 在测速前把镜像放在前面。
    """
    if os.path.isfile("/etc/network_turbo"):
        return PYPI_ROUTES_MIRROR_FIRST
    return PYPI_ROUTES_OFFICIAL_FIRST


def probe_prefix(url, expected_total=None, opener=None, limit=PROBE_BYTES):
    """Read a short prefix and reject HTML, a bad range, or a size mismatch.

    读取一小段开头，并拒绝 HTML、错误区间或总大小不一致。
    """
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "WAPR-download-route",
            "Range": "bytes=0-%d" % (limit - 1),
            "Accept-Encoding": "identity",
        },
    )
    started = time.perf_counter()
    try:
        response = (opener or urllib.request.urlopen)(request, timeout=PROBE_TIMEOUT_S)
    except (OSError, urllib.error.URLError):
        return None
    with response:
        status = getattr(response, "status", 200)
        if status not in (200, 206):
            return None
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "html" in content_type:
            return None
        content_range = response.headers.get("Content-Range")
        if status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", (content_range or "").strip())
            if match is None:
                return None
            start, end, total_text = match.groups()
            if int(start) != 0 or int(end) < int(start):
                return None
            total = None if total_text == "*" else int(total_text)
            declared_length = int(end) - int(start) + 1
        else:
            length = response.headers.get("Content-Length")
            total = int(length) if length is not None else None
        if expected_total is not None and total != expected_total:
            return None
        body = response.read(limit + 1)
        if body.lstrip().lower().startswith(b"<html") or body.lstrip().startswith(b"<!doctype"):
            return None
        if len(body) > limit:
            return None
        if status == 206 and len(body) != declared_length:
            return None
        elapsed = time.perf_counter() - started
        return {"bytes": len(body), "seconds": elapsed, "total": total, "url": public_url(url)}


def _read_cache():
    path = route_cache_path()
    try:
        with open(path, encoding="utf-8") as stream:
            cached = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cached, dict):
        return None
    if time.time() > float(cached.get("expires", 0)):
        return None
    return cached


def _write_cache(record):
    path = route_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(record, stream)
    os.replace(temporary, path)


def choose_pypi_route(force=False):
    """Return one PyPI index and its matching file host.

    返回一个 PyPI 索引及与之对应的文件主机。
    PIP_INDEX_URL is used as given and is not probed.
    已设置 PIP_INDEX_URL 时直接使用，不再测速。
    """
    user_index = os.environ.get("PIP_INDEX_URL", "").strip()
    if user_index:
        print("WAPR_DOWNLOAD_ROUTE", {"kind": "pypi", "reason": "user", "index": public_url(user_index)}, flush=True)
        return {"name": "user", "index": user_index, "files": None, "reason": "user"}
    cached = None if force else _read_cache()
    if cached and cached.get("pypi"):
        return cached["pypi"]
    measured = []
    for name, index, files in _candidate_order():
        sample = probe_prefix(files + PYPI_PROBE_RELATIVE, PYPI_PROBE_BYTES)
        if sample is None or sample["bytes"] < 1:
            print("WAPR_DOWNLOAD_ROUTE", {"kind": "pypi", "name": name, "ok": False}, flush=True)
            continue
        sample.update(name=name, index=index, files=files)
        measured.append(sample)
        print("WAPR_DOWNLOAD_ROUTE", {
            "kind": "pypi", "name": name, "ok": True,
            "seconds": round(sample["seconds"], 3), "bytes": sample["bytes"],
        }, flush=True)
    if not measured:
        chosen = {"name": "official", "index": "https://pypi.org/simple",
                  "files": "https://files.pythonhosted.org/packages", "reason": "probe_failed"}
    else:
        best = min(measured, key=lambda item: item["seconds"])
        chosen = {"name": best["name"], "index": best["index"], "files": best["files"],
                  "seconds": round(best["seconds"], 3), "reason": "measured"}
    record = _read_cache() or {}
    record["pypi"] = chosen
    record["expires"] = time.time() + ROUTE_CACHE_TTL_S
    _write_cache(record)
    return chosen


def same_pypi_file_url(url):
    """Point a Python package file at the measured host when the size matches.

    在总大小一致时，把 PyPI 文件地址改到测得更快的主机。
    The sha256 fragment stays. A user-selected index is not rewritten.
    sha256 片段保留。用户指定的索引不会被改写。
    """
    route = choose_pypi_route()
    if route.get("files") is None or route.get("reason") == "user":
        return url
    split = urllib.parse.urlsplit(url)
    marker = "/packages/"
    if marker not in split.path:
        return url
    relative = split.path[split.path.index(marker) + len("/packages"):]
    official_files = "https://files.pythonhosted.org/packages"
    if split.netloc != "files.pythonhosted.org":
        return url
    # A short range is enough to confirm this exact file exists at the same size.
    # 短区间足以确认这个文件存在且总大小相同。
    official = probe_prefix(official_files + relative, limit=1024)
    if official is None or official["total"] is None:
        return url
    target = route["files"] + relative
    target_parts = urllib.parse.urlsplit(target)
    if target_parts.netloc == split.netloc:
        return url
    checked = probe_prefix(target, official["total"], limit=1024)
    if checked is None:
        print("WAPR_DOWNLOAD_ROUTE", {"kind": "file", "ok": False, "url": public_url(target)}, flush=True)
        return url
    print("WAPR_DOWNLOAD_ROUTE", {"kind": "file", "ok": True, "url": public_url(target)}, flush=True)
    return urllib.parse.urlunsplit(("https", target_parts.netloc, target_parts.path, split.query, split.fragment))


def hub_endpoints(official, mirror):
    """Return hub addresses, user endpoint first, then the faster measured host.

    返回站点地址：用户指定的在前，其后是测得更快的主机。
    """
    chosen = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    cached = _read_cache() or {}
    hub = cached.get("hub")
    if not isinstance(hub, dict) or time.time() > float(cached.get("expires", 0)):
        samples = []
        for endpoint in (official, mirror):
            sample = probe_prefix(endpoint.rstrip("/") + HUB_PROBE_PATH)
            if sample is None:
                print("WAPR_DOWNLOAD_ROUTE", {"kind": "hub", "endpoint": public_url(endpoint), "ok": False}, flush=True)
                continue
            sample["endpoint"] = endpoint.rstrip("/")
            samples.append(sample)
            print("WAPR_DOWNLOAD_ROUTE", {
                "kind": "hub", "endpoint": public_url(endpoint), "ok": True,
                "seconds": round(sample["seconds"], 3), "bytes": sample["bytes"], "total": sample["total"],
            }, flush=True)
        totals = {item["total"] for item in samples if item["total"] is not None}
        if len(totals) > 1:
            samples = [item for item in samples if item["endpoint"] == official.rstrip("/")]
        if samples:
            fastest = min(samples, key=lambda item: item["seconds"])["endpoint"]
            rest = [item["endpoint"] for item in samples if item["endpoint"] != fastest]
            hub = {"order": [fastest] + rest, "reason": "measured"}
        else:
            hub = {"order": [official.rstrip("/"), mirror.rstrip("/")], "reason": "probe_failed"}
        record = cached if isinstance(cached, dict) else {}
        record["hub"] = hub
        record["expires"] = time.time() + ROUTE_CACHE_TTL_S
        if record.get("pypi"):
            pass
        _write_cache(record)
    order = []
    if chosen:
        order.append(chosen)
    for endpoint in hub.get("order", []):
        if endpoint not in order:
            order.append(endpoint)
    for endpoint in (official.rstrip("/"), mirror.rstrip("/")):
        if endpoint not in order:
            order.append(endpoint)
    return order
