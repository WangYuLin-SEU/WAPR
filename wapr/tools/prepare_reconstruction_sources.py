# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Fetch the tested SAM 3D source revision when absent.

缺少源码时检出已测试的 SAM 3D 版本，并应用本地兼容补丁；SAM2 使用 Ultralytics。
Run from the release root: python wapr/tools/prepare_reconstruction_sources.py
从发布根目录运行：python wapr/tools/prepare_reconstruction_sources.py
"""

import os
import shutil
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Source-only tools live under wapr/tools, two levels below the release root.
# 仅供源码使用的工具位于 wapr/tools，发布根目录在其上两级。
RELEASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)
from wapr.resources import cache_dir

THIRD_PARTY_DIR = os.path.join(cache_dir(), "sources")
SAM3D_REVISION = "f91db411c50efee93d8db7aeb323885650f6f722"


def main():
    """Prepare source checkouts without changing an existing different revision.

    准备源码检出；已有目录若不是指定版本则报告问题，不覆盖作者的改动。
    """
    if shutil.which("git") is None:
        raise RuntimeError("git is required / 需要 git")
    os.makedirs(THIRD_PARTY_DIR, exist_ok=True)

    sources = (
        ("sam-3d-objects", "https://github.com/facebookresearch/sam-3d-objects.git", SAM3D_REVISION),
    )
    for name, url, revision in sources:
        path = os.path.join(THIRD_PARTY_DIR, name)
        if not os.path.exists(path):
            print("RECON_CLONE", name, url, flush=True)
            subprocess.check_call(["git", "clone", url, path])
            subprocess.check_call(["git", "-C", path, "checkout", revision])
        if not os.path.isdir(path):
            raise RuntimeError("Source path is not a directory / 源码路径不是目录: " + path)
        try:
            current_revision = subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"], text=True,
            ).strip()
        except subprocess.CalledProcessError as error:
            raise RuntimeError("Source is not a git checkout / 源码目录不是 git 检出: " + path) from error
        if current_revision != revision:
            raise RuntimeError(
                "Existing source uses a different revision / 已有源码版本不同: "
                + name + " " + current_revision + " (expected / 预期 " + revision + ")"
            )
        print("RECON_SOURCE_REVISION", name, current_revision, flush=True)

    sam3d_root = os.path.join(THIRD_PARTY_DIR, "sam-3d-objects")
    patch_path = os.path.join(SCRIPT_DIR, "patches", "sam3d_wapr_compat.patch")
    if not os.path.isfile(patch_path):
        raise FileNotFoundError(patch_path)
    reverse = subprocess.run(
        ["git", "-C", sam3d_root, "apply", "--reverse", "--check", patch_path],
        capture_output=True, text=True, check=False,
    )
    if reverse.returncode == 0:
        print("RECON_SAM3D_PATCH", "already applied / 已应用", flush=True)
        return
    forward = subprocess.run(
        ["git", "-C", sam3d_root, "apply", "--check", patch_path],
        capture_output=True, text=True, check=False,
    )
    if forward.returncode != 0:
        raise RuntimeError(
            "SAM 3D patch does not match this checkout / SAM 3D 补丁与源码不匹配: "
            + forward.stderr.strip()
        )
    subprocess.check_call(["git", "-C", sam3d_root, "apply", patch_path])
    print("RECON_SAM3D_PATCH", "applied / 已应用", flush=True)


if __name__ == "__main__":
    main()
