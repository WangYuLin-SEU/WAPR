# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Locate WAPR weights and lesson samples in a checkout or installed wheel.

在源码目录或已安装 wheel 中定位 WAPR 权重与教程小样。
"""

import os


release_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PATHS.txt belongs to the source release and is excluded from the wheel.
# PATHS.txt 属于源码发布目录，不会打进 wheel；据此保留源码原有布局。
source_checkout = os.path.isfile(os.path.join(release_dir, "assets", "weights", "PATHS.txt"))


def resource_root():
    """Return the source release or the installed package's user cache.

    返回源码发布根目录，或已安装包使用的用户缓存目录。
    """
    if source_checkout:
        return release_dir

    override = os.environ.get("WAPR_CACHE_DIR", "").strip()
    if override:
        override = os.path.expanduser(override)
        if not os.path.isabs(override):
            raise ValueError("WAPR_CACHE_DIR must be an absolute path / 必须是绝对路径")
        return override

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache_home and os.path.isabs(xdg_cache_home):
        return os.path.join(xdg_cache_home, "wapr")
    return os.path.join(os.path.expanduser("~"), ".cache", "wapr")


def weights_dir():
    """Return the directory for checkpoints and locally built engines.

    返回权重与本机生成引擎所在目录。
    """
    if source_checkout:
        return os.path.join(resource_root(), "assets", "weights")
    return os.path.join(resource_root(), "weights")


def samples_dir():
    """Return the directory for downloaded lesson samples.

    返回下载的教程小样所在目录。
    """
    return os.path.join(resource_root(), "samples")
