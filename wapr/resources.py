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
# Detect source files without changing the common resource layout.
# 识别源码目录，但不改变共用资源布局。
source_checkout = (os.path.isdir(os.path.join(release_dir, "examples"))
                   and os.path.isdir(os.path.join(release_dir, "wheel_build")))


def resource_root():
    """Return the common writable resource root for source and wheel.

    返回源码与 wheel 共用的可写资源根目录。
    """
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


def cache_dir():
    """Return shared reusable resources. / 返回可复用资源目录。"""
    return os.path.join(resource_root(), "cache")


def outputs_dir(example=None):
    """Keep each numbered recipe's outputs separate. / 各编号示例的输出独立存放。"""
    root = os.path.join(resource_root(), "outputs")
    if example is None:
        return root
    name = os.path.splitext(os.path.basename(os.fspath(example)))[0]
    if not name or name in (".", ".."):
        raise ValueError("Invalid example output name / 示例输出名称无效")
    return os.path.join(root, name)


def weights_dir():
    """Return the directory for checkpoints and locally built engines.

    返回权重与本机生成引擎所在目录。
    """
    return os.path.join(cache_dir(), "weights")


def samples_dir():
    """Return the directory for downloaded lesson samples.

    返回下载的教程小样所在目录。
    """
    return os.path.join(cache_dir(), "samples")
