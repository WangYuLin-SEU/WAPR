# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Public entry: from wapr import WAPREstimator.
# 对外入口：from wapr import WAPREstimator。
import sys

# Also reject unsupported source checkouts before dependency preparation.
# 源码调用同样在准备依赖前拒绝不支持的 Python 版本。
if sys.version_info < (3, 10):
    raise RuntimeError("WAPR requires Python 3.10 or newer / WAPR 需要 Python 3.10 或更高版本")

__all__ = ["WAPREstimator"]


def __getattr__(name):
    """Load inference after dependency preparation. / 依赖准备后再载入推理模块。"""
    if name == "WAPREstimator":
        from wapr.estimator import WAPREstimator
        return WAPREstimator
    raise AttributeError(name)
