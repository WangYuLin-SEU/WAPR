# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Mark the staged native renderer wheel with its Python and Linux ABI.

将暂存的本地编译的渲染器 wheel 标为对应的 Python 与 Linux ABI。
"""

from setuptools import Distribution, setup


class NativeWheelDistribution(Distribution):
    """Report the prebuilt renderer as native code to the wheel builder.

    告诉 wheel 构建器：包内预编译渲染器是本地编译代码。
    """

    def has_ext_modules(self):
        """Prevent the native wheel from receiving a pure-Python tag.

        避免包含本地编译模块的 wheel 被误标为跨平台纯 Python 包。
        """
        return True


setup(distclass=NativeWheelDistribution)
