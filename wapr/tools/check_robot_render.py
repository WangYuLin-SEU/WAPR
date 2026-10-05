# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Render one ManiSkill frame to verify the robot graphics environment.

只渲染一帧 ManiSkill 图像，检查机器人案例所需的图形环境。
Run from the release root after installing ManiSkill: python wapr/tools/check_robot_render.py
安装 ManiSkill 后从发布根目录运行：python wapr/tools/check_robot_render.py
"""

import gymnasium as gym
import mani_skill.envs  # noqa: F401  Register ManiSkill environments / 注册 ManiSkill 环境。


def main():
    """Check the Panda RGB render without running a manipulation episode.

    检查 Panda RGB 渲染；不执行抓取或放置回合。
    """
    env = gym.make(
        "PickCube-v1",
        robot_uids="panda",
        obs_mode="rgbd",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
    )
    try:
        env.reset(seed=0)
        frame = env.render()
        if frame is None or frame.ndim != 4 or frame.shape[-1] != 3:
            raise RuntimeError("Expected batched RGB frames / 预期得到带批次维的 RGB 图像")
        print("ROBOT_RENDER READY", "shape / 形状", tuple(frame.shape), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
