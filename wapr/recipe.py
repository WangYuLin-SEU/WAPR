# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Inference choices. Edit these variables; they are not command-line flags.
# 推理选择。改这些变量即可，它们不是命令行参数。
# Paths are relative to the release root, the parent of this file.
# 路径相对 release 根目录，也就是本文件的上一级。
from pathlib import Path

from wapr.resources import source_checkout, weights_dir

_ROOT = Path(__file__).resolve().parents[1]

# Initial rotations: n_view camera directions × n_inplane spins about camera +Z.
# 初始旋转：n_view 个相机方向，每个再绕相机 +Z 转 n_inplane 次。
# 4/6/8/12/20 are tetrahedron, octahedron, cube, icosahedron, dodecahedron.
# 4/6/8/12/20 分别是正四面体、正八面体、立方体、正二十面体、正十二面体。
# Any other count samples the sphere with Fibonacci.
# 其他数量在球面上做斐波那契采样。
n_view = 4
n_inplane = 3
# WAPR: wide-angle pose refinement. wapr_w_mask uses the mask channel; wapr_wo_mask does not.
# WAPR：广角位姿修正。wapr_w_mask 带 mask 通道，wapr_wo_mask 不带。
# SAPR shares the WAPR net and runs after it. Its rotation cap is tighter.
# SAPR 与 WAPR 共用修正网络，排在 WAPR 之后，旋转上限更紧。
# WBPS rates each hypothesis inside its group and against the other groups.
# WBPS 给每个候选姿态打组内分，以及相对其他组的组间分。within_group 越大越好，between_group 越大越差。
# Update counts. One update is one rotation step and one translation step.
# 更新次数。一次更新包含一步旋转和一步平移。
wapr_iters = 3
sapr_iters = 2
# Per-step rotation cap, radians. The network rotation passes through tanh, then this scale.
# 每步旋转上限，弧度。网络旋转先过 tanh，再乘这个系数。
wapr_max_rot_rad = 0.70
sapr_max_rot_rad = 0.35
# Crop window as a multiple of the object diameter.
# 裁剪窗口是物体直径的倍数。
refine_crop_ratio = 1.2
wbps_crop_ratio = 1.1
# Square crop side, pixels.
# 正方形裁剪边长，像素。
crop_px = 160
# Lambert weights in the camera frame. Lit color is ambient plus diffuse times the clamped cosine.
# 相机系 Lambert 权重。着色是环境项加上漫反射乘截断后的余弦。
w_ambient = 0.8
w_diffuse = 0.2
# Light direction in the OpenCV camera frame, +Z forward.
# OpenCV 相机系的光照方向，+Z 朝前。
light_dir = (0.0, 0.0, 1.0)
# "torch" runs the modules. "trt" requires the FP16 engine assets/weights/<name>.engine.
# A missing engine raises. It does not fall back to the module.
# "torch" 走模块。"trt" 必须有 assets/weights/<name>.engine 这份 FP16 引擎。
# 缺引擎直接报错，不退回模块。
backend = "trt"
# False writes no 6D pose visualization. True writes one, from the examples that call visualize_6d_pose.
# False 不存 6D 位姿可视化。True 时，调用了 visualize_6d_pose 的示例会存一张。
visualize = False
# A directory, or a .jpg / .png file. Relative paths start at the release root.
# 目录，或者一个 .jpg / .png 文件。相对路径从 release 根目录算。
visualize_path = "outputs/vis"

# Input channels. wapr_w_mask is RGB, mask, xyz. The other three are RGB, xyz.
# 输入通道。wapr_w_mask 是 RGB、mask、xyz。其余三个是 RGB、xyz。
channels = {
    "wapr_w_mask": 7,
    "wapr_wo_mask": 6,
    "sapr": 6,
    "wbps": 6,
}


def weight_file(name):
    """
    # Resolve one weight path from the source table or the installed user cache.

    ## Args

        - name: wapr_w_mask, wapr_wo_mask, sapr, or wbps. A name that is not in the table raises KeyError. The file itself is not opened here.

    ## Returns

        - Returns a Path under the release root or installed user cache.

    ---

    # 从源码路径表或安装后的用户缓存解析一份权重路径。

    ## 参数

        - name: wapr_w_mask、wapr_wo_mask、sapr 或 wbps。表里没有这个名字时抛出 KeyError。这里不打开文件。

    ## 返回

        - 返回源码发布目录或安装后用户缓存下的 Path。

"""
    table = {}
    if source_checkout:
        # Keep the author's project-relative checkpoint mapping in a source checkout.
        # 源码目录继续使用作者维护的项目相对权重路径表。
        path = _ROOT / "assets" / "weights" / "PATHS.txt"
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, rel = line.split()
            table[key] = _ROOT / rel
    else:
        # A wheel excludes PATHS.txt; its four checkpoint names are fixed by the release.
        # wheel 不包含 PATHS.txt；四份权重文件名由发布包确定。
        for key in channels:
            table[key] = Path(weights_dir()) / (key + ".pth")
    if name not in table:
        raise KeyError(name)
    return table[name]


def engine_file(name):
    """
    # Same path as the weight, with the suffix replaced by .engine.

    ## Args

        - name: the same key as weight_file. The engine file does not have to exist yet.

    ## Returns

        - Returns a Path.

    ---

    # 与权重相同的路径，后缀换成 .engine。

    ## 参数

        - name 与 weight_file 用同一个键。
        - 引擎文件此时可以还不存在。

    ## 返回

        - 返回 Path。

"""
    return weight_file(name).with_suffix(".engine")


def resolve_visualize_path(filename):
    """
    # Resolve the 6D visualization path.

        visualize_2d_detection calls it when image is True.

        visualize_6d_pose calls it when recipe.visualize is true.

    ## Args

        - filename: the file name, such as pose.jpg. It is used when visualize_path is a directory. A visualize_path that already ends in .jpg, .jpeg, or .png is the file itself. A relative path starts at the source release or installed package's user cache.

    ## Returns

        - Returns an absolute Path.

    ---

    # 解析 6D 可视化的保存路径。

        image 为 True 时 visualize_2d_detection 调用它。

        recipe.visualize 为真时 visualize_6d_pose 调用它。

    ## 参数

        - filename: 文件名，例如 pose.jpg。当 visualize_path 是目录时使用。若 visualize_path 以 .jpg、.jpeg 或 .png 结尾，则将其作为完整输出文件路径。相对路径以源码 release 根目录或已安装包的用户缓存为基准。

    ## 返回

        - 返回绝对 Path。
    """
    raw = Path(visualize_path)
    if raw.suffix.lower() in (".jpg", ".jpeg", ".png"):
        path = raw
    else:
        path = raw / filename
    if not path.is_absolute():
        # Installed wheels write user-cache outputs, never read-only site-packages.
        # 已安装 wheel 的输出写入用户缓存，不写可能只读的 site-packages。
        from wapr.resources import resource_root
        path = Path(resource_root()) / path
    return path
