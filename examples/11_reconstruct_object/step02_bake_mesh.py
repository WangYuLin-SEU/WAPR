# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Stage 02, called by examples/11_reconstruct_object.py; the entry bakes on its own.
# 第 02 段，由 examples/11_reconstruct_object.py 调用；入口块单独烘焙。
# Stage 02 in the combined example; its standalone main writes a separate bake study.
# 综合示例的第 02 阶段；单独运行时写出独立的烘焙实验。
# Full SAM3D mesh and the UV bake. python examples/11_reconstruct_object/step02_bake_mesh.py
# 完整 SAM3D 网格和 UV 烘焙。运行：python examples/11_reconstruct_object/step02_bake_mesh.py
"""Original SAM3D mesh plus the official Gaussian texture bake.

原来的 SAM3D 网格，再加官方高斯贴图烘焙。

The bake is the official Gaussian bake: 100 views at 1024 and 2500
optimizer steps. All three objects keep the decoder mesh. Cutting it to
about 8000 vertices smears the print, and a 180 degree in-plane spin then
looks like the upright face. Edge lengths are not chosen here.
烘焙用官方高斯烘焙：100 个 1024 视角，优化 2500 步。三个物体都保留解码器网格。
减到大约 8000 顶点会把印刷糊掉，面内转 180 度之后就和正放的那一面看起来一样。
三边长度不在这里决定。
"""

import json
import os
import sys
import time

import cv2
import numpy as np


DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.dirname(os.path.dirname(DEMO_DIR))
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from fast_sam3d.session import Sam3dSession, preset_config  # noqa: E402
from step01_point_mask import SAM3D_ROOT, read_rgb_depth  # noqa: E402

if SAM3D_ROOT not in sys.path:
    sys.path.insert(0, SAM3D_ROOT)


from wapr.resources import samples_dir
DATA_ROOT = os.environ.get("RECON_DATA_ROOT", os.path.join(samples_dir(), "ycbineoat"))
DATA_ROOT = os.path.abspath(os.path.join(RELEASE_DIR, DATA_ROOT))
from wapr.resources import outputs_dir
PAGE = os.path.join(outputs_dir("11_reconstruct_object"), "stages", "step01_point_mask")
OUT_DIR = os.path.join(outputs_dir("11_reconstruct_object"), "stages", "short_texture")

# Official Gaussian bake. Views, render size, optimizer steps, and the saved texture.
# 官方高斯烘焙。视角数、渲染边长、优化步数，以及保存的贴图边长。
# Baking latency needs a separate RTX 5090 measurement for these settings.
# 此组烘焙设置的耗时需在 RTX 5090 上单独测量。
BAKE_VIEWS = 100
BAKE_RESOLUTION = 1024
BAKE_STEPS = 2500
BAKE_TEXTURE = 1024
BAKE_MODE = "opt"
# Original SAM3D for every object. The 8000-vertex cut is not used.
# 三个物体都用原来的 SAM3D。不再减到 8000 顶点。
PRESET = {
    "cracker": "sam3d",
    "sugar": "sam3d",
    "mustard": "sam3d",
}


def first_rgb(sequence):
    """
    # Return the path of the first PNG in one YCBInEOAT sequence.

    ## Args

        - sequence: the sequence folder name under DATA_ROOT. It is not None.

    ## Returns

        - The return is one filesystem path string.
        - It is not None.
        - Names: sorted, and the first name that ends with .png is used.

    ---

    # 返回一条 YCBInEOAT 序列里第一张 PNG 的路径。

    ## 参数

        - sequence: DATA_ROOT 下的序列文件夹名。不是 None。

    ## 返回

        - 返回值是一个文件系统路径字符串。
        - 不是 None。
        - 文件名排序后，取第一个以 .png 结尾的名字。

"""
    rgb_dir = os.path.join(DATA_ROOT, sequence, "rgb")
    name = sorted(name for name in os.listdir(rgb_dir) if name.endswith(".png"))[0]
    return os.path.join(rgb_dir, name)


def mask_from_prompt(rgb, prompt_path):
    """
    # Return pixels that differ from the saved prompt overlay.

    ## Args

        - rgb: (H, W, 3) uint8 RGB. It is not None.
        - prompt_path: the path of the prompt image. It is not None. The image is read as BGR and compared with rgb converted to BGR.

    ## Returns

        - The return is (H, W) uint8, 0 or 1.
        - A channel sum above 8 is 1.
        - It is not None.

    ---

    # 返回和保存的提示叠图不一样的像素。

    ## 参数

        - rgb: (H, W, 3) uint8 RGB。不是 None。
        - prompt_path: 提示图路径。不是 None。图按 BGR 读入，再和转成 BGR 的 rgb 比较。

    ## 返回

        - 返回值是 (H, W) uint8，取值 0 或 1。
        - 通道差之和大于 8 的为 1。
        - 不是 None。

"""
    prompt = cv2.imread(prompt_path)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    diff = np.abs(bgr.astype(np.int16) - prompt.astype(np.int16)).sum(axis=2)
    return (diff > 8).astype(np.uint8)


def cases():
    """
    # Return the three showcase frames.

    ## Args

        - There are no arguments.

    ## Returns

        - The return is a list of three dicts.
        - It is not None.
        - Each dict has name, rgb, and prompt, all strings.
        - No field is None.

    ---

    # 返回宣传页上的三帧。

    ## 参数

        - 没有参数。

    ## 返回

        - 返回值是三个字典的列表。
        - 不是 None。
        - 每个字典含 name、rgb、prompt，都是字符串。
        - 没有字段是 None。

"""
    return [
        {
            "name": "cracker",
            "rgb": first_rgb("cracker_box_reorient"),
            "prompt": os.path.join(PAGE, "prompt.png"),
        },
        {
            "name": "sugar",
            "rgb": os.path.join(DATA_ROOT, "sugar_box1", "rgb", "1579910587381007354.png"),
            "prompt": os.path.join(PAGE, "sugar_prompt.png"),
        },
        {
            "name": "mustard",
            "rgb": first_rgb("mustard0"),
            "prompt": os.path.join(PAGE, "mustard_prompt.png"),
        },
    ]


def texture_config(name):
    """
    # Return the SAM3D preset for this object with the UV bake turned on.

    ## Args

        - name: "cracker", "sugar", or "mustard". It is not None. It selects PRESET.

    ## Returns

        - The return is a config dict.
        - It is not None.
        - with_texture_baking: true, texture_size is BAKE_TEXTURE, and auto_fallback is false.

    ---

    # 返回这个物体的 SAM3D preset，并打开 UV 烘焙。

    ## 参数

        - name: "cracker"、"sugar" 或 "mustard"。不是 None。它用来选取 PRESET。

    ## 返回

        - 返回值是配置字典。
        - 不是 None。
        - with_texture_baking: 真，texture_size 为 BAKE_TEXTURE，auto_fallback 为假。

"""
    cfg = preset_config(PRESET[name])
    cfg["with_texture_baking"] = True
    cfg["texture_size"] = BAKE_TEXTURE
    cfg["auto_fallback"] = False
    return cfg


def apply_bake_budget():
    """
    # Point the vendored baker at this script's view count, size, and steps.

    ## Args

        - There are no arguments.

    ## Returns

        - Returns None.
        - The return is None.
        - TEXTURE_BAKE_VIEWS, TEXTURE_BAKE_RESOLUTION, TEXTURE_BAKE_STEPS, and TEXTURE_BAKE_MODE on the vendored module are replaced.

    ---

    # 把第三方烘焙指到这个脚本的视角数、边长和步数。

    ## 参数

        - 没有参数。

    ## 返回

        - 返回 None。
        - 返回值是 None。
        - 第三方模块上的 TEXTURE_BAKE_VIEWS、TEXTURE_BAKE_RESOLUTION、TEXTURE_BAKE_STEPS 和 TEXTURE_BAKE_MODE 被替换。

"""
    # The package init imports a module this checkout does not ship.
    # 这个包的初始化会导入当前这份代码里没有的模块。
    os.environ["LIDRA_SKIP_INIT"] = "true"
    from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils

    postprocessing_utils.TEXTURE_BAKE_VIEWS = BAKE_VIEWS
    postprocessing_utils.TEXTURE_BAKE_RESOLUTION = BAKE_RESOLUTION
    postprocessing_utils.TEXTURE_BAKE_STEPS = BAKE_STEPS
    postprocessing_utils.TEXTURE_BAKE_MODE = BAKE_MODE








if __name__ == "__main__":
    # Run the script stages directly in the entry block.
    # 在入口块中直接执行脚本各阶段。
    os.makedirs(OUT_DIR, exist_ok=True)
    apply_bake_budget()
    session = Sam3dSession(device="cuda:0").load()
    chosen = os.environ.get("TEXTURE_ONLY", "")
    report = []
    for case in cases():
        if chosen and case["name"] != chosen:
            continue
        rgb, _depth = read_rgb_depth(case["rgb"])
        mask = mask_from_prompt(rgb, case["prompt"])
        started = time.perf_counter()
        output = session.reconstruct(rgb, mask, cfg=texture_config(case["name"]), allow_fallback=False)
        wall_s = time.perf_counter() - started
        mesh = output["mesh"]
        stem = os.path.join(OUT_DIR, case["name"])
        mesh.export(stem + ".glb")
        texture = output.get("texture")
        if texture is not None:
            texture.convert("RGB").save(stem + "_texture.jpg", quality=90)
        profile = output.get("profile", {})
        meta = output.get("metadata", {})
        row = {
            "name": case["name"],
            "wall_s": round(wall_s, 2),
            "vertices": int(meta.get("vertices", 0)),
            "faces": int(meta.get("faces", 0)),
            "texture_size": meta.get("texture_size"),
            "mask_px": int(mask.sum()),
            "profile": {key: round(float(value), 3) for key, value in profile.items()},
        }
        report.append(row)
        print(
            case["name"], "wall", row["wall_s"],
            "bake", row["profile"].get("T_texture_baking"),
            "verts", row["vertices"],
            flush=True,
        )
    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("wrote", OUT_DIR, flush=True)
