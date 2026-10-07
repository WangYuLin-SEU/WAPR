# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Language prompt stage used by example 11. / 示例 11 使用的语言提示阶段。
# A text prompt selects the object; the observed silhouette refines its metric scale.
# 文本提示指定目标，观测轮廓进一步校正其米制尺寸。
# Entry: python examples/11_reconstruct_object.py
# 入口：python examples/11_reconstruct_object.py
"""Qwen box selection, SAM 2 segmentation, and visualization helpers.

千问目标框、SAM 2 分割与可视化辅助函数。
Example 11 owns the inputs and the reconstruction flow.
输入与重建流程统一由示例 11 管理。
"""

import json
import os
import re
import sys

import numpy as np


EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_DIR = os.path.dirname(EXAMPLES_DIR)
# Mask, mesh, and scale helpers are the reconstruction case.
# mask、网格和尺度函数在重建案例里。
CASE_DIR = os.path.join(EXAMPLES_DIR, "11_reconstruct_object")
if CASE_DIR not in sys.path:
    sys.path.insert(0, CASE_DIR)
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)

from step01_point_mask import (  # noqa: E402
    SAM2_CHECKPOINT,
    WEIGHTS_DIR,
    SAM2_CONFIG,
    SAM2_ROOT,
    project_silhouette,
)


# Local Qwen2.5-VL-3B snapshot. Children of this root are the weight files.
# 本地千问 2.5-VL-3B。权重文件都在这个根下面。
from wapr.resources import weights_dir
QWEN_ROOT = os.environ.get(
    "QWEN_VL_ROOT",
    os.path.join(weights_dir(), "Qwen2.5-VL-3B-Instruct"),
)
QWEN_ROOT = os.path.abspath(os.path.join(RELEASE_DIR, QWEN_ROOT))
# The bundled, unmodified font includes Chinese glyphs; its license is beside it.
# 随项目提供的原版字体包含中文字形；许可在字体旁。
FONT_PATH = os.path.join(RELEASE_DIR, "wapr", "fonts", "wqy-microhei.ttc")
DEVICE = "cuda:0"


def qwen_snapshot(root):
    """
    # Return a complete local Qwen snapshot, including every verified weight shard.

    ## Args

        - root: a filesystem path. Use its complete snapshot, or the first complete snapshots child.

    ## Returns

        - The return is one path string.
        - It is not None.
        - A missing snapshot raises RuntimeError.

    ---

    # 返回完整本地千问快照目录，包括已核验的全部权重分片。

    ## 参数

        - root: 文件系统路径。使用 root 的完整快照，或 snapshots 下第一个完整子快照。

    ## 返回

        - 返回值是一个路径字符串。
        - 不是 None。
        - 找不到快照时抛出 RuntimeError。

"""
    from wapr.source_setup import _qwen_weights_ready
    if _qwen_weights_ready(root):
        return root
    snap_root = os.path.join(root, "snapshots")
    names = sorted(os.listdir(snap_root)) if os.path.isdir(snap_root) else []
    for name in names:
        path = os.path.join(snap_root, name)
        if _qwen_weights_ready(path):
            return path
    raise RuntimeError("qwen snapshot")


def parse_box(text, width, height, resized_wh):
    """
    # Parse Qwen's reply into a pixel box, x right and y down.

    ## Args

        - text: the reply string. It is not None. A bbox_2d list is preferred. Otherwise the first four numbers are used. Fewer than four numbers raises RuntimeError.
        - width: the image width in pixels. It is not None.
        - height: the image height in pixels. It is not None.
        - resized_wh: (resized_w, resized_h) in pixels, the size Qwen actually saw. It is not None.

    ## Returns

        - The return is (4,) float32, (x1, y1, x2, y2) in pixels on the original image.
        - It is not None.
        - A box thinner than 8 pixels on either side raises RuntimeError.

    ---

    # 把千问的回答解析成像素框，x 向右，y 向下。

    ## 参数

        - text: 回答字符串。不是 None。优先取 bbox_2d 列表。否则用前四个数。少于四个数时抛出 RuntimeError。
        - width: 图像宽，单位像素。不是 None。
        - height: 图像高，单位像素。不是 None。
        - resized_wh: 像素 (resized_w, resized_h)，千问实际看到的尺寸。不是 None。

    ## 返回

        - 返回值是 (4,) float32，(x1, y1, x2, y2)，原图像素。
        - 不是 None。
        - 任一边短于 8 像素时抛出 RuntimeError。

"""
    match = re.search(
        r"bbox_2d\"?\s*[:=]\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]",
        text,
    )
    if match is None:
        numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", text)
        if len(numbers) < 4:
            raise RuntimeError("qwen box")
        values = [float(number) for number in numbers[:4]]
    else:
        values = [float(match.group(index)) for index in range(1, 5)]
    x1, y1, x2, y2 = values
    peak = max(x1, y1, x2, y2)
    resized_w, resized_h = resized_wh
    if peak <= 1.5:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif peak <= 1000.0 and (x2 > width or y2 > height or (resized_w == width and peak > width)):
        # 0–1000 is the other coordinate Qwen prints when it is not using pixels.
        # 0–1000 是千问不用像素时的另一种坐标。
        if x2 > width or y2 > height:
            x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
            y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
    elif resized_w > 0 and (x2 > width + 2 or y2 > height + 2):
        x1, x2 = x1 * width / float(resized_w), x2 * width / float(resized_w)
        y1, y2 = y1 * height / float(resized_h), y2 * height / float(resized_h)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = float(np.clip(x1, 0, width - 1))
    x2 = float(np.clip(x2, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    y2 = float(np.clip(y2, 0, height - 1))
    if (x2 - x1) < 8 or (y2 - y1) < 8:
        raise RuntimeError("qwen box small")
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def qwen_box(rgb, sentence):
    """
    # Ask Qwen2.5-VL for one box and return that box and the reply.

    ## Args

        - rgb: (H, W, 3) uint8 RGB. It is not None.
        - sentence: the object description. It is not None.

    ## Returns

        - The return is box (4,) float32 pixels, (x1, y1, x2, y2), and reply, a string.
        - Neither: None. The model is deleted before the return.

    ---

    # 向千问 2.5-VL 要一个框，返回这个框和回答文本。

    ## 参数

        - rgb: (H, W, 3) uint8 RGB。不是 None。
        - sentence: 物体描述。不是 None。

    ## 返回

        - 返回值是 box（(4,) float32 像素，(x1, y1, x2, y2)）和 reply（字符串）。
        - 两者都不是 None。
        - 返回前会放开模型。

"""
    import torch
    from PIL import Image
    # Qwen dependencies are optional until language localization is requested.
    # 仅在调用语言定位时准备 Qwen 可选依赖。
    from wapr.bootstrap import ensure_optional
    ensure_optional("qwen")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    try:
        snapshot = qwen_snapshot(QWEN_ROOT)
    except RuntimeError:
        default_root = os.path.join(weights_dir(), "Qwen2.5-VL-3B-Instruct")
        if QWEN_ROOT != default_root or os.environ.get("QWEN_VL_ROOT"):
            raise RuntimeError("Custom Qwen snapshot missing / 自定义 Qwen 快照缺失: " + QWEN_ROOT)
        # The public snapshot is optional until language localization is called.
        # 仅在调用语言定位时下载公开快照，不使用维护者凭据。
        from wapr.source_setup import prepare_qwen_weights
        snapshot = prepare_qwen_weights()
    image = Image.fromarray(rgb)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        snapshot, torch_dtype=torch.bfloat16, device_map=DEVICE,
    )
    processor = AutoProcessor.from_pretrained(snapshot)
    height, width = rgb.shape[:2]
    prompt = (
        "Locate the object described by this sentence. "
        'Reply with only JSON {"bbox_2d": [x1, y1, x2, y2]}. '
        "The image is %d pixels wide and %d pixels tall. "
        "x1, y1 is the top-left pixel and x2, y2 is the bottom-right pixel. "
        "Sentence: %s"
    ) % (width, height, sentence)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    # tokenize=True keeps the image tokens aligned with the pixels.
    # tokenize=True 让图像标记和像素对齐。
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)
    grid = inputs["image_grid_thw"][0].detach().cpu().tolist()
    resized_wh = (int(grid[2]) * 14, int(grid[1]) * 14)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    reply = processor.batch_decode(trimmed, skip_special_tokens=False)[0]
    print("QWEN_RAW", reply.replace("\n", " "), flush=True)
    height, width = rgb.shape[:2]
    box = parse_box(reply, width, height, resized_wh)
    print("QWEN", reply.replace("\n", " "), "BOX", [round(float(v), 1) for v in box], flush=True)
    del generated
    del model
    del processor
    torch.cuda.empty_cache()
    return box, reply


def mask_from_box(rgb, box):
    """
    # Return the SAM 2 mask for one box.

    ## Args

        - rgb: (H, W, 3) uint8 RGB. It is not None.
        - box: (4,) pixels, (x1, y1, x2, y2), x right and y down. It is not None.

    ## Returns

        - The return is (H, W) uint8, 0 or 1.
        - It is not None.
        - The highest SAM 2 score is kept.
        - The model is deleted before the return.

    ---

    # 返回一个框对应的 SAM 2 mask。

    ## 参数

        - rgb: (H, W, 3) uint8 RGB。不是 None。
        - box: (4,) 像素，(x1, y1, x2, y2)，x 向右，y 向下。不是 None。

    ## 返回

        - 返回值是 (H, W) uint8，取值 0 或 1。
        - 不是 None。
        - 留下 SAM 2 分数最高的一张。
        - 返回前会放开模型。

"""
    import torch

    from wapr.sam3d_isolated import SAM3D_ENV_ROOT
    if os.path.realpath(sys.prefix) != os.path.realpath(SAM3D_ENV_ROOT):
        # The original large SAM2 box recipe runs without replacing pose Torch.
        # 原 large SAM2 框分割配方独立运行，不替换位姿 Torch。
        from wapr.sam2_isolated import predict_mask
        isolated_checkpoint = SAM2_CHECKPOINT
        if isolated_checkpoint == os.path.join(WEIGHTS_DIR, "sam2.1_hiera_large.pt") and not os.path.isfile(isolated_checkpoint):
            isolated_checkpoint = None
        return predict_mask(rgb, box_xyxy=box, device=DEVICE,
                            checkpoint=isolated_checkpoint, config=SAM2_CONFIG)
    if SAM2_ROOT not in sys.path:
        sys.path.insert(0, SAM2_ROOT)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam)
    predictor.set_image(rgb)
    masks, scores, _low = predictor.predict(
        box=np.asarray(box, dtype=np.float32),
        multimask_output=True,
    )
    pick = int(np.argmax(scores))
    mask = (masks[pick] > 0).astype(np.uint8)
    print("SAM", int(mask.sum()), "score", round(float(scores[pick]), 3), flush=True)
    del predictor
    del sam
    torch.cuda.empty_cache()
    return mask


def obb_rotation(verts):
    """
    # Return local coordinates, the box extent, and the Open3D box axes.

    ## Args

        - verts: (N, 3) meters. It is not None.

    ## Returns

        - The return has three fields, and none is None.
        - local: (N, 3) float64 meters in the box frame.
        - extent: (3,) float64 meters.
        - rotation: (3, 3) float64. Its columns rotate local coordinates into the centered vertices.

    ---

    # 返回局部坐标、包围盒尺寸，以及 Open3D 包围盒的轴。

    ## 参数

        - verts: (N, 3)，单位米。不是 None。

    ## 返回

        - 返回值有三项，都不是 None。
        - local: 包围盒坐标系里的 (N, 3) float64，米。
        - extent: (3,) float64，米。
        - rotation: (3, 3) float64。它的列把局部坐标转进中心化后的顶点。

"""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(verts, np.float64)))
    box = cloud.get_oriented_bounding_box()
    rotation = np.asarray(box.R, dtype=np.float64)
    extent = np.asarray(box.extent, dtype=np.float64)
    center = 0.5 * (np.asarray(verts, np.float64).min(axis=0) + np.asarray(verts, np.float64).max(axis=0))
    local = (np.asarray(verts, np.float64) - center) @ rotation
    return local, extent, rotation


def display_matrix(after_verts, cad_verts):
    """
    # Return the rotation that puts the reconstruction axes onto the real mesh axes.

    ## Args

        - after_verts: (N, 3) meters, the scaled reconstruction. It is not None.
        - cad_verts: (M, 3) meters, the real mesh. It is not None. It sets axis order and sign. It does not set scale.

    ## Returns

        - The return is (3, 3) float64, unitless.
        - It is not None.

    ---

    # 返回把重建的轴转到真实网格轴上的旋转。

    ## 参数

        - after_verts: (N, 3)，米，已经缩放过的重建顶点。不是 None。
        - cad_verts: (M, 3)，米，真实网格。不是 None。它决定轴的顺序和符号。它不决定尺度。

    ## 返回

        - 返回值是 (3, 3) float64，无量纲。
        - 不是 None。

"""
    _local, extent, rotation = obb_rotation(after_verts)
    cad_ext = np.asarray(cad_verts, np.float64).max(axis=0) - np.asarray(cad_verts, np.float64).min(axis=0)
    src = np.argsort(extent)
    dst = np.argsort(cad_ext)
    center = 0.5 * (after_verts.min(axis=0) + after_verts.max(axis=0))
    local = (after_verts - center) @ rotation
    cad_c = np.asarray(cad_verts, np.float64)
    cad_c = cad_c - cad_c.mean(axis=0)
    signs = np.ones(3, dtype=np.float64)
    for rank in range(3):
        new_axis = local[:, src[rank]]
        old_axis = cad_c[:, dst[rank]]
        new_m3 = float(np.mean((new_axis - new_axis.mean()) ** 3))
        old_m3 = float(np.mean((old_axis - old_axis.mean()) ** 3))
        if new_m3 * old_m3 < 0.0:
            signs[rank] = -1.0
    mapping = np.zeros((3, 3), dtype=np.float64)
    for rank in range(3):
        mapping[src[rank], dst[rank]] = signs[rank]
    if np.linalg.det(rotation @ mapping) < 0.0:
        mapping[src[2], dst[2]] *= -1.0
    return rotation @ mapping


def apply_display(verts, matrix):
    """
    # Rotate vertices about their box center and return them centered.

    ## Args

        - verts: (N, 3) meters. It is not None.
        - matrix: (3, 3) float64 from display_matrix. It is not None.

    ## Returns

        - The return is (N, 3) float64 meters, centered on the new box.
        - It is not None.
        - The input array is copied first.

    ---

    # 绕包围盒中心旋转顶点，并返回居中后的顶点。

    ## 参数

        - verts: (N, 3)，单位米。不是 None。
        - matrix: display_matrix 返回的 (3, 3) float64。不是 None。

    ## 返回

        - 返回值是 (N, 3) float64，米，以新的包围盒为中心。
        - 不是 None。
        - 传入的数组会先被拷贝。

"""
    out = np.asarray(verts, dtype=np.float64).copy()
    center = 0.5 * (out.min(axis=0) + out.max(axis=0))
    out = (out - center) @ matrix
    center = 0.5 * (out.min(axis=0) + out.max(axis=0))
    return out - center


def load_cad_vertices():
    """
    # Return the published mustard real-mesh vertices.

    ## Args

        - There are no arguments.

    ## Returns

        - The return is (N, 3) float32 meters, from mustard_cad.bin.
        - It is not None.

    ---

    # 返回页面上芥末瓶真实网格的顶点。

    ## 参数

        - 没有参数。

    ## 返回

        - 返回值是 mustard_cad.bin 里的 (N, 3) float32，单位米。
        - 不是 None。

"""
    page = os.path.join(RELEASE_DIR, "pages", "demo", "reconstruct", "mustard_cad.json")
    blob = os.path.join(RELEASE_DIR, "pages", "demo", "reconstruct", "mustard_cad.bin")
    with open(page, encoding="utf-8") as stream:
        meta = json.load(stream)
    raw = open(blob, "rb").read()
    count = int(meta["vertices"])
    return np.frombuffer(raw, dtype=np.float32, count=count * 3).reshape(count, 3)


# The published mustard's long axis is Z. These degrees lay that axis across the
# frame and turn the printed label toward the camera. The flip keeps the label upright.
# 已发布芥末瓶的长轴是 Z。这几个角度让长轴横过来，并把印刷标签转到相机前。翻转后标签是正的。
LABEL_SIDE_DEG = 165.0
LABEL_TILT_DEG = 35.0
LABEL_UPRIGHT = True


FLOW_LABELS = (
    ("Text description", "黄色的瓶子"),
    ("Qwen localization", "千问目标定位"),
    ("SAM 2 segmentation", "SAM 2 实例分割"),
    ("Reconstructed object mesh", "重建物体网格模型"),
    ("Pose-aligned outline", "位姿与轮廓对齐"),
)


def compose_flow_board(panels_rgb, path):
    """
    # Write the five-panel language board.

    ## Args

        - panels_rgb: five (H, W, 3) uint8 RGB images, in prompt order. It is not None. A panel of another size is resized to 640 by 480.
        - path: the PNG path. It is not None.

    ## Returns

        - Returns None.
        - The return is None.

    ---

    # 写出语言提示的五格图。

    ## 参数

        - panels_rgb: 五张 (H, W, 3) uint8 RGB，顺序和提示流程一致。不是 None。其他尺寸的一格会被缩到 640 乘 480。
        - path: PNG 路径。不是 None。

    ## 返回

        - 返回 None。
        - 返回值是 None。

"""
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    tile_w, tile_h = 640, 480
    gap = 56
    title_h = 96
    label_h = 78
    row_gap = 36
    canvas_w = 3 * tile_w + 4 * gap
    canvas_h = title_h + 2 * (tile_h + label_h) + row_gap
    board = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(board)
    face_title = ImageFont.truetype(FONT_PATH, 40)
    face_sub = ImageFont.truetype(FONT_PATH, 24)
    face = ImageFont.truetype(FONT_PATH, 26)
    face_zh = ImageFont.truetype(FONT_PATH, 22)
    draw.text((canvas_w / 2, 32), "Language-guided reconstruction", font=face_title, fill=(28, 28, 28), anchor="mm")
    draw.text(
        (canvas_w / 2, 68),
        "文本定位、实例分割、三维重建与图像轮廓对齐。",
        font=face_sub,
        fill=(70, 70, 70),
        anchor="mm",
    )
    # Row two is centered under the three panels.
    # 第二行两格在三格下面居中。
    row2_span = 2 * tile_w + gap
    row2_origin = (canvas_w - row2_span) / 2
    row_top = (title_h, title_h + tile_h + label_h + row_gap)
    slots = (
        (gap + 0 * (tile_w + gap), row_top[0]),
        (gap + 1 * (tile_w + gap), row_top[0]),
        (gap + 2 * (tile_w + gap), row_top[0]),
        (row2_origin, row_top[1]),
        (row2_origin + tile_w + gap, row_top[1]),
    )
    for index, (left, top) in enumerate(slots):
        panel = panels_rgb[index]
        if panel.shape[1] != tile_w or panel.shape[0] != tile_h:
            panel = cv2.resize(panel, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        board.paste(Image.fromarray(panel), (int(left), int(top)))
        en_text, zh_text = FLOW_LABELS[index]
        draw.text((left + tile_w / 2, top + tile_h + 26), en_text, font=face, fill=(28, 28, 28), anchor="mm")
        draw.text((left + tile_w / 2, top + tile_h + 54), zh_text, font=face_zh, fill=(70, 70, 70), anchor="mm")
    # Arrows stay inside each row. The second row continues the same order.
    # 箭头留在每一行里。第二行接着同一个顺序。
    arrow_pairs = ((0, 1), (1, 2), (3, 4))
    for left_index, right_index in arrow_pairs:
        left, top = slots[left_index]
        y = top + tile_h / 2
        x0 = left + tile_w + 10
        tip = slots[right_index][0] - 8
        draw.line([(x0, y), (tip - 16, y)], fill=(40, 40, 40), width=4)
        draw.polygon([(tip, y), (tip - 16, y - 9), (tip - 16, y + 9)], fill=(40, 40, 40))
    board.save(path, "PNG", optimize=True)


def baked_label_view(width, height):
    """
    # Return a software view of the published mustard with the label toward the camera.

    ## Args

        - width: the canvas width in pixels. It is not None.
        - height: the canvas height in pixels. It is not None.

    ## Returns

        - The return is (height, width, 3) uint8 BGR.
        - It is not None.
        - Each face uses the color at its UV centroid.
        - No GL context is opened.

    ---

    # 返回已发布芥末瓶的软件视图，印刷标签朝向相机。

    ## 参数

        - width: 画布宽，单位像素。不是 None。
        - height: 画布高，单位像素。不是 None。

    ## 返回

        - 返回值是 (height, width, 3) uint8 BGR。
        - 不是 None。
        - 每个面用 UV 中心的颜色。
        - 这里不开 GL 上下文。

"""
    import cv2
    from pose_frame_align import load_viewer_mesh, texture_rgb

    stem = os.path.join(RELEASE_DIR, "pages", "demo", "reconstruct", "mustard")
    mesh = load_viewer_mesh(stem)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    local = verts - verts.mean(axis=0)
    side = np.deg2rad(LABEL_SIDE_DEG)
    tilt = np.deg2rad(LABEL_TILT_DEG)
    camera = np.array([
        np.cos(side) * np.cos(tilt),
        np.sin(side) * np.cos(tilt),
        np.sin(tilt),
    ])
    camera = camera / np.linalg.norm(camera)
    right = np.array([0.0, 0.0, 1.0])
    up = np.cross(camera, right)
    up = up / np.linalg.norm(up)
    if LABEL_UPRIGHT:
        up = -up
    right = np.cross(up, camera)
    view = np.stack([local @ right, local @ up, local @ camera], axis=1)
    span = np.ptp(view[:, :2], axis=0).max()
    if span < 1e-8:
        span = 1.0
    scale = 0.86 * min(width, height) / span
    xy = np.stack([
        width / 2 + view[:, 0] * scale,
        height / 2 - view[:, 1] * scale,
    ], axis=1)
    uv, image = texture_rgb(mesh)
    tex_h, tex_w = image.shape[:2]
    centroid = uv[faces].mean(axis=1)
    tex_x = np.clip(np.rint(centroid[:, 0] * (tex_w - 1)).astype(np.int32), 0, tex_w - 1)
    tex_y = np.clip(np.rint((1.0 - centroid[:, 1]) * (tex_h - 1)).astype(np.int32), 0, tex_h - 1)
    colors = image[tex_y, tex_x]
    order = np.argsort(view[faces].mean(axis=1)[:, 2])
    canvas = np.full((height, width, 3), 244, dtype=np.uint8)
    for face_index in order:
        poly = np.rint(xy[faces[face_index]]).astype(np.int32)
        color = colors[face_index]
        cv2.fillConvexPoly(canvas, poly, (int(color[2]), int(color[1]), int(color[0])))
    return canvas


def draw_flow(rgb, box, mask, mesh, pose, K, path):
    """
    # Write the five-panel language figure.

    ## Args

        - rgb: (H, W, 3) uint8 RGB. It is not None.
        - box: (4,) pixels, (x1, y1, x2, y2). It is not None.
        - mask: (H, W) uint8, 0 or 1. It is not None.
        - mesh: the independently reconstructed trimesh, vertices in meters. Both the mesh view and contour panel use this prediction.
        - pose: (4, 4), translation in meters. It is not None. It places mesh for the contour panel.
        - K: (3, 3) camera intrinsics in pixels. It is not None.
        - path: the PNG path. It is not None.

    ## Returns

        - Returns None.
        - The return is None.

    ---

    # 写出语言提示的五格图。

    ## 参数

        - rgb: (H, W, 3) uint8 RGB。不是 None。
        - box: (4,) 像素，(x1, y1, x2, y2)。不是 None。
        - mask: (H, W) uint8，取值 0 或 1。不是 None。
        - mesh: 独立重建的 trimesh，顶点单位米。模型视图和轮廓面板均使用这份预测。
        - pose: (4, 4)，平移单位米。不是 None。它把 mesh 放到轮廓那一格。
        - K: (3, 3) 相机内参，单位像素。不是 None。
        - path: PNG 路径。不是 None。

    ## 返回

        - 返回 None。
        - 返回值是 None。

"""
    import cv2

    height, width = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    boxed = bgr.copy()
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    cv2.rectangle(boxed, (x1, y1), (x2, y2), (40, 90, 220), 3)
    masked = bgr.copy()
    tint = masked.copy()
    # Predicted segmentation is blue; estimated pose contours are red.
    # 预测分割使用蓝色，估计位姿轮廓使用红色。
    tint[mask > 0] = (tint[mask > 0] * 0.45 + np.array([210, 110, 40]) * 0.55).astype(np.uint8)
    masked[mask > 0] = tint[mask > 0]
    _silhouette, aligned = project_silhouette(rgb, mesh, pose, K)
    mesh_panel = shaded_mesh(mesh, width, height)
    panels = [
        cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        for panel in (bgr, boxed, masked, mesh_panel, aligned)
    ]
    compose_flow_board(panels, path)


def shaded_mesh(mesh, width, height, after_rotation=None):
    """
    # Return a software view of the mesh texture.

    ## Args

        - mesh: a trimesh, vertices in meters. It is not None. UV is sampled when present. Otherwise vertex colors are used.
        - width: the canvas width in pixels. It is not None.
        - height: the canvas height in pixels. It is not None.
        - after_rotation: (3, 3) float64 applied in the view frame after label_yaw, or None. None leaves that extra turn out.

    ## Returns

        - The return is (height, width, 3) uint8 BGR.
        - It is not None.
        - No GL context is opened.

    ---

    # 返回网格贴图的软件视图。

    ## 参数

        - mesh: trimesh，顶点单位米。不是 None。有 UV 时采样贴图。否则用顶点色。
        - width: 画布宽，单位像素。不是 None。
        - height: 画布高，单位像素。不是 None。
        - after_rotation: label_yaw 之后在视角坐标系里再乘的 (3, 3) float64，或 None。None 表示不再转。

    ## 返回

        - 返回值是 (height, width, 3) uint8 BGR。
        - 不是 None。
        - 这里不开 GL 上下文。

"""
    import cv2
    from pose_frame_align import label_yaw, paint_mesh_faces

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    center = verts.mean(axis=0)
    local = verts - center
    # A fixed three-quarter view. x right, y up, z toward the camera.
    # 固定的四分之三视角。x 向右，y 向上，z 朝向相机。
    yaw, pitch = label_yaw(mesh), np.deg2rad(20.0)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rotation = np.array([
        [cy, 0, sy],
        [sy * sp, cp, -cy * sp],
        [-sy * cp, sp, cy * cp],
    ], dtype=np.float64)
    view = local @ rotation.T
    if after_rotation is not None:
        view = view @ np.asarray(after_rotation, dtype=np.float64)
    span = np.ptp(view[:, :2], axis=0).max()
    if span < 1e-8:
        span = 1.0
    scale = 0.82 * min(width, height) / span
    xy = np.stack([
        width / 2 + view[:, 0] * scale,
        height / 2 - view[:, 1] * scale,
    ], axis=1)
    canvas = np.full((height, width, 3), 244, dtype=np.uint8)
    paint_mesh_faces(canvas, xy, faces, view[faces].mean(axis=1)[:, 2], mesh)
    return canvas
