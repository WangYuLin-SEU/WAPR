# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# visualize_2d_detection draws the 2D detection result: the box and the mask.
# visualize_2d_detection 画出二维检测结果：框和 mask。
# visualize_6d_pose draws the 6D pose: the box and the rendered contour, when recipe.visualize is true.
# recipe.visualize 为真时，visualize_6d_pose 画出 6D 位姿：框和渲染轮廓。
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wapr import recipe

def pose_silhouette(mesh, pose_4x4, K, height_px, width_px):
    """Offline integer-triangle silhouette for pose projection.

    Mesh and pose share the original object frame, meters; K is in pixels.
    This CPU drawing routine is separate from the batched GPU inference raster.
    供历史投影诊断使用的离线整数三角形剪影。
    网格与位姿使用原始物体系、米制单位，K 为像素单位。
    此 CPU 绘图函数独立于推理使用的批量计算 GPU 光栅渲染。
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    pose = np.asarray(pose_4x4, dtype=np.float64).reshape(4, 4)
    cam = verts @ pose[:3, :3].T + pose[:3, 3]
    z = cam[:, 2]
    keep = z > 1.0e-4
    pix = np.zeros((verts.shape[0], 2), dtype=np.float64)
    pix[keep, 0] = (K[0, 0] * cam[keep, 0] + K[0, 2] * z[keep]) / z[keep]
    pix[keep, 1] = (K[1, 1] * cam[keep, 1] + K[1, 2] * z[keep]) / z[keep]
    face_ok = keep[faces].all(axis=1)
    mask = np.zeros((int(height_px), int(width_px)), dtype=np.uint8)
    # Fill separately to retain the union rather than fillPoly's contour-hole rule.
    # 分别填充以保留三角形并集，避免 fillPoly 的轮廓空洞规则。
    for tri in pix[faces[face_ok]].astype(np.int32):
        cv2.fillConvexPoly(mask, tri, 1)
    return mask

# Bit 0 is x, bit 1 is y, bit 2 is z. The twelve edges of one bbox.
# bit 0 是 x，bit 1 是 y，bit 2 是 z。一个包围盒的十二条边。
_BOX_EDGES = tuple((i, i ^ (1 << bit)) for i in range(8) for bit in range(3) if (i ^ (1 << bit)) > i)

# Red is the predicted pose. Green is ground truth, and only when gt_pose_4x4 is present.
# 红色是预测位姿。绿色是真值，只有记录里有 gt_pose_4x4 时才画。
# Blue is the 2D detection box from bbox or bbox_xywh.
# 蓝色是 bbox 或 bbox_xywh 的二维检测框。
_RED = (220, 30, 30)
_GREEN = (0, 170, 0)
_BLUE = (30, 120, 220)
# One tint per instance, in order. Later instances cover earlier ones where masks overlap.
# 每个实例按顺序用一种颜色。mask 重叠处，后面的实例盖住前面的。
_MASK_COLORS = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
)
_MASK_ALPHA = 110

# Ship the public font with the package; no server font installation is needed.
# 公共字体随包分发，不需要在服务器安装字体。
_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "wqy-microhei.ttc")


def _font(size):
    """
    # Load a font.

        size is in pixels.

        Uses the bundled WenQuanYi Micro Hei font relative to this module.
        A missing bundled file raises OSError, rather than selecting a server font.

        _compose_pose_view calls it at 16 px for labels and the color legend.

    ## Returns

        - Returns a PIL font.

    ---

    # 载入一种字体。

        size 的单位是像素。

        根据本模块位置读取随包的文泉驿微米黑字体。
        随包文件缺失时抛出 OSError，不依赖服务器上的替代字体。

        _compose_pose_view 用 16 像素调用它，画标签和图例。

    ## 返回

        - 返回 PIL 字体。
    """
    return ImageFont.truetype(_FONT_PATH, size)


def _rgb_uint8(rgb):
    """
    # Convert RGB to uint8.

        A contiguous HxWx3 uint8 RGB image.

        _compose_pose_view calls it before drawing.

        frame._rgb_uint8 is a separate converter.

    ## Args

        - rgb: HxWx3 or wider in the last axis. uint8 keeps the first three channels. A float image whose maximum is at most 1.5 is treated as 0–1 and scaled to 0–255. A larger float is clipped to 0–255. Fewer than three channels raises ValueError.

    ---

    # 把 RGB 转成 uint8。

        连续的 HxWx3 uint8 RGB。

        _compose_pose_view 在画图前调用它。

        frame._rgb_uint8 是另一个转换。

    ## 参数

        - rgb 最后一维至少要有 3 个通道。
        - uint8 保留前三个通道。
        - 最大值不超过 1.5 的浮点图按 0–1 放大到 0–255。
        - 更大的浮点截到 0–255。
        - 通道少于 3 抛出 ValueError。
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or int(arr.shape[-1]) < 3:
        raise ValueError("rgb must be HxWx3")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr[..., :3])
    values = arr[..., :3].astype(np.float32)
    if float(np.nanmax(values)) <= 1.5:
        values = np.clip(values, 0.0, 1.0) * 255.0
    else:
        values = np.clip(values, 0.0, 255.0)
    return np.ascontiguousarray(values.astype(np.uint8))


def _aabb_corners(vertices):
    """
    # Compute the eight corners of the axis-aligned box.

        The unit matches the vertices.

        Returns (8, 3).

        _corners_m calls it when the record has a mesh and no corners.

    ## Args

        - vertices: (V, 3) or wider. Bit 0 of the corner index is x, bit 1 is y, bit 2 is z.

    ---

    # 计算轴对齐包围盒的八个角。

        单位与顶点相同。

        返回 (8, 3)。

        记录有 mesh、没有 corners 时，_corners_m 调用它。

    ## 参数

        - vertices: (V, 3) 或更宽。角点下标的 bit 0 是 x，bit 1 是 y，bit 2 是 z。
    """
    points = np.asarray(vertices, dtype=np.float64)[:, :3]
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    corners = np.zeros((8, 3), dtype=np.float64)
    for index in range(8):
        corners[index, 0] = hi[0] if index & 1 else lo[0]
        corners[index, 1] = hi[1] if index & 2 else lo[1]
        corners[index, 2] = hi[2] if index & 4 else lo[2]
    return corners


def _corners_m(record):
    """
    # Compute the eight box corners.

        Eight corners in the original mesh frame, in meters.

        corners on the record wins and must reshape to (8, 3).

        Otherwise the mesh bounds are used.

        A centered mesh stores model_center in metadata, and that offset is added back.

        Neither corners nor mesh raises ValueError.

        Returns (8, 3), meters.

        _compose_pose_view calls this when pose_4x4 or gt_pose_4x4 is present.

    ---

    # 计算八个包围盒角点。

        原始网格坐标系下的八个角点，单位是米。

        记录里的 corners 优先，并 reshape 成 (8, 3)。

        否则用网格包围盒。

        已经居中的网格把 metadata 里的 model_center 加回去。

        两者都没有时抛出 ValueError。

        返回 (8, 3)，米。

        有 pose_4x4 或 gt_pose_4x4 时，_compose_pose_view 调用它。
    """
    if record.get("corners") is not None:
        corners = np.asarray(record["corners"], dtype=np.float64).reshape(8, 3)
        return corners
    mesh = record.get("mesh")
    if mesh is None:
        raise ValueError("a 3D bbox needs corners or mesh")
    corners = _aabb_corners(mesh.vertices)
    center = (getattr(mesh, "metadata", None) or {}).get("model_center")
    if center is not None:
        corners = corners + np.asarray(center, dtype=np.float64).reshape(1, 3)
    return corners


def _project(corners, pose, K):
    """
    # Project the box corners.

        Object-frame corners go to pixels.

        _draw_box uses it for the twelve edges.

        _compose_pose_view uses it to place a label when the record has no 2D box.

    ## Args

        - corners: (8, 3), meters.
        - pose: a 4×4 object-to-camera matrix, meters, OpenCV.
        - K: 3×3, pixels. uv is (8, 2). Division uses z of at least 1e-4, so a point on the camera plane does not explode.

    ## Returns

        - Returns uv and camera-frame z.

    ---

    # 投影包围盒角点。

        物体坐标系的角点投到像素。

        _draw_box 用它画十二条边。

        记录没有 2D 包围盒时，_compose_pose_view 用它放标签。

    ## 参数

        - corners: (8, 3)，米。
        - pose: 4×4 物体到相机矩阵，米，OpenCV。
        - K: 3×3，像素。uv 是 (8, 2)。除法用的 z 至少是 1e-4，相机平面上的点不会爆掉。

    ## 返回

        - 返回 uv 和相机系 z。
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    pts = (pose[:3, :3] @ corners.T).T + pose[:3, 3]
    z = pts[:, 2]
    uv = (np.asarray(K, dtype=np.float64).reshape(3, 3) @ pts.T).T
    uv = uv[:, :2] / np.maximum(z[:, None], 1e-4)
    return uv, z


def _draw_box(draw, corners, pose, K, color, width):
    """
    # Draw the twelve edges whose both ends are in front of the camera.

        _compose_pose_view calls it in green, width 1, for gt_pose_4x4, and in red, width 2, for pose_4x4.

    ## Args

        - draw: a PIL ImageDraw. corners, pose, and K match _project.
        - color: an RGB tuple.
        - width: the line width in pixels. An endpoint with z at or below 1e-4 is skipped. Returns None.

    ---

    # 画出两端都在相机前方的十二条边。

        _compose_pose_view 调用它：gt_pose_4x4 用绿色、线宽 1，pose_4x4 用红色、线宽 2。

    ## 参数

        - draw: PIL ImageDraw。corners、pose、K 与 _project 相同。
        - color: RGB 元组。
        - width: 线宽，像素。端点 z 不大于 1e-4 的边不画。返回 None。
    """
    uv, z = _project(corners, pose, K)
    for a, b in _BOX_EDGES:
        if z[a] <= 1e-4 or z[b] <= 1e-4:
            continue
        draw.line(
            [(float(uv[a, 0]), float(uv[a, 1])), (float(uv[b, 0]), float(uv[b, 1]))],
            fill=color,
            width=width,
        )


def _label_text(record, names):
    """
    # Build one label string.

        A detection score is appended when the record has one.

        The name is record["name"], else names[obj_id], else mesh.metadata["name"], else "obj {id}", else "object". names maps an integer obj_id to a string and may be None.

        A pose label shows score_6d; a detection label shows score_2d. Both use two decimals.

        Returns the string.

        _compose_pose_view calls it once per record.

    ---

    # 生成一行标签。

        记录里有检测分数时，加在名称后面。

        名称先取 record["name"]，否则 names[obj_id]，否则 mesh.metadata["name"]，否则 "obj {id}"，否则 "object"。

        names 把整数 obj_id 映射到字符串，可以是 None。

        位姿标签显示 score_6d，检测标签显示 score_2d，均保留两位小数。

        返回这个字符串。

        _compose_pose_view 对每条记录调用一次。
    """
    name = record.get("name")
    if not name and names is not None and record.get("obj_id") is not None:
        name = names.get(int(record["obj_id"]))
    if not name:
        mesh = record.get("mesh")
        metadata = getattr(mesh, "metadata", None) or {}
        name = metadata.get("name")
    if not name and record.get("obj_id") is not None:
        name = "obj %d" % int(record["obj_id"])
    if not name:
        name = "object"
    text = str(name)
    if record.get("score_6d") is not None:
        text = "%s 6D %.2f" % (text, float(record["score_6d"]))
    elif record.get("score_2d") is not None:
        text = "%s 2D %.2f" % (text, float(record["score_2d"]))
    return text


def _stamp(draw, text, x, y, font, taken, width, height):
    """
    # Draw one dark label in a free image region and reserve its box.

        _compose_pose_view calls it for every object label and once for the color legend.

    ## Args

        - text: the string.
        - x and y: the anchor in pixels, y down. Prefer a label above it, then search nearby positions without covering earlier labels.
        - font: a PIL font.
        - taken: a list of (x, y, w, h) already drawn, and this call appends to it.
        - width and height: the image size in pixels.

    ## Returns

        - Returns None.

    ---

    # 在图像的空闲区域画一块深色标签，并记录其占用范围。

        _compose_pose_view 给每个物体标签调用一次，图例再调用一次。

    ## 参数

        - text: 字符串。
        - x 和 y: 锚点，像素，y 向下。优先放在锚点上方；被已有标签占用时搜索附近空位。
        - font: PIL 字体。
        - taken: 已经画过的 (x, y, w, h)，这次调用会再追加一块。
        - width 和 height: 图像尺寸，像素。

    ## 返回

        - 返回 None。
    """
    box = font.getbbox(text)
    tw = box[2] - box[0]
    th = box[3] - box[1]
    pad = 3
    label_w = tw + 2 * pad
    label_h = th + 2 * pad
    max_x = max(0, width - label_w)
    max_y = max(0, height - label_h)
    preferred_x = min(max(0, int(round(x))), max_x)
    preferred_y = int(round(y - label_h))
    if preferred_y < 0:
        preferred_y = int(round(y + 2))
    preferred_y = min(max(0, preferred_y), max_y)
    # Search the full image after the preferred position; six downward steps can
    # leave overlapping labels on a crowded single-frame detection preview.
    # 首选位置被占用后搜索整幅图；拥挤的单帧检测图不能只向下尝试六次。
    candidate_xs = {
        preferred_x,
        min(max(0, preferred_x - label_w), max_x),
        min(max(0, preferred_x + label_w), max_x),
        0,
        max_x,
    }
    candidate_ys = set(range(0, max_y + 1, label_h + 2))
    candidate_ys.add(preferred_y)
    candidates = sorted(
        ((abs(cx - preferred_x) + abs(cy - preferred_y), cx, cy)
         for cx in candidate_xs for cy in candidate_ys)
    )
    lx, ly = preferred_x, preferred_y
    for _, cx, cy in candidates:
        if all(cx + label_w <= ox or ox + ow <= cx or cy + label_h <= oy or oy + oh <= cy
               for ox, oy, ow, oh in taken):
            lx, ly = cx, cy
            break
    draw.rectangle([lx, ly, lx + label_w, ly + label_h], fill=(20, 20, 20))
    draw.text((lx + pad, ly + pad - box[1]), text, fill=(255, 255, 255), font=font)
    taken.append((lx, ly, label_w, label_h))


def _view_records(poses):
    """
    # Accept pose rows and 2D detection rows in one list.

        A detection row stores bbox and score_2d.

        A pose row stores bbox_xywh, score_2d, and score_6d.

        Returns a new list of dicts.

        The input rows are not modified.

        save_pose_view, show_pose_view, and visualize_2d_detection call it before drawing.

    ---

    # 位姿行和 2D 检测行可以放在同一个列表里。

        检测行用 bbox 和 score_2d。

        位姿行用 bbox_xywh、score_2d 和 score_6d。

        返回新的字典列表。

        不改输入行。

        save_pose_view、show_pose_view 和 visualize_2d_detection 在画图前调用它。
    """
    records = []
    for row in poses:
        item = dict(row)
        if item.get("bbox_xywh") is None and item.get("bbox") is not None:
            item["bbox_xywh"] = item["bbox"]
        records.append(item)
    return records


def _mask_bool(mask, height, width):
    """
    # Decode one 2D segmentation mask.

        A dict with counts is COCO RLE.

        An array is HxW, or HxWx1.

        None returns None.

        A size that differs from the image raises ValueError.

        _compose_pose_view calls it once per row.

    ## Args

        - mask: COCO RLE, an array, or None.
        - height, width: the RGB image size, in pixels.

    ## Returns

        - Returns an HxW bool array, or None.

    ---

    # 解码一块二维分割 mask。

        带 counts 的字典是 COCO RLE。

        数组是 HxW，或 HxWx1。

        None 返回 None。

        尺寸和图像不同则抛出 ValueError。

        _compose_pose_view 对每一行调用一次。

    ## 参数

        - mask: COCO RLE、数组，或 None。
        - height、width: RGB 图像尺寸，像素。

    ## 返回

        - 返回 HxW 的 bool 数组，或 None。
    """
    if mask is None:
        return None
    if isinstance(mask, dict):
        from pycocotools import mask as mask_utils

        decoded = np.asarray(mask_utils.decode(mask))
    else:
        decoded = np.asarray(mask)
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    if decoded.shape[:2] != (height, width):
        raise ValueError("mask size differs from rgb")
    return decoded > 0


def _rendered_mask(mesh, pose_4x4, K, height, width):
    """
    # Render the silhouette of one pose.

        The mask is the union of the projected triangles.

        Each triangle is filled on its own.

        A vertex stays when camera z is above 1.0e-4.

        _compose_pose_view calls it before drawing the outer contour.

    ## Args

        - mesh: vertices and faces, meters, in the same frame as pose_4x4.
        - pose_4x4: (4, 4), object to camera, meters.
        - K: 3×3, pixels.
        - height, width: the image size, in pixels.

    ## Returns

        - Returns an HxW uint8 mask, 0 or 255.

    ---

    # 渲染单个位姿对应的剪影。

        mask 是投影三角形的并集。

        每个三角形单独填充。

        相机坐标 z 大于 1.0e-4 的顶点才保留。

        _compose_pose_view 在画外轮廓之前调用它。

    ## 参数

        - mesh: 顶点和面，米，与 pose_4x4 同一坐标系。
        - pose_4x4: (4, 4)，物体到相机，米。
        - K: 3×3，像素。
        - height、width: 图像尺寸，像素。

    ## 返回

        - 返回 HxW 的 uint8 mask，取 0 或 255。
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    pose = np.asarray(pose_4x4, dtype=np.float64).reshape(4, 4)
    camera = vertices @ pose[:3, :3].T + pose[:3, 3]
    depth = camera[:, 2]
    front = depth > 1.0e-4
    pixels = np.zeros((vertices.shape[0], 2), dtype=np.float64)
    pixels[front, 0] = (K[0, 0] * camera[front, 0] + K[0, 2] * depth[front]) / depth[front]
    pixels[front, 1] = (K[1, 1] * camera[front, 1] + K[1, 2] * depth[front]) / depth[front]
    face_ok = front[faces].all(axis=1)
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    for triangle in pixels[faces[face_ok]].astype(np.int32):
        if triangle[:, 0].max() < 0 or triangle[:, 1].max() < 0:
            continue
        if triangle[:, 0].min() >= width or triangle[:, 1].min() >= height:
            continue
        if np.any(np.abs(triangle[:, 0]) > width * 8) or np.any(np.abs(triangle[:, 1]) > height * 8):
            continue
        cv2.fillConvexPoly(mask, triangle, 255)
    return mask


def _draw_outer_contour(image_u8, mask, color, thickness):
    """
    # Draw the outer contour of one rendered silhouette.

        Internal edges are left out.

        An empty mask draws nothing and returns False.

        _compose_pose_view calls it once per pose.

    ## Args

        - image_u8: HxWx3 uint8 RGB. The contour is written into this array.
        - mask: HxW uint8 silhouette.
        - color: RGB tuple.
        - thickness: line width in pixels.

    ## Returns

        - Returns True when a contour was drawn.

    ---

    # 画出一块渲染剪影的外轮廓。

        内部边不画。

        空 mask 不画，并返回 False。

        _compose_pose_view 对每条位姿调用一次。

    ## 参数

        - image_u8: HxWx3，uint8，RGB。轮廓写进这个数组。
        - mask: HxW，uint8 剪影。
        - color: RGB 三元组。
        - thickness: 线宽，像素。

    ## 返回

        - 画出轮廓时返回 True。
    """
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(image_u8, [contour], -1, color, int(thickness), cv2.LINE_AA)
    return True


def _compose_pose_view(rgb, poses, K=None, names=None):
    """
    # Compose the 6D box and rendered contour.

    ## Args

        - rgb: HxWx3, uint8 or float, as _rgb_uint8 accepts.
        - poses: the record list from _view_records.
        - K: 3×3 pixels and is required when any record has pose_4x4 or gt_pose_4x4.
        - names: obj_id to a label and may be None.

    ## Returns

        - Nothing: written. Returns a PIL image.
        - Red is the predicted 6D box and its rendered contour. Green is ground truth. save_pose_view and show_pose_view call it.

    ---

    # 合成 6D 框和渲染轮廓。

    ## 参数

        - rgb: HxWx3，uint8 或浮点，规则与 _rgb_uint8 相同。
        - poses: _view_records 给出的记录列表。任一记录有 pose_4x4 或 gt_pose_4x4 时，K 必须是 3×3，像素。names 把 obj_id 映射到标签，可以是 None。

    ## 返回

        - 不写文件。
        - 返回 PIL 图像。
        - 红色是预测的 6D 框和它的渲染轮廓。绿色是真值。
        - save_pose_view 和 show_pose_view 调用它。
    """
    rgb_u8 = np.array(_rgb_uint8(rgb), copy=True)
    height, width = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])
    drew_contour = False
    for record in poses:
        mesh = record.get("mesh")
        if mesh is None or K is None:
            continue
        if record.get("gt_pose_4x4") is not None:
            gt_mask = _rendered_mask(mesh, record["gt_pose_4x4"], K, height, width)
            drew_contour = _draw_outer_contour(rgb_u8, gt_mask, _GREEN, 2) or drew_contour
        if record.get("pose_4x4") is not None:
            pred_mask = _rendered_mask(mesh, record["pose_4x4"], K, height, width)
            drew_contour = _draw_outer_contour(rgb_u8, pred_mask, _RED, 2) or drew_contour
    image = Image.fromarray(rgb_u8)
    draw = ImageDraw.Draw(image)
    font = _font(16)
    taken = []
    drew_pred = False
    drew_gt = False
    for record in poses:
        corners = None
        if record.get("pose_4x4") is not None or record.get("gt_pose_4x4") is not None:
            if K is None:
                raise ValueError("K is required for a 3D bbox")
            corners = _corners_m(record)
        if record.get("gt_pose_4x4") is not None:
            _draw_box(draw, corners, record["gt_pose_4x4"], K, _GREEN, 1)
            drew_gt = True
        if record.get("pose_4x4") is not None:
            _draw_box(draw, corners, record["pose_4x4"], K, _RED, 2)
            drew_pred = True
        anchor_x, anchor_y = 8, 8
        if corners is not None and record.get("pose_4x4") is not None:
            uv, z = _project(corners, record["pose_4x4"], K)
            visible = uv[z > 1e-4]
            if len(visible):
                anchor_x = float(visible[:, 0].min())
                anchor_y = float(visible[:, 1].min())
        _stamp(draw, _label_text(record, names), anchor_x, anchor_y, font, taken, width, height)
    legend = []
    if drew_pred and drew_contour:
        legend.append("red: 6D box and contour")
    elif drew_pred:
        legend.append("red: 6D box")
    if drew_gt:
        legend.append("green: ground truth")
    if legend:
        _stamp(draw, "  ".join(legend), 8, 22, font, [], width, height)
    return image


def _write_pose_image(image, path):
    """
    # Write one RGB image as JPEG or PNG.

        Parent directories are created.

        Returns the Path.

        save_pose_view calls it with the given path.

        visualize_2d_detection calls it when image is set.

    ## Args

        - image: a PIL RGB image.
        - path: the file. The suffix selects the format OpenCV writes. A failed write raises OSError. The line POSE_VIEW is printed.

    ---

    # 把一张 RGB 图写成 JPEG 或 PNG。

        上级目录会建出来。

        返回 Path。

        save_pose_view 用给定路径调用它。

        image 有值时 visualize_2d_detection 调用它。

    ## 参数

        - image: PIL 的 RGB 图。
        - path: 文件。后缀决定 OpenCV 写成哪种格式。写失败抛出 OSError。会打印一行 POSE_VIEW。
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(out), bgr):
        raise OSError("could not write %s" % out)
    print("POSE_VIEW", str(out), flush=True)
    return out


def save_pose_view(rgb, poses, path, K=None, names=None):
    """
    # Draw the 6D pose picture.

        The picture is the 6D pose: the red box and the rendered contour of mesh at pose_4x4.

        gt_pose_4x4 uses the same frame and draws the green box and contour.

        mesh supplies the contour and the 3D corners when corners is absent.

    ## Args

        - path: the image file. Parent directories are created. Returns that Path.
        - K: 3×3 pixels when a 3D box is drawn, otherwise None.
        - names: obj_id to a label and may be None. visualize_6d_pose calls this when recipe.visualize is true.

    ## Returns

        - name: the user string drawn on the image.
        - obj_id: used with names when name is absent.
        - score_6d: shown for pose rows; score_2d: shown for detection rows.
        - pose_4x4: object-to-camera, meters, OpenCV. It draws the red 6D box. mesh draws the contour.
        - The blue 2D box and the mask are visualize_2d_detection.
        - corners: (8, 3), meters, in the original mesh frame.

    ---

    # 画出 6D 位姿图。

        这张图是 6D 位姿：红色框，以及 mesh 在 pose_4x4 下的渲染轮廓。

        gt_pose_4x4 与 pose_4x4 同一坐标系，画绿色框和轮廓。

        没有 corners 时用 mesh 的包围盒。轮廓也用这份 mesh。

    ## 参数

        - path: 图像文件。上级目录会建出来。返回这个 Path。
        - 要画 3D 盒时 K 是 3×3，像素，否则可以是 None。
        - names 把 obj_id 映射到标签，可以是 None。
        - recipe.visualize 为真时，visualize_6d_pose 调用它。

    ## 返回

        - name: 画在图上的用户字符串。
        - 没有 name 时，用 obj_id 到 names 里查。
        - 位姿行显示 score_6d 并标为 6D；检测行显示 score_2d 并标为 2D。
        - pose_4x4: 物体到相机，米，OpenCV，画红色 6D 框。mesh 画轮廓。
        - 蓝色二维框和 mask 是 visualize_2d_detection。
        - corners: 原始网格坐标系下的 (8, 3)，米。
    """
    image = _compose_pose_view(rgb, _view_records(poses), K=K, names=names)
    return _write_pose_image(image, path)


def show_pose_view(rgb, poses, K=None, names=None, window="WAPR"):
    """
    # Open one window and wait until a key is pressed.

        No DISPLAY raises RuntimeError.

        The caller can still save the same image with save_pose_view.

        Returns None.

    ## Args

        - window: the OpenCV window name. The default is WAPR. rgb, poses, K, and names match save_pose_view. No function in this release calls show_pose_view. Call it when a window is wanted.

    ---

    # 打开一个窗口，按下一个键才返回。

        没有 DISPLAY 时抛出 RuntimeError。

        调用方仍可以用 save_pose_view 把同一张图存下来。

        返回 None。

    ## 参数

        - window: OpenCV 窗口名，默认 WAPR。rgb、poses、K、names 与 save_pose_view 相同。这个发布包里没有函数调用 show_pose_view。需要弹窗时直接调用它。
    """
    image = _compose_pose_view(rgb, _view_records(poses), K=K, names=names)
    _show_image(image, window)
    return None


def _show_image(image, window):
    """
    # Open one window and wait until a key is pressed.

        show_pose_view always calls it.

        visualize_2d_detection calls it when show is true.

    ## Args

        - image: a PIL RGB image.
        - window: the OpenCV window name. No DISPLAY and no WAYLAND_DISPLAY raises RuntimeError. A cv2 error is raised again as RuntimeError.

    ## Returns

        - Returns None.

    ---

    # 打开一个窗口，按下一个键才返回。

        show_pose_view 总会调用它。

        show 为真时 visualize_2d_detection 调用它。

    ## 参数

        - image: PIL 的 RGB 图。
        - window: OpenCV 窗口名。没有 DISPLAY 也没有 WAYLAND_DISPLAY 时抛出 RuntimeError。cv2 出错会再包成 RuntimeError。

    ## 返回

        - 返回 None。
    """
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("No display for the popup. Pass save= to write the image.")
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, bgr)
        print("POSE_VIEW_SHOW", window, "press a key to close", flush=True)
        cv2.waitKey(0)
        cv2.destroyWindow(window)
    except cv2.error as exc:
        raise RuntimeError("The popup did not open. Pass save= to write the image.") from exc


def _compose_2d_detection(rgb, poses, names=None):
    """
    # Compose the 2D detection picture.

        bbox or bbox_xywh is the blue box.

        mask is the tint, one color per instance. It may be COCO RLE or an HxW array.

        visualize_2d_detection calls it.

    ## Args

        - rgb: HxWx3, uint8 or float, as _rgb_uint8 accepts.
        - poses: the record list from _view_records.
        - names: obj_id to a label and may be None.

    ## Returns

        - Returns a PIL image. Nothing is written.

    ---

    # 合成二维检测图。

        bbox 或 bbox_xywh 是蓝色框。

        mask 是色块，每个实例一种颜色。可以是 COCO RLE 或 HxW 数组。

        visualize_2d_detection 调用它。

    ## 参数

        - rgb: HxWx3，uint8 或浮点，规则与 _rgb_uint8 相同。
        - poses: _view_records 给出的记录列表。
        - names 把 obj_id 映射到标签，可以是 None。

    ## 返回

        - 返回 PIL 图像。不写文件。
    """
    rgb_u8 = _rgb_uint8(rgb)
    height, width = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])
    canvas = Image.fromarray(rgb_u8).convert("RGBA")
    tint = np.zeros((height, width, 4), dtype=np.uint8)
    drew_mask = False
    for index, record in enumerate(poses):
        visible = _mask_bool(record.get("mask"), height, width)
        if visible is None:
            continue
        color = _MASK_COLORS[index % len(_MASK_COLORS)]
        tint[visible, 0] = color[0]
        tint[visible, 1] = color[1]
        tint[visible, 2] = color[2]
        tint[visible, 3] = _MASK_ALPHA
        drew_mask = True
    if drew_mask:
        canvas = Image.alpha_composite(canvas, Image.fromarray(tint, mode="RGBA"))
    image = canvas.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font(16)
    taken = []
    for record in poses:
        anchor_x, anchor_y = 8, 8
        box = record.get("bbox_xywh")
        if box is not None:
            x, y, w, h = [float(value) for value in box]
            draw.rectangle([x, y, x + w, y + h], outline=_BLUE, width=2)
            anchor_x, anchor_y = x, y
        _stamp(draw, _label_text(record, names), anchor_x, anchor_y, font, taken, width, height)
    legend = []
    if drew_mask:
        legend.append("tint: 2D mask")
    if any(record.get("bbox_xywh") is not None for record in poses):
        legend.append("blue: 2D box")
    if legend:
        _stamp(draw, "  ".join(legend), 8, 22, font, [], width, height)
    return image


def visualize_2d_detection(rgb, detections, names=None, image=None, show=False, filename="det2d.jpg", window="WAPR"):
    """
    # Visualize 2D detection.

        The picture is the 2D detection result: the blue box and the tinted mask.

        image writes that picture. show True opens a window until a key is pressed.

        This call does not read recipe.visualize. visualize_6d_pose does.

        Returns the written Path, or None when image is omitted.

    ## Args

        - detections: rows with bbox or bbox_xywh, score_2d, and mask. mask is COCO RLE or an HxW array.
        - names: obj_id to a label and may be None.
        - image: a path, or True. True uses recipe.visualize_path and filename.

    ---

    # 可视化二维检测结果。

        这张图是二维检测结果：蓝色框和带颜色的 mask。

        image 写出这张图。show 为 True 时弹出窗口，按下一个键才关闭。

        这次调用不看 recipe.visualize。visualize_6d_pose 才看。

        写出文件时返回 Path。没有 image 时返回 None。

    ## 参数

        - detections: 带 bbox 或 bbox_xywh、score_2d、mask 的行。mask 是 COCO RLE 或 HxW 数组。
        - names 把 obj_id 映射到标签，可以是 None。
        - image: 路径，或者 True。True 时用 recipe.visualize_path 和 filename。

"""
    picture = _compose_2d_detection(rgb, _view_records(detections), names=names)
    written = None
    if image is True:
        written = _write_pose_image(picture, recipe.resolve_visualize_path(filename))
    elif image:
        written = _write_pose_image(picture, image)
    if show:
        _show_image(picture, window)
    return written


def visualize_6d_pose(rgb, poses, K=None, names=None, filename="pose.jpg"):
    """
    # Visualize 6D pose.

        The picture is the 6D pose: the red box and the rendered contour of mesh at pose_4x4.

        A green box and contour are ground truth when gt_pose_4x4 is present.

        recipe.visualize false writes nothing and returns None.

        The file is recipe.visualize_path. A directory uses filename. A .jpg or .png path is the file itself.

        The 2D box and the 2D mask are visualize_2d_detection.

    ## Args

        - filename: the file name inside recipe.visualize_path when that value is a directory.
        - poses: rows with pose_4x4 and mesh. mesh is meters, in the same frame as pose_4x4.
        - K: 3×3, pixels.
        - names: obj_id to a label and may be None.

    ---

    # 可视化 6D 位姿。

        这张图是 6D 位姿：红色框，以及 mesh 在 pose_4x4 下的渲染轮廓。

        给了 gt_pose_4x4 时，绿色框和轮廓是真实位姿。

        recipe.visualize 关掉时不存图，返回 None。

        图像保存到 recipe.visualize_path 指定的位置：若该路径是目录，则使用 filename 作为文件名；若该路径是 .jpg 或 .png 文件，则直接写入该文件。

        二维框和二维 mask 是 visualize_2d_detection。

    ## 参数

        - filename: recipe.visualize_path 是目录时的文件名。
        - poses: 带 pose_4x4 和 mesh 的行。mesh 是米，与 pose_4x4 同一坐标系。
        - K: 3×3，像素。
        - names 把 obj_id 映射到标签，可以是 None。

"""
    if not recipe.visualize:
        return None
    return save_pose_view(rgb, poses, recipe.resolve_visualize_path(filename), K=K, names=names)
