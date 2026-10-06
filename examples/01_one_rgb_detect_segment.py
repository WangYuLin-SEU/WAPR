# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Example 01. Previous: none. Next: examples/02_one_category_one_instance.py
# 示例 01。上一例：无。下一例：examples/02_one_category_one_instance.py
# 2D boxes and masks on one RGB image. No 6D pose.
# 一张 RGB 上的 2D 框和 mask，没有 6D 位姿。
# Page: docs/det2d.html#call
# 页面：docs/det2d.html#call
# Run: python examples/01_one_rgb_detect_segment.py
# 运行：python examples/01_one_rgb_detect_segment.py
"""Single LM-O RGB frame with a CAD library built in GPU memory. / LM-O 单帧与显存 CAD 库的 RGB 检测示例。
Run from any directory: python /path/to/WAPR/examples/01_one_rgb_detect_segment.py.
可从任意目录执行；默认获取 LM-O 单帧示例并建库，检测器在本次运行中常驻。
"""
import json
import os
import sys
import time

import numpy as np
import torch
import trimesh
from PIL import Image
from pycocotools import mask as mask_utils

# Import the release-local package beside this example.
# 从本示例旁的 release 根目录导入项目内包。
release_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, release_dir)
from wapr.det2d import WAPRDet2D, onboard_meshes
from wapr.download_assets import check_and_fetch_pack
from wapr.view import visualize_2d_detection

# The default root is the release's LM-O sample. An external root must use the
# same BOP layout: models/ and test/000002/rgb/000001.png.
# 默认根目录是发布包中的 LM-O 样例。外部根目录也需采用相同的 BOP 布局：
# models/ 和 test/000002/rgb/000001.png。
sample_root = os.path.join(release_dir, 'samples', 'bop', 'lmo')
input_root = os.environ.get('WAPR_DET2D_INPUT_ROOT', sample_root)
image_path = os.path.join(input_root, 'test', '000002', 'rgb', '000001.png')
models_dir = os.path.join(input_root, 'models')
# Empty keeps rendered views and DINOv2 features on the GPU. For a reusable
# bank, set os.path.join(release_dir, 'outputs', 'cache', 'det2d', 'lmo.pt') here.
# 为空时渲染图和 DINOv2 特征留在显存。若需复用，可在此设为
# os.path.join(release_dir, 'outputs', 'cache', 'det2d', 'lmo.pt')。
template_path = ''
device = 'cuda:0'
backend = 'trt'
# confidence is the 2D gate. A row stays when its score is >= this value.
# confidence 是 2D 阈值。分数不低于这个值才留下。
confidence = 0.1
# dino is vits14, vitb14, or vitl14. grounding is swinb or swint.
# A missing file is downloaded into assets/weights/det2d. vitl14 + trt reuses
# dino_patches_fp16.engine, or builds it when that engine does not match.
# dino 取 vits14、vitb14、vitl14。grounding 取 swinb 或 swint。
# 缺的文件下载到 assets/weights/det2d。vitl14 且 trt 时，沿用 dino_patches_fp16.engine，不匹配才构建。
dino = 'vitl14'
grounding = 'swinb'
# show True opens a window until a key is pressed. image writes the same picture.
# show 为 True 时弹出窗口，按下一个键才关闭。image 把同一张图画下来。
show = False
# The JSON retains every detection. Limit only the illustrated preview so labels
# remain readable when the permissive detector threshold returns many boxes.
# JSON 保留全部检测；只限制插图显示的高分结果，避免宽松阈值下标签遮挡画面。
preview_top_k = 12
output_path = os.path.join(release_dir, 'results', 'det2d', 'single_image.json')
image_path_out = os.path.join(release_dir, 'results', 'det2d', 'single_image.jpg')


def prepare_custom_cads():
    """Save a one-object CAD bank at outputs/cache/det2d/custom.pt when called explicitly.

    Read input_root/models/obj_000001.ply in BOP millimeters, convert its
    vertices to meters, and return the saved bank path. The main flow builds
    the full bank in GPU memory instead.

    手动调用时，把单物体 CAD 库保存到 outputs/cache/det2d/custom.pt。
    读取 input_root/models/obj_000001.ply，将 BOP 毫米顶点换成米，返回库路径。
    主流程则在显存中建立全部物体的模板库。
    """
    if os.path.abspath(input_root) == sample_root:
        check_and_fetch_pack('lmo')
    mesh_path = os.path.join(models_dir, 'obj_000001.ply')
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError('expected one mesh: %s' % mesh_path)
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32) * .001
    return onboard_meshes({1: mesh}, os.path.join(release_dir, 'outputs', 'cache', 'det2d', 'custom.pt'), device=device, dino=dino)


if __name__ == '__main__':
    if os.path.abspath(input_root) == sample_root:
        check_and_fetch_pack('lmo')
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(models_dir)
    # Build a CAD-view bank from the sample meshes before reading the query RGB frame.
    # 先由样例 CAD 网格建立视角模板库，再读取待检测的 RGB 图像。
    # BOP CAD vertices are millimeters. onboard_meshes expects meters and
    # centers each mesh before rendering its template views.
    # BOP CAD 顶点单位是毫米；onboard_meshes 接收米制顶点，并在渲染模板前居中。
    meshes_m = {}
    for name in sorted(os.listdir(models_dir)):
        if not name.startswith('obj_') or not name.endswith('.ply'):
            continue
        obj_id = int(name[len('obj_'):-len('.ply')])
        mesh_path = os.path.join(models_dir, name)
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError('expected one mesh: %s' % mesh_path)
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32) * .001
        meshes_m[obj_id] = mesh
    if not meshes_m:
        raise ValueError('No obj_*.ply meshes under %s' % models_dir)
    torch.cuda.synchronize(device)
    setup_started = time.perf_counter()
    template = onboard_meshes(meshes_m, template_path, device=device, dino=dino)
    print({'input_root': input_root, 'image': image_path, 'obj_ids': sorted(meshes_m),
           'template': template_path if template_path else 'gpu', 'backend': backend,
           'dino': dino, 'grounding': grounding, 'confidence': confidence}, flush=True)
    with Image.open(image_path) as frame:
        rgb = np.asarray(frame.convert('RGB'))
    # Detection returns all accepted boxes and masks; no depth or pose is used here.
    # 检测返回全部保留的框与掩码；此处不使用深度，也不估计位姿。
    detector = WAPRDet2D(template, device=device, backend=backend, dino=dino, grounding=grounding)
    torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - setup_started
    # Warm the same image and confidence; reported inference excludes loading.
    # 在相同图像与置信度下预热，报告的推理时间不包含模型加载。
    warm_started = time.perf_counter()
    for _ in range(3):
        detector.detect_many_categories_many_instances(rgb, confidence=confidence)
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warm_started
    instances, timing = detector.detect_many_categories_many_instances(rgb, profile=True, confidence=confidence)
    timing["warmup_calls"] = 3
    timing["setup_seconds"] = setup_seconds
    timing["warmup_seconds"] = warmup_seconds
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as stream:
        json.dump({'instances': instances, 'timing': timing}, stream, indent=2)
    # Keep the complete JSON result; limit only the number of labels in the preview.
    # JSON 保存完整结果；仅限制预览图中的标签数量。
    preview_instances = sorted(instances, key=lambda row: -float(row['score_2d']))[:preview_top_k]
    image_out = visualize_2d_detection(rgb, preview_instances, image=image_path_out, show=show)
    # Pose consumers receive a visible HxW bool mask; bbox remains the detector bbox.
    # 姿态模块可接收可见 HxW bool mask；bbox 仍是检测包围盒，不替换成 mask 紧包围盒。
    masks = [mask_utils.decode(row['mask']).astype(bool) for row in instances]
    print({'output': output_path, 'image': str(image_out), 'instances': len(masks),
           'preview_instances': len(preview_instances), 'timing': timing})
