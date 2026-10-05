# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Prepare inference inputs from an unpacked TACO release; never load object poses.
# 从已解压的 TACO 发布内容准备推理输入；不读取物体位姿标注。
# Run: python examples/09_taco_many_instances/prepare_sample.py
# 运行：python examples/09_taco_many_instances/prepare_sample.py
import hashlib
import json
import os
import shutil
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
release_dir = os.path.dirname(os.path.dirname(script_dir))
# Change this one root to your unpacked official dataset. / 修改此根目录指向解压数据集。
raw_root = os.path.join(release_dir, "datasets", "TACO")
output_root = os.path.join(release_dir, "samples", "taco")
triplet, sequence = "(brush, roller, box)", "20231104_007"
# Calibration and saved clicks use this RGB grid; depth is nearest-resized to it.
# 标定与保存点击点使用此 RGB 网格；深度以最近邻缩放至该网格。
rgb_size_wh = (512, 376)
# Original depth is a lossless 1920×1080 uint16 stream, scale 4000.
# 原始深度是无损 1920×1080 uint16 视频，比例为 4000。
raw_depth_size_wh = (1920, 1080)


if __name__ == "__main__":
    rgb_path = os.path.join(raw_root, "Egocentric_RGB_Videos", triplet, sequence, "color.mp4")
    depth_path = os.path.join(raw_root, "Egocentric_Depth_Videos", triplet, sequence, "egocentric_depth.avi")
    camera_path = os.path.join(raw_root, "Egocentric_Camera_Parameters", triplet, sequence, "egocentric_intrinsic.txt")
    mesh_paths = [os.path.join(raw_root, "Object_Models", name) for name in ("093_cm.obj", "076_cm.obj")]
    for path in (rgb_path, depth_path, camera_path, *mesh_paths):
        if not os.path.isfile(path):
            raise FileNotFoundError("Unpack the required TACO sequence first / 请先解压对应 TACO 序列: " + path)
    capture = cv2.VideoCapture(rgb_path)
    frame_count = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if (bgr.shape[1], bgr.shape[0]) != rgb_size_wh:
            raise ValueError("This calibration/click set requires 512x376 RGB / 标定及点击点需要 512x376 RGB")
        frame_count += 1
    capture.release()
    if frame_count < 110:
        raise ValueError("The example needs raw frames 80–109 / 示例需要原始第 80–109 帧")
    K = np.loadtxt(camera_path).reshape(3, 3)
    if not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Invalid TACO calibration / TACO 标定无效")
    os.makedirs(os.path.join(output_root, "clip"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "seq_cam"), exist_ok=True)
    output_depth = os.path.join(output_root, "clip", "depth_u16_scale4000_512x376.npy")
    pending_depth = output_depth + ".pending"
    depth_array = np.lib.format.open_memmap(
        pending_depth, mode="w+", dtype=np.uint16,
        shape=(frame_count, rgb_size_wh[1], rgb_size_wh[0]))
    # Decode the original lossless values, not the three-channel OpenCV preview.
    # 解码原始无损数值，不使用 OpenCV 的三通道深度预览。
    process = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", depth_path,
         "-f", "rawvideo", "-pix_fmt", "gray16le", "pipe:1"], stdout=subprocess.PIPE)
    frame_bytes = raw_depth_size_wh[0] * raw_depth_size_wh[1] * np.dtype("<u2").itemsize
    try:
        for frame_id in range(frame_count):
            chunks, read_bytes = [], 0
            while read_bytes < frame_bytes:
                chunk = process.stdout.read(frame_bytes - read_bytes)
                if not chunk:
                    raise RuntimeError("Missing lossless depth frame / 缺少无损深度帧: %d" % frame_id)
                chunks.append(chunk)
                read_bytes += len(chunk)
            raw_depth = np.frombuffer(b"".join(chunks), dtype="<u2").reshape(raw_depth_size_wh[1], raw_depth_size_wh[0])
            depth_array[frame_id] = cv2.resize(raw_depth, rgb_size_wh, interpolation=cv2.INTER_NEAREST)
        if process.stdout.read(1):
            raise RuntimeError("RGB and depth frame counts differ / RGB 与深度帧数不符")
        if process.wait() != 0:
            raise RuntimeError("Lossless depth decoder failed / 无损深度解码失败")
        depth_array.flush()
        del depth_array
        os.replace(pending_depth, output_depth)
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
    shutil.copyfile(rgb_path, os.path.join(output_root, "clip", "color.mp4"))
    shutil.copyfile(camera_path, os.path.join(output_root, "seq_cam", "egocentric_intrinsic.txt"))
    for path in mesh_paths:
        shutil.copyfile(path, os.path.join(output_root, "meshes", os.path.basename(path)))
    source_hashes = {}
    for path in (rgb_path, depth_path, camera_path, *mesh_paths):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        source_hashes[os.path.relpath(path, raw_root)] = digest.hexdigest()
    with open(os.path.join(output_root, "sample.json"), "w", encoding="utf-8") as stream:
        json.dump({"sequence": triplet + "/" + sequence, "frames": frame_count,
                   "rgb_size_wh": rgb_size_wh, "depth_scale": 4000, "mesh_unit": "cm",
                   "source": "https://github.com/leolyliu/TACO-Instructions",
                   "source_sha256": source_hashes, "annotation_files_read": False}, stream, indent=2)
    print("TACO_SAMPLE", {"root": output_root, "frames": frame_count, "rgb_size_wh": rgb_size_wh,
                          "depth_dtype": "uint16", "annotation_files_read": False}, flush=True)
