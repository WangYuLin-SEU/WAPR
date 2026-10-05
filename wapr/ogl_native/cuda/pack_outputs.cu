// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#include "pack_outputs.cuh"
#include <cuda_runtime.h>

namespace wapr_ogl {
namespace cuda_kernels {

__global__ void pack_tiles_kernel(
    const float* depth_atlas,
    const float* rgb_atlas,
    const float* normal_atlas,
    float* out_depth,
    float* out_rgb,
    float* out_normal,
    int atlas_w,
    int atlas_h,
    int tile_w,
    int tile_h,
    int batch_size,
    int cols,
    int y_flip,
    int output_normal) {
    const int n = blockIdx.x;
    if (n >= batch_size) return;
    const int ty = n / cols;
    const int tx = n % cols;
    const int ox0 = tx * tile_w;
    const int oy0 = ty * tile_h;

    for (int y = threadIdx.x; y < tile_h; y += blockDim.x) {
        const int sy = y_flip ? (oy0 + tile_h - 1 - y) : (oy0 + y);
        for (int x = 0; x < tile_w; ++x) {
            const int sx = ox0 + x;
            const int si = sy * atlas_w + sx;
            const int di = n * tile_w * tile_h + y * tile_w + x;
            if (out_depth && depth_atlas) {
                out_depth[di] = depth_atlas[si];
            }
            if (out_rgb && rgb_atlas) {
                out_rgb[di * 3 + 0] = rgb_atlas[si * 4 + 0];
                out_rgb[di * 3 + 1] = rgb_atlas[si * 4 + 1];
                out_rgb[di * 3 + 2] = rgb_atlas[si * 4 + 2];
            }
            if (output_normal && out_normal && normal_atlas) {
                out_normal[di * 3 + 0] = normal_atlas[si * 4 + 0];
                out_normal[di * 3 + 1] = normal_atlas[si * 4 + 1];
                out_normal[di * 3 + 2] = normal_atlas[si * 4 + 2];
            }
        }
    }
}

void pack_tiles_rgb_depth(
    const float* d_depth_atlas,
    const float* d_rgb_atlas,
    const float* d_normal_atlas,
    float* d_out_depth,
    float* d_out_rgb,
    float* d_out_normal,
    uint32_t atlas_w,
    uint32_t atlas_h,
    uint32_t tile_w,
    uint32_t tile_h,
    uint32_t batch_size,
    uint32_t cols,
    bool y_flip,
    bool output_normal,
    cudaStream_t stream) {
    pack_tiles_kernel<<<batch_size, 256, 0, stream>>>(
        d_depth_atlas, d_rgb_atlas, d_normal_atlas,
        d_out_depth, d_out_rgb, d_out_normal,
        static_cast<int>(atlas_w), static_cast<int>(atlas_h),
        static_cast<int>(tile_w), static_cast<int>(tile_h),
        static_cast<int>(batch_size), static_cast<int>(cols),
        y_flip ? 1 : 0, output_normal ? 1 : 0);
}

__global__ void pack_scene_kernel(
    const float* depth,
    const float* rgb,
    float* out_depth,
    float* out_rgb,
    int w, int h, int y_flip) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = w * h;
    if (idx >= n) return;
    const int x = idx % w;
    const int y = idx / w;
    const int sy = y_flip ? (h - 1 - y) : y;
    const int si = sy * w + x;
    if (out_depth && depth) {
        out_depth[idx] = depth[si];
    }
    if (out_rgb && rgb) {
        out_rgb[idx * 3 + 0] = rgb[si * 4 + 0];
        out_rgb[idx * 3 + 1] = rgb[si * 4 + 1];
        out_rgb[idx * 3 + 2] = rgb[si * 4 + 2];
    }
}

void pack_scene_rgb_depth(
    const float* d_depth,
    const float* d_rgb,
    float* d_out_depth,
    float* d_out_rgb,
    uint32_t w,
    uint32_t h,
    bool y_flip,
    cudaStream_t stream) {
    const int n = static_cast<int>(w * h);
    pack_scene_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        d_depth, d_rgb, d_out_depth, d_out_rgb,
        static_cast<int>(w), static_cast<int>(h), y_flip ? 1 : 0);
}

__global__ void pack_tiles_id_kernel(
    const uint32_t* id_atlas,
    uint32_t* out_id,
    int atlas_w,
    int atlas_h,
    int tile_w,
    int tile_h,
    int batch_size,
    int cols,
    int y_flip) {
    const int n = blockIdx.x;
    if (n >= batch_size) return;
    const int ty = n / cols;
    const int tx = n % cols;
    const int ox0 = tx * tile_w;
    const int oy0 = ty * tile_h;
    for (int y = threadIdx.x; y < tile_h; y += blockDim.x) {
        const int sy = y_flip ? (oy0 + tile_h - 1 - y) : (oy0 + y);
        for (int x = 0; x < tile_w; ++x) {
            const int sx = ox0 + x;
            const int si = sy * atlas_w + sx;
            const int di = n * tile_w * tile_h + y * tile_w + x;
            out_id[di] = id_atlas[si];
        }
    }
}

void pack_tiles_instance_id(
    const uint32_t* d_id_atlas,
    uint32_t* d_out_id,
    uint32_t atlas_w,
    uint32_t atlas_h,
    uint32_t tile_w,
    uint32_t tile_h,
    uint32_t batch_size,
    uint32_t cols,
    bool y_flip,
    cudaStream_t stream) {
    pack_tiles_id_kernel<<<batch_size, 256, 0, stream>>>(
        d_id_atlas, d_out_id,
        static_cast<int>(atlas_w), static_cast<int>(atlas_h),
        static_cast<int>(tile_w), static_cast<int>(tile_h),
        static_cast<int>(batch_size), static_cast<int>(cols),
        y_flip ? 1 : 0);
}

}
}
