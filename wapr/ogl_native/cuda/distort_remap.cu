// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#include "distort_remap.cuh"
#include <cmath>

namespace wapr_ogl {
namespace cuda_kernels {

__device__ inline void opencv5_forward(float x, float y,
                                       float k1, float k2, float p1, float p2, float k3,
                                       float& xd, float& yd) {
    const float r2 = x * x + y * y;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;
    const float cdist = 1.f + k1 * r2 + k2 * r4 + k3 * r6;
    xd = x * cdist + 2.f * p1 * x * y + p2 * (r2 + 2.f * x * x);
    yd = y * cdist + p1 * (r2 + 2.f * y * y) + 2.f * p2 * x * y;
}

__device__ inline bool undistort_norm_newton(float xd, float yd,
                                             float k1, float k2, float p1, float p2, float k3,
                                             float& x, float& y) {
    x = xd;
    y = yd;
    for (int it = 0; it < 16; ++it) {
        float fx, fy;
        opencv5_forward(x, y, k1, k2, p1, p2, k3, fx, fy);
        const float ex = fx - xd;
        const float ey = fy - yd;
        if (fabsf(ex) < 1e-6f && fabsf(ey) < 1e-6f) return true;
        const float r2 = x * x + y * y;
        const float r4 = r2 * r2;
        const float r6 = r4 * r2;
        const float cdist = 1.f + k1 * r2 + k2 * r4 + k3 * r6;
        const float dfx_dx = cdist + 2.f * k1 * x * x + 4.f * k2 * r2 * x * x + 6.f * k3 * r4 * x * x
                             + 2.f * p1 * y + 6.f * p2 * x;
        const float dfx_dy = 2.f * k1 * x * y + 4.f * k2 * r2 * x * y + 6.f * k3 * r4 * x * y
                             + 2.f * p1 * x + 2.f * p2 * y;
        const float dfy_dx = 2.f * k1 * x * y + 4.f * k2 * r2 * x * y + 6.f * k3 * r4 * x * y
                             + 2.f * p1 * x + 2.f * p2 * y;
        const float dfy_dy = cdist + 2.f * k1 * y * y + 4.f * k2 * r2 * y * y + 6.f * k3 * r4 * y * y
                             + 6.f * p1 * y + 2.f * p2 * x;
        const float det = dfx_dx * dfy_dy - dfx_dy * dfy_dx;
        if (fabsf(det) < 1e-8f) break;
        const float inv_det = 1.f / det;
        x -= inv_det * (dfy_dy * ex - dfx_dy * ey);
        y -= inv_det * (-dfy_dx * ex + dfx_dx * ey);
    }
    float fx_final, fy_final;
    opencv5_forward(x, y, k1, k2, p1, p2, k3, fx_final, fy_final);
    return isfinite(x) && isfinite(y) &&
           fabsf(fx_final - xd) < 1e-6f && fabsf(fy_final - yd) < 1e-6f;
}

__global__ void build_undistort_map_kernel(
    float fx, float fy, float cx, float cy,
    float k1, float k2, float p1, float p2, float k3,
    int width, int height,
    float* map_x, float* map_y) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= height || j >= width) return;
    const float u = static_cast<float>(j);
    const float v = static_cast<float>(i);
    const float xd = (u - cx) / fx;
    const float yd = (v - cy) / fy;
    float x, y;
    if (!undistort_norm_newton(xd, yd, k1, k2, p1, p2, k3, x, y)) {
        const int idx = i * width + j;
        map_x[idx] = -1.f;
        map_y[idx] = -1.f;
        return;
    }
    const int idx = i * width + j;
    map_x[idx] = fx * x + cx;
    map_y[idx] = fy * y + cy;
}

__device__ inline float sample_bilinear(const float* img, int w, int h, float u, float v) {
    if (u < 0.f || v < 0.f || u >= static_cast<float>(w - 1) || v >= static_cast<float>(h - 1)) {
        return 0.f;
    }
    const int x0 = static_cast<int>(floorf(u));
    const int y0 = static_cast<int>(floorf(v));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float tx = u - static_cast<float>(x0);
    const float ty = v - static_cast<float>(y0);
    const float v00 = img[y0 * w + x0];
    const float v10 = img[y0 * w + x1];
    const float v01 = img[y1 * w + x0];
    const float v11 = img[y1 * w + x1];
    const float a = v00 * (1.f - tx) + v10 * tx;
    const float b = v01 * (1.f - tx) + v11 * tx;
    return a * (1.f - ty) + b * ty;
}

__global__ void remap_bilinear_rgb_kernel(
    const float* src, float* dst,
    const float* map_x, const float* map_y,
    int width, int height) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= height || j >= width) return;
    const int idx = i * width + j;
    const float u = map_x[idx];
    const float v = map_y[idx];
    const int di = idx * 3;
    dst[di + 0] = sample_bilinear(src, width, height, u, v);
}

__global__ void remap_bilinear_rgb_interleaved_kernel(
    const float* src, float* dst,
    const float* map_x, const float* map_y,
    int width, int height) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= height || j >= width) return;
    const int idx = i * width + j;
    const float u = map_x[idx];
    const float v = map_y[idx];
    if (u < 0.f || v < 0.f || u >= static_cast<float>(width - 1) || v >= static_cast<float>(height - 1)) {
        dst[idx * 3 + 0] = dst[idx * 3 + 1] = dst[idx * 3 + 2] = 0.f;
        return;
    }
    const int x0 = static_cast<int>(floorf(u));
    const int y0 = static_cast<int>(floorf(v));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float tx = u - static_cast<float>(x0);
    const float ty = v - static_cast<float>(y0);
    for (int c = 0; c < 3; ++c) {
        const float v00 = src[(y0 * width + x0) * 3 + c];
        const float v10 = src[(y0 * width + x1) * 3 + c];
        const float v01 = src[(y1 * width + x0) * 3 + c];
        const float v11 = src[(y1 * width + x1) * 3 + c];
        const float a = v00 * (1.f - tx) + v10 * tx;
        const float b = v01 * (1.f - tx) + v11 * tx;
        dst[idx * 3 + c] = a * (1.f - ty) + b * ty;
    }
}

__global__ void remap_nearest_depth_kernel(
    const float* src, float* dst,
    const float* map_x, const float* map_y,
    int width, int height) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= height || j >= width) return;
    const int idx = i * width + j;
    const float u = map_x[idx];
    const float v = map_y[idx];
    const int x = static_cast<int>(roundf(u));
    const int y = static_cast<int>(roundf(v));
    if (x < 0 || y < 0 || x >= width || y >= height) {
        dst[idx] = 0.f;
        return;
    }
    dst[idx] = src[y * width + x];
}

void build_undistort_map(
    float fx, float fy, float cx, float cy,
    const float* dist5,
    uint32_t width, uint32_t height,
    float* map_x, float* map_y,
    cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    build_undistort_map_kernel<<<grid, block, 0, stream>>>(
        fx, fy, cx, cy,
        dist5[0], dist5[1], dist5[2], dist5[3], dist5[4],
        static_cast<int>(width), static_cast<int>(height),
        map_x, map_y);
}

void remap_bilinear_rgb(
    const float* src, float* dst,
    const float* map_x, const float* map_y,
    uint32_t width, uint32_t height,
    cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    remap_bilinear_rgb_interleaved_kernel<<<grid, block, 0, stream>>>(
        src, dst, map_x, map_y, static_cast<int>(width), static_cast<int>(height));
}

void remap_nearest_depth(
    const float* src, float* dst,
    const float* map_x, const float* map_y,
    uint32_t width, uint32_t height,
    cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    remap_nearest_depth_kernel<<<grid, block, 0, stream>>>(
        src, dst, map_x, map_y, static_cast<int>(width), static_cast<int>(height));
}

}
}
