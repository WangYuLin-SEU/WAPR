// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#include "cuda_distort_context.h"
#include "cuda_device_guard.h"
#include <cstring>
#include <cmath>
#include <string>
#include <stdexcept>

namespace wapr_ogl {

namespace {
void check_cuda(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
    }
}
}

CudaDistortContext::CudaDistortContext(int device) : device_(device) {
    CudaDeviceGuard device_guard(device_, "CudaDistortContext ctor");
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags(CudaDistortContext)");
}

CudaDistortContext::~CudaDistortContext() {
    try {
        CudaDeviceGuard device_guard(device_, "CudaDistortContext dtor");
        if (d_map_x_) cudaFree(d_map_x_);
        if (d_map_y_) cudaFree(d_map_y_);
        if (stream_) cudaStreamDestroy(stream_);
    } catch (...) {
    }
}

void CudaDistortContext::ensure_map(float fx, float fy, float cx, float cy,
                                    const float* dist5, uint32_t width, uint32_t height) {
    CudaDeviceGuard device_guard(device_, "CudaDistortContext::ensure_map");
    if (!dist5 || width == 0 || height == 0 || !std::isfinite(fx) ||
        !std::isfinite(fy) || !std::isfinite(cx) || !std::isfinite(cy) ||
        !(fx > 0.f) || !(fy > 0.f)) {
        throw std::invalid_argument("invalid distortion map intrinsics or dimensions");
    }
    const size_t need = static_cast<size_t>(width) * height;
    if (need > map_cap_) {
        if (d_map_x_) cudaFree(d_map_x_);
        if (d_map_y_) cudaFree(d_map_y_);
        check_cuda(cudaMalloc(&d_map_x_, need * sizeof(float)), "cudaMalloc(distort map x)");
        try {
            check_cuda(cudaMalloc(&d_map_y_, need * sizeof(float)), "cudaMalloc(distort map y)");
        } catch (...) {
            cudaFree(d_map_x_);
            d_map_x_ = nullptr;
            throw;
        }
        map_cap_ = need;
    }
    const bool same =
        width == last_w_ && height == last_h_ &&
        fx == last_fx_ && fy == last_fy_ && cx == last_cx_ && cy == last_cy_ &&
        std::memcmp(dist5, last_dist_, sizeof(last_dist_)) == 0;
    if (same) return;

    cuda_kernels::build_undistort_map(
        fx, fy, cx, cy, dist5, width, height, d_map_x_, d_map_y_, stream_);
    check_cuda(cudaGetLastError(), "build_undistort_map launch");
    check_cuda(cudaStreamSynchronize(stream_), "build_undistort_map synchronize");
    last_fx_ = fx;
    last_fy_ = fy;
    last_cx_ = cx;
    last_cy_ = cy;
    std::memcpy(last_dist_, dist5, sizeof(last_dist_));
    last_w_ = width;
    last_h_ = height;
}

void CudaDistortContext::remap_rgb_depth_host(const float* src_rgb, const float* src_depth,
                                              float* dst_rgb, float* dst_depth,
                                              uint32_t width, uint32_t height) {
    CudaDeviceGuard device_guard(device_, "CudaDistortContext::remap_host");
    if (!src_rgb || !src_depth || !dst_rgb || !dst_depth || width == 0 || height == 0) {
        throw std::invalid_argument("distort host remap requires non-null buffers and dimensions");
    }
    const size_t pix = static_cast<size_t>(width) * height;
    float *d_src_rgb = nullptr, *d_src_depth = nullptr, *d_dst_rgb = nullptr, *d_dst_depth = nullptr;
    check_cuda(cudaMalloc(&d_src_rgb, pix * 3 * sizeof(float)), "cudaMalloc(src rgb)");
    check_cuda(cudaMalloc(&d_src_depth, pix * sizeof(float)), "cudaMalloc(src depth)");
    check_cuda(cudaMalloc(&d_dst_rgb, pix * 3 * sizeof(float)), "cudaMalloc(dst rgb)");
    check_cuda(cudaMalloc(&d_dst_depth, pix * sizeof(float)), "cudaMalloc(dst depth)");
    cudaMemcpyAsync(d_src_rgb, src_rgb, pix * 3 * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_src_depth, src_depth, pix * sizeof(float), cudaMemcpyHostToDevice, stream_);
    remap_rgb_depth_device(d_src_rgb, d_src_depth, d_dst_rgb, d_dst_depth, width, height);
    cudaMemcpyAsync(dst_rgb, d_dst_rgb, pix * 3 * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(dst_depth, d_dst_depth, pix * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    check_cuda(cudaStreamSynchronize(stream_), "distort host synchronize");
    cudaFree(d_src_rgb);
    cudaFree(d_src_depth);
    cudaFree(d_dst_rgb);
    cudaFree(d_dst_depth);
}

void CudaDistortContext::remap_rgb_depth_device(const float* src_rgb, const float* src_depth,
                                                float* dst_rgb, float* dst_depth,
                                                uint32_t width, uint32_t height) {
    CudaDeviceGuard device_guard(device_, "CudaDistortContext::remap_device");
    if (!src_rgb || !src_depth || !dst_rgb || !dst_depth || width == 0 || height == 0) {
        throw std::invalid_argument("distort device remap requires non-null buffers and dimensions");
    }
    if (!d_map_x_ || !d_map_y_ || width != last_w_ || height != last_h_) {
        throw std::runtime_error("distort map missing or dimensions changed; call ensure_map first");
    }
    cuda_kernels::remap_bilinear_rgb(src_rgb, dst_rgb, d_map_x_, d_map_y_, width, height, stream_);
    cuda_kernels::remap_nearest_depth(
        src_depth, dst_depth, d_map_x_, d_map_y_, width, height, stream_);
    check_cuda(cudaGetLastError(), "distort remap launch");
    check_cuda(cudaStreamSynchronize(stream_), "distort remap synchronize");
}

}
