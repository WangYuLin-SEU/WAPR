// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#pragma once

#include "distort_remap.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace wapr_ogl {

class CudaDistortContext {
public:
    explicit CudaDistortContext(int device = 0);
    ~CudaDistortContext();

    void ensure_map(float fx, float fy, float cx, float cy,
                    const float* dist5, uint32_t width, uint32_t height);

    void remap_rgb_depth_host(const float* src_rgb, const float* src_depth,
                              float* dst_rgb, float* dst_depth,
                              uint32_t width, uint32_t height);

    void remap_rgb_depth_device(const float* src_rgb, const float* src_depth,
                                float* dst_rgb, float* dst_depth,
                                uint32_t width, uint32_t height);

    const float* device_map_x() const { return d_map_x_; }
    const float* device_map_y() const { return d_map_y_; }

private:
    int device_ = 0;
    cudaStream_t stream_ = nullptr;
    float* d_map_x_ = nullptr;
    float* d_map_y_ = nullptr;
    size_t map_cap_ = 0;
    float last_fx_ = 0.f, last_fy_ = 0.f, last_cx_ = 0.f, last_cy_ = 0.f;
    float last_dist_[5] = {};
    uint32_t last_w_ = 0, last_h_ = 0;
};

}
