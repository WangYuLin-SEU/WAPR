// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace wapr_ogl {


class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(int device, const char* where = "CUDA device guard") {
        cudaError_t err = cudaGetDevice(&previous_);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(where) + " cudaGetDevice: " +
                                     cudaGetErrorString(err));
        }
        if (previous_ != device) {
            err = cudaSetDevice(device);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string(where) + " cudaSetDevice: " +
                                         cudaGetErrorString(err));
            }
            restore_ = true;
        }
    }

    ~CudaDeviceGuard() {
        if (restore_) cudaSetDevice(previous_);
    }

    CudaDeviceGuard(const CudaDeviceGuard&) = delete;
    CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

private:
    int previous_ = 0;
    bool restore_ = false;
};

}
