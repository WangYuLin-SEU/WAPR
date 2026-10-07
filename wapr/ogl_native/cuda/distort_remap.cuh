#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace wapr_ogl {
namespace cuda_kernels {

void build_undistort_map(
    float fx, float fy, float cx, float cy,
    const float* dist5,
    uint32_t width, uint32_t height,
    float* map_x, float* map_y,
    cudaStream_t stream);

void remap_bilinear_rgb(
    const float* src, float* dst,
    const float* map_x, const float* map_y,
    uint32_t width, uint32_t height,
    cudaStream_t stream);

void remap_nearest_depth(
    const float* src, float* dst,
    const float* map_x, const float* map_y,
    uint32_t width, uint32_t height,
    cudaStream_t stream);

}
}
