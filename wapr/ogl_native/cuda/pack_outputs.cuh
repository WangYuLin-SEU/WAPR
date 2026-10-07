#pragma once

#include <cstdint>

namespace wapr_ogl {
namespace cuda_kernels {

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
    cudaStream_t stream);

void pack_scene_rgb_depth(
    const float* d_depth,
    const float* d_rgb,
    float* d_out_depth,
    float* d_out_rgb,
    uint32_t w,
    uint32_t h,
    bool y_flip,
    cudaStream_t stream);

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
    cudaStream_t stream);

}
}
