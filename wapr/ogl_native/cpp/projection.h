// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#pragma once

#include "types.h"
#include <array>
#include <cstdint>
#include <vector>

namespace wapr_ogl {

inline std::array<float, 16> glcam_in_cvcam() {
    return {1, 0, 0, 0,
            0, -1, 0, 0,
            0, 0, -1, 0,
            0, 0, 0, 1};
}

std::array<float, 16> projection_matrix_from_intrinsics(
    const CameraIntrinsics& K, const char* window_coords = "y_down");

void mat4_transpose(const float* in, float* out);
void mat4_mul(const float* A, const float* B, float* out);
void transform_point(const float* T, float x, float y, float z, float* out);


void clip_affine_from_window_rect(float left, float bottom, float right, float top,
                                  float frame_w, float frame_h, float* mat4_row_out);


CameraIntrinsics pinhole_intrinsics_for_crop(const CameraIntrinsics& cam,
                                             float bx, float by, float bw, float bh,
                                             uint32_t out_w, uint32_t out_h);

void build_mvp(const CameraIntrinsics& cam,
               const float* T_cam_obj,
               const float* clip_affine4_row,
               bool apply_clip_affine,
               float* mvp_out,
               float* model_cam_out);


void tile_atlas_place_row(float* mvp_row,
                          uint32_t tile_x, uint32_t tile_y,
                          uint32_t tile_w, uint32_t tile_h,
                          uint32_t atlas_w, uint32_t atlas_h);

void projection_depth_coeffs(const CameraIntrinsics& cam, float& q, float& qn);

TileAtlasLayout compute_tile_atlas_layout(uint32_t n,
                                          uint32_t tile_w,
                                          uint32_t tile_h,
                                          uint32_t max_atlas_w,
                                          uint32_t max_atlas_h);

uint32_t max_tiles_per_atlas_chunk(uint32_t tile_w,
                                   uint32_t tile_h,
                                   uint32_t max_atlas_w,
                                   uint32_t max_atlas_h);

inline uint32_t max_scene_slot(const std::vector<RenderInstance>& instances) {
    uint32_t m = 0;
    for (const auto& inst : instances) {
        m = std::max(m, inst.tile_slot);
    }
    return m;
}

inline bool scene_batch_active(const RenderOutputSpec& spec,
                               const std::vector<RenderInstance>& instances) {
    return spec.mode == RenderMode::Scene &&
           (spec.scene_batch || max_scene_slot(instances) > 0);
}

inline uint32_t scene_batch_count(const std::vector<RenderInstance>& instances) {
    return max_scene_slot(instances) + 1;
}


inline uint32_t tile_batch_count(const std::vector<RenderInstance>& instances) {
    return max_scene_slot(instances) + 1;
}

}
