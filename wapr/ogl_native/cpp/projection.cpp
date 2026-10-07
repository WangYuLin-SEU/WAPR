// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#include "projection.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <climits>

namespace wapr_ogl {

std::array<float, 16> projection_matrix_from_intrinsics(
    const CameraIntrinsics& K, const char* window_coords) {
    const float x0 = 0.f, y0 = 0.f;
    const float w = static_cast<float>(K.width);
    const float h = static_cast<float>(K.height);
    const float nc = K.znear, fc = K.zfar;
    const float depth = fc - nc;
    const float q = -(fc + nc) / depth;
    const float qn = -2.f * (fc * nc) / depth;

    std::array<float, 16> proj{};
    if (std::strcmp(window_coords, "y_down") == 0) {
        proj = {
            2.f * K.fx / w, -2.f * 0.f / w, (-2.f * K.cx + w + 2.f * x0) / w, 0.f,
            0.f, 2.f * K.fy / h, (2.f * K.cy - h + 2.f * y0) / h, 0.f,
            0.f, 0.f, q, qn,
            0.f, 0.f, -1.f, 0.f};
    } else {
        proj = {
            2.f * K.fx / w, -2.f * 0.f / w, (-2.f * K.cx + w + 2.f * x0) / w, 0.f,
            0.f, -2.f * K.fy / h, (-2.f * K.cy + h + 2.f * y0) / h, 0.f,
            0.f, 0.f, q, qn,
            0.f, 0.f, -1.f, 0.f};
    }
    return proj;
}

void mat4_transpose(const float* in, float* out) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[j * 4 + i] = in[i * 4 + j];
        }
    }
}

void mat4_mul(const float* A, const float* B, float* out) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[i * 4 + j] = 0.f;
            for (int k = 0; k < 4; ++k) {
                out[i * 4 + j] += A[i * 4 + k] * B[k * 4 + j];
            }
        }
    }
}

void transform_point(const float* T, float x, float y, float z, float* out) {
    out[0] = T[0] * x + T[1] * y + T[2] * z + T[3];
    out[1] = T[4] * x + T[5] * y + T[6] * z + T[7];
    out[2] = T[8] * x + T[9] * y + T[10] * z + T[11];
}

void clip_affine_from_window_rect(float left, float bottom, float right, float top,
                                  float frame_w, float frame_h, float* mat4_row_out) {
    const float rw = std::max(right - left, 1e-6f);
    const float rh = std::max(top - bottom, 1e-6f);
    const float sx = frame_w / rw;
    const float sy = frame_h / rh;
    const float tx = (frame_w - right - left) / rw;
    const float ty = (frame_h - top - bottom) / rh;
    const float m[16] = {
        sx, 0.f, 0.f, 0.f,
        0.f, sy, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        tx, ty, 0.f, 1.f};
    std::memcpy(mat4_row_out, m, sizeof(m));
}

CameraIntrinsics pinhole_intrinsics_for_crop(const CameraIntrinsics& cam,
                                             float bx, float by, float bw, float bh,
                                             uint32_t out_w, uint32_t out_h) {
    CameraIntrinsics out = cam;
    if (bw <= 1e-3f || bh <= 1e-3f) return out;
    const float ax = static_cast<float>(out_w) / bw;
    const float ay = static_cast<float>(out_h) / bh;
    out.fx = cam.fx * ax;
    out.fy = cam.fy * ay;
    out.cx = cam.cx * ax - bx * ax;
    out.cy = cam.cy * ay - by * ay;
    out.width = out_w;
    out.height = out_h;
    return out;
}

void build_mvp(const CameraIntrinsics& cam,
               const float* T_cam_obj,
               const float* clip_affine4_row,
               bool apply_clip_affine,
               float* mvp_out,
               float* model_cam_out) {
    auto glcam = glcam_in_cvcam();
    float T_glcam_obj[16];
    mat4_mul(glcam.data(), T_cam_obj, T_glcam_obj);

    auto proj = projection_matrix_from_intrinsics(cam);
    mat4_mul(proj.data(), T_glcam_obj, mvp_out);

    if (model_cam_out) {
        std::memcpy(model_cam_out, T_cam_obj, 16 * sizeof(float));
    }

    if (apply_clip_affine && clip_affine4_row) {
        float aT[16];
        mat4_transpose(clip_affine4_row, aT);
        float tmp[16];
        mat4_mul(aT, mvp_out, tmp);
        std::memcpy(mvp_out, tmp, 16 * sizeof(float));
    }
}

void tile_atlas_place_row(float* mvp_row,
                          uint32_t tile_x, uint32_t tile_y,
                          uint32_t tile_w, uint32_t tile_h,
                          uint32_t atlas_w, uint32_t atlas_h) {
    if (atlas_w == 0 || atlas_h == 0) return;
    const float sx = static_cast<float>(tile_w) / static_cast<float>(atlas_w);
    const float sy = static_cast<float>(tile_h) / static_cast<float>(atlas_h);
    const float tx = (2.f * static_cast<float>(tile_x) + static_cast<float>(tile_w))
                     / static_cast<float>(atlas_w) - 1.f;
    const float ty = (2.f * static_cast<float>(tile_y) + static_cast<float>(tile_h))
                     / static_cast<float>(atlas_h) - 1.f;
    const float place[16] = {
        sx, 0.f, 0.f, 0.f,
        0.f, sy, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        tx, ty, 0.f, 1.f};
    float placeT[16];
    mat4_transpose(place, placeT);
    float tmp[16];
    mat4_mul(placeT, mvp_row, tmp);
    std::memcpy(mvp_row, tmp, 16 * sizeof(float));
}

void projection_depth_coeffs(const CameraIntrinsics& cam, float& q, float& qn) {
    const float depth = cam.zfar - cam.znear;
    q = -(cam.zfar + cam.znear) / depth;
    qn = -2.f * cam.zfar * cam.znear / depth;
}

static uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

TileAtlasLayout compute_tile_atlas_layout(uint32_t n,
                                          uint32_t tile_w,
                                          uint32_t tile_h,
                                          uint32_t max_atlas_w,
                                          uint32_t max_atlas_h) {
    TileAtlasLayout layout{};
    if (n == 0) {
        layout.atlas_w = tile_w;
        layout.atlas_h = tile_h;
        return layout;
    }
    layout.cols = std::max(1u, static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<double>(n)))));
    layout.rows = ceil_div_u32(n, layout.cols);
    if (max_atlas_w > 0 && max_atlas_h > 0 && tile_w > 0 && tile_h > 0) {
        const uint32_t max_cols = std::max(1u, max_atlas_w / tile_w);
        const uint32_t max_rows = std::max(1u, max_atlas_h / tile_h);
        if (layout.cols > max_cols) {
            layout.cols = max_cols;
        }
        layout.rows = ceil_div_u32(n, layout.cols);
        if (layout.rows > max_rows) {
            layout.rows = max_rows;
        }
    }
    layout.atlas_w = layout.cols * tile_w;
    layout.atlas_h = layout.rows * tile_h;
    return layout;
}

uint32_t max_tiles_per_atlas_chunk(uint32_t tile_w,
                                   uint32_t tile_h,
                                   uint32_t max_atlas_w,
                                   uint32_t max_atlas_h) {
    if (tile_w == 0 || tile_h == 0) {
        return 1u;
    }
    if (max_atlas_w == 0 || max_atlas_h == 0) {
        return UINT32_MAX;
    }
    const uint32_t nx = max_atlas_w / tile_w;
    const uint32_t ny = max_atlas_h / tile_h;
    if (nx == 0 || ny == 0) {
        return 1u;
    }
    return nx * ny;
}

}
