// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#pragma once

#include "types.h"
#include <cuda_runtime.h>

namespace wapr_ogl {

// Per-row geometry metadata; poses and boxes remain on CUDA.
// 每行的几何元数据；位姿和框始终保留在 CUDA。
struct TileMeshData {
    uint32_t mesh_id;
    int32_t tex_layer;
    uint32_t force_flat_normal;
    float bmin[3];
    float bmax[3];
    uint32_t has_bounds;
};

// Write one indexed or layered tile record per pose. Poses are row-major 4x4, meters.
// bboxes are left, top, right, bottom in full-frame pixels, y down.
// 每个位姿写一条分块记录。位姿是行主序 4x4，米。
// bbox 是整幅图像素的 left, top, right, bottom，y 向下。
void fill_indexed_tile_instances(const float* poses,
                                 const float* bboxes,
                                 InstanceRecordGPU* records,
                                 uint32_t count,
                                 float fx, float fy, float cx, float cy,
                                 float frame_w, float frame_h,
                                 uint32_t tile_w, uint32_t tile_h, uint32_t cols,
                                 const TileMeshData* meshes,
                                 uint32_t layered,
                                 cudaStream_t stream);

}
