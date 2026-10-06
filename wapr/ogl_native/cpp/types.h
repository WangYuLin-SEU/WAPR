// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace wapr_ogl {

constexpr uint32_t kDepthFuncLess = 0x0201;
constexpr uint32_t kDepthFuncGreater = 0x0204;

enum class DataLayout : uint32_t {
    NHWC = 0,
    NCHW = 1,
};

enum class RenderMode : uint32_t {
    Tile = 0,
    Scene = 1,
};


enum class ProjectionMode : uint32_t {
    ClipAffine = 0,
    PinholeCrop = 1,
};

struct CameraIntrinsics {
    float fx, fy, cx, cy;
    uint32_t width, height;
    float znear = 0.001f;
    float zfar = 100.f;
    float dist[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    uint8_t use_distortion = 0;
};

struct RenderInstance {
    uint32_t mesh_id = 0;
    float T_cam_obj[16] = {};
    float view_warp[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    uint8_t has_view_warp = 0;
    
    float clip_affine4[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    uint8_t has_clip_affine = 0;
    float mvp_pre[16] = {};
    float model_cam_pre[16] = {};
    uint8_t has_precomputed_mvp = 0;
    uint32_t depth_func = kDepthFuncLess;
    uint32_t tile_slot = 0;
    
    float bbox_xywh[4] = {0, 0, 0, 0};
    uint8_t has_bbox = 0;
    
    float fx = 0.f, fy = 0.f, cx = 0.f, cy = 0.f;
    uint8_t has_K = 0;
    
    float metallic = 0.f;
    float roughness = 0.5f;
    uint8_t has_material = 0;
    
    uint32_t instance_id = 0;
};

struct TileAtlasLayout {
    uint32_t cols = 1;
    uint32_t rows = 1;
    uint32_t atlas_w = 0;
    uint32_t atlas_h = 0;
};

struct RenderOutputSpec {
    bool rgb = true;
    bool metric_depth = true;
    bool normal = false;
    bool object_coord = false;
    uint32_t tile_w = 160;
    uint32_t tile_h = 160;
    RenderMode mode = RenderMode::Tile;
    ProjectionMode projection = ProjectionMode::ClipAffine;
    DataLayout layout = DataLayout::NCHW;
    bool y_flip = true;
    uint32_t max_atlas_w = 0;
    uint32_t max_atlas_h = 0;
    bool use_scissor = false;
    
    bool indexed_viewport_scissor = false;
    
    bool viewport_per_tile = false;
    bool scene_batch = false;
    
    bool layered_tiles = false;
    
    bool output_instance_id = false;
    
    uint32_t msaa_samples = 0;
    
    float sample_shading = 0.f;
    
    bool persistent_buffers = false;
};


struct GlCaps {
    int max_viewports = 1;
    bool multi_draw_indirect = false;
    bool shader_draw_parameters = false;
    bool viewport_array = false;
    bool shader_viewport_layer_array = false;
    bool buffer_storage = false;
    bool map_persistent = false;
    bool sync = false;
};

struct MeshUpload {
    std::string name;
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<uint32_t> indices;
    std::vector<float> vertex_colors;
    std::vector<float> uvs;
    std::vector<uint32_t> uv_indices;
    std::vector<uint8_t> texture_rgb;
    uint32_t tex_w = 0, tex_h = 0;
    uint8_t uv_per_corner = 0;
    uint8_t force_flat_normal = 0;
};

struct MeshTableEntry {
    uint32_t vertex_offset;
    uint32_t vertex_count = 0;
    uint32_t index_offset;
    uint32_t index_count;
    uint32_t tex_layer;
    uint32_t has_texture;
    uint32_t force_flat_normal;
    uint32_t valid = 1;
    float bounds_min[3] = {0.f, 0.f, 0.f};
    float bounds_max[3] = {0.f, 0.f, 0.f};
    uint32_t has_bounds = 0;
};

inline void adapt_clip_planes_for_scene(CameraIntrinsics& cam,
                                        const std::vector<RenderInstance>& instances,
                                        const std::vector<MeshTableEntry>& mesh_table) {
    float zmin = 1e30f;
    float zmax = 0.f;
    bool any = false;
    for (const auto& inst : instances) {
        if (inst.mesh_id >= mesh_table.size() || !mesh_table[inst.mesh_id].valid ||
            !mesh_table[inst.mesh_id].has_bounds) {
            continue;
        }
        const auto& me = mesh_table[inst.mesh_id];
        for (int c = 0; c < 8; ++c) {
            const float lx = (c & 1) ? me.bounds_max[0] : me.bounds_min[0];
            const float ly = (c & 2) ? me.bounds_max[1] : me.bounds_min[1];
            const float lz = (c & 4) ? me.bounds_max[2] : me.bounds_min[2];
            const float* T = inst.T_cam_obj;
            const float z = T[8] * lx + T[9] * ly + T[10] * lz + T[11];
            if (z <= 1e-3f) {
                continue;
            }
            zmin = std::min(zmin, z);
            zmax = std::max(zmax, z);
            any = true;
        }
    }
    if (any && zmin < zmax) {
        cam.znear = std::max(1e-3f, zmin * 0.8f);
        cam.zfar = std::max(cam.znear * 10.f, zmax * 1.2f);
    }
}

struct LightParams {
    float ambient = 0.8f;
    float diffuse = 0.2f;
    float light_dir[3] = {0.f, 0.f, 1.f};
    
    int32_t use_pbr = 0;
    float ambient_color[3] = {0.15f, 0.15f, 0.15f};
    float point_pos[3] = {0.f, 0.f, 0.f};
    float point_color[3] = {1.f, 1.f, 1.f};
    float point_intensity = 0.f;
    float dir_color[3] = {1.f, 1.f, 1.f};
    float dir_intensity = 0.f;
};


struct alignas(16) InstanceRecordGPU {
    float mvp[16];
    float model_cam[16];
    float clip_affine[16];
    uint32_t mesh_id = 0;
    uint32_t tile_x = 0;
    uint32_t tile_y = 0;
    uint32_t depth_func = kDepthFuncLess;
    int32_t tex_layer = -1;
    uint32_t force_flat_normal = 0;
    float metallic = 0.f;
    float roughness = 0.5f;
    int32_t tile_slot = 0;
    int32_t instance_id = 0;
    float fx = 0.f;
    float fy = 0.f;
    float cx = 0.f;
    float cy = 0.f;
    int32_t has_clip_affine = 0;
    int32_t _pad1 = 0;
};
static_assert(alignof(InstanceRecordGPU) == 16, "InstanceRecordGPU std430 alignment");
static_assert(sizeof(InstanceRecordGPU) == 256, "InstanceRecordGPU std430 stride");
static_assert(offsetof(InstanceRecordGPU, mesh_id) == 192, "InstanceRecordGPU mesh offset");
static_assert(offsetof(InstanceRecordGPU, tile_slot) == 224, "InstanceRecordGPU tile offset");
static_assert(offsetof(InstanceRecordGPU, has_clip_affine) == 248,
              "InstanceRecordGPU clip-affine flag offset");


struct LightRecordGPU {
    float ambient = 0.8f;
    float diffuse = 0.2f;
    int32_t use_pbr = 0;
    int32_t _pad0 = 0;
    float light_dir[4] = {0.f, 0.f, 1.f, 0.f};
    float ambient_color[4] = {0.15f, 0.15f, 0.15f, 0.f};
    float point_pos[4] = {0.f, 0.f, 0.f, 0.f};
    float point_color[4] = {1.f, 1.f, 1.f, 0.f};
    float dir_color[4] = {1.f, 1.f, 1.f, 0.f};
};

inline LightRecordGPU light_record_from_params(const LightParams& lp) {
    LightRecordGPU r;
    r.ambient = lp.ambient;
    r.diffuse = lp.diffuse;
    r.use_pbr = lp.use_pbr;
    r.light_dir[0] = lp.light_dir[0];
    r.light_dir[1] = lp.light_dir[1];
    r.light_dir[2] = lp.light_dir[2];
    r.ambient_color[0] = lp.ambient_color[0];
    r.ambient_color[1] = lp.ambient_color[1];
    r.ambient_color[2] = lp.ambient_color[2];
    r.point_pos[0] = lp.point_pos[0];
    r.point_pos[1] = lp.point_pos[1];
    r.point_pos[2] = lp.point_pos[2];
    r.point_color[0] = lp.point_color[0];
    r.point_color[1] = lp.point_color[1];
    r.point_color[2] = lp.point_color[2];
    r.point_color[3] = lp.point_intensity;
    r.dir_color[0] = lp.dir_color[0];
    r.dir_color[1] = lp.dir_color[1];
    r.dir_color[2] = lp.dir_color[2];
    r.dir_color[3] = lp.dir_intensity;
    return r;
}

struct DrawElementsIndirectCommand {
    uint32_t count = 0;
    uint32_t instanceCount = 1;
    uint32_t firstIndex = 0;
    int32_t baseVertex = 0;
    uint32_t baseInstance = 0;
};

struct CudaBufferSet {
    uint32_t batch_size = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    void* rgb = nullptr;
    void* depth = nullptr;
    void* normal = nullptr;
    void* instance_id = nullptr;
    bool on_device = true;
};

}
