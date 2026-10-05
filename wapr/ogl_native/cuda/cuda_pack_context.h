// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#pragma once

#include "types.h"
#include <cuda_runtime.h>
#include <vector>
#include <GL/gl.h>

namespace wapr_ogl {

class CudaPackContext {
public:
    explicit CudaPackContext(int device = 0);
    ~CudaPackContext();

    void register_gl_texture(GLuint tex, uint32_t w, uint32_t h, int channel_index,
                             bool texture_array = false);
    
    void unregister_gl_textures();
    CudaBufferSet pack_tiles(const RenderOutputSpec& spec,
                             uint32_t atlas_w, uint32_t atlas_h,
                             uint32_t batch_size,
                             uint32_t tile_w, uint32_t tile_h,
                             uint32_t atlas_cols = 0,
                             uint32_t out_tile_offset = 0);
    
    CudaBufferSet pack_layers(const RenderOutputSpec& spec,
                              uint32_t batch_size,
                              uint32_t tile_w, uint32_t tile_h,
                              uint32_t out_tile_offset = 0);

    CudaBufferSet pack_scene(const RenderOutputSpec& spec,
                             uint32_t w, uint32_t h);

    CudaBufferSet pack_tiles_from_cpu(const RenderOutputSpec& spec,
                                      const float* depth_atlas,
                                      const float* rgb_rgba_atlas,
                                      uint32_t atlas_w, uint32_t atlas_h,
                                      uint32_t batch_size,
                                      uint32_t tile_w, uint32_t tile_h,
                                      uint32_t atlas_cols = 0);

    CudaBufferSet pack_scene_from_cpu(const RenderOutputSpec& spec,
                                      const float* depth_atlas,
                                      const float* rgb_rgba_atlas,
                                      uint32_t w, uint32_t h);

    void copy_tiles_to_offset(void* dst_depth, void* dst_rgb,
                              uint32_t tile_offset, uint32_t tile_count) const;

    void synchronize();

    void ensure_batch_output(uint32_t batch, uint32_t tw, uint32_t th, const RenderOutputSpec& spec);

    void* device_rgb() const { return last_has_rgb_ ? d_rgb_ : nullptr; }
    void* device_depth() const { return last_has_depth_ ? d_depth_ : nullptr; }
    void* device_normal() const { return last_has_normal_ ? d_normal_ : nullptr; }
    void* device_instance_id() const { return last_has_instance_id_ ? d_instance_id_ : nullptr; }
    uint32_t last_batch() const { return last_batch_; }
    uint32_t last_width() const { return last_w_; }
    uint32_t last_height() const { return last_h_; }
    void set_last_pack_dims(uint32_t batch, uint32_t w, uint32_t h) {
        last_batch_ = batch;
        last_w_ = w;
        last_h_ = h;
    }

private:
    int device_ = 0;
    uint32_t last_batch_ = 0;
    uint32_t last_w_ = 0;
    uint32_t last_h_ = 0;
    bool last_has_rgb_ = false;
    bool last_has_depth_ = false;
    bool last_has_normal_ = false;
    bool last_has_instance_id_ = false;
    cudaStream_t stream_ = nullptr;

    struct RegisteredTex {
        cudaGraphicsResource* resource = nullptr;
        GLuint tex_id = 0;
        uint32_t w = 0, h = 0;
        bool is_array = false;
    };
    RegisteredTex rgb_tex_{};
    RegisteredTex depth_tex_{};
    RegisteredTex normal_tex_{};
    RegisteredTex id_tex_{};

    void* d_rgb_ = nullptr;
    void* d_depth_ = nullptr;
    void* d_normal_ = nullptr;
    void* d_coord_ = nullptr;
    void* d_instance_id_ = nullptr;
    size_t cap_rgb_ = 0;
    size_t cap_depth_ = 0;
    size_t cap_normal_ = 0;
    size_t cap_instance_id_ = 0;

    float* d_depth_stage_ = nullptr;
    float* d_rgb_stage_ = nullptr;
    float* d_normal_stage_ = nullptr;
    uint32_t* d_id_stage_ = nullptr;
    size_t cap_depth_stage_ = 0;
    size_t cap_rgb_stage_ = 0;
    size_t cap_normal_stage_ = 0;
    size_t cap_id_stage_ = 0;

    void ensure_capacity(uint32_t batch, uint32_t tw, uint32_t th, const RenderOutputSpec& spec);
    void ensure_stage(uint32_t atlas_w, uint32_t atlas_h,
                      bool need_depth, bool need_rgb,
                      bool need_normal, bool need_id=false);
};

}
