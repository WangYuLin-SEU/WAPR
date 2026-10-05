// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#pragma once

#include "asset_manager.h"
#include "egl_context.h"
#include "types.h"
#include <functional>
#include <memory>
#include <vector>

namespace wapr_ogl {

class CudaPackContext;

class BatchRenderer {
public:
    BatchRenderer(EglContext& egl, AssetManager& assets, const std::string& shader_dir);
    ~BatchRenderer() noexcept;

    
    void set_pre_fbo_recreate(std::function<void()> cb) { pre_fbo_recreate_ = std::move(cb); }

    void render(const std::vector<RenderInstance>& instances,
                const CameraIntrinsics& cam,
                const RenderOutputSpec& spec,
                const LightParams& light,
                const std::vector<LightParams>* tile_lights = nullptr);

    // Poses (N, 16) and bboxes (N, 4) stay on the CUDA stream. The next render consumes them.
    // 位姿 (N, 16) 和框 (N, 4) 留在这条 CUDA 流上。下一次 render 用完即弃。
    void arm_device_tiles(const float* poses, const float* bboxes, uint32_t count, void* stream);
    void disarm_device_tiles();
    void set_cuda_device(int device);

    uint32_t atlas_width() const { return atlas_w_; }
    uint32_t atlas_height() const { return atlas_h_; }
    uint32_t atlas_cols() const { return atlas_cols_; }

    GLuint fbo() const { return fbo_; }
    GLuint tex_depth() const { return tex_depth_; }
    GLuint tex_rgb() const { return tex_rgb_; }
    GLuint tex_normal() const { return tex_normal_; }
    GLuint tex_coord() const { return tex_coord_; }
    GLuint tex_instance_id() const { return tex_instance_id_; }
    const GlCaps& gl_caps() const { return caps_; }
    uint32_t last_draw_api_calls() const { return last_draw_api_calls_; }

    void readback_rgb(std::vector<uint8_t>& out) const;
    void readback_depth(std::vector<float>& out) const;
    void readback_depth_f32(std::vector<float>& out) const;
    void readback_rgb_rgba_f32(std::vector<float>& out) const;

private:
    EglContext& egl_;
    AssetManager& assets_;
    GLFunctions& gl_;
    GLuint program_ = 0;
    GLuint fbo_ = 0;
    GLuint fbo_alt_ = 0;
    uint32_t fbo_ping_ = 0;
    GLuint tex_depth_ = 0;
    GLuint tex_rgb_ = 0;
    GLuint tex_normal_ = 0;
    GLuint tex_coord_ = 0;
    GLuint tex_instance_id_ = 0;
    GLuint tex_hw_depth_ = 0;
    GLuint depth_rb_ = 0;
    GLuint fbo_ms_ = 0;
    GLuint rb_ms_depth_ = 0;
    GLuint rb_ms_rgb_ = 0;
    GLuint rb_ms_normal_ = 0;
    GLuint rb_ms_coord_ = 0;
    GLuint rb_ms_instance_id_ = 0;
    GLuint rb_ms_hw_depth_ = 0;
    uint32_t fbo_msaa_samples_ = 0;
    bool fbo_output_instance_id_ = false;
    bool fbo_has_rgb_ = false;
    bool fbo_has_metric_depth_ = false;
    bool fbo_has_normal_ = false;
    bool fbo_has_object_coord_ = false;
    GLuint instance_ssbo_ = 0;
    GLuint mesh_table_ssbo_ = 0;
    GLuint light_ssbo_ = 0;
    GLuint indirect_buf_ = 0;
    static constexpr int kPersistSlots = 3;
    int persist_slot_ = -1;
    GLsizeiptr persist_inst_cap_ = 0;
    GLsizeiptr persist_cmd_cap_ = 0;
    GLsizeiptr persist_inst_stride_ = 0;
    GLsizeiptr persist_cmd_stride_ = 0;
    GLsizeiptr persist_inst_offset_ = 0;
    GLsizeiptr persist_cmd_offset_ = 0;
    void* persist_inst_ptr_ = nullptr;
    void* persist_cmd_ptr_ = nullptr;
    GLsync persist_fences_[3] = {nullptr, nullptr, nullptr};
    bool persist_active_ = false;
    void (*glBufferSubData_)(GLenum, GLintptr, GLsizeiptr, const void*) = nullptr;
    void (*glScissor_)(GLint, GLint, GLsizei, GLsizei) = nullptr;
    GLint loc_tile_ofs_ = -1;
    GLint loc_viewport_ = -1;
    GLint loc_K_ = -1;
    GLint loc_proj_z_ = -1;
    GLint loc_has_clip_affine_ = -1;
    GLint loc_clip_affine_ = -1;
    uint32_t atlas_w_ = 0;
    uint32_t atlas_h_ = 0;
    uint32_t atlas_cols_ = 1;
    uint32_t fbo_layers_ = 1;
    bool fbo_layered_ = false;
    std::function<void()> pre_fbo_recreate_;
    GlCaps caps_{};
    uint32_t last_draw_api_calls_ = 0;

    std::vector<InstanceRecordGPU> last_gpu_instances_;
    std::vector<DrawElementsIndirectCommand> last_cmds_;
    std::vector<CameraIntrinsics> last_proj_cams_;
    bool last_scene_batch_ = false;
    uint32_t last_scene_w_ = 0;
    uint32_t last_scene_h_ = 0;
    const float* device_poses_ = nullptr;
    const float* device_bboxes_ = nullptr;
    uint32_t device_count_ = 0;
    void* device_stream_ = nullptr;
    bool cuda_instance_active_ = false;
    GLuint cuda_inst_ssbo_ = 0;
    GLuint cuda_indirect_buf_ = 0;
    void* cuda_inst_resource_ = nullptr;
    size_t cuda_inst_cap_ = 0;
    int cuda_device_ = 0;

    void query_caps();
    void ensure_persistent_buffers(GLsizeiptr inst_bytes, GLsizeiptr cmd_bytes);
    void ensure_fbo(uint32_t w, uint32_t h, uint32_t layers, const RenderOutputSpec& spec);
    void destroy_msaa_targets();
    void resolve_msaa(const RenderOutputSpec& spec) const;
    void upload_instances(const std::vector<RenderInstance>& instances,
                          const CameraIntrinsics& cam,
                          const RenderOutputSpec& spec);
    void upload_device_instances(const std::vector<RenderInstance>& instances,
                                 const CameraIntrinsics& cam,
                                 const RenderOutputSpec& spec);
    void ensure_cuda_instance_buffer(size_t bytes);
    void draw_batch(const std::vector<RenderInstance>& instances, uint32_t depth_func,
                    const CameraIntrinsics& cam, const RenderOutputSpec& spec);
};

}
