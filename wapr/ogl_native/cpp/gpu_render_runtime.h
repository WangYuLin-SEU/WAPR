// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#pragma once

#include "batch_renderer.h"
#include "cuda/cuda_pack_context.h"
#include "types.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace wapr_ogl {

class CudaDistortContext;

class GpuRenderRuntime {
public:
    explicit GpuRenderRuntime(int device = 0, const std::string& shader_dir = "");
    ~GpuRenderRuntime();

    int device() const { return device_; }

    uint32_t load_mesh(const MeshUpload& mesh);
    std::vector<uint32_t> load_meshes(const std::vector<MeshUpload>& meshes);
    bool unload_mesh(uint32_t mesh_id);
    void clear_all_meshes();
    bool update_mesh_texture(uint32_t mesh_id, const uint8_t* rgb, uint32_t w, uint32_t h);
    bool mesh_valid(uint32_t mesh_id) const;
    uint32_t mesh_count() const;
    uint32_t mesh_id(const std::string& name) const;

    CudaBufferSet render(const std::vector<RenderInstance>& instances,
                         const CameraIntrinsics& cam,
                         const RenderOutputSpec& spec,
                         const LightParams& light = {},
                         const std::vector<LightParams>* tile_lights = nullptr);

    // One mesh, poses and bboxes already on device. stream is the CUDA stream that wrote them.
    // 一个网格，位姿和框已经在设备上。stream 是写出它们的 CUDA 流。
    void render_device_tiles(uint32_t mesh_id,
                             const float* poses,
                             const float* bboxes,
                             uint32_t count,
                             float fx, float fy, float cx, float cy,
                             uint32_t frame_w, uint32_t frame_h,
                             const RenderOutputSpec& spec,
                             const LightParams& light,
                             void* stream);

    // Mixed meshes share one device tile draw; only metadata is on the host.
    // 混合网格共用一次设备分块绘制；只有元数据在主机端。
    void render_device_mesh_tiles(const std::vector<uint32_t>& mesh_ids,
                                  const float* poses, const float* bboxes,
                                  float fx, float fy, float cx, float cy,
                                  uint32_t frame_w, uint32_t frame_h,
                                  const RenderOutputSpec& spec, const LightParams& light,
                                  void* stream);

    std::string gl_version_string() const;
    GlCaps gl_caps() const;
    uint32_t last_draw_api_calls() const;

    void readback_atlas_rgb(std::vector<uint8_t>& out) const;
    void readback_atlas_depth(std::vector<float>& out) const;

    void synchronize();
    void warmup();
    void set_vram_budget_mb(uint32_t mb) { vram_budget_mb_ = mb; }
    void set_skip_cuda_pack(bool v) { env_skip_cuda_pack_ = v; }
    void set_force_gl_finish(bool v) { env_force_gl_finish_ = v; }
    bool skip_cuda_pack() const { return env_skip_cuda_pack_; }
    bool force_gl_finish() const { return env_force_gl_finish_; }

    void copy_pack_to_host(std::vector<float>& depth,
                           std::vector<float>& rgb,
                           uint32_t& batch, uint32_t& w, uint32_t& h) const;

    
    bool pack_device_ptrs(uintptr_t& depth_ptr, uintptr_t& rgb_ptr,
                          uint32_t& batch, uint32_t& w, uint32_t& h) const;
    bool pack_device_ptrs(uintptr_t& depth_ptr, uintptr_t& rgb_ptr, uintptr_t& id_ptr,
                          uint32_t& batch, uint32_t& w, uint32_t& h) const;
    bool pack_device_ptrs(uintptr_t& depth_ptr, uintptr_t& rgb_ptr, uintptr_t& id_ptr,
                          uintptr_t& normal_ptr, uint32_t& batch, uint32_t& w, uint32_t& h) const;

    
    void distort_remap_host(const float* src_rgb, const float* src_depth,
                            float* dst_rgb, float* dst_depth,
                            float fx, float fy, float cx, float cy,
                            const float* dist5, uint32_t width, uint32_t height);

    
    void distort_remap_device(const float* src_rgb, const float* src_depth,
                              float* dst_rgb, float* dst_depth,
                              float fx, float fy, float cx, float cy,
                              const float* dist5, uint32_t width, uint32_t height);

private:
    int device_ = 0;
    uint32_t vram_budget_mb_ = 0;
    bool warmed_up_ = false;
    bool env_force_cpu_pack_ = false;
    bool env_debug_pack_ = false;
    bool env_force_gl_finish_ = false;
    bool env_skip_cuda_pack_ = false;
    bool warmup_skip_cuda_pack_ = false;
    void do_warmup();
    CudaBufferSet render_once(const std::vector<RenderInstance>& instances,
                              const CameraIntrinsics& cam,
                              const RenderOutputSpec& spec,
                              const LightParams& light,
                              uint32_t pack_tile_offset = 0,
                              const std::vector<LightParams>* tile_lights = nullptr);
    std::string shader_dir_;
    std::unique_ptr<EglContext> egl_;
    std::unique_ptr<AssetManager> assets_;
    std::unique_ptr<BatchRenderer> renderer_;
    std::unique_ptr<CudaPackContext> pack_;
    std::unique_ptr<CudaDistortContext> distort_;
};

}
