// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#include "gpu_render_runtime.h"
#include "projection.h"
#include "shader_utils.h"
#include "cuda/cuda_distort_context.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <GL/gl.h>
#ifndef GL_TEXTURE_2D_ARRAY
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#endif
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace wapr_ogl {

static void check_cuda(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
    }
}

class ScopedCudaDevice {
public:
    explicit ScopedCudaDevice(int device) {
        check_cuda(cudaGetDevice(&previous_), "cudaGetDevice");
        if (previous_ != device) {
            check_cuda(cudaSetDevice(device), "cudaSetDevice");
            restore_ = true;
        }
    }
    ~ScopedCudaDevice() {
        if (restore_) cudaSetDevice(previous_);
    }
    ScopedCudaDevice(const ScopedCudaDevice&) = delete;
    ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

private:
    int previous_ = 0;
    bool restore_ = false;
};

static uint32_t validate_contiguous_slots(const std::vector<RenderInstance>& instances,
                                          const char* label) {
    uint32_t max_slot = 0;
    for (const auto& inst : instances) max_slot = std::max(max_slot, inst.tile_slot);
    if (max_slot == std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(std::string(label) + " tile_slot is too large");
    }
    std::vector<uint8_t> seen(static_cast<size_t>(max_slot) + 1u, 0u);
    for (const auto& inst : instances) seen[inst.tile_slot] = 1u;
    for (uint32_t slot = 0; slot <= max_slot; ++slot) {
        if (!seen[slot]) {
            throw std::invalid_argument(std::string(label) +
                                        " tile_slot values must cover 0..max without holes");
        }
    }
    return max_slot + 1u;
}

GpuRenderRuntime::GpuRenderRuntime(int device, const std::string& shader_dir)
    : device_(device), shader_dir_(shader_dir.empty() ? default_shader_dir() : shader_dir) {
    ScopedCudaDevice device_guard(device_);
    egl_ = std::make_unique<EglContext>(device_);
    const char* set_gl = std::getenv("WAPR_OGL_CUDA_GL_SET_DEVICE");
    const bool want_set_gl = set_gl && (set_gl[0] == '1' || set_gl[0] == 't' || set_gl[0] == 'T');
    if (want_set_gl) {
        cudaError_t gl_dev = cudaGLSetGLDevice(device_);
        if (gl_dev != cudaSuccess) {
            cudaGetLastError();
        }
    }
    assets_ = std::make_unique<AssetManager>(*egl_);
    renderer_ = std::make_unique<BatchRenderer>(*egl_, *assets_, shader_dir_);
    renderer_->set_cuda_device(device_);
    renderer_->set_pre_fbo_recreate([this]() {
        if (pack_) {
            pack_->unregister_gl_textures();
        }
    });
    pack_ = std::make_unique<CudaPackContext>(device_);
    distort_ = std::make_unique<CudaDistortContext>(device_);
    env_force_cpu_pack_ = std::getenv("WAPR_OGL_FORCE_CPU_PACK") != nullptr;
    env_debug_pack_ = std::getenv("WAPR_OGL_DEBUG") != nullptr;
    env_force_gl_finish_ = std::getenv("WAPR_OGL_GL_FINISH") != nullptr;
    env_skip_cuda_pack_ = std::getenv("WAPR_OGL_SKIP_CUDA_PACK") != nullptr;
    do_warmup();
}

void GpuRenderRuntime::synchronize() {
    ScopedCudaDevice device_guard(device_);
    if (pack_) pack_->synchronize();
}

void GpuRenderRuntime::warmup() {
    do_warmup();
}

void GpuRenderRuntime::do_warmup() {
    if (warmed_up_) return;
    if (std::getenv("WAPR_OGL_SKIP_WARMUP")) {
        warmed_up_ = true;
        return;
    }
    MeshUpload tri{};
    tri.name = "__warmup__";
    tri.vertices = {0.f, 0.f, 0.5f, 0.05f, 0.f, 0.5f, 0.f, 0.05f, 0.5f};
    tri.indices = {0, 1, 2};
    const uint32_t mid = assets_->load_mesh(tri);
    RenderInstance inst{};
    inst.mesh_id = mid;
    inst.T_cam_obj[0] = inst.T_cam_obj[5] = inst.T_cam_obj[10] = inst.T_cam_obj[15] = 1.f;
    inst.T_cam_obj[14] = 0.6f;
    CameraIntrinsics cam{};
    cam.fx = cam.fy = 600.f;
    cam.cx = 320.f;
    cam.cy = 240.f;
    cam.width = 640;
    cam.height = 480;
    RenderOutputSpec spec{};
    spec.mode = RenderMode::Tile;
    spec.tile_w = spec.tile_h = 160;
    spec.rgb = spec.metric_depth = true;
    LightParams lp{};
    warmup_skip_cuda_pack_ = true;
    render({inst}, cam, spec, lp);
    warmup_skip_cuda_pack_ = false;
    if (pack_) pack_->synchronize();
    warmed_up_ = true;
}

GpuRenderRuntime::~GpuRenderRuntime() {
    try {
        ScopedCudaDevice device_guard(device_);
        distort_.reset();
        pack_.reset();
        renderer_.reset();
        assets_.reset();
        egl_.reset();
    } catch (...) {
    }
}

uint32_t GpuRenderRuntime::load_mesh(const MeshUpload& mesh) {
    return assets_->load_mesh(mesh);
}

std::vector<uint32_t> GpuRenderRuntime::load_meshes(const std::vector<MeshUpload>& meshes) {
    return assets_->load_meshes(meshes);
}

bool GpuRenderRuntime::unload_mesh(uint32_t mesh_id) {
    return assets_->unload_mesh(mesh_id);
}

void GpuRenderRuntime::clear_all_meshes() {
    assets_->clear_all();
}

bool GpuRenderRuntime::update_mesh_texture(uint32_t mesh_id, const uint8_t* rgb, uint32_t w, uint32_t h) {
    return assets_->update_mesh_texture(mesh_id, rgb, w, h);
}

bool GpuRenderRuntime::mesh_valid(uint32_t mesh_id) const {
    return assets_->mesh_valid(mesh_id);
}

uint32_t GpuRenderRuntime::mesh_count() const {
    return assets_->mesh_count();
}

uint32_t GpuRenderRuntime::mesh_id(const std::string& name) const {
    return assets_->mesh_id_by_name(name);
}

CudaBufferSet GpuRenderRuntime::render(const std::vector<RenderInstance>& instances,
                                       const CameraIntrinsics& cam,
                                       const RenderOutputSpec& spec,
                                       const LightParams& light,
                                       const std::vector<LightParams>* tile_lights) {
    ScopedCudaDevice device_guard(device_);
    if (instances.empty()) {
        return {};
    }
    if (cam.width == 0 || cam.height == 0 || !std::isfinite(cam.fx) ||
        !std::isfinite(cam.fy) || !std::isfinite(cam.cx) || !std::isfinite(cam.cy) ||
        !(cam.fx > 0.f) || !(cam.fy > 0.f)) {
        throw std::invalid_argument("camera must have positive dimensions/focal lengths and finite intrinsics");
    }
    if (spec.object_coord) {
        throw std::invalid_argument(
            "object_coord is not implemented: the current MRT stores camera-space position "
            "and CUDA does not pack attachment 3");
    }
    if (spec.mode == RenderMode::Scene && !scene_batch_active(spec, instances) &&
        (spec.normal || spec.output_instance_id)) {
        throw std::invalid_argument(
            "scene mode does not pack normal or instance-id outputs; use tile/scene_batch or disable them");
    }
    if (spec.viewport_per_tile && !spec.use_scissor) {
        throw std::invalid_argument("viewport_per_tile requires use_scissor so clears stay inside each tile");
    }
    bool has_less = false;
    bool has_greater = false;
    for (const auto& inst : instances) {
        if (!assets_->mesh_valid(inst.mesh_id)) {
            throw std::invalid_argument("render instance references an invalid or unloaded mesh_id");
        }
        if (inst.depth_func == kDepthFuncLess) has_less = true;
        else if (inst.depth_func == kDepthFuncGreater) has_greater = true;
        else throw std::invalid_argument("depth_func must be DEPTH_LESS or DEPTH_GREATER");
    }
    if (has_less && has_greater) {
        throw std::invalid_argument(
            "mixing DEPTH_LESS and DEPTH_GREATER in one render call is undefined; split the passes");
    }

    uint32_t output_count = 1u;
    if (spec.mode == RenderMode::Tile) {
        if (spec.tile_w == 0 || spec.tile_h == 0) {
            throw std::invalid_argument("tile dimensions must be > 0");
        }
        output_count = validate_contiguous_slots(instances, "tile batch");
    } else if (scene_batch_active(spec, instances)) {
        output_count = validate_contiguous_slots(instances, "scene batch");
    }
    if (tile_lights != nullptr && tile_lights->size() != output_count) {
        throw std::invalid_argument("tile_lights length must equal the number of output slots");
    }
    if (spec.mode == RenderMode::Tile) {
        const uint32_t n_tiles = tile_batch_count(instances);
        uint32_t max_chunk = spec.layered_tiles
            ? (spec.max_atlas_w > 0 ? spec.max_atlas_w : 64u)
            : max_tiles_per_atlas_chunk(
                  spec.tile_w, spec.tile_h, spec.max_atlas_w, spec.max_atlas_h);
        if (max_chunk == 0) {
            throw std::runtime_error("tile max_chunk is 0 (atlas smaller than tile or invalid spec)");
        }
        if (n_tiles <= max_chunk) {
            return render_once(instances, cam, spec, light, 0, tile_lights);
        }

        pack_->ensure_batch_output(n_tiles, spec.tile_w, spec.tile_h, spec);

        CudaBufferSet merged{};
        merged.batch_size = n_tiles;
        merged.width = spec.tile_w;
        merged.height = spec.tile_h;
        merged.depth = pack_->device_depth();
        merged.rgb = pack_->device_rgb();
        merged.normal = pack_->device_normal();
        merged.instance_id = pack_->device_instance_id();
        merged.on_device = true;

        std::vector<std::vector<RenderInstance>> by_tile(n_tiles);
        for (const auto& inst : instances) {
            if (inst.tile_slot >= n_tiles) {
                throw std::runtime_error("tile_slot out of range for tile batch");
            }
            by_tile[inst.tile_slot].push_back(inst);
        }
        for (uint32_t off = 0; off < n_tiles; off += max_chunk) {
            const uint32_t chunk_n = std::min(max_chunk, n_tiles - off);
            std::vector<RenderInstance> chunk;
            chunk.reserve(instances.size());
            for (uint32_t s = 0; s < chunk_n; ++s) {
                for (auto inst : by_tile[off + s]) {
                    inst.tile_slot = s;
                    chunk.push_back(std::move(inst));
                }
            }
            const std::vector<LightParams>* chunk_lights = nullptr;
            std::vector<LightParams> lights_chunk;
            if (tile_lights != nullptr && tile_lights->size() >= off + chunk_n) {
                lights_chunk.assign(tile_lights->begin() + off,
                                    tile_lights->begin() + off + chunk_n);
                chunk_lights = &lights_chunk;
            }
            render_once(chunk, cam, spec, light, off, chunk_lights);
        }
        pack_->set_last_pack_dims(n_tiles, spec.tile_w, spec.tile_h);
        return merged;
    }

    if (scene_batch_active(spec, instances)) {
        const uint32_t n_scenes = scene_batch_count(instances);
        uint32_t max_chunk = max_tiles_per_atlas_chunk(
            cam.width, cam.height, spec.max_atlas_w, spec.max_atlas_h);
        if (max_chunk == 0) {
            throw std::runtime_error("scene_batch max_chunk is 0 (atlas smaller than frame or invalid spec)");
        }
        if (n_scenes <= max_chunk) {
            return render_once(instances, cam, spec, light, 0, tile_lights);
        }

        std::vector<std::vector<RenderInstance>> by_scene(n_scenes);
        for (const auto& inst : instances) {
            if (inst.tile_slot >= n_scenes) {
                throw std::runtime_error("tile_slot out of range for scene batch");
            }
            by_scene[inst.tile_slot].push_back(inst);
        }

        pack_->ensure_batch_output(n_scenes, cam.width, cam.height, spec);

        CudaBufferSet merged{};
        merged.batch_size = n_scenes;
        merged.width = cam.width;
        merged.height = cam.height;
        merged.depth = pack_->device_depth();
        merged.rgb = pack_->device_rgb();
        merged.normal = pack_->device_normal();
        merged.instance_id = pack_->device_instance_id();
        merged.on_device = true;

        for (uint32_t off = 0; off < n_scenes; off += max_chunk) {
            const uint32_t chunk_n = std::min(max_chunk, n_scenes - off);
            std::vector<RenderInstance> chunk;
            chunk.reserve(instances.size());
            for (uint32_t s = 0; s < chunk_n; ++s) {
                for (auto inst : by_scene[off + s]) {
                    inst.tile_slot = s;
                    chunk.push_back(std::move(inst));
                }
            }
            const std::vector<LightParams>* chunk_lights = nullptr;
            std::vector<LightParams> lights_chunk;
            if (tile_lights != nullptr && tile_lights->size() >= off + chunk_n) {
                lights_chunk.assign(tile_lights->begin() + off,
                                    tile_lights->begin() + off + chunk_n);
                chunk_lights = &lights_chunk;
            }
            render_once(chunk, cam, spec, light, off, chunk_lights);
        }
        pack_->set_last_pack_dims(n_scenes, cam.width, cam.height);
        return merged;
    }

    return render_once(instances, cam, spec, light, 0, tile_lights);
}

void GpuRenderRuntime::render_device_tiles(uint32_t mesh_id,
                                           const float* poses,
                                           const float* bboxes,
                                           uint32_t count,
                                           float fx, float fy, float cx, float cy,
                                           uint32_t frame_w, uint32_t frame_h,
                                           const RenderOutputSpec& spec,
                                           const LightParams& light,
                                           void* stream) {
    render_device_mesh_tiles(std::vector<uint32_t>(count, mesh_id), poses, bboxes,
                             fx, fy, cx, cy, frame_w, frame_h, spec, light, stream);
}

void GpuRenderRuntime::render_device_mesh_tiles(const std::vector<uint32_t>& mesh_ids,
                                               const float* poses, const float* bboxes,
                                               float fx, float fy, float cx, float cy,
                                               uint32_t frame_w, uint32_t frame_h,
                                               const RenderOutputSpec& spec,
                                               const LightParams& light, void* stream) {
    const uint32_t count = static_cast<uint32_t>(mesh_ids.size());
    ScopedCudaDevice device_guard(device_);
    if (count == 0 || count > 256) {
        throw std::invalid_argument("device tile count must be 1..256");
    }
    if (poses == nullptr || bboxes == nullptr) {
        throw std::invalid_argument("device tile poses and bboxes are required");
    }
    if (spec.mode != RenderMode::Tile || spec.tile_w == 0 || spec.tile_h == 0) {
        throw std::invalid_argument("device tiles require a tile spec");
    }
    for (uint32_t mesh_id : mesh_ids) {
        if (!assets_->mesh_valid(mesh_id)) {
            throw std::invalid_argument("device tiles reference an invalid mesh");
        }
    }
    if (!(fx > 0.f) || !(fy > 0.f) || frame_w == 0 || frame_h == 0) {
        throw std::invalid_argument("device tiles require a positive camera");
    }
    const uint32_t max_chunk = spec.layered_tiles
                                   ? (spec.max_atlas_w > 0 ? spec.max_atlas_w : 64u)
                                   : max_tiles_per_atlas_chunk(
                                         spec.tile_w, spec.tile_h, spec.max_atlas_w, spec.max_atlas_h);
    if (max_chunk == 0 || count > max_chunk) {
        throw std::invalid_argument("device tile batch does not fit one draw");
    }
    renderer_->arm_device_tiles(poses, bboxes, count, stream);
    struct Disarm {
        BatchRenderer* renderer;
        ~Disarm() { renderer->disarm_device_tiles(); }
    } disarm{renderer_.get()};
    std::vector<RenderInstance> instances(count);
    for (uint32_t i = 0; i < count; ++i) {
        instances[i].mesh_id = mesh_ids[i];
        instances[i].tile_slot = i;
        instances[i].depth_func = kDepthFuncLess;
        instances[i].has_clip_affine = 1;
    }
    CameraIntrinsics cam{};
    cam.fx = fx;
    cam.fy = fy;
    cam.cx = cx;
    cam.cy = cy;
    cam.width = frame_w;
    cam.height = frame_h;
    render(instances, cam, spec, light, nullptr);
}

CudaBufferSet GpuRenderRuntime::render_once(const std::vector<RenderInstance>& instances,
                                            const CameraIntrinsics& cam,
                                            const RenderOutputSpec& spec,
                                            const LightParams& light,
                                            uint32_t pack_tile_offset,
                                            const std::vector<LightParams>* tile_lights) {
    if (instances.empty()) {
        return {};
    }
    LightParams lp = light;

    RenderOutputSpec eff = spec;
    std::vector<RenderInstance> eff_inst = instances;
    const bool scene_batch = scene_batch_active(spec, instances);
    if (spec.mode == RenderMode::Scene) {
        if (scene_batch) {
            eff.scene_batch = true;
            eff.use_scissor = true;
            eff.viewport_per_tile = true;
            eff.tile_w = cam.width;
            eff.tile_h = cam.height;
            if (eff.projection == ProjectionMode::PinholeCrop) {
                for (auto& inst : eff_inst) {
                    if (!inst.has_bbox) {
                        inst.bbox_xywh[0] = 0.f;
                        inst.bbox_xywh[1] = 0.f;
                        inst.bbox_xywh[2] = static_cast<float>(cam.width);
                        inst.bbox_xywh[3] = static_cast<float>(cam.height);
                        inst.has_bbox = 1;
                    }
                }
            }
        } else {
            for (auto& inst : eff_inst) {
                inst.tile_slot = 0;
                if (!inst.has_bbox && eff.projection == ProjectionMode::PinholeCrop) {
                    inst.bbox_xywh[0] = 0.f;
                    inst.bbox_xywh[1] = 0.f;
                    inst.bbox_xywh[2] = static_cast<float>(cam.width);
                    inst.bbox_xywh[3] = static_cast<float>(cam.height);
                    inst.has_bbox = 1;
                }
            }
        }
    }

    renderer_->render(eff_inst, cam, eff, lp, tile_lights);
    egl_->make_current();

    const uint32_t aw = renderer_->atlas_width();
    const uint32_t ah = renderer_->atlas_height();
    const bool use_cpu_pack = env_force_cpu_pack_;
    const bool debug_pack = env_debug_pack_;
    const bool skip_cuda_pack = warmup_skip_cuda_pack_ || env_skip_cuda_pack_;
    const bool force_gl_finish = env_force_gl_finish_;
    if (debug_pack) {
        fprintf(stderr, "pack atlas=%ux%u scene=%d use_cpu=%d\n",
                aw, ah, eff.mode == RenderMode::Scene ? 1 : 0, use_cpu_pack ? 1 : 0);
    }

    CudaBufferSet out{};
    if (skip_cuda_pack) {
        auto& gl = egl_->gl();
        if (force_gl_finish && gl.Finish) {
            gl.Finish();
        } else if (gl.Flush) {
            gl.Flush();
        } else if (gl.Finish) {
            gl.Finish();
        }
        if (eff.mode == RenderMode::Scene && scene_batch) {
            out.batch_size = scene_batch_count(eff_inst);
            out.width = cam.width;
            out.height = cam.height;
        } else if (eff.mode == RenderMode::Tile) {
            out.batch_size = tile_batch_count(eff_inst);
            out.width = eff.tile_w;
            out.height = eff.tile_h;
        } else {
            out.batch_size = static_cast<uint32_t>(instances.size());
            out.width = spec.mode == RenderMode::Scene ? cam.width : spec.tile_w;
            out.height = spec.mode == RenderMode::Scene ? cam.height : spec.tile_h;
        }
        return out;
    }

    if (use_cpu_pack) {
        auto& gl = egl_->gl();
        if (gl.Finish) {
            gl.Finish();
        }
        std::vector<float> depth_cpu;
        std::vector<float> rgb_cpu;
        renderer_->readback_depth_f32(depth_cpu);
        if (eff.rgb) {
            renderer_->readback_rgb_rgba_f32(rgb_cpu);
        }
        if (debug_pack) {
            size_t nz = 0;
            for (float v : depth_cpu) {
                if (v > 0.f) ++nz;
            }
            fprintf(stderr, "cpu_pack %ux%u depth_nonzero=%zu rgb_max=%f\n",
                    aw, ah, nz,
                    rgb_cpu.empty() ? 0.f : *std::max_element(rgb_cpu.begin(), rgb_cpu.end()));
        }
        if (eff.mode == RenderMode::Scene) {
            if (scene_batch) {
                const uint32_t n_scenes = scene_batch_count(eff_inst);
                return pack_->pack_tiles_from_cpu(
                    eff, depth_cpu.data(), eff.rgb ? rgb_cpu.data() : nullptr, aw, ah,
                    n_scenes, cam.width, cam.height, renderer_->atlas_cols());
            }
            return pack_->pack_scene_from_cpu(eff, depth_cpu.data(),
                                              eff.rgb ? rgb_cpu.data() : nullptr, aw, ah);
        }
        const uint32_t n_pack = tile_batch_count(eff_inst);
        return pack_->pack_tiles_from_cpu(
            eff, depth_cpu.data(), eff.rgb ? rgb_cpu.data() : nullptr, aw, ah,
            n_pack, eff.tile_w, eff.tile_h,
            renderer_->atlas_cols());
    }

    auto& gl = egl_->gl();
    if (gl.BindFramebuffer) {
        gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    if (gl.ActiveTexture) {
        gl.ActiveTexture(0x84C0);
    }
    if (gl.BindTexture) {
        gl.BindTexture(GL_TEXTURE_2D, 0);
        gl.BindTexture(GL_TEXTURE_2D_ARRAY, 0);
    }
    if (force_gl_finish && gl.Finish) {
        gl.Finish();
    } else if (gl.Flush) {
        gl.Flush();
    } else if (gl.Finish) {
        gl.Finish();
    }

    const bool layered = eff.layered_tiles && eff.mode == RenderMode::Tile;
    if (spec.metric_depth) {
        pack_->register_gl_texture(renderer_->tex_depth(), aw, ah, 0, layered);
    }
    if (spec.rgb) {
        pack_->register_gl_texture(renderer_->tex_rgb(), aw, ah, 1, layered);
    }
    if (spec.normal) {
        pack_->register_gl_texture(renderer_->tex_normal(), aw, ah, 2, layered);
    }
    if (spec.output_instance_id) {
        pack_->register_gl_texture(renderer_->tex_instance_id(), aw, ah, 4, layered);
    }

    if (eff.mode == RenderMode::Scene) {
        if (scene_batch) {
            const uint32_t n_scenes = scene_batch_count(eff_inst);
            return pack_->pack_tiles(eff, aw, ah, n_scenes, cam.width, cam.height,
                                     renderer_->atlas_cols(), pack_tile_offset);
        }
        return pack_->pack_scene(eff, aw, ah);
    }
    if (layered) {
        return pack_->pack_layers(eff, tile_batch_count(eff_inst),
                                  eff.tile_w, eff.tile_h, pack_tile_offset);
    }
    return pack_->pack_tiles(eff, aw, ah,
                             tile_batch_count(eff_inst),
                             eff.tile_w, eff.tile_h,
                             renderer_->atlas_cols(), pack_tile_offset);
}

void GpuRenderRuntime::readback_atlas_rgb(std::vector<uint8_t>& out) const {
    renderer_->readback_rgb(out);
}

void GpuRenderRuntime::readback_atlas_depth(std::vector<float>& out) const {
    renderer_->readback_depth(out);
}

std::string GpuRenderRuntime::gl_version_string() const {
    return egl_ ? egl_->gl_version_string() : "no_egl";
}

GlCaps GpuRenderRuntime::gl_caps() const {
    return renderer_ ? renderer_->gl_caps() : GlCaps{};
}

uint32_t GpuRenderRuntime::last_draw_api_calls() const {
    return renderer_ ? renderer_->last_draw_api_calls() : 0u;
}

void GpuRenderRuntime::copy_pack_to_host(std::vector<float>& depth,
                                         std::vector<float>& rgb,
                                         uint32_t& batch, uint32_t& w, uint32_t& h) const {
    ScopedCudaDevice device_guard(device_);
    if (!pack_) return;
    pack_->synchronize();
    batch = pack_->last_batch();
    w = pack_->last_width();
    h = pack_->last_height();
    const size_t dpix = static_cast<size_t>(batch) * w * h;
    depth.clear();
    rgb.clear();
    if (pack_->device_depth()) {
        depth.resize(dpix);
        cudaError_t err = cudaMemcpy(depth.data(), pack_->device_depth(), dpix * sizeof(float),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("copy_pack_to_host depth: ") + cudaGetErrorString(err));
        }
    }
    if (pack_->device_rgb()) {
        rgb.resize(dpix * 3);
        cudaError_t err = cudaMemcpy(rgb.data(), pack_->device_rgb(), dpix * 3 * sizeof(float),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("copy_pack_to_host rgb: ") + cudaGetErrorString(err));
        }
    }
}

bool GpuRenderRuntime::pack_device_ptrs(uintptr_t& depth_ptr, uintptr_t& rgb_ptr,
                                        uint32_t& batch, uint32_t& w, uint32_t& h) const {
    uintptr_t id_ptr = 0;
    uintptr_t normal_ptr = 0;
    return pack_device_ptrs(depth_ptr, rgb_ptr, id_ptr, normal_ptr, batch, w, h);
}

bool GpuRenderRuntime::pack_device_ptrs(uintptr_t& depth_ptr, uintptr_t& rgb_ptr, uintptr_t& id_ptr,
                                        uint32_t& batch, uint32_t& w, uint32_t& h) const {
    uintptr_t normal_ptr = 0;
    return pack_device_ptrs(depth_ptr, rgb_ptr, id_ptr, normal_ptr, batch, w, h);
}

bool GpuRenderRuntime::pack_device_ptrs(uintptr_t& depth_ptr, uintptr_t& rgb_ptr, uintptr_t& id_ptr,
                                        uintptr_t& normal_ptr, uint32_t& batch, uint32_t& w, uint32_t& h) const {
    ScopedCudaDevice device_guard(device_);
    if (!pack_) {
        depth_ptr = rgb_ptr = id_ptr = normal_ptr = 0;
        batch = w = h = 0;
        return false;
    }
    pack_->synchronize();
    batch = pack_->last_batch();
    w = pack_->last_width();
    h = pack_->last_height();
    depth_ptr = reinterpret_cast<uintptr_t>(pack_->device_depth());
    rgb_ptr = reinterpret_cast<uintptr_t>(pack_->device_rgb());
    id_ptr = reinterpret_cast<uintptr_t>(pack_->device_instance_id());
    normal_ptr = reinterpret_cast<uintptr_t>(pack_->device_normal());
    return depth_ptr != 0 || rgb_ptr != 0 || id_ptr != 0 || normal_ptr != 0;
}

void GpuRenderRuntime::distort_remap_host(const float* src_rgb, const float* src_depth,
                                          float* dst_rgb, float* dst_depth,
                                          float fx, float fy, float cx, float cy,
                                          const float* dist5, uint32_t width, uint32_t height) {
    ScopedCudaDevice device_guard(device_);
    if (!distort_) return;
    distort_->ensure_map(fx, fy, cx, cy, dist5, width, height);
    distort_->remap_rgb_depth_host(src_rgb, src_depth, dst_rgb, dst_depth, width, height);
}

void GpuRenderRuntime::distort_remap_device(const float* src_rgb, const float* src_depth,
                                            float* dst_rgb, float* dst_depth,
                                            float fx, float fy, float cx, float cy,
                                            const float* dist5, uint32_t width, uint32_t height) {
    ScopedCudaDevice device_guard(device_);
    if (!distort_) return;
    distort_->ensure_map(fx, fy, cx, cy, dist5, width, height);
    distort_->remap_rgb_depth_device(src_rgb, src_depth, dst_rgb, dst_depth, width, height);
}

}
