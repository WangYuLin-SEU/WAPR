// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#include "cuda_pack_context.h"
#include "cuda_device_guard.h"
#include "pack_outputs.cuh"
#include <cuda_gl_interop.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace wapr_ogl {

namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

void check_cuda_last(const char* what) {
    check_cuda(cudaGetLastError(), what);
}

struct CudaGraphicsMapGuard {
    cudaGraphicsResource** list = nullptr;
    int n = 0;
    cudaStream_t stream = nullptr;
    bool live = false;

    void map(const char* what) {
        if (n > 0) {
            check_cuda(cudaGraphicsMapResources(n, list, stream), what);
            live = true;
        }
    }

    void unmap(const char* what) {
        if (!live) {
            return;
        }
        live = false;
        check_cuda(cudaGraphicsUnmapResources(n, list, stream), what);
    }

    ~CudaGraphicsMapGuard() {
        if (live) {
            cudaGraphicsUnmapResources(n, list, stream);
            live = false;
        }
    }
};

}

CudaPackContext::CudaPackContext(int device) : device_(device) {
    CudaDeviceGuard device_guard(device_, "CudaPackContext ctor");
    check_cuda(cudaSetDevice(device_), "cudaSetDevice(CudaPackContext)");
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags(CudaPackContext)");
}

CudaPackContext::~CudaPackContext() {
    try {
        CudaDeviceGuard device_guard(device_, "CudaPackContext dtor");
        if (rgb_tex_.resource) cudaGraphicsUnregisterResource(rgb_tex_.resource);
    if (depth_tex_.resource) cudaGraphicsUnregisterResource(depth_tex_.resource);
    if (normal_tex_.resource) cudaGraphicsUnregisterResource(normal_tex_.resource);
    if (id_tex_.resource) cudaGraphicsUnregisterResource(id_tex_.resource);
    if (d_rgb_) cudaFree(d_rgb_);
    if (d_depth_) cudaFree(d_depth_);
    if (d_normal_) cudaFree(d_normal_);
    if (d_coord_) cudaFree(d_coord_);
    if (d_instance_id_) cudaFree(d_instance_id_);
    if (d_depth_stage_) cudaFree(d_depth_stage_);
    if (d_rgb_stage_) cudaFree(d_rgb_stage_);
    if (d_normal_stage_) cudaFree(d_normal_stage_);
    if (d_id_stage_) cudaFree(d_id_stage_);
        if (stream_) cudaStreamDestroy(stream_);
    } catch (...) {
    }
}

void CudaPackContext::synchronize() {
    cudaStreamSynchronize(stream_);
}

void CudaPackContext::ensure_batch_output(uint32_t batch, uint32_t tw, uint32_t th,
                                          const RenderOutputSpec& spec) {
    ensure_capacity(batch, tw, th, spec);
    last_batch_ = batch;
    last_w_ = tw;
    last_h_ = th;
    last_has_rgb_ = spec.rgb;
    last_has_depth_ = spec.metric_depth;
    last_has_normal_ = spec.normal;
    last_has_instance_id_ = spec.output_instance_id;
}

void CudaPackContext::unregister_gl_textures() {
    auto unreg = [](RegisteredTex& slot) {
        if (slot.resource) {
            cudaGraphicsUnregisterResource(slot.resource);
            slot.resource = nullptr;
        }
        slot.tex_id = 0;
        slot.w = 0;
        slot.h = 0;
    };
    unreg(depth_tex_);
    unreg(rgb_tex_);
    unreg(normal_tex_);
    unreg(id_tex_);
}

void CudaPackContext::register_gl_texture(GLuint tex, uint32_t w, uint32_t h, int channel_index,
                                          bool texture_array) {
    RegisteredTex* slot = nullptr;
    if (channel_index == 0) slot = &depth_tex_;
    else if (channel_index == 1) slot = &rgb_tex_;
    else if (channel_index == 2) slot = &normal_tex_;
    else if (channel_index == 4) slot = &id_tex_;
    if (!slot) return;

#ifndef GL_TEXTURE_2D_ARRAY
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#endif
    if (slot->resource && slot->tex_id == tex && slot->w == w && slot->h == h &&
        slot->is_array == texture_array) {
        return;
    }
    if (slot->resource) {
        cudaGraphicsUnregisterResource(slot->resource);
        slot->resource = nullptr;
    }
    slot->tex_id = tex;
    slot->w = w;
    slot->h = h;
    slot->is_array = texture_array;
    const GLenum target = texture_array ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;
    check_cuda(cudaGraphicsGLRegisterImage(&slot->resource, tex, target, cudaGraphicsRegisterFlagsReadOnly),
               "cudaGraphicsGLRegisterImage");
}

void CudaPackContext::ensure_capacity(uint32_t batch, uint32_t tw, uint32_t th, const RenderOutputSpec& spec) {
    const size_t pix = static_cast<size_t>(batch) * tw * th;
    if (spec.rgb) {
        const size_t need = pix * 3 * sizeof(float);
        if (need > cap_rgb_) {
            if (d_rgb_) cudaFree(d_rgb_);
            check_cuda(cudaMalloc(&d_rgb_, need), "cudaMalloc(d_rgb)");
            cap_rgb_ = need;
        }
    }
    if (spec.metric_depth) {
        const size_t need = pix * sizeof(float);
        if (need > cap_depth_) {
            if (d_depth_) cudaFree(d_depth_);
            check_cuda(cudaMalloc(&d_depth_, need), "cudaMalloc(d_depth)");
            cap_depth_ = need;
        }
    }
    if (spec.normal) {
        const size_t need = pix * 3 * sizeof(float);
        if (need > cap_normal_) {
            if (d_normal_) cudaFree(d_normal_);
            check_cuda(cudaMalloc(&d_normal_, need), "cudaMalloc(d_normal)");
            cap_normal_ = need;
        }
    }
    if (spec.output_instance_id) {
        const size_t need = pix * sizeof(uint32_t);
        if (need > cap_instance_id_) {
            if (d_instance_id_) cudaFree(d_instance_id_);
            check_cuda(cudaMalloc(&d_instance_id_, need), "cudaMalloc(d_instance_id)");
            cap_instance_id_ = need;
        }
    }
}

void CudaPackContext::ensure_stage(uint32_t atlas_w, uint32_t atlas_h,
                                   bool need_depth, bool need_rgb,
                                   bool need_normal, bool need_id) {
    const size_t dpix = static_cast<size_t>(atlas_w) * atlas_h;
    const size_t need_d = dpix * sizeof(float);
    const size_t need_r = dpix * 4 * sizeof(float);
    const size_t need_n = dpix * 4 * sizeof(float);
    const size_t need_i = dpix * sizeof(uint32_t);
    if (need_depth && need_d > cap_depth_stage_) {
        if (d_depth_stage_) cudaFree(d_depth_stage_);
        check_cuda(cudaMalloc(&d_depth_stage_, need_d), "cudaMalloc(d_depth_stage)");
        cap_depth_stage_ = need_d;
    }
    if (need_rgb && need_r > cap_rgb_stage_) {
        if (d_rgb_stage_) cudaFree(d_rgb_stage_);
        check_cuda(cudaMalloc(&d_rgb_stage_, need_r), "cudaMalloc(d_rgb_stage)");
        cap_rgb_stage_ = need_r;
    }
    if (need_normal && need_n > cap_normal_stage_) {
        if (d_normal_stage_) cudaFree(d_normal_stage_);
        check_cuda(cudaMalloc(&d_normal_stage_, need_n), "cudaMalloc(d_normal_stage)");
        cap_normal_stage_ = need_n;
    }
    if (need_id && need_i > cap_id_stage_) {
        if (d_id_stage_) cudaFree(d_id_stage_);
        check_cuda(cudaMalloc(&d_id_stage_, need_i), "cudaMalloc(d_id_stage)");
        cap_id_stage_ = need_i;
    }
}

CudaBufferSet CudaPackContext::pack_layers(const RenderOutputSpec& spec,
                                           uint32_t batch_size,
                                           uint32_t tile_w, uint32_t tile_h,
                                           uint32_t out_tile_offset) {
    ensure_capacity(out_tile_offset + batch_size, tile_w, tile_h, spec);
    const uint32_t stack_h = tile_h * batch_size;
    ensure_stage(tile_w, stack_h, spec.metric_depth, spec.rgb,
                 spec.normal, spec.output_instance_id);

    cudaGraphicsResource* map_list[4];
    int nres = 0;
    if (spec.metric_depth && depth_tex_.resource) map_list[nres++] = depth_tex_.resource;
    if (spec.rgb && rgb_tex_.resource) map_list[nres++] = rgb_tex_.resource;
    if (normal_tex_.resource && spec.normal) map_list[nres++] = normal_tex_.resource;
    if (id_tex_.resource && spec.output_instance_id) map_list[nres++] = id_tex_.resource;
    CudaGraphicsMapGuard mapped;
    mapped.list = map_list;
    mapped.n = nres;
    mapped.stream = stream_;
    mapped.map("cudaGraphicsMapResources(layers)");

    const size_t tile_pix = static_cast<size_t>(tile_w) * tile_h;
    const size_t dpitch = tile_w * sizeof(float);
    const size_t rpitch = tile_w * 4 * sizeof(float);
    const size_t ipitch = tile_w * sizeof(uint32_t);

    for (uint32_t layer = 0; layer < batch_size; ++layer) {
        if (spec.metric_depth && depth_tex_.resource) {
            cudaArray_t arr;
            check_cuda(cudaGraphicsSubResourceGetMappedArray(&arr, depth_tex_.resource, layer, 0),
                       "cudaGraphicsSubResourceGetMappedArray(depth layer)");
            float* dst = d_depth_stage_ + static_cast<size_t>(layer) * tile_pix;
            check_cuda(cudaMemcpy2DFromArrayAsync(dst, dpitch, arr, 0, 0, tile_w * sizeof(float), tile_h,
                                                  cudaMemcpyDeviceToDevice, stream_),
                       "cudaMemcpy2DFromArrayAsync(depth layer)");
        }
        if (spec.rgb && rgb_tex_.resource) {
            cudaArray_t arr;
            check_cuda(cudaGraphicsSubResourceGetMappedArray(&arr, rgb_tex_.resource, layer, 0),
                       "cudaGraphicsSubResourceGetMappedArray(rgb layer)");
            float* dst = d_rgb_stage_ + static_cast<size_t>(layer) * tile_pix * 4;
            check_cuda(cudaMemcpy2DFromArrayAsync(dst, rpitch, arr, 0, 0, tile_w * 4 * sizeof(float), tile_h,
                                                  cudaMemcpyDeviceToDevice, stream_),
                       "cudaMemcpy2DFromArrayAsync(rgb layer)");
        }
        if (normal_tex_.resource && spec.normal) {
            cudaArray_t arr;
            check_cuda(cudaGraphicsSubResourceGetMappedArray(&arr, normal_tex_.resource, layer, 0),
                       "cudaGraphicsSubResourceGetMappedArray(normal layer)");
            float* dst = d_normal_stage_ + static_cast<size_t>(layer) * tile_pix * 4;
            check_cuda(cudaMemcpy2DFromArrayAsync(dst, rpitch, arr, 0, 0, tile_w * 4 * sizeof(float), tile_h,
                                                  cudaMemcpyDeviceToDevice, stream_),
                       "cudaMemcpy2DFromArrayAsync(normal layer)");
        }
        if (id_tex_.resource && spec.output_instance_id && d_id_stage_) {
            cudaArray_t arr;
            check_cuda(cudaGraphicsSubResourceGetMappedArray(&arr, id_tex_.resource, layer, 0),
                       "cudaGraphicsSubResourceGetMappedArray(id layer)");
            uint32_t* dst = d_id_stage_ + static_cast<size_t>(layer) * tile_pix;
            check_cuda(cudaMemcpy2DFromArrayAsync(dst, ipitch, arr, 0, 0, tile_w * sizeof(uint32_t), tile_h,
                                                  cudaMemcpyDeviceToDevice, stream_),
                       "cudaMemcpy2DFromArrayAsync(id layer)");
        }
    }

    float* depth_out = spec.metric_depth
        ? static_cast<float*>(d_depth_) + static_cast<size_t>(out_tile_offset) * tile_pix
        : nullptr;
    float* rgb_out = (spec.rgb && d_rgb_)
        ? static_cast<float*>(d_rgb_) + static_cast<size_t>(out_tile_offset) * tile_pix * 3
        : nullptr;
    float* normal_out =
        spec.normal ? static_cast<float*>(d_normal_) + static_cast<size_t>(out_tile_offset) * tile_pix * 3
                    : nullptr;
    cuda_kernels::pack_tiles_rgb_depth(
        spec.metric_depth ? d_depth_stage_ : nullptr, spec.rgb ? d_rgb_stage_ : nullptr,
        spec.normal ? d_normal_stage_ : nullptr, depth_out, rgb_out,
        normal_out, tile_w, stack_h, tile_w, tile_h, batch_size, 1, spec.y_flip,
        spec.normal, stream_);
    check_cuda_last("pack_tiles_rgb_depth(layers)");
    if (spec.output_instance_id && d_instance_id_ && d_id_stage_) {
        uint32_t* id_out = static_cast<uint32_t*>(d_instance_id_) +
                           static_cast<size_t>(out_tile_offset) * tile_pix;
        cuda_kernels::pack_tiles_instance_id(
            d_id_stage_, id_out, tile_w, stack_h, tile_w, tile_h, batch_size, 1,
            spec.y_flip, stream_);
    }

    mapped.unmap("cudaGraphicsUnmapResources(layers)");

    CudaBufferSet out{};
    out.batch_size = out_tile_offset + batch_size;
    out.width = tile_w;
    out.height = tile_h;
    out.rgb = spec.rgb ? d_rgb_ : nullptr;
    out.depth = spec.metric_depth ? d_depth_ : nullptr;
    out.normal = spec.normal ? d_normal_ : nullptr;
    out.instance_id = spec.output_instance_id ? d_instance_id_ : nullptr;
    out.on_device = true;
    last_batch_ = out_tile_offset + batch_size;
    last_w_ = tile_w;
    last_h_ = tile_h;
    last_has_rgb_ = spec.rgb;
    last_has_depth_ = spec.metric_depth;
    last_has_normal_ = spec.normal;
    last_has_instance_id_ = spec.output_instance_id;
    return out;
}

CudaBufferSet CudaPackContext::pack_tiles(const RenderOutputSpec& spec,
                                          uint32_t atlas_w, uint32_t atlas_h,
                                          uint32_t batch_size,
                                          uint32_t tile_w, uint32_t tile_h,
                                          uint32_t atlas_cols,
                                          uint32_t out_tile_offset) {
    ensure_capacity(out_tile_offset + batch_size, tile_w, tile_h, spec);
    ensure_stage(atlas_w, atlas_h, spec.metric_depth, spec.rgb,
                 spec.normal, spec.output_instance_id);
    const uint32_t cols = atlas_cols > 0
        ? atlas_cols
        : std::max(1u, static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<double>(batch_size)))));

    const size_t dpitch = atlas_w * sizeof(float);
    const size_t rpitch = atlas_w * 4 * sizeof(float);
    const size_t tile_pix = static_cast<size_t>(tile_w) * tile_h;
    float* depth_out = spec.metric_depth
        ? static_cast<float*>(d_depth_) + static_cast<size_t>(out_tile_offset) * tile_pix
        : nullptr;
    float* rgb_out = (spec.rgb && d_rgb_)
        ? static_cast<float*>(d_rgb_) + static_cast<size_t>(out_tile_offset) * tile_pix * 3
        : nullptr;
    float* normal_out =
        spec.normal ? static_cast<float*>(d_normal_) + static_cast<size_t>(out_tile_offset) * tile_pix * 3
                    : nullptr;

    cudaGraphicsResource* map_list[4];
    int nres = 0;
    if (spec.metric_depth && depth_tex_.resource) map_list[nres++] = depth_tex_.resource;
    if (rgb_tex_.resource && spec.rgb) map_list[nres++] = rgb_tex_.resource;
    if (normal_tex_.resource && spec.normal) map_list[nres++] = normal_tex_.resource;
    if (id_tex_.resource && spec.output_instance_id) map_list[nres++] = id_tex_.resource;
    CudaGraphicsMapGuard mapped_tiles;
    mapped_tiles.list = map_list;
    mapped_tiles.n = nres;
    mapped_tiles.stream = stream_;
    mapped_tiles.map("cudaGraphicsMapResources(tiles)");

    if (spec.metric_depth && depth_tex_.resource) {
        cudaArray_t arr;
        check_cuda(cudaGraphicsSubResourceGetMappedArray(&arr, depth_tex_.resource, 0, 0),
                   "cudaGraphicsSubResourceGetMappedArray(depth atlas)");
        check_cuda(cudaMemcpy2DFromArrayAsync(d_depth_stage_, dpitch, arr, 0, 0, atlas_w * sizeof(float), atlas_h,
                                              cudaMemcpyDeviceToDevice, stream_),
                   "cudaMemcpy2DFromArrayAsync(depth atlas)");
    }
    if (rgb_tex_.resource && spec.rgb) {
        cudaArray_t arr;
        check_cuda(cudaGraphicsSubResourceGetMappedArray(&arr, rgb_tex_.resource, 0, 0),
                   "cudaGraphicsSubResourceGetMappedArray(rgb atlas)");
        check_cuda(cudaMemcpy2DFromArrayAsync(d_rgb_stage_, rpitch, arr, 0, 0, atlas_w * 4 * sizeof(float), atlas_h,
                                              cudaMemcpyDeviceToDevice, stream_),
                   "cudaMemcpy2DFromArrayAsync(rgb atlas)");
    }
    if (normal_tex_.resource && spec.normal) {
        cudaArray_t arr;
        check_cuda(cudaGraphicsSubResourceGetMappedArray(&arr, normal_tex_.resource, 0, 0),
                   "cudaGraphicsSubResourceGetMappedArray(normal atlas)");
        check_cuda(cudaMemcpy2DFromArrayAsync(d_normal_stage_, rpitch, arr, 0, 0,
                                              atlas_w * 4 * sizeof(float), atlas_h,
                                              cudaMemcpyDeviceToDevice, stream_),
                   "cudaMemcpy2DFromArrayAsync(normal atlas)");
    }
    if (id_tex_.resource && spec.output_instance_id) {
        cudaArray_t arr;
        check_cuda(cudaGraphicsSubResourceGetMappedArray(&arr, id_tex_.resource, 0, 0),
                   "cudaGraphicsSubResourceGetMappedArray(id atlas)");
        check_cuda(cudaMemcpy2DFromArrayAsync(d_id_stage_, atlas_w * sizeof(uint32_t), arr, 0, 0,
                                              atlas_w * sizeof(uint32_t), atlas_h,
                                              cudaMemcpyDeviceToDevice, stream_),
                   "cudaMemcpy2DFromArrayAsync(id atlas)");
    }

    cuda_kernels::pack_tiles_rgb_depth(
        spec.metric_depth ? d_depth_stage_ : nullptr, spec.rgb ? d_rgb_stage_ : nullptr,
        spec.normal ? d_normal_stage_ : nullptr,
        depth_out, rgb_out, normal_out,
        atlas_w, atlas_h, tile_w, tile_h, batch_size, cols, spec.y_flip, spec.normal, stream_);
    check_cuda_last("pack_tiles_rgb_depth(atlas)");
    if (spec.output_instance_id && d_instance_id_ && d_id_stage_) {
        uint32_t* id_out = static_cast<uint32_t*>(d_instance_id_) +
                           static_cast<size_t>(out_tile_offset) * tile_pix;
        cuda_kernels::pack_tiles_instance_id(
            d_id_stage_, id_out, atlas_w, atlas_h, tile_w, tile_h, batch_size, cols,
            spec.y_flip, stream_);
    }

    mapped_tiles.unmap("cudaGraphicsUnmapResources(tiles)");

    CudaBufferSet out{};
    out.batch_size = out_tile_offset + batch_size;
    out.width = tile_w;
    out.height = tile_h;
    out.rgb = spec.rgb ? d_rgb_ : nullptr;
    out.depth = spec.metric_depth ? d_depth_ : nullptr;
    out.normal = spec.normal ? d_normal_ : nullptr;
    out.instance_id = spec.output_instance_id ? d_instance_id_ : nullptr;
    out.on_device = true;
    last_batch_ = out_tile_offset + batch_size;
    last_w_ = tile_w;
    last_h_ = tile_h;
    last_has_rgb_ = spec.rgb;
    last_has_depth_ = spec.metric_depth;
    last_has_normal_ = spec.normal;
    last_has_instance_id_ = spec.output_instance_id;
    return out;
}

CudaBufferSet CudaPackContext::pack_scene(const RenderOutputSpec& spec, uint32_t w, uint32_t h) {
    ensure_capacity(1, w, h, spec);
    ensure_stage(w, h, spec.metric_depth, spec.rgb, false);

    cudaGraphicsResource* map_list[2];
    int nres = 0;
    if (spec.metric_depth && depth_tex_.resource) map_list[nres++] = depth_tex_.resource;
    if (rgb_tex_.resource && spec.rgb) map_list[nres++] = rgb_tex_.resource;
    CudaGraphicsMapGuard mapped_scene;
    mapped_scene.list = map_list;
    mapped_scene.n = nres;
    mapped_scene.stream = stream_;
    mapped_scene.map("cudaGraphicsMapResources(scene)");
    if (spec.metric_depth && depth_tex_.resource) {
        cudaArray_t arr;
        check_cuda(cudaGraphicsSubResourceGetMappedArray(&arr, depth_tex_.resource, 0, 0),
                   "cudaGraphicsSubResourceGetMappedArray(scene depth)");
        check_cuda(cudaMemcpy2DFromArrayAsync(d_depth_stage_, w * sizeof(float), arr, 0, 0,
                                              w * sizeof(float), h, cudaMemcpyDeviceToDevice, stream_),
                   "cudaMemcpy2DFromArrayAsync(scene depth)");
    }
    if (rgb_tex_.resource && spec.rgb) {
        cudaArray_t arr;
        check_cuda(cudaGraphicsSubResourceGetMappedArray(&arr, rgb_tex_.resource, 0, 0),
                   "cudaGraphicsSubResourceGetMappedArray(scene rgb)");
        check_cuda(cudaMemcpy2DFromArrayAsync(d_rgb_stage_, w * 4 * sizeof(float), arr, 0, 0,
                                              w * 4 * sizeof(float), h, cudaMemcpyDeviceToDevice, stream_),
                   "cudaMemcpy2DFromArrayAsync(scene rgb)");
    }
    mapped_scene.unmap("cudaGraphicsUnmapResources(scene)");

    cuda_kernels::pack_scene_rgb_depth(
        spec.metric_depth ? d_depth_stage_ : nullptr, spec.rgb ? d_rgb_stage_ : nullptr,
        spec.metric_depth ? static_cast<float*>(d_depth_) : nullptr,
        spec.rgb ? static_cast<float*>(d_rgb_) : nullptr,
        w, h, spec.y_flip, stream_);
    check_cuda_last("pack_scene_rgb_depth");

    CudaBufferSet out{};
    out.batch_size = 1;
    out.width = w;
    out.height = h;
    out.rgb = spec.rgb ? d_rgb_ : nullptr;
    out.depth = spec.metric_depth ? d_depth_ : nullptr;
    out.on_device = true;
    last_batch_ = 1;
    last_w_ = w;
    last_h_ = h;
    last_has_rgb_ = spec.rgb;
    last_has_depth_ = spec.metric_depth;
    last_has_normal_ = false;
    last_has_instance_id_ = false;
    return out;
}

CudaBufferSet CudaPackContext::pack_tiles_from_cpu(const RenderOutputSpec& spec,
                                                   const float* depth_atlas,
                                                   const float* rgb_rgba_atlas,
                                                   uint32_t atlas_w, uint32_t atlas_h,
                                                   uint32_t batch_size,
                                                   uint32_t tile_w, uint32_t tile_h,
                                                   uint32_t atlas_cols) {
    ensure_capacity(batch_size, tile_w, tile_h, spec);
    ensure_stage(atlas_w, atlas_h, spec.metric_depth, spec.rgb, spec.normal);
    const size_t dpix = static_cast<size_t>(atlas_w) * atlas_h;
    cudaMemcpyAsync(d_depth_stage_, depth_atlas, dpix * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_rgb_stage_, rgb_rgba_atlas, dpix * 4 * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    const uint32_t cols = atlas_cols > 0
        ? atlas_cols
        : std::max(1u, static_cast<uint32_t>(
              std::ceil(std::sqrt(static_cast<double>(batch_size)))));
    cuda_kernels::pack_tiles_rgb_depth(
        d_depth_stage_, d_rgb_stage_, spec.normal ? d_normal_stage_ : nullptr,
        static_cast<float*>(d_depth_), static_cast<float*>(d_rgb_),
        spec.normal ? static_cast<float*>(d_normal_) : nullptr,
        atlas_w, atlas_h, tile_w, tile_h, batch_size, cols, spec.y_flip, spec.normal, stream_);
    cudaStreamSynchronize(stream_);
    CudaBufferSet out{};
    out.batch_size = batch_size;
    out.width = tile_w;
    out.height = tile_h;
    out.rgb = spec.rgb ? d_rgb_ : nullptr;
    out.depth = spec.metric_depth ? d_depth_ : nullptr;
    out.normal = spec.normal ? d_normal_ : nullptr;
    out.on_device = true;
    last_batch_ = batch_size;
    last_w_ = tile_w;
    last_h_ = tile_h;
    last_has_rgb_ = spec.rgb;
    last_has_depth_ = spec.metric_depth;
    last_has_normal_ = spec.normal;
    last_has_instance_id_ = false;
    return out;
}

CudaBufferSet CudaPackContext::pack_scene_from_cpu(const RenderOutputSpec& spec,
                                                   const float* depth_atlas,
                                                   const float* rgb_rgba_atlas,
                                                   uint32_t w, uint32_t h) {
    ensure_capacity(1, w, h, spec);
    ensure_stage(w, h, spec.metric_depth, spec.rgb, false);
    const size_t dpix = static_cast<size_t>(w) * h;
    cudaMemcpyAsync(d_depth_stage_, depth_atlas, dpix * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_rgb_stage_, rgb_rgba_atlas, dpix * 4 * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    cuda_kernels::pack_scene_rgb_depth(
        spec.metric_depth ? d_depth_stage_ : nullptr, spec.rgb ? d_rgb_stage_ : nullptr,
        spec.metric_depth ? static_cast<float*>(d_depth_) : nullptr,
        spec.rgb ? static_cast<float*>(d_rgb_) : nullptr,
        w, h, spec.y_flip, stream_);
    cudaStreamSynchronize(stream_);
    CudaBufferSet out{};
    out.batch_size = 1;
    out.width = w;
    out.height = h;
    out.rgb = spec.rgb ? d_rgb_ : nullptr;
    out.depth = spec.metric_depth ? d_depth_ : nullptr;
    out.on_device = true;
    last_batch_ = 1;
    last_w_ = w;
    last_h_ = h;
    last_has_rgb_ = spec.rgb;
    last_has_depth_ = spec.metric_depth;
    last_has_normal_ = false;
    last_has_instance_id_ = false;
    return out;
}

void CudaPackContext::copy_tiles_to_offset(void* dst_depth, void* dst_rgb,
                                           uint32_t tile_offset, uint32_t tile_count) const {
    if (!dst_depth || !d_depth_ || tile_count == 0 || last_w_ == 0 || last_h_ == 0) {
        return;
    }
    const size_t tile_pix = static_cast<size_t>(last_w_) * last_h_;
    const size_t tile_depth_bytes = tile_pix * sizeof(float);
    const size_t tile_rgb_bytes = tile_pix * 3 * sizeof(float);
    const size_t off = static_cast<size_t>(tile_offset);
    cudaMemcpyAsync(static_cast<char*>(dst_depth) + off * tile_depth_bytes,
                    d_depth_, tile_count * tile_depth_bytes,
                    cudaMemcpyDeviceToDevice, stream_);
    if (dst_rgb && d_rgb_) {
        cudaMemcpyAsync(static_cast<char*>(dst_rgb) + off * tile_rgb_bytes,
                        d_rgb_, tile_count * tile_rgb_bytes,
                        cudaMemcpyDeviceToDevice, stream_);
    }
    cudaStreamSynchronize(stream_);
}

}
