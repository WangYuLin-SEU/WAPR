// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#include "fill_tile_instances.h"

#include <stdexcept>
#include <string>

namespace wapr_ogl {

struct FillArgs {
    const float* poses;
    const float* bboxes;
    InstanceRecordGPU* records;
    uint32_t count;
    float fx, fy, cx, cy;
    float frame_w, frame_h;
    uint32_t tile_w, tile_h, cols;
    const TileMeshData* meshes;
    uint32_t layered;
};

__device__ void dev_mat4_mul(const float* a, const float* b, float* out) {
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float sum = 0.f;
            for (int k = 0; k < 4; ++k) {
                sum += a[row * 4 + k] * b[k * 4 + col];
            }
            out[row * 4 + col] = sum;
        }
    }
}

__device__ void dev_transpose(const float* in, float* out) {
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            out[col * 4 + row] = in[row * 4 + col];
        }
    }
}

__global__ void fill_indexed_tile_instances_kernel(FillArgs args) {
    const int lane = threadIdx.x;
    float zmin = 1e30f;
    float zmax = 0.f;
    int any = 0;
    if (lane < static_cast<int>(args.count) && args.meshes[lane].has_bounds) {
        const TileMeshData& mesh = args.meshes[lane];
        const float* pose = args.poses + static_cast<size_t>(lane) * 16;
        for (int corner = 0; corner < 8; ++corner) {
            const float lx = (corner & 1) ? mesh.bmax[0] : mesh.bmin[0];
            const float ly = (corner & 2) ? mesh.bmax[1] : mesh.bmin[1];
            const float lz = (corner & 4) ? mesh.bmax[2] : mesh.bmin[2];
            const float z = pose[8] * lx + pose[9] * ly + pose[10] * lz + pose[11];
            if (z <= 1e-3f) {
                continue;
            }
            zmin = fminf(zmin, z);
            zmax = fmaxf(zmax, z);
            any = 1;
        }
    }
    __shared__ float shared_min[256];
    __shared__ float shared_max[256];
    __shared__ int shared_any[256];
    shared_min[lane] = zmin;
    shared_max[lane] = zmax;
    shared_any[lane] = any;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            shared_any[lane] |= shared_any[lane + stride];
            shared_min[lane] = fminf(shared_min[lane], shared_min[lane + stride]);
            shared_max[lane] = fmaxf(shared_max[lane], shared_max[lane + stride]);
        }
        __syncthreads();
    }
    __shared__ float znear;
    __shared__ float zfar;
    if (lane == 0) {
        if (shared_any[0] && shared_min[0] < shared_max[0]) {
            znear = fmaxf(1e-3f, shared_min[0] * 0.8f);
            zfar = fmaxf(znear * 10.f, shared_max[0] * 1.2f);
        } else {
            znear = 0.001f;
            zfar = 100.f;
        }
    }
    __syncthreads();
    if (lane >= static_cast<int>(args.count)) {
        return;
    }

    const float* pose = args.poses + static_cast<size_t>(lane) * 16;
    const float* box = args.bboxes + static_cast<size_t>(lane) * 4;
    const double left = static_cast<double>(box[0]);
    const double top_down = static_cast<double>(box[1]);
    const double right = static_cast<double>(box[2]);
    const double bottom_down = static_cast<double>(box[3]);
    const double frame_w = static_cast<double>(args.frame_w);
    const double frame_h = static_cast<double>(args.frame_h);
    const double top = frame_h - top_down;
    const double bottom = frame_h - bottom_down;
    const double box_w = right - left;
    const double box_h = top - bottom;
    const double sx = frame_w / box_w;
    const double sy = frame_h / box_h;
    const double tx = (frame_w - right - left) / box_w;
    const double ty = (frame_h - top - bottom) / box_h;
    float clip[16] = {
        static_cast<float>(sx), 0.f, 0.f, 0.f,
        0.f, static_cast<float>(sy), 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        static_cast<float>(tx), static_cast<float>(ty), 0.f, 1.f};

    const float width = args.frame_w;
    const float height = args.frame_h;
    const float depth = zfar - znear;
    const float q = -(zfar + znear) / depth;
    const float qn = -2.f * (zfar * znear) / depth;
    const float proj[16] = {
        2.f * args.fx / width, 0.f, (-2.f * args.cx + width) / width, 0.f,
        0.f, 2.f * args.fy / height, (2.f * args.cy - height) / height, 0.f,
        0.f, 0.f, q, qn,
        0.f, 0.f, -1.f, 0.f};
    float gl_pose[16];
    for (int col = 0; col < 4; ++col) {
        gl_pose[col] = pose[col];
        gl_pose[4 + col] = -pose[4 + col];
        gl_pose[8 + col] = -pose[8 + col];
        gl_pose[12 + col] = pose[12 + col];
    }
    float mvp[16];
    dev_mat4_mul(proj, gl_pose, mvp);
    float clip_t[16];
    dev_transpose(clip, clip_t);
    float mvp_clip[16];
    dev_mat4_mul(clip_t, mvp, mvp_clip);

    InstanceRecordGPU record{};
    dev_transpose(mvp_clip, record.mvp);
    dev_transpose(pose, record.model_cam);
    for (int i = 0; i < 16; ++i) {
        record.clip_affine[i] = clip[i];
    }
    record.mesh_id = args.meshes[lane].mesh_id;
    if (args.layered) {
        record.tile_x = 0;
        record.tile_y = 0;
    } else {
        const uint32_t slot = static_cast<uint32_t>(lane);
        record.tile_x = (slot % args.cols) * args.tile_w;
        record.tile_y = (slot / args.cols) * args.tile_h;
    }
    record.depth_func = 0x0201u;
    record.tex_layer = args.meshes[lane].tex_layer;
    record.force_flat_normal = args.meshes[lane].force_flat_normal;
    record.metallic = 0.f;
    record.roughness = 0.5f;
    record.tile_slot = lane;
    record.instance_id = 0;
    record.fx = args.fx;
    record.fy = args.fy;
    record.cx = args.cx;
    record.cy = args.cy;
    record.has_clip_affine = 1;
    record._pad1 = 0;
    args.records[lane] = record;
}

void fill_indexed_tile_instances(const float* poses,
                                 const float* bboxes,
                                 InstanceRecordGPU* records,
                                 uint32_t count,
                                 float fx, float fy, float cx, float cy,
                                 float frame_w, float frame_h,
                                 uint32_t tile_w, uint32_t tile_h, uint32_t cols,
                                 const TileMeshData* meshes,
                                 uint32_t layered,
                                 cudaStream_t stream) {
    if (count == 0 || count > 256) {
        throw std::runtime_error("device tile count must be 1..256");
    }
    FillArgs args{};
    args.poses = poses;
    args.bboxes = bboxes;
    args.records = records;
    args.count = count;
    args.fx = fx;
    args.fy = fy;
    args.cx = cx;
    args.cy = cy;
    args.frame_w = frame_w;
    args.frame_h = frame_h;
    args.tile_w = tile_w;
    args.tile_h = tile_h;
    args.cols = cols == 0 ? 1u : cols;
    args.meshes = meshes;
    args.layered = layered;
    fill_indexed_tile_instances_kernel<<<1, 256, 0, stream>>>(args);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("fill_indexed_tile_instances: ") + cudaGetErrorString(err));
    }
}

}
