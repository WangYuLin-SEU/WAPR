#version 460 core
// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#extension GL_ARB_shader_viewport_layer_array : enable

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_color;
layout(location = 3) in vec2 a_uv;

struct InstanceRecord {
    mat4 mvp;
    mat4 model_cam;
    mat4 clip_affine;
    uint mesh_id;
    uint tile_x;
    uint tile_y;
    uint depth_func;
    int tex_layer;
    uint force_flat_normal;
    float metallic;
    float roughness;
    int tile_slot;
    int instance_id;
    float fx;
    float fy;
    float cx;
    float cy;
    int has_clip_affine;
    int _pad1;
};

layout(std430, binding = 0) readonly buffer InstanceRecords {
    InstanceRecord instances[];
};

uniform vec2 u_atlas_size;
uniform vec2 u_tile_size;
uniform int u_scene_mode;
uniform int u_apply_tile_remap;
uniform int u_layered;
uniform int u_indexed_viewport;
uniform vec2 u_tile_offset;

uniform int u_use_distortion;
uniform int u_has_clip_affine;
uniform mat4 u_clip_affine;
uniform vec4 u_K;
uniform vec4 u_dist_k;
uniform float u_dist_k3;
uniform vec2 u_viewport;
uniform vec2 u_proj_z;

out VS_OUT {
    vec3 pos_cam;
    vec3 normal_cam;
    vec3 color;
    vec2 uv;
    flat int tex_layer;
    flat int force_flat_normal;
    flat float metallic;
    flat float roughness;
    flat int tile_slot;
    flat int instance_id;
} vs_out;

vec2 opencv5_forward(vec2 xyn) {
    float x = xyn.x;
    float y = xyn.y;
    float r2 = x * x + y * y;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    float cdist = 1.0 + u_dist_k.x * r2 + u_dist_k.y * r4 + u_dist_k3 * r6;
    float xd = x * cdist + 2.0 * u_dist_k.z * x * y + u_dist_k.w * (r2 + 2.0 * x * x);
    float yd = y * cdist + u_dist_k.z * (r2 + 2.0 * y * y) + 2.0 * u_dist_k.w * x * y;
    return vec2(xd, yd);
}

vec4 project_distorted(vec3 pc, vec4 K) {
    float z = pc.z;
    if (z < 1e-4) {
        return vec4(0.0, 0.0, 2.0, 1.0);
    }
    vec2 xyn = vec2(pc.x / z, pc.y / z);
    vec2 d = opencv5_forward(xyn);
    float u = K.x * d.x + K.z;
    float v = K.y * d.y + K.w;
    float ndc_x = 2.0 * u / max(u_viewport.x, 1.0) - 1.0;
    float ndc_y = 1.0 - 2.0 * v / max(u_viewport.y, 1.0);
    float clip_z = u_proj_z.x * (-z) + u_proj_z.y;
    float clip_w = z;
    return vec4(ndc_x * clip_w, ndc_y * clip_w, clip_z, clip_w);
}

void main() {
    InstanceRecord rec = instances[gl_BaseInstance + gl_InstanceID];
    vec4 clip;
    if (u_use_distortion > 0) {
        vec3 pc = (rec.model_cam * vec4(a_pos, 1.0)).xyz;
        vec4 K = (rec.fx > 0.0) ? vec4(rec.fx, rec.fy, rec.cx, rec.cy) : u_K;
        clip = project_distorted(pc, K);
        if (rec.has_clip_affine > 0) {
            clip = rec.clip_affine * clip;
        }
        if (u_scene_mode == 0 && u_apply_tile_remap > 0) {
            vec2 ndc = clip.xy / max(abs(clip.w), 1e-6);
            float tw = u_tile_size.x;
            float th = u_tile_size.y;
            float aw = u_atlas_size.x;
            float ah = u_atlas_size.y;
            float nx = (ndc.x + 1.0) * 0.5;
            float ny = (ndc.y + 1.0) * 0.5;
            nx = (nx * tw + float(rec.tile_x)) / aw * 2.0 - 1.0;
            ny = (ny * th + float(rec.tile_y)) / ah * 2.0 - 1.0;
            clip.xy = vec2(nx, ny) * clip.w;
        }
    } else {
        clip = rec.mvp * vec4(a_pos, 1.0);
        if (u_scene_mode == 0 && u_apply_tile_remap > 0) {
            vec2 ndc = clip.xy / max(abs(clip.w), 1e-6);
            float tw = u_tile_size.x;
            float th = u_tile_size.y;
            float aw = u_atlas_size.x;
            float ah = u_atlas_size.y;
            float nx = (ndc.x + 1.0) * 0.5;
            float ny = (ndc.y + 1.0) * 0.5;
            nx = (nx * tw + float(rec.tile_x)) / aw * 2.0 - 1.0;
            ny = (ny * th + float(rec.tile_y)) / ah * 2.0 - 1.0;
            clip.xy = vec2(nx, ny) * clip.w;
        }
    }
    vs_out.pos_cam = (rec.model_cam * vec4(a_pos, 1.0)).xyz;
    vs_out.normal_cam = mat3(rec.model_cam) * a_normal;
    vs_out.color = a_color;
    vs_out.uv = a_uv;
    vs_out.tex_layer = int(rec.tex_layer);
    vs_out.force_flat_normal = int(rec.force_flat_normal);
    vs_out.metallic = rec.metallic;
    vs_out.roughness = rec.roughness;
    vs_out.tile_slot = rec.tile_slot;
    vs_out.instance_id = rec.instance_id;
    gl_Position = clip;
    gl_ClipDistance[0] = vs_out.pos_cam.z - 1e-3;
    if (u_layered > 0) {
        gl_Layer = max(rec.tile_slot, 0);
    }
    if (u_indexed_viewport > 0) {
        gl_ViewportIndex = max(rec.tile_slot, 0);
    }
}
