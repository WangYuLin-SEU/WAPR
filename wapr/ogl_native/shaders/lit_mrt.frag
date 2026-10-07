#version 460 core
// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.


layout(location = 0) out float out_depth;
layout(location = 1) out vec4 out_rgb;
layout(location = 2) out vec4 out_normal;
layout(location = 3) out vec4 out_coord;
layout(location = 4) out uint out_instance_id;

in VS_OUT {
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
} fs_in;

uniform float u_ambient;
uniform float u_diffuse;
uniform vec3 u_light_dir;
uniform int u_output_normal;
uniform int u_output_coord;
uniform int u_output_instance_id;
uniform int u_use_texture;
uniform sampler2DArray u_tex_array;

uniform int u_use_pbr;
uniform vec3 u_ambient_color;
uniform vec3 u_point_pos;
uniform vec3 u_point_color;
uniform float u_point_intensity;
uniform vec3 u_dir_color;
uniform float u_dir_intensity;

uniform int u_use_tile_lights;
struct LightRecord {
    float ambient;
    float diffuse;
    int use_pbr;
    int _pad0;
    vec4 light_dir;
    vec4 ambient_color;
    vec4 point_pos;
    vec4 point_color;
    vec4 dir_color;
};
layout(std430, binding = 2) readonly buffer TileLights {
    LightRecord lights[];
};

const float PI = 3.14159265358979323846;
const float MIN_ROUGHNESS = 0.04;

vec3 safe_normal(vec3 n, vec3 pos_cam) {
    float len = length(n);
    if (len < 1e-4) {
        return normalize(-pos_cam);
    }
    n = n / len;
    if (dot(n, normalize(-pos_cam)) < 0.0) {
        n = -n;
    }
    return n;
}

vec3 flat_normal_from_position(vec3 pos_cam) {
    vec3 n = cross(dFdx(pos_cam), dFdy(pos_cam));
    float len = length(n);
    if (len < 1e-8) {
        return normalize(vec3(0.0, 0.0, 1.0));
    }
    n = n / len;
    if (dot(n, normalize(-pos_cam)) <= 0.0) {
        n = -n;
    }
    return n;
}

vec3 srgb_to_linear(vec3 srgb) {
    bvec3 cutoff = lessThanEqual(srgb, vec3(0.04045));
    vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    vec3 lower = srgb / vec3(12.92);
    return mix(higher, lower, cutoff);
}

vec3 linear_to_srgb(vec3 lin) {
    bvec3 cutoff = lessThanEqual(lin, vec3(0.0031308));
    vec3 higher = vec3(1.055) * pow(lin, vec3(1.0 / 2.4)) - vec3(0.055);
    vec3 lower = lin * vec3(12.92);
    return mix(higher, lower, cutoff);
}

vec3 soft_clip_highlight(vec3 x) {
    const float knee = 0.90;
    const float head = 0.08;
    vec3 y = min(x, vec3(knee));
    vec3 over = max(x - knee, 0.0);
    return y + head * over / (1.0 + over);
}

vec3 mild_spec_cap(vec3 lit, vec3 base) {
    vec3 floorc = max(base, vec3(0.04));
    vec3 cap = floorc * 1.15 + vec3(0.10);
    return min(lit, cap);
}

vec3 compute_brdf(vec3 n, vec3 v, vec3 l, float roughness, float metallic,
                  vec3 f0, vec3 c_diff, vec3 radiance) {
    vec3 h = normalize(l + v);
    float nl = clamp(dot(n, l), 0.001, 1.0);
    float nv = clamp(abs(dot(n, v)), 0.001, 1.0);
    float nh = clamp(dot(n, h), 0.0, 1.0);
    float vh = clamp(dot(v, h), 0.0, 1.0);

    float a = roughness * roughness;
    float a2 = a * a;
    float denom = (nh * nh) * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * denom * denom);

    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float g1 = nv / (nv * (1.0 - k) + k);
    float g2 = nl / (nl * (1.0 - k) + k);
    float G = g1 * g2;

    vec3 F = f0 + (1.0 - f0) * pow(clamp(1.0 - vh, 0.0, 1.0), 5.0);
    vec3 diffuse_contrib = (1.0 - F) * c_diff / PI;
    vec3 spec_contrib = F * G * D / (4.0 * nl * nv + 0.001);
    return nl * radiance * (diffuse_contrib + spec_contrib);
}

void main() {
    float z = fs_in.pos_cam.z;
    if (z < 1e-3) {
        discard;
    }
    out_depth = z;
    vec3 base_color = fs_in.color;
    if (u_use_texture > 0 && fs_in.tex_layer >= 0) {
        base_color = texture(u_tex_array, vec3(fs_in.uv, float(fs_in.tex_layer))).rgb;
    }
    vec3 n = (fs_in.force_flat_normal > 0)
        ? flat_normal_from_position(fs_in.pos_cam)
        : safe_normal(fs_in.normal_cam, fs_in.pos_cam);

    float ambient = u_ambient;
    float diffuse = u_diffuse;
    vec3 light_dir = u_light_dir;
    int use_pbr = u_use_pbr;
    vec3 ambient_color = u_ambient_color;
    vec3 point_pos = u_point_pos;
    vec3 point_color = u_point_color;
    float point_intensity = u_point_intensity;
    vec3 dir_color = u_dir_color;
    float dir_intensity = u_dir_intensity;
    if (u_use_tile_lights > 0) {
        LightRecord lr = lights[max(fs_in.tile_slot, 0)];
        ambient = lr.ambient;
        diffuse = lr.diffuse;
        light_dir = lr.light_dir.xyz;
        use_pbr = lr.use_pbr;
        ambient_color = lr.ambient_color.xyz;
        point_pos = lr.point_pos.xyz;
        point_color = lr.point_color.rgb;
        point_intensity = lr.point_color.w;
        dir_color = lr.dir_color.rgb;
        dir_intensity = lr.dir_color.w;
    }

    vec3 lit;
    if (use_pbr > 0) {
        base_color = clamp(base_color, 0.0, 1.0);
        float metallic = clamp(fs_in.metallic, 0.0, 1.0);
        float roughness = clamp(fs_in.roughness, MIN_ROUGHNESS, 1.0);
        vec3 dialectric_spec = vec3(MIN_ROUGHNESS);
        vec3 c_diff = mix(vec3(0.0), base_color * (1.0 - MIN_ROUGHNESS), 1.0 - metallic);
        vec3 f0 = mix(dialectric_spec, base_color, metallic);
        vec3 v = normalize(-fs_in.pos_cam);

        lit = base_color * ambient_color;

        vec3 to_l = point_pos - fs_in.pos_cam;
        float dist = max(length(to_l), 1e-3);
        vec3 l_p = to_l / dist;
        float atten = point_intensity / (dist * dist);
        vec3 radiance_p = atten * point_color;
        lit += compute_brdf(n, v, l_p, roughness, metallic, f0, c_diff, radiance_p);

        vec3 l_d = normalize(-light_dir);
        vec3 radiance_d = dir_intensity * dir_color;
        lit += compute_brdf(n, v, l_d, roughness, metallic, f0, c_diff, radiance_d);
    } else {
        vec3 l = normalize(-light_dir);
        float diff = max(dot(n, l), 0.0);
        lit = base_color * ambient + base_color * diff * diffuse;
        if (dir_intensity > 1e-6) {
            vec3 v = normalize(-fs_in.pos_cam);
            vec3 h = normalize(l + v);
            float shininess = mix(8.0, 128.0, 1.0 - clamp(fs_in.roughness, 0.04, 1.0));
            float spec = pow(max(dot(n, h), 0.0), shininess);
            lit += spec * dir_intensity * dir_color * base_color;
        }
    }
    out_rgb = vec4(clamp(soft_clip_highlight(mild_spec_cap(lit, base_color)), 0.0, 1.0), 1.0);
    out_normal = (u_output_normal > 0) ? vec4(n * 0.5 + 0.5, 1.0) : vec4(0.0);
    out_coord = (u_output_coord > 0) ? vec4(fs_in.pos_cam, 1.0) : vec4(0.0);
    out_instance_id = (u_output_instance_id > 0) ? uint(max(fs_in.instance_id, 0)) : 0u;
}
