// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#include <stdexcept>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gpu_render_runtime.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace py = pybind11;

template <typename T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
static CArray<T> require_c_array(py::handle value, const char* name) {
    auto out = CArray<T>::ensure(value);
    if (!out) {
        throw std::invalid_argument(std::string(name) + " must be convertible to a C-contiguous array");
    }
    return out;
}

static void require_finite_f32(const float* values, size_t count, const char* name) {
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            throw std::invalid_argument(std::string(name) + " contains NaN or Inf");
        }
    }
}

static wapr_ogl::CameraIntrinsics parse_K(py::array K, uint32_t w, uint32_t h) {
    wapr_ogl::CameraIntrinsics cam{};
    cam.width = w;
    cam.height = h;
    auto arr = require_c_array<double>(K, "K");
    auto buf = arr.request();
    if (buf.ndim != 2 || buf.shape[0] != 3 || buf.shape[1] != 3) {
        throw std::invalid_argument("K must have shape (3, 3)");
    }
    const double* p = static_cast<const double*>(buf.ptr);
    for (size_t i = 0; i < 9; ++i) {
        if (!std::isfinite(p[i])) throw std::invalid_argument("K contains NaN or Inf");
    }
    cam.fx = static_cast<float>(p[0]);
    cam.fy = static_cast<float>(p[4]);
    cam.cx = static_cast<float>(p[2]);
    cam.cy = static_cast<float>(p[5]);
    if (!(cam.fx > 0.f) || !(cam.fy > 0.f)) {
        throw std::invalid_argument("K focal lengths must be finite and > 0");
    }
    return cam;
}

static void apply_distortion(wapr_ogl::CameraIntrinsics& cam, py::object dist_coeffs) {
    if (dist_coeffs.is_none()) return;
    auto d = require_c_array<float>(dist_coeffs, "dist_coeffs");
    auto dr = d.request();
    if (dr.ndim > 2 || dr.size > 5) {
        throw std::invalid_argument("dist_coeffs must contain at most 5 values");
    }
    const float* p = static_cast<const float*>(dr.ptr);
    const size_t n = static_cast<size_t>(dr.size);
    require_finite_f32(p, n, "dist_coeffs");
    for (size_t i = 0; i < n; ++i) cam.dist[i] = p[i];
    cam.use_distortion = 0;
    for (int i = 0; i < 5; ++i) {
        if (std::abs(cam.dist[i]) > 1e-12f) {
            cam.use_distortion = 1;
            break;
        }
    }
}

static wapr_ogl::MeshUpload mesh_upload_from_numpy(const std::string& name,
                                              py::array vertices,
                                              py::array faces,
                                              py::object normals,
                                              py::object vertex_colors,
                                              py::object uvs,
                                              py::object texture_rgb,
                                              uint32_t tex_w,
                                              uint32_t tex_h,
                                              bool uv_per_corner,
                                              bool force_flat_normal) {
    wapr_ogl::MeshUpload mesh{};
    mesh.name = name;
    auto vertices_c = require_c_array<float>(vertices, "vertices");
    auto faces_c = require_c_array<uint32_t>(faces, "faces");
    auto vr = vertices_c.request();
    auto fr = faces_c.request();
    if (vr.ndim != 2 || vr.shape[1] != 3 || vr.shape[0] <= 0) {
        throw std::invalid_argument("vertices must have non-empty shape (N, 3)");
    }
    const bool faces_shaped = fr.ndim == 2 && fr.shape[1] == 3 && fr.shape[0] > 0;
    const bool faces_flat = fr.ndim == 1 && fr.size > 0 && (fr.size % 3) == 0;
    if (!faces_shaped && !faces_flat) {
        throw std::invalid_argument("faces must have shape (F, 3) or flat length F*3");
    }
    require_finite_f32(static_cast<const float*>(vr.ptr), static_cast<size_t>(vr.size), "vertices");
    const auto* face_ptr = static_cast<const uint32_t*>(fr.ptr);
    for (py::ssize_t i = 0; i < fr.size; ++i) {
        if (face_ptr[i] >= static_cast<uint32_t>(vr.shape[0])) {
            throw std::invalid_argument("faces contains an out-of-range vertex index");
        }
    }
    mesh.vertices.assign(static_cast<float*>(vr.ptr),
                         static_cast<float*>(vr.ptr) + vr.size);
    mesh.indices.assign(static_cast<uint32_t*>(fr.ptr),
                         static_cast<uint32_t*>(fr.ptr) + fr.size);
    if (!normals.is_none()) {
        auto n = require_c_array<float>(normals, "normals");
        auto nr = n.request();
        const bool shaped = nr.ndim == 2 && nr.shape[0] == vr.shape[0] && nr.shape[1] == 3;
        const bool flat = nr.ndim == 1 && nr.size == vr.shape[0] * 3;
        if (!shaped && !flat) {
            throw std::invalid_argument("normals must have shape (N, 3) or flat length N*3");
        }
        require_finite_f32(static_cast<const float*>(nr.ptr), static_cast<size_t>(nr.size), "normals");
        mesh.normals.assign(static_cast<float*>(nr.ptr),
                             static_cast<float*>(nr.ptr) + nr.size);
    }
    if (!vertex_colors.is_none()) {
        auto c = require_c_array<float>(vertex_colors, "vertex_colors");
        auto cr = c.request();
        const bool shaped = cr.ndim == 2 && cr.shape[0] == vr.shape[0] && cr.shape[1] == 3;
        const bool flat = cr.ndim == 1 && cr.size == vr.shape[0] * 3;
        if (!shaped && !flat) {
            throw std::invalid_argument("vertex_colors must have shape (N, 3) or flat length N*3");
        }
        require_finite_f32(static_cast<const float*>(cr.ptr), static_cast<size_t>(cr.size), "vertex_colors");
        mesh.vertex_colors.assign(static_cast<float*>(cr.ptr),
                                  static_cast<float*>(cr.ptr) + cr.size);
    }
    if (!uvs.is_none()) {
        auto u = require_c_array<float>(uvs, "uvs");
        auto ur = u.request();
        const py::ssize_t expected_uv_rows = uv_per_corner ? fr.size : vr.shape[0];
        const bool shaped = ur.ndim == 2 && ur.shape[0] == expected_uv_rows && ur.shape[1] == 2;
        const bool flat = ur.ndim == 1 && ur.size == expected_uv_rows * 2;
        if (!shaped && !flat) {
            throw std::invalid_argument(uv_per_corner
                ? "corner uvs must have shape (F*3, 2)"
                : "vertex uvs must have shape (N, 2), matching vertices");
        }
        require_finite_f32(static_cast<const float*>(ur.ptr), static_cast<size_t>(ur.size), "uvs");
        mesh.uvs.assign(static_cast<float*>(ur.ptr),
                        static_cast<float*>(ur.ptr) + ur.size);
    }
    if (!texture_rgb.is_none()) {
        auto t = require_c_array<uint8_t>(texture_rgb, "texture_rgb");
        auto tr = t.request();
        const bool shaped = tr.ndim == 3 && tr.shape[2] == 3 && tr.shape[0] > 0 &&
                            tr.shape[1] > 0 && tex_w == static_cast<uint32_t>(tr.shape[1]) &&
                            tex_h == static_cast<uint32_t>(tr.shape[0]);
        const bool flat = tr.ndim == 1 && tex_w > 0 && tex_h > 0 &&
                          static_cast<size_t>(tr.size) ==
                              static_cast<size_t>(tex_w) * tex_h * 3u;
        if (!shaped && !flat) {
            throw std::invalid_argument(
                "texture_rgb must have shape (H, W, 3) or flat length H*W*3 matching tex_w/tex_h");
        }
        if (tex_w == 0 || tex_h == 0) {
            throw std::invalid_argument("tex_w/tex_h must match texture_rgb shape");
        }
        mesh.texture_rgb.assign(static_cast<uint8_t*>(tr.ptr),
                                static_cast<uint8_t*>(tr.ptr) + tr.size);
        mesh.tex_w = tex_w;
        mesh.tex_h = tex_h;
    }
    mesh.uv_per_corner = uv_per_corner ? 1 : 0;
    mesh.force_flat_normal = force_flat_normal ? 1 : 0;
    return mesh;
}

static void copy_f32_exact(py::handle value, float* dst, size_t count, const char* name) {
    auto arr = require_c_array<float>(value, name);
    const auto buf = arr.request();
    if (static_cast<size_t>(buf.size) != count) {
        throw std::invalid_argument(std::string(name) + " has the wrong number of elements");
    }
    const auto* src = static_cast<const float*>(buf.ptr);
    require_finite_f32(src, count, name);
    std::memcpy(dst, src, count * sizeof(float));
}

static std::vector<wapr_ogl::RenderInstance> parse_instances(py::list instances) {
    std::vector<wapr_ogl::RenderInstance> out;
    out.reserve(instances.size());
    for (auto item : instances) {
        py::dict d = py::cast<py::dict>(item);
        wapr_ogl::RenderInstance ri{};
        ri.mesh_id = d.contains("mesh_id") ? py::cast<uint32_t>(d["mesh_id"]) : 0u;
        ri.depth_func = d.contains("depth_func") ? py::cast<uint32_t>(d["depth_func"]) : wapr_ogl::kDepthFuncLess;
        ri.tile_slot = d.contains("tile_slot") ? py::cast<uint32_t>(d["tile_slot"]) : 0u;
        ri.instance_id = d.contains("instance_id") ? py::cast<uint32_t>(d["instance_id"]) : 0u;
        if (d.contains("T_cam_obj")) {
            copy_f32_exact(d["T_cam_obj"], ri.T_cam_obj, 16, "T_cam_obj");
        }
        if (!d.contains("T_cam_obj") && !d.contains("mvp")) {
            throw std::invalid_argument(
                "each render instance requires T_cam_obj, or both mvp and model_cam");
        }
        if (d.contains("mvp") && !d.contains("model_cam")) {
            throw std::invalid_argument("precomputed mvp requires model_cam");
        }
        if (d.contains("view_warp")) {
            copy_f32_exact(d["view_warp"], ri.view_warp, 9, "view_warp");
            ri.has_view_warp = 1;
        }
        if (d.contains("clip_affine4") || d.contains("clip_warp4") || d.contains("roi_zoom4")) {
            py::object key = py::str("clip_affine4");
            if (!d.contains("clip_affine4")) {
                key = d.contains("clip_warp4") ? py::str("clip_warp4") : py::str("roi_zoom4");
            }
            copy_f32_exact(d[key], ri.clip_affine4, 16, "clip_affine4");
            ri.has_clip_affine = 1;
        }
        if (d.contains("bbox_xywh")) {
            copy_f32_exact(d["bbox_xywh"], ri.bbox_xywh, 4, "bbox_xywh");
            if (!(ri.bbox_xywh[2] > 0.f) || !(ri.bbox_xywh[3] > 0.f)) {
                throw std::invalid_argument("bbox_xywh width and height must be > 0");
            }
            ri.has_bbox = 1;
        }
        if (d.contains("K")) {
            py::array Ki = py::cast<py::array>(d["K"]);
            auto camk = parse_K(Ki, 0, 0);
            ri.fx = camk.fx;
            ri.fy = camk.fy;
            ri.cx = camk.cx;
            ri.cy = camk.cy;
            ri.has_K = 1;
        }
        if (d.contains("mvp")) {
            copy_f32_exact(d["mvp"], ri.mvp_pre, 16, "mvp");
            ri.has_precomputed_mvp = 1;
        }
        if (d.contains("model_cam")) {
            copy_f32_exact(d["model_cam"], ri.model_cam_pre, 16, "model_cam");
        }
        if (d.contains("metallic") || d.contains("roughness")) {
            ri.has_material = 1;
            if (d.contains("metallic")) {
                ri.metallic = py::cast<float>(d["metallic"]);
            }
            if (d.contains("roughness")) {
                ri.roughness = py::cast<float>(d["roughness"]);
            }
        }
        out.push_back(ri);
    }
    return out;
}

PYBIND11_MODULE(_gpu_render, m) {
    m.doc() = "";

    py::enum_<wapr_ogl::ProjectionMode>(m, "ProjectionMode")
        .value("ClipAffine", wapr_ogl::ProjectionMode::ClipAffine)
        .value("PinholeCrop", wapr_ogl::ProjectionMode::PinholeCrop);

    py::enum_<wapr_ogl::RenderMode>(m, "RenderMode")
        .value("Tile", wapr_ogl::RenderMode::Tile)
        .value("Scene", wapr_ogl::RenderMode::Scene);

    py::enum_<wapr_ogl::DataLayout>(m, "DataLayout")
        .value("NHWC", wapr_ogl::DataLayout::NHWC)
        .value("NCHW", wapr_ogl::DataLayout::NCHW);

    py::class_<wapr_ogl::RenderOutputSpec>(m, "RenderOutputSpec")
        .def(py::init<>())
        .def_readwrite("rgb", &wapr_ogl::RenderOutputSpec::rgb)
        .def_readwrite("metric_depth", &wapr_ogl::RenderOutputSpec::metric_depth)
        .def_readwrite("normal", &wapr_ogl::RenderOutputSpec::normal)
        .def_readwrite("object_coord", &wapr_ogl::RenderOutputSpec::object_coord)
        .def_readwrite("tile_w", &wapr_ogl::RenderOutputSpec::tile_w)
        .def_readwrite("tile_h", &wapr_ogl::RenderOutputSpec::tile_h)
        .def_readwrite("mode", &wapr_ogl::RenderOutputSpec::mode)
        .def_readwrite("projection", &wapr_ogl::RenderOutputSpec::projection)
        .def_readwrite("layout", &wapr_ogl::RenderOutputSpec::layout)
        .def_readwrite("y_flip", &wapr_ogl::RenderOutputSpec::y_flip)
        .def_readwrite("max_atlas_w", &wapr_ogl::RenderOutputSpec::max_atlas_w)
        .def_readwrite("max_atlas_h", &wapr_ogl::RenderOutputSpec::max_atlas_h)
        .def_readwrite("use_scissor", &wapr_ogl::RenderOutputSpec::use_scissor)
        .def_readwrite("indexed_viewport_scissor", &wapr_ogl::RenderOutputSpec::indexed_viewport_scissor)
        .def_readwrite("viewport_per_tile", &wapr_ogl::RenderOutputSpec::viewport_per_tile)
        .def_readwrite("scene_batch", &wapr_ogl::RenderOutputSpec::scene_batch)
        .def_readwrite("layered_tiles", &wapr_ogl::RenderOutputSpec::layered_tiles)
        .def_readwrite("output_instance_id", &wapr_ogl::RenderOutputSpec::output_instance_id)
        .def_readwrite("msaa_samples", &wapr_ogl::RenderOutputSpec::msaa_samples)
        .def_readwrite("sample_shading", &wapr_ogl::RenderOutputSpec::sample_shading)
        .def_readwrite("persistent_buffers", &wapr_ogl::RenderOutputSpec::persistent_buffers);

    py::class_<wapr_ogl::GlCaps>(m, "GlCaps")
        .def(py::init<>())
        .def_readonly("max_viewports", &wapr_ogl::GlCaps::max_viewports)
        .def_readonly("multi_draw_indirect", &wapr_ogl::GlCaps::multi_draw_indirect)
        .def_readonly("shader_draw_parameters", &wapr_ogl::GlCaps::shader_draw_parameters)
        .def_readonly("viewport_array", &wapr_ogl::GlCaps::viewport_array)
        .def_readonly("shader_viewport_layer_array", &wapr_ogl::GlCaps::shader_viewport_layer_array)
        .def_readonly("buffer_storage", &wapr_ogl::GlCaps::buffer_storage)
        .def_readonly("map_persistent", &wapr_ogl::GlCaps::map_persistent)
        .def_readonly("sync", &wapr_ogl::GlCaps::sync);

    auto set_vec3 = [](float* dst, py::object obj) {
        if (obj.is_none()) return;
        py::array_t<float> a = py::cast<py::array_t<float>>(obj);
        auto r = a.request();
        if (r.size < 3) throw std::runtime_error("vec3 needs 3 floats");
        const float* p = static_cast<const float*>(r.ptr);
        dst[0] = p[0];
        dst[1] = p[1];
        dst[2] = p[2];
    };
    auto get_vec3 = [](const float* src) {
        py::array_t<float> out(3);
        auto r = out.request();
        float* p = static_cast<float*>(r.ptr);
        p[0] = src[0];
        p[1] = src[1];
        p[2] = src[2];
        return out;
    };

    py::class_<wapr_ogl::LightParams>(m, "LightParams")
        .def(py::init<>())
        .def(py::init([](float ambient, float diffuse, py::object light_dir) {
                wapr_ogl::LightParams lp;
                lp.ambient = ambient;
                lp.diffuse = diffuse;
                if (!light_dir.is_none()) {
                    py::sequence s = py::cast<py::sequence>(light_dir);
                    if (py::len(s) >= 3) {
                        lp.light_dir[0] = py::cast<float>(s[0]);
                        lp.light_dir[1] = py::cast<float>(s[1]);
                        lp.light_dir[2] = py::cast<float>(s[2]);
                    }
                }
                return lp;
             }),
             py::arg("ambient") = 0.8f,
             py::arg("diffuse") = 0.2f,
             py::arg("light_dir") = py::none())
        .def_readwrite("ambient", &wapr_ogl::LightParams::ambient)
        .def_readwrite("diffuse", &wapr_ogl::LightParams::diffuse)
        .def_readwrite("use_pbr", &wapr_ogl::LightParams::use_pbr)
        .def_readwrite("point_intensity", &wapr_ogl::LightParams::point_intensity)
        .def_readwrite("dir_intensity", &wapr_ogl::LightParams::dir_intensity)
        .def_property(
            "light_dir",
            [get_vec3](const wapr_ogl::LightParams& lp) { return get_vec3(lp.light_dir); },
            [set_vec3](wapr_ogl::LightParams& lp, py::object d) { set_vec3(lp.light_dir, d); })
        .def_property(
            "ambient_color",
            [get_vec3](const wapr_ogl::LightParams& lp) { return get_vec3(lp.ambient_color); },
            [set_vec3](wapr_ogl::LightParams& lp, py::object d) { set_vec3(lp.ambient_color, d); })
        .def_property(
            "point_pos",
            [get_vec3](const wapr_ogl::LightParams& lp) { return get_vec3(lp.point_pos); },
            [set_vec3](wapr_ogl::LightParams& lp, py::object d) { set_vec3(lp.point_pos, d); })
        .def_property(
            "point_color",
            [get_vec3](const wapr_ogl::LightParams& lp) { return get_vec3(lp.point_color); },
            [set_vec3](wapr_ogl::LightParams& lp, py::object d) { set_vec3(lp.point_color, d); })
        .def_property(
            "dir_color",
            [get_vec3](const wapr_ogl::LightParams& lp) { return get_vec3(lp.dir_color); },
            [set_vec3](wapr_ogl::LightParams& lp, py::object d) { set_vec3(lp.dir_color, d); });

    py::class_<wapr_ogl::GpuRenderRuntime>(m, "GpuRenderRuntime")
        .def(py::init<int, std::string>(), py::arg("device") = 0, py::arg("shader_dir") = "")
        .def("load_mesh_from_arrays", [](wapr_ogl::GpuRenderRuntime& self, const std::string& name,
                                         py::array_t<float> vertices,
                                         py::array_t<uint32_t> faces,
                                         py::object normals,
                                         py::object vertex_colors,
                                         py::object uvs,
                                         py::object texture_rgb,
                                         uint32_t tex_w,
                                         uint32_t tex_h,
                                         bool uv_per_corner,
                                         bool force_flat_normal) {
            return self.load_mesh(mesh_upload_from_numpy(
                name, vertices, faces, normals, vertex_colors, uvs, texture_rgb,
                tex_w, tex_h, uv_per_corner, force_flat_normal));
        },
             py::arg("name"),
             py::arg("vertices"),
             py::arg("faces"),
             py::arg("normals") = py::none(),
             py::arg("vertex_colors") = py::none(),
             py::arg("uvs") = py::none(),
             py::arg("texture_rgb") = py::none(),
             py::arg("tex_w") = 0,
             py::arg("tex_h") = 0,
             py::arg("uv_per_corner") = false,
             py::arg("force_flat_normal") = false)
        .def("load_meshes_from_arrays", [](wapr_ogl::GpuRenderRuntime& self, py::list items) {
            std::vector<wapr_ogl::MeshUpload> meshes;
            meshes.reserve(static_cast<size_t>(py::len(items)));
            for (auto handle : items) {
                py::dict d = py::cast<py::dict>(handle);
                meshes.push_back(mesh_upload_from_numpy(
                    py::cast<std::string>(d["name"]),
                    py::cast<py::array_t<float>>(d["vertices"]),
                    py::cast<py::array_t<uint32_t>>(d["faces"]),
                    d.contains("normals") ? py::object(d["normals"]) : py::none(),
                    d.contains("vertex_colors") ? py::object(d["vertex_colors"]) : py::none(),
                    d.contains("uvs") ? py::object(d["uvs"]) : py::none(),
                    d.contains("texture_rgb") ? py::object(d["texture_rgb"]) : py::none(),
                    d.contains("tex_w") ? py::cast<uint32_t>(d["tex_w"]) : 0u,
                    d.contains("tex_h") ? py::cast<uint32_t>(d["tex_h"]) : 0u,
                    d.contains("uv_per_corner") ? py::cast<bool>(d["uv_per_corner"]) : false,
                    d.contains("force_flat_normal") ? py::cast<bool>(d["force_flat_normal"]) : false));
            }
            return self.load_meshes(meshes);
        }, py::arg("items"))
        .def("render", [](wapr_ogl::GpuRenderRuntime& self, py::list instances, py::array K,
                          uint32_t frame_w, uint32_t frame_h, wapr_ogl::RenderOutputSpec spec,
                          wapr_ogl::LightParams light, py::object dist_coeffs,
                          py::object tile_lights) {
            auto cam = parse_K(K, frame_w, frame_h);
            apply_distortion(cam, dist_coeffs);
            auto inst = parse_instances(instances);
            std::vector<wapr_ogl::LightParams> lights_vec;
            const std::vector<wapr_ogl::LightParams>* lights_ptr = nullptr;
            if (!tile_lights.is_none()) {
                py::list lst = py::cast<py::list>(tile_lights);
                lights_vec.reserve(py::len(lst));
                for (auto item : lst) {
                    lights_vec.push_back(py::cast<wapr_ogl::LightParams>(item));
                }
                lights_ptr = &lights_vec;
            }
            auto out = self.render(inst, cam, spec, light, lights_ptr);
            py::dict result;
            result["batch_size"] = out.batch_size;
            result["width"] = out.width;
            result["height"] = out.height;
            result["device_ptr_rgb"] = reinterpret_cast<uintptr_t>(out.rgb);
            result["device_ptr_depth"] = reinterpret_cast<uintptr_t>(out.depth);
            result["device_ptr_normal"] = reinterpret_cast<uintptr_t>(out.normal);
            result["device_ptr_instance_id"] = reinterpret_cast<uintptr_t>(out.instance_id);
            return result;
        }, py::arg("instances"), py::arg("K"), py::arg("frame_w"), py::arg("frame_h"),
           py::arg("spec"), py::arg("light"), py::arg("dist_coeffs") = py::none(),
           py::arg("tile_lights") = py::none())
        .def("render_device_tiles", [](wapr_ogl::GpuRenderRuntime& self, uint32_t mesh_id,
                                       uintptr_t pose_ptr, uintptr_t bbox_ptr, uint32_t count,
                                       float fx, float fy, float cx, float cy,
                                       uint32_t frame_w, uint32_t frame_h, uintptr_t stream,
                                       wapr_ogl::RenderOutputSpec spec, wapr_ogl::LightParams light) {
            self.render_device_tiles(
                mesh_id, reinterpret_cast<const float*>(pose_ptr),
                reinterpret_cast<const float*>(bbox_ptr), count, fx, fy, cx, cy, frame_w, frame_h,
                spec, light, reinterpret_cast<void*>(stream));
        }, py::arg("mesh_id"), py::arg("pose_ptr"), py::arg("bbox_ptr"), py::arg("count"),
           py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
           py::arg("frame_w"), py::arg("frame_h"), py::arg("stream"),
           py::arg("spec"), py::arg("light"))
        .def("render_device_mesh_tiles", [](wapr_ogl::GpuRenderRuntime& self,
                                            const std::vector<uint32_t>& mesh_ids,
                                            uintptr_t pose_ptr, uintptr_t bbox_ptr,
                                            float fx, float fy, float cx, float cy,
                                            uint32_t frame_w, uint32_t frame_h, uintptr_t stream,
                                            wapr_ogl::RenderOutputSpec spec, wapr_ogl::LightParams light) {
            self.render_device_mesh_tiles(mesh_ids, reinterpret_cast<const float*>(pose_ptr),
                reinterpret_cast<const float*>(bbox_ptr), fx, fy, cx, cy, frame_w, frame_h,
                spec, light, reinterpret_cast<void*>(stream));
        }, py::arg("mesh_ids"), py::arg("pose_ptr"), py::arg("bbox_ptr"),
           py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
           py::arg("frame_w"), py::arg("frame_h"), py::arg("stream"),
           py::arg("spec"), py::arg("light"))
        .def("readback_atlas_rgb", [](wapr_ogl::GpuRenderRuntime& self) {
            std::vector<uint8_t> buf;
            self.readback_atlas_rgb(buf);
            return py::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
        })
        .def("readback_atlas_depth", [](wapr_ogl::GpuRenderRuntime& self) {
            std::vector<float> buf;
            self.readback_atlas_depth(buf);
            py::array_t<float> arr(buf.size());
            std::memcpy(arr.mutable_data(), buf.data(), buf.size() * sizeof(float));
            return arr;
        })
        .def("copy_pack_to_host", [](wapr_ogl::GpuRenderRuntime& self) {
            std::vector<float> depth, rgb;
            uint32_t batch = 0, w = 0, h = 0;
            self.copy_pack_to_host(depth, rgb, batch, w, h);
            py::dict result;
            result["batch_size"] = batch;
            result["width"] = w;
            result["height"] = h;
            py::array_t<float> darr(depth.size());
            py::array_t<float> rarr(rgb.size());
            if (!depth.empty()) std::memcpy(darr.mutable_data(), depth.data(), depth.size() * sizeof(float));
            if (!rgb.empty()) std::memcpy(rarr.mutable_data(), rgb.data(), rgb.size() * sizeof(float));
            result["depth"] = darr;
            result["rgb"] = rarr;
            return result;
        })
        .def("pack_device_ptrs", [](wapr_ogl::GpuRenderRuntime& self) {
            uintptr_t depth_ptr = 0, rgb_ptr = 0, id_ptr = 0, normal_ptr = 0;
            uint32_t batch = 0, w = 0, h = 0;
            const bool ok = self.pack_device_ptrs(depth_ptr, rgb_ptr, id_ptr, normal_ptr, batch, w, h);
            py::dict result;
            result["ok"] = ok;
            result["depth_ptr"] = depth_ptr;
            result["rgb_ptr"] = rgb_ptr;
            result["instance_id_ptr"] = id_ptr;
            result["normal_ptr"] = normal_ptr;
            result["batch_size"] = batch;
            result["width"] = w;
            result["height"] = h;
            return result;
        })
        .def("synchronize", &wapr_ogl::GpuRenderRuntime::synchronize)
        .def("warmup", &wapr_ogl::GpuRenderRuntime::warmup)
        .def("set_vram_budget_mb", &wapr_ogl::GpuRenderRuntime::set_vram_budget_mb)
        .def("set_skip_cuda_pack", &wapr_ogl::GpuRenderRuntime::set_skip_cuda_pack)
        .def("set_force_gl_finish", &wapr_ogl::GpuRenderRuntime::set_force_gl_finish)
        .def("skip_cuda_pack", &wapr_ogl::GpuRenderRuntime::skip_cuda_pack)
        .def("force_gl_finish", &wapr_ogl::GpuRenderRuntime::force_gl_finish)
        .def("unload_mesh", &wapr_ogl::GpuRenderRuntime::unload_mesh)
        .def("clear_all_meshes", &wapr_ogl::GpuRenderRuntime::clear_all_meshes)
        .def(
            "update_mesh_texture",
            [](wapr_ogl::GpuRenderRuntime& self, uint32_t mesh_id, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> rgb) {
                const auto buf = rgb.request();
                if (buf.ndim != 3 || buf.shape[2] < 3) {
                    throw std::runtime_error("update_mesh_texture expects HxWx3 uint8");
                }
                const auto h = static_cast<uint32_t>(buf.shape[0]);
                const auto w = static_cast<uint32_t>(buf.shape[1]);
                if (buf.shape[2] == 3 && buf.strides[0] == static_cast<py::ssize_t>(w * 3) &&
                    buf.strides[1] == 3 && buf.strides[2] == 1) {
                    return self.update_mesh_texture(mesh_id, static_cast<const uint8_t*>(buf.ptr), w, h);
                }
                std::vector<uint8_t> packed(static_cast<size_t>(w) * h * 3u);
                const auto* src = static_cast<const uint8_t*>(buf.ptr);
                const auto s0 = buf.strides[0], s1 = buf.strides[1], s2 = buf.strides[2];
                for (uint32_t y = 0; y < h; ++y) {
                    for (uint32_t x = 0; x < w; ++x) {
                        const size_t o = (static_cast<size_t>(y) * w + x) * 3u;
                        const auto base = y * s0 + x * s1;
                        packed[o + 0] = src[base + 0 * s2];
                        packed[o + 1] = src[base + 1 * s2];
                        packed[o + 2] = src[base + 2 * s2];
                    }
                }
                return self.update_mesh_texture(mesh_id, packed.data(), w, h);
            },
            py::arg("mesh_id"),
            py::arg("rgb"))
        .def("mesh_valid", &wapr_ogl::GpuRenderRuntime::mesh_valid)
        .def("mesh_count", &wapr_ogl::GpuRenderRuntime::mesh_count)
        .def("device", &wapr_ogl::GpuRenderRuntime::device)
        .def("gl_version_string", &wapr_ogl::GpuRenderRuntime::gl_version_string)
        .def("gl_caps", &wapr_ogl::GpuRenderRuntime::gl_caps)
        .def("last_draw_api_calls", &wapr_ogl::GpuRenderRuntime::last_draw_api_calls);

    m.attr("DEPTH_LESS") = py::int_(wapr_ogl::kDepthFuncLess);
    m.attr("DEPTH_GREATER") = py::int_(wapr_ogl::kDepthFuncGreater);
}
