// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#include "asset_manager.h"
#ifndef _WIN32
#include <EGL/egl.h>
#endif
#include <GL/gl.h>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace wapr_ogl {

using PFNGLTEXIMAGE3DPROC = void (*)(GLenum, GLint, GLint, GLsizei, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*);
using PFNGLTEXSUBIMAGE3DPROC = void (*)(GLenum, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei, GLenum, GLenum, const void*);


static void validate_mesh_upload(const MeshUpload& mesh) {
    if (mesh.vertices.empty() || mesh.vertices.size() % 3 != 0) {
        throw std::invalid_argument("mesh vertices must be non-empty xyz triples");
    }
    if (mesh.indices.empty() || mesh.indices.size() % 3 != 0) {
        throw std::invalid_argument("mesh indices must be non-empty triangles");
    }
    const size_t nv = mesh.vertices.size() / 3;
    for (uint32_t index : mesh.indices) {
        if (static_cast<size_t>(index) >= nv) {
            throw std::out_of_range("mesh index exceeds vertex count");
        }
    }
    if (!mesh.normals.empty() && mesh.normals.size() != mesh.vertices.size()) {
        throw std::invalid_argument("mesh normals must be empty or match vertices");
    }
    if (!mesh.vertex_colors.empty() && mesh.vertex_colors.size() != mesh.vertices.size()) {
        throw std::invalid_argument("mesh vertex colors must be empty or match vertices");
    }
    const size_t expected_uv = mesh.uv_per_corner ? mesh.indices.size() * 2 : nv * 2;
    if (!mesh.uvs.empty() && mesh.uvs.size() != expected_uv) {
        throw std::invalid_argument("mesh UV count does not match UV layout");
    }
    const bool has_texture = !mesh.texture_rgb.empty() || mesh.tex_w > 0 || mesh.tex_h > 0;
    if (has_texture) {
        if (mesh.tex_w == 0 || mesh.tex_h == 0 ||
            mesh.texture_rgb.size() != static_cast<size_t>(mesh.tex_w) * mesh.tex_h * 3u) {
            throw std::invalid_argument("texture must be exactly tex_h*tex_w*3 RGB bytes");
        }
        if (mesh.uvs.empty()) {
            throw std::invalid_argument("textured mesh requires UV coordinates");
        }
    }
}

AssetManager::AssetManager(EglContext& egl) : egl_(egl), gl_(egl.gl()) {
    egl_.make_current();
    gl_.GenVertexArrays(1, &vao_);
    gl_.GenBuffers(1, &vbo_);
    gl_.GenBuffers(1, &ebo_);
    gl_.GenTextures(1, &tex_array_);
    gl_.BindTexture(GL_TEXTURE_2D_ARRAY, tex_array_);
    gl_.TexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    gl_.TexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#endif
    auto glTexParameterf =
        reinterpret_cast<void (*)(GLenum, GLenum, GLfloat)>(eglGetProcAddress("glTexParameterf"));
    if (glTexParameterf) {
        glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8.0f);
    }
}

void AssetManager::rebuild_gpu_buffers() {
    struct Vertex {
        float px, py, pz;
        float nx, ny, nz;
        float cr, cg, cb;
        float u, v;
    };
    const size_t n = vertices_.size() / 3;
    std::vector<Vertex> interleaved(n);
    for (size_t i = 0; i < n; ++i) {
        interleaved[i].px = vertices_[i * 3 + 0];
        interleaved[i].py = vertices_[i * 3 + 1];
        interleaved[i].pz = vertices_[i * 3 + 2];
        interleaved[i].nx = normals_.size() >= (i + 1) * 3 ? normals_[i * 3 + 0] : 0.f;
        interleaved[i].ny = normals_.size() >= (i + 1) * 3 ? normals_[i * 3 + 1] : 1.f;
        interleaved[i].nz = normals_.size() >= (i + 1) * 3 ? normals_[i * 3 + 2] : 0.f;
        interleaved[i].cr = colors_.size() >= (i + 1) * 3 ? colors_[i * 3 + 0] : 0.5f;
        interleaved[i].cg = colors_.size() >= (i + 1) * 3 ? colors_[i * 3 + 1] : 0.5f;
        interleaved[i].cb = colors_.size() >= (i + 1) * 3 ? colors_[i * 3 + 2] : 0.5f;
        interleaved[i].u = uvs_.size() >= (i + 1) * 2 ? uvs_[i * 2 + 0] : 0.f;
        interleaved[i].v = uvs_.size() >= (i + 1) * 2 ? uvs_[i * 2 + 1] : 0.f;
    }

    gl_.BindVertexArray(vao_);
    gl_.BindBuffer(GL_ARRAY_BUFFER, vbo_);
    gl_.BufferData(GL_ARRAY_BUFFER,
                   static_cast<GLsizeiptr>(interleaved.size() * sizeof(Vertex)),
                   interleaved.data(), GL_STATIC_DRAW);
    gl_.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    gl_.BufferData(GL_ELEMENT_ARRAY_BUFFER,
                   static_cast<GLsizeiptr>(indices_.size() * sizeof(uint32_t)),
                   indices_.data(), GL_STATIC_DRAW);

    const GLsizei stride = static_cast<GLsizei>(sizeof(Vertex));
    gl_.EnableVertexAttribArray(0);
    gl_.VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(0));
    gl_.EnableVertexAttribArray(1);
    gl_.VertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(3 * sizeof(float)));
    gl_.EnableVertexAttribArray(2);
    gl_.VertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(6 * sizeof(float)));
    gl_.EnableVertexAttribArray(3);
    gl_.VertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(9 * sizeof(float)));

    if (std::getenv("WAPR_OGL_DEBUG") && !interleaved.empty()) {
        fprintf(stderr, "vbo first vert=(%.4f, %.4f, %.4f) n=%zu indices=%zu\n",
                interleaved[0].px, interleaved[0].py, interleaved[0].pz,
                interleaved.size(), indices_.size());
    }
}

uint32_t AssetManager::append_mesh_cpu(const MeshUpload& mesh) {
    if (!mesh.name.empty() && name_to_id_.count(mesh.name)) {
        return name_to_id_.at(mesh.name);
    }

    MeshTableEntry entry{};
    entry.vertex_offset = static_cast<uint32_t>(vertices_.size() / 3);
    entry.index_offset = static_cast<uint32_t>(indices_.size());
    entry.has_texture = (!mesh.texture_rgb.empty() && mesh.tex_w > 0 && mesh.tex_h > 0) ? 1u : 0u;
    entry.tex_layer = entry.has_texture ? num_tex_layers_++ : 0u;
    entry.force_flat_normal = mesh.force_flat_normal;

    const size_t nv = mesh.vertices.size() / 3;
    const size_t tri_count = mesh.indices.size() / 3;

    if (mesh.uv_per_corner && mesh.uvs.size() >= tri_count * 6) {
        for (size_t t = 0; t < tri_count; ++t) {
            for (int k = 0; k < 3; ++k) {
                const uint32_t vi = mesh.indices[t * 3 + k];
                vertices_.push_back(mesh.vertices[vi * 3 + 0]);
                vertices_.push_back(mesh.vertices[vi * 3 + 1]);
                vertices_.push_back(mesh.vertices[vi * 3 + 2]);
                if (mesh.normals.size() >= nv * 3) {
                    normals_.push_back(mesh.normals[vi * 3 + 0]);
                    normals_.push_back(mesh.normals[vi * 3 + 1]);
                    normals_.push_back(mesh.normals[vi * 3 + 2]);
                } else {
                    normals_.insert(normals_.end(), {0.f, 1.f, 0.f});
                }
                if (mesh.vertex_colors.size() >= nv * 3) {
                    colors_.push_back(mesh.vertex_colors[vi * 3 + 0]);
                    colors_.push_back(mesh.vertex_colors[vi * 3 + 1]);
                    colors_.push_back(mesh.vertex_colors[vi * 3 + 2]);
                } else {
                    colors_.insert(colors_.end(), {0.5f, 0.5f, 0.5f});
                }
                const size_t uvi = (t * 3 + k) * 2;
                uvs_.push_back(mesh.uvs[uvi + 0]);
                uvs_.push_back(mesh.uvs[uvi + 1]);
                indices_.push_back(static_cast<uint32_t>(vertices_.size() / 3 - 1));
            }
        }
    } else {
        const uint32_t base_v = entry.vertex_offset;
        vertices_.insert(vertices_.end(), mesh.vertices.begin(), mesh.vertices.end());
        if (mesh.normals.size() >= nv * 3) {
            normals_.insert(normals_.end(), mesh.normals.begin(), mesh.normals.end());
        } else {
            for (size_t i = 0; i < nv; ++i) {
                normals_.insert(normals_.end(), {0.f, 1.f, 0.f});
            }
        }
        if (mesh.vertex_colors.size() >= nv * 3) {
            colors_.insert(colors_.end(), mesh.vertex_colors.begin(), mesh.vertex_colors.end());
        } else {
            for (size_t i = 0; i < nv; ++i) {
                colors_.insert(colors_.end(), {0.5f, 0.5f, 0.5f});
            }
        }
        if (mesh.uvs.size() >= nv * 2) {
            uvs_.insert(uvs_.end(), mesh.uvs.begin(), mesh.uvs.end());
        } else {
            for (size_t i = 0; i < nv; ++i) {
                uvs_.insert(uvs_.end(), {0.f, 0.f});
            }
        }
        for (size_t t = 0; t < tri_count; ++t) {
            const uint32_t i0 = mesh.indices[t * 3 + 0];
            const uint32_t i1 = mesh.indices[t * 3 + 1];
            const uint32_t i2 = mesh.indices[t * 3 + 2];
            indices_.push_back(base_v + i0);
            indices_.push_back(base_v + i1);
            indices_.push_back(base_v + i2);
        }
    }
    entry.index_count = static_cast<uint32_t>(indices_.size()) - entry.index_offset;
    entry.vertex_count = static_cast<uint32_t>(vertices_.size() / 3) - entry.vertex_offset;

    if (!mesh.vertices.empty()) {
        entry.bounds_min[0] = entry.bounds_min[1] = entry.bounds_min[2] = mesh.vertices[0];
        entry.bounds_max[0] = entry.bounds_max[1] = entry.bounds_max[2] = mesh.vertices[0];
        for (size_t vi = 0; vi < nv; ++vi) {
            for (int ax = 0; ax < 3; ++ax) {
                const float v = mesh.vertices[vi * 3 + ax];
                entry.bounds_min[ax] = std::min(entry.bounds_min[ax], v);
                entry.bounds_max[ax] = std::max(entry.bounds_max[ax], v);
            }
        }
        entry.has_bounds = 1;
    }

    if (entry.has_texture) {
        TextureEntry tex;
        tex.tex_w = mesh.tex_w;
        tex.tex_h = mesh.tex_h;
        tex.rgb = mesh.texture_rgb;
        texture_store_.push_back(std::move(tex));
    } else {
        texture_store_.push_back({});
    }

    const uint32_t id = static_cast<uint32_t>(mesh_table_.size());
    mesh_table_.push_back(entry);
    if (!mesh.name.empty()) {
        name_to_id_[mesh.name] = id;
    }
    return id;
}

uint32_t AssetManager::load_mesh(const MeshUpload& mesh) {
    validate_mesh_upload(mesh);
    if (!mesh.texture_rgb.empty() && tex_array_w_ > 0 &&
        (mesh.tex_w != tex_array_w_ || mesh.tex_h != tex_array_h_)) {
        throw std::invalid_argument(
            "texture size mismatch across meshes; resize before upload");
    }
    const bool existed = !mesh.name.empty() && name_to_id_.count(mesh.name);
    const uint32_t id = append_mesh_cpu(mesh);
    if (existed) {
        return id;
    }
    egl_.make_current();
    rebuild_gpu_buffers();
    if (id < mesh_table_.size() && mesh_table_[id].has_texture) {
        rebuild_texture_array();
    }
    return id;
}

std::vector<uint32_t> AssetManager::load_meshes(const std::vector<MeshUpload>& meshes) {
    uint32_t expected_w = tex_array_w_;
    uint32_t expected_h = tex_array_h_;
    for (const auto& mesh : meshes) {
        validate_mesh_upload(mesh);
        if (!mesh.name.empty() && name_to_id_.count(mesh.name)) continue;
        if (!mesh.texture_rgb.empty()) {
            if (expected_w == 0) {
                expected_w = mesh.tex_w;
                expected_h = mesh.tex_h;
            } else if (mesh.tex_w != expected_w || mesh.tex_h != expected_h) {
                throw std::invalid_argument(
                    "texture size mismatch across meshes; resize before batch upload");
            }
        }
    }
    std::vector<uint32_t> ids;
    ids.reserve(meshes.size());
    bool any_new = false;
    bool any_tex = false;
    for (const auto& mesh : meshes) {
        const bool existed = !mesh.name.empty() && name_to_id_.count(mesh.name);
        const uint32_t id = append_mesh_cpu(mesh);
        ids.push_back(id);
        if (!existed) {
            any_new = true;
            if (id < mesh_table_.size() && mesh_table_[id].has_texture) {
                any_tex = true;
            }
        }
    }
    if (any_new) {
        egl_.make_current();
        rebuild_gpu_buffers();
        if (any_tex) {
            rebuild_texture_array();
        }
    }
    return ids;
}

void AssetManager::rebuild_texture_array() {
    num_tex_layers_ = 0;
    uint32_t array_w = 0;
    uint32_t array_h = 0;
    for (size_t i = 0; i < mesh_table_.size(); ++i) {
        auto& me = mesh_table_[i];
        if (!me.valid || !me.has_texture) {
            me.tex_layer = 0;
            continue;
        }
        const auto& tex = texture_store_[i];
        if (array_w == 0) {
            array_w = tex.tex_w;
            array_h = tex.tex_h;
        } else if (tex.tex_w != array_w || tex.tex_h != array_h) {
            throw std::runtime_error("texture size mismatch across meshes");
        }
        me.tex_layer = num_tex_layers_++;
    }
    if (num_tex_layers_ == 0) {
        tex_array_layers_alloc_ = 0;
        tex_array_w_ = 0;
        tex_array_h_ = 0;
        return;
    }

    auto tex_fn = reinterpret_cast<PFNGLTEXSUBIMAGE3DPROC>(eglGetProcAddress("glTexSubImage3D"));
    auto tex_alloc = reinterpret_cast<PFNGLTEXIMAGE3DPROC>(eglGetProcAddress("glTexImage3D"));
    if (!tex_fn || !tex_alloc) {
        throw std::runtime_error("OpenGL 3D texture functions unavailable for textured mesh");
    }

    tex_array_w_ = array_w;
    tex_array_h_ = array_h;
    tex_array_layers_alloc_ = std::max(num_tex_layers_, 1u);
    gl_.BindTexture(GL_TEXTURE_2D_ARRAY, tex_array_);
    tex_alloc(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8,
              static_cast<GLsizei>(tex_array_w_),
              static_cast<GLsizei>(tex_array_h_),
              static_cast<GLsizei>(tex_array_layers_alloc_),
              0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    for (size_t i = 0; i < mesh_table_.size(); ++i) {
        const auto& me = mesh_table_[i];
        if (!me.valid || !me.has_texture) {
            continue;
        }
        const auto& tex = texture_store_[i];
        tex_fn(GL_TEXTURE_2D_ARRAY, 0, 0, 0, static_cast<GLint>(me.tex_layer),
               static_cast<GLsizei>(tex.tex_w),
               static_cast<GLsizei>(tex.tex_h), 1,
               GL_RGB, GL_UNSIGNED_BYTE, tex.rgb.data());
    }
    gl_.TexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    gl_.TexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#endif
    {
        auto glTexParameterf =
            reinterpret_cast<void (*)(GLenum, GLenum, GLfloat)>(eglGetProcAddress("glTexParameterf"));
        if (glTexParameterf) {
            glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8.0f);
        }
    }
    if (gl_.GenerateMipmap) {
        gl_.GenerateMipmap(GL_TEXTURE_2D_ARRAY);
    }
}

uint32_t AssetManager::mesh_id_by_name(const std::string& name) const {
    auto it = name_to_id_.find(name);
    if (it == name_to_id_.end()) {
        throw std::runtime_error("mesh not found: " + name);
    }
    if (it->second >= mesh_table_.size() || !mesh_table_[it->second].valid) {
        throw std::runtime_error("mesh unloaded: " + name);
    }
    return it->second;
}

bool AssetManager::mesh_valid(uint32_t mesh_id) const {
    return mesh_id < mesh_table_.size() && mesh_table_[mesh_id].valid != 0 &&
           mesh_table_[mesh_id].index_count > 0;
}

bool AssetManager::update_mesh_texture(uint32_t mesh_id, const uint8_t* rgb, uint32_t w, uint32_t h) {
    if (!mesh_valid(mesh_id) || !mesh_table_[mesh_id].has_texture || rgb == nullptr) {
        return false;
    }
    if (w != tex_array_w_ || h != tex_array_h_) {
        return false;
    }
    if (mesh_id >= texture_store_.size()) {
        return false;
    }
    auto& tex = texture_store_[mesh_id];
    const size_t nbytes = static_cast<size_t>(w) * h * 3u;
    tex.rgb.assign(rgb, rgb + nbytes);
    tex.tex_w = w;
    tex.tex_h = h;

    egl_.make_current();
    auto tex_fn = reinterpret_cast<PFNGLTEXSUBIMAGE3DPROC>(eglGetProcAddress("glTexSubImage3D"));
    if (!tex_fn) {
        return false;
    }
    gl_.BindTexture(GL_TEXTURE_2D_ARRAY, tex_array_);
    tex_fn(GL_TEXTURE_2D_ARRAY, 0, 0, 0, static_cast<GLint>(mesh_table_[mesh_id].tex_layer),
           static_cast<GLsizei>(w), static_cast<GLsizei>(h), 1,
           GL_RGB, GL_UNSIGNED_BYTE, tex.rgb.data());
    if (gl_.GenerateMipmap) {
        gl_.GenerateMipmap(GL_TEXTURE_2D_ARRAY);
    }
    return true;
}

void AssetManager::clear_all() {
    egl_.make_current();
    mesh_table_.clear();
    name_to_id_.clear();
    vertices_.clear();
    normals_.clear();
    colors_.clear();
    uvs_.clear();
    indices_.clear();
    texture_store_.clear();
    vertices_.shrink_to_fit();
    normals_.shrink_to_fit();
    colors_.shrink_to_fit();
    uvs_.shrink_to_fit();
    indices_.shrink_to_fit();
    texture_store_.shrink_to_fit();
    num_tex_layers_ = 0;
    tex_array_layers_alloc_ = 0;
    tex_array_w_ = 0;
    tex_array_h_ = 0;
    rebuild_gpu_buffers();
    auto tex_alloc = reinterpret_cast<PFNGLTEXIMAGE3DPROC>(eglGetProcAddress("glTexImage3D"));
    if (tex_alloc && tex_array_) {
        gl_.BindTexture(GL_TEXTURE_2D_ARRAY, tex_array_);
        tex_alloc(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8, 1, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    }
}

bool AssetManager::unload_mesh(uint32_t mesh_id) {
    if (mesh_id >= mesh_table_.size() || !mesh_table_[mesh_id].valid) {
        return false;
    }
    const bool had_texture = mesh_table_[mesh_id].has_texture != 0;
    mesh_table_[mesh_id].valid = 0;
    mesh_table_[mesh_id].index_count = 0;
    mesh_table_[mesh_id].has_texture = 0;
    for (auto it = name_to_id_.begin(); it != name_to_id_.end();) {
        if (it->second == mesh_id) {
            it = name_to_id_.erase(it);
        } else {
            ++it;
        }
    }
    egl_.make_current();
    compact_geometry();
    if (had_texture) {
        rebuild_texture_array();
    }
    return true;
}

void AssetManager::compact_geometry() {
    std::vector<float> nv, nn, nc, nu;
    std::vector<uint32_t> ni;
    const size_t nmesh = mesh_table_.size();
    nv.reserve(vertices_.size());
    nn.reserve(normals_.size());
    nc.reserve(colors_.size());
    nu.reserve(uvs_.size());
    ni.reserve(indices_.size());
    for (size_t i = 0; i < nmesh; ++i) {
        auto& me = mesh_table_[i];
        if (!me.valid || me.vertex_count == 0) {
            me.vertex_offset = 0;
            me.vertex_count = 0;
            me.index_offset = 0;
            me.index_count = 0;
            if (i < texture_store_.size()) {
                texture_store_[i].rgb.clear();
                texture_store_[i].rgb.shrink_to_fit();
            }
            continue;
        }
        const uint32_t v0 = me.vertex_offset;
        const uint32_t nvtx = me.vertex_count;
        const uint32_t new_v = static_cast<uint32_t>(nv.size() / 3);
        const uint32_t new_i = static_cast<uint32_t>(ni.size());
        nv.insert(nv.end(), vertices_.begin() + static_cast<size_t>(v0) * 3,
                  vertices_.begin() + static_cast<size_t>(v0 + nvtx) * 3);
        if (normals_.size() >= static_cast<size_t>(v0 + nvtx) * 3) {
            nn.insert(nn.end(), normals_.begin() + static_cast<size_t>(v0) * 3,
                      normals_.begin() + static_cast<size_t>(v0 + nvtx) * 3);
        }
        if (colors_.size() >= static_cast<size_t>(v0 + nvtx) * 3) {
            nc.insert(nc.end(), colors_.begin() + static_cast<size_t>(v0) * 3,
                      colors_.begin() + static_cast<size_t>(v0 + nvtx) * 3);
        }
        if (uvs_.size() >= static_cast<size_t>(v0 + nvtx) * 2) {
            nu.insert(nu.end(), uvs_.begin() + static_cast<size_t>(v0) * 2,
                      uvs_.begin() + static_cast<size_t>(v0 + nvtx) * 2);
        }
        for (uint32_t k = 0; k < me.index_count; ++k) {
            ni.push_back(indices_[me.index_offset + k] - v0 + new_v);
        }
        me.vertex_offset = new_v;
        me.vertex_count = nvtx;
        me.index_offset = new_i;
    }
    vertices_.swap(nv);
    normals_.swap(nn);
    colors_.swap(nc);
    uvs_.swap(nu);
    indices_.swap(ni);
    vertices_.shrink_to_fit();
    normals_.shrink_to_fit();
    colors_.shrink_to_fit();
    uvs_.shrink_to_fit();
    indices_.shrink_to_fit();
    rebuild_gpu_buffers();
}

void AssetManager::bind_for_draw() {
    gl_.BindVertexArray(vao_);
}

}
