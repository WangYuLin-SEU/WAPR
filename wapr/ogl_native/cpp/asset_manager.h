// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#pragma once

#include "egl_context.h"
#include "types.h"
#include <unordered_map>
#include <algorithm>
#include <string>
#include <vector>

namespace wapr_ogl {

class AssetManager {
public:
    explicit AssetManager(EglContext& egl);

    uint32_t load_mesh(const MeshUpload& mesh);
    
    std::vector<uint32_t> load_meshes(const std::vector<MeshUpload>& meshes);
    bool unload_mesh(uint32_t mesh_id);
    
    void clear_all();
    
    bool update_mesh_texture(uint32_t mesh_id, const uint8_t* rgb, uint32_t w, uint32_t h);
    bool mesh_valid(uint32_t mesh_id) const;
    uint32_t mesh_count() const {
        return static_cast<uint32_t>(std::count_if(
            mesh_table_.begin(), mesh_table_.end(),
            [](const MeshTableEntry& e) { return e.valid != 0 && e.index_count > 0; }));
    }
    uint32_t mesh_id_by_name(const std::string& name) const;
    const std::vector<MeshTableEntry>& mesh_table() const { return mesh_table_; }

    void bind_for_draw();
    GLuint tex_array() const { return tex_array_; }
    bool has_textures() const { return num_tex_layers_ > 0; }

private:
    struct TextureEntry {
        uint32_t tex_w = 0;
        uint32_t tex_h = 0;
        std::vector<uint8_t> rgb;
    };

    EglContext& egl_;
    GLFunctions& gl_;
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    GLuint ebo_ = 0;
    GLuint tex_array_ = 0;
    uint32_t num_tex_layers_ = 0;
    uint32_t tex_array_layers_alloc_ = 0;
    uint32_t tex_array_w_ = 0;
    uint32_t tex_array_h_ = 0;

    std::vector<float> vertices_;
    std::vector<float> normals_;
    std::vector<float> colors_;
    std::vector<float> uvs_;
    std::vector<uint32_t> indices_;
    std::vector<MeshTableEntry> mesh_table_;
    std::vector<TextureEntry> texture_store_;
    std::unordered_map<std::string, uint32_t> name_to_id_;

    void rebuild_gpu_buffers();
    void rebuild_texture_array();
    void compact_geometry();
    uint32_t append_mesh_cpu(const MeshUpload& mesh);
};

}
