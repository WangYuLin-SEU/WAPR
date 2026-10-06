// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#include "batch_renderer.h"
#include "projection.h"
#include "shader_utils.h"
#include "cuda/fill_tile_instances.h"
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <EGL/egl.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <unordered_set>

#ifndef GL_SHADER_STORAGE_BUFFER
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#endif
#ifndef GL_DRAW_INDIRECT_BUFFER
#define GL_DRAW_INDIRECT_BUFFER 0x8F3F
#endif
#ifndef GL_RENDERBUFFER
#define GL_RENDERBUFFER 0x8D41
#endif
#ifndef GL_DEPTH_COMPONENT32F
#define GL_DEPTH_COMPONENT32F 0x81A7
#endif
#ifndef GL_COLOR_ATTACHMENT0
#define GL_COLOR_ATTACHMENT0 0x8CE0
#endif
#ifndef GL_COLOR_ATTACHMENT1
#define GL_COLOR_ATTACHMENT1 0x8CE1
#endif
#ifndef GL_COLOR_ATTACHMENT2
#define GL_COLOR_ATTACHMENT2 0x8CE2
#endif
#ifndef GL_COLOR_ATTACHMENT3
#define GL_COLOR_ATTACHMENT3 0x8CE3
#endif
#ifndef GL_COLOR_ATTACHMENT4
#define GL_COLOR_ATTACHMENT4 0x8CE4
#endif
#ifndef GL_R32UI
#define GL_R32UI 0x8236
#endif
#ifndef GL_R8
#define GL_R8 0x8229
#endif
#ifndef GL_RED_INTEGER
#define GL_RED_INTEGER 0x8D94
#endif
#ifndef GL_MAX_VIEWPORTS
#define GL_MAX_VIEWPORTS 0x825B
#endif
#ifndef GL_MAP_WRITE_BIT
#define GL_MAP_WRITE_BIT 0x0002
#endif
#ifndef GL_MAP_PERSISTENT_BIT
#define GL_MAP_PERSISTENT_BIT 0x0040
#endif
#ifndef GL_MAP_COHERENT_BIT
#define GL_MAP_COHERENT_BIT 0x0080
#endif
#ifndef GL_DYNAMIC_STORAGE_BIT
#define GL_DYNAMIC_STORAGE_BIT 0x0100
#endif
#ifndef GL_CLIENT_STORAGE_BIT
#define GL_CLIENT_STORAGE_BIT 0x0200
#endif
#ifndef GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT
#define GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT 0x90DF
#endif
#ifndef GL_WAIT_FAILED
#define GL_WAIT_FAILED 0x911D
#endif
#ifndef GL_SYNC_GPU_COMMANDS_COMPLETE
#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
#endif
#ifndef GL_SYNC_FLUSH_COMMANDS_BIT
#define GL_SYNC_FLUSH_COMMANDS_BIT 0x00000001
#endif
#ifndef GL_TIMEOUT_EXPIRED
#define GL_TIMEOUT_EXPIRED 0x911B
#endif
#ifndef GL_ALREADY_SIGNALED
#define GL_ALREADY_SIGNALED 0x911A
#endif
#ifndef GL_CONDITION_SATISFIED
#define GL_CONDITION_SATISFIED 0x911C
#endif
#ifndef GL_TEXTURE_2D_ARRAY
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#endif
#ifndef GL_SCISSOR_TEST
#define GL_SCISSOR_TEST 0x0C11
#endif
#ifndef GL_READ_FRAMEBUFFER
#define GL_READ_FRAMEBUFFER 0x8CA8
#endif
#ifndef GL_DRAW_FRAMEBUFFER
#define GL_DRAW_FRAMEBUFFER 0x8CA9
#endif
#ifndef GL_DRAW_BUFFER
#define GL_DRAW_BUFFER 0x0C01
#endif
#ifndef GL_SAMPLE_SHADING
#define GL_SAMPLE_SHADING 0x8C36
#endif
#ifndef GL_COLOR
#define GL_COLOR 0x1800
#endif

namespace wapr_ogl {

static uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

static GLsizeiptr align_up(GLsizeiptr value, GLsizeiptr alignment) {
    alignment = std::max<GLsizeiptr>(alignment, 1);
    return ((value + alignment - 1) / alignment) * alignment;
}

static uint32_t normalize_msaa_samples(uint32_t samples, bool layered) {
    if (layered || samples < 2) return 0;
    if (samples >= 8) return 8;
    if (samples >= 4) return 4;
    return 2;
}

static void mat4_to_gl_colmajor(const float* row_major, float* col_major) {
    mat4_transpose(row_major, col_major);
}

BatchRenderer::BatchRenderer(EglContext& egl, AssetManager& assets, const std::string& shader_dir)
    : egl_(egl), assets_(assets), gl_(egl.gl()) {
    egl_.make_current();
    const auto vert = load_shader_file(shader_dir + "/lit_mrt.vert");
    const auto frag = load_shader_file(shader_dir + "/lit_mrt.frag");
    program_ = compile_shader_program(gl_, vert, frag);

    gl_.GenBuffers(1, &instance_ssbo_);
    gl_.GenBuffers(1, &mesh_table_ssbo_);
    gl_.GenBuffers(1, &light_ssbo_);
    gl_.GenBuffers(1, &indirect_buf_);

    if (gl_.BufferSubData) {
        glBufferSubData_ = gl_.BufferSubData;
    } else {
        glBufferSubData_ = reinterpret_cast<void (*)(GLenum, GLintptr, GLsizeiptr, const void*)>(
            eglGetProcAddress("glBufferSubData"));
    }
    glScissor_ = reinterpret_cast<void (*)(GLint, GLint, GLsizei, GLsizei)>(
        eglGetProcAddress("glScissor"));
    loc_tile_ofs_ = gl_.GetUniformLocation(program_, "u_tile_offset");
    loc_viewport_ = gl_.GetUniformLocation(program_, "u_viewport");
    loc_K_ = gl_.GetUniformLocation(program_, "u_K");
    loc_proj_z_ = gl_.GetUniformLocation(program_, "u_proj_z");
    loc_has_clip_affine_ = gl_.GetUniformLocation(program_, "u_has_clip_affine");
    loc_clip_affine_ = gl_.GetUniformLocation(program_, "u_clip_affine");
    query_caps();
}

BatchRenderer::~BatchRenderer() noexcept {
    try {
        egl_.make_current();
        for (auto& fence : persist_fences_) {
            if (fence && gl_.DeleteSync) gl_.DeleteSync(fence);
            fence = nullptr;
        }
        if (persist_inst_ptr_ && gl_.UnmapBuffer) {
            gl_.BindBuffer(GL_SHADER_STORAGE_BUFFER, instance_ssbo_);
            gl_.UnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            persist_inst_ptr_ = nullptr;
        }
        if (persist_cmd_ptr_ && gl_.UnmapBuffer) {
            gl_.BindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_buf_);
            gl_.UnmapBuffer(GL_DRAW_INDIRECT_BUFFER);
            persist_cmd_ptr_ = nullptr;
        }
        if (cuda_inst_resource_ != nullptr) {
            cudaSetDevice(cuda_device_);
            cudaGraphicsUnregisterResource(
                reinterpret_cast<cudaGraphicsResource_t>(cuda_inst_resource_));
            cuda_inst_resource_ = nullptr;
        }
        if (cuda_inst_ssbo_ && gl_.DeleteBuffers) gl_.DeleteBuffers(1, &cuda_inst_ssbo_);
        if (cuda_indirect_buf_ && gl_.DeleteBuffers) gl_.DeleteBuffers(1, &cuda_indirect_buf_);
        const GLuint buffers[] = {instance_ssbo_, mesh_table_ssbo_, light_ssbo_, indirect_buf_};
        if (gl_.DeleteBuffers) gl_.DeleteBuffers(4, buffers);
        if (program_ && gl_.DeleteProgram) gl_.DeleteProgram(program_);
    } catch (...) {
    }
}

void BatchRenderer::query_caps() {
    caps_ = GlCaps{};
    caps_.multi_draw_indirect = gl_.MultiDrawElementsIndirect != nullptr;
    caps_.viewport_array = gl_.ViewportArrayv != nullptr && gl_.ScissorArrayv != nullptr;
    caps_.buffer_storage = gl_.BufferStorage != nullptr;
    caps_.map_persistent = gl_.MapBufferRange != nullptr && caps_.buffer_storage;
    caps_.sync = gl_.FenceSync != nullptr && gl_.ClientWaitSync != nullptr && gl_.DeleteSync != nullptr;
    caps_.max_viewports = 1;
    if (gl_.GetIntegerv) {
        GLint mv = 1;
        gl_.GetIntegerv(GL_MAX_VIEWPORTS, &mv);
        caps_.max_viewports = std::max(1, static_cast<int>(mv));
    }
    auto has_ext = [&](const char* name) -> bool {
        if (!gl_.GetStringi || !gl_.GetIntegerv) return false;
        GLint n = 0;
        gl_.GetIntegerv(0x821D , &n);
        for (GLint i = 0; i < n; ++i) {
            const char* e = reinterpret_cast<const char*>(gl_.GetStringi(0x1F03 , static_cast<GLuint>(i)));
            if (e && std::strcmp(e, name) == 0) return true;
        }
        return false;
    };
    int gl_major = 0, gl_minor = 0;
    if (gl_.GetString) {
        const char* version = reinterpret_cast<const char*>(gl_.GetString(GL_VERSION));
        if (version) std::sscanf(version, "%d.%d", &gl_major, &gl_minor);
    }
    const bool core_46 = gl_major > 4 || (gl_major == 4 && gl_minor >= 6);
    caps_.shader_draw_parameters =
        core_46 || has_ext("GL_ARB_shader_draw_parameters");
    caps_.shader_viewport_layer_array =
        core_46 || has_ext("GL_ARB_shader_viewport_layer_array");
    fprintf(stderr,
            "[gpu_render] GL caps: max_viewports=%d mdi=%d viewport_array=%d "
            "shader_viewport_layer=%d buffer_storage=%d sync=%d\n",
            caps_.max_viewports, caps_.multi_draw_indirect ? 1 : 0,
            caps_.viewport_array ? 1 : 0, caps_.shader_viewport_layer_array ? 1 : 0,
            caps_.buffer_storage ? 1 : 0, caps_.sync ? 1 : 0);
}

void BatchRenderer::destroy_msaa_targets() {
    auto glDeleteRenderbuffers =
        reinterpret_cast<void (*)(GLsizei, const GLuint*)>(eglGetProcAddress("glDeleteRenderbuffers"));
    auto del_rb = [&](GLuint& rb) {
        if (rb && glDeleteRenderbuffers) {
            glDeleteRenderbuffers(1, &rb);
            rb = 0;
        }
    };
    if (fbo_ms_) {
        gl_.DeleteFramebuffers(1, &fbo_ms_);
        fbo_ms_ = 0;
    }
    del_rb(rb_ms_depth_);
    del_rb(rb_ms_rgb_);
    del_rb(rb_ms_normal_);
    del_rb(rb_ms_coord_);
    del_rb(rb_ms_instance_id_);
    del_rb(rb_ms_hw_depth_);
    fbo_msaa_samples_ = 0;
}

void BatchRenderer::resolve_msaa(const RenderOutputSpec& spec) const {
    if (!fbo_ms_ || fbo_msaa_samples_ < 2) return;
    auto glBlitFramebuffer = reinterpret_cast<void (*)(
        GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum)>(
        eglGetProcAddress("glBlitFramebuffer"));
    auto glDrawBuffer =
        reinterpret_cast<void (*)(GLenum)>(eglGetProcAddress("glDrawBuffer"));
    if (!glBlitFramebuffer || !gl_.ReadBuffer || !glDrawBuffer) {
        throw std::runtime_error("MSAA resolve requires glBlitFramebuffer/glReadBuffer/glDrawBuffer");
    }
    const GLsizei w = static_cast<GLsizei>(atlas_w_);
    const GLsizei h = static_cast<GLsizei>(atlas_h_);
    gl_.BindFramebuffer(GL_READ_FRAMEBUFFER, fbo_ms_);
    gl_.BindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_);

    auto blit_color = [&](GLenum att, GLenum filter) {
        gl_.ReadBuffer(att);
        glDrawBuffer(att);
        glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, filter);
    };
    GLenum color_filter = GL_LINEAR;
    if (const char* mode = std::getenv("ONLINE_OGL_MSAA_COLOR_RESOLVE")) {
        if (mode[0] == 'n' || mode[0] == 'N') color_filter = GL_NEAREST;
    }
    if (spec.metric_depth) blit_color(GL_COLOR_ATTACHMENT0, GL_NEAREST);
    if (spec.rgb) blit_color(GL_COLOR_ATTACHMENT1, color_filter);
    if (spec.normal) blit_color(GL_COLOR_ATTACHMENT2, GL_NEAREST);
    if (spec.object_coord) blit_color(GL_COLOR_ATTACHMENT3, GL_NEAREST);
    if (spec.output_instance_id) blit_color(GL_COLOR_ATTACHMENT4, GL_NEAREST);

    GLenum bufs[5] = {
        GL_COLOR_ATTACHMENT0,
        GL_COLOR_ATTACHMENT1,
        spec.normal ? static_cast<GLenum>(GL_COLOR_ATTACHMENT2) : static_cast<GLenum>(GL_NONE),
        spec.object_coord ? static_cast<GLenum>(GL_COLOR_ATTACHMENT3) : static_cast<GLenum>(GL_NONE),
        spec.output_instance_id ? static_cast<GLenum>(GL_COLOR_ATTACHMENT4) : static_cast<GLenum>(GL_NONE),
    };
    const int n = spec.output_instance_id ? 5 : (spec.object_coord ? 4 : (spec.normal ? 3 : 2));
    gl_.BindFramebuffer(GL_FRAMEBUFFER, fbo_);
    if (n > 0) gl_.DrawBuffers(n, bufs);
}

void BatchRenderer::ensure_fbo(uint32_t w, uint32_t h, uint32_t layers, const RenderOutputSpec& spec) {
    const bool want_layered = spec.layered_tiles && layers >= 1;
    const uint32_t use_layers = want_layered ? std::max(1u, layers) : 1u;
    uint32_t want_msaa = normalize_msaa_samples(spec.msaa_samples, want_layered);
    if (fbo_ && atlas_w_ == w && atlas_h_ == h && fbo_layers_ == use_layers &&
        fbo_layered_ == want_layered && fbo_msaa_samples_ == want_msaa &&
        fbo_output_instance_id_ == spec.output_instance_id && fbo_has_rgb_ == spec.rgb &&
        fbo_has_metric_depth_ == spec.metric_depth && fbo_has_normal_ == spec.normal &&
        fbo_has_object_coord_ == spec.object_coord) {
        return;
    }

    if (pre_fbo_recreate_) {
        pre_fbo_recreate_();
    }

#ifndef GL_TEXTURE_2D_ARRAY
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#endif
#ifndef GL_DEPTH_COMPONENT32F
#define GL_DEPTH_COMPONENT32F 0x8CAC
#endif

    auto del_tex = [&](GLuint& tex) {
        if (tex) {
            gl_.DeleteTextures(1, &tex);
            tex = 0;
        }
    };
    if (fbo_layered_ != want_layered || fbo_output_instance_id_ != spec.output_instance_id ||
        fbo_has_rgb_ != spec.rgb || fbo_has_metric_depth_ != spec.metric_depth ||
        fbo_has_normal_ != spec.normal || fbo_has_object_coord_ != spec.object_coord) {
        del_tex(tex_depth_);
        del_tex(tex_rgb_);
        del_tex(tex_normal_);
        del_tex(tex_coord_);
        del_tex(tex_instance_id_);
        del_tex(tex_hw_depth_);
    }

    destroy_msaa_targets();

    atlas_w_ = w;
    atlas_h_ = h;
    fbo_layers_ = use_layers;
    fbo_layered_ = want_layered;
    fbo_output_instance_id_ = spec.output_instance_id;
    fbo_has_rgb_ = spec.rgb;
    fbo_has_metric_depth_ = spec.metric_depth;
    fbo_has_normal_ = spec.normal;
    fbo_has_object_coord_ = spec.object_coord;

    if (!fbo_) gl_.GenFramebuffers(1, &fbo_);
    gl_.BindFramebuffer(GL_FRAMEBUFFER, fbo_);

    auto glGenRenderbuffers =
        reinterpret_cast<void (*)(GLsizei, GLuint*)>(eglGetProcAddress("glGenRenderbuffers"));
    auto glBindRenderbuffer =
        reinterpret_cast<void (*)(GLenum, GLuint)>(eglGetProcAddress("glBindRenderbuffer"));
    auto glRenderbufferStorage = reinterpret_cast<void (*)(GLenum, GLenum, GLsizei, GLsizei)>(
        eglGetProcAddress("glRenderbufferStorage"));
    auto glRenderbufferStorageMultisample =
        reinterpret_cast<void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei)>(
            eglGetProcAddress("glRenderbufferStorageMultisample"));
    auto glFramebufferRenderbuffer = reinterpret_cast<void (*)(GLenum, GLenum, GLenum, GLuint)>(
        eglGetProcAddress("glFramebufferRenderbuffer"));

    if (want_layered) {
        if (!gl_.TexImage3D || !gl_.FramebufferTexture) {
            throw std::runtime_error("layered_tiles requires glTexImage3D + glFramebufferTexture");
        }
        auto reg_array = [&](GLuint& tex, GLint internal, GLenum format, GLenum type) {
            if (!tex) gl_.GenTextures(1, &tex);
            gl_.BindTexture(GL_TEXTURE_2D_ARRAY, tex);
            gl_.TexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            gl_.TexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            gl_.TexImage3D(GL_TEXTURE_2D_ARRAY, 0, internal, static_cast<GLsizei>(w),
                           static_cast<GLsizei>(h), static_cast<GLsizei>(use_layers), 0, format, type,
                           nullptr);
        };
        if (spec.metric_depth) {
            reg_array(tex_depth_, GL_R32F, GL_RED, GL_FLOAT);
        } else {
            reg_array(tex_depth_, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
        }
        gl_.FramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex_depth_, 0);
        if (spec.rgb) {
            reg_array(tex_rgb_, GL_RGBA32F, GL_RGBA, GL_FLOAT);
        } else {
            reg_array(tex_rgb_, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
        }
        gl_.FramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, tex_rgb_, 0);
        if (spec.normal) {
            reg_array(tex_normal_, GL_RGBA32F, GL_RGBA, GL_FLOAT);
            gl_.FramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, tex_normal_, 0);
        }
        if (spec.object_coord) {
            reg_array(tex_coord_, GL_RGBA16F, GL_RGBA, GL_FLOAT);
            gl_.FramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, tex_coord_, 0);
        }
        if (spec.output_instance_id) {
            reg_array(tex_instance_id_, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT);
            gl_.FramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, tex_instance_id_, 0);
        }
        reg_array(tex_hw_depth_, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
        gl_.FramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, tex_hw_depth_, 0);
    } else {
        auto reg_tex = [&](GLuint& tex, GLenum internal, GLenum format, GLenum type) {
            if (!tex) gl_.GenTextures(1, &tex);
            gl_.BindTexture(GL_TEXTURE_2D, tex);
            gl_.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            gl_.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            gl_.TexImage2D(GL_TEXTURE_2D, 0, internal, static_cast<GLsizei>(w),
                           static_cast<GLsizei>(h), 0, format, type, nullptr);
        };
        if (spec.metric_depth) {
            reg_tex(tex_depth_, GL_R32F, GL_RED, GL_FLOAT);
        } else {
            reg_tex(tex_depth_, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
        }
        gl_.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_depth_, 0);
        if (spec.rgb) {
            reg_tex(tex_rgb_, GL_RGBA32F, GL_RGBA, GL_FLOAT);
        } else {
            reg_tex(tex_rgb_, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
        }
        gl_.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, tex_rgb_, 0);
        if (spec.normal) {
            reg_tex(tex_normal_, GL_RGBA32F, GL_RGBA, GL_FLOAT);
            gl_.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, tex_normal_, 0);
        }
        if (spec.object_coord) {
            reg_tex(tex_coord_, GL_RGBA16F, GL_RGBA, GL_FLOAT);
            gl_.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, tex_coord_, 0);
        }
        if (spec.output_instance_id) {
            reg_tex(tex_instance_id_, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT);
            gl_.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, tex_instance_id_, 0);
        }
        if (want_msaa == 0) {
            if (!depth_rb_ && glGenRenderbuffers) glGenRenderbuffers(1, &depth_rb_);
            if (glBindRenderbuffer && glRenderbufferStorage && glFramebufferRenderbuffer && depth_rb_) {
                glBindRenderbuffer(GL_RENDERBUFFER, depth_rb_);
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, static_cast<GLsizei>(w),
                                      static_cast<GLsizei>(h));
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER,
                                          depth_rb_);
            }
        }
    }

    GLenum bufs[5] = {
        GL_COLOR_ATTACHMENT0,
        GL_COLOR_ATTACHMENT1,
        spec.normal ? static_cast<GLenum>(GL_COLOR_ATTACHMENT2) : static_cast<GLenum>(GL_NONE),
        spec.object_coord ? static_cast<GLenum>(GL_COLOR_ATTACHMENT3) : static_cast<GLenum>(GL_NONE),
        spec.output_instance_id ? static_cast<GLenum>(GL_COLOR_ATTACHMENT4) : static_cast<GLenum>(GL_NONE),
    };
    const int n = spec.output_instance_id ? 5 : (spec.object_coord ? 4 : (spec.normal ? 3 : 2));
    gl_.DrawBuffers(n, bufs);

    const GLenum fb_status = gl_.CheckFramebufferStatus(GL_FRAMEBUFFER);
    if (fb_status != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("FBO incomplete status=" + std::to_string(fb_status) +
                                 (want_layered ? " (layered)" : ""));
    }

    if (want_msaa >= 2) {
        if (!glGenRenderbuffers || !glBindRenderbuffer || !glRenderbufferStorageMultisample ||
            !glFramebufferRenderbuffer) {
            throw std::runtime_error("MSAA requires glRenderbufferStorageMultisample");
        }
        if (!fbo_ms_) gl_.GenFramebuffers(1, &fbo_ms_);
        gl_.BindFramebuffer(GL_FRAMEBUFFER, fbo_ms_);
        auto alloc_ms = [&](GLuint& rb, GLenum internal) {
            if (!rb) glGenRenderbuffers(1, &rb);
            glBindRenderbuffer(GL_RENDERBUFFER, rb);
            glRenderbufferStorageMultisample(GL_RENDERBUFFER, static_cast<GLsizei>(want_msaa),
                                             internal, static_cast<GLsizei>(w),
                                             static_cast<GLsizei>(h));
        };
        if (spec.metric_depth) {
            alloc_ms(rb_ms_depth_, GL_R32F);
        } else {
            alloc_ms(rb_ms_depth_, GL_R8);
        }
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
                                  rb_ms_depth_);
        if (spec.rgb) {
            alloc_ms(rb_ms_rgb_, GL_RGBA32F);
        } else {
            alloc_ms(rb_ms_rgb_, GL_R8);
        }
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER,
                                  rb_ms_rgb_);
        if (spec.normal) {
            alloc_ms(rb_ms_normal_, GL_RGBA32F);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_RENDERBUFFER,
                                      rb_ms_normal_);
        }
        if (spec.object_coord) {
            alloc_ms(rb_ms_coord_, GL_RGBA16F);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_RENDERBUFFER,
                                      rb_ms_coord_);
        }
        if (spec.output_instance_id) {
            alloc_ms(rb_ms_instance_id_, GL_R32UI);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_RENDERBUFFER,
                                      rb_ms_instance_id_);
        }
        alloc_ms(rb_ms_hw_depth_, GL_DEPTH_COMPONENT32F);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER,
                                  rb_ms_hw_depth_);
        if (n > 0) gl_.DrawBuffers(n, bufs);
        const GLenum ms_status = gl_.CheckFramebufferStatus(GL_FRAMEBUFFER);
        if (ms_status != GL_FRAMEBUFFER_COMPLETE) {
            destroy_msaa_targets();
            throw std::runtime_error("MSAA FBO incomplete status=" + std::to_string(ms_status) +
                                     " samples=" + std::to_string(want_msaa));
        }
        fbo_msaa_samples_ = want_msaa;
        gl_.BindFramebuffer(GL_FRAMEBUFFER, fbo_);
    }

    if (std::getenv("WAPR_OGL_DEBUG")) {
        fprintf(stderr,
                "ensure_fbo %ux%u layers=%u layered=%d msaa=%u status=0x%x attachments=%d\n", w, h,
                use_layers, want_layered ? 1 : 0, fbo_msaa_samples_, fb_status, n);
    }
}

void BatchRenderer::set_cuda_device(int device) {
    cuda_device_ = device;
}

void BatchRenderer::arm_device_tiles(const float* poses, const float* bboxes,
                                     uint32_t count, void* stream) {
    device_poses_ = poses;
    device_bboxes_ = bboxes;
    device_count_ = count;
    device_stream_ = stream;
}

void BatchRenderer::disarm_device_tiles() {
    device_poses_ = nullptr;
    device_bboxes_ = nullptr;
    device_count_ = 0;
    device_stream_ = nullptr;
}

void BatchRenderer::ensure_cuda_instance_buffer(size_t bytes) {
    if (cuda_inst_resource_ != nullptr && bytes <= cuda_inst_cap_) {
        return;
    }
    cudaSetDevice(cuda_device_);
    if (cuda_inst_resource_ != nullptr) {
        const cudaError_t unreg = cudaGraphicsUnregisterResource(
            reinterpret_cast<cudaGraphicsResource_t>(cuda_inst_resource_));
        cuda_inst_resource_ = nullptr;
        if (unreg != cudaSuccess) {
            throw std::runtime_error(std::string("cudaGraphicsUnregisterResource: ") +
                                     cudaGetErrorString(unreg));
        }
    }
    if (cuda_inst_ssbo_ != 0) {
        gl_.DeleteBuffers(1, &cuda_inst_ssbo_);
        cuda_inst_ssbo_ = 0;
    }
    const size_t cap = std::max(bytes, size_t(64) * sizeof(InstanceRecordGPU));
    gl_.GenBuffers(1, &cuda_inst_ssbo_);
    gl_.BindBuffer(GL_SHADER_STORAGE_BUFFER, cuda_inst_ssbo_);
    gl_.BufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(cap), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsResource_t resource = nullptr;
    const cudaError_t reg = cudaGraphicsGLRegisterBuffer(
        &resource, cuda_inst_ssbo_, cudaGraphicsRegisterFlagsWriteDiscard);
    if (reg != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGraphicsGLRegisterBuffer: ") +
                                 cudaGetErrorString(reg));
    }
    cuda_inst_resource_ = resource;
    cuda_inst_cap_ = cap;
}

void BatchRenderer::upload_device_instances(const std::vector<RenderInstance>& instances,
                                            const CameraIntrinsics& cam,
                                            const RenderOutputSpec& spec) {
    const float* poses = device_poses_;
    const float* bboxes = device_bboxes_;
    const uint32_t armed = device_count_;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(device_stream_);
    disarm_device_tiles();
    if (poses == nullptr || bboxes == nullptr) {
        throw std::runtime_error("device tile poses are missing");
    }
    if (spec.mode != RenderMode::Tile || spec.viewport_per_tile) {
        throw std::runtime_error("device tiles are only for the tile renderer");
    }
    const uint32_t count = static_cast<uint32_t>(instances.size());
    if (count == 0 || count > 256 || count != armed) {
        throw std::runtime_error("device tile count does not match the armed poses");
    }
    const auto& mesh_table = assets_.mesh_table();
    for (const auto& inst : instances) {
        if (inst.mesh_id >= mesh_table.size() || !mesh_table[inst.mesh_id].valid ||
            mesh_table[inst.mesh_id].index_count == 0) {
            throw std::runtime_error("invalid mesh for device tiles");
        }
    }
    const bool layered = spec.layered_tiles;
    const TileAtlasLayout layout = compute_tile_atlas_layout(
        count, spec.tile_w, spec.tile_h, spec.max_atlas_w, spec.max_atlas_h);
    if (layered) {
        atlas_w_ = spec.tile_w;
        atlas_h_ = spec.tile_h;
        atlas_cols_ = 1;
    } else {
        atlas_w_ = layout.atlas_w;
        atlas_h_ = layout.atlas_h;
        atlas_cols_ = layout.cols;
    }
    last_scene_batch_ = false;
    last_proj_cams_.clear();

    std::vector<TileMeshData> mesh_rows(count);
    last_gpu_instances_.assign(count, InstanceRecordGPU{});
    last_cmds_.assign(count, DrawElementsIndirectCommand{});
    for (uint32_t i = 0; i < count; ++i) {
        const auto& mesh = mesh_table[instances[i].mesh_id];
        auto& row = mesh_rows[i];
        row.mesh_id = instances[i].mesh_id;
        row.tex_layer = mesh.has_texture ? static_cast<int32_t>(mesh.tex_layer) : -1;
        row.force_flat_normal = mesh.force_flat_normal;
        row.has_bounds = mesh.has_bounds;
        std::copy(mesh.bounds_min, mesh.bounds_min + 3, row.bmin);
        std::copy(mesh.bounds_max, mesh.bounds_max + 3, row.bmax);
        last_gpu_instances_[i].tile_slot = static_cast<int32_t>(i);
        if (layered) {
            last_gpu_instances_[i].tile_x = 0;
            last_gpu_instances_[i].tile_y = 0;
        } else {
            last_gpu_instances_[i].tile_x = (i % layout.cols) * spec.tile_w;
            last_gpu_instances_[i].tile_y = (i / layout.cols) * spec.tile_h;
        }
        last_cmds_[i].count = mesh.index_count;
        last_cmds_[i].instanceCount = 1;
        last_cmds_[i].firstIndex = mesh.index_offset;
        last_cmds_[i].baseVertex = 0;
        last_cmds_[i].baseInstance = i;
    }

    const size_t record_bytes = static_cast<size_t>(count) * sizeof(InstanceRecordGPU);
    const size_t mesh_bytes = static_cast<size_t>(count) * sizeof(TileMeshData);
    ensure_cuda_instance_buffer(record_bytes + mesh_bytes);
    cudaGraphicsResource_t resource = reinterpret_cast<cudaGraphicsResource_t>(cuda_inst_resource_);
    cudaError_t err = cudaGraphicsMapResources(1, &resource, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGraphicsMapResources: ") + cudaGetErrorString(err));
    }
    void* mapped = nullptr;
    size_t mapped_bytes = 0;
    err = cudaGraphicsResourceGetMappedPointer(&mapped, &mapped_bytes, resource);
    if (err != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &resource, stream);
        throw std::runtime_error(std::string("cudaGraphicsResourceGetMappedPointer: ") +
                                 cudaGetErrorString(err));
    }
    // One metadata upload and one CUDA kernel for the entire mixed-mesh batch.
    // 整批混合网格仅上传一次元数据，并执行一次 CUDA 核函数。
    auto* mesh_device = reinterpret_cast<TileMeshData*>(static_cast<char*>(mapped) + record_bytes);
    err = cudaMemcpyAsync(mesh_device, mesh_rows.data(), mesh_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &resource, stream);
        throw std::runtime_error(std::string("tile metadata upload: ") + cudaGetErrorString(err));
    }
    fill_indexed_tile_instances(
        poses, bboxes, static_cast<InstanceRecordGPU*>(mapped), count,
        cam.fx, cam.fy, cam.cx, cam.cy,
        static_cast<float>(cam.width), static_cast<float>(cam.height),
        spec.tile_w, spec.tile_h, layout.cols, mesh_device,
        layered ? 1u : 0u, stream);
    err = cudaGraphicsUnmapResources(1, &resource, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGraphicsUnmapResources: ") + cudaGetErrorString(err));
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaStreamSynchronize: ") + cudaGetErrorString(err));
    }

    cuda_instance_active_ = true;
    persist_active_ = false;
    persist_inst_offset_ = 0;
    persist_cmd_offset_ = 0;

    gl_.BindBuffer(GL_SHADER_STORAGE_BUFFER, mesh_table_ssbo_);
    gl_.BufferData(GL_SHADER_STORAGE_BUFFER,
                   static_cast<GLsizeiptr>(mesh_table.size() * sizeof(MeshTableEntry)),
                   mesh_table.data(), GL_DYNAMIC_DRAW);
    gl_.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mesh_table_ssbo_);

    if (cuda_indirect_buf_ == 0) {
        gl_.GenBuffers(1, &cuda_indirect_buf_);
    }
    gl_.BindBuffer(GL_DRAW_INDIRECT_BUFFER, cuda_indirect_buf_);
    gl_.BufferData(GL_DRAW_INDIRECT_BUFFER,
                   static_cast<GLsizeiptr>(count * sizeof(DrawElementsIndirectCommand)),
                   last_cmds_.data(), GL_DYNAMIC_DRAW);
}

void BatchRenderer::upload_instances(const std::vector<RenderInstance>& instances,
                                     const CameraIntrinsics& cam,
                                     const RenderOutputSpec& spec) {
    if (device_poses_ != nullptr) {
        upload_device_instances(instances, cam, spec);
        return;
    }
    cuda_instance_active_ = false;
    std::vector<InstanceRecordGPU> gpu_instances(instances.size());
    std::vector<DrawElementsIndirectCommand> cmds(instances.size());

    const auto& mesh_table = assets_.mesh_table();
    const uint32_t n_inst = static_cast<uint32_t>(instances.size());
    const uint32_t n_tiles = (spec.mode == RenderMode::Tile && !spec.layered_tiles)
                                 ? tile_batch_count(instances)
                                 : n_inst;
    TileAtlasLayout layout{};
    uint32_t atlas_w = cam.width, atlas_h = cam.height;
    CameraIntrinsics scene_cam = cam;
    adapt_clip_planes_for_scene(scene_cam, instances, assets_.mesh_table());
    const bool scene_batch = scene_batch_active(spec, instances);
    last_scene_batch_ = scene_batch;
    last_scene_w_ = cam.width;
    last_scene_h_ = cam.height;

    if (spec.layered_tiles && spec.mode == RenderMode::Tile) {
        layout.cols = 1;
        layout.rows = n_inst;
        layout.atlas_w = spec.tile_w;
        layout.atlas_h = spec.tile_h;
        atlas_w = spec.tile_w;
        atlas_h = spec.tile_h;
        atlas_cols_ = 1;
    } else if (spec.mode == RenderMode::Tile) {
        layout = compute_tile_atlas_layout(
            n_tiles, spec.tile_w, spec.tile_h, spec.max_atlas_w, spec.max_atlas_h);
        atlas_w = layout.atlas_w;
        atlas_h = layout.atlas_h;
        atlas_cols_ = layout.cols;
    } else if (scene_batch) {
        const uint32_t n_scenes = scene_batch_count(instances);
        layout = compute_tile_atlas_layout(
            n_scenes, cam.width, cam.height, spec.max_atlas_w, spec.max_atlas_h);
        atlas_w = layout.atlas_w;
        atlas_h = layout.atlas_h;
        atlas_cols_ = layout.cols;
    } else {
        atlas_cols_ = 1;
    }

    const bool skip_atlas_warp =
        spec.viewport_per_tile || spec.layered_tiles || spec.indexed_viewport_scissor;

    last_proj_cams_.clear();
    last_proj_cams_.reserve(instances.size());

    for (size_t i = 0; i < instances.size(); ++i) {
        const auto& inst = instances[i];
        if (inst.mesh_id >= mesh_table.size() || !mesh_table[inst.mesh_id].valid ||
            mesh_table[inst.mesh_id].index_count == 0) {
            throw std::runtime_error("invalid or unloaded mesh_id=" + std::to_string(inst.mesh_id));
        }
        auto& gi = gpu_instances[i];
        gi.mesh_id = inst.mesh_id;
        gi.depth_func = inst.depth_func;
        if (spec.layered_tiles && spec.mode == RenderMode::Tile) {
            gi.tile_x = 0;
            gi.tile_y = 0;
        } else if (spec.mode == RenderMode::Tile) {
            const uint32_t slot = inst.tile_slot;
            gi.tile_x = (slot % layout.cols) * spec.tile_w;
            gi.tile_y = (slot / layout.cols) * spec.tile_h;
        } else if (scene_batch) {
            const uint32_t slot = inst.tile_slot;
            gi.tile_x = (slot % layout.cols) * cam.width;
            gi.tile_y = (slot / layout.cols) * cam.height;
        } else {
            gi.tile_x = 0;
            gi.tile_y = 0;
        }

        CameraIntrinsics proj_base = scene_cam;
        if (inst.has_K) {
            proj_base.fx = inst.fx;
            proj_base.fy = inst.fy;
            proj_base.cx = inst.cx;
            proj_base.cy = inst.cy;
        }
        CameraIntrinsics proj_cam = proj_base;
        if (spec.projection == ProjectionMode::PinholeCrop && inst.has_bbox) {
            proj_cam = pinhole_intrinsics_for_crop(
                proj_base, inst.bbox_xywh[0], inst.bbox_xywh[1], inst.bbox_xywh[2], inst.bbox_xywh[3],
                spec.mode == RenderMode::Tile ? spec.tile_w : cam.width,
                spec.mode == RenderMode::Tile ? spec.tile_h : cam.height);
        }
        last_proj_cams_.push_back(proj_cam);

        float mvp[16], model[16], mvp_gl[16], model_gl[16];
        float warp4[16] = {0};
        bool use_warp = false;
        if (inst.has_clip_affine) {
            std::memcpy(warp4, inst.clip_affine4, sizeof(warp4));
            use_warp = true;
        } else if (inst.has_view_warp && spec.projection == ProjectionMode::ClipAffine) {
            warp4[0] = inst.view_warp[0];
            warp4[1] = inst.view_warp[1];
            warp4[3] = inst.view_warp[2];
            warp4[4] = inst.view_warp[3];
            warp4[5] = inst.view_warp[4];
            warp4[7] = inst.view_warp[5];
            warp4[10] = 1.f;
            warp4[15] = 1.f;
            use_warp = true;
        }

        if (inst.has_precomputed_mvp) {
            float mvp_row[16];
            mat4_transpose(inst.mvp_pre, mvp_row);
            if (spec.mode == RenderMode::Tile && !skip_atlas_warp) {
                tile_atlas_place_row(mvp_row, gi.tile_x, gi.tile_y, spec.tile_w, spec.tile_h, atlas_w, atlas_h);
            }
            mat4_to_gl_colmajor(mvp_row, mvp_gl);
            std::memcpy(model_gl, inst.model_cam_pre, sizeof(model_gl));
        } else {
            build_mvp(proj_cam, inst.T_cam_obj, use_warp ? warp4 : nullptr, use_warp, mvp, model);
            if (spec.mode == RenderMode::Tile && !skip_atlas_warp) {
                tile_atlas_place_row(mvp, gi.tile_x, gi.tile_y, spec.tile_w, spec.tile_h, atlas_w, atlas_h);
            }
            mat4_to_gl_colmajor(mvp, mvp_gl);
            mat4_to_gl_colmajor(model, model_gl);
        }
        std::memcpy(gi.mvp, mvp_gl, sizeof(mvp_gl));
        std::memcpy(gi.model_cam, model_gl, sizeof(model_gl));
        std::memcpy(gi.clip_affine, inst.clip_affine4, sizeof(gi.clip_affine));
        gi.has_clip_affine = inst.has_clip_affine ? 1 : 0;

        const auto& me = mesh_table.at(inst.mesh_id);
        gi.tex_layer = me.has_texture ? static_cast<int32_t>(me.tex_layer) : -1;
        gi.force_flat_normal = me.force_flat_normal;
        gi.metallic = inst.has_material ? inst.metallic : 0.f;
        gi.roughness = inst.has_material ? inst.roughness : 0.5f;
        gi.tile_slot = static_cast<int32_t>(inst.tile_slot);
        gi.instance_id = static_cast<int32_t>(inst.instance_id);
        gi.fx = proj_cam.fx;
        gi.fy = proj_cam.fy;
        gi.cx = proj_cam.cx;
        gi.cy = proj_cam.cy;
        cmds[i].count = me.index_count;
        cmds[i].instanceCount = 1;
        cmds[i].firstIndex = me.index_offset;
        cmds[i].baseVertex = 0;
        cmds[i].baseInstance = static_cast<uint32_t>(i);
    }

    const GLsizeiptr inst_bytes =
        static_cast<GLsizeiptr>(gpu_instances.size() * sizeof(InstanceRecordGPU));
    const GLsizeiptr cmd_bytes =
        static_cast<GLsizeiptr>(cmds.size() * sizeof(DrawElementsIndirectCommand));
    const bool can_persist =
        caps_.map_persistent && caps_.sync && gl_.BindBufferRange != nullptr;
    const bool use_persist =
        can_persist && (spec.persistent_buffers || persist_inst_ptr_ || persist_cmd_ptr_);
    if (use_persist) {
        ensure_persistent_buffers(inst_bytes, cmd_bytes);
        persist_slot_ = (persist_slot_ + 1) % kPersistSlots;
        if (persist_fences_[persist_slot_] && gl_.ClientWaitSync) {
            for (;;) {
                const GLenum status = gl_.ClientWaitSync(
                    persist_fences_[persist_slot_], GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull);
                if (status == GL_ALREADY_SIGNALED || status == GL_CONDITION_SATISFIED) break;
                if (status == GL_WAIT_FAILED) {
                    throw std::runtime_error("glClientWaitSync failed for persistent buffer slot");
                }
            }
            gl_.DeleteSync(persist_fences_[persist_slot_]);
            persist_fences_[persist_slot_] = nullptr;
        }
        persist_inst_offset_ = persist_slot_ * persist_inst_stride_;
        persist_cmd_offset_ = persist_slot_ * persist_cmd_stride_;
        std::memcpy(static_cast<char*>(persist_inst_ptr_) + persist_inst_offset_,
                    gpu_instances.data(), static_cast<size_t>(inst_bytes));
        std::memcpy(static_cast<char*>(persist_cmd_ptr_) + persist_cmd_offset_,
                    cmds.data(), static_cast<size_t>(cmd_bytes));
        gl_.BindBuffer(GL_SHADER_STORAGE_BUFFER, instance_ssbo_);
        gl_.BindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, instance_ssbo_,
                            persist_inst_offset_, inst_bytes);

        gl_.BindBuffer(GL_SHADER_STORAGE_BUFFER, mesh_table_ssbo_);
        gl_.BufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<GLsizeiptr>(mesh_table.size() * sizeof(MeshTableEntry)),
                       mesh_table.data(), GL_DYNAMIC_DRAW);
        gl_.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mesh_table_ssbo_);
        gl_.BindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_buf_);
        persist_active_ = true;
    } else {
        persist_active_ = false;
        persist_inst_offset_ = 0;
        persist_cmd_offset_ = 0;
        gl_.BindBuffer(GL_SHADER_STORAGE_BUFFER, instance_ssbo_);
        gl_.BufferData(GL_SHADER_STORAGE_BUFFER, inst_bytes, gpu_instances.data(), GL_DYNAMIC_DRAW);
        gl_.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, instance_ssbo_);

        gl_.BindBuffer(GL_SHADER_STORAGE_BUFFER, mesh_table_ssbo_);
        gl_.BufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<GLsizeiptr>(mesh_table.size() * sizeof(MeshTableEntry)),
                       mesh_table.data(), GL_DYNAMIC_DRAW);
        gl_.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mesh_table_ssbo_);

        gl_.BindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_buf_);
        gl_.BufferData(GL_DRAW_INDIRECT_BUFFER, cmd_bytes, cmds.data(), GL_DYNAMIC_DRAW);
    }
    last_gpu_instances_ = std::move(gpu_instances);
    last_cmds_ = std::move(cmds);
}

void BatchRenderer::ensure_persistent_buffers(GLsizeiptr inst_bytes, GLsizeiptr cmd_bytes) {
    GLint ssbo_alignment_i = 1;
    if (gl_.GetIntegerv) {
        gl_.GetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &ssbo_alignment_i);
    }
    const GLsizeiptr inst_stride = std::max(
        persist_inst_stride_,
        align_up(std::max<GLsizeiptr>(inst_bytes, 1), std::max(1, ssbo_alignment_i)));
    const GLsizeiptr cmd_stride = std::max(
        persist_cmd_stride_, align_up(std::max<GLsizeiptr>(cmd_bytes, 1), 16));
    const GLsizeiptr need_inst = inst_stride * kPersistSlots;
    const GLsizeiptr need_cmd = cmd_stride * kPersistSlots;
    const bool grow_inst = !persist_inst_ptr_ || need_inst > persist_inst_cap_;
    const bool grow_cmd = !persist_cmd_ptr_ || need_cmd > persist_cmd_cap_;
    if (!grow_inst && !grow_cmd) {
        persist_inst_stride_ = inst_stride;
        persist_cmd_stride_ = cmd_stride;
        return;
    }

    for (auto& fence : persist_fences_) {
        if (!fence) continue;
        for (;;) {
            const GLenum status =
                gl_.ClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull);
            if (status == GL_ALREADY_SIGNALED || status == GL_CONDITION_SATISFIED) break;
            if (status == GL_WAIT_FAILED) {
                throw std::runtime_error("glClientWaitSync failed while growing persistent buffers");
            }
        }
        gl_.DeleteSync(fence);
        fence = nullptr;
    }

    const GLbitfield storage_flags =
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_DYNAMIC_STORAGE_BIT;
    const GLbitfield map_flags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    if (grow_inst) {
        if (persist_inst_ptr_ && gl_.UnmapBuffer) {
            gl_.BindBuffer(GL_SHADER_STORAGE_BUFFER, instance_ssbo_);
            gl_.UnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        }
        if (instance_ssbo_) gl_.DeleteBuffers(1, &instance_ssbo_);
        gl_.GenBuffers(1, &instance_ssbo_);
        gl_.BindBuffer(GL_SHADER_STORAGE_BUFFER, instance_ssbo_);
        gl_.BufferStorage(GL_SHADER_STORAGE_BUFFER, need_inst, nullptr, storage_flags);
        persist_inst_ptr_ = gl_.MapBufferRange(
            GL_SHADER_STORAGE_BUFFER, 0, need_inst, map_flags);
        if (!persist_inst_ptr_) throw std::runtime_error("persistent instance buffer map failed");
        persist_inst_cap_ = need_inst;
    }
    if (grow_cmd) {
        if (persist_cmd_ptr_ && gl_.UnmapBuffer) {
            gl_.BindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_buf_);
            gl_.UnmapBuffer(GL_DRAW_INDIRECT_BUFFER);
        }
        if (indirect_buf_) gl_.DeleteBuffers(1, &indirect_buf_);
        gl_.GenBuffers(1, &indirect_buf_);
        gl_.BindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_buf_);
        gl_.BufferStorage(GL_DRAW_INDIRECT_BUFFER, need_cmd, nullptr, storage_flags);
        persist_cmd_ptr_ = gl_.MapBufferRange(
            GL_DRAW_INDIRECT_BUFFER, 0, need_cmd, map_flags);
        if (!persist_cmd_ptr_) throw std::runtime_error("persistent indirect buffer map failed");
        persist_cmd_cap_ = need_cmd;
    }
    persist_inst_stride_ = inst_stride;
    persist_cmd_stride_ = cmd_stride;
    persist_slot_ = -1;
}

void BatchRenderer::draw_batch(const std::vector<RenderInstance>& instances,
                               uint32_t depth_func,
                               const CameraIntrinsics& cam,
                               const RenderOutputSpec& spec) {
    gl_.Enable(GL_DEPTH_TEST);
#ifndef GL_CLIP_DISTANCE0
#define GL_CLIP_DISTANCE0 0x3000
#endif
    gl_.Enable(GL_CLIP_DISTANCE0);
    gl_.Disable(0x0B44);
    if (depth_func == kDepthFuncGreater) {
        gl_.DepthFunc(kDepthFuncGreater);
        gl_.ClearDepthf(0.f);
    } else {
        gl_.DepthFunc(kDepthFuncLess);
        gl_.ClearDepthf(1.f);
    }
    if (!spec.viewport_per_tile && !last_scene_batch_) {
        gl_.Clear(GL_DEPTH_BUFFER_BIT);
    }

    assets_.bind_for_draw();
    gl_.UseProgram(program_);
    gl_.BindBuffer(GL_DRAW_INDIRECT_BUFFER,
                   cuda_instance_active_ ? cuda_indirect_buf_ : indirect_buf_);
    if (cuda_instance_active_) {
        gl_.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cuda_inst_ssbo_);
    } else if (persist_active_) {
        gl_.BindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, instance_ssbo_,
                            persist_inst_offset_,
                            static_cast<GLsizeiptr>(last_gpu_instances_.size() *
                                                   sizeof(InstanceRecordGPU)));
    } else {
        gl_.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, instance_ssbo_);
    }

    const GLsizei draw_count = static_cast<GLsizei>(instances.size());
    if (draw_count <= 0) return;

    const bool indexed_ok =
        spec.indexed_viewport_scissor && caps_.viewport_array &&
        caps_.shader_viewport_layer_array && gl_.ViewportArrayv && gl_.ScissorArrayv;
    const bool mdi_fast =
        !spec.viewport_per_tile && !last_scene_batch_ &&
        (!spec.use_scissor || indexed_ok) &&
        caps_.shader_draw_parameters &&
        gl_.MultiDrawElementsIndirect != nullptr;

    static const bool debug_draw = std::getenv("WAPR_OGL_DEBUG") != nullptr;
    if (debug_draw) {
        fprintf(stderr,
                "draw count=%d mdi_fast=%d scissor=%d indexed=%d vp_tile=%d\n",
                draw_count, mdi_fast ? 1 : 0, spec.use_scissor ? 1 : 0,
                indexed_ok ? 1 : 0, spec.viewport_per_tile ? 1 : 0);
    }

    last_draw_api_calls_ = 0;
    if (mdi_fast) {
        if (loc_tile_ofs_ >= 0) {
            gl_.Uniform2f(loc_tile_ofs_, 0.f, 0.f);
        }

        if (indexed_ok) {
            const int max_vp = std::max(1, caps_.max_viewports);
            std::vector<GLfloat> vps(static_cast<size_t>(max_vp) * 4, 0.f);
            std::vector<GLint> scs(static_cast<size_t>(max_vp) * 4, 0);
            int used = 0;
            for (const auto& gi : last_gpu_instances_) {
                const int slot = std::max(0, gi.tile_slot);
                if (slot >= max_vp) {
                    throw std::runtime_error(
                        "tile_slot exceeds GL_MAX_VIEWPORTS; lower ONLINE_OGL_TILE_BATCH");
                }
                used = std::max(used, slot + 1);
                vps[static_cast<size_t>(slot) * 4 + 0] = static_cast<GLfloat>(gi.tile_x);
                vps[static_cast<size_t>(slot) * 4 + 1] = static_cast<GLfloat>(gi.tile_y);
                vps[static_cast<size_t>(slot) * 4 + 2] = static_cast<GLfloat>(spec.tile_w);
                vps[static_cast<size_t>(slot) * 4 + 3] = static_cast<GLfloat>(spec.tile_h);
                scs[static_cast<size_t>(slot) * 4 + 0] = static_cast<GLint>(gi.tile_x);
                scs[static_cast<size_t>(slot) * 4 + 1] = static_cast<GLint>(gi.tile_y);
                scs[static_cast<size_t>(slot) * 4 + 2] = static_cast<GLint>(spec.tile_w);
                scs[static_cast<size_t>(slot) * 4 + 3] = static_cast<GLint>(spec.tile_h);
            }
            if (used <= 0) used = 1;
            gl_.Enable(GL_SCISSOR_TEST);
            gl_.ViewportArrayv(0, used, vps.data());
            gl_.ScissorArrayv(0, used, scs.data());
            if (loc_viewport_ >= 0) {
                gl_.Uniform2f(loc_viewport_,
                              static_cast<GLfloat>(spec.tile_w),
                              static_cast<GLfloat>(spec.tile_h));
            }
        }

        bool same_mesh = !last_cmds_.empty();
        for (GLsizei i = 1; same_mesh && i < draw_count; ++i) {
            if (last_cmds_[static_cast<size_t>(i)].count != last_cmds_[0].count ||
                last_cmds_[static_cast<size_t>(i)].firstIndex != last_cmds_[0].firstIndex) {
                same_mesh = false;
            }
        }
        if (same_mesh && gl_.DrawElementsIndirect) {
            DrawElementsIndirectCommand cmd = last_cmds_[0];
            cmd.instanceCount = static_cast<uint32_t>(draw_count);
            cmd.baseInstance = 0;
            if (persist_active_) {
                std::memcpy(static_cast<char*>(persist_cmd_ptr_) + persist_cmd_offset_,
                            &cmd, sizeof(cmd));
            } else if (glBufferSubData_) {
                glBufferSubData_(GL_DRAW_INDIRECT_BUFFER, 0, sizeof(cmd), &cmd);
            }
            gl_.DrawElementsIndirect(
                GL_TRIANGLES, GL_UNSIGNED_INT,
                reinterpret_cast<void*>(static_cast<uintptr_t>(persist_cmd_offset_)));
            last_draw_api_calls_ = 1;
        } else {
            gl_.MultiDrawElementsIndirect(
                GL_TRIANGLES, GL_UNSIGNED_INT,
                reinterpret_cast<void*>(static_cast<uintptr_t>(persist_cmd_offset_)),
                draw_count, 0);
            last_draw_api_calls_ = 1;
        }
        if (indexed_ok) {
            gl_.Disable(GL_SCISSOR_TEST);
            gl_.Viewport(0, 0, static_cast<GLsizei>(atlas_w_), static_cast<GLsizei>(atlas_h_));
        }
        if (persist_active_ && gl_.FenceSync) {
            if (persist_fences_[persist_slot_]) {
                gl_.DeleteSync(persist_fences_[persist_slot_]);
            }
            persist_fences_[persist_slot_] = gl_.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        }
        return;
    }

    const GLint loc_tile_ofs = loc_tile_ofs_;
    const GLint loc_viewport = loc_viewport_;
    const GLint loc_K = loc_K_;
    const GLint loc_proj_z = loc_proj_z_;
    const GLint loc_has_clip_affine = loc_has_clip_affine_;
    const GLint loc_clip_affine = loc_clip_affine_;

    if (spec.use_scissor && glScissor_) {
        gl_.Enable(GL_SCISSOR_TEST);
    }

    std::unordered_set<uint32_t> cleared_slots;
    const bool force_scene_scissor = last_scene_batch_ && glScissor_;
    if (force_scene_scissor) {
        gl_.Enable(GL_SCISSOR_TEST);
    }

    for (GLsizei i = 0; i < draw_count; ++i) {
        if (i < static_cast<GLsizei>(last_gpu_instances_.size())) {
            const auto& gi = last_gpu_instances_[static_cast<size_t>(i)];
            const GLint vp_x = static_cast<GLint>(gi.tile_x);
            const GLint vp_y = static_cast<GLint>(gi.tile_y);
            const bool vp_per_cell = spec.viewport_per_tile || last_scene_batch_;
            const GLsizei vp_w = last_scene_batch_
                                     ? static_cast<GLsizei>(last_scene_w_)
                                     : static_cast<GLsizei>(spec.tile_w);
            const GLsizei vp_h = last_scene_batch_
                                     ? static_cast<GLsizei>(last_scene_h_)
                                     : static_cast<GLsizei>(spec.tile_h);

            if (vp_per_cell) {
                gl_.Viewport(vp_x, vp_y, vp_w, vp_h);
                if ((spec.use_scissor || last_scene_batch_) && glScissor_) {
                    glScissor_(vp_x, vp_y, vp_w, vp_h);
                }
                gl_.Uniform2f(loc_viewport,
                              static_cast<GLfloat>(vp_w),
                              static_cast<GLfloat>(vp_h));
                if (cam.use_distortion && i < static_cast<GLsizei>(last_proj_cams_.size())) {
                    const CameraIntrinsics& kcam = last_proj_cams_[static_cast<size_t>(i)];
                    gl_.Uniform4f(loc_K, kcam.fx, kcam.fy, kcam.cx, kcam.cy);
                    float pq = 0.f, pqn = 0.f;
                    projection_depth_coeffs(kcam, pq, pqn);
                    gl_.Uniform2f(loc_proj_z, pq, pqn);
                }
                if (last_scene_batch_) {
                    const uint32_t slot = instances[static_cast<size_t>(i)].tile_slot;
                    if (cleared_slots.insert(slot).second) {
                        gl_.Clear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
                    }
                } else {
                    const uint32_t slot = instances[static_cast<size_t>(i)].tile_slot;
                    if (cleared_slots.insert(slot).second) {
                        gl_.Clear(GL_DEPTH_BUFFER_BIT);
                    }
                }
            } else if (spec.use_scissor && glScissor_) {
                glScissor_(vp_x, vp_y,
                           static_cast<GLsizei>(spec.tile_w),
                           static_cast<GLsizei>(spec.tile_h));
            }
            if (!vp_per_cell && cam.use_distortion && loc_K >= 0 &&
                i < static_cast<GLsizei>(last_proj_cams_.size())) {
                const CameraIntrinsics& kcam = last_proj_cams_[static_cast<size_t>(i)];
                gl_.Uniform4f(loc_K, kcam.fx, kcam.fy, kcam.cx, kcam.cy);
                float pq = 0.f, pqn = 0.f;
                projection_depth_coeffs(kcam, pq, pqn);
                if (loc_proj_z >= 0) {
                    gl_.Uniform2f(loc_proj_z, pq, pqn);
                }
            }

            if (cam.use_distortion && loc_has_clip_affine >= 0) {
                const auto& src = instances[static_cast<size_t>(i)];
                if (src.has_clip_affine) {
                    float tfT[16], warp_gl[16];
                    mat4_transpose(src.clip_affine4, tfT);
                    mat4_to_gl_colmajor(tfT, warp_gl);
                    gl_.Uniform1i(loc_has_clip_affine, 1);
                    gl_.UniformMatrix4fv(loc_clip_affine, 1, GL_FALSE, warp_gl);
                } else {
                    gl_.Uniform1i(loc_has_clip_affine, 0);
                }
            }
            gl_.Uniform2f(loc_tile_ofs,
                          static_cast<GLfloat>(gi.tile_x),
                          static_cast<GLfloat>(gi.tile_y));
        }
        if (gl_.DrawElementsIndirect) {
            gl_.DrawElementsIndirect(
                GL_TRIANGLES,
                GL_UNSIGNED_INT,
                reinterpret_cast<void*>(
                    static_cast<uintptr_t>(persist_cmd_offset_) +
                    static_cast<uintptr_t>(i) * sizeof(DrawElementsIndirectCommand)));
            last_draw_api_calls_ += 1;
        }
    }

    if (spec.use_scissor || last_scene_batch_) {
        gl_.Disable(GL_SCISSOR_TEST);
    }
    if (spec.viewport_per_tile || last_scene_batch_) {
        gl_.Viewport(0, 0, static_cast<GLsizei>(atlas_w_), static_cast<GLsizei>(atlas_h_));
    }
    if (persist_active_ && gl_.FenceSync) {
        if (persist_fences_[persist_slot_]) gl_.DeleteSync(persist_fences_[persist_slot_]);
        persist_fences_[persist_slot_] = gl_.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    }
}

void BatchRenderer::render(const std::vector<RenderInstance>& instances,
                           const CameraIntrinsics& cam,
                           const RenderOutputSpec& spec,
                           const LightParams& light,
                           const std::vector<LightParams>* tile_lights) {
    if (instances.empty()) return;
    egl_.make_current();

    uint32_t fb_w = cam.width, fb_h = cam.height;
    uint32_t fb_layers = 1;
    const bool scene_batch = scene_batch_active(spec, instances);
    if (spec.layered_tiles && spec.mode == RenderMode::Tile) {
        fb_w = spec.tile_w;
        fb_h = spec.tile_h;
        fb_layers = std::max(1u, tile_batch_count(instances));
        atlas_cols_ = 1;
    } else if (spec.mode == RenderMode::Tile) {
        const auto layout = compute_tile_atlas_layout(
            tile_batch_count(instances),
            spec.tile_w, spec.tile_h, spec.max_atlas_w, spec.max_atlas_h);
        fb_w = layout.atlas_w;
        fb_h = layout.atlas_h;
        atlas_cols_ = layout.cols;
    } else if (scene_batch) {
        const auto layout = compute_tile_atlas_layout(
            scene_batch_count(instances), cam.width, cam.height,
            spec.max_atlas_w, spec.max_atlas_h);
        fb_w = layout.atlas_w;
        fb_h = layout.atlas_h;
        atlas_cols_ = layout.cols;
    }

    ensure_fbo(fb_w, fb_h, fb_layers, spec);
    fbo_ping_ = 1u - fbo_ping_;
    const GLuint draw_fbo = (fbo_msaa_samples_ >= 2 && fbo_ms_) ? fbo_ms_ : fbo_;
    gl_.BindFramebuffer(GL_FRAMEBUFFER, draw_fbo);
    gl_.Viewport(0, 0, static_cast<GLsizei>(fb_w), static_cast<GLsizei>(fb_h));
    gl_.ClearColor(0.f, 0.f, 0.f, 0.f);
    gl_.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (spec.output_instance_id) {
        auto glClearBufferuiv = reinterpret_cast<void (*)(GLenum, GLint, const GLuint*)>(
            eglGetProcAddress("glClearBufferuiv"));
        if (glClearBufferuiv) {
            const GLuint zero = 0u;
            glClearBufferuiv(GL_COLOR, 4, &zero);
        }
    }

    const bool want_sample_shading =
        fbo_msaa_samples_ >= 2 && fbo_ms_ && spec.sample_shading > 0.f;
    auto glMinSampleShading =
        reinterpret_cast<void (*)(GLfloat)>(eglGetProcAddress("glMinSampleShading"));
    if (want_sample_shading) {
        gl_.Enable(GL_SAMPLE_SHADING);
        if (glMinSampleShading) {
            const float frac = std::min(1.f, std::max(spec.sample_shading, 1e-3f));
            glMinSampleShading(frac);
        }
    } else {
        gl_.Disable(GL_SAMPLE_SHADING);
    }

    gl_.UseProgram(program_);
    gl_.Uniform2f(gl_.GetUniformLocation(program_, "u_atlas_size"),
                  static_cast<GLfloat>(fb_w), static_cast<GLfloat>(fb_h));
    gl_.Uniform2f(gl_.GetUniformLocation(program_, "u_tile_size"),
                  static_cast<GLfloat>(spec.tile_w), static_cast<GLfloat>(spec.tile_h));
    gl_.Uniform1i(gl_.GetUniformLocation(program_, "u_scene_mode"),
                  spec.mode == RenderMode::Scene ? 1 : 0);
    gl_.Uniform1i(gl_.GetUniformLocation(program_, "u_layered"),
                  (spec.layered_tiles && spec.mode == RenderMode::Tile) ? 1 : 0);
    const bool indexed_ok =
        spec.indexed_viewport_scissor && caps_.viewport_array &&
        caps_.shader_viewport_layer_array;
    gl_.Uniform1i(gl_.GetUniformLocation(program_, "u_indexed_viewport"),
                  (indexed_ok && spec.mode == RenderMode::Tile) ? 1 : 0);
    gl_.Uniform1i(gl_.GetUniformLocation(program_, "u_apply_tile_remap"),
                  (spec.mode == RenderMode::Tile && !spec.viewport_per_tile &&
                   !spec.layered_tiles && !indexed_ok && cam.use_distortion)
                      ? 1
                      : 0);
    gl_.Uniform1f(gl_.GetUniformLocation(program_, "u_ambient"), light.ambient);
    gl_.Uniform1f(gl_.GetUniformLocation(program_, "u_diffuse"), light.diffuse);
    gl_.Uniform3f(gl_.GetUniformLocation(program_, "u_light_dir"),
                  light.light_dir[0], light.light_dir[1], light.light_dir[2]);
    gl_.Uniform1i(gl_.GetUniformLocation(program_, "u_use_pbr"), light.use_pbr);
    gl_.Uniform3f(gl_.GetUniformLocation(program_, "u_ambient_color"),
                  light.ambient_color[0], light.ambient_color[1], light.ambient_color[2]);
    gl_.Uniform3f(gl_.GetUniformLocation(program_, "u_point_pos"),
                  light.point_pos[0], light.point_pos[1], light.point_pos[2]);
    gl_.Uniform3f(gl_.GetUniformLocation(program_, "u_point_color"),
                  light.point_color[0], light.point_color[1], light.point_color[2]);
    gl_.Uniform1f(gl_.GetUniformLocation(program_, "u_point_intensity"), light.point_intensity);
    gl_.Uniform3f(gl_.GetUniformLocation(program_, "u_dir_color"),
                  light.dir_color[0], light.dir_color[1], light.dir_color[2]);
    gl_.Uniform1f(gl_.GetUniformLocation(program_, "u_dir_intensity"), light.dir_intensity);

    const bool use_tile_lights = tile_lights != nullptr && !tile_lights->empty();
    gl_.Uniform1i(gl_.GetUniformLocation(program_, "u_use_tile_lights"), use_tile_lights ? 1 : 0);
    if (use_tile_lights) {
        std::vector<LightRecordGPU> gpu_lights(tile_lights->size());
        for (size_t i = 0; i < tile_lights->size(); ++i) {
            gpu_lights[i] = light_record_from_params((*tile_lights)[i]);
        }
        gl_.BindBuffer(GL_SHADER_STORAGE_BUFFER, light_ssbo_);
        gl_.BufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<GLsizeiptr>(gpu_lights.size() * sizeof(LightRecordGPU)),
                       gpu_lights.data(), GL_DYNAMIC_DRAW);
        gl_.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, light_ssbo_);
    }
    gl_.Uniform1i(gl_.GetUniformLocation(program_, "u_output_normal"), spec.normal ? 1 : 0);
    gl_.Uniform1i(gl_.GetUniformLocation(program_, "u_output_coord"), spec.object_coord ? 1 : 0);
    gl_.Uniform1i(gl_.GetUniformLocation(program_, "u_output_instance_id"),
                  spec.output_instance_id ? 1 : 0);
    gl_.ActiveTexture(0x84C0);
    gl_.BindTexture(GL_TEXTURE_2D_ARRAY, assets_.tex_array());
    gl_.Uniform1i(gl_.GetUniformLocation(program_, "u_tex_array"), 0);
    gl_.Uniform1i(gl_.GetUniformLocation(program_, "u_use_texture"), assets_.has_textures() ? 1 : 0);

    float proj_q = 0.f, proj_qn = 0.f;
    projection_depth_coeffs(cam, proj_q, proj_qn);
    gl_.Uniform1i(gl_.GetUniformLocation(program_, "u_use_distortion"), cam.use_distortion ? 1 : 0);
    gl_.Uniform1i(gl_.GetUniformLocation(program_, "u_has_clip_affine"), 0);
    gl_.Uniform4f(gl_.GetUniformLocation(program_, "u_K"),
                  cam.fx, cam.fy, cam.cx, cam.cy);
    gl_.Uniform4f(gl_.GetUniformLocation(program_, "u_dist_k"),
                  cam.dist[0], cam.dist[1], cam.dist[2], cam.dist[3]);
    gl_.Uniform1f(gl_.GetUniformLocation(program_, "u_dist_k3"), cam.dist[4]);
    GLfloat vp_w = static_cast<GLfloat>(fb_w);
    GLfloat vp_h = static_cast<GLfloat>(fb_h);
    if (cam.use_distortion && spec.mode == RenderMode::Tile &&
        spec.projection == ProjectionMode::ClipAffine) {
        vp_w = static_cast<GLfloat>(cam.width);
        vp_h = static_cast<GLfloat>(cam.height);
    } else if (spec.layered_tiles && spec.mode == RenderMode::Tile) {
        vp_w = static_cast<GLfloat>(spec.tile_w);
        vp_h = static_cast<GLfloat>(spec.tile_h);
    } else if (indexed_ok && spec.mode == RenderMode::Tile) {
        vp_w = static_cast<GLfloat>(spec.tile_w);
        vp_h = static_cast<GLfloat>(spec.tile_h);
    } else if (cam.use_distortion && spec.mode == RenderMode::Tile && !spec.viewport_per_tile) {
        vp_w = static_cast<GLfloat>(cam.width);
        vp_h = static_cast<GLfloat>(cam.height);
    }
    gl_.Uniform2f(gl_.GetUniformLocation(program_, "u_viewport"), vp_w, vp_h);
    gl_.Uniform2f(gl_.GetUniformLocation(program_, "u_proj_z"), proj_q, proj_qn);

    bool has_less = false, has_greater = false;
    for (const auto& inst : instances) {
        if (inst.depth_func == kDepthFuncGreater) has_greater = true;
        else has_less = true;
    }

    if (has_less) {
        std::vector<RenderInstance> subset;
        for (const auto& inst : instances) {
            if (inst.depth_func != kDepthFuncGreater) subset.push_back(inst);
        }
        if (!subset.empty()) {
            upload_instances(subset, cam, spec);
            draw_batch(subset, kDepthFuncLess, cam, spec);
        }
    }
    if (has_greater) {
        std::vector<RenderInstance> subset;
        for (const auto& inst : instances) {
            if (inst.depth_func == kDepthFuncGreater) subset.push_back(inst);
        }
        if (!subset.empty()) {
            upload_instances(subset, cam, spec);
            draw_batch(subset, kDepthFuncGreater, cam, spec);
        }
    }

    if (want_sample_shading) {
        gl_.Disable(GL_SAMPLE_SHADING);
    }
    if (fbo_msaa_samples_ >= 2 && fbo_ms_) {
        resolve_msaa(spec);
    }
    cuda_instance_active_ = false;
}

void BatchRenderer::readback_depth_f32(std::vector<float>& out) const {
    out.resize(static_cast<size_t>(atlas_w_) * atlas_h_);
    gl_.BindFramebuffer(GL_FRAMEBUFFER, fbo_);
    if (gl_.ReadBuffer) gl_.ReadBuffer(GL_COLOR_ATTACHMENT0);
    gl_.ReadPixels(0, 0, static_cast<GLsizei>(atlas_w_), static_cast<GLsizei>(atlas_h_),
                   GL_RED, GL_FLOAT, out.data());
}

void BatchRenderer::readback_rgb_rgba_f32(std::vector<float>& out) const {
    out.resize(static_cast<size_t>(atlas_w_) * atlas_h_ * 4);
    gl_.BindFramebuffer(GL_FRAMEBUFFER, fbo_);
    if (gl_.ReadBuffer) gl_.ReadBuffer(GL_COLOR_ATTACHMENT1);
    gl_.ReadPixels(0, 0, static_cast<GLsizei>(atlas_w_), static_cast<GLsizei>(atlas_h_),
                   GL_RGBA, GL_FLOAT, out.data());
}

void BatchRenderer::readback_rgb(std::vector<uint8_t>& out) const {
    if (!std::getenv("WAPR_OGL_ALLOW_READBACK")) {
        throw std::runtime_error("readback disabled; set WAPR_OGL_ALLOW_READBACK=1");
    }
    std::vector<float> rgba(static_cast<size_t>(atlas_w_) * atlas_h_ * 4);
    gl_.BindFramebuffer(GL_FRAMEBUFFER, fbo_);
    if (gl_.ReadBuffer) gl_.ReadBuffer(GL_COLOR_ATTACHMENT1);
    gl_.ReadPixels(0, 0, static_cast<GLsizei>(atlas_w_), static_cast<GLsizei>(atlas_h_),
                   GL_RGBA, GL_FLOAT, rgba.data());
    out.resize(rgba.size());
    for (size_t i = 0; i < rgba.size(); ++i) {
        out[i] = static_cast<uint8_t>(std::max(0.f, std::min(1.f, rgba[i])) * 255.f);
    }
}

void BatchRenderer::readback_depth(std::vector<float>& out) const {
    if (!std::getenv("WAPR_OGL_ALLOW_READBACK")) {
        throw std::runtime_error("readback disabled; set WAPR_OGL_ALLOW_READBACK=1");
    }
    out.resize(static_cast<size_t>(atlas_w_) * atlas_h_);
    gl_.BindFramebuffer(GL_FRAMEBUFFER, fbo_);
    if (gl_.ReadBuffer) gl_.ReadBuffer(GL_COLOR_ATTACHMENT0);
    gl_.ReadPixels(0, 0, static_cast<GLsizei>(atlas_w_), static_cast<GLsizei>(atlas_h_),
                   GL_RED, GL_FLOAT, out.data());
}

}
