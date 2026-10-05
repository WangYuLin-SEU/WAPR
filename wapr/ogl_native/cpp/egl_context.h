// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#pragma once

#include "gl_loader.h"
#include <EGL/egl.h>
#include <string>

namespace wapr_ogl {

class EglContext {
public:
    EglContext(int gpu_id = 0);
    ~EglContext();

    EglContext(const EglContext&) = delete;
    EglContext& operator=(const EglContext&) = delete;

    bool valid() const { return valid_; }
    EGLDisplay display() const { return display_; }
    EGLContext context() const { return context_; }
    const GLFunctions& gl() const { return gl_; }
    GLFunctions& gl() { return gl_; }

    void make_current();
    void release_current();
    std::string gl_version_string() const;

private:
    bool valid_ = false;
    EGLDisplay display_ = EGL_NO_DISPLAY;
    EGLContext context_ = EGL_NO_CONTEXT;
    EGLSurface surface_ = EGL_NO_SURFACE;
    GLFunctions gl_{};
};

}
