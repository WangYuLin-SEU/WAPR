// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#include "egl_context.h"
#include <EGL/eglext.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef EGL_PLATFORM_DEVICE_EXT
#define EGL_PLATFORM_DEVICE_EXT 0x313F
#endif
#ifndef EGL_CUDA_DEVICE_NV
#define EGL_CUDA_DEVICE_NV 0x323A
#endif

namespace wapr_ogl {

static void check_egl(EGLBoolean ok, const char* msg) {
    if (!ok) {
        throw std::runtime_error(std::string("EGL error: ") + msg);
    }
}

static void ensure_nvidia_egl_vendor() {
    const char* vendor = std::getenv("__EGL_VENDOR_LIBRARY_FILENAMES");
    if (!vendor) {
        setenv("__EGL_VENDOR_LIBRARY_FILENAMES",
               "/usr/share/glvnd/egl_vendor.d/10_nvidia.json", 0);
    }
}


static int find_egl_index_for_cuda(int cuda_id, EGLDeviceEXT* devices, EGLint count,
                                   PFNEGLQUERYDEVICEATTRIBEXTPROC query_attr) {
    if (!query_attr || count <= 0) {
        return -1;
    }
    for (EGLint i = 0; i < count; ++i) {
        EGLAttrib mapped = -1;
        if (query_attr(devices[i], EGL_CUDA_DEVICE_NV, &mapped) == EGL_TRUE &&
            static_cast<int>(mapped) == cuda_id) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

static EGLDisplay create_nvidia_display(int cuda_device_id) {
    ensure_nvidia_egl_vendor();

    EGLDisplay boot = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (boot != EGL_NO_DISPLAY) {
        EGLint major = 0, minor = 0;
        eglInitialize(boot, &major, &minor);
    }

    auto eglQueryDevicesEXT = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(
        eglGetProcAddress("eglQueryDevicesEXT"));
    auto eglQueryDeviceAttribEXT = reinterpret_cast<PFNEGLQUERYDEVICEATTRIBEXTPROC>(
        eglGetProcAddress("eglQueryDeviceAttribEXT"));
    auto eglGetPlatformDisplayEXT = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
        eglGetProcAddress("eglGetPlatformDisplayEXT"));

    if (eglQueryDevicesEXT && eglGetPlatformDisplayEXT) {
        EGLDeviceEXT devices[32];
        EGLint count = 0;
        if (eglQueryDevicesEXT(32, devices, &count) && count > 0) {
            int idx = find_egl_index_for_cuda(cuda_device_id, devices, count, eglQueryDeviceAttribEXT);
            if (idx < 0) {
                idx = cuda_device_id;
                if (idx < 0) {
                    idx = 0;
                }
                if (idx >= count) {
                    idx = count - 1;
                }
            }
            EGLDisplay dpy =
                eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices[idx], nullptr);
            if (dpy != EGL_NO_DISPLAY) {
                return dpy;
            }
        }
    }
    if (boot != EGL_NO_DISPLAY) {
        return boot;
    }
    return eglGetDisplay(EGL_DEFAULT_DISPLAY);
}

EglContext::EglContext(int gpu_id) {
    display_ = create_nvidia_display(gpu_id);
    check_egl(display_ != EGL_NO_DISPLAY, "eglGetDisplay");

    EGLint major = 0, minor = 0;
    check_egl(eglInitialize(display_, &major, &minor), "eglInitialize");

    const EGLint cfg_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_STENCIL_SIZE, 8,
        EGL_NONE};

    EGLConfig config = nullptr;
    EGLint num = 0;
    check_egl(eglChooseConfig(display_, cfg_attribs, &config, 1, &num) && num > 0,
              "eglChooseConfig");

    const EGLint pbuffer_attribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    surface_ = eglCreatePbufferSurface(display_, config, pbuffer_attribs);
    check_egl(surface_ != EGL_NO_SURFACE, "eglCreatePbufferSurface");

    check_egl(eglBindAPI(EGL_OPENGL_API), "eglBindAPI");

    const EGLint ctx_attribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_NONE};
    context_ = eglCreateContext(display_, config, EGL_NO_CONTEXT, ctx_attribs);
    check_egl(context_ != EGL_NO_CONTEXT, "eglCreateContext");

    make_current();

    if (!load_gl_functions(gl_, display_)) {
        throw std::runtime_error("Failed to load OpenGL 4.3 functions");
    }

    valid_ = true;
}

EglContext::~EglContext() {
    if (display_ != EGL_NO_DISPLAY) {
        eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (context_ != EGL_NO_CONTEXT) {
            eglDestroyContext(display_, context_);
        }
        if (surface_ != EGL_NO_SURFACE) {
            eglDestroySurface(display_, surface_);
        }
        eglTerminate(display_);
    }
}

void EglContext::make_current() {
    check_egl(eglMakeCurrent(display_, surface_, surface_, context_), "eglMakeCurrent");
}

void EglContext::release_current() {
    eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

std::string EglContext::gl_version_string() const {
    if (gl_.GetString) {
        const char* v = reinterpret_cast<const char*>(gl_.GetString(0x1F02));
        if (v) return v;
    }
    return "unknown";
}

}
