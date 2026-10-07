// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#include "egl_context.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#ifndef WGL_CONTEXT_MAJOR_VERSION_ARB
#define WGL_CONTEXT_MAJOR_VERSION_ARB 0x2091
#endif
#ifndef WGL_CONTEXT_MINOR_VERSION_ARB
#define WGL_CONTEXT_MINOR_VERSION_ARB 0x2092
#endif
#ifndef WGL_CONTEXT_PROFILE_MASK_ARB
#define WGL_CONTEXT_PROFILE_MASK_ARB 0x9126
#endif
#ifndef WGL_CONTEXT_CORE_PROFILE_BIT_ARB
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001
#endif

// Ask hybrid laptops to create this process's OpenGL context on the NVIDIA GPU.
// 混合显卡笔记本上，让本进程的 OpenGL 上下文建在 NVIDIA GPU 上。
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

extern "C" void* eglGetProcAddress(const char* name) {
    using Proc = PROC(WINAPI*)(LPCSTR);
    static auto wgl_get = reinterpret_cast<Proc>(
        GetProcAddress(GetModuleHandleA("opengl32.dll"), "wglGetProcAddress"));
    void* fn = nullptr;
    if (wgl_get != nullptr) {
        fn = reinterpret_cast<void*>(wgl_get(name));
    }
    if (fn == nullptr || fn == reinterpret_cast<void*>(1) || fn == reinterpret_cast<void*>(2) ||
        fn == reinterpret_cast<void*>(3) || fn == reinterpret_cast<void*>(-1)) {
        fn = reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("opengl32.dll"), name));
    }
    return fn;
}

namespace wapr_ogl {
namespace {

void check_win(bool ok, const char* msg) {
    if (!ok) {
        throw std::runtime_error(std::string("WGL error: ") + msg);
    }
}

HWND create_hidden_window() {
    HINSTANCE instance = GetModuleHandleW(nullptr);
    const wchar_t* class_name = L"WaprOglHiddenWindow";
    WNDCLASSW existing = {};
    if (!GetClassInfoW(instance, class_name, &existing)) {
        WNDCLASSW window_class = {};
        window_class.style = CS_OWNDC;
        window_class.lpfnWndProc = DefWindowProcW;
        window_class.hInstance = instance;
        window_class.lpszClassName = class_name;
        if (RegisterClassW(&window_class) == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
            throw std::runtime_error("WGL error: RegisterClassW");
        }
    }
    HWND window = CreateWindowExW(0, class_name, L"wapr-ogl", WS_OVERLAPPEDWINDOW,
                                   0, 0, 64, 64, nullptr, nullptr, instance, nullptr);
    check_win(window != nullptr, "CreateWindowExW");
    return window;
}

}  // namespace

EglContext::EglContext(int gpu_id) {
    cudaSetDevice(gpu_id);
    surface_ = create_hidden_window();
    display_ = GetDC(static_cast<HWND>(surface_));
    check_win(display_ != nullptr, "GetDC");

    PIXELFORMATDESCRIPTOR format = {};
    format.nSize = sizeof(format);
    format.nVersion = 1;
    format.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    format.iPixelType = PFD_TYPE_RGBA;
    format.cColorBits = 32;
    format.cDepthBits = 24;
    format.cStencilBits = 8;
    int format_index = ChoosePixelFormat(static_cast<HDC>(display_), &format);
    check_win(format_index != 0, "ChoosePixelFormat");
    check_win(SetPixelFormat(static_cast<HDC>(display_), format_index, &format) == TRUE, "SetPixelFormat");

    HGLRC bootstrap = wglCreateContext(static_cast<HDC>(display_));
    check_win(bootstrap != nullptr, "wglCreateContext");
    check_win(wglMakeCurrent(static_cast<HDC>(display_), bootstrap) == TRUE, "wglMakeCurrent");

    using CreateAttribs = HGLRC(WINAPI*)(HDC, HGLRC, const int*);
    auto create_attribs = reinterpret_cast<CreateAttribs>(wglGetProcAddress("wglCreateContextAttribsARB"));
    check_win(create_attribs != nullptr, "wglCreateContextAttribsARB");
    const int attribs[] = {
        WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
        WGL_CONTEXT_MINOR_VERSION_ARB, 3,
        WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
        0};
    HGLRC core = create_attribs(static_cast<HDC>(display_), nullptr, attribs);
    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(bootstrap);
    check_win(core != nullptr, "wglCreateContextAttribsARB");
    context_ = core;
    make_current();

    if (!load_gl_functions(gl_, display_)) {
        throw std::runtime_error("Failed to load OpenGL 4.3 functions");
    }
    valid_ = true;
}

EglContext::~EglContext() {
    if (context_ != nullptr) {
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(static_cast<HGLRC>(context_));
        context_ = nullptr;
    }
    if (display_ != nullptr && surface_ != nullptr) {
        ReleaseDC(static_cast<HWND>(surface_), static_cast<HDC>(display_));
        display_ = nullptr;
    }
    if (surface_ != nullptr) {
        DestroyWindow(static_cast<HWND>(surface_));
        surface_ = nullptr;
    }
}

void EglContext::make_current() {
    check_win(wglMakeCurrent(static_cast<HDC>(display_), static_cast<HGLRC>(context_)) == TRUE,
              "wglMakeCurrent");
}

void EglContext::release_current() {
    wglMakeCurrent(nullptr, nullptr);
}

std::string EglContext::gl_version_string() const {
    if (gl_.GetString) {
        const char* version = reinterpret_cast<const char*>(gl_.GetString(0x1F02));
        if (version) return version;
    }
    return "unknown";
}

}  // namespace wapr_ogl
