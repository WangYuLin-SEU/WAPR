// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#include "gl_loader.h"
#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace wapr_ogl {

template <typename T>
static T load_fn(EGLDisplay, const char* name) {
    // EGL returns a function pointer; WGL's wrapper returns void*.
    // EGL 返回函数指针，WGL 包装返回 void*；显式转换使两种平台都能编译。
    void* raw = reinterpret_cast<void*>(eglGetProcAddress(name));
#ifdef _WIN32
    if (!raw || raw == reinterpret_cast<void*>(1) || raw == reinterpret_cast<void*>(2) ||
        raw == reinterpret_cast<void*>(3) || raw == reinterpret_cast<void*>(-1)) {
        raw = reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("opengl32.dll"), name));
    }
#else
    if (!raw) {
        raw = dlsym(RTLD_DEFAULT, name);
    }
#endif
    return reinterpret_cast<T>(raw);
}

bool load_gl_functions(GLFunctions& gl, EGLDisplay display) {
    gl.GenVertexArrays = load_fn<PFNGLGENVERTEXARRAYSPROC>(display, "glGenVertexArrays");
    gl.BindVertexArray = load_fn<PFNGLBINDVERTEXARRAYPROC>(display, "glBindVertexArray");
    gl.DeleteVertexArrays = load_fn<PFNGLDELETEVERTEXARRAYSPROC>(display, "glDeleteVertexArrays");
    gl.GenBuffers = load_fn<PFNGLGENBUFFERSPROC>(display, "glGenBuffers");
    gl.BindBuffer = load_fn<PFNGLBINDBUFFERPROC>(display, "glBindBuffer");
    gl.BufferData = load_fn<PFNGLBUFFERDATAPROC>(display, "glBufferData");
    gl.BufferStorage = load_fn<PFNGLBUFFERSTORAGEPROC>(display, "glBufferStorage");
    gl.MapBufferRange = load_fn<PFNGLMAPBUFFERRANGEPROC>(display, "glMapBufferRange");
    gl.UnmapBuffer = load_fn<PFNGLUNMAPBUFFERPROC>(display, "glUnmapBuffer");
    gl.BufferSubData = load_fn<PFNGLBUFFERSUBDATAPROC>(display, "glBufferSubData");
    gl.FlushMappedBufferRange =
        load_fn<PFNGLFLUSHMAPPEDBUFFERRANGEPROC>(display, "glFlushMappedBufferRange");
    gl.DeleteBuffers = load_fn<PFNGLDELETEBUFFERSPROC>(display, "glDeleteBuffers");
    gl.CreateShader = load_fn<PFNGLCREATESHADERPROC>(display, "glCreateShader");
    gl.ShaderSource = load_fn<PFNGLSHADERSOURCEPROC>(display, "glShaderSource");
    gl.CompileShader = load_fn<PFNGLCOMPILESHADERPROC>(display, "glCompileShader");
    gl.GetShaderiv = load_fn<PFNGLGETSHADERIVPROC>(display, "glGetShaderiv");
    gl.GetShaderInfoLog = load_fn<PFNGLGETSHADERINFOLOGPROC>(display, "glGetShaderInfoLog");
    gl.CreateProgram = load_fn<PFNGLCREATEPROGRAMPROC>(display, "glCreateProgram");
    gl.AttachShader = load_fn<PFNGLATTACHSHADERPROC>(display, "glAttachShader");
    gl.LinkProgram = load_fn<PFNGLLINKPROGRAMPROC>(display, "glLinkProgram");
    gl.GetProgramiv = load_fn<PFNGLGETPROGRAMIVPROC>(display, "glGetProgramiv");
    gl.GetProgramInfoLog = load_fn<PFNGLGETPROGRAMINFOLOGPROC>(display, "glGetProgramInfoLog");
    gl.UseProgram = load_fn<PFNGLUSEPROGRAMPROC>(display, "glUseProgram");
    gl.DeleteShader = load_fn<PFNGLDELETESHADERPROC>(display, "glDeleteShader");
    gl.DeleteProgram = load_fn<PFNGLDELETEPROGRAMPROC>(display, "glDeleteProgram");
    gl.GetUniformLocation = load_fn<PFNGLGETUNIFORMLOCATIONPROC>(display, "glGetUniformLocation");
    gl.Uniform1i = load_fn<PFNGLUNIFORM1IPROC>(display, "glUniform1i");
    gl.Uniform1f = load_fn<PFNGLUNIFORM1FPROC>(display, "glUniform1f");
    gl.Uniform2f = load_fn<PFNGLUNIFORM2FPROC>(display, "glUniform2f");
    gl.Uniform3f = load_fn<PFNGLUNIFORM3FPROC>(display, "glUniform3f");
    gl.Uniform4f = load_fn<PFNGLUNIFORM4FPROC>(display, "glUniform4f");
    gl.UniformMatrix4fv = load_fn<PFNGLUNIFORMMATRIX4FVPROC>(display, "glUniformMatrix4fv");
    gl.EnableVertexAttribArray = load_fn<PFNGLENABLEVERTEXATTRIBARRAYPROC>(display, "glEnableVertexAttribArray");
    gl.VertexAttribPointer = load_fn<PFNGLVERTEXATTRIBPOINTERPROC>(display, "glVertexAttribPointer");
    gl.DisableVertexAttribArray = load_fn<PFNGLDISABLEVERTEXATTRIBARRAYPROC>(display, "glDisableVertexAttribArray");
    gl.GenTextures = load_fn<PFNGLGENTEXTURESPROC>(display, "glGenTextures");
    gl.BindTexture = load_fn<PFNGLBINDTEXTUREPROC>(display, "glBindTexture");
    gl.TexImage2D = load_fn<PFNGLTEXIMAGE2DPROC>(display, "glTexImage2D");
    gl.TexImage3D = load_fn<PFNGLTEXIMAGE3DPROC>(display, "glTexImage3D");
    gl.TexParameteri = load_fn<PFNGLTEXPARAMETERIPROC>(display, "glTexParameteri");
    gl.GenerateMipmap = load_fn<PFNGLGENERATEMIPMAPPROC>(display, "glGenerateMipmap");
    gl.DeleteTextures = load_fn<PFNGLDELETETEXTURESPROC>(display, "glDeleteTextures");
    gl.GenFramebuffers = load_fn<PFNGLGENFRAMEBUFFERSPROC>(display, "glGenFramebuffers");
    gl.BindFramebuffer = load_fn<PFNGLBINDFRAMEBUFFERPROC>(display, "glBindFramebuffer");
    gl.FramebufferTexture2D = load_fn<PFNGLFRAMEBUFFERTEXTURE2DPROC>(display, "glFramebufferTexture2D");
    gl.FramebufferTexture = load_fn<PFNGLFRAMEBUFFERTEXTUREPROC>(display, "glFramebufferTexture");
    gl.CheckFramebufferStatus = load_fn<PFNGLCHECKFRAMEBUFFERSTATUSPROC>(display, "glCheckFramebufferStatus");
    gl.DeleteFramebuffers = load_fn<PFNGLDELETEFRAMEBUFFERSPROC>(display, "glDeleteFramebuffers");
    gl.Viewport = load_fn<PFNGLVIEWPORTPROC>(display, "glViewport");
    gl.Clear = load_fn<PFNGLCLEARPROC>(display, "glClear");
    gl.ClearColor = load_fn<PFNGLCLEARCOLORPROC>(display, "glClearColor");
    gl.ClearDepthf = load_fn<PFNGLCLEARDEPTHFPROC>(display, "glClearDepthf");
    gl.DepthFunc = load_fn<PFNGLDEPTHFUNCPROC>(display, "glDepthFunc");
    gl.Enable = load_fn<PFNGLENABLEPROC>(display, "glEnable");
    gl.Disable = load_fn<PFNGLDISABLEPROC>(display, "glDisable");
    gl.DrawElements = load_fn<PFNGLDRAWELEMENTSPROC>(display, "glDrawElements");
    gl.DrawArrays = load_fn<PFNGLDRAWARRAYSPROC>(display, "glDrawArrays");
    gl.ActiveTexture = load_fn<PFNGLACTIVETEXTUREPROC>(display, "glActiveTexture");
    gl.ReadPixels = load_fn<PFNGLREADPIXELSPROC>(display, "glReadPixels");
    gl.ReadBuffer = load_fn<PFNGLREADBUFFERPROC>(display, "glReadBuffer");
    gl.DrawBuffers = load_fn<PFNGLDRAWBUFFERSPROC>(display, "glDrawBuffers");
    gl.BindBufferBase = load_fn<PFNGLBINDBUFFERBASEPROC>(display, "glBindBufferBase");
    gl.BindBufferRange = load_fn<PFNGLBINDBUFFERRANGEPROC>(display, "glBindBufferRange");
    gl.MultiDrawElementsIndirect = load_fn<PFNGLMULTIDRAWELEMENTSINDIRECTPROC>(display, "glMultiDrawElementsIndirect");
    gl.DrawElementsIndirect = load_fn<PFNGLDRAWELEMENTSINDIRECTPROC>(display, "glDrawElementsIndirect");
    gl.ViewportArrayv = load_fn<PFNGLVIEWPORTARRAYVPROC>(display, "glViewportArrayv");
    gl.ScissorArrayv = load_fn<PFNGLSCISSORARRAYVPROC>(display, "glScissorArrayv");
    gl.FenceSync = load_fn<PFNGLFENCESYNCPROC>(display, "glFenceSync");
    gl.ClientWaitSync = load_fn<PFNGLCLIENTWAITSYNCPROC>(display, "glClientWaitSync");
    gl.DeleteSync = load_fn<PFNGLDELETESYNCPROC>(display, "glDeleteSync");
    gl.GetIntegerv = load_fn<PFNGLGETINTEGERVPROC>(display, "glGetIntegerv");
    gl.GetStringi = load_fn<PFNGLGETSTRINGIPROC>(display, "glGetStringi");
    gl.GetString = load_fn<PFNGLGETSTRINGPROC>(display, "glGetString");
    gl.Finish = load_fn<PFNGLFINISHPROC>(display, "glFinish");
    gl.Flush = load_fn<PFNGLFLUSHPROC>(display, "glFlush");

    return gl.CreateProgram && gl.MultiDrawElementsIndirect && gl.GenVertexArrays;
}

}
