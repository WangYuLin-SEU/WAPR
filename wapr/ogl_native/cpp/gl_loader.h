// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#pragma once

#include <EGL/egl.h>
#include <GL/gl.h>
#include <cstdint>

#ifndef GL_SYNC_FLUSH_COMMANDS_BIT
struct __GLsync;
typedef struct __GLsync* GLsync;
#endif
#ifndef GLuint64
using GLuint64 = uint64_t;
#endif

namespace wapr_ogl {

using PFNGLGENVERTEXARRAYSPROC = void (*)(GLsizei, GLuint*);
using PFNGLBINDVERTEXARRAYPROC = void (*)(GLuint);
using PFNGLDELETEVERTEXARRAYSPROC = void (*)(GLsizei, const GLuint*);
using PFNGLGENBUFFERSPROC = void (*)(GLsizei, GLuint*);
using PFNGLBINDBUFFERPROC = void (*)(GLenum, GLuint);
using PFNGLBUFFERDATAPROC = void (*)(GLenum, GLsizeiptr, const void*, GLenum);
using PFNGLDELETEBUFFERSPROC = void (*)(GLsizei, const GLuint*);
using PFNGLCREATESHADERPROC = GLuint (*)(GLenum);
using PFNGLSHADERSOURCEPROC = void (*)(GLuint, GLsizei, const GLchar* const*, const GLint*);
using PFNGLCOMPILESHADERPROC = void (*)(GLuint);
using PFNGLGETSHADERIVPROC = void (*)(GLuint, GLenum, GLint*);
using PFNGLGETSHADERINFOLOGPROC = void (*)(GLuint, GLsizei, GLsizei*, GLchar*);
using PFNGLCREATEPROGRAMPROC = GLuint (*)(void);
using PFNGLATTACHSHADERPROC = void (*)(GLuint, GLuint);
using PFNGLLINKPROGRAMPROC = void (*)(GLuint);
using PFNGLGETPROGRAMIVPROC = void (*)(GLuint, GLenum, GLint*);
using PFNGLGETPROGRAMINFOLOGPROC = void (*)(GLuint, GLsizei, GLsizei*, GLchar*);
using PFNGLUSEPROGRAMPROC = void (*)(GLuint);
using PFNGLDELETESHADERPROC = void (*)(GLuint);
using PFNGLDELETEPROGRAMPROC = void (*)(GLuint);
using PFNGLGETUNIFORMLOCATIONPROC = GLint (*)(GLuint, const GLchar*);
using PFNGLUNIFORM1IPROC = void (*)(GLint, GLint);
using PFNGLUNIFORM1FPROC = void (*)(GLint, GLfloat);
using PFNGLUNIFORM2FPROC = void (*)(GLint, GLfloat, GLfloat);
using PFNGLUNIFORM3FPROC = void (*)(GLint, GLfloat, GLfloat, GLfloat);
using PFNGLUNIFORM4FPROC = void (*)(GLint, GLfloat, GLfloat, GLfloat, GLfloat);
using PFNGLUNIFORMMATRIX4FVPROC = void (*)(GLint, GLsizei, GLboolean, const GLfloat*);
using PFNGLENABLEVERTEXATTRIBARRAYPROC = void (*)(GLuint);
using PFNGLVERTEXATTRIBPOINTERPROC = void (*)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*);
using PFNGLDISABLEVERTEXATTRIBARRAYPROC = void (*)(GLuint);
using PFNGLGENTEXTURESPROC = void (*)(GLsizei, GLuint*);
using PFNGLBINDTEXTUREPROC = void (*)(GLenum, GLuint);
using PFNGLTEXIMAGE2DPROC = void (*)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*);
using PFNGLTEXIMAGE3DPROC = void (*)(GLenum, GLint, GLint, GLsizei, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*);
using PFNGLTEXPARAMETERIPROC = void (*)(GLenum, GLenum, GLint);
using PFNGLGENERATEMIPMAPPROC = void (*)(GLenum);
using PFNGLDELETETEXTURESPROC = void (*)(GLsizei, const GLuint*);
using PFNGLGENFRAMEBUFFERSPROC = void (*)(GLsizei, GLuint*);
using PFNGLBINDFRAMEBUFFERPROC = void (*)(GLenum, GLuint);
using PFNGLFRAMEBUFFERTEXTURE2DPROC = void (*)(GLenum, GLenum, GLenum, GLuint, GLint);
using PFNGLFRAMEBUFFERTEXTUREPROC = void (*)(GLenum, GLenum, GLuint, GLint);
using PFNGLCHECKFRAMEBUFFERSTATUSPROC = GLenum (*)(GLenum);
using PFNGLDELETEFRAMEBUFFERSPROC = void (*)(GLsizei, const GLuint*);
using PFNGLVIEWPORTPROC = void (*)(GLint, GLint, GLsizei, GLsizei);
using PFNGLCLEARPROC = void (*)(GLbitfield);
using PFNGLCLEARCOLORPROC = void (*)(GLfloat, GLfloat, GLfloat, GLfloat);
using PFNGLCLEARDEPTHFPROC = void (*)(GLfloat);
using PFNGLDEPTHFUNCPROC = void (*)(GLenum);
using PFNGLENABLEPROC = void (*)(GLenum);
using PFNGLDISABLEPROC = void (*)(GLenum);
using PFNGLDRAWELEMENTSPROC = void (*)(GLenum, GLsizei, GLenum, const void*);
using PFNGLACTIVETEXTUREPROC = void (*)(GLenum);
using PFNGLREADPIXELSPROC = void (*)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*);
using PFNGLDRAWBUFFERSPROC = void (*)(GLsizei, const GLenum*);
using PFNGLBINDBUFFERBASEPROC = void (*)(GLenum, GLuint, GLuint);
using PFNGLBINDBUFFERRANGEPROC = void (*)(GLenum, GLuint, GLuint, GLintptr, GLsizeiptr);
using PFNGLDRAWARRAYSPROC = void (*)(GLenum, GLint, GLsizei);
using PFNGLREADBUFFERPROC = void (*)(GLenum);
using PFNGLGETSTRINGPROC = const GLubyte* (*)(GLenum);
using PFNGLFINISHPROC = void (*)();
using PFNGLFLUSHPROC = void (*)();
using PFNGLMULTIDRAWELEMENTSINDIRECTPROC = void (*)(GLenum, GLenum, const void*, GLsizei, GLsizei);
using PFNGLDRAWELEMENTSINDIRECTPROC = void (*)(GLenum, GLenum, const void*);
using PFNGLVIEWPORTARRAYVPROC = void (*)(GLuint, GLsizei, const GLfloat*);
using PFNGLSCISSORARRAYVPROC = void (*)(GLuint, GLsizei, const GLint*);
using PFNGLBUFFERSTORAGEPROC = void (*)(GLenum, GLsizeiptr, const void*, GLbitfield);
using PFNGLMAPBUFFERRANGEPROC = void* (*)(GLenum, GLintptr, GLsizeiptr, GLbitfield);
using PFNGLUNMAPBUFFERPROC = GLboolean (*)(GLenum);
using PFNGLBUFFERSUBDATAPROC = void (*)(GLenum, GLintptr, GLsizeiptr, const void*);
using PFNGLFENCESYNCPROC = GLsync (*)(GLenum, GLbitfield);
using PFNGLCLIENTWAITSYNCPROC = GLenum (*)(GLsync, GLbitfield, GLuint64);
using PFNGLDELETESYNCPROC = void (*)(GLsync);
using PFNGLGETINTEGERVPROC = void (*)(GLenum, GLint*);
using PFNGLGETSTRINGIPROC = const GLubyte* (*)(GLenum, GLuint);
using PFNGLFLUSHMAPPEDBUFFERRANGEPROC = void (*)(GLenum, GLintptr, GLsizeiptr);

struct GLFunctions {
    PFNGLGENVERTEXARRAYSPROC GenVertexArrays = nullptr;
    PFNGLBINDVERTEXARRAYPROC BindVertexArray = nullptr;
    PFNGLDELETEVERTEXARRAYSPROC DeleteVertexArrays = nullptr;
    PFNGLGENBUFFERSPROC GenBuffers = nullptr;
    PFNGLBINDBUFFERPROC BindBuffer = nullptr;
    PFNGLBUFFERDATAPROC BufferData = nullptr;
    PFNGLBUFFERSTORAGEPROC BufferStorage = nullptr;
    PFNGLMAPBUFFERRANGEPROC MapBufferRange = nullptr;
    PFNGLUNMAPBUFFERPROC UnmapBuffer = nullptr;
    PFNGLBUFFERSUBDATAPROC BufferSubData = nullptr;
    PFNGLFLUSHMAPPEDBUFFERRANGEPROC FlushMappedBufferRange = nullptr;
    PFNGLDELETEBUFFERSPROC DeleteBuffers = nullptr;
    PFNGLCREATESHADERPROC CreateShader = nullptr;
    PFNGLSHADERSOURCEPROC ShaderSource = nullptr;
    PFNGLCOMPILESHADERPROC CompileShader = nullptr;
    PFNGLGETSHADERIVPROC GetShaderiv = nullptr;
    PFNGLGETSHADERINFOLOGPROC GetShaderInfoLog = nullptr;
    PFNGLCREATEPROGRAMPROC CreateProgram = nullptr;
    PFNGLATTACHSHADERPROC AttachShader = nullptr;
    PFNGLLINKPROGRAMPROC LinkProgram = nullptr;
    PFNGLGETPROGRAMIVPROC GetProgramiv = nullptr;
    PFNGLGETPROGRAMINFOLOGPROC GetProgramInfoLog = nullptr;
    PFNGLUSEPROGRAMPROC UseProgram = nullptr;
    PFNGLDELETESHADERPROC DeleteShader = nullptr;
    PFNGLDELETEPROGRAMPROC DeleteProgram = nullptr;
    PFNGLGETUNIFORMLOCATIONPROC GetUniformLocation = nullptr;
    PFNGLUNIFORM1IPROC Uniform1i = nullptr;
    PFNGLUNIFORM1FPROC Uniform1f = nullptr;
    PFNGLUNIFORM2FPROC Uniform2f = nullptr;
    PFNGLUNIFORM3FPROC Uniform3f = nullptr;
    PFNGLUNIFORM4FPROC Uniform4f = nullptr;
    PFNGLUNIFORMMATRIX4FVPROC UniformMatrix4fv = nullptr;
    PFNGLENABLEVERTEXATTRIBARRAYPROC EnableVertexAttribArray = nullptr;
    PFNGLVERTEXATTRIBPOINTERPROC VertexAttribPointer = nullptr;
    PFNGLDISABLEVERTEXATTRIBARRAYPROC DisableVertexAttribArray = nullptr;
    PFNGLGENTEXTURESPROC GenTextures = nullptr;
    PFNGLBINDTEXTUREPROC BindTexture = nullptr;
    PFNGLTEXIMAGE2DPROC TexImage2D = nullptr;
    PFNGLTEXIMAGE3DPROC TexImage3D = nullptr;
    PFNGLTEXPARAMETERIPROC TexParameteri = nullptr;
    PFNGLGENERATEMIPMAPPROC GenerateMipmap = nullptr;
    PFNGLDELETETEXTURESPROC DeleteTextures = nullptr;
    PFNGLGENFRAMEBUFFERSPROC GenFramebuffers = nullptr;
    PFNGLBINDFRAMEBUFFERPROC BindFramebuffer = nullptr;
    PFNGLFRAMEBUFFERTEXTURE2DPROC FramebufferTexture2D = nullptr;
    PFNGLFRAMEBUFFERTEXTUREPROC FramebufferTexture = nullptr;
    PFNGLCHECKFRAMEBUFFERSTATUSPROC CheckFramebufferStatus = nullptr;
    PFNGLDELETEFRAMEBUFFERSPROC DeleteFramebuffers = nullptr;
    PFNGLVIEWPORTPROC Viewport = nullptr;
    PFNGLCLEARPROC Clear = nullptr;
    PFNGLCLEARCOLORPROC ClearColor = nullptr;
    PFNGLCLEARDEPTHFPROC ClearDepthf = nullptr;
    PFNGLDEPTHFUNCPROC DepthFunc = nullptr;
    PFNGLENABLEPROC Enable = nullptr;
    PFNGLDISABLEPROC Disable = nullptr;
    PFNGLDRAWELEMENTSPROC DrawElements = nullptr;
    PFNGLDRAWARRAYSPROC DrawArrays = nullptr;
    PFNGLACTIVETEXTUREPROC ActiveTexture = nullptr;
    PFNGLREADPIXELSPROC ReadPixels = nullptr;
    PFNGLREADBUFFERPROC ReadBuffer = nullptr;
    PFNGLDRAWBUFFERSPROC DrawBuffers = nullptr;
    PFNGLBINDBUFFERBASEPROC BindBufferBase = nullptr;
    PFNGLBINDBUFFERRANGEPROC BindBufferRange = nullptr;
    PFNGLMULTIDRAWELEMENTSINDIRECTPROC MultiDrawElementsIndirect = nullptr;
    PFNGLDRAWELEMENTSINDIRECTPROC DrawElementsIndirect = nullptr;
    PFNGLVIEWPORTARRAYVPROC ViewportArrayv = nullptr;
    PFNGLSCISSORARRAYVPROC ScissorArrayv = nullptr;
    PFNGLFENCESYNCPROC FenceSync = nullptr;
    PFNGLCLIENTWAITSYNCPROC ClientWaitSync = nullptr;
    PFNGLDELETESYNCPROC DeleteSync = nullptr;
    PFNGLGETINTEGERVPROC GetIntegerv = nullptr;
    PFNGLGETSTRINGIPROC GetStringi = nullptr;
    PFNGLGETSTRINGPROC GetString = nullptr;
    PFNGLFINISHPROC Finish = nullptr;
    PFNGLFLUSHPROC Flush = nullptr;
};

bool load_gl_functions(GLFunctions& gl, EGLDisplay display);

}
