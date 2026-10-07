// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#include "shader_utils.h"
#include "gl_loader.h"
#include <GL/gl.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace wapr_ogl {

std::string default_shader_dir() {
    const char* env = std::getenv("WAPR_OGL_SHADER_DIR");
    if (env) return env;
#ifdef WAPR_OGL_SHADER_DIR
    return WAPR_OGL_SHADER_DIR;
#else
    return "wapr/ogl_native/shaders";
#endif
}

std::string load_shader_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("Cannot open shader: " + path);
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

GLuint compile_shader_program(const GLFunctions& gl,
                              const std::string& vert_src,
                              const std::string& frag_src) {
    auto compile = [&](GLenum type, const std::string& src) {
        GLuint sh = gl.CreateShader(type);
        const char* csrc = src.c_str();
        gl.ShaderSource(sh, 1, &csrc, nullptr);
        gl.CompileShader(sh);
        GLint ok = 0;
        gl.GetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[4096];
            gl.GetShaderInfoLog(sh, 4096, nullptr, log);
            throw std::runtime_error(std::string("Shader compile error: ") + log);
        }
        return sh;
    };

    GLuint vs = compile(GL_VERTEX_SHADER, vert_src);
    GLuint fs = compile(GL_FRAGMENT_SHADER, frag_src);
    GLuint prog = gl.CreateProgram();
    gl.AttachShader(prog, vs);
    gl.AttachShader(prog, fs);
    gl.LinkProgram(prog);
    gl.DeleteShader(vs);
    gl.DeleteShader(fs);

    GLint linked = 0;
    gl.GetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[4096];
        gl.GetProgramInfoLog(prog, 4096, nullptr, log);
        throw std::runtime_error(std::string("Program link error: ") + log);
    }
    return prog;
}

}
