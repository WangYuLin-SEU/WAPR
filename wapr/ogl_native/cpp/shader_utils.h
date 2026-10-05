// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

#pragma once

#include "gl_loader.h"
#include <GL/gl.h>
#include <string>

namespace wapr_ogl {

struct GLFunctions;

GLuint compile_shader_program(const GLFunctions& gl,
                              const std::string& vert_src,
                              const std::string& frag_src);

std::string load_shader_file(const std::string& path);
std::string default_shader_dir();

}
