# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

cmake_minimum_required(VERSION 3.18)
# macOS cannot use the Windows WGL context or the Linux EGL context.
# macOS 不能使用 Windows 的 WGL 上下文，也不能使用 Linux 的 EGL 上下文。
# Stop before the CUDA language is enabled, so this is the message CMake prints.
# 在启用 CUDA 语言之前停下，这样 Mac 上看到的是这句，而不是缺少 CUDA 编译器。
if(APPLE)
    message(FATAL_ERROR "WAPR OGL does not support macOS. Windows uses WGL, Linux uses EGL. / WAPR 的 OGL 不支持 macOS。Windows 用 WGL，Linux 用 EGL。")
endif()
project(wapr_ogl LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})

set(WAPR_OGL_SOURCES
    cpp/projection.cpp
    cpp/gl_loader.cpp
    cpp/shader_utils.cpp
    cpp/asset_manager.cpp
    cpp/batch_renderer.cpp
    cpp/gpu_render_runtime.cpp
    cuda/pack_outputs.cu
    cuda/fill_tile_instances.cu
    cuda/cuda_pack_context.cpp
    cuda/distort_remap.cu
    cuda/cuda_distort_context.cpp
    python/gpu_render_pybind.cpp
)
if(WIN32)
    list(APPEND WAPR_OGL_SOURCES cpp/wgl_context.cpp)
else()
    list(APPEND WAPR_OGL_SOURCES cpp/egl_context.cpp)
endif()

# Keep pybind11 from adding LTO after our compiler flags; some system linkers
# cannot read the Conda compiler's LTO objects and silently lose PyInit symbols.
# 禁止 pybind11 在编译选项之后再追加 LTO；部分系统链接器无法读取镜像编译器的
# LTO 对象，导致生成的库丢失 PyInit 导出符号。
pybind11_add_module(_gpu_render NO_EXTRAS ${WAPR_OGL_SOURCES})

target_include_directories(_gpu_render PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)
if(WIN32)
    # Windows SDK gl.h is OpenGL 1.1. The Khronos glext header supplies the 4.x tokens.
    # Windows SDK 的 gl.h 只有 OpenGL 1.1。Khronos glext 头提供 4.x 常量。
    target_include_directories(_gpu_render PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    # Source comments are UTF-8. MSVC otherwise reads them as the system code page.
    # 源码注释是 UTF-8。不指定时 MSVC 会按系统代码页读取。
    target_compile_options(_gpu_render PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

target_compile_definitions(_gpu_render PRIVATE
    WAPR_OGL_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

if(WIN32)
    target_link_libraries(_gpu_render PRIVATE opengl32 user32 gdi32)
else()
    find_library(EGL_LIBRARY NAMES EGL PATHS /usr/lib/x86_64-linux-gnu)
    if(EGL_LIBRARY)
        target_link_libraries(_gpu_render PRIVATE ${EGL_LIBRARY})
    endif()
    if(TARGET EGL::EGL)
        target_link_libraries(_gpu_render PRIVATE EGL::EGL)
    endif()
endif()

target_link_libraries(_gpu_render PRIVATE
    CUDA::cudart
    OpenGL::GL
    ${CMAKE_DL_LIBS}
)

if(NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

set_target_properties(_gpu_render PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
    INTERPROCEDURAL_OPTIMIZATION FALSE
    # Build output stays in the selected binary/cache directory.
    # 编译输出保留在指定的构建或缓存目录，不写只读源码目录。
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_gpu_render PRIVATE -fno-lto)
    target_link_options(_gpu_render PRIVATE "-Wl,--no-gc-sections" "-fno-lto")
endif()
