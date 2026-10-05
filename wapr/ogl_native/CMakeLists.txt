# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

cmake_minimum_required(VERSION 3.18)
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
    cpp/egl_context.cpp
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

pybind11_add_module(_gpu_render ${WAPR_OGL_SOURCES})

target_include_directories(_gpu_render PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

target_compile_definitions(_gpu_render PRIVATE
    WAPR_OGL_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

find_library(EGL_LIBRARY NAMES EGL PATHS /usr/lib/x86_64-linux-gnu)
if(EGL_LIBRARY)
    target_link_libraries(_gpu_render PRIVATE ${EGL_LIBRARY})
endif()

target_link_libraries(_gpu_render PRIVATE
    CUDA::cudart
    OpenGL::GL
    ${CMAKE_DL_LIBS}
)
if(TARGET EGL::EGL)
    target_link_libraries(_gpu_render PRIVATE EGL::EGL)
endif()

set_target_properties(_gpu_render PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
    INTERPROCEDURAL_OPTIMIZATION FALSE
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_gpu_render PRIVATE -fno-lto)
    target_link_options(_gpu_render PRIVATE "-Wl,--no-gc-sections" "-fno-lto")
endif()
