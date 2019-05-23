#/***************************************************************************
# *
# *  @license
# *  Copyright (C) Codeplay Software Limited
# *  Licensed under the Apache License, Version 2.0 (the "License");
# *  you may not use this file except in compliance with the License.
# *  You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# *  For your convenience, a copy of the License has been included in this
# *  repository.
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# *
# *  SYCL-BLAS: BLAS implementation using SYCL
# *
# *  @filename CMakeLists.txt
# *
# **************************************************************************/

cmake_minimum_required(VERSION 3.4.3)
project(SYCL-BLAS LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_BENCHMARKS "Build the benchmarks" OFF)
option(BUILD_DOCUMENTATION "Build the documentation" OFF)
option(SYCL_BLAS_ALWAYS_INLINE "Setting always inline attribute for all kernels" ON)
option(DOUBLE_SUPPORT "Disable double support when testing." OFF)
option(NAIVE_GEMM "Default to naive GEMM implementations" OFF)
set(TARGET "DEFAULT_CPU" CACHE STRING "Defines the platform for which the sycl library is built")
set_property(CACHE TARGET PROPERTY STRINGS
    DEFAULT_CPU INTEL_GPU AMD_GPU ARM_GPU RCAR ARM_GPU)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include("${CMAKE_BINARY_DIR}/conan_paths.cmake" OPTIONAL)
include(CTest)

add_subdirectory(external)
find_package(ComputeCpp REQUIRED)

list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS
    -O3
    -fsycl-split-modules=20
    -mllvm -inline-threshold=10000
    -no-serial-memop
    -Xclang -cl-mad-enable
)

add_library(sycl-blas-header-only INTERFACE)
add_library(codeplay::sycl-blas-header-only ALIAS sycl-blas-header-only)
target_include_directories(sycl-blas-header-only INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS sycl-blas-header-only
    EXPORT SYCLBlasTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

add_subdirectory(src bin)
add_subdirectory(tools)

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()
if(BUILD_TESTING)
    add_subdirectory(test)
endif()

export(EXPORT SYCLBlasTargets
  FILE "${PROJECT_BINARY_DIR}/sycl-blas-targets.cmake"
  NAMESPACE codeplay::)
configure_file(cmake/sycl-blas-config.cmake
    "${PROJECT_BINARY_DIR}/sycl-blas-config.cmake"
    COPYONLY)

set(config_package_location lib/cmake/SYCL-BLAS)
install(DIRECTORY include/
	    DESTINATION include/sycl-blas
		COMPONENT Devel)
install(EXPORT SYCLBlasTargets
  FILE sycl-blas-targets.cmake
  NAMESPACE codeplay::
  DESTINATION ${config_package_location})
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/sycl-blas-config.cmake
  DESTINATION ${config_package_location}
  COMPONENT Devel)
