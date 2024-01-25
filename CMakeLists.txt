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
# *  portBLAS: BLAS implementation using SYCL
# *
# *  @filename CMakeLists.txt
# *
# **************************************************************************/
cmake_minimum_required(VERSION 3.4.3)
project(portBLAS VERSION 0.1.0 LANGUAGES CXX)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
  include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
  conan_basic_setup(TARGETS NO_OUTPUT_DIRS)
  # We copy the pkgconfig files here
  list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
endif()

option(INSTALL_HEADER_ONLY "Install portBLAS as a header only library" OFF)

set(BUILD_SHARED_LIBS ON CACHE BOOL "")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations  -Wno-deprecated-copy-with-user-provided-copy") 

if(DEFINED SYSTEM_BLAS_ROOT)
  message(DEPRECATION
    "SYSTEM_BLAS_ROOT is deprecated. Add the path to the reference BLAS to CMAKE_PREFIX_PATH instead")
  list(INSERT CMAKE_PREFIX_PATH 0 ${SYSTEM_BLAS_ROOT})
endif()

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
)

# This turns the long file names that are generated into short versions using a hash
# This is required to build portBLAS on Windows and on WSL since Windows doesn't
# have good support for long file names
option(PORTBLAS_USE_SHORT_NAMES "Whether to use short names for generated files" OFF)
if (WIN32)
  # On WIN32 we can force this option to be true, but when building on WSL this
  # must be set in the command line
  set(PORTBLAS_USE_SHORT_NAMES ON CACHE BOOL "Enable short names for generated files in Windows" FORCE)
endif()

# Setup datatypes, workgroup sizes and other options.
# NB: This has to be included before CmakeFunctionHelper as it declares various options.
if (NOT INSTALL_HEADER_ONLY)
  include(ConfigurePORTBLAS)
  include(SYCL)
  find_package(PythonInterp 3 REQUIRED)
endif()

if (MSVC)
  # The device compiler needs C++14 to parse the Windows headers
  set(BUILD_SHARED_LIBS FALSE CACHE BOOL
    "Force portBLAS to be built as a static library on Windows"
    FORCE
  )
endif()

# By default, tall and skinny Gemm is enabled (for better performance)
option(GEMM_TALL_SKINNY_SUPPORT "Whether to enable tall and skinny Gemm" ON)
# By default vectorization in gemm kernels is enabled as it imrpove the performance on all Devices.
option(GEMM_VECTORIZATION_SUPPORT "Whether to enable vectorization in Gemm kernels" ON)

add_definitions(-DCL_TARGET_OPENCL_VERSION=220)

set(CBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/external/cblas/include)
set(PORTBLAS_GENERATED_SRC ${CMAKE_CURRENT_BINARY_DIR}/generated_src)
set(PORTBLAS_INCLUDE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
set(PORTBLAS_COMMON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common/include)
set(PORTBLAS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(PORTBLAS_SRC_GENERATOR ${CMAKE_CURRENT_SOURCE_DIR}/python_generator)
list(APPEND THIRD_PARTIES_INCLUDE ${CBLAS_INCLUDE})

if(IMGDNN_DIR)
  add_definitions(-DIMGDNN_LIBRARY)
  find_package(IMGDNN REQUIRED)
  list(APPEND THIRD_PARTIES_INCLUDE ${IMGDNN_INCLUDE_DIRS})
endif()

option(BLAS_ENABLE_EXTENSIONS "Whether to enable portBLAS extensions" ON)
option(BLAS_ENABLE_COMPLEX "Whether to enable complex data type for supported operators" ON)

# CmakeFunctionHelper has to be included after any options that it depends on are declared.
# These include:
# * TARGET
# * GEMM_TALL_SKINNY_SUPPORT
# * GEMM_VECTORIZATION_SUPPORT
# * BLAS_DATA_TYPES
# * BLAS_INDEX_TYPES
# * NAIVE_GEMM
# * BLAS_ENABLE_COMPLEX
include(CmakeFunctionHelper)

if (INSTALL_HEADER_ONLY)
  add_library(portblas INTERFACE)
  set_target_properties(portblas PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${PORTBLAS_INCLUDE};$<INSTALL_INTERFACE:src>"
  )
else()
  add_subdirectory(src)
  build_library(portblas ${BLAS_ENABLE_EXTENSIONS})
  if (WIN32)
    # On Windows, all symbols must be resolved at link time for DLLs.
    # The portblas target is just a collection of other objects so
    # the linked libraries are not going to be propagated to this target.
    # This requires manual linking against SYCL on Windows.
    if(is_computecpp)
      target_link_libraries(portblas PUBLIC ComputeCpp::ComputeCpp)
    elseif(is_dpcpp)
      target_link_libraries(portblas PUBLIC DPCPP::DPCPP)
    elseif(is_hipsycl)
      target_link_libraries(portblas PUBLIC hipSYCL::hipSYCL-rt)
    endif()
  endif()
  if(is_computecpp)
    set(sycl_impl ComputeCpp::ComputeCpp)
  elseif(is_dpcpp)
    set(sycl_impl DPCPP::DPCPP)
    add_sycl_to_target(TARGET portblas SOURCES)
  elseif(is_hipsycl)
    set(sycl_impl hipSYCL::hipSYCL-rt)
    add_sycl_to_target(TARGET portblas SOURCES)
  endif()
  if(IMGDNN_DIR)
    target_link_libraries(portblas PUBLIC IMGDNN::IMGDNN)
  endif()
  set_target_properties(portblas PROPERTIES
    INTERFACE_LINK_LIBRARIES ${sycl_impl}
    INTERFACE_INCLUDE_DIRECTORIES "${PORTBLAS_INCLUDE}"
  )
  set_target_properties(portblas PROPERTIES
    VERSION ${PROJECT_VERSION}
  )
endif()

include(CMakePackageConfigHelpers)
set(version_file "${CMAKE_CURRENT_BINARY_DIR}/cmake/portblas-version.cmake")
write_basic_package_version_file(${version_file}
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

include(GNUInstallDirs)
install(TARGETS portblas
  EXPORT portblas
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY ${PORTBLAS_INCLUDE}
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  COMPONENT portblas
  FILES_MATCHING PATTERN "*.h"
)
if (INSTALL_HEADER_ONLY)
  install(DIRECTORY ${PORTBLAS_SRC}
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    COMPONENT portblas
    FILES_MATCHING PATTERN "*.hpp"
  )
endif()

install(FILES ${version_file} DESTINATION ${CMAKE_INSTALL_PREFIX})
install(EXPORT portblas
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  NAMESPACE PORTBLAS::
  FILE portblas-config.cmake
)

export(EXPORT portblas
  NAMESPACE PORTBLAS::
  FILE portblas-config.cmake
)

option(BLAS_ENABLE_TESTING "Whether to enable testing" ON)
option(ENABLE_EXPRESSION_TESTS "Whether to build expression tree fusion tests" OFF)
if (INSTALL_HEADER_ONLY AND BLAS_ENABLE_TESTING)
  message(STATUS "Tests are disabled when installing portBLAS in header only mode")
  set(BLAS_ENABLE_TESTING OFF)
endif()

if(${BLAS_ENABLE_TESTING})
  enable_testing()
  add_subdirectory(test)
endif()

option(BLAS_ENABLE_CONST_INPUT "Whether to enable kernel instantiation with const input buffer" ON)
option(BLAS_ENABLE_BENCHMARK "Whether to enable benchmarking" ON)
option(BLAS_VERIFY_BENCHMARK "Whether to verify the results of benchmarks" ON)
option(BLAS_MEMPOOL_BENCHMARK "Whether to use the memory pool in benchmarks" OFF)
option(BUILD_CLBLAST_BENCHMARKS "Whether to build clBLAST benchmarks" OFF)
option(BUILD_CLBLAS_BENCHMARKS "Whether to build clBLAS benchmarks" OFF)
option(BUILD_CUBLAS_BENCHMARKS "Whether to build cuBLAS benchmarks" OFF)
option(BUILD_ROCBLAS_BENCHMARKS "Whether to build rocBLAS benchmarks" OFF)
option(BUILD_ACL_BENCHMARKS "Whether to build ARM Compute Library benchmarks" OFF)
option(BLAS_BUILD_SAMPLES "Whether to build portBLAS samples" ON)
if (INSTALL_HEADER_ONLY AND BLAS_ENABLE_BENCHMARK)
  message(STATUS "Benchmarks are disabled when installing portBLAS in header only mode")
  set(BLAS_ENABLE_BENCHMARK OFF)
endif()
if (INSTALL_HEADER_ONLY AND BLAS_BUILD_SAMPLES)
  message(STATUS "Samples are disabled when installing portBLAS in header only mode")
  set(BLAS_BUILD_SAMPLES OFF)
endif()

if(${BLAS_ENABLE_BENCHMARK})
  add_subdirectory(benchmark)
endif()

if (BLAS_BUILD_SAMPLES)
  add_subdirectory(samples)
endif()

option(BLAS_ENABLE_AUTO_TUNERS "Whether to enable building GEMM auto tuners" OFF)
option(BLAS_ENABLE_AUTO_TUNER_MEMPOOL "Whether to enable memory pool for GEMM auto tuners" OFF)
if(${BLAS_ENABLE_AUTO_TUNERS})
  # Note that the auto tuners are very slow to compile, so we avoid adding
  # them to the ALL target.
  add_subdirectory(tools/auto_tuner EXCLUDE_FROM_ALL)
endif()
