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
cmake_minimum_required (VERSION 3.2.2)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(${CMAKE_VERSION} VERSION_LESS "3.9.0") 
  project(sycl-blas VERSION 0.1.0)
else()
  project(sycl-blas VERSION 0.1.0 DESCRIPTION "sycl-blas library")
endif()

set(COMPUTECPP_SDK_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/external/computecpp-sdk/include)
set(SYCLBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SYCLBLAS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(SYCLBLAS_SRC_GENERATOR ${CMAKE_CURRENT_SOURCE_DIR}/python_generator)
set(SYCLBLAS_GENERATED_SRC ${CMAKE_CURRENT_BINARY_DIR}/generated_src)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external/computecpp-sdk/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external/sample/cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -w -Wall -Wno-long-long -pedantic")
list(APPEND COMPUTECPP_USER_FLAGS "-O3" "-fsycl-split-modules=20" "-mllvm" "-inline-threshold=10000" "-no-serial-memop")
list(APPEND THIRD_PARTIES_INCLUDE ${ComputeCpp_INCLUDE_DIRS} ${COMPUTECPP_SDK_INCLUDE})
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)
find_package(ComputeCpp REQUIRED)
include(ConfigureSYCLBLAS)
find_package(PythonInterp 2.7 REQUIRED)
include(CmakeFunctionHelper)

#by default always inlining the kernels
option(SYCL_BLAS_ALWAYS_INLINE "Setting always inline attribute for all kernels" ON)

add_subdirectory(src)
#shared lib
build_library(sycl_blas SHARED)
target_include_directories(sycl_blas PUBLIC ${SYCLBLAS_INCLUDE} ${THIRD_PARTIES_INCLUDE})
set_target_properties(sycl_blas PROPERTIES VERSION ${PROJECT_VERSION})

install(TARGETS sycl_blas 
LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/sycl_blas/lib
)

option(BLAS_ENABLE_STATIC_LIBRARY "Whether to enable STATIC LIBRARY" OFF)

#static_lib
if(BLAS_ENABLE_STATIC_LIBRARY)
build_library(sycl_blas_static STATIC)
install(TARGETS sycl_blas_static 
ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/sycl_blas/lib
)
endif()

install(DIRECTORY ${SYCLBLAS_INCLUDE} DESTINATION ${CMAKE_INSTALL_PREFIX}/sycl_blas
        FILES_MATCHING PATTERN "*.h")

option(BLAS_ENABLE_TESTING "Whether to enable testing" ON)

if(${BLAS_ENABLE_TESTING})
  enable_testing()
  add_subdirectory(test)
endif()

option(BLAS_ENABLE_BENCHMARK "Whether to enable benchmarking" ON)

if(${BLAS_ENABLE_BENCHMARK})
  add_subdirectory(gbench)
  add_subdirectory(bench)
endif()

