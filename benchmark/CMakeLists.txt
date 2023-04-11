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
find_package(Clara REQUIRED)

# compile googlebench
include(ExternalProject)


set(BLAS_BENCH ${CMAKE_CURRENT_SOURCE_DIR})
set(BLAS_BENCH_INCLUDE "${BLAS_BENCH}/include")

# Download and build googlebench
# The following operations are derived from the official advice on building
# googlebench as a subproject, from the following documentation:
# https://github.com/google/googletest/blob/master/googletest/README.md
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/templates/GBench.txt.in
  ${CMAKE_CURRENT_BINARY_DIR}/googlebench-download/CMakeLists.txt
)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googlebench-download )
if(result)
  message(FATAL_ERROR "CMake step for googlebench failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googlebench-download )
if(result)
  message(FATAL_ERROR "Build step for googlebench failed: ${result}")
endif()

# set BENCHMARK_ENABLE_TESTING to OFF for google bench
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googlebench-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googlebench-build
                 EXCLUDE_FROM_ALL)

# Set SYCL-BLAS options
if(CMAKE_CROSSCOMPILING)
    set(cmake_toolchain
      "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
       )
     message(STATUS "Crossing : ${cmake_toolchain}")
endif()

if(VERBOSE)
  add_definitions(-DVERBOSE=VERBOSE)
endif(VERBOSE)

if(SYCL_DEVICE)
  add_definitions(-DSYCL_DEVICE=${SYCL_DEVICE})
endif(SYCL_DEVICE)

include_directories(${BLAS_BENCH})
add_subdirectory(${BLAS_BENCH}/syclblas)

if (BUILD_CLBLAST_BENCHMARKS)
  add_subdirectory(${BLAS_BENCH}/clBench/clblast)
endif()

if (BUILD_ACL_BENCHMARKS)
  add_subdirectory(${BLAS_BENCH}/clBench/acl)
endif()

if(BUILD_CLBLAS_BENCHMARKS)
  add_subdirectory(${BLAS_BENCH}/clBench/clblas)
endif()

if(BUILD_CUBLAS_BENCHMARKS)
  add_subdirectory(${BLAS_BENCH}/cublas/)
endif()

if(BUILD_ROCBLAS_BENCHMARKS)
  add_subdirectory(${BLAS_BENCH}/rocblas)
endif()
