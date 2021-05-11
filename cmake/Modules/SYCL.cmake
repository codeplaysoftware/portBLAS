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
include(CheckCXXCompilerFlag)
include(ConfigureSYCLBLAS)

# find_package(hipSYCL) requires HIPSYCL_TARGETS to be set, so set it to a default value before find_package(hipSYCL)
if(NOT HISPYCL_TARGETS AND NOT ENV{HIPSYCL_TARGETS})
  if(${TARGET} STREQUAL "NVIDIA_GPU")
    set(HIPSYCL_TARGETS "cuda:sm_35")
  elseif(${TARGET} STREQUAL "AMD_GPU")
    set(HISPYCL_TARGETS "hip:gfx900")
  elseif(${TARGET} STREQUAL "INTEL_GPU")
    set(HISPYCL_TARGETS "spirv")
  else()
    set(HIPSYCL_TARGETS "omp")
  endif()
endif()

check_cxx_compiler_flag("-fsycl" has_fsycl)

if(NOT SYCL_COMPILER)
  if(has_fsycl)
    set(is_dpcpp ON)
    set(SYCL_COMPILER "dpcpp")
  else()
    find_package(hipSYCL QUIET)
    set(is_hipsycl ${hipSYCL_FOUND})
    set(SYCL_COMPILER "hipsycl")
    if(NOT is_hipsycl)
      set(is_computecpp ON)
      set(SYCL_COMPILER "computecpp")
    endif()
  endif()
else()
  if(SYCL_COMPILER MATCHES "dpcpp")
    set(is_dpcpp ON)
    if(NOT has_fsycl)
      message(WARNING "Selected DPC++ as backend, but -fsycl not supported")
    endif()
  elseif(SYCL_COMPILER MATCHES "hipsycl")
    find_package(hipSYCL CONFIG)
    set(is_hipsycl ON)
  elseif(SYCL_COMPILER MATCHES "computecpp")
    set(is_computecpp ON)
  else()
    message(WARNING "SYCL_COMPILER <${SYCL_COMPILER}> is unknown.")
  endif()
endif()

message(STATUS "Using SYCL implementation: ${SYCL_COMPILER}")

if(is_computecpp)
  find_package(ComputeCpp REQUIRED)
  # Add some performance flags to the calls to compute++.
  # NB: This must be after finding ComputeCpp
  list(APPEND COMPUTECPP_USER_FLAGS
    -O3
    -fsycl-split-modules=20
    -mllvm -inline-threshold=10000
    -Xclang -cl-mad-enable
    # We add some flags to workaround OpenCL platform bugs, see ComputeCpp documentation
    -no-serial-memop
  )
  set(SYCL_INCLUDE_DIRS ${ComputeCpp_INCLUDE_DIRS})

  
elseif(is_dpcpp)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__SYCL_DISABLE_NAMESPACE_INLINE__=ON -O3 -Xclang -cl-mad-enable")
  if(${BACKEND_DEVICE} STREQUAL "DEFAULT_CPU") 
    set(DPCPP_SYCL_TARGET spir64_x86_64-unknown-unknown-sycldevice)
  elseif(${BACKEND_DEVICE} STREQUAL "INTEL_GPU")
  # the correct target-triple for intel gpu is: spir64_gen-unknown-unknown-sycldevice
  # however, the current version of DPCPP fails to link with the following error
  #  Error: Device name missing.
  #clang++: error: gen compiler command failed with exit code 226 (use -v to see invocation)
  #clang version 12.0.0 (https://github.com/intel/llvm.git 3582cb07f9f1acf3bee986008ec10265c4614346)
  #Target: x86_64-unknown-linux-gnu
  #Thread model: posix
  #InstalledDir: /home/mehdi/soft/intel/dpcpp_compiler/bin
  #clang++: note: diagnostic msg: Error generating preprocessed source(s) - no preprocessable inputs.
  #ninja: build stopped: subcommand failed.
  #TODO : (MEHDI) Create BuG report on intel/llvm 
    set(DPCPP_SYCL_TARGET spir64-unknown-unknown-sycldevice)
  elseif(${BACKEND_DEVICE} STREQUAL "NVIDIA_GPU")
    set(DPCPP_SYCL_TARGET nvptx64-nvidia-cuda-sycldevice)
  endif()
  find_package(DPCPP REQUIRED)
  get_target_property(SYCL_INCLUDE_DIRS DPCPP::DPCPP INTERFACE_INCLUDE_DIRECTORIES)
elseif(is_hipsycl)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  get_target_property(SYCL_INCLUDE_DIRS hipSYCL::hipSYCL-rt INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT ${BLAS_ENABLE_BENCHMARK})
    # hipSYCL currently does not support queue profiling. Thus disable benchmarks by default.
    set(BLAS_ENABLE_BENCHMARK Off)
  endif()
endif()
