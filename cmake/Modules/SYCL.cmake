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
include(CheckCXXCompilerFlag)
include(ConfigurePORTBLAS)

# find_package(AdaptiveCpp) requires ACPP_TARGETS to be set, so set it to a default value before find_package(AdaptiveCpp)
if(SYCL_COMPILER MATCHES "adaptivecpp")
  if(NOT ACPP_TARGETS AND NOT ENV{ACPP_TARGETS})
    message(STATUS "Using `omp` as ACPP_TARGETS")
    set(ACPP_TARGETS "omp")
  else()
    message(STATUS "Using ${ACPP_TARGETS} as ACPP_TARGETS")
  endif()
endif()

check_cxx_compiler_flag("--acpp-targets" has_acpp)
check_cxx_compiler_flag("-fsycl" has_fsycl)

if(NOT SYCL_COMPILER)
  if(has_acpp)
    find_package(AdaptiveCpp QUIET)
    set(is_adaptivecpp ${AdaptiveCpp_FOUND})
    set(SYCL_COMPILER "adaptivecpp")
  else()
    set(is_dpcpp ON)
    set(SYCL_COMPILER "dpcpp")
  endif()
else()
  if(SYCL_COMPILER MATCHES "dpcpp")
    set(is_dpcpp ON)
    if(NOT has_fsycl)
      message(WARNING "Selected DPC++ as backend, but -fsycl not supported")
    endif()
  elseif(SYCL_COMPILER MATCHES "adaptivecpp")
    find_package(AdaptiveCpp CONFIG REQUIRED)
    set(is_adaptivecpp ${AdaptiveCpp_FOUND})
    if(NOT has_acpp)
      message(WARNING "Selected AdaptiveCpp as backend, but the compiler is not 
              fully supported")
    endif()
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
  if(NOT DEFINED DPCPP_SYCL_TARGET)
    message(STATUS "Using `spir64` as target triplet")
    set(DPCPP_SYCL_TARGET spir64)
  endif()
  find_package(DPCPP REQUIRED)
  get_target_property(SYCL_INCLUDE_DIRS DPCPP::DPCPP INTERFACE_INCLUDE_DIRECTORIES)
elseif(is_adaptivecpp)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  get_target_property(SYCL_INCLUDE_DIRS AdaptiveCpp::acpp-rt INTERFACE_INCLUDE_DIRECTORIES)
  set(HIP_BENCH_UNSUPPORTED_TARGETS "INTEL_GPU" "DEFAULT_CPU")
  if((${BLAS_ENABLE_BENCHMARK}) AND (${TUNING_TARGET} IN_LIST HIP_BENCH_UNSUPPORTED_TARGETS))
    message(STATUS "Benchmarks are not supported when targetting OpenCL/LevelZero backend 
            devices. portBLAS Benchmarks are disabled.")
    set(BLAS_ENABLE_BENCHMARK OFF)
  endif()
endif()
