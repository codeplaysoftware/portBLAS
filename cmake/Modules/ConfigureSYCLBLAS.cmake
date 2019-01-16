
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
# *  @filename ConfigureSYCLBLAS.cmake
# *
# **************************************************************************/
# We add some flags to workaround OpenCL platform bugs, see ComputeCpp documentation
list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS
    ${COMPUTECPP_DEVICE_COMPILER_FLAGS} -no-serial-memop -Xclang -cl-mad-enable -O3)
message(STATUS "${COMPUTECPP_DEVICE_COMPILER_FLAGS}")

# Check to see if we've disabled double support in the tests
option(DOUBLE_SUPPORT "Disable double support when testing." off)
if(DOUBLE_SUPPORT)
  # Define NO_DOUBLE_SUPPORT for the host cxx compiler
  add_definitions(-DDOUBLE_SUPPORT)
endif()

# If the user has specified a specific workgroup size for tests, pass that on to the compiler
if(WG_SIZE)
  add_definitions(-DWG_SIZE=${WG_SIZE})
endif()

# If the user has specified that we should use naive gemm, enable that
option(NAIVE_GEMM "Default to naive GEMM implementations" off)
if(NAIVE_GEMM)
  add_definitions(-DNAIVE_GEMM)
endif()

# the TARGET variable defines the platform for which the sycl library is built
SET(TARGET "DEFAULT_CPU" CACHE STRING "Default Platform 'DEFAULT_CPU'")
message(STATUS "${TARGET} is chosen as a backend platform")
