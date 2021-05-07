
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

set(BLAS_DATA_TYPES "float" CACHE STRING "Data types to test")

# Check to see if we've enabled double support in tests
option(DOUBLE_SUPPORT "Enable double support when testing." off)
if(DOUBLE_SUPPORT)
  message(DEPRECATION
    "Please add \"double\" to BLAS_DATA_TYPES instead of enabling DOUBLE_SUPPORT")
  if(NOT ("double" IN_LIST BLAS_DATA_TYPES))
    list(APPEND BLAS_DATA_TYPES "double")
  endif()
endif()

if(NOT ("float" IN_LIST BLAS_DATA_TYPES))
  message(FATAL_ERROR "float must be specified in BLAS_DATA_TYPES")
endif()

if("double" IN_LIST BLAS_DATA_TYPES)
  add_definitions(-DBLAS_DATA_TYPE_DOUBLE)
endif()

if("half" IN_LIST BLAS_DATA_TYPES)
  add_definitions(-DBLAS_DATA_TYPE_HALF)
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
SET(BACKEND_DEVICE ${TARGET})
message(STATUS "${TARGET} is chosen as a backend platform")

# the BLAS_MODEL_OPTIMIZATION variable defines which model optimized configs should
# be enabled for. Currently only affects ARM_GPU configs.
SET(BLAS_MODEL_OPTIMIZATION "DEFAULT" CACHE STRING "Default Model 'DEFAULT'")
