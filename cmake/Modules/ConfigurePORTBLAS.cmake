
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
# *  @filename ConfigurePORTBLAS.cmake
# *
# **************************************************************************/

set(BLAS_DATA_TYPES "float" CACHE STRING "Data types to test")
set(BLAS_INDEX_TYPES "int" CACHE STRING "Supported index/increment types")

# Select an index type to run the tests and benchmark with. Use first given index type.
list(GET BLAS_INDEX_TYPES 0 BLAS_TEST_INDEX_TYPE)
list(GET BLAS_INDEX_TYPES 0 BLAS_BENCHMARK_INDEX_TYPE)

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

# the TUNING_TARGET variable defines the platform for which the sycl library is tuned
SET(TUNING_TARGET "DEFAULT_CPU" CACHE STRING "Default Platform 'DEFAULT_CPU'")
message(STATUS "${TUNING_TARGET} is chosen as a tuning target")

if(DEFINED TARGET)
  message(FATAL_ERROR
            "\nSetting the TARGET CMake variable is no longer supported. "
            "Set TUNING_TARGET instead, it accepts the same options.\n"
            "Further details can be found in README.md.\n"
            "You can remove this error by unsetting TARGET invoking cmake "
            "with -UTARGET argument.\n" )
endif()

if (WIN32)
  # On Win32, shared library symbols need to be explicitly exported.
  if (BUILD_SHARED_LIBS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  endif()
endif()
