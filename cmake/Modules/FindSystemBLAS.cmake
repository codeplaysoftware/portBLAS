
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
# *  @filename FindSystemBLAS.cmake
# *
# **************************************************************************/

include(FindPackageHandleStandardArgs)

find_library(OPENBLAS_LIBRARIES NAMES openblas libopenblas)
find_path(OPENBLAS_INCLUDE_DIRS openblas_config.h)
if(OPENBLAS_LIBRARIES AND OPENBLAS_INCLUDE_DIRS)
  find_package(Threads REQUIRED)
  find_package_handle_standard_args(SystemBLAS REQUIRED_VARS OPENBLAS_LIBRARIES OPENBLAS_INCLUDE_DIRS)
  add_library(blas::blas UNKNOWN IMPORTED)
  set_target_properties(blas::blas PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OPENBLAS_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES Threads::Threads
    IMPORTED_LOCATION "${OPENBLAS_LIBRARIES}"
  )
  return()
endif()

find_package(BLAS QUIET)
if(NOT BLAS_FOUND)
  set(BLA_STATIC ON)
  find_package(BLAS QUIET)
endif()

if(BLAS_FOUND AND NOT TARGET blas::blas)
  find_package_handle_standard_args(SystemBLAS REQUIRED_VARS BLAS_LIBRARIES)

  add_library(blas::blas INTERFACE IMPORTED)
  set_target_properties(blas::blas PROPERTIES
    INTERFACE_LINK_LIBRARIES "${BLAS_LINKER_FLAGS};${BLAS_LIBRARIES}"
  )
endif()
