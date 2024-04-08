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
# *  @filename FindSystemBLAS.cmake
# *
# **************************************************************************/
include(FindPackageHandleStandardArgs)

# Let user possibility to set blas path to similar libraries like lapack
if(BLAS_LIBRARIES AND BLAS_INCLUDE_DIRS)
  find_package(Threads REQUIRED)
  add_library(blas::blas UNKNOWN IMPORTED)
  set_target_properties(blas::blas PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${BLAS_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES Threads::Threads
    IMPORTED_LOCATION "${BLAS_LIBRARIES}"
  )
  set(SystemBLAS_LIBRARIES ${BLAS_LIBRARIES})
else()
  set(BLA_VENDOR OpenBLAS)
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    set(SystemBLAS_LIBRARIES ${BLAS_LIBRARIES})
  endif()
  if(NOT BLAS_FOUND)
    message(WARNING "openBLAS library was not found on your system")
    unset(BLA_VENDOR)
    set(BLA_VENDOR All)
    find_package(BLAS QUIET)
    if (BLAS_FOUND)
      message(WARNING "Found another BLAS library on your system. Not using openBLAS may cause some tests to fail.")
      message("-- BLAS library found at ${BLAS_LIBRARIES}")
      set(SystemBLAS_LIBRARIES ${BLAS_LIBRARIES})
    endif()
  endif()

  if(NOT BLAS_FOUND)
    set(BLA_STATIC ON)
    find_package(BLAS QUIET)
    set(SystemBLAS_LIBRARIES ${BLAS_LIBRARIES})
  endif()

  if(BLAS_FOUND AND NOT TARGET blas::blas)
    add_library(blas::blas INTERFACE IMPORTED)
    set_target_properties(blas::blas PROPERTIES
      INTERFACE_LINK_LIBRARIES "${BLAS_LINKER_FLAGS};${BLAS_LIBRARIES}"
    )
  endif()

endif()

find_package_handle_standard_args(SystemBLAS
  REQUIRED_VARS SystemBLAS_LIBRARIES)
