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
# *  @filename FindPORTBLAS.cmake
# *
# **************************************************************************/

find_path(PORTBLAS_INCLUDE_DIR
  NAMES portblas.h
  PATH_SUFFIXES include
  HINTS ${PORTBLAS_DIR}
  DOC "The PORTBLAS include directory"
)

find_path(PORTBLAS_SRC_DIR
  NAMES portblas.hpp
  PATH_SUFFIXES src
  HINTS ${PORTBLAS_DIR}
  DOC "The PORTBLAS source directory"
)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PORTBLAS
  FOUND_VAR PORTBLAS_FOUND
  REQUIRED_VARS PORTBLAS_INCLUDE_DIR
                PORTBLAS_SRC_DIR
)

mark_as_advanced(PORTBLAS_FOUND
                 PORTBLAS_SRC_DIR
                 PORTBLAS_INCLUDE_DIR
)

if(PORTBLAS_FOUND)
  set(PORTBLAS_INCLUDE_DIRS
    ${PORTBLAS_INCLUDE_DIR}
    ${PORTBLAS_SRC_DIR}
  )
endif()

if(PORTBLAS_FOUND AND NOT TARGET PORTBLAS::PORTBLAS)
  add_library(PORTBLAS::PORTBLAS INTERFACE IMPORTED)
  set_target_properties(PORTBLAS::PORTBLAS PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${PORTBLAS_INCLUDE_DIRS}"
  )
endif()
