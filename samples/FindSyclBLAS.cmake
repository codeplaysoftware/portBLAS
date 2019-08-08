# Copyright 2018 Codeplay Software Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use these files except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Try to find the SyclBLAS library.
#
# If the library is found then the `SyclBLAS::SyclBLAS` target will be exported
# with the required include directories.
#
# Sets the following variables:
#   SyclBLAS_FOUND        - whether the system has SyclBLAS
#   SyclBLAS_INCLUDE_DIRS - the SyclBLAS include directory

find_path(SyclBLAS_INCLUDE_DIR
  NAMES sycl_blas.h
  PATH_SUFFIXES include
  HINTS ${SyclBLAS_DIR}
  DOC "The SyclBLAS include directory"
)

find_path(SyclBLAS_SRC_DIR
  NAMES sycl_blas.hpp
  PATH_SUFFIXES src
  HINTS ${SyclBLAS_DIR}
  DOC "The SyclBLAS source directory"
)

find_path(SyclBLAS_VPTR_INCLUDE_DIR
  NAMES vptr/virtual_ptr.hpp
  PATH_SUFFIXES external/computecpp-sdk/include
  HINTS ${SyclBLAS_DIR}
  DOC "The SyclBLAS virtual pointer include directory"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SyclBLAS
  FOUND_VAR SyclBLAS_FOUND
  REQUIRED_VARS SyclBLAS_INCLUDE_DIR
                SyclBLAS_SRC_DIR
                SyclBLAS_VPTR_INCLUDE_DIR
)

mark_as_advanced(SyclBLAS_FOUND
                 SyclBLAS_SRC_DIR
                 SyclBLAS_VPTR_INCLUDE_DIR
                 SyclBLAS_INCLUDE_DIR
)

if(SyclBLAS_FOUND)
  set(SyclBLAS_INCLUDE_DIRS
    ${SyclBLAS_INCLUDE_DIR}
    ${SyclBLAS_SRC_DIR}
    ${SyclBLAS_VPTR_INCLUDE_DIR}
  )
endif()

if(SyclBLAS_FOUND AND NOT TARGET SyclBLAS::SyclBLAS)
  add_library(SyclBLAS::SyclBLAS INTERFACE IMPORTED)
  set_target_properties(SyclBLAS::SyclBLAS PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SyclBLAS_INCLUDE_DIRS}"
    INTERFACE_COMPUTECPP_FLAGS "-DSYCL_BLAS_ALWAYS_INLINE=1"
  )
endif()
