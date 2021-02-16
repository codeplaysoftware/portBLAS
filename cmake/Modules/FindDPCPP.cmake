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
# *  @filename FindDPCPP.cmake
# *
# **************************************************************************/

include_guard()

include(CheckCXXCompilerFlag)
include(FindPackageHandleStandardArgs)

get_filename_component(DPCPP_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
find_library(DPCPP_LIB_DIR NAMES sycl PATHS "${DPCPP_BIN_DIR}/../lib")

add_library(DPCPP::DPCPP INTERFACE IMPORTED)

if(UNIX)
  set_target_properties(DPCPP::DPCPP PROPERTIES
    INTERFACE_COMPILE_OPTIONS "-fsycl;-fsycl-targets=${DPCPP_SYCL_TARGET}"
    INTERFACE_LINK_OPTIONS "-fsycl;-fsycl-targets=${DPCPP_SYCL_TARGET}"
    INTERFACE_LINK_LIBRARIES ${DPCPP_LIB_DIR}
    INTERFACE_INCLUDE_DIRECTORIES "${DPCPP_BIN_DIR}/../include/sycl")
else()
  set_target_properties(DPCPP::DPCPP PROPERTIES
    INTERFACE_COMPILE_OPTIONS "-fsycl;-fsycl-targets=${DPCPP_SYCL_TARGET}"
    INTERFACE_LINK_LIBRARIES ${DPCPP_LIB_DIR}
    INTERFACE_INCLUDE_DIRECTORIES "${DPCPP_BIN_DIR}/../include/sycl")
endif()

function(add_sycl_to_target)
  set(options)
  set(one_value_args TARGET)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(SB_ADD_SYCL
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  target_compile_options(${SB_ADD_SYCL_TARGET} PUBLIC -fsycl
                          PUBLIC -fsycl-targets=${DPCPP_SYCL_TARGET})
  get_target_property(target_type ${SB_ADD_SYCL_TARGET} TYPE)
  if (NOT target_type STREQUAL "OBJECT_LIBRARY")
    target_link_options(${SB_ADD_SYCL_TARGET} PUBLIC -fsycl
                        PUBLIC -fsycl-targets=${DPCPP_SYCL_TARGET})
  endif()                             
endfunction()
