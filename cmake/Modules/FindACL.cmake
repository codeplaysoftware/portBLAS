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
# *  @filename FindCLBlast.cmake
# *
# **************************************************************************/

find_package(OpenCL)
find_package(npy)
find_path(ACL_INCLUDE_DIR "arm_compute/graph.h" HINTS ${ACL_ROOT})
find_path(ACL_SYSTEM_INCLUDE_DIR "half/half.hpp" HINTS ${ACL_ROOT} PATH_SUFFIXES include/)

find_library(ACL_LIBRARY NAME arm_compute HINTS ${ACL_ROOT} PATH_SUFFIXES build lib)
find_library(ACL_CORE_LIBRARY NAME arm_compute_core HINTS ${ACL_ROOT} PATH_SUFFIXES build lib)
find_library(ACL_GRAPH_LIBRARY NAME arm_compute_graph HINTS ${ACL_ROOT} PATH_SUFFIXES build lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ACL REQUIRED_VARS ACL_LIBRARY ACL_CORE_LIBRARY
                                                    ACL_GRAPH_LIBRARY ACL_INCLUDE_DIR ACL_SYSTEM_INCLUDE_DIR
                                                    OpenCL_FOUND npy_FOUND)

if(ACL_FOUND AND NOT TARGET acl)
    add_library(acl INTERFACE IMPORTED)
    set_target_properties(acl PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ACL_INCLUDE_DIR};${ACL_SYSTEM_INCLUDE_DIR};${OpenCL_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${OpenCL_LIBRARIES};${CMAKE_DL_LIBS};${ACL_LIBRARY};${ACL_CORE_LIBRARY};${ACL_GRAPH_LIBRARY};npy::npy"
    )
    mark_as_advanced(ACL_LIBRARY ACL_CORE_LIBRARY ACL_GRAPH_LIBRARY
      ACL_INCLUDE_DIR ACL_SYSTEM_INCLUDE_DIR)
endif()
