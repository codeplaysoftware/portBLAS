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
find_path(MYCLBLAST_INCLUDE_DIR clblast.h HINTS ${CLBLAST_ROOT} PATH_SUFFIXES include)
find_library(MYCLBLAST_LIBRARY NAME clblast libclblast.so HINTS ${CLBLAST_ROOT} PATH_SUFFIXES lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MYCLBlast REQUIRED_VARS MYCLBLAST_LIBRARY MYCLBLAST_INCLUDE_DIR OpenCL_FOUND)

message(STATUS "Found MYCLBLAST: ${MYCLBLAST_LIBRARY} - ${MYCLBLAST_INCLUDE_DIR} - ${OpenCL_FOUND}")

if(MYCLBlast_FOUND AND NOT TARGET clblast)
    add_library(clblast UNKNOWN IMPORTED)
    set_target_properties(clblast PROPERTIES
        IMPORTED_LOCATION "${MYCLBLAST_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MYCLBLAST_INCLUDE_DIR};${OpenCL_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${OpenCL_LIBRARIES}"
    )
endif()
