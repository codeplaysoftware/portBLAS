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
# *  @filename CMakeLists.txt
# *
# **************************************************************************/

include(FindPackageHandleStandardArgs)

find_package(BLAS)
find_package(OpenBLAS)
find_package(Threads REQUIRED)

if(OpenBLAS_DIR)
    set(SystemBLAS_LIBRARIES ${OpenBLAS_LIBRARIES})
    set(SystemBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIRS})
elseif(BLAS_FOUND)
    set(SystemBLAS_LIBRARIES ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
endif()
message(STATUS "${SystemBLAS_INCLUDE_DIRS}")
find_package_handle_standard_args(SystemBLAS REQUIRED_VARS SystemBLAS_LIBRARIES)

if(SystemBLAS_FOUND AND NOT TARGET SystemBLAS::BLAS)
    add_library(SystemBLAS::BLAS INTERFACE IMPORTED)
    set_target_properties(SystemBLAS::BLAS PROPERTIES
        INTERFACE_LINK_LIBRARIES "${SystemBLAS_LIBRARIES};Threads::Threads"
        INTERFACE_INCLUDE_DIRECTORIES "${SystemBLAS_INCLUDE_DIRS}"
    )
endif()