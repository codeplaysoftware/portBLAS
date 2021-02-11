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
add_library(sycl_policy OBJECT ${SYCLBLAS_SRC}/policy/sycl_policy_handler.cpp)
set_target_compile_def(sycl_policy)

# This can be replaced by target_link_libraries(sycl_policy PRIVATE sycl_blas_common)
# in CMake 3.12, but travis uses CMake 3.5 so the include directories for the common
# library need to be manually set here
get_target_property(_sycl_blas_common_include_dir sycl_blas_common INCLUDE_DIRECTORIES)

target_include_directories(sycl_policy PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE}
                           ${ComputeCpp_INCLUDE_DIRS} ${COMPUTECPP_SDK_INCLUDE}
                           ${_sycl_blas_common_include_dir})

add_sycl_to_target(TARGET sycl_policy SOURCES ${SYCLBLAS_SRC}/policy/sycl_policy_handler.cpp)
