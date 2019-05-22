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

function(unary)
endfunction()

function(binary)
    foreach(func ${ARGV})
        syclblas_add_binary_function_instantiations(
            MODEL "${CMAKE_CURRENT_LIST_DIR}/model.yaml"
            TEMPLATE ${func}.cpp.in
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${func}.cpp"
        )
        set(sycl_blas_extern_instantiations ${sycl_blas_extern_instantiations} ${CMAKE_CURRENT_BINARY_DIR}/${func}.cpp)
    endforeach()
    set(sycl_blas_extern_instantiations ${sycl_blas_extern_instantiations} PARENT_SCOPE)
endfunction()

function(binary_special)
endfunction()

function(ternary)
endfunction()

add_subdirectory(blas1)
#add_subdirectory(blas2)
#add_subdirectory(blas3)

set(sycl_blas_extern_instantiations ${sycl_blas_extern_instantiations} PARENT_SCOPE)
