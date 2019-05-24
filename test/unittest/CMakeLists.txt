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
# *  @filename executor_base.hpp
# *
# **************************************************************************/

find_package(Clara REQUIRED)

set(sources
    blas1/blas1_axpy_test.cpp
    blas1/blas1_copy_test.cpp
    blas1/blas1_swap_test.cpp
    blas1/blas1_scal_test.cpp
    blas1/blas1_asum_test.cpp
    blas1/blas1_dot_test.cpp
    blas1/blas1_nrm2_test.cpp
    blas1/blas1_rotg_test.cpp
    blas1/blas1_iamax_test.cpp
    blas1/blas1_iamin_test.cpp

    blas2/blas2_gemv_test.cpp
    blas2/blas2_ger_test.cpp
    blas2/blas2_trmv_test.cpp
    blas2/blas2_syr_test.cpp
    blas2/blas2_syr2_test.cpp
    blas2/blas2_symv_test.cpp

    blas3/blas3_gemm_test.cpp
    blas3/blas3_gemm_batched_test.cpp

    buffers/sycl_buffer_test.cpp
)
add_executable(unittests main.cpp ${sources})
target_link_libraries(unittests PRIVATE test_common Clara::Clara)
add_sycl_to_target(TARGET unittests SOURCES ${sources})

if(STRESS_TESTING)
    target_compile_definitions(unittests PRIVATE STRESS_TESTING)
endif()
