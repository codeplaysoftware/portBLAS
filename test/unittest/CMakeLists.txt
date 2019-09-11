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
cmake_minimum_required(VERSION 3.4.2)

project(syclblas_unittest)

set(SYCLBLAS_UNITTEST ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${SYCLBLAS_TEST} ${BLAS_INCLUDE_DIRS})

# compiling tests
set(SYCL_UNITTEST_SRCS
  # Blas 1 tests
  ${SYCLBLAS_UNITTEST}/blas1/blas1_axpy_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1/blas1_copy_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1/blas1_swap_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1/blas1_scal_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1/blas1_asum_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1/blas1_dot_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1/blas1_nrm2_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1/blas1_rotg_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1/blas1_iamax_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1/blas1_iamin_test.cpp
  # Blas 2 tests
  ${SYCLBLAS_UNITTEST}/blas2/blas2_gemv_test.cpp
  ${SYCLBLAS_UNITTEST}/blas2/blas2_ger_test.cpp
  ${SYCLBLAS_UNITTEST}/blas2/blas2_trmv_test.cpp
  ${SYCLBLAS_UNITTEST}/blas2/blas2_syr_test.cpp
  ${SYCLBLAS_UNITTEST}/blas2/blas2_syr2_test.cpp
  ${SYCLBLAS_UNITTEST}/blas2/blas2_symv_test.cpp
  # Blas 3 tests
  ${SYCLBLAS_UNITTEST}/blas3/blas3_gemm_test.cpp
  ${SYCLBLAS_UNITTEST}/blas3/blas3_gemm_batched_test.cpp
  ${SYCLBLAS_UNITTEST}/blas3/blas3_experimental_test.cpp
  # Blas buffer tests
  ${SYCLBLAS_UNITTEST}/buffers/sycl_buffer_test.cpp
)

if(GEMM_TALL_SKINNY_SUPPORT)
  list(APPEND SYCL_UNITTEST_SRCS ${SYCLBLAS_UNITTEST}/blas3/blas3_gemm_tall_skinny_test.cpp)
endif()

foreach(blas_test ${SYCL_UNITTEST_SRCS})
  get_filename_component(test_exec ${blas_test} NAME_WE)
  set(test_exec, ${blas_test})
  add_executable(${test_exec} main.cpp ${blas_test})
  set_property(TARGET ${test_exec} PROPERTY CXX_STANDARD 11)
  add_dependencies(${test_exec} gtest_main sycl_blas)
  if (STRESS_TESTING)
  target_compile_definitions(${test_exec} PRIVATE STRESS_TESTING)
  endif()
  target_link_libraries(${test_exec} PUBLIC gtest_main ${BLAS_LIBRARIES} sycl_blas ComputeCpp::ComputeCpp)
  if (TEST_DEVICE)
    add_test(NAME ${test_exec} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${test_exec} --device ${TEST_DEVICE})
  else()
    add_test(NAME ${test_exec} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${test_exec})
  endif()
  message("-- Created google test ${test_exec}")
  install(TARGETS ${test_exec} RUNTIME DESTINATION sycl_blas/bin)

endforeach()
