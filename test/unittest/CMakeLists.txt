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

set(SYCLBLAS_UNITTEST ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${SYCLBLAS_TEST} ${BLAS_INCLUDE_DIRS})

# compiling tests
set(SYCL_UNITTEST_SRCS
  # Blas 1 tests
  ${SYCLBLAS_UNITTEST}/blas1/blas1_axpy_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1/blas1_scal_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1/blas1_rot_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1/blas1_rotm_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1/blas1_rotmg_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1/blas1_rotg_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1/blas1_sdsdot_test.cpp
  # Blas 2 tests
  ${SYCLBLAS_UNITTEST}/blas2/blas2_gbmv_test.cpp
  ${SYCLBLAS_UNITTEST}/blas2/blas2_gemv_test.cpp
  ${SYCLBLAS_UNITTEST}/blas2/blas2_ger_test.cpp
  ${SYCLBLAS_UNITTEST}/blas2/blas2_sbmv_test.cpp
  ${SYCLBLAS_UNITTEST}/blas2/blas2_syr_test.cpp
  ${SYCLBLAS_UNITTEST}/blas2/blas2_syr2_test.cpp
  ${SYCLBLAS_UNITTEST}/blas2/blas2_symv_test.cpp
  # Blas 3 tests
  ${SYCLBLAS_UNITTEST}/blas3/blas3_gemm_test.cpp
  ${SYCLBLAS_UNITTEST}/blas3/blas3_gemm_batched_test.cpp
  ${SYCLBLAS_UNITTEST}/blas3/blas3_trsm_test.cpp
)

# Temporary disabling the following tests fro Intel DPC++ as currently Intel compiler crashes while running the following tests
if(is_computecpp)
  set(SYCL_UNITTEST_SRCS ${SYCL_UNITTEST_SRCS}
    # Blas 1 tests
    ${SYCLBLAS_UNITTEST}/blas1/blas1_copy_test.cpp
    ${SYCLBLAS_UNITTEST}/blas1/blas1_swap_test.cpp
    ${SYCLBLAS_UNITTEST}/blas1/blas1_asum_test.cpp
    ${SYCLBLAS_UNITTEST}/blas1/blas1_dot_test.cpp
    ${SYCLBLAS_UNITTEST}/blas1/blas1_nrm2_test.cpp
    ${SYCLBLAS_UNITTEST}/blas1/blas1_iamax_test.cpp
    ${SYCLBLAS_UNITTEST}/blas1/blas1_iamin_test.cpp
    # Blas 2 tests
    ${SYCLBLAS_UNITTEST}/blas2/blas2_trmv_test.cpp
    ${SYCLBLAS_UNITTEST}/blas2/blas2_tbmv_test.cpp
    # Blas buffer tests
    ${SYCLBLAS_UNITTEST}/buffers/sycl_buffer_test.cpp
  )

  if(${BLAS_ENABLE_EXTENSIONS})
    list(APPEND SYCL_UNITTEST_SRCS "${SYCLBLAS_UNITTEST}/reduction/reduction_test.cpp")
  endif()
endif()

if(GEMM_TALL_SKINNY_SUPPORT)
  list(APPEND SYCL_UNITTEST_SRCS ${SYCLBLAS_UNITTEST}/blas3/blas3_gemm_tall_skinny_test.cpp)
endif()

foreach(blas_test ${SYCL_UNITTEST_SRCS})
  get_filename_component(test_exec ${blas_test} NAME_WE)
  add_executable(${test_exec} main.cpp ${blas_test})
  if(is_computecpp)
    set_property(TARGET ${test_exec} PROPERTY CXX_STANDARD 14)
  endif()
  if(STRESS_TESTING)
    target_compile_definitions(${test_exec} PRIVATE STRESS_TESTING)
  endif()
  target_compile_definitions(${test_exec} PRIVATE -DBLAS_INDEX_T=${BLAS_TEST_INDEX_TYPE})
  target_link_libraries(${test_exec} PRIVATE gtest_main Clara::Clara blas::blas sycl_blas)
  target_include_directories(${test_exec} PRIVATE ${CBLAS_INCLUDE} ${SYCLBLAS_COMMON_INCLUDE_DIR})
  if(TEST_DEVICE)
    add_test(NAME ${test_exec} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${test_exec} --device ${TEST_DEVICE} --gtest_output=xml:output/)
  else()
    add_test(NAME ${test_exec} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${test_exec} --gtest_output=xml:output/)
  endif()
  message(STATUS "Created google test ${test_exec}")
  install(TARGETS ${test_exec}
    RUNTIME
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT tests
  )
endforeach()
