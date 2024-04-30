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
# *  portBLAS: BLAS implementation using SYCL
# *
# *  @filename CMakeLists.txt
# *
# **************************************************************************/

set(PORTBLAS_UNITTEST ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${PORTBLAS_TEST} ${BLAS_INCLUDE_DIRS})

# compiling tests
set(SYCL_UNITTEST_SRCS
  # Blas 1 tests
  ${PORTBLAS_UNITTEST}/blas1/blas1_asum_test.cpp
  ${PORTBLAS_UNITTEST}/blas1/blas1_axpy_test.cpp
  ${PORTBLAS_UNITTEST}/blas1/blas1_copy_test.cpp
  ${PORTBLAS_UNITTEST}/blas1/blas1_scal_test.cpp
  ${PORTBLAS_UNITTEST}/blas1/blas1_swap_test.cpp
  ${PORTBLAS_UNITTEST}/blas1/blas1_rot_test.cpp
  ${PORTBLAS_UNITTEST}/blas1/blas1_rotm_test.cpp
  ${PORTBLAS_UNITTEST}/blas1/blas1_rotmg_test.cpp
  ${PORTBLAS_UNITTEST}/blas1/blas1_rotg_test.cpp
  ${PORTBLAS_UNITTEST}/blas1/blas1_sdsdot_test.cpp
  ${PORTBLAS_UNITTEST}/blas1/blas1_nrm2_test.cpp
  ${PORTBLAS_UNITTEST}/blas1/blas1_dot_test.cpp
  ${PORTBLAS_UNITTEST}/blas1/blas1_iamax_test.cpp
  ${PORTBLAS_UNITTEST}/blas1/blas1_iamin_test.cpp
  # # Blas 2 tests
  ${PORTBLAS_UNITTEST}/blas2/blas2_gbmv_test.cpp
  ${PORTBLAS_UNITTEST}/blas2/blas2_gemv_test.cpp
  ${PORTBLAS_UNITTEST}/blas2/blas2_ger_test.cpp
  ${PORTBLAS_UNITTEST}/blas2/blas2_sbmv_test.cpp
  ${PORTBLAS_UNITTEST}/blas2/blas2_spmv_test.cpp
  ${PORTBLAS_UNITTEST}/blas2/blas2_syr_test.cpp
  ${PORTBLAS_UNITTEST}/blas2/blas2_spr_test.cpp
  ${PORTBLAS_UNITTEST}/blas2/blas2_spr2_test.cpp
  ${PORTBLAS_UNITTEST}/blas2/blas2_syr2_test.cpp
  ${PORTBLAS_UNITTEST}/blas2/blas2_symv_test.cpp
  ${PORTBLAS_UNITTEST}/blas2/blas2_tpmv_test.cpp
  ${PORTBLAS_UNITTEST}/blas2/blas2_trmv_test.cpp
  ${PORTBLAS_UNITTEST}/blas2/blas2_tbmv_test.cpp
  # Blas 3 tests
  ${PORTBLAS_UNITTEST}/blas3/blas3_gemm_test.cpp
  ${PORTBLAS_UNITTEST}/blas3/blas3_gemm_batched_test.cpp
  ${PORTBLAS_UNITTEST}/blas3/blas3_trsm_test.cpp
  ${PORTBLAS_UNITTEST}/blas3/blas3_symm_test.cpp
  # Blas extension
  ${PORTBLAS_UNITTEST}/extension/omatcopy_test.cpp
  ${PORTBLAS_UNITTEST}/extension/omatadd_test.cpp
  ${PORTBLAS_UNITTEST}/extension/omatcopy2_test.cpp
  ${PORTBLAS_UNITTEST}/extension/omatcopy_batched_test.cpp
  ${PORTBLAS_UNITTEST}/extension/omatadd_batched_test.cpp
  ${PORTBLAS_UNITTEST}/extension/axpy_batch_test.cpp
  ${PORTBLAS_UNITTEST}/buffers/sycl_buffer_test.cpp
)

if(is_adaptivecpp)
  # Skip these tests for AdaptiveCpp for SPIRV/OpenCL targets
  # that use SYCL 2020 features like group reduction or hang
  # during execution (https://github.com/AdaptiveCpp/AdaptiveCpp/issues/1309)
  set(TESTS_TO_SKIP
    ${PORTBLAS_UNITTEST}/blas1/blas1_asum_test.cpp
    ${PORTBLAS_UNITTEST}/blas1/blas1_sdsdot_test.cpp
    ${PORTBLAS_UNITTEST}/blas1/blas1_nrm2_test.cpp
    ${PORTBLAS_UNITTEST}/blas1/blas1_dot_test.cpp
    ${PORTBLAS_UNITTEST}/blas1/blas1_rot_test.cpp
    # Hang during execution (without failing)
    ${PORTBLAS_UNITTEST}/blas3/blas3_trsm_test.cpp
  )
endif()

if(${BLAS_ENABLE_EXTENSIONS})
  list(APPEND SYCL_UNITTEST_SRCS ${PORTBLAS_UNITTEST}/extension/transpose_test.cpp
                                 ${PORTBLAS_UNITTEST}/extension/reduction_test.cpp
)
endif()

# Enable testing of the sycl 2020 routines just for Intel DPC++
if(is_dpcpp)
  set(SYCL_UNITTEST_SRCS ${SYCL_UNITTEST_SRCS}
    ${PORTBLAS_UNITTEST}/blas2/blas2_trsv_test.cpp
    ${PORTBLAS_UNITTEST}/blas2/blas2_tbsv_test.cpp
    ${PORTBLAS_UNITTEST}/blas2/blas2_tpsv_test.cpp
  )
endif()

if(is_dpcpp AND ${TUNING_TARGET} STREQUAL "DEFAULT")
  if (${DPCPP_SYCL_TARGET} MATCHES "nvidia")
    set(TESTS_TO_SKIP
      ${PORTBLAS_UNITTEST}/blas3/blas3_trsm_test.cpp
      )
    message(WARNING "Targetting NVIDIA hardware with DEFAULT TUNING_TARGET.
    Disabling tests for following operators: iamax, iamin, trsv, tbsv, tpsv, trsm.")
  elseif (${DPCPP_SYCL_TARGET} MATCHES "amd")
    set(TESTS_TO_SKIP
      ${PORTBLAS_UNITTEST}/blas1/blas1_iamax_test.cpp
      ${PORTBLAS_UNITTEST}/blas1/blas1_iamin_test.cpp
      ${PORTBLAS_UNITTEST}/blas2/blas2_tbsv_test.cpp
      ${PORTBLAS_UNITTEST}/blas2/blas2_tpsv_test.cpp
      ${PORTBLAS_UNITTEST}/blas2/blas2_trsv_test.cpp
      )
    message(WARNING "Targetting AMD hardware with DEFAULT TUNING_TARGET.
    Disabling tests for following operators: iamax, iamin, tbsv, tpsv, trsv.")
  endif()
endif()

if(GEMM_TALL_SKINNY_SUPPORT)
  list(APPEND SYCL_UNITTEST_SRCS ${PORTBLAS_UNITTEST}/blas3/blas3_gemm_tall_skinny_test.cpp)
endif()

set(HALF_DATA_OPS "blas1_axpy_test" 
                  "blas1_scal_test"
                  "blas3_gemm_test"
                  "blas3_gemm_batched_test"
                  )

foreach(blas_test ${SYCL_UNITTEST_SRCS})
  if(${blas_test} IN_LIST TESTS_TO_SKIP)
    continue()
  endif()
  get_filename_component(test_exec ${blas_test} NAME_WE)
  add_executable(${test_exec} main.cpp ${blas_test})
  if(is_computecpp)
    set_property(TARGET ${test_exec} PROPERTY CXX_STANDARD 14)
  endif()
  if(STRESS_TESTING)
    target_compile_definitions(${test_exec} PRIVATE STRESS_TESTING)
  endif()
  if(${BLAS_ENABLE_COMPLEX})
    if(${test_exec} MATCHES "gemm")
      target_compile_definitions(${test_exec} PRIVATE BLAS_ENABLE_COMPLEX=1)
    endif()
  endif()
  if((${BLAS_ENABLE_HALF}) AND (${test_exec} IN_LIST HALF_DATA_OPS))
    target_compile_definitions(${test_exec} PRIVATE BLAS_ENABLE_HALF=1)
  endif()
  target_compile_definitions(${test_exec} PRIVATE -DBLAS_INDEX_T=${BLAS_TEST_INDEX_TYPE})
  target_link_libraries(${test_exec} PRIVATE gtest_main Clara::Clara blas::blas portblas)
  target_include_directories(${test_exec} PRIVATE ${CBLAS_INCLUDE} ${PORTBLAS_COMMON_INCLUDE_DIR})

  # Cmake identifies compilers as IntelLLVM only those distributed with Intel oneAPI releases,
  # so this flag doesn't apply to intel/llvm open source compiler.
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM")
    target_compile_options(${test_exec} PRIVATE "-fno-fast-math")
  endif()

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

if(${ENABLE_JOINTMATRIX_TESTS})
  if (${DPCPP_SYCL_TARGET} STREQUAL "nvptx64-nvidia-cuda")
    string(FIND ${DPCPP_SYCL_ARCH} "_" start_idx)
    if(start_idx)
      MATH(EXPR start_idx "${start_idx} + 1")
      string(SUBSTRING ${DPCPP_SYCL_ARCH} ${start_idx} "2" sm_val)
    endif()

    if (${start_idx} AND ${sm_val} GREATER_EQUAL "80")
      add_subdirectory(joint_matrix)
    else()
      message(FATAL_ERROR "Joint Matrix Tests only supported for NVIDIA GPUs with sm_80 arch and above.")
    endif()
  else()
    message(FATAL_ERROR "Joint Matrix Tests only supported for NVIDIA GPUs.")
  endif()
endif()
