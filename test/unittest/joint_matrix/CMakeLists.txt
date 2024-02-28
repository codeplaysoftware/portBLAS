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

set(PORTBLAS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
set(PORTBLAS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../src)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
include(ConfigurePORTBLAS)
include(SYCL)
find_package(PORTBLAS REQUIRED)

set(PORTBLAS_JOINTMATRIX_TEST ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${PORTBLAS_TEST} ${BLAS_INCLUDE_DIRS})

# compiling tests
set(SYCL_UNITTEST_SRCS
  ${PORTBLAS_JOINTMATRIX_TEST}/half_half_16_16_16.cpp
  ${PORTBLAS_JOINTMATRIX_TEST}/half_half_32_8_16.cpp
  ${PORTBLAS_JOINTMATRIX_TEST}/half_half_8_32_16.cpp
  ${PORTBLAS_JOINTMATRIX_TEST}/half_float_16_16_16.cpp
  ${PORTBLAS_JOINTMATRIX_TEST}/half_float_32_8_16.cpp
  ${PORTBLAS_JOINTMATRIX_TEST}/half_float_8_32_16.cpp
  ${PORTBLAS_JOINTMATRIX_TEST}/bfloat16_float_16_16_16.cpp
  ${PORTBLAS_JOINTMATRIX_TEST}/bfloat16_float_32_8_16.cpp
  ${PORTBLAS_JOINTMATRIX_TEST}/bfloat16_float_8_32_16.cpp
  ${PORTBLAS_JOINTMATRIX_TEST}/tf32_float_16_16_8.cpp
)

foreach(blas_test ${SYCL_UNITTEST_SRCS})
  get_filename_component(test_exec ${blas_test} NAME_WE)
  add_executable(joint_matrix_${test_exec}_test ../main.cpp ${blas_test})
  target_compile_definitions(joint_matrix_${test_exec}_test PRIVATE -DBLAS_INDEX_T=${BLAS_TEST_INDEX_TYPE})
  target_link_libraries(joint_matrix_${test_exec}_test PRIVATE gtest_main Clara::Clara blas::blas PORTBLAS::PORTBLAS)
  target_include_directories(joint_matrix_${test_exec}_test PRIVATE ${SYCL_INCLUDE_DIRS})
  target_include_directories(joint_matrix_${test_exec}_test PRIVATE ${CBLAS_INCLUDE} ${PORTBLAS_COMMON_INCLUDE_DIR})
  target_compile_options(joint_matrix_${test_exec}_test PRIVATE ${DPCPP_FLAGS})
  target_link_options(joint_matrix_${test_exec}_test PRIVATE ${DPCPP_FLAGS})

  if(TEST_DEVICE)
    add_test(NAME joint_matrix_${test_exec}_test COMMAND ${CMAKE_CURRENT_BINARY_DIR}/joint_matrix_${test_exec}_test --device ${TEST_DEVICE} --gtest_output=xml:output/)
  else()
    add_test(NAME joint_matrix_${test_exec}_test COMMAND ${CMAKE_CURRENT_BINARY_DIR}/joint_matrix_${test_exec}_test --gtest_output=xml:output/)
  endif()
  message(STATUS "Created google test joint_matrix_${test_exec}_test")
  install(TARGETS joint_matrix_${test_exec}_test
    RUNTIME
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT tests
  )
endforeach()
