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
cmake_minimum_required(VERSION 3.2.2)

project(syclblas_exprtest)

set(SYCLBLAS_EXPRTEST ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${SYCLBLAS_TEST} )

# compiling tests
file(GLOB SYCL_EXPRTEST_SRCS
  #${SYCLBLAS_EXPRTEST}/blas1_iface_test.cpp
)

foreach(blas_test ${SYCL_EXPRTEST_SRCS})
  message("-- blas test: ${blas_test}")
  get_filename_component(test_exec ${blas_test} NAME_WE)
  message("-- test exec: ${test_exec}")
  set(test_exec, ${blas_test})
  add_executable(${test_exec} main.cpp ${blas_test})
  set_property(TARGET ${test_exec} PROPERTY CXX_STANDARD 11)
  add_dependencies(${test_exec} gtest_main)
  target_link_libraries(${test_exec} PUBLIC gtest_main ${BLAS_LIBRARIES})
  add_sycl_to_target(
    TARGET ${test_exec}
    SOURCES main.cpp ${blas_test}
  )
  if (TEST_DEVICE)
    add_test(NAME ${test_exec} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${test_exec} --device ${TEST_DEVICE})
  else()
    add_test(NAME ${test_exec} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${test_exec})
  endif()
  message("-- Created google test ${test_exec}")
  install(TARGETS ${test_exec} RUNTIME DESTINATION sycl_blas/bin)

endforeach()

# header only expression tree tests
set(SYCL_HEADER_ONLY_EXPRTEST_SRCS
  ${SYCLBLAS_EXPRTEST}/blas1_axpy_copy_test.cpp
  #${SYCLBLAS_EXPRTEST}/blas1_reduction_asum_test.cpp
)

foreach(blas_test ${SYCL_HEADER_ONLY_EXPRTEST_SRCS})
  include_directories(${SYCLBLAS_SRC})
  get_filename_component(test_exec ${blas_test} NAME_WE)
  set(test_exec, ${blas_test})
  add_executable(${test_exec} main.cpp ${blas_test})
  add_sycl_to_target(TARGET ${test_exec} SOURCES  main.cpp ${blas_test})
  set_property(TARGET ${test_exec} PROPERTY CXX_STANDARD 11)
  add_dependencies(${test_exec} gtest_main)
  target_link_libraries(${test_exec} PUBLIC gtest_main ${BLAS_LIBRARIES} sycl_blas ComputeCpp::ComputeCpp)
  add_test(NAME ${test_exec} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${test_exec})
  message("-- Created google test ${test_exec}")
  install(TARGETS ${test_exec} RUNTIME DESTINATION sycl_blas/bin)
endforeach()
