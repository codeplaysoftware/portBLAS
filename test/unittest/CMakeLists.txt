cmake_minimum_required(VERSION 3.2.2)

project(syclblas_unittest)

set(SYCLBLAS_UNITTEST ${CMAKE_CURRENT_SOURCE_DIR})
set(SYCLBLAS_TEST_INCLUDE "${SYCLBLAS_TEST}/include")

include_directories(${SYCLBLAS_TEST})

# compiling tests
file(GLOB SYCL_UNITTEST_SRCS
  ${SYCLBLAS_UNITTEST}/blas1_copy_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1_swap_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1_axpy_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1_scal_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1_asum_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1_dot_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1_nrm2_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1_rotg_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1_iamax_test.cpp
  ${SYCLBLAS_UNITTEST}/blas1_iamin_test.cpp
)

foreach(blas_test ${SYCL_UNITTEST_SRCS})
  get_filename_component(test_exec ${blas_test} NAME_WE)
  set(test_exec, ${blas_test})
  add_executable(${test_exec} main.cpp ${blas_test})
  set_property(TARGET ${test_exec} PROPERTY CXX_STANDARD 11)
  target_link_libraries(${test_exec} PUBLIC libgtest libgmock)
  add_sycl_to_target(${test_exec} ${CMAKE_CURRENT_BINARY_DIR} ${blas_test})
  add_test(NAME ${test_exec} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${test_exec})
  message("-- Created google test ${test_exec}")
endforeach()
