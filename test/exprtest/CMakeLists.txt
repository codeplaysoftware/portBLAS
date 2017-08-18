cmake_minimum_required(VERSION 3.0.2)

project(syclblas_unittest)

set(SYCLBLAS_EXPRTEST ${CMAKE_CURRENT_SOURCE_DIR})
set(SYCLBLAS_TEST_INCLUDE "${SYCLBLAS_TEST}/include")

include_directories(${SYCLBLAS_TEST})

# compiling tests
file(GLOB SYCL_EXPRTEST_SRCS
  ${SYCLBLAS_EXPRTEST}/blas1_iface_test.cpp
)

foreach(blas_test ${SYCL_EXPRTEST_SRCS})
  message("-- blas test: ${blas_test}")
  get_filename_component(test_exec ${blas_test} NAME_WE)
  message("-- test exec: ${test_exec}")
  set(test_exec, ${blas_test})
  add_executable(${test_exec} main.cpp ${blas_test})
  set_property(TARGET ${test_exec} PROPERTY CXX_STANDARD 11)
  target_link_libraries(${test_exec} PUBLIC libgtest libgmock)
  add_sycl_to_target(${test_exec} ${CMAKE_CURRENT_BINARY_DIR} ${blas_test})
  add_test(NAME ${test_exec} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${test_exec})
  message("-- Created google test ${test_exec}")
endforeach()
