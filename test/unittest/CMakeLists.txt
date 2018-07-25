set(test_sources
  blas1_copy_test.cpp
  blas1_swap_test.cpp
  blas1_axpy_test.cpp
  blas1_scal_test.cpp
  blas1_asum_test.cpp
  blas1_dot_test.cpp
  blas1_nrm2_test.cpp
  blas1_rotg_test.cpp
  blas1_iamax_test.cpp
  blas1_iamin_test.cpp
  blas2_gemv_test.cpp
  blas2_ger_test.cpp
  blas3_gemm_test.cpp
  sycl_buffer_test.cpp)

add_executable(unittest ${test_sources} main.cpp)
target_link_libraries(unittest PRIVATE sycl-blas ${GTEST_LIBRARIES} Threads::Threads)
target_include_directories(unittest PRIVATE ${GTEST_INCLUDE_DIRS})
gtest_add_tests(unittest "" AUTO)
add_sycl_to_target(TARGET unittest SOURCES ${test_sources})
