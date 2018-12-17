cmake_minimum_required(VERSION 3.2.2)

project(syclblas_benchmarks)

set(SYCLBLAS_BENCH ${CMAKE_CURRENT_SOURCE_DIR})
# set(SYCLBLAS_TEST_INCLUDE "${SYCLBLAS_TEST}/include")

# include_directories(${SYCLBLAS_TEST})

add_executable(example ${SYCLBLAS_BENCH}/example.cpp)
set_property(TARGET example PROPERTY CXX_STANDARD 11)
add_dependencies(example benchmark)
target_link_libraries(example benchmark)
install(TARGETS ${bench_exec} RUNTIME DESTINATION bin)

# compiling tests
# file(GLOB SYCLBLAS_BENCH_SRCS
#   ${SYCLBLAS_BENCH}/example.cpp

  # Blas 1 tests
  # ${SYCLBLAS_BENCH}/blas1/blas1_copy_test.cpp
  # ${SYCLBLAS_BENCH}/blas1/blas1_swap_test.cpp
  # ${SYCLBLAS_BENCH}/blas1/blas1_axpy_test.cpp
  # ${SYCLBLAS_BENCH}/blas1/blas1_scal_test.cpp
  # ${SYCLBLAS_BENCH}/blas1/blas1_asum_test.cpp
  # ${SYCLBLAS_BENCH}/blas1/blas1_dot_test.cpp
  # ${SYCLBLAS_BENCH}/blas1/blas1_nrm2_test.cpp
  # ${SYCLBLAS_BENCH}/blas1/blas1_rotg_test.cpp
  # ${SYCLBLAS_BENCH}/blas1/blas1_iamax_test.cpp
  # ${SYCLBLAS_BENCH}/blas1/blas1_iamin_test.cpp

  # Blas 2 tests
  # ${SYCLBLAS_BENCH}/blas2/blas2_gemv_test.cpp
  # ${SYCLBLAS_BENCH}/blas2/blas2_ger_test.cpp

  # Blas 3 tests
  # ${SYCLBLAS_BENCH}/blas3/gemm/blas3_gemm_nn_test.cpp
  # ${SYCLBLAS_BENCH}/blas3/gemm/blas3_gemm_nt_test.cpp
  # ${SYCLBLAS_BENCH}/blas3/gemm/blas3_gemm_nc_test.cpp
  # ${SYCLBLAS_BENCH}/blas3/gemm/blas3_gemm_tn_test.cpp
  # ${SYCLBLAS_BENCH}/blas3/gemm/blas3_gemm_tt_test.cpp
  # ${SYCLBLAS_BENCH}/blas3/gemm/blas3_gemm_tc_test.cpp
  # ${SYCLBLAS_BENCH}/blas3/gemm/blas3_gemm_cc_test.cpp
  # ${SYCLBLAS_BENCH}/blas3/gemm/blas3_gemm_cn_test.cpp
  # ${SYCLBLAS_BENCH}/blas3/gemm/blas3_gemm_ct_test.cpp

  # Blas buffer tests
  # ${SYCLBLAS_BENCH}/buffers/sycl_buffer_test.cpp
# )

# foreach(blas_bench ${SYCLBLAS_BENCH_SRCS})
  # get_filename_component(bench_exec ${blas_bench} NAME_WE)
  # set(bench_exec, ${blas_bench})

  # add_executable(${bench_exec} main.cpp ${blas_bench})
  # set_property(TARGET ${bench_exec} PROPERTY CXX_STANDARD 11)
  # add_dependencies(${bench_exec} gtest_main)

  # target_link_libraries(${bench_exec} PUBLIC gtest_main ${BLAS_LIBRARIES})
  # add_sycl_to_target(
  #   TARGET ${bench_exec}
  #   SOURCES ${blas_bench}
  # )
  # add_test(NAME ${bench_exec} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${bench_exec})
  # message("-- Created google test ${bench_exec}")
  # install(TARGETS ${bench_exec} RUNTIME DESTINATION bin)

# endforeach()
