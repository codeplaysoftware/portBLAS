cmake_minimum_required(VERSION 3.2.2)

project(syclblas_benchmarks)

set(SYCLBLAS_BENCH ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${SYCLBLAS_BENCH} ${CLARA_INCLUDE})

if(${BLAS_VERIFY_BENCHMARK})
  add_definitions(-DBLAS_VERIFY_BENCHMARK)
  find_package(SystemBLAS REQUIRED)
endif()

set(SYCLBLAS_BENCH_SRCS
  # Level 1 blas
  ${SYCLBLAS_BENCH}/blas1/axpy.cpp
  ${SYCLBLAS_BENCH}/blas1/asum.cpp
  ${SYCLBLAS_BENCH}/blas1/dot.cpp
  ${SYCLBLAS_BENCH}/blas1/iamax.cpp
  ${SYCLBLAS_BENCH}/blas1/iamin.cpp
  ${SYCLBLAS_BENCH}/blas1/nrm2.cpp
  ${SYCLBLAS_BENCH}/blas1/scal.cpp
  # Level 2 blas
  ${SYCLBLAS_BENCH}/blas2/gemv.cpp
  # Level 3 blas
  ${SYCLBLAS_BENCH}/blas3/gemm.cpp
)

# Add individual benchmarks for each method
foreach(syclblas_bench ${SYCLBLAS_BENCH_SRCS})
  get_filename_component(bench_exec ${syclblas_bench} NAME_WE)
  add_executable(bench_${bench_exec} ${syclblas_bench} ${SYCLBLAS_BENCH}/main.cpp )
  set_target_properties(bench_${bench_exec} PROPERTIES CXX_STANDARD 11)
  add_dependencies(bench_${bench_exec} benchmark)
  target_link_libraries(bench_${bench_exec} PUBLIC benchmark ${BLAS_LIBRARIES} sycl_blas ComputeCpp::ComputeCpp)

  message("-- Created benchmark: ${bench_exec}")
  install(TARGETS bench_${bench_exec} RUNTIME DESTINATION sycl_blas/bin)
endforeach()
