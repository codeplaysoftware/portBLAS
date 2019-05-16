cmake_minimum_required(VERSION 3.2.2)

project(clblast_benchmarks)

set(CLBLAST_BENCH ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CLBLAST_BENCH})

if(${BLAS_VERIFY_BENCHMARK})
  add_definitions(-DBLAS_VERIFY_BENCHMARK)
  find_package(SystemBLAS REQUIRED)
endif()

find_package(SB_CLBlast REQUIRED)

set(CLBLAST_BENCH_SRCS
  # Level 1 blas
  ${CLBLAST_BENCH}/blas1/asum.cpp
  ${CLBLAST_BENCH}/blas1/axpy.cpp
  ${CLBLAST_BENCH}/blas1/dot.cpp
  ${CLBLAST_BENCH}/blas1/iamax.cpp
  ${CLBLAST_BENCH}/blas1/iamin.cpp
  ${CLBLAST_BENCH}/blas1/nrm2.cpp
  ${CLBLAST_BENCH}/blas1/scal.cpp
  # Level 2 blas
  ${CLBLAST_BENCH}/blas2/gemv.cpp
  # Level 3 blas
  ${CLBLAST_BENCH}/blas3/gemm.cpp
  ${CLBLAST_BENCH}/blas3/gemm_batched.cpp
)

# Add individual benchmarks for each method
foreach(clblast_benchmark ${CLBLAST_BENCH_SRCS})
  get_filename_component(clblast_bench_exec ${clblast_benchmark} NAME_WE)
  add_executable(bench_clblast_${clblast_bench_exec} ${clblast_benchmark} ${CLBLAST_BENCH}/main.cpp ${CLBLAST_BENCH}/clwrap.cpp)
  set_target_properties(bench_clblast_${clblast_bench_exec} PROPERTIES CXX_STANDARD 11)
  add_dependencies(bench_clblast_${clblast_bench_exec} benchmark)
  target_link_libraries(bench_clblast_${clblast_bench_exec} PUBLIC benchmark ${OpenCL_LIBRARIES} ${BLAS_LIBRARIES})
  target_link_libraries(bench_clblast_${clblast_bench_exec} PRIVATE clblast)
  target_include_directories(bench_clblast_${clblast_bench_exec} PUBLIC ${SYCLBLAS_INCLUDE} ${THIRD_PARTIES_INCLUDE} ${BLAS_INCLUDE_DIRS})

  message("-- Created benchmark: ${clblast_bench_exec}")
  install(TARGETS bench_clblast_${clblast_bench_exec} RUNTIME DESTINATION sycl_blas/bin)
endforeach()
