cmake_minimum_required(VERSION 3.2.2)

project(syclblas_benchmarks)

set(SYCLBLAS_BENCH ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${SYCLBLAS_BENCH})

file(GLOB SYCLBLAS_BENCH_SRCS
  # Level 1 blas
  ${SYCLBLAS_BENCH}/blas1/axpy.cpp
  ${SYCLBLAS_BENCH}/blas1/asum.cpp
  ${SYCLBLAS_BENCH}/blas1/axpy3op.cpp
  ${SYCLBLAS_BENCH}/blas1/blas1.cpp
  ${SYCLBLAS_BENCH}/blas1/dot.cpp
  ${SYCLBLAS_BENCH}/blas1/iamax.cpp
  ${SYCLBLAS_BENCH}/blas1/iamin.cpp
  ${SYCLBLAS_BENCH}/blas1/nrm2.cpp
  ${SYCLBLAS_BENCH}/blas1/scal.cpp
  ${SYCLBLAS_BENCH}/blas1/scal2op.cpp
  ${SYCLBLAS_BENCH}/blas1/scal3op.cpp
  # Level 2 blas
  ${SYCLBLAS_BENCH}/blas2/gemv.cpp
)

# Add individual benchmarks for each method
foreach(syclblas_bench ${SYCLBLAS_BENCH_SRCS})
  get_filename_component(bench_exec ${syclblas_bench} NAME_WE)
  set(bench_exec, ${syclblas_bench})
  add_executable(${bench_exec} ${syclblas_bench} ${SYCLBLAS_BENCH}/main.cpp )
  set_property(TARGET ${bench_exec} PROPERTY CXX_STANDARD 11)

  add_dependencies(${bench_exec} benchmark)

  target_link_libraries(${bench_exec} benchmark)
  add_sycl_to_target(
    TARGET ${bench_exec}
    SOURCES ${syclblas_bench}
  )

  message("-- Created benchmark: ${bench_exec}")
  install(TARGETS ${bench_exec} RUNTIME DESTINATION bin)
endforeach()

# Add an overall benchmark for level 1
add_executable(sycl_blas1
  ${SYCLBLAS_BENCH}/blas1/axpy.cpp
  ${SYCLBLAS_BENCH}/blas1/asum.cpp
  ${SYCLBLAS_BENCH}/blas1/axpy3op.cpp
  ${SYCLBLAS_BENCH}/blas1/blas1.cpp
  ${SYCLBLAS_BENCH}/blas1/dot.cpp
  ${SYCLBLAS_BENCH}/blas1/iamax.cpp
  ${SYCLBLAS_BENCH}/blas1/iamin.cpp
  ${SYCLBLAS_BENCH}/blas1/nrm2.cpp
  ${SYCLBLAS_BENCH}/blas1/scal.cpp
  ${SYCLBLAS_BENCH}/blas1/scal2op.cpp
  ${SYCLBLAS_BENCH}/blas1/scal3op.cpp
  ${SYCLBLAS_BENCH}/main.cpp
)
set_property(TARGET sycl_blas1 PROPERTY CXX_STANDARD 11)
add_dependencies(sycl_blas1 benchmark)
target_link_libraries(sycl_blas1 benchmark)
add_sycl_to_target(TARGET sycl_blas1)
message("-- Created benchmark: sycl_blas1 ")
install(TARGETS sycl_blas1 RUNTIME DESTINATION bin)