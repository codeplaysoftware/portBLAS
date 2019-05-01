cmake_minimum_required(VERSION 3.2.2)

project(clblast_benchmarks)

set(CLBLAST_BENCH ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CLBLAST_BENCH})

# Try and build CLBlast benchmarks.
# Check to see if we've been passed an explicit path.
if (DEFINED CLBLAST_ROOT)
  message(STATUS "CLBlast root specified as : ${CLBLAST_ROOT}")
  set(CMAKE_PREFIX_PATH ${CLBLAST_ROOT})
else()
  # set a default path if not specified
  message(STATUS "CLBlast root not specified by user, searching default path /usr/local/lib/")
  set(CMAKE_PREFIX_PATH "/usr/local/lib/")
endif()
find_package(CLBlast REQUIRED)

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
)

# Add individual benchmarks for each method
foreach(clblast_benchmark ${CLBLAST_BENCH_SRCS})
  get_filename_component(clblast_bench_exec ${clblast_benchmark} NAME_WE)
  set(clblast_bench_exec, ${clblast_benchmark})
  add_executable(bench_clblast_${clblast_bench_exec} ${clblast_benchmark} ${CLBLAST_BENCH}/main.cpp ${CLBLAST_BENCH}/clwrap.cpp)
  set_property(TARGET bench_clblast_${clblast_bench_exec} PROPERTY CXX_STANDARD 11)
  add_dependencies(bench_clblast_${clblast_bench_exec} benchmark)
  target_link_libraries(bench_clblast_${clblast_bench_exec} PUBLIC ${OpenCL_LIBRARIES})
  target_link_libraries(bench_clblast_${clblast_bench_exec} PUBLIC benchmark)
  target_link_libraries(bench_clblast_${clblast_bench_exec} PRIVATE clblast)
  target_include_directories(bench_clblast_${clblast_bench_exec} PUBLIC ${ComputeCpp_INCLUDE_DIRS} ${SYCLBLAS_INCLUDE} ${CLARA_INCLUDE})

  message("-- Created benchmark: ${clblast_bench_exec}")
  install(TARGETS bench_clblast_${clblast_bench_exec} RUNTIME DESTINATION sycl_blas/bin)
endforeach()
