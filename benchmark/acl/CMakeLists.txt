cmake_minimum_required(VERSION 3.2.2)

project(acl_benchmarks)

set(ACL_BENCH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(ACL REQUIRED)
if(ACL_BACKEND STREQUAL "NEON")
  add_definitions(-DACL_BACKEND_NEON)
else()
  add_definitions(-DACL_BACKEND_OPENCL)
endif()

set(ACL_BENCH_SRCS
  # Level 3 blas
  ${ACL_BENCH}/blas3/gemm.cpp
)

# Add individual benchmarks for each method
foreach(acl_benchmark ${ACL_BENCH_SRCS})
  get_filename_component(acl_bench_exec ${acl_benchmark} NAME_WE)
  add_executable(bench_acl_${acl_bench_exec} ${acl_benchmark} ${ACL_BENCH}/main.cpp)
  set_target_properties(bench_acl_${acl_bench_exec} PROPERTIES CXX_STANDARD 11)
  add_dependencies(bench_acl_${acl_bench_exec} benchmark)
  target_link_libraries(bench_acl_${acl_bench_exec} PUBLIC benchmark)
  target_link_libraries(bench_acl_${acl_bench_exec} PRIVATE acl)
  target_include_directories(bench_acl_${acl_bench_exec} PUBLIC ${ACL_BENCH} ${SYCLBLAS_INCLUDE} ${THIRD_PARTIES_INCLUDE})

  message("-- Created benchmark: ${acl_bench_exec}")
  install(TARGETS bench_acl_${acl_bench_exec} RUNTIME DESTINATION sycl_blas/bin)
endforeach()
