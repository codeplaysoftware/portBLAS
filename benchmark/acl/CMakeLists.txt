set(ACL_BENCH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(ACL REQUIRED)
if(ACL_BACKEND STREQUAL "NEON")
  add_definitions(-DACL_BACKEND_NEON)
else()
  add_definitions(-DACL_BACKEND_OPENCL)
endif()

if(${BLAS_VERIFY_BENCHMARK})
  add_definitions(-DBLAS_VERIFY_BENCHMARK)
  find_package(SystemBLAS REQUIRED)
endif()

set(ACL_BENCH_SRCS
  # Level 3 blas
  ${ACL_BENCH}/blas3/gemm.cpp
)

# Add individual benchmarks for each method
foreach(acl_benchmark ${ACL_BENCH_SRCS})
  get_filename_component(acl_bench_exec ${acl_benchmark} NAME_WE)
  add_executable(bench_acl_${acl_bench_exec} ${acl_benchmark} ${ACL_BENCH}/main.cpp)
  target_link_libraries(bench_acl_${acl_bench_exec} PRIVATE benchmark ${BLAS_LIBRARIES} acl sycl_blas)
  target_include_directories(bench_acl_${acl_bench_exec} PRIVATE ${ACL_BENCH} ${CLARA_INCLUDE} ${CBLAS_INCLUDE})

  message("-- Created benchmark: ${acl_bench_exec}")
  install(TARGETS bench_acl_${acl_bench_exec}
    RUNTIME
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT benchmarks
  )
endforeach()
