find_package(ACL REQUIRED)

if(BLAS_VERIFY_BENCHMARK)
  find_package(SystemBLAS REQUIRED)
endif()

set(sources
  # Level 3 blas
  blas3/gemm.cpp
)

# Add individual benchmarks for each method
foreach(acl_benchmark ${sources})
  get_filename_component(acl_bench_exec ${acl_benchmark} NAME_WE)
  add_executable(bench_acl_${acl_bench_exec} ${acl_benchmark} main.cpp)
  target_link_libraries(bench_acl_${acl_bench_exec} PRIVATE benchmark acl sycl_blas)
  target_include_directories(bench_acl_${acl_bench_exec} PRIVATE ${CLARA_INCLUDE} ${CBLAS_INCLUDE})

  if(BLAS_VERIFY_BENCHMARK)
    target_compile_definitions(bench_acl_${acl_bench_exec} PRIVATE BLAS_VERIFY_BENCHMARK)
    target_link_libraries(bench_acl_${acl_bench_exec} PRIVATE blas::blas)
  endif()

  if(ACL_BACKEND STREQUAL "NEON")
    target_compile_definitions(bench_acl_${acl_bench_exec} PRIVATE ACL_BACKEND_NEON)
  else()
    target_compile_definitions(bench_acl_${acl_bench_exec} PRIVATE ACL_BACKEND_OPENCL)
  endif()

  message(STATUS "Created benchmark: ${acl_bench_exec}")
  install(TARGETS bench_acl_${acl_bench_exec}
    RUNTIME
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT benchmarks
  )
endforeach()
