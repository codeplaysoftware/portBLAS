if(BLAS_VERIFY_BENCHMARK)
  find_package(SystemBLAS REQUIRED)
endif()

set(sources
  # Level 1 blas
  blas1/axpy.cpp
  blas1/asum.cpp
  blas1/dot.cpp
  blas1/iamax.cpp
  blas1/iamin.cpp
  blas1/nrm2.cpp
  blas1/scal.cpp
  # Level 2 blas
  blas2/gemv.cpp
  # Level 3 blas
  blas3/gemm.cpp
  blas3/gemm_batched.cpp
  blas3/trsm.cpp
)

# Add individual benchmarks for each method
foreach(syclblas_bench ${sources})
  get_filename_component(bench_exec ${syclblas_bench} NAME_WE)
  add_executable(bench_${bench_exec} ${syclblas_bench} main.cpp)
  target_link_libraries(bench_${bench_exec} PRIVATE benchmark Clara::Clara sycl_blas_common sycl_blas)
  add_sycl_to_target(
    TARGET bench_${bench_exec}
    SOURCES ${syclblas_bench}
  )
  target_include_directories(bench_${bench_exec} PRIVATE ${CBLAS_INCLUDE})

  if(BLAS_VERIFY_BENCHMARK)
    target_compile_definitions(bench_${bench_exec} PRIVATE BLAS_VERIFY_BENCHMARK)
    target_link_libraries(bench_${bench_exec} PRIVATE blas::blas)
  endif()

  message(STATUS "Created benchmark: ${bench_exec}")
  install(TARGETS bench_${bench_exec}
    RUNTIME
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT benchmarks
  )
endforeach()

# Expression benchmarks: use source, not Library
if(BUILD_EXPRESSION_BENCHMARKS)
  set(extensions expression/reduction_rows.cpp)

  foreach(syclblas_bench ${extensions})
    get_filename_component(bench_exec ${syclblas_bench} NAME_WE)
    add_executable(bench_${bench_exec} ${syclblas_bench} main.cpp)
    target_link_libraries(bench_${bench_exec} PUBLIC benchmark Clara::Clara sycl_blas_common sycl_blas)
    add_sycl_to_target(
      TARGET bench_${bench_exec}
      SOURCES ${syclblas_bench}
    )
    target_include_directories(bench_${bench_exec} PRIVATE ${SYCLBLAS_SRC} ${CBLAS_INCLUDE})

    if(BLAS_VERIFY_BENCHMARK)
      target_compile_definitions(bench_${bench_exec} PRIVATE BLAS_VERIFY_BENCHMARK)
      target_link_libraries(bench_${bench_exec} PRIVATE blas::blas)
    endif()

    message(STATUS "Created expression benchmark: ${bench_exec}")
    install(TARGETS bench_${bench_exec}
      RUNTIME
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT benchmarks
    )
  endforeach()
endif()
