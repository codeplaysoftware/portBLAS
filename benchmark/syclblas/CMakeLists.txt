set(SYCLBLAS_BENCH ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${SYCLBLAS_BENCH})

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
  ${SYCLBLAS_BENCH}/blas3/gemm_batched.cpp
)

# Add individual benchmarks for each method
foreach(syclblas_bench ${SYCLBLAS_BENCH_SRCS})
  get_filename_component(bench_exec ${syclblas_bench} NAME_WE)
  add_executable(bench_${bench_exec} ${syclblas_bench} ${SYCLBLAS_BENCH}/main.cpp)
  target_link_libraries(bench_${bench_exec} PRIVATE benchmark ${BLAS_LIBRARIES} sycl_blas)
  add_sycl_to_target(
    TARGET bench_${bench_exec}
    SOURCES ${syclblas_bench}
  )
  target_include_directories(bench_${bench_exec} PRIVATE ${CLARA_INCLUDE} ${CBLAS_INCLUDE})

  message("-- Created benchmark: ${bench_exec}")
  install(TARGETS bench_${bench_exec}
    RUNTIME
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT benchmarks
  )
endforeach()

# Expression benchmarks: use source, not Library
if(${BUILD_EXPRESSION_BENCHMARKS})
  set(SYCLBLAS_EXPR_BENCH_SRCS
    # Extensions
    ${SYCLBLAS_BENCH}/expression/reduction_rows.cpp
  )

  foreach(syclblas_bench ${SYCLBLAS_EXPR_BENCH_SRCS})
    get_filename_component(bench_exec ${syclblas_bench} NAME_WE)
    add_executable(bench_${bench_exec} ${syclblas_bench} ${SYCLBLAS_BENCH}/main.cpp)
    target_link_libraries(bench_${bench_exec} PUBLIC benchmark ${BLAS_LIBRARIES} sycl_blas)
    add_sycl_to_target(
      TARGET bench_${bench_exec}
      SOURCES ${syclblas_bench}
    )
    target_include_directories(bench_${bench_exec} PRIVATE ${SYCLBLAS_SRC} ${CLARA_INCLUDE} ${CBLAS_INCLUDE})

    message("-- Created expression benchmark: ${bench_exec}")
    install(TARGETS bench_${bench_exec}
      RUNTIME
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT benchmarks
    )
  endforeach()
endif()
