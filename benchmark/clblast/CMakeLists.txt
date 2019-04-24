find_package(CLBlast REQUIRED)

set(sources
    blas1/asum.cpp
    blas1/axpy.cpp
    blas1/dot.cpp
    blas1/iamax.cpp
    blas1/iamin.cpp
    blas1/nrm2.cpp
    blas1/scal.cpp

    blas2/gemv.cpp

    blas3/gemm.cpp

    main.cpp
)

add_executable(bench_clblast ${sources})
add_sycl_to_target(TARGET bench_clblast SOURCES ${sources})
target_link_libraries(bench_clblast PRIVATE clblast benchmark_common codeplay::sycl-blas-header-only)
target_include_directories(bench_clblast PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
