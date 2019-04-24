find_package(CLBlast REQUIRED)

add_executable(syclblas_benchmarks syclblas_benchmark.cpp)
target_link_libraries(syclblas_benchmarks PRIVATE codeplay::SYCL-BLAS clblast)
add_sycl_to_target(TARGET syclblas_benchmarks SOURCES syclblas_benchmark.cpp)
