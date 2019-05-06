set(benchmarks
    asum
    axpy
    dot
    iamax
    iamin
    nrm2
    scal
)

foreach(b ${benchmarks})
    add_executable(benchmark-syclblas-blas1-${b} ${b}.cpp)
    target_link_libraries(benchmark-syclblas-blas1-${b} PRIVATE benchmark-syclblas-common)
endforeach()
