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
    add_executable(benchmark-clblast-blas1-${b} ${b}.cpp)
    target_link_libraries(benchmark-clblast-blas1-${b} PRIVATE benchmark-clblast-common)
endforeach()
