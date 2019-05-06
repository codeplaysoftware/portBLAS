add_library(benchmark-common INTERFACE)
target_include_directories(benchmark-common INTERFACE include)
target_link_libraries(benchmark-common INTERFACE Clara::Clara benchmark::benchmark codeplay::sycl-blas-header-only)
