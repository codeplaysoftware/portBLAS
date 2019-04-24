add_library(benchmark_common INTERFACE)
target_include_directories(benchmark_common INTERFACE include)
target_link_libraries(benchmark_common INTERFACE Clara::Clara benchmark::benchmark)
