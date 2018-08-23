cmake_minimum_required(VERSION 3.2.2)

find_package(PkgConfig)

message(STATUS "Path to the SYCL implementation ")

set(SYCLBLAS_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules")

include(FindOpenCL)
include(FindComputeCpp)

if (CLBLAST_FOUND)
  pkg_search_module(CLBLAST REQUIRED clblast)
  pkg_check_modules(CLBLAST clblast OPTIONAL)
  add_executable(clblast_benchmarks clblast_benchmark.cpp)
  set_property(TARGET clblast_benchmarks PROPERTY CXX_STANDARD 11)
  include_directories(${CLBLAST_INCLUDE_DIRS})
  target_link_libraries(clblast_benchmarks PUBLIC ${CLBLAST_LIBRARIES})
  target_link_libraries(clblast_benchmarks PUBLIC ${OpenCL_LIBRARIES})
endif(CLBLAST_FOUND)

if (CLBLAS_FOUND)
  pkg_search_module(CLBLAS REQUIRED clBLAS)
  pkg_check_modules(CLBLAS clBLAS OPTIONAL)
  add_executable(clblas_benchmarks clblas_benchmark.cpp)
  set_property(TARGET clblas_benchmarks PROPERTY CXX_STANDARD 11)
  include_directories(${CLBLAS_INCLUDE_DIRS})
  target_link_libraries(clblas_benchmarks ${CLBLAS_LIBRARIES})
  target_link_libraries(clblas_benchmarks ${OpenCL_LIBRARIES})
endif(CLBLAS_FOUND)

set(COMPUTECPP_DEVICE_COMPILER_FLAGS "${COMPUTECPP_DEVICE_COMPILER_FLAGS} -no-serial-memop -sycl-compress-name")
message(STATUS "${COMPUTECPP_DEVICE_COMPILER_FLAGS}")

include_directories(${SYCLBLAS_INCLUDE} ${COMPUTECPP_INCLUDE_DIRECTORY})

add_executable(syclblas_benchmarks_level1 syclblas/syclblas_benchmark_level1.cpp)
set_property(TARGET syclblas_benchmarks_level1 PROPERTY CXX_STANDARD 11)
add_sycl_to_target(syclblas_benchmarks_level1 ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/syclblas/syclblas_benchmark_level1.cpp)

add_executable(syclblas_benchmarks_level2 syclblas/syclblas_benchmark_level2.cpp)
set_property(TARGET syclblas_benchmarks_level2 PROPERTY CXX_STANDARD 11)
add_sycl_to_target(syclblas_benchmarks_level2 ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/syclblas/syclblas_benchmark_level2.cpp)


add_executable(syclblas_benchmarks_level3 syclblas/syclblas_benchmark_level3.cpp)
set_property(TARGET syclblas_benchmarks_level3 PROPERTY CXX_STANDARD 11)
add_sycl_to_target(syclblas_benchmarks_level3 ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/syclblas/syclblas_benchmark_level3.cpp)
