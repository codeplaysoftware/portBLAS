cmake_minimum_required(VERSION 3.2.2)

find_package(PkgConfig)

message(STATUS "Path to the SYCL implementation ")

set(SYCLBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules)

include(FindOpenCL)
include(FindComputeCpp)
include(ConfigureSYCLBLAS)

# Add syclblas benchmarks.
file(GLOB SYCL_BENCHMARK_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/syclblas/syclblas_benchmark_level1.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/syclblas/syclblas_benchmark_level2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/syclblas/syclblas_benchmark_level2_jose.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/syclblas/syclblas_benchmark_level3.cpp
)
foreach(syclblas_benchmark ${SYCL_BENCHMARK_SRCS})
  get_filename_component(sycl_bench_exec ${syclblas_benchmark} NAME_WE)
  set(TARGET sycl_bench_exec ${syclblas_benchmark})
  add_executable(${sycl_bench_exec} ${syclblas_benchmark})
  set_property(TARGET ${sycl_bench_exec} PROPERTY CXX_STANDARD 11)
  add_sycl_to_target(
    TARGET ${sycl_bench_exec}
    SOURCES ${syclblas_benchmark}
  )
endforeach()
include_directories(${SYCLBLAS_INCLUDE} ${ComputeCpp_INCLUDE_DIRS})

# Try and also build CLBlast benchmarks.
# Check to see if we've been passed an explicit path. 
if (DEFINED CLBLAST_ROOT) 
  message(STATUS "CLBlast root specified as : ${CLBLAST_ROOT}")
  set(CMAKE_PREFIX_PATH ${CLBLAST_ROOT})
else() 
  # set a default path if not specified
  message(STATUS "CLBlast root not specified by user, searching default path /usr/lib/clblast")
  set(CMAKE_PREFIX_PATH "/usr/lib/clblast")
endif()
pkg_search_module(CLBLAST required clblast)
pkg_check_modules(CLBLAST clblast)

# If found, add the clblast benchmarks.
if (CLBLAST_FOUND)
  message(STATUS "CLBlast found!")
  message(STATUS "CLBlast headers: ${CLBLAST_INCLUDE_DIRS}")
  message(STATUS "CLBlast libraries: ${CLBLAST_LIBRARY_DIRS}")

  include_directories(${CLBLAST_INCLUDE_DIRS})
  link_directories(${CLBLAST_LIBRARY_DIRS})

  file(GLOB CLBLAST_BENCHMARK_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/clblast/clblast_benchmark_level1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/clblast/clblast_benchmark_level2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/clblast/clblast_benchmark_level3.cpp
  )

  foreach(clblast_benchmark ${CLBLAST_BENCHMARK_SRCS})
    get_filename_component(clblast_bench_exec ${clblast_benchmark} NAME_WE)
    set(TARGET clblast_bench_exec ${clblast_benchmark})
    add_executable(${clblast_bench_exec} ${clblast_benchmark})
    set_property(TARGET ${clblast_bench_exec} PROPERTY CXX_STANDARD 11)
    target_link_libraries(${clblast_bench_exec} PUBLIC ${CLBLAST_LIBRARIES})
    target_link_libraries(${clblast_bench_exec} PUBLIC ${OpenCL_LIBRARIES})
  endforeach()
else()
  message(STATUS "CLBlast not found, only building benchmarks for syclblas")
endif()
