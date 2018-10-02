cmake_minimum_required(VERSION 3.2.2)

find_package(PkgConfig)

message(STATUS "Path to the SYCL implementation ")

set(SYCLBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules)

include(FindOpenCL)
include(FindComputeCpp)
include(ConfigureSYCLBLAS)

# Manually search for CLBLAST
if (DEFINED CLBLAST_ROOT) 
  message(STATUS "CLBlast root specified as : ${CLBLAST_ROOT}")
  set(CMAKE_PREFIX_PATH ${CLBLAST_ROOT})
else() 
  message(STATUS "Looking for CLBlast at : /usr/lib/clblast")
  set(CMAKE_PREFIX_PATH "/usr/lib/clblast")
endif()

pkg_search_module(CLBLAST required clblast)
pkg_check_modules(CLBLAST clblast)

if (CLBLAST_FOUND)
  message(STATUS "CLBlast found!")
  message(STATUS "CLBlast libraries found at: ${CLBLAST_LIBRARIES}")
  message(STATUS "CLBlast headers found at: ${CLBLAST_INCLUDE_DIRS}")
  message(STATUS "CLBlast libraries found at: ${CLBLAST_LIBRARY_DIRS}")
  message(STATUS "CLBLAST_LIBRARIES     found at: ${CLBLAST_LIBRARIES}")
  message(STATUS "CLBLAST_LIBRARY_DIRS  found at: ${CLBLAST_LIBRARY_DIRS}")
  message(STATUS "CLBLAST_LDFLAGS       found at: ${CLBLAST_LDFLAGS}")
  message(STATUS "CLBLAST_INCLUDE_DIRS  found at: ${CLBLAST_INCLUDE_DIRS}")
  message(STATUS "CLBLAST_CFLAGS        found at: ${CLBLAST_CFLAGS}")
  message(STATUS "CLBLAST_CFLAGS_OTHER  found at: ${CLBLAST_CFLAGS_OTHER}")

  include_directories(${CLBLAST_INCLUDE_DIRS})
  link_directories(${CLBLAST_LIBRARY_DIRS})
  
  add_executable(clblast_benchmark_level1 clblast/clblast_benchmark_level1.cpp)
  set_property(TARGET clblast_benchmark_level1 PROPERTY CXX_STANDARD 11)
  
  target_link_libraries(clblast_benchmark_level1 PUBLIC ${CLBLAST_LIBRARIES})
  target_link_libraries(clblast_benchmark_level1 PUBLIC ${OpenCL_LIBRARIES})
endif()

# Manually search for CLBLAS_ROOT
if (DEFINED CLBLAS_ROOT) 
  message(STATUS "CLBlas root specified as : ${CLBLAS_ROOT}")
  set(CMAKE_PREFIX_PATH ${CLBLAS_ROOT})
else ()
message(STATUS "Looking for CLBlas at : /usr/lib/clblas/lib64/cmake/clBLAS")
  set(CMAKE_PREFIX_PATH "/usr/lib/clblas/lib64/cmake/clBLAS")
endif()

# clBLAS needs to be found using find_package, as it's cmake doesn't support pkg_search_module
find_package(clBLAS REQUIRED)

if (clBLAS_FOUND)
  message(STATUS "CLBlas found!")
  message(STATUS "CLBlas libraries found at: ${CLBLAS_LIBRARIES}")
  message(STATUS "CLBlas headers found at: ${CLBLAS_INCLUDE_DIRS}")
  message(STATUS "CLBLAS_LIBRARIES     found at: ${CLBLAS_LIBRARIES}")
  message(STATUS "CLBLAS_INCLUDE_DIRS  found at: ${CLBLAS_INCLUDE_DIRS}")
  # add_executable(clblas_benchmarks clblas_benchmark.cpp)
  # set_property(TARGET clblas_benchmarks PROPERTY CXX_STANDARD 11)
  # include_directories(${CLBLAS_INCLUDE_DIRS})
  # target_link_libraries(clblas_benchmarks ${CLBLAS_LIBRARIES})
  # target_link_libraries(clblas_benchmarks ${OpenCL_LIBRARIES})
endif()

add_executable(syclblas_benchmarks_level1 syclblas/syclblas_benchmark_level1.cpp)
set_property(TARGET syclblas_benchmarks_level1 PROPERTY CXX_STANDARD 11)
add_sycl_to_target(
    TARGET syclblas_benchmarks_level1
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/syclblas/syclblas_benchmark_level1.cpp
)

add_executable(syclblas_benchmarks_level2 syclblas/syclblas_benchmark_level2.cpp)
set_property(TARGET syclblas_benchmarks_level2 PROPERTY CXX_STANDARD 11)
add_sycl_to_target(
    TARGET syclblas_benchmarks_level2
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/syclblas/syclblas_benchmark_level2.cpp
)

add_executable(syclblas_benchmarks_level3 syclblas/syclblas_benchmark_level3.cpp)
set_property(TARGET syclblas_benchmarks_level3 PROPERTY CXX_STANDARD 11)
add_sycl_to_target(
    TARGET syclblas_benchmarks_level3
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/syclblas/syclblas_benchmark_level3.cpp
)

include_directories(${SYCLBLAS_INCLUDE} ${ComputeCpp_INCLUDE_DIRS})
