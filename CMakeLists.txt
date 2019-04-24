cmake_minimum_required(VERSION 3.4.3)
project(SYCL-BLAS LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include("${CMAKE_BINARY_DIR}/conan_paths.cmake" OPTIONAL)
include(CTest)

option(BUILD_BENCHMARKS "Build the benchmarks" OFF)
option(BUILD_DOCUMENTATION "Build the documentation" OFF)

find_package(ComputeCpp REQUIRED)
# We add some flags to workaround OpenCL platform bugs, see ComputeCpp documentation
list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -no-serial-memop -sycl-compress-name)

add_library(SYCL-BLAS INTERFACE)
add_library(codeplay::SYCL-BLAS ALIAS SYCL-BLAS)
target_include_directories(SYCL-BLAS INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()
if(BUILD_TESTING)
    add_subdirectory(test)
endif()


install(TARGETS SYCL-BLAS EXPORT SYCLBlasTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

export(EXPORT SYCLBlasTargets
  FILE "${PROJECT_BINARY_DIR}/sycl-blas-targets.cmake"
  NAMESPACE codeplay::)
configure_file(cmake/sycl-blas-config.cmake
    "${PROJECT_BINARY_DIR}/sycl-blas-config.cmake"
    COPYONLY)

set(config_package_location lib/cmake/SYCL-BLAS)
install(DIRECTORY include
	    DESTINATION .
		COMPONENT Devel)
install(EXPORT SYCLBlasTargets
  FILE sycl-blas-targets.cmake
  NAMESPACE codeplay::
  DESTINATION ${config_package_location})
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/sycl-blas-config.cmake
  DESTINATION ${config_package_location}
  COMPONENT Devel)
