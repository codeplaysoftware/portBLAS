cmake_minimum_required (VERSION 3.2.2)

project(sycl-blas)

enable_testing()

message(STATUS "Path to the SYCL implementation ")

set(SYCLBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

include(FindOpenCL)

include(FindComputeCpp)

include(FindBLAS)

if (DEFINED OPENBLAS_ROOT)
  set(USE_OPENBLAS ON)
else()
  set(USE_OPENBLAS OFF)
endif()

if(USE_OPENBLAS)
  message(STATUS "Will enable comparison with clBLAS")
  set(OPENBLAS_LIBRARIES "${OPENBLAS_ROOT}/lib/libopenblas.so")
  set(OPENBLAS_INCLUDE "${OPENBLAS_ROOT}/include/")
endif(USE_OPENBLAS)

message(STATUS "${COMPUTECPP_DEVICE_COMPILER_FLAGS}")

# We add some flags to workaround OpenCL platform bugs, see ComputeCpp documentation
set(COMPUTECPP_DEVICE_COMPILER_FLAGS
    "${COMPUTECPP_DEVICE_COMPILER_FLAGS} -no-serial-memop -sycl-compress-name")
message(STATUS "${COMPUTECPP_DEVICE_COMPILER_FLAGS}")

include_directories(${SYCLBLAS_INCLUDE} ${COMPUTECPP_INCLUDE_DIRECTORY})

if (USE_OPENBLAS)
  include_directories(${OPENBLAS_INCLUDE})
endif()


add_subdirectory(test)
