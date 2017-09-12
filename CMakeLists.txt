cmake_minimum_required (VERSION 3.2.2)

project(sycl-blas)

enable_testing()

message(STATUS "Path to the SYCL implementation ")

set(SYCLBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

include(FindOpenCL)

include(FindComputeCpp)

include(FindBLAS)

if (DEFINED CLBLAS_ROOT)
  set(USE_CLBLAS ON)
else()
  set(USE_CLBLAS OFF)
endif()

if(USE_CLBLAS)
  message(STATUS "Will enable comparison with clBLAS")
  set(CLBLAS_LIBRARIES "${CLBLAS_ROOT}/lib64/libclBLAS.so")
  set(CLBLAS_INCLUDE "${CLBLAS_ROOT}/include/")
endif(USE_CLBLAS)

message(STATUS "${COMPUTECPP_DEVICE_COMPILER_FLAGS}")

# We add some flags to workaround OpenCL platform bugs, see ComputeCpp documentation
set(COMPUTECPP_DEVICE_COMPILER_FLAGS
    "${COMPUTECPP_DEVICE_COMPILER_FLAGS} -no-serial-memop -sycl-compress-name")
message(STATUS "${COMPUTECPP_DEVICE_COMPILER_FLAGS}")

include_directories(${SYCLBLAS_INCLUDE} ${COMPUTECPP_INCLUDE_DIRECTORY})

if (USE_CLBLAS)
  include_directories(${CLBLAS_INCLUDE})
endif()


add_subdirectory(test)
add_subdirectory(tests)
