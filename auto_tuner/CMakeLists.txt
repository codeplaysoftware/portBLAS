cmake_minimum_required(VERSION 3.2.2)

find_package(PkgConfig)

message(STATUS "Path to the SYCL implementation ")

set(COMPUTECPP_SDK_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../external/computecpp-sdk/include)
set(SYCLBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../external/computecpp-sdk/cmake/Modules)

find_package(OpenCL REQUIRED)
find_package(ComputeCpp REQUIRED)
include(ConfigureSYCLBLAS)

include(FindBLAS)

if (DEFINED OPENBLAS_ROOT)
  # If OPENBLAS_ROOT is defined, then use it explicitly, and set the BLAS paths and
  # libraries based on the explicit path given 
  message(STATUS "Using explicit OpenBLAS installation path for unit tests")
  set(BLAS_LIBRARIES "${OPENBLAS_ROOT}/lib/libopenblas.so")
  set(BLAS_INCLUDE_DIRS "${OPENBLAS_ROOT}/include/")
else()
  message(STATUS "Using Cmake FindBLAS to locate a BLAS library for unit tests")
  set(BLA_STATIC on)
  # If we want to use a specific BLAS vendor, we could set it here:
  # by calling: set(BLA_VENDOR OpenBLAS) 
  find_package(BLAS REQUIRED) # We need BLAS for the tests - require it
  message(STATUS "Found BLAS library at: ${BLAS_LIBRARIES}")
endif()


# Add syclblas benchmarks.
file(GLOB SYCL_AUTO_TUNNER_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/tune_nn.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tune_nt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tune_tn.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tune_tt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tune_all.cpp
)
foreach(blas_tunner ${SYCL_AUTO_TUNNER_SRCS})
  get_filename_component(tunner_exec ${blas_tunner} NAME_WE)
  set(TARGET tunner_exec ${blas_tunner})
  add_executable(${tunner_exec} ${blas_tunner})
  set_property(TARGET ${tunner_exec} PROPERTY CXX_STANDARD 11)
  target_link_libraries(${tunner_exec} PUBLIC ${BLAS_LIBRARIES})
  add_sycl_to_target(
    TARGET ${tunner_exec}
    SOURCES ${blas_tunner}
  )
  install(TARGETS ${tunner_exec} RUNTIME DESTINATION bin)
endforeach()
include_directories(${SYCLBLAS_INCLUDE} ${ComputeCpp_INCLUDE_DIRS} ${COMPUTECPP_SDK_INCLUDE} ${BLAS_INCLUDE_DIRS})