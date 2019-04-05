cmake_minimum_required(VERSION 3.2.2)

find_package(PkgConfig)

message(STATUS "Path to the SYCL implementation ")

set(COMPUTECPP_SDK_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../external/computecpp-sdk/include)
set(SYCLBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../external/computecpp-sdk/cmake/Modules)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src/)

find_package(OpenCL REQUIRED)
find_package(ComputeCpp REQUIRED)
include(ConfigureSYCLBLAS)

include(FindSystemBLAS)


# Add syclblas benchmarks.
file(GLOB SYCL_AUTO_TUNNER_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tune_nn.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tune_nt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tune_tn.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tune_tt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tune_all.cpp
)
foreach(blas_tuner ${SYCL_AUTO_TUNNER_SRCS})
  get_filename_component(tunner_exec ${blas_tuner} NAME_WE)
  set(TARGET tunner_exec ${blas_tuner})
  add_executable(${tunner_exec} ${blas_tuner})
  set_property(TARGET ${tunner_exec} PROPERTY CXX_STANDARD 11)
  message(${BLAS_LIBRARIES})
  target_link_libraries(${tunner_exec} PUBLIC m PUBLIC ${BLAS_LIBRARIES})
  add_sycl_to_target(
    TARGET ${tunner_exec}
    SOURCES ${blas_tuner}
  )
  install(TARGETS ${tunner_exec} RUNTIME DESTINATION bin)
endforeach()
include_directories(${SYCLBLAS_INCLUDE} ${ComputeCpp_INCLUDE_DIRS} ${COMPUTECPP_SDK_INCLUDE} ${BLAS_INCLUDE_DIRS})
