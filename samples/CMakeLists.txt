cmake_minimum_required(VERSION 3.4.3)

set(COMPUTECPP_SDK_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../external/computecpp-sdk/include)
set(SYCLBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../external/computecpp-sdk/cmake/Modules)

find_package(OpenCL REQUIRED)
find_package(ComputeCpp REQUIRED)
find_package(SyclBLAS REQUIRED)
include(ConfigureSYCLBLAS)

file(GLOB SRC_LIST
  ${CMAKE_CURRENT_SOURCE_DIR}/gemv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm.cpp
)
foreach(src_file ${SRC_LIST})
  get_filename_component(sample_exec ${src_file} NAME_WE)
  set(TARGET sample_exec ${src_file})
  add_executable(${sample_exec} ${src_file})
  set_property(TARGET ${sample_exec} PROPERTY CXX_STANDARD 11)
  add_sycl_to_target(
    TARGET ${sample_exec}
    SOURCES ${src_file}
  )
  target_link_libraries(${sample_exec} PUBLIC SyclBLAS::SyclBLAS )
  install(TARGETS ${sample_exec} RUNTIME DESTINATION bin)
endforeach()
include_directories(${SYCLBLAS_INCLUDE} ${ComputeCpp_INCLUDE_DIRS} ${COMPUTECPP_SDK_INCLUDE})
