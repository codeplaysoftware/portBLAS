set(SyclBLAS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)
set(SyclBLAS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../external/computecpp-sdk/cmake/Modules)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)

find_package(ComputeCpp REQUIRED)
find_package(SyclBLAS REQUIRED)
include(ConfigureSYCLBLAS)

set(SAMPLES_LIST
  gemv.cpp
  gemm.cpp
)

foreach(src_file ${SAMPLES_LIST})
  get_filename_component(sample_exec ${src_file} NAME_WE)
  set(sample_exec "sample_${sample_exec}")
  add_executable(${sample_exec} ${src_file})
  set_property(TARGET ${sample_exec} PROPERTY CXX_STANDARD 11)
  add_sycl_to_target(
    TARGET ${sample_exec}
    SOURCES ${src_file}
  )
  target_link_libraries(${sample_exec} PUBLIC SyclBLAS::SyclBLAS)
  install(TARGETS ${sample_exec} RUNTIME DESTINATION bin)
endforeach()
include_directories(${SYCLBLAS_INCLUDE} ${ComputeCpp_INCLUDE_DIRS} ${COMPUTECPP_SDK_INCLUDE})
