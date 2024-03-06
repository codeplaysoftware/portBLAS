cmake_minimum_required(VERSION 3.4.3)

project(portBLASSample LANGUAGES CXX)

set(PORTBLAS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)
set(PORTBLAS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)

include(ConfigurePORTBLAS)
include(SYCL)
find_package(PORTBLAS REQUIRED)

set(SAMPLES_LIST
  gemv.cpp
  gemm.cpp
  symm.cpp
)

foreach(src_file ${SAMPLES_LIST})
  get_filename_component(sample_exec ${src_file} NAME_WE)
  set(sample_exec "sample_${sample_exec}")
  add_executable(${sample_exec} ${src_file})
  if(is_adaptivecpp OR is_dpcpp)
    set_target_properties(${sample_exec} PROPERTIES CXX_STANDARD 17)
  else()
    set_target_properties(${sample_exec} PROPERTIES CXX_STANDARD 14)
  endif()
  target_compile_definitions(${sample_exec} PRIVATE -D${TUNING_TARGET})
  target_compile_definitions(${sample_exec} PUBLIC GEMM_VECTORIZATION_SUPPORT=1)

  add_sycl_to_target(
    TARGET ${sample_exec}
    SOURCES ${src_file}
  )
  target_link_libraries(${sample_exec} PRIVATE PORTBLAS::PORTBLAS)
  if(IMGDNN_DIR)
    target_link_libraries(${sample_exec} PRIVATE IMGDNN::IMGDNN)
  endif()
  install(TARGETS ${sample_exec} RUNTIME DESTINATION bin)
endforeach()
include_directories(${PORTBLAS_INCLUDE} ${SYCL_INCLUDE_DIRS} ${THIRD_PARTIES_INCLUDE})
