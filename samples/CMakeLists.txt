set(SyclBLAS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)
set(SyclBLAS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../external/computecpp-sdk/cmake/Modules)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)

include(ConfigureSYCLBLAS)
include(SYCL)
find_package(SyclBLAS REQUIRED)

set(SAMPLES_LIST
  gemv.cpp
  gemm.cpp
)

foreach(src_file ${SAMPLES_LIST})
  get_filename_component(sample_exec ${src_file} NAME_WE)
  set(sample_exec "sample_${sample_exec}")
  add_executable(${sample_exec} ${src_file})
  if(is_hipsycl)
    set_target_properties(${sample_exec} PROPERTIES CXX_STANDARD 17)
  elseif (WIN32)
    set_target_properties(${sample_exec} PROPERTIES CXX_STANDARD 14)
  else()
    set_target_properties(${sample_exec} PROPERTIES CXX_STANDARD 11)
  endif()
  add_sycl_to_target(
    TARGET ${sample_exec}
    SOURCES ${src_file}
  )
  target_link_libraries(${sample_exec} PRIVATE SyclBLAS::SyclBLAS)
  install(TARGETS ${sample_exec} RUNTIME DESTINATION bin)
endforeach()
include_directories(${SYCLBLAS_INCLUDE} ${SYCL_INCLUDE_DIRS} ${COMPUTECPP_SDK_INCLUDE})
