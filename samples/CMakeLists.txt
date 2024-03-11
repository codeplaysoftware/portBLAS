#/***************************************************************************
# *
# *  @license
# *  Copyright (C) Codeplay Software Limited
# *  Licensed under the Apache License, Version 2.0 (the "License");
# *  you may not use this file except in compliance with the License.
# *  You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# *  For your convenience, a copy of the License has been included in this
# *  repository.
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# *
# *  portBLAS: BLAS implementation using SYCL
# *
# *  @filename CMakeLists.txt
# *
# **************************************************************************/
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
