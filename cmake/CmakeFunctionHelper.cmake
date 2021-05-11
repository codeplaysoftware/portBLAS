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
# *  SYCL-BLAS: BLAS implementation using SYCL
# *
# *  @filename CmakeFUnctionHelper.cmake
# *
# **************************************************************************/
# represent the list of supported handler for executor
set(executor_list "PolicyHandler<codeplay_policy>")
#represent the list of supported index/increment type
set(index_list "int" )

# BLAS_DATA_TYPES was provided by the user
#Each data type in a data list determines the container types.
#The container type for SYCLbackend is BufferIterator<${data}, codeplay_policy>
set(data_list "${BLAS_DATA_TYPES}")

# Converts a user specified type name into a C++ type
function(cpp_type output data)
  if (${data} STREQUAL "half")
    set(${output} "cl::sycl::half" PARENT_SCOPE)
    return()
  endif()
  set(${output} "${data}" PARENT_SCOPE)
endfunction()

## represent the list of bolean options
set(boolean_list "true" "false")

# Cleans up the proposed file name so that it can be used in the file system
function(sanitize_file_name output file_name)
  string(REGEX REPLACE "(:|\\*|<| |,|>)" "_" file_name ${file_name})
  string(REGEX REPLACE "(_____|____|___|__)" "_" file_name ${file_name})
  if (SYCLBLAS_USE_SHORT_NAMES)
    # Long paths are problematic on Windows and WSL so we hash the filename
    # to reduce its size
    string(MD5 file_name ${file_name})
    set(file_name ${file_name}.cpp)
  endif()
  set(${output} "${file_name}" PARENT_SCOPE)
endfunction()

function(set_target_compile_def in_target)
  #setting compiler flag for backend
  message(STATUS "Adding ${BACKEND_DEVICE} backend to target ${in_target}")
  if(${BACKEND_DEVICE} STREQUAL "INTEL_GPU")
    target_compile_definitions(${in_target} PUBLIC INTEL_GPU=1)
  elseif(${BACKEND_DEVICE} STREQUAL "AMD_GPU")
    target_compile_definitions(${in_target} PUBLIC AMD_GPU=1)
  elseif(${BACKEND_DEVICE} STREQUAL "ARM_GPU")
    target_compile_definitions(${in_target} PUBLIC ARM_GPU=1)
  elseif(${BACKEND_DEVICE} STREQUAL "RCAR")
    target_compile_definitions(${in_target} PUBLIC RCAR=1)
  elseif(${BACKEND_DEVICE} STREQUAL "POWER_VR")
    target_compile_definitions(${in_target} PUBLIC POWER_VR=1)
  elseif(${BACKEND_DEVICE} STREQUAL "NVIDIA_GPU")
    target_compile_definitions(${in_target} PUBLIC NVIDIA_GPU=1)
  else()
    target_compile_definitions(${in_target} PUBLIC DEFAULT_CPU=1)
  endif()
  #setting tall skinny support
  if(${GEMM_TALL_SKINNY_SUPPORT})
    message(STATUS "Tall and skinny Gemm support enabled for target ${in_target}")
    target_compile_definitions(${in_target} PUBLIC GEMM_TALL_SKINNY_SUPPORT=1)
  endif()
  #setting vectorization support
  if(${GEMM_VECTORIZATION_SUPPORT})
    message(STATUS "Gemm vectorization support enabled for target ${in_target}")
    target_compile_definitions(${in_target} PUBLIC GEMM_VECTORIZATION_SUPPORT=1)
  endif()
  #Set optimized model configs
  if(${BLAS_MODEL_OPTIMIZATION} STREQUAL "RESNET_50")
    target_compile_definitions(${in_target} PUBLIC MODEL_RESNET_50=1)
  elseif(${BLAS_MODEL_OPTIMIZATION} STREQUAL "VGG_16")
    target_compile_definitions(${in_target} PUBLIC MODEL_VGG_16=1)
  endif()
endfunction()


# blas unary function for generating source code
function(generate_blas_unary_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
foreach(executor ${executor_list})
  foreach(data ${data_list})
    cpp_type(cpp_data ${data})
    set(container_list "BufferIterator<${cpp_data},codeplay_policy>")
    foreach(index ${index_list})
      foreach(container0 ${container_list})
        foreach(increment ${index_list})
          sanitize_file_name(file_name
            "${func}_${executor}_${data}_${index}_${container0}_${increment}.cpp")
          add_custom_command(OUTPUT "${LOCATION}/${file_name}"
            COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_unary.py
              ${PROJECT_SOURCE_DIR}/external/
              ${SYCLBLAS_SRC_GENERATOR}/gen
              ${blas_level}
              ${func}
              ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
              ${executor}
              ${cpp_data}
              ${index}
              ${increment}
              ${container0}
              ${file_name}
            MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
            DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_unary.py
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            VERBATIM
          )
          list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
        endforeach(increment)
      endforeach(container0)
    endforeach(index)
  endforeach(data)
endforeach(executor)
add_library(${func} OBJECT ${FUNC_SRC})
set_target_compile_def(${func})
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE}
                           ${SYCLBLAS_COMMON_INCLUDE_DIR} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_unary_objects)


# blas binary function for generating source code
function(generate_blas_binary_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
foreach(executor ${executor_list})
  foreach(data ${data_list})
    cpp_type(cpp_data ${data})
    set(container_list "BufferIterator<${cpp_data},codeplay_policy>")
    foreach(index ${index_list})
      foreach(container0 ${container_list})
        foreach(container1 ${container_list})
          set(container_names "${container0}_${container1}")
          foreach(increment ${index_list})
            sanitize_file_name(file_name
              "${func}_${executor}_${data}_${index}_${container_names}_${increment}.cpp")
            add_custom_command(OUTPUT "${LOCATION}/${file_name}"
              COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_binary.py
                ${PROJECT_SOURCE_DIR}/external/
                ${SYCLBLAS_SRC_GENERATOR}/gen
                ${blas_level}
                ${func}
                ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                ${executor}
                ${cpp_data}
                ${index}
                ${increment}
                ${container0}
                ${container1}
                ${file_name}
              MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
              DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_binary.py
              WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
              VERBATIM
            )
            list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
            endforeach(increment)
        endforeach(container1)
      endforeach(container0)
    endforeach(index)
  endforeach(data)
endforeach(executor)
add_library(${func} OBJECT ${FUNC_SRC})
set_target_compile_def(${func})
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE}
                           ${SYCLBLAS_COMMON_INCLUDE_DIR} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_binary_objects)



# blas special binary function for generating source code
function(generate_blas_binary_special_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
foreach(executor ${executor_list})
  foreach(data ${data_list})
    cpp_type(cpp_data ${data})
    set(container_list_in "BufferIterator<${cpp_data},codeplay_policy>")
    foreach(index ${index_list})
      set(container_list_out
        "BufferIterator<IndexValueTuple<${index},${cpp_data}>,codeplay_policy>")
      foreach(container0 ${container_list_in})
        foreach(container1 ${container_list_out})
          set(container_names "${container0}_${container1}")
          foreach(increment ${index_list})
            sanitize_file_name(file_name
              "${func}_${executor}_${data}_${index}_${container_names}_${increment}.cpp")
            add_custom_command(OUTPUT "${LOCATION}/${file_name}"
              COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_binary_special.py
                ${PROJECT_SOURCE_DIR}/external/
                ${SYCLBLAS_SRC_GENERATOR}/gen
                ${blas_level}
                ${func}
                ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                ${executor}
                ${cpp_data}
                ${index}
                ${increment}
                ${container0}
                ${container1}
                ${file_name}
              MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
              DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_binary_special.py
              WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
              VERBATIM
            )
            list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
            endforeach(increment)
        endforeach(container1)
      endforeach(container0)
    endforeach(index)
  endforeach(data)
endforeach(executor)
add_library(${func} OBJECT ${FUNC_SRC})
set_target_compile_def(${func})
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE}
                           ${SYCLBLAS_COMMON_INCLUDE_DIR} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_binary_special_objects)



# blas ternary function for generating source code
function(generate_blas_ternary_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
foreach(executor ${executor_list})
  foreach(data ${data_list})
    cpp_type(cpp_data ${data})
    set(container_list "BufferIterator<${cpp_data},codeplay_policy>")
    foreach(index ${index_list})
      foreach(container0 ${container_list})
        foreach(container1 ${container_list})
          foreach(container2 ${container_list})
            set(container_names
              "${container0}_${container1}_${container2}")
            foreach(increment ${index_list})
              sanitize_file_name(file_name
                "${func}_${executor}_${data}_${index}_${container_names}_${increment}.cpp")
              add_custom_command(OUTPUT "${LOCATION}/${file_name}"
                COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_ternary.py
                  ${PROJECT_SOURCE_DIR}/external/
                  ${SYCLBLAS_SRC_GENERATOR}/gen
                  ${blas_level}
                  ${func}
                  ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                  ${executor}
                  ${cpp_data}
                  ${index}
                  ${increment}
                  ${container0}
                  ${container1}
                  ${container2}
                  ${file_name}
                MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_ternary.py
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                VERBATIM
              )
              list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
            endforeach(increment)
          endforeach(container2)
        endforeach(container1)
      endforeach(container0)
    endforeach(index)
  endforeach(data)
endforeach(executor)
add_library(${func} OBJECT ${FUNC_SRC})
set_target_compile_def(${func})
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE}
                           ${SYCLBLAS_COMMON_INCLUDE_DIR} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_ternary_objects)


# blas gemm function for generating source code
function(generate_blas_gemm_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
set(gemm_sources "")

# Generates a file for a new GEMM configuration
# Adds the file to gemm_sources
# If the configuration is not supported by the current settings
# (e.g. double type not enabled), it's ignored
function(add_gemm_configuration
  data
  wg_size
  double_buffer
  conflict_a
  conflict_b
  cache_line_size
  tir
  tic
  twr
  twc
  tsr
  tsc
  tlr
  tlc
  item_batch wg_batch
  gemm_memory_type
  gemm_shape_type
  gemm_vectorize_type
  vector_size
  batch_type
)
  if(NOT ("${data}" IN_LIST data_list))
    # Data type not enabled, skip configuration
    return()
  endif()
  if(("${gemm_shape_type}" STREQUAL "tall_skinny") AND NOT GEMM_TALL_SKINNY_SUPPORT)
    # Tall/skinny configurations not enabled, skip
    return()
  endif()
  cpp_type(cpp_data ${data})
  foreach(trans_a ${boolean_list})
    foreach(trans_b ${boolean_list})
      foreach(is_beta_zero ${boolean_list})
        foreach(executor ${executor_list})
            foreach(index ${index_list})
              set(file_name "${func}_${double_buffer}_${conflict_a}_"
                            "${conflict_b}_${trans_a}_${trans_b}_"
                            "${is_beta_zero}_${gemm_memory_type}_"
                            "${gemm_shape_type}_${gemm_vectorize_type}_"
                            "${vector_size}_${batch_type}_${executor}_"
                            "${data}_${index}_${tir}_${tic}_${twr}_"
                            "${twc}_${tsr}_${tsc}_${tlr}_${tlc}_"
                            "${item_batch}_${wg_batch}_"
                            "${wg_size}_${cache_line_size}.cpp")
              sanitize_file_name(file_name "${file_name}")
              add_custom_command(OUTPUT "${LOCATION}/${file_name}"
                COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_gemm_launcher.py
                  ${PROJECT_SOURCE_DIR}/external/
                  ${SYCLBLAS_SRC_GENERATOR}/gen
                  ${blas_level}
                  ${func}
                  ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                  ${executor}
                  ${cpp_data}
                  ${index}
                  ${double_buffer}
                  ${conflict_a}
                  ${conflict_b}
                  ${trans_a}
                  ${trans_b}
                  ${is_beta_zero}
                  ${gemm_memory_type}
                  ${gemm_shape_type}
                  ${tir}
                  ${tic}
                  ${twr}
                  ${twc}
                  ${tsr}
                  ${tsc}
                  ${tlr}
                  ${tlc}
                  ${item_batch}
                  ${wg_batch}
                  ${wg_size}
                  ${cache_line_size}
                  ${file_name}
                  ${gemm_vectorize_type}
                  ${vector_size}
                  ${batch_type}
                MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_gemm_launcher.py
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                VERBATIM
              )
              list(APPEND gemm_sources "${LOCATION}/${file_name}")
              set(gemm_sources "${gemm_sources}" PARENT_SCOPE)
            endforeach(index)
        endforeach(executor)
      endforeach(is_beta_zero)
    endforeach(trans_b)
  endforeach(trans_a)
endfunction()
if(${TARGET} STREQUAL "INTEL_GPU")
  set(supported_types
    "float"
    "double"
    "half"
  )
  foreach(data ${supported_types})
    add_gemm_configuration(
      "${data}" 64 "true" "false" "false"
      64 4 4 8 8 1 1 1 1 1 1 "local" "standard" "full" 4 "strided")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 8 8 8 8 1 1 1 1 1 1 "local" "standard" "full" 4 "strided")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 8 8 8 8 1 1 1 1 1 1 "no_local" "standard" "partial" 4 "strided")

    if (${data} STREQUAL "half")
      add_gemm_configuration(
         "${data}" 16 "true" "false" "false"
         64 1 1 8 8 1 1 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
      add_gemm_configuration(
        "${data}" 16 "true" "false" "false"
         64 2 2 8 8 1 1 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
    else()
      add_gemm_configuration(
         "${data}" 16 "true" "false" "false"
         64 1 1 4 4 1 1 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
      add_gemm_configuration(
        "${data}" 16 "true" "false" "false"
         64 2 2 4 4 1 1 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
    endif()

    add_gemm_configuration(
      "${data}" 64 "true" "true" "true"
      64 2 2 8 8 1 1 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
    add_gemm_configuration(
      "${data}" 64 "true" "true" "true"
      64 4 4 8 8 1 1 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")

    if (${data} STREQUAL "double")
      add_gemm_configuration(
        "${data}" 256 "true" "true" "true"
        64 4 4 8 8 1 1 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
    else()
      add_gemm_configuration(
        "${data}" 256 "true" "true" "true"
        64 4 4 16 16 1 1 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
    endif()

    add_gemm_configuration(
      "${data}" 32 "true" "true" "true"
      64 2 1 8 4 1 1 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")
    add_gemm_configuration(
      "${data}" 32 "true" "true" "true"
      64 2 2 8 4 1 1 1 1 1 1 "local" "tall_skinny" "none" 4 "strided")

    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 4 4 4 1 1 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")
  endforeach()
elseif(${TARGET} STREQUAL "RCAR") # need investigation
  set(supported_types
    "float"
  )
  foreach(data ${supported_types})
    add_gemm_configuration(
      "${data}" 32 "false" "false" "false"
      128 4 8 8 4 1 1 1 1 1 1 "local" "standard" "full" 4 "strided")
    add_gemm_configuration(
      "${data}" 32 "false" "false" "false"
      128 8 4 4 8 1 1 1 1 1 1 "local" "standard" "full" 4 "strided")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 4 4 4 1 1 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")
  endforeach()
elseif(${TARGET} STREQUAL "ARM_GPU")
  set(supported_types
    "float"
    "half"
  )
  foreach(data ${supported_types})
    if(${BLAS_MODEL_OPTIMIZATION} STREQUAL "RESNET_50")
      add_gemm_configuration(
        "${data}" 64 "false" "false" "false"
        64 4 4 8 8 1 1 1 1 1 1 "no_local" "standard" "full" 4 "strided")
      add_gemm_configuration(
        "${data}" 32 "false" "false" "false"
        64 8 4 4 8 1 1 1 1 1 1 "no_local" "standard" "partial" 4 "strided")
      add_gemm_configuration(
        "${data}" 32 "false" "false" "false"
        64 4 8 8 4 1 1 1 1 1 1 "no_local" "standard" "partial" 4 "strided")
      add_gemm_configuration(
        "${data}" 64 "false" "false" "false"
        64 4 4 8 8 1 1 1 1 1 1 "no_local" "standard" "partial" 4 "strided")
      add_gemm_configuration(
        "${data}" 32 "false" "false" "false"
        64 8 4 4 8 1 1 1 1 1 1 "no_local" "standard" "partial" 1 "strided")
      add_gemm_configuration(
        "${data}" 32 "false" "false" "false"
        64 8 4 4 8 1 1 1 1 1 1 "no_local" "standard" "partial" 2 "strided")
      add_gemm_configuration(
        "${data}" 16 "false" "false" "false"
        64 4 4 4 4 1 1 1 1 1 1 "no_local" "standard" "partial" 4 "strided")
      add_gemm_configuration(
        "${data}" 16 "false" "false" "false"
        64 4 4 4 4 1 1 1 1 1 1 "no_local" "standard" "partial" 1 "strided")
      add_gemm_configuration(
        "${data}" 64 "false" "false" "false"
        64 4 4 8 8 1 1 1 1 1 1 "no_local" "standard" "partial" 2 "strided")
      add_gemm_configuration(
        "${data}" 128 "false" "false" "false"
        64 4 8 16 8 1 1 1 1 1 1 "no_local" "standard" "partial" 4 "strided")
      add_gemm_configuration(
        "${data}" 16 "false" "false" "false"
        64 4 4 4 4 1 1 1 1 1 1 "no_local" "standard" "partial" 2 "strided")
    elseif(${BLAS_MODEL_OPTIMIZATION} STREQUAL "VGG_16")
      add_gemm_configuration(
        "${data}" 64 "false" "false" "false"
        64 4 4 8 8 1 1 1 1 1 1 "no_local" "standard" "partial" 2 "strided")
      add_gemm_configuration(
        "${data}" 128 "false" "false" "false"
        64 4 8 16 8 1 1 1 1 1 1 "no_local" "standard" "partial" 4 "strided")
      add_gemm_configuration(
        "${data}" 64 "false" "false" "false"
        64 4 4 4 4 1 1 1 1 1 1 "no_local" "standard" "partial" 2 "strided")
    else()
      add_gemm_configuration(
        "${data}" 64 "false" "false" "false"
        64 4 4 8 8 1 1 1 1 1 1 "no_local" "standard" "partial" 4 "strided")
      add_gemm_configuration(
        "${data}" 128 "false" "false" "false"
        64 4 8 16 8 1 1 1 1 1 1 "no_local" "standard" "partial" 4 "strided")
      add_gemm_configuration(
        "${data}" 32 "false" "false" "false"
        64 8 4 4 8 1 1 1 1 1 1 "no_local" "standard" "partial" 4 "strided")
    endif()
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 2 2 4 4 1 1 1 1 4 4 "no_local" "standard" "full" 2 "interleaved")
  endforeach()
elseif(${TARGET} STREQUAL "POWER_VR" AND NOT IMGDNN_DIR)
  set(supported_types
    "float"
    "half"
  )
  foreach(data ${supported_types})
    add_gemm_configuration(
      "${data}" 96 "true" "false" "false"
      16 4 6 12 8 1 1 1 1 1 1 "local" "standard" "full" 1 "strided")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      128 1 1 8 8 1 1 1 1 1 1 "local" "standard" "full" 1 "strided")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 4 8 8 1 1 1 1 1 1 "no_local" "standard" "full" 1 "strided")
    add_gemm_configuration(
      "${data}" 128 "false" "false" "false"
      16 4 8 16 8 1 1 1 1 1 1 "local" "standard" "full" 1 "strided")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      32 4 4 8 8 1 1 1 1 1 1 "local" "standard" "full" 1 "strided")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 4 4 4 1 1 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")
  endforeach()
elseif(${TARGET} STREQUAL "AMD_GPU")  # need investigation
  set(supported_types
    "float"
    "double"
    "half"
  )
  set(workgroup_float 16)
  set(workgroup_double 8)
  set(workgroup_half 32)
  foreach(data ${supported_types})
    set(twr "${workgroup_${data}}")
    set(twc "${workgroup_${data}}")

    add_gemm_configuration(
      "${data}" 256 "false" "false" "false"
      64 1 1 ${twr} ${twc} 1 1 1 1 1 1 "local" "standard" "full" 1 "strided")
    add_gemm_configuration(
      "${data}" 256 "false" "false" "false"
      64 4 4 ${twr} ${twc} 1 1 1 1 1 1 "local" "standard" "full" 2 "strided")

    add_gemm_configuration(
      "${data}" 256 "true" "true" "true"
      64 1 1 ${twr} ${twc} 1 1 1 1 1 1 "local" "tall_skinny" "none" 2 "strided")
    add_gemm_configuration(
      "${data}" 256 "true" "true" "true"
      64 2 2 ${twr} ${twc} 1 1 1 1 1 1 "local" "tall_skinny" "none" 2 "strided")
    add_gemm_configuration(
      "${data}" 256 "true" "true" "true"
      64 4 4 ${twr} ${twc} 1 1 1 1 1 1 "local" "tall_skinny" "none" 2 "strided")
    add_gemm_configuration(
      "${data}" 256 "true" "true" "true"
      64 1 4 ${twr} ${twc} 1 1 1 1 1 1 "local" "tall_skinny" "none" 2 "strided")
    add_gemm_configuration(
      "${data}" 256 "true" "true" "true"
      64 4 1 ${twr} ${twc} 1 1 1 1 1 1 "local" "tall_skinny" "none" 2 "strided")

    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 4 4 4 1 1 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")
  endforeach()
elseif(${TARGET} STREQUAL "NVIDIA_GPU")
 set(supported_types
    "float"
  )
  foreach(data ${supported_types})
    add_gemm_configuration(
        "${data}" 128 "false" "false" "true"
        128 2 2 8 8 1 1 1 1 1 1 "local" "standard" "full" 1 "strided")
    add_gemm_configuration(
        "${data}"  64 "false" "false" "true"
        64 8 8 8 8 1 1 2 2 1 1 "local" "standard" "full" 1 "strided")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 2 2 4 4 1 1 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")
  endforeach()
else() # default cpu backend
  set(supported_types
    "float"
    "double"
  )
  foreach(data ${supported_types})
    if(NAIVE_GEMM)
      add_gemm_configuration(
        "${data}"  64 "false" "false" "false"
        64 8 8 8 8 1 1 1 1 1 1 "no_local" "naive" "none" 1 "strided")
    else()
      add_gemm_configuration(
        "${data}"  64 "false" "false" "false"
        64 2 2 8 8 1 1 1 1 1 1 "no_local" "standard" "full" 2 "strided")
      add_gemm_configuration(
        "${data}"  64 "false" "false" "false"
        64 8 8 8 8 1 1 1 1 1 1 "no_local" "standard" "partial" 1 "strided")
    endif()

    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 2 2 4 4 1 1 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")
  endforeach()
endif()
add_library(${func} OBJECT ${gemm_sources})
set_target_compile_def(${func})
# The blas library depends on FindComputeCpp
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE}
                           ${SYCLBLAS_COMMON_INCLUDE_DIR} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${gemm_sources})
endfunction(generate_blas_gemm_objects)


# Generate quantization instantiations
function(generate_quantize)
  set(LOCATION "${SYCLBLAS_GENERATED_SRC}/quantize")
  set(quantize_data_list "${data_list}")
  # float and double don't need to be quantized
  list(REMOVE_ITEM quantize_data_list "float" "double")
  foreach(executor ${executor_list})
    # First generate quantize_base.cpp.in for float and double
    sanitize_file_name(file_name
      "quantize_${executor}_base.cpp")
    add_custom_command(OUTPUT "${LOCATION}/${file_name}"
      COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_quantize.py
        ${PROJECT_SOURCE_DIR}/external/ #1
        ${SYCLBLAS_SRC_GENERATOR}/gen #2
        ${SYCLBLAS_SRC}/quantize/quantize_base.cpp.in #3
        ${executor} #4
        "DATA_TYPE" #5 # data is ignored in the file template
        ${file_name} #6
      MAIN_DEPENDENCY ${SYCLBLAS_SRC}/quantize/quantize_base.cpp.in
      DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_quantize.py
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      VERBATIM
    )
    list(APPEND FUNC_SRC "${LOCATION}/${file_name}")

    # Generate quantize.cpp.in for each special data type
    foreach(data ${quantize_data_list})
      cpp_type(cpp_data ${data})
      sanitize_file_name(file_name
        "quantize_${executor}_${data}.cpp")
      add_custom_command(OUTPUT "${LOCATION}/${file_name}"
        COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_quantize.py
          ${PROJECT_SOURCE_DIR}/external/ #1
          ${SYCLBLAS_SRC_GENERATOR}/gen #2
          ${SYCLBLAS_SRC}/quantize/quantize.cpp.in #3
          ${executor} #4
          ${cpp_data} #5
          ${file_name} #6
        MAIN_DEPENDENCY ${SYCLBLAS_SRC}/quantize/quantize.cpp.in
        DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_quantize.py
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        VERBATIM
      )
      list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
    endforeach(data)
  endforeach(executor)

  add_library(quantize OBJECT ${FUNC_SRC})
  set_target_compile_def(quantize)
  target_include_directories(quantize PRIVATE
    ${SYCLBLAS_SRC}
    ${SYCLBLAS_INCLUDE}
    ${SYCLBLAS_COMMON_INCLUDE_DIR}
    ${SYCL_INCLUDE_DIRS}
    ${COMPUTECPP_SDK_INCLUDE})
  message(STATUS "Adding SYCL to target quantize")
  add_sycl_to_target(TARGET quantize SOURCES ${FUNC_SRC})
endfunction(generate_quantize)


function (build_library LIB_NAME)
add_library(${LIB_NAME}
                             $<TARGET_OBJECTS:sycl_policy>
                             $<TARGET_OBJECTS:quantize>
                             $<TARGET_OBJECTS:axpy>
                             $<TARGET_OBJECTS:asum>
                             $<TARGET_OBJECTS:asum_return>
                             $<TARGET_OBJECTS:copy>
                             $<TARGET_OBJECTS:dot>
                             $<TARGET_OBJECTS:dot_return>
                             $<TARGET_OBJECTS:iamax>
                             $<TARGET_OBJECTS:iamax_return>
                             $<TARGET_OBJECTS:iamin>
                             $<TARGET_OBJECTS:iamin_return>
                             $<TARGET_OBJECTS:nrm2>
                             $<TARGET_OBJECTS:nrm2_return>
                             $<TARGET_OBJECTS:rot>
                             $<TARGET_OBJECTS:scal>
                             $<TARGET_OBJECTS:swap>
                             $<TARGET_OBJECTS:gemv>
                             $<TARGET_OBJECTS:ger>
                             $<TARGET_OBJECTS:symv>
                             $<TARGET_OBJECTS:syr>
                             $<TARGET_OBJECTS:syr2>
                             $<TARGET_OBJECTS:trmv>
                             $<TARGET_OBJECTS:gemm_launcher>
                             $<TARGET_OBJECTS:gemm>
                             $<TARGET_OBJECTS:trsm>
                            )
endfunction(build_library)
