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

# represent the list of supported index/increment types
set(index_list "${BLAS_INDEX_TYPES}" )

# BLAS_DATA_TYPES was provided by the user
#Each data type in a data list determines the container types.
#The container type for SYCLbackend is BufferIterator<${data}>
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
  message(STATUS "Adding ${TUNING_TARGET} backend to target ${in_target}")
  if(${TUNING_TARGET} STREQUAL "INTEL_GPU")
    target_compile_definitions(${in_target} PUBLIC INTEL_GPU=1)
  elseif(${TUNING_TARGET} STREQUAL "AMD_GPU")
    target_compile_definitions(${in_target} PUBLIC AMD_GPU=1)
  elseif(${TUNING_TARGET} STREQUAL "ARM_GPU")
    target_compile_definitions(${in_target} PUBLIC ARM_GPU=1)
  elseif(${TUNING_TARGET} STREQUAL "RCAR")
    target_compile_definitions(${in_target} PUBLIC RCAR=1)
  elseif(${TUNING_TARGET} STREQUAL "POWER_VR")
    target_compile_definitions(${in_target} PUBLIC POWER_VR=1)
  elseif(${TUNING_TARGET} STREQUAL "NVIDIA_GPU")
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
foreach(data ${data_list})
  cpp_type(cpp_data ${data})
  set(container_list "BufferIterator<${cpp_data}>")
  foreach(index ${index_list})
    foreach(container0 ${container_list})
      foreach(increment ${index_list})
        sanitize_file_name(file_name
          "${func}_${data}_${index}_${container0}_${increment}.cpp")
        add_custom_command(OUTPUT "${LOCATION}/${file_name}"
          COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_unary.py
            ${PROJECT_SOURCE_DIR}/external/
            ${SYCLBLAS_SRC_GENERATOR}/gen
            ${blas_level}
            ${func}
            ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
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
foreach(data ${data_list})
  cpp_type(cpp_data ${data})
  set(container_list "BufferIterator<${cpp_data}>")
  foreach(index ${index_list})
    foreach(container0 ${container_list})
      foreach(container1 ${container_list})
        set(container_names "${container0}_${container1}")
        foreach(increment ${index_list})
          sanitize_file_name(file_name
            "${func}_${data}_${index}_${container_names}_${increment}.cpp")
          add_custom_command(OUTPUT "${LOCATION}/${file_name}"
            COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_binary.py
              ${PROJECT_SOURCE_DIR}/external/
              ${SYCLBLAS_SRC_GENERATOR}/gen
              ${blas_level}
              ${func}
              ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
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
add_library(${func} OBJECT ${FUNC_SRC})
set_target_compile_def(${func})
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE}
                           ${SYCLBLAS_COMMON_INCLUDE_DIR} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_binary_objects)


# blas binary function for generating source code
function(generate_blas_reduction_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
set(operator_list "AddOperator" "MinOperator" "MaxOperator" "ProductOperator" "AbsoluteAddOperator" "MeanOperator")
string(FIND ${func} "_const" pos)
if(pos)
  string(REPLACE "_const" "" actualfunc ${func})
endif()
foreach(data ${data_list})
  cpp_type(cpp_data ${data})
  set(container_list_in)
  if(pos EQUAL -1)
    list(APPEND container_list_in "BufferIterator<${cpp_data}>")
  else()
    list(APPEND container_list_in "BufferIterator<${cpp_data} const>")
  endif()
  set(container_list_out "BufferIterator<${cpp_data}>")
  foreach(index ${index_list})
    set(container_list "BufferIterator<${cpp_data}>")
    foreach(operator ${operator_list})
      foreach(container0 ${container_list_in})
        foreach(container1 ${container_list_out})
          set(container_names "${container0}_${container1}")
          foreach(increment ${index_list})
            sanitize_file_name(file_name
              "${func}_${operator}_${data}_${index}_${container0}_${increment}.cpp")
            add_custom_command(OUTPUT "${LOCATION}/${file_name}"
              COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_reduction.py
                ${PROJECT_SOURCE_DIR}/external/
                ${SYCLBLAS_SRC_GENERATOR}/gen
                ${blas_level}
                ${func}
                ${SYCLBLAS_SRC}/interface/${blas_level}/${actualfunc}.cpp.in
                ${cpp_data}
                ${index}
                ${increment}
                ${container0}
                ${container1}
                ${operator}
                ${file_name}
              MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${actualfunc}.cpp.in
              DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_reduction.py
              WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
              VERBATIM
            )
            list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
            endforeach(increment)
        endforeach(container1)
      endforeach(container0)
    endforeach(operator)
  endforeach(index)
endforeach(data)
add_library(${func} OBJECT ${FUNC_SRC})
set_target_compile_def(${func})
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE}
                           ${SYCLBLAS_COMMON_INCLUDE_DIR} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_reduction_objects)


# blas special binary function for generating source code
function(generate_blas_binary_special_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
foreach(data ${data_list})
  cpp_type(cpp_data ${data})
  set(container_list_in "BufferIterator<${cpp_data}>")
  foreach(index ${index_list})
    set(container_list_out
      "BufferIterator<IndexValueTuple<${index},${cpp_data}>>")
    foreach(container0 ${container_list_in})
      foreach(container1 ${container_list_out})
        set(container_names "${container0}_${container1}")
        foreach(increment ${index_list})
          sanitize_file_name(file_name
            "${func}_${data}_${index}_${container_names}_${increment}.cpp")
          add_custom_command(OUTPUT "${LOCATION}/${file_name}"
            COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_binary_special.py
              ${PROJECT_SOURCE_DIR}/external/
              ${SYCLBLAS_SRC_GENERATOR}/gen
              ${blas_level}
              ${func}
              ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
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
string(FIND ${func} "_const" const_pos)
if(const_pos)
  string(REPLACE "_const" "" actualfunc ${func})
endif()
foreach(data ${data_list})
  cpp_type(cpp_data ${data})
  set(container_list_in)
  if(const_pos EQUAL -1)
    list(APPEND container_list_in "BufferIterator<${cpp_data}>")
  else()
    list(APPEND container_list_in "BufferIterator<${cpp_data} const>")
  endif()
  set(container_list_out "BufferIterator<${cpp_data}>")
  foreach(index ${index_list})
    foreach(container0 ${container_list_in})
      foreach(container1 ${container_list_in})
        foreach(container2 ${container_list_out})
          set(container_names
            "${container0}_${container1}_${container2}")
          foreach(increment ${index_list})
            sanitize_file_name(file_name
              "${func}_${data}_${index}_${container_names}_${increment}.cpp")
            add_custom_command(OUTPUT "${LOCATION}/${file_name}"
              COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_ternary.py
                ${PROJECT_SOURCE_DIR}/external/
                ${SYCLBLAS_SRC_GENERATOR}/gen
                ${blas_level}
                ${func}
                ${SYCLBLAS_SRC}/interface/${blas_level}/${actualfunc}.cpp.in
                ${cpp_data}
                ${index}
                ${increment}
                ${container0}
                ${container1}
                ${container2}
                ${file_name}
              MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${actualfunc}.cpp.in
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
add_library(${func} OBJECT ${FUNC_SRC})
set_target_compile_def(${func})
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE}
                           ${SYCLBLAS_COMMON_INCLUDE_DIR} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_ternary_objects)


# blas function for generating source code for the rotg operator (asynchronous version with containers)
function(generate_blas_rotg_objects blas_level func)
  set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
  foreach (data ${data_list})
    cpp_type(cpp_data ${data})
    set(container_list_in_out "BufferIterator<${cpp_data}>")
    foreach (container0 ${container_list_in_out})
      foreach (container1 ${container_list_in_out})
        foreach (container2 ${container_list_in_out})
          foreach (container3 ${container_list_in_out})
            set(container_names "${container0}_${container1}_${container2}_${container3}")
            sanitize_file_name(file_name
                    "${func}_${data}_${index}_${container_names}_${increment}.cpp")
            add_custom_command(OUTPUT "${LOCATION}/${file_name}"
                    COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_rotg.py
                    ${PROJECT_SOURCE_DIR}/external/
                    ${SYCLBLAS_SRC_GENERATOR}/gen
                    ${blas_level}
                    ${func}
                    ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                    ${cpp_data}
                    ${container0}
                    ${container1}
                    ${container2}
                    ${container3}
                    ${file_name}
                    MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                    DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_rotg.py
                    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                    VERBATIM
                    )
            list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
          endforeach (container3)
        endforeach (container2)
      endforeach (container1)
    endforeach (container0)
  endforeach (data)
  add_library(${func} OBJECT ${FUNC_SRC})
  set_target_compile_def(${func})
  target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE}
          ${SYCLBLAS_COMMON_INCLUDE_DIR} ${THIRD_PARTIES_INCLUDE})
  message(STATUS "Adding SYCL to target ${func}")
  add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_rotg_objects)


# blas function for generating source code for the rotg operator (synchronous version)
function(generate_blas_rotg_return_objects blas_level func)
  set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
  foreach (data ${data_list})
    cpp_type(cpp_data ${data})
    set(container_list "BufferIterator<${cpp_data}>")
    sanitize_file_name(file_name
            "${func}_${data}_${index}_${container0}_${increment}.cpp")
    add_custom_command(OUTPUT "${LOCATION}/${file_name}"
            COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_rotg_return.py
            ${PROJECT_SOURCE_DIR}/external/
            ${SYCLBLAS_SRC_GENERATOR}/gen
            ${blas_level}
            ${func}
            ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
            ${cpp_data}
            ${file_name}
            MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
            DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_rotg_return.py
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            VERBATIM
            )
    list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
  endforeach (data)
  add_library(${func} OBJECT ${FUNC_SRC})
  set_target_compile_def(${func})
  target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE}
          ${SYCLBLAS_COMMON_INCLUDE_DIR} ${THIRD_PARTIES_INCLUDE})
  message(STATUS "Adding SYCL to target ${func}")
  add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_rotg_return_objects)

# blas function for generating source code for the rotg operator (asynchronous version with containers)
function(generate_blas_rotmg_objects blas_level func)
  set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
  foreach (data ${data_list})
    cpp_type(cpp_data ${data})
    set(container_list_in_out "BufferIterator<${cpp_data}>")
    foreach (container0 ${container_list_in_out})
      foreach (container1 ${container_list_in_out})
        foreach (container2 ${container_list_in_out})
          foreach (container3 ${container_list_in_out})
            foreach (container4 ${container_list_in_out})
              set(container_names "${container0}_${container1}_${container2}_${container3}")
              sanitize_file_name(file_name "${func}_${data}_${container_names}.cpp")
              add_custom_command(OUTPUT "${LOCATION}/${file_name}"
                      COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_rotmg.py
                      ${PROJECT_SOURCE_DIR}/external/
                      ${SYCLBLAS_SRC_GENERATOR}/gen
                      ${blas_level}
                      ${func}
                      ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                      ${cpp_data}
                      ${container0}
                      ${container1}
                      ${container2}
                      ${container3}
                      ${container4}
                      ${file_name}
                      MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                      DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_rotmg.py
                      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                      VERBATIM
                      )
              list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
            endforeach (container4)
          endforeach (container3)
        endforeach (container2)
      endforeach (container1)
    endforeach (container0)
  endforeach (data)
  add_library(${func} OBJECT ${FUNC_SRC})
  set_target_compile_def(${func})
  target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE}
          ${SYCLBLAS_COMMON_INCLUDE_DIR} ${THIRD_PARTIES_INCLUDE})
  message(STATUS "Adding SYCL to target ${func}")
  add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_rotmg_objects)

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
  jm_m jm_n jm_k 
  jm_in_type jm_out_type
  gemm_memory_type
  gemm_shape_type
  gemm_vectorize_type
  vector_size
  batch_type
  use_joint_matrix
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
          foreach(index ${index_list})
            set(file_name "${func}_${double_buffer}_${conflict_a}_"
                          "${conflict_b}_${trans_a}_${trans_b}_"
                          "${is_beta_zero}_${gemm_memory_type}_"
                          "${gemm_shape_type}_${gemm_vectorize_type}_"
                          "${vector_size}_${batch_type}_${use_joint_matrix}_"
                          "${data}_${index}_${tir}_${tic}_${twr}_"
                          "${twc}_${tsr}_${tsc}_${tlr}_${tlc}_"
                          "${item_batch}_${wg_batch}_"
                          "${jm_m}_${jm_n}_${jm_k}_${jm_in_type}_${jm_out_type}_"
                          "${wg_size}_${cache_line_size}.cpp")
            sanitize_file_name(file_name "${file_name}")
            add_custom_command(OUTPUT "${LOCATION}/${file_name}"
              COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_gemm_launcher.py
                ${PROJECT_SOURCE_DIR}/external/
                ${SYCLBLAS_SRC_GENERATOR}/gen
                ${blas_level}
                ${func}
                ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
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
                ${jm_m}
                ${jm_n}
                ${jm_k}
                ${jm_in_type}
                ${jm_out_type}
                ${wg_size}
                ${cache_line_size}
                ${file_name}
                ${gemm_vectorize_type}
                ${vector_size}
                ${batch_type}
                ${use_joint_matrix}
              MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
              DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_gemm_launcher.py
              WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
              VERBATIM
            )
            list(APPEND gemm_sources "${LOCATION}/${file_name}")
            set(gemm_sources "${gemm_sources}" PARENT_SCOPE)
          endforeach(index)
      endforeach(is_beta_zero)
    endforeach(trans_b)
  endforeach(trans_a)
endfunction()
if(${TUNING_TARGET} STREQUAL "INTEL_GPU")
  set(supported_types
    "float"
    "double"
    "half"
  )
  foreach(data ${supported_types})
    add_gemm_configuration(
      "${data}" 64 "true" "false" "false"
      64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 4 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 8 8 8 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 4 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 8 8 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 4 "strided" "false")

    if (${data} STREQUAL "half")
      add_gemm_configuration(
         "${data}" 16 "true" "false" "false"
         64 1 1 8 8 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
      add_gemm_configuration(
        "${data}" 16 "true" "false" "false"
         64 2 2 8 8 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
    else()
      add_gemm_configuration(
         "${data}" 16 "true" "false" "false"
         64 1 1 4 4 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
      add_gemm_configuration(
        "${data}" 16 "true" "false" "false"
         64 2 2 4 4 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
    endif()

    add_gemm_configuration(
      "${data}" 64 "true" "true" "true"
      64 2 2 8 8 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "true" "true" "true"
      64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")

    if (${data} STREQUAL "double")
      add_gemm_configuration(
        "${data}" 256 "true" "true" "true"
        64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
    else()
      add_gemm_configuration(
        "${data}" 256 "true" "true" "true"
        64 4 4 16 16 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
    endif()

    add_gemm_configuration(
      "${data}" 32 "true" "true" "true"
      64 2 1 8 4 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")
    add_gemm_configuration(
      "${data}" 32 "true" "true" "true"
      64 2 2 8 4 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 4 "strided" "false")

    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 4 4 4 1 1 1 1 4 4 1 1 1 float float "no_local" "standard" "full" 4 "interleaved" "false")
  endforeach()
elseif(${TUNING_TARGET} STREQUAL "RCAR") # need investigation
  set(supported_types
    "float"
  )
  foreach(data ${supported_types})
    add_gemm_configuration(
      "${data}" 32 "false" "false" "false"
      128 4 8 8 4 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 4 "strided" "false")
    add_gemm_configuration(
      "${data}" 32 "false" "false" "false"
      128 8 4 4 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 4 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 4 4 4 1 1 1 1 4 4 1 1 1 float float "no_local" "standard" "full" 4 "interleaved" "false")
  endforeach()
elseif(${TUNING_TARGET} STREQUAL "ARM_GPU")
  set(supported_types
    "float"
    "half"
  )
  foreach(data ${supported_types})
    if(${BLAS_MODEL_OPTIMIZATION} STREQUAL "RESNET_50")
      add_gemm_configuration(
        "${data}" 64 "false" "false" "false"
        64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "full" 4 "strided" "false")
      add_gemm_configuration(
        "${data}" 32 "false" "false" "false"
        64 8 4 4 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 4 "strided" "false")
      add_gemm_configuration(
        "${data}" 32 "false" "false" "false"
        64 4 8 8 4 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 4 "strided" "false")
      add_gemm_configuration(
        "${data}" 64 "false" "false" "false"
        64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 4 "strided" "false")
      add_gemm_configuration(
        "${data}" 32 "false" "false" "false"
        64 8 4 4 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 1 "strided" "false")
      add_gemm_configuration(
        "${data}" 32 "false" "false" "false"
        64 8 4 4 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 2 "strided" "false")
      add_gemm_configuration(
        "${data}" 16 "false" "false" "false"
        64 4 4 4 4 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 4 "strided" "false")
      add_gemm_configuration(
        "${data}" 16 "false" "false" "false"
        64 4 4 4 4 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 1 "strided" "false")
      add_gemm_configuration(
        "${data}" 64 "false" "false" "false"
        64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 2 "strided" "false")
      add_gemm_configuration(
        "${data}" 128 "false" "false" "false"
        64 4 8 16 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 4 "strided" "false")
      add_gemm_configuration(
        "${data}" 16 "false" "false" "false"
        64 4 4 4 4 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 2 "strided" "false")
    elseif(${BLAS_MODEL_OPTIMIZATION} STREQUAL "VGG_16")
      add_gemm_configuration(
        "${data}" 64 "false" "false" "false"
        64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 2 "strided" "false")
      add_gemm_configuration(
        "${data}" 128 "false" "false" "false"
        64 4 8 16 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 4 "strided" "false")
      add_gemm_configuration(
        "${data}" 64 "false" "false" "false"
        64 4 4 4 4 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 2 "strided" "false")
    else()
      add_gemm_configuration(
        "${data}" 64 "false" "false" "false"
        64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 4 "strided" "false")
      add_gemm_configuration(
        "${data}" 128 "false" "false" "false"
        64 4 8 16 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 4 "strided" "false")
      add_gemm_configuration(
        "${data}" 32 "false" "false" "false"
        64 8 4 4 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 4 "strided" "false")
    endif()
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 2 2 4 4 1 1 1 1 4 4 1 1 1 float float "no_local" "standard" "full" 2 "interleaved" "false")
  endforeach()
elseif(${TUNING_TARGET} STREQUAL "POWER_VR" AND NOT IMGDNN_DIR)
  set(supported_types
    "float"
    "half"
  )
  foreach(data ${supported_types})
    add_gemm_configuration(
      "${data}" 96 "true" "false" "false"
      16 4 6 12 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      128 1 1 8 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
      "${data}" 128 "false" "false" "false"
      16 4 8 16 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      32 4 4 8 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 4 4 4 1 1 1 1 4 4 1 1 1 float float "no_local" "standard" "full" 4 "interleaved" "false")
  endforeach()
elseif(${TUNING_TARGET} STREQUAL "AMD_GPU")  # need investigation
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
      64 1 1 ${twr} ${twc} 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
      "${data}" 256 "false" "false" "false"
      64 4 4 ${twr} ${twc} 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 2 "strided" "false")

    add_gemm_configuration(
      "${data}" 256 "true" "true" "true"
      64 1 1 ${twr} ${twc} 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 2 "strided" "false")
    add_gemm_configuration(
      "${data}" 256 "true" "true" "true"
      64 2 2 ${twr} ${twc} 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 2 "strided" "false")
    add_gemm_configuration(
      "${data}" 256 "true" "true" "true"
      64 4 4 ${twr} ${twc} 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 2 "strided" "false")
    add_gemm_configuration(
      "${data}" 256 "true" "true" "true"
      64 1 4 ${twr} ${twc} 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 2 "strided" "false")
    add_gemm_configuration(
      "${data}" 256 "true" "true" "true"
      64 4 1 ${twr} ${twc} 1 1 1 1 1 1 1 1 1 float float "local" "tall_skinny" "none" 2 "strided" "false")

    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 4 4 4 1 1 1 1 4 4 1 1 1 float float "no_local" "standard" "full" 4 "interleaved" "false")
  endforeach()
elseif(${TUNING_TARGET} STREQUAL "NVIDIA_GPU")
 set(supported_types
    "float"
    "double"
  )
  string(FIND ${DPCPP_SYCL_ARCH} "_" start_idx)
  if(start_idx)
    MATH(EXPR start_idx "${start_idx} + 1")
    string(SUBSTRING ${DPCPP_SYCL_ARCH} ${start_idx} "2" sm_val)
  endif()
  foreach(data ${supported_types})
    # Joint Matrix specific GEMM configurations (only for float)
    if(${start_idx} AND ${sm_val} GREATER_EQUAL "80")
      add_gemm_configuration(
          "float" 128 "false" "true" "true"
          128 2 4 16 8 16 2 1 1 1 1 16 16 16 cl::sycl::half float "local" "standard" "none" 1 "strided" "true")
      add_gemm_configuration(
          "float" 128 "false" "true" "true"
          128 4 8 16 8 16 2 1 1 1 1 16 16 16 cl::sycl::half float "local" "standard" "none" 1 "strided" "true")
      add_gemm_configuration(
          "float" 256 "false" "true" "true"
          128 8 8 16 16 16 2 1 1 1 1 16 16 16 cl::sycl::half float "local" "standard" "none" 1 "strided" "true")
    endif()
    # Non-Joint Matrix specific GEMM Configurations
    add_gemm_configuration(
      "${data}" 128 "false" "false" "true"
      128 2 2 8 8 1 1 1 1 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
        "${data}"  64 "false" "false" "true"
          64 8 8 8 8 1 1 2 2 1 1 1 1 1 float float "local" "standard" "full" 1 "strided" "false")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 2 2 4 4 1 1 1 1 4 4 1 1 1 float float "no_local" "standard" "full" 4 "interleaved" "false")
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
        64 8 8 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "naive" "none" 1 "strided" "false" "false")
    else()
      add_gemm_configuration(
        "${data}"  64 "false" "false" "false"
        64 2 2 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "full" 2 "strided" "false" "false")
      add_gemm_configuration(
        "${data}"  64 "false" "false" "false"
        64 8 8 8 8 1 1 1 1 1 1 1 1 1 float float "no_local" "standard" "partial" 1 "strided" "false" "false")
    endif()

    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 2 2 4 4 1 1 1 1 4 4 1 1 1 float float "no_local" "standard" "full" 4 "interleaved" "false" "false")
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


function (build_library LIB_NAME ENABLE_EXTENSIONS)
  set(LIB_SRCS  $<TARGET_OBJECTS:axpy>
                $<TARGET_OBJECTS:asum>
                $<TARGET_OBJECTS:asum_return>
                $<TARGET_OBJECTS:copy>
                $<TARGET_OBJECTS:dot>
                $<TARGET_OBJECTS:dot_return>
                $<TARGET_OBJECTS:sdsdot>
                $<TARGET_OBJECTS:sdsdot_return>
                $<TARGET_OBJECTS:iamax>
                $<TARGET_OBJECTS:iamax_return>
                $<TARGET_OBJECTS:iamin>
                $<TARGET_OBJECTS:iamin_return>
                $<TARGET_OBJECTS:nrm2>
                $<TARGET_OBJECTS:nrm2_return>
                $<TARGET_OBJECTS:rot>
                $<TARGET_OBJECTS:rotm>
                $<TARGET_OBJECTS:rotmg>
                $<TARGET_OBJECTS:rotg>
                $<TARGET_OBJECTS:rotg_return>
                $<TARGET_OBJECTS:scal>
                $<TARGET_OBJECTS:swap>
                $<TARGET_OBJECTS:gbmv>
                $<TARGET_OBJECTS:gemv>
                $<TARGET_OBJECTS:ger>
                $<TARGET_OBJECTS:sbmv>
                $<TARGET_OBJECTS:symv>
                $<TARGET_OBJECTS:syr>
                $<TARGET_OBJECTS:syr2>
                $<TARGET_OBJECTS:trmv>
                $<TARGET_OBJECTS:gemm_launcher>
                $<TARGET_OBJECTS:gemm>
                $<TARGET_OBJECTS:trsm>)

  if (${ENABLE_EXTENSIONS})
    list(APPEND LIB_SRCS $<TARGET_OBJECTS:reduction>)
  endif()

  add_library(${LIB_NAME} ${LIB_SRCS})

  if(BLAS_ENABLE_CONST_INPUT)
    set(CONST_SRCS $<TARGET_OBJECTS:gemv_const>
                   $<TARGET_OBJECTS:gemm_const>)

    if(${ENABLE_EXTENSIONS})
      list(APPEND CONST_SRCS $<TARGET_OBJECTS:reduction_const>)
    endif()

    target_sources(${LIB_NAME} PRIVATE ${CONST_SRCS})
  endif()
endfunction(build_library)
