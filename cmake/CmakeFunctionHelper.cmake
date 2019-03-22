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
#represent the list of supported data type.
#Each data type in a data list determines the container types.
#The container type for SYCLbackend is BufferIterator<${data}, codeplay_policy>
set(data_list "float")
 #if double supported we add double as a data type
if(DOUBLE_SUPPORT)
  set(data_list "float" "double")
endif()

## represent the list of bolean options
set(boolean_list "true" "false")

# gemm_configuration(work_group_size, double_buffer, conflict_a, conflict_b, 
#                    cache_line_size, tir, tic, twr, twc, tlr, tlc, local_mem)
set(gemm_configuration_lists "")

#intel GPU
if(${TARGET} STREQUAL "INTEL_GPU")
  set(gemm_configuration_0 256 "true" "false" "false" 64 4 4 16 16 1 1 "local_memory")
  set(gemm_configuration_1 256 "false" "false" "false" 64 8 8 16 16 1 1 "no_local_memory")
  set(gemm_configuration_2 64 "true" "false" "false" 64 4 4 8 8 1 1 "local_memory")
  set(gemm_configuration_3 64 "false" "false" "false" 64 8 8 8 8 1 1 "no_local_memory")
  set(gemm_configuration_4 64 "true" "false" "false" 64 8 8 8 8 1 1 "local_memory")
  list(APPEND gemm_configuration_lists gemm_configuration_0 gemm_configuration_1 
                                       gemm_configuration_2 gemm_configuration_3 
                                       gemm_configuration_4)
elseif(${TARGET} STREQUAL "RCAR") # need investigation

  set(gemm_configuration_0 32 "false" "false" "false" 128 4 8 8 4 1 1 "local_memory")
  set(gemm_configuration_1 32 "false" "false" "false" 128 8 4 4 8 1 1 "local_memory")
  list(APPEND gemm_configuration_lists gemm_configuration_0 gemm_configuration_1)
elseif(${TARGET} STREQUAL "ARM_GPU")
  set(gemm_configuration_0 64 "false" "false" "false" 64 4 4 8 8 1 1 "no_local_memory")
  set(gemm_configuration_1 128 "false" "false" "false" 64 4 8 16 8 1 1 "no_local_memory")
  set(gemm_configuration_2 32 "false" "false" "false" 64 8 4 4 8 1 1 "no_local_memory")
  list(APPEND gemm_configuration_lists gemm_configuration_0 gemm_configuration_1 
                                       gemm_configuration_2)
elseif(${TARGET} STREQUAL "AMD_GPU")  # need investigation
  set(gemm_configuration_0 256 "true" "false" "false" 64 1 1 16 16 1 1 "local_memory")
  set(gemm_configuration_1 256 "false" "false" "false" 64 8 8 16 16 1 1 "local_memory")
  list(APPEND gemm_configuration_lists gemm_configuration_0 gemm_configuration_1)
else() # default cpu backend
  set(gemm_type "no_local_memory" )
  if(NAIVE_GEMM)
    set(gemm_type "naive")
  endif()
  set(gemm_configuration_0 64 "false" "false" "false" 64 8 8 8 8 1 1 "${gemm_type}")
  set(gemm_configuration_lists "")
  list(APPEND gemm_configuration_lists gemm_configuration_0)
endif()


function(set_target_compile_def in_target)
  #setting compiler flag for backend
  if(${TARGET} STREQUAL "INTEL_GPU")
    target_compile_definitions(${in_target} PUBLIC INTEL_GPU=1)
  elseif(${TARGET} STREQUAL "AMD_GPU")
    target_compile_definitions(${in_target} PUBLIC AMD_GPU=1)
  elseif(${TARGET} STREQUAL "ARM_GPU")
    target_compile_definitions(${in_target} PUBLIC ARM_GPU=1)
  elseif(${TARGET} STREQUAL "RCAR")
    target_compile_definitions(${in_target} PUBLIC RCAR=1)
  elseif(${TARGET} STREQUAL "ARM_GPU")
    target_compile_definitions(${in_target} PUBLIC ARM_GPU=1)
  else()
    target_compile_definitions(${in_target} PUBLIC DEFAULT_CPU=1)
  endif()
  #setting always inline attribute
  if(${SYCL_BLAS_ALWAYS_INLINE})
    target_compile_definitions(${in_target} PUBLIC SYCL_BLAS_ALWAYS_INLINE=1)
  endif()

endfunction()


# blas unary function for generating source code
function(generate_blas_unary_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
foreach(executor ${executor_list})
  foreach(data ${data_list})
    set(container_list "BufferIterator<${data},codeplay_policy>")
    foreach(index ${index_list})
      foreach(container0 ${container_list})
        foreach(increment ${index_list})
          set(file_name "${func}_${executor}_${data}_${index}_${container0}_${increment}.cpp")
          STRING(REGEX REPLACE "(\\*|<| |,|>)" "_" file_name ${file_name})
          STRING(REGEX REPLACE "(___|__)" "_" file_name ${file_name})
          add_custom_command(OUTPUT "${LOCATION}/${file_name}"
            COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_unary.py 
              ${PROJECT_SOURCE_DIR}/external/
              ${SYCLBLAS_SRC_GENERATOR}/gen 
              ${blas_level}
              ${func}
              ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
              ${executor}
              ${data}
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
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_unary_objects)


# blas binary function for generating source code
function(generate_blas_binary_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
foreach(executor ${executor_list})
  foreach(data ${data_list})
    set(container_list "BufferIterator<${data},codeplay_policy>")
    foreach(index ${index_list})
      foreach(container0 ${container_list})
        foreach(container1 ${container_list})
          foreach(increment ${index_list})
            set(file_name "${func}_${executor}_${data}_${index}_${container0}_${container1}_${increment}.cpp")
            STRING(REGEX REPLACE "(\\*|<| |,|>)" "_" file_name ${file_name})
            STRING(REGEX REPLACE "(___|__)" "_" file_name ${file_name})
            add_custom_command(OUTPUT "${LOCATION}/${file_name}"
              COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_binary.py 
                ${PROJECT_SOURCE_DIR}/external/
                ${SYCLBLAS_SRC_GENERATOR}/gen 
                ${blas_level}
                ${func}
                ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                ${executor}
                ${data}
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
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_binary_objects)



# blas special binary function for generating source code
function(generate_blas_binary_special_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
foreach(executor ${executor_list})
  foreach(data ${data_list})
    set(container_list_in "BufferIterator<${data},codeplay_policy>")
    foreach(index ${index_list})
      set(container_list_out "BufferIterator<IndexValueTuple<${data},${index}>,codeplay_policy>")
      foreach(container0 ${container_list_in})
        foreach(container1 ${container_list_out})
          foreach(increment ${index_list})
            set(file_name "${func}_${executor}_${data}_${index}_${container1}_${container0}_${increment}.cpp")
            STRING(REGEX REPLACE "(\\*|<| |,|>)" "_" file_name ${file_name})
            STRING(REGEX REPLACE "(___|__)" "_" file_name ${file_name})
            add_custom_command(OUTPUT "${LOCATION}/${file_name}"
              COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_binary_special.py 
                ${PROJECT_SOURCE_DIR}/external/
                ${SYCLBLAS_SRC_GENERATOR}/gen 
                ${blas_level}
                ${func}
                ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                ${executor}
                ${data}
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
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_binary_special_objects)



# blas ternary function for generating source code
function(generate_blas_ternary_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
foreach(executor ${executor_list})
  foreach(data ${data_list})
    set(container_list "BufferIterator<${data},codeplay_policy>")
    foreach(index ${index_list})
      foreach(container0 ${container_list})
        foreach(container1 ${container_list})
          foreach(container2 ${container_list})
            foreach(increment ${index_list})
              set(file_name "${func}_${executor}_${data}_${index}_${container0}_${container1}_${container2}_${increment}.cpp")
              STRING(REGEX REPLACE "(\\*|<| |,|>)" "_" file_name ${file_name})
              STRING(REGEX REPLACE "(___|__)" "_" file_name ${file_name})
              add_custom_command(OUTPUT "${LOCATION}/${file_name}"
                COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_ternary.py 
                  ${PROJECT_SOURCE_DIR}/external/
                  ${SYCLBLAS_SRC_GENERATOR}/gen 
                  ${blas_level}
                  ${func}
                  ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                  ${executor}
                  ${data}
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
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_ternary_objects)


# blas gemm function for generating source code
function(generate_blas_gemm_objects blas_level func)
set(LOCATION "${SYCLBLAS_GENERATED_SRC}/${blas_level}/${func}/")
      foreach(trans_a ${boolean_list})
        foreach(trans_b ${boolean_list})
          foreach(is_beta_zero ${boolean_list})
            foreach(executor ${executor_list})
              foreach(data ${data_list})
                foreach(index ${index_list})
                  foreach(gemm_list ${gemm_configuration_lists})
                    list(GET ${gemm_list} 0 wg_size)
                    list(GET ${gemm_list} 1 double_buffer)
                    list(GET ${gemm_list} 2 conflict_a)
                    list(GET ${gemm_list} 3 conflict_b)
                    list(GET ${gemm_list} 4 cl_size)
                    list(GET ${gemm_list} 5 tir)
                    list(GET ${gemm_list} 6 tic)
                    list(GET ${gemm_list} 7 twr)
                    list(GET ${gemm_list} 8 twc)
                    list(GET ${gemm_list} 9 tlr)
                    list(GET ${gemm_list} 10 tlc)
                    list(GET ${gemm_list} 11 gemm_type)
                    set(file_name "${func}_${double_buffer}_${conflict_a}_"
                                    "${conflict_b}_${trans_a}_${trans_b}_"
                                    "${is_beta_zero}_${gemm_type}_${executor}_"
                                    "${data}_${index}_${tir}_${tic}_${twr}_"
                                    "${twc}_${tlr}_${tlc}_${wg_size}_"
                                    "${cl_size}.cpp")
                    STRING(REGEX REPLACE "(\\*|<| |,|>)" "_" file_name ${file_name})
                    STRING(REGEX REPLACE "(___|__)" "_" file_name ${file_name})
                    add_custom_command(OUTPUT "${LOCATION}/${file_name}"
                      COMMAND ${PYTHON_EXECUTABLE} ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_gemm_launcher.py 
                        ${PROJECT_SOURCE_DIR}/external/
                        ${SYCLBLAS_SRC_GENERATOR}/gen 
                        ${blas_level}
                        ${func}
                        ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                        ${executor}
                        ${data}
                        ${index}
                        ${double_buffer}
                        ${conflict_a}
                        ${conflict_b}
                        ${trans_a}
                        ${trans_b}
                        ${is_beta_zero}
                        ${gemm_type}
                        ${tir}
                        ${tic}
                        ${twr}
                        ${twc}
                        ${tlr}
                        ${tlc}
                        ${wg_size}
                        ${cl_size}
                        ${file_name}
                      MAIN_DEPENDENCY ${SYCLBLAS_SRC}/interface/${blas_level}/${func}.cpp.in
                      DEPENDS ${SYCLBLAS_SRC_GENERATOR}/py_gen_blas_gemm_launcher.py
                      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                      VERBATIM
                    )
                    list(APPEND FUNC_SRC "${LOCATION}/${file_name}")
                  endforeach(gemm_list)
                endforeach(index)
              endforeach(data)
            endforeach(executor)
          endforeach(is_beta_zero)
        endforeach(trans_b)
      endforeach(trans_a)
add_library(${func} OBJECT ${FUNC_SRC})
set_target_compile_def(${func})
# The blas library depends on FindComputeCpp
target_include_directories(${func} PRIVATE ${SYCLBLAS_SRC} ${SYCLBLAS_INCLUDE} ${THIRD_PARTIES_INCLUDE})
message(STATUS "Adding SYCL to target ${func}")
add_sycl_to_target(TARGET ${func} SOURCES ${FUNC_SRC})
endfunction(generate_blas_gemm_objects)

function (build_library LIB_NAME LIB_TYPE)
add_library(${LIB_NAME} ${LIB_TYPE}
                             $<TARGET_OBJECTS:sycl_iterator>
                             $<TARGET_OBJECTS:sycl_policy> 
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
                             $<TARGET_OBJECTS:gemv_legacy>
                             $<TARGET_OBJECTS:gemm_launcher>
                             $<TARGET_OBJECTS:gemm>
                            )
endfunction(build_library)
