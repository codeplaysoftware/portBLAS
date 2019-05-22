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

# blas unary function for generating source code
function(generate_blas_unary_objects blas_level func)
    set(LOCATION "${PROJECT_SOURCE_DIR}/src/interface/${blas_level}/${func}/")
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
                  ${PROJECT_SOURCE_DIR}/src//interface/${blas_level}/${func}.cpp.in
                  ${executor}
                  ${data}
                  ${index}
                  ${increment}
                  ${container0}
                  ${file_name}
                MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/src//interface/${blas_level}/${func}.cpp.in
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
endfunction(generate_blas_unary_objects)


# blas binary function for generating source code
function(syclblas_add_binary_function_instantiations)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "TEMPLATE;MODEL;OUTPUT" "")
    add_custom_command(OUTPUT ${ARG_OUTPUT}
        COMMAND ${PROJECT_SOURCE_DIR}/python_generator/jinja.py
                    --model=${ARG_MODEL} --template=${ARG_TEMPLATE}
                    --output=${ARG_OUTPUT}
        MAIN_DEPENDENCY ${ARG_TEMPLATE}
        DEPENDS ${ARG_MODEL} ${PROJECT_SOURCE_DIR}/python_generator/jinja.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM
    )
endfunction()



# blas special binary function for generating source code
function(generate_blas_binary_special_objects blas_level func)
    set(LOCATION "${PROJECT_SOURCE_DIR}/src/interface/${blas_level}/${func}/")
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
                    ${PROJECT_SOURCE_DIR}/src//interface/${blas_level}/${func}.cpp.in
                    ${executor}
                    ${data}
                    ${index}
                    ${increment}
                    ${container0}
                    ${container1}
                    ${file_name}
                  MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/src//interface/${blas_level}/${func}.cpp.in
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
endfunction(generate_blas_binary_special_objects)



# blas ternary function for generating source code
function(generate_blas_ternary_objects blas_level func)
    set(LOCATION "${PROJECT_SOURCE_DIR}/src/interface/${blas_level}/${func}/")
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
                      ${PROJECT_SOURCE_DIR}/src//interface/${blas_level}/${func}.cpp.in
                      ${executor}
                      ${data}
                      ${index}
                      ${increment}
                      ${container0}
                      ${container1}
                      ${container2}
                      ${file_name}
                    MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/src//interface/${blas_level}/${func}.cpp.in
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
endfunction(generate_blas_ternary_objects)


# blas gemm function for generating source code
function(generate_blas_gemm_objects blas_level func)
set(LOCATION "${PROJECT_SOURCE_DIR}/src/interface/${blas_level}/${func}/")
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
                    ${PROJECT_SOURCE_DIR}/src//interface/${blas_level}/${func}.cpp.in
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
                  MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/src//interface/${blas_level}/${func}.cpp.in
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
endfunction(generate_blas_gemm_objects)
