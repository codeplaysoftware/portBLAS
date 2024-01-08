# /***************************************************************************
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
# *  @filename py_gen_blas_gemm_launcher.py
# *
# **************************************************************************/
# py_gen import
import errno
import os
import sys

if __name__ == '__main__':

    generator_path = sys.argv[1]
    sys.path.insert(0, generator_path)
    from py_gen import generate_file
    from py_gen import *
    from string import Template
    input_template = sys.argv[2]
    blas_level_name = sys.argv[3]
    blas_function_name = sys.argv[4]
    blas_template_impl = sys.argv[5]
    data = sys.argv[6]
    index = sys.argv[7]
    double_buffer = sys.argv[8]
    conflict_a = sys.argv[9]
    conflict_b = sys.argv[10]
    trans_a = sys.argv[11]
    trans_b = sys.argv[12]
    is_beta_zero = sys.argv[13]
    gemm_memory_type = sys.argv[14]
    gemm_shape_type = sys.argv[15]
    tir = sys.argv[16]
    tic = sys.argv[17]
    twr = sys.argv[18]
    twc = sys.argv[19]
    tsr = sys.argv[20]
    tsc = sys.argv[21]
    tlr = sys.argv[22]
    tlc = sys.argv[23]
    tib = sys.argv[24]
    twb = sys.argv[25]
    jm_m = sys.argv[26]
    jm_n = sys.argv[27]
    jm_k = sys.argv[28]
    jm_in_type = sys.argv[29]
    jm_out_type = sys.argv[30]
    wg_size = sys.argv[31]
    cl_size = sys.argv[32]
    file_name = sys.argv[33]
    gemm_vectorize_type = sys.argv[34]
    vector_size = sys.argv[35]
    batch_type = sys.argv[36]
    use_joint_matrix = sys.argv[37]
    symm_a = sys.argv[38]
    symm_b = sys.argv[39]
    source = 'generated_src/' + blas_level_name + '/' + blas_function_name + '/'
    try:
        os.makedirs(source)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    f = open(blas_template_impl, "r")
    template = Template(f.read())
    f.close()
    iterables = [
        Iterable(
            key='WG_SIZE',
            vals=[wg_size],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='DOUBLE_BUFFER',
            vals=[double_buffer],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='CONFLICT_A',
            vals=[conflict_a],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='CONFLICT_B',
            vals=[conflict_b],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='CL_SIZE',
            vals=[cl_size],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='TRANS_A',
            vals=[trans_a],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='TRANS_B',
            vals=[trans_b],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='TIR',
            vals=[tir],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='TIC',
            vals=[tic],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='TWR',
            vals=[twr],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='TWC',
            vals=[twc],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='TSR',
            vals=[tsr],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='TSC',
            vals=[tsc],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='TLR',
            vals=[tlr],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='TLC',
            vals=[tlc],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='TIB',
            vals=[tib],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='TWB',
            vals=[twb],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='JM_M',
            vals=[jm_m],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='JM_N',
            vals=[jm_n],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='JM_K',
            vals=[jm_k],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='JM_IN_T',
            vals=[jm_in_type],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='JM_OUT_T',
            vals=[jm_out_type],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='IS_BETA_ZERO',
            vals=[is_beta_zero],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='GEMM_MEMORY_TYPE',
            vals=[gemm_memory_type],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='GEMM_SHAPE_TYPE',
            vals=[gemm_shape_type],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='DATA_TYPE',
            vals=[data],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='INDEX_TYPE',
            vals=[index],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='GEMM_VECTORIZE_TYPE',
            vals=[gemm_vectorize_type],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='VECTOR_SIZE',
            vals=[vector_size],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='BATCH_TYPE',
            vals=[batch_type],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='USE_JOINT_MATRIX',
            vals=[use_joint_matrix],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='SYMM_A',
            vals=[symm_a],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='SYMM_B',
            vals=[symm_b],
            itermode=Itermode.combinations,
            iter_modifier=1)
    ]
    iter_groups = [IterGroup('@ip1@', template, iterables, combine_iters=True)]
    generate_file(
        input_template,
        source + file_name,
        iter_groups,
        format_generated=False,
        format_script="")
