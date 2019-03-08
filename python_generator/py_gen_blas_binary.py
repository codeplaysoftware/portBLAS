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
# *  @filename py_gen_blas_binary.py
# *
# **************************************************************************/

# py_gen import
import sys
import os

if __name__ == '__main__':

    generator_path = sys.argv[1]
    sys.path.insert(0, generator_path)
    from py_gen import generate_file
    from py_gen import *
    from string import Template

    executor_list = ['PolicyHandler<codeplay_policy>']
    index_list = ['int', 'long', 'long long']
    data_list = ['float', 'double']

    input_template = sys.argv[2]
    blas_level_name = sys.argv[3]
    blas_function_name = sys.argv[4]
    blas_template_impl = sys.argv[5]
    executor = sys.argv[6]
    data = sys.argv[7]
    index = sys.argv[8]
    increment = sys.argv[9]
    container0 = sys.argv[10]
    container1 = sys.argv[11]
    file_name = sys.argv[12]
    source = 'generated_src/' + blas_level_name + '/' + blas_function_name + '/'

    if not os.path.exists(source):
        os.makedirs(source)
    f = open(blas_template_impl, "r")
    template = Template(f.read())
    f.close()
    iterables = [
        Iterable(
            key='EXECUTOR',
            vals=[executor],
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
            key='INCREMENT_TYPE',
            vals=[increment],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='container_t0',
            vals=[container0],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='container_t1',
            vals=[container1],
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
