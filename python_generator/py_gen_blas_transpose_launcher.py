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
# *  SYCL-BLAS: BLAS implementation using SYCL
# *
# *  @filename py_gen_blas_transpose_launcher.py
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
    in_place = sys.argv[8]
    tile_size = sys.argv[9]
    local_memory = sys.argv[10]
    file_name = sys.argv[11]

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
            key='IN_PLACE',
            vals=[in_place],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='TILE_SIZE',
            vals=[tile_size],
            itermode=Itermode.combinations,
            iter_modifier=1),
        Iterable(
            key='LOCAL_MEM',
            vals=[local_memory],
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
