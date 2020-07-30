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
# *  @filename py_gen_quantize.py
# *
# **************************************************************************/
# py_gen import
import errno
import os
import sys


def generate(argv):
    generator_path = argv[1]
    sys.path.insert(0, generator_path)
    from py_gen import generate_file, Iterable, Itermode, IterGroup
    from string import Template

    input_template = argv[2]
    blas_template_impl = sys.argv[3]
    executor = argv[4]
    data = argv[5]
    file_name = argv[6]
    source = 'generated_src/quantize/'

    try:
        os.makedirs(source)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    f = open(blas_template_impl, "r")
    template = Template(f.read())
    f.close()
    iterables = [
        Iterable(key='EXECUTOR',
                 vals=[executor],
                 itermode=Itermode.combinations,
                 iter_modifier=1),
        Iterable(key='DATA_TYPE',
                 vals=[data],
                 itermode=Itermode.combinations,
                 iter_modifier=1),
    ]
    iter_groups = [IterGroup('@ip1@', template, iterables, combine_iters=True)]
    generate_file(input_template,
                  source + file_name,
                  iter_groups,
                  format_generated=False,
                  format_script="")


if __name__ == '__main__':
    generate(sys.argv)
