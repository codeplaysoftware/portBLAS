#!/usr/bin/env python3
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
# *  @filename gen_param.py
# *
# **************************************************************************/

"""This tool generates CSV parameter files for the BLAS benchmarks, based on
   expressions written in a domain-specific language.
   See the documentation in README.md for more information.
"""

import itertools
import argparse

def main(args):
    """Generate the csv file according to the given arguments
    """
    # Match DSL to Python names
    nd_range = itertools.product
    value_range = lambda *v: list(v)
    def size_range(low, high, mult):
        val = low
        while val <= high:
            yield val
            val *= mult
    concat_ranges = itertools.chain

    gen_machine = eval(args.expr)

    with open(args.output_file, "w") as f_write:
        for line in gen_machine:
            f_write.write(",".join(map(str, line)) + "\n")


def get_args(args_str=""):
    """Parse the command line arguments (displays information if the -h or
       --help option is used)
    """
    description = ("Tool to generate a csv file containing the parameters for "
                   " the benchmarks")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-o", dest="output_file",
                        default="params.csv", metavar="filepath",
                        help="Specify the name of the resulting CSV file")
    parser.add_argument("-e", dest="expr", metavar="expression", required=True,
                        help="Expression used to generate the file, in a"
                        " domain-specific language (see README.md)")
    args = parser.parse_args(args_str) if args_str else parser.parse_args()
    return args


if __name__ == "__main__":
    main(args=get_args())
