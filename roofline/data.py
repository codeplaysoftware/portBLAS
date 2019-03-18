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
# *  SYCL-BLAS: BLAS implementation using SYCL
# *
# *  @filename roofline/data.py
# *
# **************************************************************************/

"""Provide helpful functions to read the results of benchmarks, and the specs
   of the testing machines (hardcoded at the moment)
"""

import json
from collections import namedtuple


# Defining data types
KernelPoint = namedtuple("KernelPoint", "name flop seconds bytes")
MachineData = namedtuple("MachineData",
                         "name bandwidth maxGflops mem_limits perf_limits")


def read_gbench_json(filepath, time_type):
    """Read the results exported by Google Benchmarks under the JSON format,
       and return them as a list of KernelPoints.
    """
    try:
        with open(filepath, 'r') as f_read:
            json_data = json.load(f_read)
    except IOError:
        print("The file %s couldn't be read. It will be ignored." % filepath)
        return None
    except json.decoder.JSONDecodeError as error:
        print("The file %s is not valid JSON. It will be ignored. (error: %s)"
              % (filepath, error.msg))
        return None

    kernel_points = []
    for benchmark in json_data["benchmarks"]:
        try:
            kernel_points.append(
                KernelPoint(name=benchmark["name"],
                            flop=int(benchmark["n_fl_ops"]),
                            seconds=benchmark[time_type] * 10**-9,
                            bytes=int(benchmark["bytes_processed"])))
        except KeyError as error:
            print("The file %s is missing a parameter %s. It will be ignored."
                  % (filepath, error.args[0]))
            return None

    return kernel_points


def read_machine_info(machine_id):
    """Read the machine information for the given machine ID and return it as a
       MachineData variable.
    """
    # Temporary: hardcoded machine info
    machines = {
        "test": MachineData(bandwidth=34.128 * 10**9, maxGflops=460.8,
                            mem_limits=[], perf_limits=[], name="Intel NEO"),
        "r-car-v3h": MachineData(bandwidth=1.583 * 10**9, maxGflops=102,
                                 mem_limits=[0.4 * 10**9], perf_limits=[75.8],
                                 name="R-Car V3H"),
    }

    if machine_id not in machines:
        print("Error: Machine with id %s doesn't exist." % machine_id)
        exit(1)

    return machines[machine_id]
