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
# *  @filename roofline/analyze.py
# *
# **************************************************************************/

"""Create a roofline analysis of the benchmark results given as command-line
   args, for a machine which theoretical performance is known.
"""

import argparse
import datetime


def main(args):
    """Entry point. Read machine info and benchmark results and exports the
       graph as an image.
    """
    graph_data = []
    for i in range(len(args.gbench_file or [])):
        title, filepath = args.gbench_file[i]
        time_type = args.time_types[min(i, len(args.time_types) - 1)]
        data_points = data.read_gbench_json(filepath, time_type)
        if data_points:
            graph_data.append((title, data_points))

    # In case no file was given, or a file failed to be read
    if not graph_data:
        print("No data to plot")
        exit(1)

    machine_data = data.read_machine_info(args.machine_id)
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    title = "%s on %s (%s)" % (args.kernel, machine_data.name, today)

    draw.draw_roofline(machine_data, graph_data, args.output_file,
                       args.x_bounds, args.y_bounds, title)


def get_args(args_str=""):
    """Parse the command line arguments (displays information if the -h or
       --help option is used)
    """
    description = "Tool to generate the roofline model."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-o", dest="output_file",
                        default="roofline.png", metavar="filepath",
                        help="Specify the name of the resulting image file")
    parser.add_argument("-g", dest="gbench_file", nargs=2, action="append",
                        metavar=("title", "filepath"),
                        help="Add a gbench-generated JSON file. Specify the"
                        " title of the graph and the file path.")
    parser.add_argument("-m", dest="machine_id", default="test",
                        metavar="machine_id",
                        help="Id of the machine (to find the specs)")
    parser.add_argument("-x", dest="x_bounds", nargs=2, default=(1, 1000),
                        type=float, metavar=("xmin", "xmax"),
                        help="x bounds for the graph (NB: it's a log scale)")
    parser.add_argument("-y", dest="y_bounds", nargs=2, default=(.01, 200),
                        type=float, metavar=("ymin", "ymax"),
                        help="y bounds for the graph (NB: it's a log scale)")
    parser.add_argument("-k", dest="kernel", default="GEMM",
                        metavar="kernel_name", help="Name of the kernel")
    parser.add_argument("-t", dest="time_types", metavar="time_type",
                        action="append",
                        help="Which time indicator to use in the results file."
                        " If you want to use different time types for different"
                        " files, use this option multiple times.")
    args = parser.parse_args(args_str) if args_str else parser.parse_args()
    args.time_types = args.time_types or ["best_event_time"]

    return args


if __name__ == "__main__":
    import draw
    import data
    main(args=get_args())
else:
    from . import draw
    from . import data
