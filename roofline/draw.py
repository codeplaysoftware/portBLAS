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
# *  @filename roofline/draw.py
# *
# **************************************************************************/

"""Provide an abstraction to draw the graph and save it in a file.
"""

import numpy as np

# Matplotlib initialization
import matplotlib
backends = matplotlib.rcsetup.all_backends
matplotlib.use("agg" if "agg" in backends else backends[0])
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def draw_roofline(machine_data, data_points, output_file, x_bounds, y_bounds,
                  graph_title):
    """Draws the roofline graph for the given data with Matplotlib.
    """
    # Initialization
    fig, axes = plt.subplots()

    # Plotting data for each input file
    i = 0
    colors = "brgmcyk"
    for title, kernel_points in data_points:
        flop = np.array([kp.flop for kp in kernel_points])
        seconds = np.array([kp.seconds for kp in kernel_points])
        mem_bytes = np.array([kp.bytes for kp in kernel_points])
        x = flop / mem_bytes
        y = flop / seconds / 10**9
        axes.plot(x, y, "%s+" % colors[i % 7], label=title)
        i += 1

    # Axes and legend setup
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set(xlabel='Operational intensity (flop/byte)',
             ylabel='Performance (Gflop/s)',
             title=graph_title)
    axes.grid()
    axes.legend(loc=1)
    axes.set_xlim(x_bounds)
    axes.set_ylim(y_bounds)

    # Machine-specific rooftop lines
    # Hard limits
    draw_linear(axes, 10**-9 * machine_data.bandwidth, 0)  # Memory roof
    draw_linear(axes, 0, machine_data.maxGflops)           # Compute roof
    # Other perf limits
    for bandwidth in machine_data.mem_limits:
        draw_linear(axes, 10**-9 * bandwidth, 0, dashed=True)
    for maxGflops in machine_data.perf_limits:
        draw_linear(axes, 0, maxGflops, dashed=True)

    # Final export
    fig.savefig(output_file, bbox_inches="tight")
    print("Roofline graph exported as %s" % output_file)


def draw_linear(axes, a, b, dashed=False):
    """Draw the line y=ax+b on the given axes, stretching to the x axis bounds.
       Don't modify the x axis bounds after calling this function.
    """
    x1, x2 = axes.get_xbound()
    y1 = a * x1 + b
    y2 = a * x2 + b
    line = mlines.Line2D([x1, x2], [y1, y2], color=(0, 0, 0),
                         linestyle="--" if dashed else "-")
    axes.add_line(line)
