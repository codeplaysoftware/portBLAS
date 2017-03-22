from utils import *
import os

def plot_graph(per_exp, found_variants, output_file_name, platform_info):
  """ Plots the output using matplot lib into a file """
  import matplotlib as mpl
  mpl.use('Agg')

  import matplotlib.pyplot as plt; 
  plt.rcdefaults()
  import numpy as np
  import matplotlib.pyplot as plt
   
  y_pos = np.arange(len(per_exp.keys()))
  performance_bars_height = {}
  model_color = [ 'g', 'r', 'w', 'b' ]
  for v in found_variants:
    if not v in performance_bars_height:
      performance_bars_height[v] = []

    for exp in per_exp:
      performance_bars_height[v].extend(
          [ float(val[1]) for val in per_exp[exp] if val[0] == v  ])

  width = 0.3
  i = -1;
  for v in performance_bars_height:
    plt.bar(y_pos + ( (i + 0.5) * width), performance_bars_height[v], 
              width, align='center', alpha=0.5, label = v,
              color = model_color[i])
    i += 1

  plt.xticks(y_pos, per_exp.keys())
  plt.legend(loc="lower right")
  plt.xlabel("Experiments")
  plt.ylabel('Time')
  plt.suptitle('Time per function')
  plt.title(platform_info)
  plt.savefig(output_file_name)



def process_gpu_sycl(stdout):
    """ Process the stdout of the SYCLBLAS papre benchmarks.
        stdout is a string containing multiple lines.
    Example output:

    t_copy , 0.000486338, 0.000144124, 9.4301e-05, 6.7637e-05
    t_axpy , 0.000449574, 0.000166723, 0.000120264, 9.2823e-05
    t_add  , 0.000529654, 0.000364608, 0.00034186, 0.0002719
    """
    headers = []
    values = {}
    lines = stdout.splitlines(True)
    for l in lines:
        # Field Delimiter
        fd = "\s*,\s*"
        p = re.compile("(?P<name>t_[A-Za-z]+)" + fd + "(?P<first>[0-9]+\.[0-9]+e?-?[0-9]*)"
                        + fd + "(?P<normal>[0-9]+\.[0-9]+e?-?[0-9]*)"
                        + fd + "(?P<twofold>[0-9]+\.[0-9]+e?-?[0-9]*)"
                        + fd + "(?P<fourfold>[0-9]+\.[0-9]+e?-?[0-9]*).*");
        m = p.match(l)
        if m:
            values[m.group("name")] = m.group("first");

    return values
 
def process_gpu_clblas(stdout):
    """ Process the stdout of the SYCLBLAS papre benchmarks.
        stdout is a string containing multiple lines.
    Example output:
    t_copy, 0.000111871
    t_axpy, 0.000119333
    t_add, 0.00017468
    """
    headers = []
    values = {}
    lines = stdout.splitlines(True)
    for l in lines:
        # Field Delimiter
        fd = "\s*,\s*"
        p = re.compile("(?P<name>t_[A-Za-z]+)" + fd + "(?P<first>[0-9]+\.[0-9]+e?[0-9]*)")
        m = p.match(l)
        if m:
            values[m.group("name")] = m.group("first");
    return values



variants = { 'ocl' : 
                { 
                    'binary' : "paper_clblas_test", 
                    'file' : "paper_clblas.out",
                    'output_process_func' : process_gpu_clblas
                }, 
             'sycl' : 
                { 
                    'binary' : "paper_blas1_test", 
                    'file' : "paper_blas1.out",
                    'output_process_func' : process_gpu_sycl
                } 
            }

import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run benchmark")
  parser.add_argument("path", metavar="path", type=str,
                      help="Path to the benchmark")
  parser.add_argument("description", metavar="description", type=str,
                      help="Description of the benchmark")
  parser.add_argument("--save", dest="save", help="Save the results",
                      type=str)
  parser.add_argument("--load", dest="load", help="Load the results",
                      type=str)
  params = parser.parse_args()

  # Path to the benchmark
  path = params.path
  import datetime
  now = datetime.datetime.now()

  date_string = "%d_%d_%d-%d:%d:%d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)

  if params.save and params.load:
    print " Cannot load and save at the same time "
    sys.exit()
  elif params.save:
    output_file_name_dump = params.save + date_string
  elif params.load:
    output_file_name_dump = params.load

  output_file_name = "sycl-blas--" + date_string
  output_file_name_png = output_file_name + ".png"
  per_exp = {}
  found_variants = []
  platform_info = []

  # Gathers the result
  if params.load:
    with open(output_file_name_dump, 'rb') as f:
      per_exp = pickle.load(f)
      found_variants = pickle.load(f)
      platform_info = pickle.load(f)
  else:
    # Executes the benchmark
    platform_info = os.uname()
    results = execution(variants, path, date_string)
    print results
    [per_exp, found_variants] = gather_results(variants, results)

  if params.save and not params.load:
    with open(output_file_name_dump, 'w') as f:
      pickle.dump(per_exp, f)
      pickle.dump(found_variants, f)
      pickle.dump(platform_info, f)

  # Plots the data
  plot_graph(per_exp, found_variants, output_file_name_png, params.description)
