import os
import re
import subprocess
import pickle
import sys

def execution(variants, path, datestring, size = 1200):
  """ Executes the given variants of the benchmark """
  results = {}
  platform_info = {}
  # Execution
  for model,d in variants.iteritems():
      binary = d['binary']
      exe_file = path + binary
      print "Running: %s"%(exe_file)
      if not os.path.exists(exe_file):
        print "Combination not found"
        continue

      popen = subprocess.Popen([exe_file, str(size)], stdout=subprocess.PIPE)
      popen.wait()
      output = popen.stdout.read();
      with open(d['file'] + '-' + datestring, "w") as f:
        f.writelines(output)
      results[model] = d['output_process_func'](output)
  return results

def gather_results(variants, results):
  """ Gather results and organises them per-experiment """
  # Gathering results
  per_exp = {}
  found_variants = []
  for model in variants:
      if not model in results:
        print "Skipping from results : %s"%(model)
        continue
      else:
        found_variants.append(model)

      for experiment,result in results[model].iteritems():
          if per_exp.has_key(experiment):
              per_exp[experiment].append( ((model, result)) )
          else:
              per_exp[experiment] = [ ((model, result)) ];
  return [per_exp, found_variants]

def gather_speedup(variants, results, base_model):
  """ Gather results and organises them per-experiment """
  # Gathering results
  per_exp = {}
  found_variants = []

  def speedup(experiment, value):
    """ Compute the speedup of value over the baseline """
    baseline = results[base_model][experiment];
    return float(baseline) / float(value)

  for model in variants:
      if not model in results:
        print "Skipping from results : %s"%(model)
        continue
      else:
        if not model == base_model:
          found_variants.append(model)

      for experiment,result in results[model].iteritems():
          if model == base_model:
              pass
          if per_exp.has_key(experiment):
              per_exp[experiment].append( ((model, speedup(experiment, result))) )
          else:
              per_exp[experiment] = [ ((model, speedup(experiment, result))) ];
  return [per_exp, found_variants, "Speedup", "Speedup over " + base_model]


