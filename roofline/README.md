Python tool for roofline analysis
===

This tool uses the data generated from our benchmarks to generate a graph
with the roofline model.

For more information about the roofline model, see
[its Wikipedia page](https://en.wikipedia.org/wiki/Roofline_model).

## Installation

You need Python 3. You can install the dependencies with
[pip](https://pypi.org/project/pip/) with the following command:

```bash
pip3 install -r requirements.txt
```

## How to use

The script `analyze.py` takes the following options:

| option | parameters | description | default |
|--------|------------|-------------|---------|
| -h |   | show a help message and exit | |
| -o | filepath | specify the name of the resulting image file | roofline.png |
| -g | title filepath | Add a gbench-generated JSON file. Specify the title of the graph and the file path. Use this option once per JSON file. | |
| -m | machine_id | Identifier of the machine (to find the specs) | test |
| -x | xmin xmax | x bounds for the graph (NB: it's a log scale) | 1 1000 |
| -y | ymin ymax | y bounds for the graph (NB: it's a log scale) | 0.01 200 |
| -k | kernel_name | Name of the kernel | GEMM |
| -t | time_type | Which time indicator to use in the results file. If you want to use different time types for different files, use this option multiple times. | best_event_time |

Example of use:

```bash
python3 analyze.py -o syclblas_vs_clblast.png \
    -g syclblas syclblas.json -t best_overall_time \
    -g clblast clblast.json -t best_event_time
```

## Documentation of the tool

To generate the documentation, you need the module `pdoc3` that
you can install with:

```bash
pip3 install pdoc3
```

You can then run `generate_doc.sh` from the `roofline` folder.
