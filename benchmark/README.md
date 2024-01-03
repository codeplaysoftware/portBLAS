Benchmarks
===

## General information

The portBLAS benchmarks are intended to measure the evolution of the
performance of this BLAS implementation and how it compares with other tuned
implementations, such as [CUBLAS](https://docs.nvidia.com/cuda/cublas/index.html) native and optimized CUDA BLAS library for Nvidia GPUs, 
[ROCBLAS](https://github.com/ROCm/rocBLAS) which is the native HIP/Rocm BLAS 
library optimized for AMD GPUs, and [CLBLAST](https://github.com/CNugteren/CLBlast),
a very performant OpenCL BLAS library for CPU targets. 

The benchmarks use Google's [benchmark](https://github.com/google/benchmark)
library and generate a report with indicative metrics *(see instructions below)*.

## How to compile the benchmarks
### portBLAS Benchmarks
The portBLAS default benchmarks are compiled with the project if the 
`BLAS_ENABLE_BENCHMARK` CMake option is activated *(which is the case by 
default)*. For the results to be relevant, portBLAS needs to be built in Release
mode *(Passing `-DCMAKE_BUILD_TYPE=Release` to `cmake` command)*.

### clBLAST Benchmarks
The CLBLAST benchmarks are compiled only if the `BUILD_CLBLAST_BENCHMARKS` CMake
option is activated. If so, if CLBlast cannot be found the build will fail. The
location of CLBlast can be given with `CLBLAST_ROOT`.
To install CLBlast, see:
[CLBlast: Building and installing](
https://github.com/CNugteren/CLBlast/blob/master/doc/installation.md)

### cuBLAS Benchmarks
cuBLAS benchmarks can be built by enabling the `BUILD_CUBLAS_BENCHMARKS` option and 
require an installation of CUDA *(>=11.x)* and cuBLAS library *(Both can be obtained by 
installing the cuda Toolkit : 
[installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html))*.

### rocBLAS Benchmarks
cuBLAS benchmarks can be built by enabling the `BUILD_ROCBLAS_BENCHMARKS` option 
and require an installation of Rocm *(>=4.5.x)* and rocBLAS library *(Please refer 
to this [installation guide](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/Linux_Install_Guide.html) 
for the setup)*

## General Notes
After the compilation, the binaries will be available:
* in the build folder, in `benchmark/[Library Name]/` 
* if you provide an installation directory with the CMake variable
    `CMAKE_INSTALL_PREFIX`, and run the installation command, e.g
    `ninja install`, in your installation folder, in `portblas/bin/`

A verification of the results is enabled by default and can be disabled with the
CMake option `BLAS_VERIFY_BENCHMARK` set to `OFF` or `0`. The verification will
be run a small number of times *(more than once because of the way the benchmark
library works, but much less than the usual number of iterations of the
benchmarks)*. The verification requires that a reference implementation of BLAS
like OpenBLAS is installed, which path can be given with the `CMAKE_PREFIX_PATH`
CMake parameter.

## How to run the benchmarks

The benchmarks take two kinds of command-line options: those for the benchmark
library and those specific to portBLAS: 
- Benchmark library options : these options help specify for instance the output 
format, the verbosity level and help filter specific benchmarks based on regex 
arguments. 
- portBLAS options : these options are portBLAS specific and they help specify 
the target SYCL device *(portBLAS benchmarks only)* as well as pass a custom set 
of operator-specfic configurations *(applies to rocBLAS and cuBLAS benchmarks as 
well)*. If no csv file is specified, the benchmark will use defaults values 
*(found in `include/common/common_utils.hpp`)*. 

For portBLAS and clBLASt benchmarks, and if the target machine has more than one 
OpenCL device, you can specify which one to use at runtime(\*). 

(*): Preferably, for portBLAS benchmarks, this device has to match the 
`TUNING_TARGET` as some operators are configured differently depending on the 
SYCL target for optimal performance.  

The most useful options for us are:

|option|parameter|description|
|------|:-------:|-----------|
| `--help` |   | Show help message |
| `--device` | device name | Select a device to run on (e.g `intel:gpu`), useful for portBLAS and clBLASt benchmarks|
| `--csv-param` | file path | Path to a CSV file with the benchmark parameters |
| `--benchmark_format` | `console` / `json` / `csv` | Specify the format of the standard output |
| `--benchmark_out` | file path | Specify a file where to write the report |
| `--benchmark_out_format` | `console` / `json` / `csv` | Specify the format of the file output |
| `--benchmark_list_tests` | `true` / `false` | Display the names of all the registered benchmarks and abort (one line per tuple of parameters) |
| `--benchmark_filter` | regex | Run only the benchmark which name contains a substring that matches the given regex |
| `--benchmark_min_time` | min time | Override the minimum time that the library will run the benchmark for every tuple of parameters. Useful if the benchmarks take too long |
| `--v` | verbosity | Verbose level of the benchmark library |

You can check the [GitHub repository](https://github.com/google/benchmark) of
the library for information about the other supported command-line arguments.

Here is an example of an invocation of the portBLAS GEMM benchmark running on 
Intel GPU, displaying the results in the console and saving a json report:

```bash
./benchmark/portblas/bench_gemm --device=intel:gpu --csv-param=parameters.csv \
    --benchmark_out=../results.json --benchmark_out_format=json \
    --benchmark_format=console
```

Here is the same benchmark through cuBLAS: 
```bash
./benchmark/cublas/bench_cublas_gemm --csv-param=parameters.csv --benchmark_out=../results.json \ 
--benchmark_out_format=json --benchmark_format=console
```

### CSV format

The benchmarks can be given a CSV file containing the parameters to run with
*(matrix/vector dimensions, transpose or not, etc)*, in the following format: one
line corresponds to one set of parameters, i.e. one name for the library *(though
it will be iterated many times for statistical accuracy)*.

The formats for the different BLAS levels are:

|operations|format|description|
|:--------:|:----:|-----------|
| blas 1 | *size* | Vector size |
| blas 2 | *transpose_A,m,n,alpha,beta* | Action on the matrix (`n`, `t`, `c`), dimensions, and scalars alpha and beta |
| blas 3 |  | |
| gemm | *transpose_A,transpose_B,m,k,n,alpha,beta* | Action on the matrices (`n`, `t`, `c`), dimensions (A: mk, B:kn, C: mn), and scalars alpha and beta |
| gemm (Batched) | *transpose_A,transpose_B,m,k,n,alpha,beta,batch_size,batch_type* | Action on the matrices (`n`, `t`, `c`), dimensions (A: mk, B:kn, C: mn), scalars alpha and beta, batch size, batch_type |
| trsm | *side,triangle,transpose,diagonal,m,n,alpha* | Position of A (`l`, `r`), A is upper or lower triangular (`u`, `l`), transposition of A (`n`, `t`), A is unit or non-unit diagonal(`u`,`n`),dimensions, scalar alpha |

### Notes: 

For operations that support an increment, the benchmarks will use an increment of
1 *(contiguous values)*, except for the GEMM batched operation where valid 
default increment values are used depending on `batch_type` *(strided or 
interleaved)*. For operations that support a leading dimension, the
benchmarks use the minimum possible value *(the actual leading dimension 
of the matrix)*.

For batched-strided operations that expect a stride(s), the benchmarking suite
expects **stride multipliers** instead of explicit stride values. For example, in
`gemm_batched_strided` operator, a `stride_a_mul=3` for matrix `A` of `size_a=m*k` 
is equivalent to an actual stride of `stride_a = stride_a_mul * size_a`. 

For operations that support Complex data type *(and when cmake option 
BLAS_ENABLE_COMPLEX is enabled)*, scalars such as `alpha` and `beta` are 
expected to have real and imaginary parts as separate values, if only single
scalar values are passed in the csv configuration files, complex values
for these arguments will be constructed by duplicating the same real value
for both imaginary and real parts. (e.g. alpha_cplx={alpha_real, alpha_real}). 


Here are two examples of a valid CSV files for the GEMM benchmark:

- Scalar only alpha & beta values

```
n,n,42,42,42,1,0
n,t,64,128,64,0.5,0.5
t,n,13,3,7,0,0.7
```

- Complex alpha & beta values *(two scalars each)*
```
n,n,42,42,42,1,2,3,0
n,t,64,128,64,0.5,1,0.5,1
t,n,13,3,7,0,1,0.7,3
```


The folder `config_csv` provides a few files corresponding to sizes that are
relevant for neural networks, but you can provide your own, see the next
section for more info on how to generate them.

### Python tool to generate a CSV file

If you don't yet have a file containing the parameters you want to run the
benchmarks with, we provide a Python generator in the script `gen_param.py`.
The script takes two command-line args:
* `-o` specifies the output CSV file
* `-e` specifies a generator expression in the DSL described below

The goal of the generator expression is to create an iterable where each element
is a tuple that will generate a line of the output file. This is achieved by
combining *ranges*, where ranges can be 1-dimensional (they generate one
parameter) or multidimensional (they generate a tuple).

Here are the bricks to build expressions with (you can find examples below):

|function|parameters|description|
|--------|----------|-----------|
| `size_range` | *low, high, mult* | Generates a 1-dimensional range from *low* to *high* with a multiplier *mult* |
| `value_range` | *val_1, ..., val_n* | Generates a range with exactly the given values |
| `concat_ranges` | *range_1, ..., range_n* | Generates a range with the union of all the values from all the given ranges |
| `nd_range` | *range_1, ..., range_n* | Generates a n-dimensional range by combining 1-dimensional ranges. It generates all the combinations so the cardinality of the resulting range can be really high |

#### Examples

A bunch of flat matrices:
```python
concat_ranges(nd_range(value_range(8), size_range(128, 512, 2)),
             nd_range(size_range(128, 512, 2), value_range(8)))
```
```
8,128
8,256
8,512
128,8
256,8
512,8
```

GEMV parameters:
```python
nd_range(value_range('n', 't'),
         concat_ranges(value_range(32, 64), value_range(1024)),
         size_range(128, 256, 2),
         value_range(1),
         value_range(0))
```
```
n,32,128,1,0
n,32,256,1,0
n,64,128,1,0
n,64,256,1,0
n,1024,128,1,0
n,1024,256,1,0
t,32,128,1,0
t,32,256,1,0
t,64,128,1,0
t,64,256,1,0
t,1024,128,1,0
t,1024,256,1,0
```

#### Some ranges to start with

The following ranges are a good starting point:

|blas level|range expression|
|----------|----------------|
| 1 | `nd_range(size_range(1024, 1048576, 2))` |
| 2 | `nd_range(value_range('n', 't'), size_range(128, 1024, 2), size_range(128, 1024, 2), value_range(1), value_range(0))` |
| 3 (gemm) | `nd_range(value_range('n', 't'), value_range('n', 't'), size_range(128, 1024, 2), size_range(128, 1024, 2), size_range(128, 1024, 2), value_range(1), value_range(0))` |


### Default parameters

If no CSV file is provided, the default ranges will be used, as described below.
These ranges only use powers of two and have been carefully fine-tuned to 
ensure that the problem sizes are meaningful for modern GPGPUs, and that all 
benchmarks complete within a reasonable time.

If you need to use specific sizes or run less benchmarks, you can use the CSV
parameter files as described above.

#### BLAS 1

|parameter|values|
|---------|------|
| size | 4096, 8192, ..., 1048576 |

#### BLAS 2

|parameter|values|
|---------|------|
| transpose A | `"n"`, `"t"` |
| m | 64, 128, ..., 1024 |
| n | 64, 128, ..., 1024 |
| alpha | 1 |
| beta | 0 |

#### BLAS 3
##### GEMM

|parameter|values|
|---------|------|
| transpose A | `"n"`, `"t"` |
| transpose B | `"n"`, `"t"` |
| m | 64, 128, ..., 1024 |
| k | 64, 128, ..., 1024 |
| n | 64, 128, ..., 1024 |
| alpha | 1 |
| beta | 0 |

##### TRSM
|parameter|values|
|---------|------|
| side | `"l"`, `"r"` |
| triangle | `"u"`, `"l"` |
| transpose | `"n"`, `"t"` |
| diagonal | `"u"`, `"n"` |
| m | 64, 128, ..., 1024 |
| n | 64, 128, ..., 1024 |
| alpha | 1 |

## Output files

The benchmarks can create reports as CSV or JSON files (or output text e.g to
follow the execution in the console).

The JSON output contains more information, with a `context` and a `benchmarks`
section as described below. The CSV output corresponds roughly to the
`benchmarks` section of the JSON file described below (though some context
information is printed as text before the actual CSV).

### Context

The `context` section is an object containing the following keys:
* `date`: when the benchmarks **start**. Format: `YYYY-MM-DD hh:mm:ss`
* `host_name`: the value of `$HOSTNAME`
* `executable`: relative path to the exectuable from where it was invoked
* `num_cpus`, `mhz_per_cpu`, `cpu_scaling_enabled`, `caches`: information
     provided by the benchmark library. Not always relevant, especially if the
     CL device used is not the CPU.

### Benchmarks

The `benchmarks` section contains a list of objects corresponding to each tuple
of parameters the benchmark has been run with. Every object contains the
following keys:
* `name`: name of the benchmark and parameters, separated by slashes,
    e.g `BM_Gemm<float>/n/n/64/64/64`
* `iterations`: how many times the benchmark has been run
* `real_time`: total time between the start and end of the benchmark
* `cpu_time`: actual CPU time spent running the benchmark
* `time_unit`: unit used for these times. Should be `ns`, if not please file an
    issue.
* `label`: Contains the name of the benchmark backend *(`@backend` : "portblas",
    "cublas" or "rocblas")*, and target device related informations *(`device_name`,
    `device_version`, `driver_version` etc..)* 
* `avg_event_time`: the average of the CL/SYCL event times in nanoseconds. This
    time depends on the events returned by the BLAS functions used and might not
    be accurate in some cases
* `best_event_time`: the best of the CL/SYCL event times in nanoseconds. See
    warning above.
* `avg_overall_time`: the average wall time in nanoseconds (the wall time is
    the difference between the timestamps before and after running the blas
    operation).
* `best_overall_time`: the best wall time in nanoseconds.
* `total_event_time`: this is the event time of all iterations. Warning: the
    number of iterations is variable.
* `total_overall_time`: this is the wall time of all iterations. Warning: the
    number of iterations is variable.
* `n_fl_ops`: total number of floating-point operations. It is calculated
    theoretically based on the operations that we think the benchmark is doing.
* `bytes_processed`: total number of bytes read and written in memory. It is
    calculated theoretically based on the operations that we think the benchmark
    is doing.
* other operator-specific parameters that affect the computations *(e.g `m`, `n`, 
`k` for GEMM, but `alpha` is skipped as it doens't affect the operation directly.)*
* some other keys from the benchmark library

**Note:** to calculate the performance in Gflops, you can divide `n_fl_ops` by one
of the best or average time metrics, e.g `avg_overall_time` *(the event and wall
time usually converge for large dimensions)*.
