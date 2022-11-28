# SYCL-BLAS Autotuner Developer Documentation
This documentation aims to cover the inner workings of the autotuner from a developer perspective.
It should be useful if you need to work on or extend the autotuner in the future, or just want to better understand how it works.

For information on building and using the autotuner please see the readme [here](../tools/auto_tuner/README.md).
# Table of Contents
- [**Overview**](#overview)
- [**C++ Binaries**](#c++-binaries)
- [**Parameter Configurations**](#parameter-configurations)
- [**Python Generation**](#python-generation)
    - [generated_combinations.def](#generated_combinations.def)
    - [Instantiation Source Files](#instantiation-source-files)
- [**CMake**](#cmake)
- [**Common Tasks**](#common-tasks)
    - [Adding a new set of configurations](#adding-a-new-set-of-configurations)

# Overview
The Autotuner is comprised of several parts:
- The `C++ binaries` that are run by the user to do the tuning. 
Comprised of five different binaries which benchmark different combinations of transposition of the inputs and print the results.
- `Python scripts` which generate macro calls and source files for each GEMM configuration specified.
- `JSON files` which specify the parameter combinations to test when tuning.
# C++ Binaries
There are five different binaries produced in total: four of which (`tune_nn/nt/tn/tt`) run different combinations of whether one or both of the input matrices are transposed. 
The fifth binary (`tune_all`) runs all four combinations one after the other.

The bulk of the work is done inside `gemm_tuner.hpp` in the `run_tune_gemm()` function. 
This function is called by each `tune_` binary with different combinations of template parameters for transposition.
Before testing the generated combinations the application first benchmarks the provided `system BLAS`, along with the current `SYCL-BLAS`. 
The generated combinations are then run with the following macro:
```c++
#define BENCH_PARAMS(MEM, ALG, BATCH, VEC, ...)                             \
  do {                                                                      \
    auto result =                                                           \
        tune<__VA_ARGS__, GemmConfig<TransA, TransB, MEM, ALG, BATCH, VEC>, \
             DataType>(rep, args);                                          \
    results.push_back(result);                                              \
  } while (0);

#include "generated_combinations.def"

#undef BENCH_PARAMS
```
The file `generated_combinations.def` contains calls to `BENCH_PARAMS` for each combination to be tested,
and the stored results are sorted from worst to best before being printed.
For more information on this file and how it is generated, see [this section](#generated_combinations.def).

There are several wrapper functions which wrap the function `run_tune()`: `tune()` and `tune_syclblas()`. 
The `run_tune()` function takes a function object, runs it and measures the execution time. 
The `tune()` wrapper function is used in the above macro, and `tune_syclblas()` is used for testing the current SYCL-BLAS. 

Calling `run_tune()` happens directly in `run_tune_gemm()` for testing the system BLAS.

# Parameter Configurations
Parameter configurations are provided as JSON files. 
The autotuner includes configs for various target backends along with a set of default configs.

Detailed information about the config parameters can be found in the autotuner's [readme](../tools/auto_tuner/README.md#configuration).
# Python Generation
The python script `tools/auto_tuner/gen/generate_combinations.py` produces two types of output:
- A single file, `generated_combinations.def`, which contains generated calls to the `BENCH_PARAMS` macro.
  As seen earlier this file is included inside the `run_tune_gemm()` function after the `BENCH_PARAMS` macro is defined.
  This handles the execution of each test configuration.
- A number of `C++` source files which instantiate calls to the `tune<>` method, with one file for each configuration.
  This keeps the kernels for each configuration in separate translation units.

Most of the work in the python script is in parsing the configurations from `JSON`. 
```python
gemm_configs = get_gemm_configs_from_json(args.config)
```
This line from the main function shows calling `get_gemm_configs_from_json()` passing in the desired `JSON` config file path.
That function then loads the `JSON` and begins parsing it.
The top level of `JSON` objects names the `GEMM` algorithm to be used.
See the following excerpt from `get_gemm_configs_from_json()`
```python
gemm_configs = []
    config_json = json.load(json_file)
    for r in config_json.get("local", []):
        gemm_configs += generate_local_gemm_configs(
            r["cache_line_size"], r["work_item_sizes"], r["work_group_sizes"],
            r["block_level_tiles"], r["double_buffer"],
            r["no_bank_conflict_a"], r["no_bank_conflict_b"],
            r["vectorization_size"])
```
Here we loop over every config inside the `local` object (if it exists) and call `generate_local_gemm_configs()` for each one.
There are several different functions, one for each algorithm (except for naive which doesn't require one).
For example, `generate_no_local_gemm_stride_configs()` for no local memory, strided configurations.
Each function takes different amounts of parameters, as not all are relevant for every algorithm, with the `local` configs taking the most.

The purpose of these `generate_` functions is to create the configurations, and validate them before adding them to the list.
For example, here is `generate_local_gemm_configs()`:
```python
def generate_local_gemm_configs(cache_sizes, item_sizes, group_sizes,
                                tile_sizes, double_buffers, bank_conflicts_a,
                                bank_conflicts_b, vec_sizes):
    """
    Generate a list of possible configurations using local memory.

    The parameters match those to the `Gemm` kernel. Each parameter should be a
    list of possible values to use when generating the configurations. Only
    those configurations which would give valid kernels will be generated.
    """

    configs = []
    for cls, item, wg, tl, db, ncba, ncbb, vs in product(
            cache_sizes, item_sizes, group_sizes, tile_sizes, double_buffers,
            bank_conflicts_a, bank_conflicts_b, vec_sizes):
        new_config = LocalGemm(cache_size=cls,
                               tile=_construct_tile(item, wg, tl),
                               double_buffer=db,
                               bank_conf_a=ncba,
                               bank_conf_b=ncbb,
                               vec_size=vs)
        if new_config.is_valid():
            configs.append(new_config)
    return configs
```
Here we are simply constructing an object of the `LocalGemm` class and calling its `is_valid()` method to check validity.
This method should mirror any restrictions present in the kernel itself (usually in the form of `static_asserts`).
For example, in `gemm_no_local_partial_vec.hpp` the following assserts are present:
```c++
static_assert(wg_cols * item_cols == item_rows * wg_rows,
                "Work group size should be a multiple "
                "of the number of rows in a block\n"
                " --- this is ensured iff: item_rows | wg_cols");

  static_assert(item_rows % packetize_t::packet_size == 0,
                "Item rows must be a multiple of the vector packet size");
  static_assert(item_cols % packetize_t::packet_size == 0,
                "Item cols must be a multiple of the vector packet size");
```
and you can see these mirrored in the `is_valid()` method of the `NonLocalGemmStrided` class (note that this also covers the fully vectorized version of the `no_local` kernel as well, through checking the `vec_type`):
```python
def is_valid(self):
        """
        Check whether this config is valid and can be compiled.

        The requirements here should match any asserts in the kernel.
        """
        tile_valid = (self.tile.group_cols *
                      self.tile.item_cols == self.tile.group_rows *
                      self.tile.item_rows)
        vec_size_valid = False
        if (self.vec_type == 'partial'):
            vec_size_valid = (self.tile.item_rows % self.vec_size == 0
                              and self.tile.item_cols % self.vec_size == 0)
        else:
            vec_size_valid = (self.tile.item_rows == self.vec_size
                              and self.tile.item_cols == self.vec_size)
        return (tile_valid and vec_size_valid)
```

The final component here is the `GemmParams` class which stores gemm parameters and also converts them to a string for writing to both types of output files.

```Python
class GemmParams(
        namedtuple('GemmParams', [
            'cache_size', 'tile', 'double_buffer', 'bank_conf_a',
            'bank_conf_b', 'mem_type', 'algo_type', 'batch_type', 'vec_type',
            'vec_size'
        ])):
    """ A parameter set for the non-local memory GEMM kernel.  """
    def __str__(self):
        return "{mem}, {algo}, {batch}, {vec}, {vec_s}, {cls}, {tile}, {db}, {bca}, {bcb}".format(
            cls=self.cache_size,
            tile=self.tile,
            db=_bool_to_str(self.double_buffer),
            bca=_bool_to_str(self.bank_conf_a),
            bcb=_bool_to_str(self.bank_conf_b),
            mem=_BLAS_MEM_STRING[self.mem_type],
            algo=_BLAS_ALGO_STRING[self.algo_type],
            batch=_BLAS_BATCH_TYPE[self.batch_type],
            vec=_BLAS_VECTORIZATION_TYPE[self.vec_type],
            vec_s=self.vec_size)

    def to_xmacro(self):
        return "BENCH_PARAMS({})".format(self)
```
## generated_combinations.def
The function `write_output_definition_file()` creates the file `generated_combinations.def`.
The function definition looks as follows:
```python
def write_output_definition_file(config_list, config_source, output_file):
    """
    Generate X-macro definition file of GEMM configurations.

    Args:
        config_list (`list` of `GemmParams`): List of GEMM configurations to
            provide definitions for.
        config_source (str): Source of the configuration list to add to the
            generated file.
        output_file (file object): File object to write definitions to.
    """
    output_strings = [
        "// **** FILE AUTOGENERATED BY gen/generate_combinations.py ****",
        "// Config from: {}".format(config_source),
        "#ifndef BENCH_PARAMS",
        "#error XMacro file expects BENCH_PARAMS macro to be defined",
        "#endif",
    ]
    output_strings += [conf.to_xmacro() for conf in config_list]
    output = "\n".join(output_strings)
    output_file.write(output)
```
You can see here that `to_xmacro()` is called for every config, this produces a `BENCH_PARAMS(...)` string containing the appropriate GEMM arguments for that configuration.
The generated strings look like this:
```c++
BENCH_PARAMS(::blas::gemm_memory_t::no_local, ::blas::gemm_algorithm_t::standard, ::blas::gemm_batch_type_t::strided, ::blas::gemm_vectorization_t::partial, 1, 0, ::blas::Tile<4, 4, 4, 4, 1, 1, 1, 1, 1, 1>, false, false, false)
```
These strings are then written to the output file.

## Instantiation Source Files
For generating the source files, where `tune()` is instantiated for each config, let's look at the `write_source_files()` function:
```python
def write_source_files(config_list, config_source, output_dir):
    """
    Generate source files that provide GEMM kernels for given configurations.

    Args:
        config_list (`list` of `GemmParams`): List of GEMM configurations to
            provide kernels for in the generated source files.
        config_source (str): Source of the configuration list to add to
            generated files.
        output_dir (str): Directory path to write files into.
    """
    base_file_strings = [
        "// File generated by gen/generate_combinations.py",
        "// Config from: {}".format(config_source),
        r'''
#include "tune_impl.hpp"

#define INSTANTIATE_TUNE(DTYPE, TRA, TRB, MEM, ALGO, BATCH, VEC, ...)    \
  template TestResultEntry                                   \
  tune<__VA_ARGS__, GemmConfig<TRA, TRB, MEM, ALGO, BATCH, VEC>, DTYPE>( \
  int r, GemmArgs<DTYPE> a);

#define BENCH_PARAMS(MEM, ALGO, BATCH, VEC, ...) \
  INSTANTIATE_TUNE(float, false, false, MEM, ALGO, BATCH, VEC, __VA_ARGS__) \
  INSTANTIATE_TUNE(float, true, false, MEM, ALGO, BATCH, VEC ,__VA_ARGS__)  \
  INSTANTIATE_TUNE(float, false, true, MEM, ALGO, BATCH, VEC, __VA_ARGS__)  \
  INSTANTIATE_TUNE(float, true, true, MEM, ALGO, BATCH, VEC ,__VA_ARGS__)
''',
    ]
    for config in config_list:
        filename = os.path.join(output_dir, _get_filename(config))
        output_strings = base_file_strings + [config.to_xmacro()]
        with open(filename, 'w+') as f:
            f.write("\n".join(output_strings))
```
You can see that this is very similar to the previous function with some key differences.
The initial `base_file_strings` define some extra macros to control instantiation inside the source file.
`BENCH_PARAMS` is reused as a macro name here in order to use the same `to_xmacro()` method for generating both types of file.
This then calls `INSTANTIATE_TUNE` for all combinations of tranpose, which itself contains the actual instantiation of `tune()`.
Finally each config is written to a separate output file.

# CMake
The generation of the definitions and source files is driven through CMake, which calls the python script at several stages:
```CMake
add_custom_command(OUTPUT ${tuner_def_file}
  COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/gen/generate_combinations.py
    ${GEN_CONFIG} ${tuner_def_file}
  MAIN_DEPENDENCY ${GEN_CONFIG}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen/generate_combinations.py
  COMMENT "Generating tuning configurations"
  VERBATIM
)
add_custom_target(tuner_generate_def_file DEPENDS ${tuner_def_file})
```
First the `generated_combinations.def` file is generated (see [the previous section](#generated_combinations.def) for more information). This file is then added to a custom target which is later added as a dependency for each C++ binary later on.
```CMake
set(_gen_src_dir ${CMAKE_CURRENT_BINARY_DIR}/gen)
# older version of python cannot generate the dir when it does not exist
file(MAKE_DIRECTORY ${_gen_src_dir})
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} ${_generator_script}
    ${GEN_CONFIG} --source_dir=${_gen_src_dir} --list_files
  OUTPUT_VARIABLE _gen_src_files
  RESULT_VARIABLE _gen_src_files_exitcode
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
```
Then the script is run once more with the `--list_files` option passed. This will output a list of the source files that will be generated in the next step.
```CMake
add_custom_command(OUTPUT ${tuner_kernel_srcs}
  COMMAND ${PYTHON_EXECUTABLE} ${_generator_script}
    ${GEN_CONFIG} --source_dir=${_gen_src_dir}
  MAIN_DEPENDENCY ${GEN_CONFIG}
  DEPENDS ${_generator_script}
)
add_custom_target(tuner_generate_kernels DEPENDS ${tuner_kernel_srcs})
```
The third execution of the script generates the source file instantiations (covered in [this section](#instantiation-source-files))
and adds the list of sources to another custom target.
The source files are added as a library, `tuner_kernel_lib` and the `tuner_generate_kernels` target is added as a dependency of this library.

# Common Tasks
## Adding a new set of configurations
In order to add a new set of parameter configurations to the autotuner you must do the following things:

1. Add a new `.json` file in `tools/auto_tuner/gen/` with your configurations inside. 
    Look at others for reference or see [this section](../tools/auto_tuner/README.md#configuration) of the readme for details.
2. In the autotuner's `CMakeLists.txt` you will find some code like this:
    ```cmake
    # The generator's configuration file - add any new jsons to this
    set(GEN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/gen/default.json)
    if(${TUNING_TARGET} STREQUAL "RCAR")
    set(GEN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/gen/rcar.json)
    endif()
    ```
    with `if`s for all the tuning targets. If you are adding a new tuning target you will need some mechanism to decide when to set `GEN_CONFIG` with your new configuration.
3. That's it, your new configuration should be used when building the autotuner when passing the new target or through whatever mechanism you have added!
