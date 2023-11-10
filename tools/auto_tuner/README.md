GEMM Parameter Tuner
====================

The GEMM Parameter Tuner is a standalone tool that can calculate the optimal
parameters for a GEMM operation through experimentation.

The Tuner is provided `M`, `N` and `K` values, iterates through a number of
potential configurations and then prints a list of them and their performance.

Building
--------

1. Clone the portBLAS repository, making sure to pass the `--recursive` option, in order to clone submodule(s).
2. Create a build directory as `tools/auto_tuner/build`.
3. Run `CMake` and `Ninja` from the build directory:

```
$ cmake -GNinja ../ -DComputeCpp_DIR=/path/to/computecpp [-DTUNING_TARGET=supported backend]
$ ninja
```

See the Setup section in this repository's main readme for more details.

Make options
------------

CMake options are given using `-D` immediately followed by the option name, the
symbol `=` and a value (`ON` and `OFF` can be used for boolean options and are
equivalent to 1 and 0). Example: `-DBLAS_ENABLE_TESTING=OFF`

| name | value | description |
|---|---|---|
| `BLAS_MEMPOOL_BENCHMARK` | `ON`/`OFF` | Enable the scratchpad memory pool, useful just in case of tall skinny matrices. `OFF` by default |

Usage
-----

Upon a successful build, five binaries will be created which will print the
optimal (highest gflops) tile sizes for a specific transposition of A and B:

| Binary    | Matrix A   | Matrix B   |
|-----------|------------|------------|
| `tune_nn` | Normal     | Normal     |
| `tune_tn` | Transposed | Normal     |
| `tune_nt` | Normal     | Transposed |
| `tune_tt` | Transposed | Transposed |

The `tune_all` binary runs through each combination in turn, printing out
separate results for each of them.

All these binaries are invoked as follows:

```
$ tune M K N bs rep [batch_type]
```

Where the provided options mean the following:

| Option        | Meaning                                                                                            |
|---------------|----------------------------------------------------------------------------------------------------|
| `M`, `N`, `K` | Values for these parameters in the GEMM algorithm                                                  |
| `bs`          | The number of batches to use for batched GEMM. Set to 1 to use regular GEMM                        |
| `rep`         | The number of times to run GEMM for each combination. The mean average is taken off all executions |
| `batch_type`  | The type of batching to be used. It can be interleaved or strided. The default is strided.         |

This will execute GEMM on a number of different combinations depending on the
current platform, and display the results of each in order from worst to best
performance.


Configuration
-------------

A number of configurations are available in `gen/` which describe all the
combinations of tunable parameters which will be used. These are automatically
used by the build system and integrated into the binary at compile time. As an
end user, you will not need to modify any of these files.

However, if you want to add or modify these values, the various `.json` files
can be used. If a new target is to be added, it should be included in the
`CMakeLists.txt` file as appropriate.

The root of the json file is an object containing three arrays. Each array (
`local`, `non_local`, `naive`) is for a different GEMM algorithm, and contain
a number of Configuration Generators.

A configuration Generator is an object, with all of its values being arrays. A
generator adds all combinations (as a cartesian product) of its array elements
to the list of Gemm parameters to try.

The parameters are listed as follows:

| Parameter            | Algorithm            | Description                                                                 |
|----------------------|------------          |-----------------------------------------------------------------------------|
| `cache_line_size`    | All                  | The size of the cache line                                                  |
| `work_item_tiles`    | Not Naive            | The `[rows, cols]` processed by each work item                              |
| `work_group_tiles`   | Not Naive            | The number of item-level tiles within each `[row, col]` of block-level tile |
| `block_level_tiles`  | Local Only           | The number of block-level tiles within each `[row, col]` of top-level tile  |
| `batch_level_tiles`  | non local Interleave | The number of batch work_item and batch work group . default to 1           |
| `vectorization_size` | Not Naive            | the size of vector type                                                     |
| `double_buffer`      | Local Only           | Enable the use of double buffering                                          |
| `no_bank_conflict_a` | Local Only           | Avoids bank conflicts when accessing blocks of matrix A in local memory     |
| `no_bank_conflict_b` | Local Only           | Avoids bank conflicts when accessing blocks of matrix B in local memory     |
