#!/usr/bin/env python

import argparse
import json
import sys
import os

from collections import namedtuple
from itertools import product

_BLAS_MEM_STRING = {
    "local": "::blas::gemm_memory_t::local",
    "no_local": "::blas::gemm_memory_t::no_local",
}

_BLAS_ALGO_STRING = {
    "standard": "::blas::gemm_algorithm_t::standard",
    "naive": "::blas::gemm_algorithm_t::naive",
}

_BLAS_VECTORIZATION_TYPE = {
    "none": "::blas::gemm_vectorization_t::none",
    "partial": "::blas::gemm_vectorization_t::partial",
    "full": "::blas::gemm_vectorization_t::full",
}

_BLAS_BATCH_TYPE = {
    "strided": "::blas::gemm_batch_type_t::strided",
    "interleaved": "::blas::gemm_batch_type_t::interleaved",
}


def _bool_to_str(val):
    """
    Convert Python bool to string 'true' or 'false'.

    Note we cannot use the standard conversion to str, as Python uses 'True'
    and 'False', whereas C++ uses 'true' and 'false'.
    """
    return 'true' if val else 'false'


class Tile(
        namedtuple('Tile', [
            'item_rows',
            'item_cols',
            'group_rows',
            'group_cols',
            'sub_group_row',
            'sub_group_col',
            'tile_rows',
            'tile_cols',
            'item_batchs',
            'wg_batchs',
        ])):
    """ A parameter set of tile sizes. """
    __slots__ = ()

    def __str__(self):
        return "::blas::Tile<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>".format(
            self.item_rows, self.item_cols, self.group_rows, self.group_cols,
            self.sub_group_row, self.sub_group_col, self.tile_rows, self.tile_cols,
            self.item_batchs, self.wg_batchs)

    def to_list(self):
        return [
            self.item_rows, self.item_cols, self.group_rows, self.group_cols,
            self.sub_group_row, self.sub_group_col, self.tile_rows, self.tile_cols, 
            self.item_batchs, self.wg_batchs
        ]


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


class LocalGemm(GemmParams):
    """ A parameter set for the non-local memory GEMM kernel.  """
    __slots__ = ()

    def __new__(self, cache_size, tile, double_buffer, bank_conf_a,
                bank_conf_b, vec_size):
        return super(LocalGemm,
                     self).__new__(self, cache_size, tile, double_buffer,
                                   bank_conf_a, bank_conf_b, 'local',
                                   'standard', 'strided', 'full', vec_size)

    def is_valid(self):
        """
        Check whether this config is valid and can be compiled.

        The requirements here should match any asserts in the kernel.
        """
        return (self.tile.group_rows % self.tile.item_cols == 0
                and self.tile.group_cols % self.tile.item_rows == 0
                and self.tile.group_rows * self.tile.group_cols %
                (self.cache_size / 4) == 0)


class NonLocalGemmStrided(GemmParams):
    """
    A parameter set for the non-local memory GEMM kernel.

    The only kernel parameters currently supported by the auto-tuner are the
    tile sizes. The kernel does not require parameters for double buffering or
    handling bank conflicts. The kernel does not use the `ClSize` parameter but
    requires it to be set, so we don't store it here and hardcode it as `0` in
    the function.
    """
    __slots__ = ()

    def __new__(self, tile, vectorization_type, vec_size):
        return super(NonLocalGemmStrided,
                     self).__new__(self, 0, tile, False, False, False,
                                   'no_local', 'standard', 'strided',
                                   vectorization_type, vec_size)

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


class NonLocalGemmnInterleaved(GemmParams):
    """
    A parameter set for the non-local memory GEMM kernel.

    The only kernel parameters currently supported by the auto-tuner are the
    tile sizes. The kernel does not require parameters for double buffering or
    handling bank conflicts. The kernel does not use the `ClSize` parameter but
    requires it to be set, so we don't store it here and hardcode it as `0` in
    the function.
    """
    __slots__ = ()

    def __new__(self, tile, vec_size):
        return super(NonLocalGemmnInterleaved,
                     self).__new__(self, 0, tile, False, False, False,
                                   'no_local', 'standard', 'interleaved',
                                   'full', vec_size)

    def is_valid(self):
        """
        Check whether this config is valid and can be compiled.

        The requirements here should match any asserts in the kernel.
        """
        return ((self.tile.item_batchs % self.vec_size == 0)
                and ((self.tile.wg_batchs * self.tile.group_rows *
                      self.tile.group_rows) <= 256)
                and self.tile.item_rows % self.vec_size == 0
                and self.tile.item_cols % self.vec_size == 0)


class NaiveGemm(GemmParams):
    """
    A parameter set for the naive GEMM kernel.

    The only kernel parameters currently supported by the auto-tuner are the
    cache line sizes. The kernel does not support any tiling or memory caching.
    """
    __slots__ = ()

    def __new__(self, cache_size):
        return super(NaiveGemm,
                     self).__new__(self, cache_size,
                                   _construct_tile((1, 1), (1, 1), (1, 1), (1, 1)),
                                   False, False, False, 'no_local', 'naive',
                                   'strided', 'none', 1)

    def is_valid(self):
        """ Check whether this config is valid and can be compiled."""
        return True


def _construct_tile(item_sizes,
                    work_group_sizes,
                    sub_group_sizes=[1, 1],
                    tile_sizes=[1, 1],
                    batch_sizes=[1, 1]):
    """ Helper function to create a new Tile parameter set. """
    return Tile(item_rows=item_sizes[0],
                item_cols=item_sizes[1],
                group_rows=work_group_sizes[0],
                group_cols=work_group_sizes[1],
                sub_group_row=sub_group_sizes[0], 
                sub_group_col=sub_group_sizes[1], 
                tile_rows=tile_sizes[0],
                tile_cols=tile_sizes[1],
                item_batchs=batch_sizes[0],
                wg_batchs=batch_sizes[1])


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


def generate_no_local_gemm_strided_configs(item_sizes, group_sizes, vec_sizes):
    """
    Generate a list of possible configurations without local memory and strided batch.

    The parameters match those to the `Gemm` kernel. Each parameter should be a
    list of possible values to use when generating the configurations. Only
    those configurations which would give valid kernels will be generated.
    """
    configs = []
    for item, wg, vs, vt in product(item_sizes, group_sizes, vec_sizes,
                                    ['partial', 'full']):
        new_config = NonLocalGemmStrided(
            tile=_construct_tile(item, wg),
            vectorization_type=vt,
            vec_size=vs,
        )
        if new_config.is_valid():
            configs.append(new_config)
    return configs


def generate_no_local_gemm_interleaved_configs(item_sizes, group_sizes,
                                               batch_sizes, vec_sizes):
    """
    Generate a list of possible configurations without local memory and interleaved batch.

    The parameters match those to the `Gemm` kernel. Each parameter should be a
    list of possible values to use when generating the configurations. Only
    those configurations which would give valid kernels will be generated.
    """
    configs = []
    for item, wg, bs, vs in product(item_sizes, group_sizes, batch_sizes,
                                    vec_sizes):
        new_config = NonLocalGemmnInterleaved(tile=_construct_tile(
            item, wg, [1, 1], [1, 1], bs),
                                              vec_size=vs)
        if new_config.is_valid():
            configs.append(new_config)
    return configs


def get_gemm_configs_from_json(json_file):
    gemm_configs = []
    config_json = json.load(json_file)
    for r in config_json.get("local", []):
        gemm_configs += generate_local_gemm_configs(
            r["cache_line_size"], r["work_item_sizes"], r["work_group_sizes"],
            r["block_level_tiles"], r["double_buffer"],
            r["no_bank_conflict_a"], r["no_bank_conflict_b"],
            r["vectorization_size"])

    for r in config_json.get("non_local", []):
        gemm_configs += generate_no_local_gemm_strided_configs(
            r["work_item_sizes"], r["work_group_sizes"],
            r["vectorization_size"])

    for r in config_json.get("non_local_interleaved", []):
        gemm_configs += generate_no_local_gemm_interleaved_configs(
            r["work_item_sizes"], r["work_group_sizes"],
            r["batch_level_tiles"], r["vectorization_size"])

    for r in config_json.get("naive", []):
        gemm_configs += [
            NaiveGemm(cache_size=cls) for cls in r["cache_line_size"]
        ]
    #Remove duplicates from the configs by converting to dictionary then back to list
    return list(dict.fromkeys(gemm_configs))


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


def _get_filename(params):
    """ Return the filename generated from given configuration params. """
    def _mem_to_int(mem_type):
        return {'local': 0, 'no_local': 1}[mem_type]

    def _algo_to_int(algo_type):
        return {'standard': 0, 'naive': 1}[algo_type]

    def _batch_to_int(batch_type):
        return {'strided': 0, 'interleaved': 1}[batch_type]

    def _vec_type_to_int(vec_type):
        return {'none': 0, 'partial': 1, 'full': 2}[vec_type]

    def _bool_to_int(val):
        return 1 if val else 0

    int_list = [
        _mem_to_int(params.mem_type),
        _algo_to_int(params.algo_type),
        _batch_to_int(params.batch_type),
        _bool_to_int(params.double_buffer),
        _bool_to_int(params.bank_conf_a),
        _bool_to_int(params.bank_conf_b),
        _vec_type_to_int(params.vec_type),
        params.cache_size,
        params.vec_size,
    ] + params.tile.to_list()
    return "tune_gemm_" + "_".join(map(str, int_list)) + ".cpp"


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


def list_source_files(config_list, output_dir):
    """
    Print a list of source files that would be generated from the given list of
    configurations.

    Args:
        config_list (`list` of `GemmParams`): List of GEMM configurations that
            would be used to generate source files.
        output_dir (str): Directory path where source files would be saved.
    """
    for config in config_list:
        filename = os.path.join(output_dir, _get_filename(config))
        print(filename)


def main():
    parser = argparse.ArgumentParser(
        description='Generate GEMM configurations and source files.')
    parser.add_argument(
        'config',
        type=argparse.FileType('r'),
        help='JSON configuration file of the GEMM configurations.')
    parser.add_argument('def_file',
                        nargs='?',
                        type=argparse.FileType('w'),
                        help='Filename to write output to',
                        default=sys.stdout)
    parser.add_argument('--source_dir',
                        type=str,
                        help='Directory to store generated source files')
    parser.add_argument(
        '--list_files',
        action='store_true',
        help='Write list of source files that would be generated')
    args = parser.parse_args()
    gemm_configs = get_gemm_configs_from_json(args.config)

    if args.source_dir:
        if args.list_files:
            list_source_files(gemm_configs, args.source_dir)
        else:
            write_source_files(gemm_configs, args.config, args.source_dir)
    else:
        write_output_definition_file(gemm_configs, args.config.name,
                                     args.def_file)


if __name__ == "__main__":
    main()
