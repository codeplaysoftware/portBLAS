#!/usr/bin/env python

import argparse
import json
import sys

from collections import namedtuple
from itertools import product


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
            'tile_rows',
            'tile_cols',
        ])):
    """ A parameter set of tile sizes. """
    __slots__ = ()

    def __str__(self):
        return "Tile<{}, {}, {}, {}, {}, {}>".format(
            self.item_rows, self.item_cols, self.group_rows, self.group_cols,
            self.tile_rows, self.tile_cols)


class LocalGemm(
        namedtuple('LocalGemm', [
            'cache_size',
            'tile',
            'double_buffer',
            'bank_conf_a',
            'bank_conf_b',
        ])):
    """ A parameter set for the non-local memory GEMM kernel.  """
    __slots__ = ()

    def __str__(self):
        return "{}, {}, {}, {}, {}, Local".format(
            self.cache_size, self.tile, _bool_to_str(self.double_buffer),
            _bool_to_str(self.bank_conf_a), _bool_to_str(self.bank_conf_b))

    def is_valid(self):
        """
        Check whether this config is valid and can be compiled.

        The requirements here should match any asserts in the kernel.
        """
        return (self.tile.group_rows % self.tile.item_cols == 0
                and self.tile.group_cols % self.tile.item_rows == 0
                and self.tile.group_rows * self.tile.group_cols %
                (self.cache_size / 4) == 0)


class NonLocalGemm(namedtuple('NonLocalGemm', ['tile'])):
    """
    A parameter set for the non-local memory GEMM kernel.

    The only kernel parameters currently supported by the auto-tuner are the
    tile sizes. The kernel does not require parameters for double buffering or
    handling bank conflicts. The kernel does not use the `ClSize` parameter but
    requires it to be set, so we don't store it here and hardcode it as `0` in
    the function.
    """
    __slots__ = ()

    def __str__(self):
        return "0, {}, false, false, false, NonLocal".format(self.tile)

    def is_valid(self):
        """
        Check whether this config is valid and can be compiled.

        The requirements here should match any asserts in the kernel.
        """
        return (self.tile.group_cols *
                self.tile.item_cols == self.tile.group_rows *
                self.tile.item_rows)


class NaiveGemm(namedtuple('NaiveGemm', ['cache_size'])):
    """
    A parameter set for the naive GEMM kernel.

    The only kernel parameters currently supported by the auto-tuner are the
    cache line sizes. The kernel does not support any tiling or memory caching.
    """
    __slots__ = ()

    def __str__(self):
        return "{}, Tile<>, false, false, false, Naive".format(self.cache_size)

    def is_valid(self):
        """ Check whether this config is valid and can be compiled."""
        return True


def _construct_tile(item_sizes, work_group_sizes, tile_sizes=[1, 1]):
    """ Helper function to create a new Tile parameter set. """
    return Tile(item_rows=item_sizes[0],
                item_cols=item_sizes[1],
                group_rows=work_group_sizes[0],
                group_cols=work_group_sizes[1],
                tile_rows=tile_sizes[0],
                tile_cols=tile_sizes[1])


def generate_local_gemm_configs(cache_sizes, item_sizes, group_sizes,
                                tile_sizes, double_buffers, bank_conflicts_a,
                                bank_conflicts_b):
    """
    Generate a list of possible configurations using local memory.

    The parameters match those to the `Gemm` kernel. Each parameter should be a
    list of possible values to use when generating the configurations. Only
    those configurations which would give valid kernels will be generated.
    """

    configs = []
    for cls, item, wg, tl, db, ncba, ncbb in product(cache_sizes, item_sizes,
                                                     group_sizes, tile_sizes,
                                                     double_buffers,
                                                     bank_conflicts_a,
                                                     bank_conflicts_b):
        new_config = LocalGemm(cache_size=cls,
                               tile=_construct_tile(item, wg, tl),
                               double_buffer=db,
                               bank_conf_a=ncba,
                               bank_conf_b=ncbb)
        if new_config.is_valid():
            configs.append(new_config)
    return configs


def generate_nonlocal_gemm_configs(item_sizes, group_sizes):
    """
    Generate a list of possible configurations without local memory.

    The parameters match those to the `Gemm` kernel. Each parameter should be a
    list of possible values to use when generating the configurations. Only
    those configurations which would give valid kernels will be generated.
    """
    configs = []
    for item, wg in product(item_sizes, group_sizes):
        new_config = NonLocalGemm(tile=_construct_tile(item, wg))
        if new_config.is_valid():
            configs.append(new_config)
    return configs


def get_gemm_configs_from_json(json_file):
    gemm_configs = []
    config_json = json.load(json_file)
    for r in config_json.get("local", []):
        gemm_configs += generate_local_gemm_configs(
            r["cache_line_size"], r["item"], r["item_level_tiles"],
            r["block_level_tiles"], r["double_buffer"],
            r["no_bank_conflict_a"], r["no_bank_conflict_b"])

    for r in config_json.get("non_local", []):
        gemm_configs += generate_nonlocal_gemm_configs(r["item"],
                                                       r["item_level_tiles"])

    for r in config_json.get("naive", []):
        gemm_configs += [
            NaiveGemm(cache_size=cls) for cls in r["cache_line_size"]
        ]
    return gemm_configs


def main():
    parser = argparse.ArgumentParser(
        description='Generate GEMM configurations and source files.')
    parser.add_argument(
        'config',
        type=argparse.FileType('r'),
        help='JSON configuration file of the GEMM configurations.')
    parser.add_argument('output',
                        nargs='?',
                        type=argparse.FileType('w'),
                        help='Filename to write output to',
                        default=sys.stdout)
    args = parser.parse_args()

    output_strings = [
        "// **** FILE AUTOGENERATED BY gen/generate_combinations.py ****",
        "// Config from: {}".format(args.config.name),
    ]
    gemm_configs = get_gemm_configs_from_json(args.config)
    output_strings += [
        "tune<{}>(rep, args);".format(gemm) for gemm in gemm_configs
    ]
    output = "\n".join(output_strings)

    args.output.write(output)


if __name__ == "__main__":
    main()
