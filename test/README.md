Tests
===

## General information

The tests are written with the [Google Test](https://github.com/google/googletest)
framework.

There are two kinds of tests:

* **Unit tests** test the BLAS interface as available to the users.
* **Expression tests** include the header-only framework to test the expression
  trees and features that are not available through the BLAS interface.

## Build the tests

The tests are built with the project by default. This can be set or unset with
the CMake option `BLAS_ENABLE_TESTING`.

A reference BLAS implementation like OpenBLAS is required to be installed on
the machine. Use the CMake `CMAKE_PREFIX_PATH` option to point to its root folder
if it is installed in a custom location.

By default, only the unit tests are build. Use `-DENABLE_EXPRESSION_TESTS=ON` to
build the expression tests.

You can set a specific device to run the tests with, using the CMake
`TEST_DEVICE` option. The format of this string is a vendor name, followed by
`gpu`, `cpu` or `accel`. A `*` may take the place of either a vendor or device
type, and means "any". If the specified device isn't found, a default device is
used instead. Example: `intel:gpu`. If `TEST_DEVICE` is not specified, the
device used will be chosen based on the behaviour of SYCL's default selector.

## Run the tests

You can run all tests with CTest by simply running the following command in the
build directory:

```bash
ctest
```

You can also run individual tests by executing their executable found in the
build directory, in `test/unittest` or `test/exprtest`. In that case, you can
select the device with the `--device` argument. Example:

```bash
./test/unittest/blas3_gemm_test --device=intel:gpu
```
