# Reduction

The following is documentation for the Reduction kernel within `portBLAS`.

## What is Reduction?

`Reduction` is not a standard BLAS operation. It is provided as an extension in
`portBLAS`. It refers to an operation where some or all elements of a matrix
are reduced to a single scalar value via a binary operator. The supported
operators are:

- `AddOperator`
- `AbsoluteAddOperator`
- `ProductOperator`
- `MinOperator`
- `MaxOperator`
- `MeanOperator`

Currently in `portBLAS` only partial reduction is supported. Unlike full
reduction, it reduces the columns or rows of a matrix to a single row or column
depending on the reduction dimension specified. The reduction dimension is
specified with the enumeration `reduction_dim_t`.

## Example

Below are examples of the output of partial reduction using `AddOperator` with
the input matrix

```
1 4 7
2 5 8
3 6 9
```

- `reduction_dim_t::inner` - Computes partial reduction along the *columns*,
  i.e. the sum of each column. The output in this case is a row vector:

```
6 15 24
```

- `reduction_dim_t::outer` - Computes partial reduction along the *rows*, i.e.
  the sum of each row of the input matrix. The output in this case is a column
  vector:

```
12
15
18
```

## Relevant CMake Variables

The CMake option `BLAS_ENABLE_EXTENSIONS` (`ON` by default) can be used to
enable/disable compilation of the `Reduction` operation.

## portBLAS Reduction kernel

Currently `portBLAS` supports a partial reduction kernel. Its implementation
can be found in
[src/operations/extension/reduction.hpp](../src/operations/extension/reduction.hpp).

## Kernel Structure

There is a single Reduction kernel: `Reduction`. The main implementation of the
kernel is in `Reduction::eval()`.

The algorithm is as follows:

1. Load multiple values from global memory.
2. Reduce them together using the reduction operator (`operator_t`)
3. Store the result in local memory.
4. Perform the reduction operation on the element in local memory with current
   local id and the corresponding element in the second half of local memory.
5. Store the result in the appropriate part of the output vector.

Some optimizations that were implemented:
- Loop unrolling
- Using an offset when storing data in local memory to prevent bank conflicts
- Using template metaprogramming to compute as much as possible at compile time
  to avoid if-statements in the kernel
