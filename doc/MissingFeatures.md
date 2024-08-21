## Missing Features

The following is a list of features missing in portBLAS for supporting entirely the [oneAPI oneMKL BLAS interface](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/blas).
The order does not reflect any kind of priority.

For questions regarding input types or operators support, please refer to the link above.

- Add row-major support to level-1 operators.
- Add row-major support to level-2 operators.
- Add row-major support to level-3 operators.
- Add complex support to level-1 operators that required it: asum, axpy, copy, nrm2, rot, rotg, scal, swap, iamax, iamin.
- Implement [dotc](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/dotc.html#onemkl-blas-dotc) operator.
- Implement [dotu](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/dotu.html#onemkl-blas-dotu) operator.
- Add complex support to level-2 operators that required it: gbmv, gemv, symv, syr, syr2, tbmv, tbsv, tpmv, tpsv, trmv, trsv.
- Implement [gerc](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/gerc.html#onemkl-blas-gerc) level-2 operator.
- Implement [geru](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/geru.html#onemkl-blas-geru) level-2 operator.
- Implement [hbmv](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/hbmv.html#onemkl-blas-hbmv) level-2 operator.
- Implement [hemv](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/hemv.html#onemkl-blas-hemv) level-2 operator.
- Implement [her](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/her.html#onemkl-blas-her) level-2 operator.
- Implement [her2](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/her2.html#onemkl-blas-her2) level-2 operator.
- Implement [hpmv](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/hpmv.html#onemkl-blas-hpmv) level-2 operator.
- Implement [hpr](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/hpr.html#onemkl-blas-hpr) level-2 operator.
- Implement [hpr2](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/hpr2.html#onemkl-blas-hpr2) level-2 operator.
- Add complex support to level-3 operators that required it: trsm.
- Implement [hemm](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/hemm#onemkl-blas-hemm) level-3 operator.
- Implement [herk](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/herk#onemkl-blas-herk) level-3 operator.
- Implement [her2k](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/her2k#onemkl-blas-her2k) level-3 operator.
- Implement [syrk](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/syrk.html#onemkl-blas-syrk) level-3 operator.
- Implement [syr2k](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/syr2k.html#onemkl-blas-syr2k) level-3 operator.
- Implement [trmm](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/trmm.html#onemkl-blas-trmm) level-3 operator.
- Add complex support to extenstion operators that required it: axpy_batch, omatcopy, omatcopy_batch, omatcopy2, omatadd, omatadd_batch.
- Implement [trsm_batch](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/trsm_batch.html#onemkl-blas-trsm-batch) extension operator.
- Implement [gemmt](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/gemmt.html#onemkl-blas-gemmt) extension operator.
- Implement [imatcopy](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/imatcopy#onemkl-blas-imatcopy) extension operator.
- Implement [imatcopy_batch](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/imatcopy_batch#onemkl-blas-imatcopy-batch) extension operator.
- Implement [gemm_bias](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/gemm_bias.html#onemkl-blas-gemm-bias) extension operator.
- Add different input types support to [gemm](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/gemm#onemkl-blas-gemm)/[gemm_batch](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/domains/blas/gemm_batch#onemkl-blas-gemm-batch). 
- Add half support to level-1 operators that required it: dot, nrm2, rot.
- Add bfloat16 support to level-1 operators that required it: axpy, copy, dot, nrm2, rot, scal.
- Add interface support for scalar value on device for level-1 operators: axpy, rot, scal.
- Add interface support for scalar value on device for level-2 operators: gbmv, gemv, ger, sbmv, spmv, spr, spr2, symv, syr, syr2.
- Add interface support for scalar value on device for level-3 operators: gemm, symm, trmm.
- Add interface support for scalar value on device for extension operators: axpy_batch, omatcopy, omatcopy2, omatadd, omatcopy_batch, omatadd_batch.
