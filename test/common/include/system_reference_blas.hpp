/***************************************************************************
 *
 *  @license
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename system_reference_blas.hpp
 *
 **************************************************************************/

#ifndef SYSTEM_REFERENCE_BLAS_HPP
#define SYSTEM_REFERENCE_BLAS_HPP

#define ENABLE_SYSTEM_GEMV(_type, _system_name)                              \
  extern "C" void _system_name(const char *, const int *, const int *,       \
                               const _type *, const _type *, const int *,    \
                               const _type *, const int *, const _type *,    \
                               _type *, const int *);                        \
  inline void gemv(const char *trans, int m, int n, _type alpha,             \
                   const _type a[], int lda, const _type b[], int incX,      \
                   _type beta, _type c[], int incY) {                        \
    _system_name(trans, &m, &n, &alpha, a, &lda, b, &incX, &beta, c, &incY); \
  }

ENABLE_SYSTEM_GEMV(float, sgemv_)
ENABLE_SYSTEM_GEMV(double, dgemv_)

#undef ENABLE_SYSTEM_GEMV

#define ENABLE_SYSTEM_GER(_type, _system_name)                            \
  extern "C" void _system_name(const int *, const int *, const _type *,   \
                               const _type *, const int *, const _type *, \
                               const int *, _type *, const int *);        \
  inline void ger(int m, int n, _type alpha, const _type a[], int incX,   \
                  const _type b[], int incY, _type c[], int lda) {        \
    _system_name(&m, &n, &alpha, a, &incX, b, &incY, c, &lda);            \
  }

ENABLE_SYSTEM_GER(float, sger_)
ENABLE_SYSTEM_GER(double, dger_)

#undef ENABLE_SYSTEM_GER

#define ENABLE_SYSTEM_GEMM(_type, _system_name)                                \
  extern "C" void _system_name(                                                \
      const char *, const char *, const int *, const int *, const int *,       \
      const _type *, const _type *, const int *, const _type *, const int *,   \
      const _type *, _type *, const int *);                                    \
  inline void gemm(const char *transA, const char *transB, int m, int n,       \
                   int k, _type alpha, const _type a[], int lda,               \
                   const _type b[], int ldb, _type beta, _type c[], int ldc) { \
    _system_name(transA, transB, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta,  \
                 c, &ldc);                                                     \
  }

ENABLE_SYSTEM_GEMM(float, sgemm_)
ENABLE_SYSTEM_GEMM(double, dgemm_)

#undef ENABLE_SYSTEM_GEMM

#endif /* end of include guard: SYSTEM_REFERENCE_BLAS_HPP */
