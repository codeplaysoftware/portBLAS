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

#include <cblas.h>
#include <cmath>
#include <iostream>

namespace {
CBLAS_TRANSPOSE c_trans(char x) {
  switch (x) {
    case 't':
    case 'T':
      return CblasTrans;
    case 'n':
    case 'N':
      return CblasNoTrans;
    case 'c':
    case 'C':
      return CblasConjTrans;
    default:
      std::cerr << "Transpose value " << x << " is invalid.\n";
      abort();
  }
}

CBLAS_UPLO c_uplo(char x) {
  switch (x) {
    case 'u':
    case 'U':
      return CblasUpper;
    case 'l':
    case 'L':
      return CblasLower;
    default:
      std::cerr << "Uplo value " << x << " is invalid.\n";
      abort();
  }
}

CBLAS_DIAG c_diag(char x) {
  switch (x) {
    case 'u':
    case 'U':
      return CblasUnit;
    case 'n':
    case 'N':
      return CblasNonUnit;
    default:
      assert(!"unknown case");
  }
}

// i*amin is an extension, provide an implementation
template <typename scalar_t>
inline int iamin(const int N, const scalar_t *X, const int incX) {
  int best = 0;
  for (int i = incX; i < N * incX; i += incX) {
    if (std::abs(X[i]) < std::abs(X[best])) {
      best = i;
    }
  }
  return best / incX;
}

inline int isamin(const int N, const float *X, const int incX) {
  return iamin<float>(N, X, incX);
}

inline int idamin(const int N, const double *X, const int incX) {
  return iamin<double>(N, X, incX);
}
}  // namespace

namespace reference_blas {

template <typename selector_t>
struct TypeDispatcher;

template <>
struct TypeDispatcher<float> {
  template <typename ret_t = void, typename floatfn_t, typename doublefn_t,
            typename... Args>
  static ret_t call(floatfn_t ffn, doublefn_t dfn, Args... args) {
    return ffn(args...);
  }
};

#if DOUBLE_SUPPORT
template <>
struct TypeDispatcher<double> {
  template <typename ret_t = void, typename floatfn_t, typename doublefn_t,
            typename... Args>
  static ret_t call(floatfn_t ffn, doublefn_t dfn, Args... args) {
    return dfn(args...);
  }
};
#endif

// =======
// Level 1
// =======
template <typename scalar_t>
scalar_t asum(const int n, scalar_t x[], const int incX) {
  return TypeDispatcher<scalar_t>::template call<scalar_t>(
      &cblas_sasum, &cblas_dasum, n, x, incX);
}

template <typename scalar_t>
void axpy(const int n, scalar_t alpha, const scalar_t x[], const int incX,
          scalar_t y[], const int incY) {
  TypeDispatcher<scalar_t>::call(&cblas_saxpy, &cblas_daxpy, n, alpha, x, incX,
                                 y, incY);
}

template <typename scalar_t>
void copy(const int n, const scalar_t x[], const int incX, scalar_t y[],
          const int incY) {
  TypeDispatcher<scalar_t>::call(&cblas_scopy, &cblas_dcopy, n, x, incX, y,
                                 incY);
}

template <typename scalar_t>
scalar_t dot(const int n, const scalar_t x[], const int incX, scalar_t y[],
             const int incY) {
  return TypeDispatcher<scalar_t>::template call<scalar_t>(
      &cblas_sdot, &cblas_ddot, n, x, incX, y, incY);
}

template <typename scalar_t>
int iamax(const int n, const scalar_t x[], const int incX) {
  return TypeDispatcher<scalar_t>::template call<int>(
      &cblas_isamax, &cblas_idamax, n, x, incX);
}

template <typename scalar_t>
int iamin(const int n, const scalar_t x[], const int incX) {
  return TypeDispatcher<scalar_t>::template call<int>(&isamin, &idamin, n, x,
                                                      incX);
}

template <typename scalar_t>
scalar_t nrm2(const int n, const scalar_t x[], const int incX) {
  return TypeDispatcher<scalar_t>::template call<scalar_t>(
      &cblas_snrm2, &cblas_dnrm2, n, x, incX);
}

template <typename scalar_t>
void rot(const int n, scalar_t x[], const int incX, scalar_t y[],
         const int incY, const scalar_t c, const scalar_t s) {
  TypeDispatcher<scalar_t>::call(&cblas_srot, &cblas_drot, n, x, incX, y, incY,
                                 c, s);
}

template <typename scalar_t>
void rotg(scalar_t *sa, scalar_t *sb, scalar_t *c, scalar_t *s) {
  TypeDispatcher<scalar_t>::call(&cblas_srotg, &cblas_drotg, sa, sb, c, s);
}

template <typename scalar_t>
void scal(const int n, const scalar_t alpha, scalar_t x[], const int incX) {
  TypeDispatcher<scalar_t>::call(&cblas_sscal, &cblas_dscal, n, alpha, x, incX);
}

template <typename scalar_t>
void swap(const int n, scalar_t x[], const int incX, scalar_t y[],
          const int incY) {
  TypeDispatcher<scalar_t>::call(&cblas_sswap, &cblas_dswap, n, x, incX, y,
                                 incY);
}

// =======
// Level 2
// =======
template <typename scalar_t>
void gemv(const char *trans, int m, int n, scalar_t alpha, const scalar_t a[],
          int lda, const scalar_t b[], int incX, scalar_t beta, scalar_t c[],
          int incY) {
  TypeDispatcher<scalar_t>::call(&cblas_sgemv, &cblas_dgemv, CblasColMajor,
                                 c_trans(*trans), m, n, alpha, a, lda, b, incX,
                                 beta, c, incY);
}

template <typename scalar_t>
void ger(int m, int n, scalar_t alpha, const scalar_t a[], int incX,
         const scalar_t b[], int incY, scalar_t c[], int lda) {
  TypeDispatcher<scalar_t>::call(&cblas_sger, &cblas_dger, CblasColMajor, m, n,
                                 alpha, a, incX, b, incY, c, lda);
}

template <typename scalar_t>
void trmv(const char *uplo, const char *trans, const char *diag, const int n,
          const scalar_t *a, const int lda, scalar_t *x, const int incX) {
  TypeDispatcher<scalar_t>::call(&cblas_strmv, &cblas_dtrmv, CblasColMajor,
                                 c_uplo(*uplo), c_trans(*trans), c_diag(*diag),
                                 n, a, lda, x, incX);
}

template <typename scalar_t>
void syr(const char *uplo, const int n, const scalar_t alpha, const scalar_t *x,
         const int incX, scalar_t *a, const int lda) {
  TypeDispatcher<scalar_t>::call(&cblas_ssyr, &cblas_dsyr, CblasColMajor,
                                 c_uplo(*uplo), n, alpha, x, incX, a, lda);
}

template <typename scalar_t>
void syr2(const char *uplo, const int n, const scalar_t alpha,
          const scalar_t *x, const int incX, const scalar_t *y, const int incY,
          scalar_t *a, const int lda) {
  TypeDispatcher<scalar_t>::call(&cblas_ssyr2, &cblas_dsyr2, CblasColMajor,
                                 c_uplo(*uplo), n, alpha, x, incX, y, incY, a,
                                 lda);
}

template <typename scalar_t>
void symv(const char *uplo, const int n, const scalar_t alpha,
          const scalar_t *a, const int lda, const scalar_t *x, const int incX,
          const scalar_t beta, scalar_t *y, const int incY) {
  TypeDispatcher<scalar_t>::call(&cblas_ssymv, &cblas_dsymv, CblasColMajor,
                                 c_uplo(*uplo), n, alpha, a, lda, x, incX, beta,
                                 y, incY);
}

// =======
// Level 3
// =======
template <typename scalar_t>
void gemm(const char *transA, const char *transB, int m, int n, int k,
          scalar_t alpha, const scalar_t a[], int lda, const scalar_t b[],
          int ldb, scalar_t beta, scalar_t c[], int ldc) {
  TypeDispatcher<scalar_t>::call(&cblas_sgemm, &cblas_dgemm, CblasColMajor,
                                 c_trans(*transA), c_trans(*transB), m, n, k,
                                 alpha, a, lda, b, ldb, beta, c, ldc);
}

#undef COROUTINE_SELECT
}  // namespace reference_blas

#endif /* end of include guard: SYSTEM_REFERENCE_BLAS_HPP */
