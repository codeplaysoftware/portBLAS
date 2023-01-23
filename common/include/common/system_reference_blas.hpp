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

#include "cblas.h"
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
      std::cerr << "Upper/lower value " << x << " is invalid.\n";
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
      std::cerr << "Diag value " << x << " is invalid.\n";
      abort();
  }
}

CBLAS_SIDE c_side(char x) {
  switch (std::tolower(x)) {
    case 'l':
      return CblasLeft;
    case 'r':
      return CblasRight;
    default:
      std::cerr << "Side value " << x << " is invalid.\n";
      abort();
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

template <typename scalar_t>
struct BlasSystemFunction {
  template <typename floatfn_t, typename doublefn_t>
  static floatfn_t get(floatfn_t ffn, doublefn_t) noexcept {
    return ffn;
  }
};

template <>
struct BlasSystemFunction<double> {
  template <typename floatfn_t, typename doublefn_t>
  static doublefn_t get(floatfn_t, doublefn_t dfn) noexcept {
    return dfn;
  }
};

template <typename scalar_t, typename floatfn_t, typename doublefn_t>
auto blas_system_function(floatfn_t ffn, doublefn_t dfn)
    -> decltype(BlasSystemFunction<scalar_t>::get(ffn, dfn)) {
  return BlasSystemFunction<scalar_t>::get(ffn, dfn);
}

// =======
// Level 1
// =======
template <typename scalar_t>
scalar_t asum(const int n, scalar_t x[], const int incX) {
  auto func = blas_system_function<scalar_t>(&cblas_sasum, &cblas_dasum);
  return static_cast<scalar_t>(func(n, x, incX));
}

template <typename scalar_t>
void axpy(const int n, scalar_t alpha, const scalar_t x[], const int incX,
          scalar_t y[], const int incY) {
  auto func = blas_system_function<scalar_t>(&cblas_saxpy, &cblas_daxpy);
  func(n, alpha, x, incX, y, incY);
}

template <typename scalar_t>
void copy(const int n, const scalar_t x[], const int incX, scalar_t y[],
          const int incY) {
  auto func = blas_system_function<scalar_t>(&cblas_scopy, &cblas_dcopy);
  func(n, x, incX, y, incY);
}

template <typename scalar_t>
scalar_t dot(const int n, const scalar_t x[], const int incX,
             const scalar_t y[], const int incY) {
  auto func = blas_system_function<scalar_t>(&cblas_sdot, &cblas_ddot);
  return static_cast<scalar_t>(func(n, x, incX, y, incY));
}

/* Note: Not a template hence inlined to avoid having more than one definition
 */
inline float sdsdot(const int n, const float sb, const float x[],
                    const int incX, const float y[], const int incY) {
  auto func = &cblas_sdsdot;
  return static_cast<float>(func(n, sb, x, incX, y, incY));
}

template <typename scalar_t>
int iamax(const int n, const scalar_t x[], const int incX) {
  auto func = blas_system_function<scalar_t>(&cblas_isamax, &cblas_idamax);
  return func(n, x, incX);
}

template <typename scalar_t>
int iamin(const int n, const scalar_t x[], const int incX) {
  auto func = blas_system_function<scalar_t>(&isamin, &idamin);
  return func(n, x, incX);
}

template <typename scalar_t>
scalar_t nrm2(const int n, const scalar_t x[], const int incX) {
  auto func = blas_system_function<scalar_t>(&cblas_snrm2, &cblas_dnrm2);
  return static_cast<scalar_t>(func(n, x, incX));
}

template <typename scalar_t>
void rot(const int n, scalar_t x[], const int incX, scalar_t y[],
         const int incY, const scalar_t c, const scalar_t s) {
  auto func = blas_system_function<scalar_t>(&cblas_srot, &cblas_drot);
  func(n, x, incX, y, incY, c, s);
}

template <typename scalar_t>
void rotm(const int n, scalar_t x[], const int incX, scalar_t y[],
          const int incY, scalar_t param[]) {
  auto func = blas_system_function<scalar_t>(&cblas_srotm, &cblas_drotm);
  func(n, x, incX, y, incY, param);
}

template <typename scalar_t>
void rotg(scalar_t *sa, scalar_t *sb, scalar_t *c, scalar_t *s) {
  auto func = blas_system_function<scalar_t>(&cblas_srotg, &cblas_drotg);
  func(sa, sb, c, s);
}

template <typename scalar_t>
void rotmg(scalar_t *d1, scalar_t *d2, scalar_t *x1, scalar_t *y1,
           scalar_t param[]) {
  auto func = blas_system_function<scalar_t>(&cblas_srotmg, &cblas_drotmg);
  func(d1, d2, x1, *y1, param);
}

template <typename scalar_t>
void scal(const int n, const scalar_t alpha, scalar_t x[], const int incX) {
  auto func = blas_system_function<scalar_t>(&cblas_sscal, &cblas_dscal);
  func(n, alpha, x, incX);
}

template <typename scalar_t>
void swap(const int n, scalar_t x[], const int incX, scalar_t y[],
          const int incY) {
  auto func = blas_system_function<scalar_t>(&cblas_sswap, &cblas_dswap);
  func(n, x, incX, y, incY);
}

// =======
// Level 2
// =======
template <typename scalar_t>
void gbmv(const char *trans, int m, int n, int kl, int ku, scalar_t alpha,
          const scalar_t a[], int lda, const scalar_t x[], int incX,
          scalar_t beta, scalar_t y[], int incY) {
  auto func = blas_system_function<scalar_t>(&cblas_sgbmv, &cblas_dgbmv);
  func(CblasColMajor, c_trans(*trans), m, n, kl, ku, alpha, a, lda, x, incX,
       beta, y, incY);
}

template <typename scalar_t>
void gemv(const char *trans, int m, int n, scalar_t alpha, const scalar_t a[],
          int lda, const scalar_t x[], int incX, scalar_t beta, scalar_t y[],
          int incY) {
  auto func = blas_system_function<scalar_t>(&cblas_sgemv, &cblas_dgemv);
  func(CblasColMajor, c_trans(*trans), m, n, alpha, a, lda, x, incX, beta, y,
       incY);
}

template <typename scalar_t>
void ger(int m, int n, scalar_t alpha, const scalar_t a[], int incX,
         const scalar_t x[], int incY, scalar_t y[], int lda) {
  auto func = blas_system_function<scalar_t>(&cblas_sger, &cblas_dger);
  func(CblasColMajor, m, n, alpha, a, incX, x, incY, y, lda);
}

template <typename scalar_t>
void tbmv(const char *uplo, const char *trans, const char *diag, int n, int k,
          const scalar_t *a, int lda, scalar_t *x, int incX) {
  auto func = blas_system_function<scalar_t>(&cblas_stbmv, &cblas_dtbmv);
  func(CblasColMajor, c_uplo(*uplo), c_trans(*trans), c_diag(*diag), n, k, a,
       lda, x, incX);
}

template <typename scalar_t>
void trmv(const char *uplo, const char *trans, const char *diag, const int n,
          const scalar_t *a, const int lda, scalar_t *x, const int incX) {
  auto func = blas_system_function<scalar_t>(&cblas_strmv, &cblas_dtrmv);
  func(CblasColMajor, c_uplo(*uplo), c_trans(*trans), c_diag(*diag), n, a, lda,
       x, incX);
}

template <typename scalar_t>
void sbmv(const char *uplo, int n, int k, scalar_t alpha, const scalar_t a[],
          int lda, const scalar_t x[], int incX, scalar_t beta, scalar_t y[],
          int incY) {
  auto func = blas_system_function<scalar_t>(&cblas_ssbmv, &cblas_dsbmv);
  func(CblasColMajor, c_uplo(*uplo), n, k, alpha, a, lda, x, incX, beta, y,
       incY);
}

template <typename scalar_t>
void syr(const char *uplo, const int n, const scalar_t alpha, const scalar_t *x,
         const int incX, scalar_t *a, const int lda) {
  auto func = blas_system_function<scalar_t>(&cblas_ssyr, &cblas_dsyr);
  func(CblasColMajor, c_uplo(*uplo), n, alpha, x, incX, a, lda);
}

template <typename scalar_t>
void syr2(const char *uplo, const int n, const scalar_t alpha,
          const scalar_t *x, const int incX, const scalar_t *y, const int incY,
          scalar_t *a, const int lda) {
  auto func = blas_system_function<scalar_t>(&cblas_ssyr2, &cblas_dsyr2);
  func(CblasColMajor, c_uplo(*uplo), n, alpha, x, incX, y, incY, a, lda);
}

template <typename scalar_t>
void symv(const char *uplo, const int n, const scalar_t alpha,
          const scalar_t *a, const int lda, const scalar_t *x, const int incX,
          const scalar_t beta, scalar_t *y, const int incY) {
  auto func = blas_system_function<scalar_t>(&cblas_ssymv, &cblas_dsymv);
  func(CblasColMajor, c_uplo(*uplo), n, alpha, a, lda, x, incX, beta, y, incY);
}

// =======
// Level 3
// =======
template <typename scalar_t>
void gemm(const char *transA, const char *transB, int m, int n, int k,
          scalar_t alpha, const scalar_t a[], int lda, const scalar_t b[],
          int ldb, scalar_t beta, scalar_t c[], int ldc) {
  auto func = blas_system_function<scalar_t>(&cblas_sgemm, &cblas_dgemm);
  func(CblasColMajor, c_trans(*transA), c_trans(*transB), m, n, k, alpha, a,
       lda, b, ldb, beta, c, ldc);
}

template <typename scalar_t>
void trsm(const char *side, const char *uplo, const char *trans,
          const char *diag, int m, int n, scalar_t alpha, const scalar_t A[],
          int lda, scalar_t B[], int ldb) {
  auto func = blas_system_function<scalar_t>(&cblas_strsm, &cblas_dtrsm);
  func(CblasColMajor, c_side(*side), c_uplo(*uplo), c_trans(*trans),
       c_diag(*diag), m, n, alpha, A, lda, B, ldb);
}

}  // namespace reference_blas

#endif /* end of include guard: SYSTEM_REFERENCE_BLAS_HPP */
