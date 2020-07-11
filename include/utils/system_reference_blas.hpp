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
#include <vector>

#ifdef BLAS_DATA_TYPE_HALF
#include <CL/sycl.hpp>
#endif  // BLAS_DATA_TYPE_HALF

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

template <typename scalar_t, typename storage_t>
class ConvertedStorageScalarBase {
 public:
  using value_type = storage_t;

  explicit ConvertedStorageScalarBase(const scalar_t input) noexcept
      : m_data{static_cast<storage_t>(input)} {}

  operator storage_t() noexcept { return m_data; }
  storage_t *operator&() noexcept { return &m_data; }
  const storage_t *operator&() const noexcept { return &m_data; }

  void convert_back(scalar_t *output) const noexcept {
    *output = static_cast<scalar_t>(m_data);
  }

 private:
  storage_t m_data;
};

template <typename scalar_t>
struct ConvertedStorageScalar
    : public ConvertedStorageScalarBase<scalar_t, float> {
  using base_t = ConvertedStorageScalarBase<scalar_t, float>;
  using base_t::base_t;
};

template <>
struct ConvertedStorageScalar<double>
    : public ConvertedStorageScalarBase<double, double> {
  using base_t = ConvertedStorageScalarBase<double, double>;
  using base_t::base_t;
};

template <typename scalar_t, typename storage_t>
class ConvertedStorageVectorBase {
 public:
  using value_type = storage_t;

  explicit ConvertedStorageVectorBase(const scalar_t *input,
                                      const size_t size) {
    m_data.reserve(size);
    for (unsigned i = 0; i < size; ++i) {
      m_data.push_back(static_cast<storage_t>(input[i]));
    }
  }

  ConvertedStorageVectorBase(const ConvertedStorageVectorBase &) = delete;
  ConvertedStorageVectorBase &operator=(const ConvertedStorageVectorBase &) =
      delete;

  ConvertedStorageVectorBase(ConvertedStorageVectorBase &&) = default;
  ConvertedStorageVectorBase &operator=(ConvertedStorageVectorBase &&) =
      default;

  ConvertedStorageVectorBase() = default;

  operator storage_t *() noexcept { return m_data.data(); }
  operator const storage_t *() const noexcept { return m_data.data(); }

  void convert_back(scalar_t *output) const noexcept {
    for (unsigned i = 0; i < m_data.size(); ++i) {
      output[i] = static_cast<scalar_t>(m_data[i]);
    }
  }

 private:
  std::vector<storage_t> m_data;
};

template <typename scalar_t>
struct ConvertedStorageNative {
 public:
  explicit ConvertedStorageNative(const scalar_t *input, const size_t)
      : m_data{const_cast<scalar_t *>(input)} {}

  operator scalar_t *() const noexcept { return m_data; }

  void convert_back(scalar_t *output) const noexcept {
    if (output != m_data) {
      // Pointers not matching is a logical error
      std::cerr
          << "ConvertedStorageNative is intended to not have any overhead\n";
      std::exit(1);
    }
    // Do nothing
  }

 private:
  scalar_t *m_data;
};

template <typename scalar_t>
struct ConvertedStorageVector
    : public ConvertedStorageVectorBase<scalar_t, float> {
  using base_t = ConvertedStorageVectorBase<scalar_t, float>;
  using base_t::base_t;
};

template <>
struct ConvertedStorageVector<float> : public ConvertedStorageNative<float> {
  using base_t = ConvertedStorageNative<float>;
  using base_t::base_t;
};
template <>
struct ConvertedStorageVector<double> : public ConvertedStorageNative<double> {
  using base_t = ConvertedStorageNative<double>;
  using base_t::base_t;
};

template <typename scalar_t>
ConvertedStorageScalar<scalar_t> store_converted(const scalar_t input) {
  return ConvertedStorageScalar<scalar_t>{input};
}

template <typename scalar_t>
ConvertedStorageVector<scalar_t> store_converted(const scalar_t *input,
                                                 const size_t size) {
  return ConvertedStorageVector<scalar_t>{input, size};
}

inline int buffer_length(const int n, const int inc) noexcept {
  return 1 + (n - 1) * std::abs(inc);
}

template <typename scalar_t>
ConvertedStorageVector<scalar_t> store_converted_vec(const scalar_t *input,
                                                     const int n,
                                                     const int inc) {
  return store_converted(input, buffer_length(n, inc));
}

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
  auto x_ = store_converted_vec(x, n, incX);
  return static_cast<scalar_t>(func(n, x_, incX));
}

template <typename scalar_t>
void axpy(const int n, scalar_t alpha, const scalar_t x[], const int incX,
          scalar_t y[], const int incY) {
  auto func = blas_system_function<scalar_t>(&cblas_saxpy, &cblas_daxpy);
  auto alpha_ = store_converted(alpha);
  auto x_ = store_converted_vec(x, n, incX);
  auto y_ = store_converted_vec(y, n, incY);
  func(n, alpha_, x_, incX, y_, incY);
  y_.convert_back(y);
}

template <typename scalar_t>
void copy(const int n, const scalar_t x[], const int incX, scalar_t y[],
          const int incY) {
  auto func = blas_system_function<scalar_t>(&cblas_scopy, &cblas_dcopy);
  auto x_ = store_converted_vec(x, n, incX);
  auto y_ = store_converted_vec(y, n, incY);
  func(n, x_, incX, y_, incY);
  y_.convert_back(y);
}

template <typename scalar_t>
scalar_t dot(const int n, const scalar_t x[], const int incX,
             const scalar_t y[], const int incY) {
  auto func = blas_system_function<scalar_t>(&cblas_sdot, &cblas_ddot);
  auto x_ = store_converted_vec(x, n, incX);
  auto y_ = store_converted_vec(y, n, incY);
  return static_cast<scalar_t>(func(n, x_, incX, y_, incY));
}

template <typename scalar_t>
int iamax(const int n, const scalar_t x[], const int incX) {
  auto func = blas_system_function<scalar_t>(&cblas_isamax, &cblas_idamax);
  auto x_ = store_converted_vec(x, n, incX);
  return func(n, x_, incX);
}

template <typename scalar_t>
int iamin(const int n, const scalar_t x[], const int incX) {
  auto func = blas_system_function<scalar_t>(&isamin, &idamin);
  auto x_ = store_converted_vec(x, n, incX);
  return func(n, x_, incX);
}

template <typename scalar_t>
scalar_t nrm2(const int n, const scalar_t x[], const int incX) {
  auto func = blas_system_function<scalar_t>(&cblas_snrm2, &cblas_dnrm2);
  auto x_ = store_converted_vec(x, n, incX);
  return static_cast<scalar_t>(func(n, x_, incX));
}

template <typename scalar_t>
void rot(const int n, scalar_t x[], const int incX, scalar_t y[],
         const int incY, const scalar_t c, const scalar_t s) {
  auto func = blas_system_function<scalar_t>(&cblas_srot, &cblas_drot);
  auto x_ = store_converted_vec(x, n, incX);
  auto y_ = store_converted_vec(y, n, incY);
  auto c_ = store_converted(c);
  auto s_ = store_converted(s);
  func(n, x_, incX, y_, incY, c_, s_);
  x_.convert_back(x);
  y_.convert_back(y);
}

template <typename scalar_t>
void rotg(scalar_t *sa, scalar_t *sb, scalar_t *c, scalar_t *s) {
  auto func = blas_system_function<scalar_t>(&cblas_srotg, &cblas_drotg);
  auto sa_ = store_converted(*sa);
  auto sb_ = store_converted(*sb);
  auto c_ = store_converted(*c);
  auto s_ = store_converted(*s);
  func(&sa_, &sb_, &c_, &s_);
  c_.convert_back(c);
  s_.convert_back(s);
}

template <typename scalar_t>
void scal(const int n, const scalar_t alpha, scalar_t x[], const int incX) {
  auto func = blas_system_function<scalar_t>(&cblas_sscal, &cblas_dscal);
  auto alpha_ = store_converted(alpha);
  auto x_ = store_converted_vec(x, n, incX);
  func(n, alpha_, x_, incX);
  x_.convert_back(x);
}

template <typename scalar_t>
void swap(const int n, scalar_t x[], const int incX, scalar_t y[],
          const int incY) {
  auto func = blas_system_function<scalar_t>(&cblas_sswap, &cblas_dswap);
  auto x_ = store_converted_vec(x, n, incX);
  auto y_ = store_converted_vec(y, n, incY);
  func(n, x_, incX, y_, incY);
  x_.convert_back(x);
  y_.convert_back(y);
}

inline int transposed_dim(const CBLAS_TRANSPOSE trans, const int n_trans,
                          const int n_notrans) noexcept {
  if (trans == CBLAS_TRANSPOSE::CblasTrans) {
    return n_trans;
  } else {
    return n_notrans;
  }
}

// =======
// Level 2
// =======
template <typename scalar_t>
void gemv(const char *trans, int m, int n, scalar_t alpha, const scalar_t a[],
          int lda, const scalar_t x[], int incX, scalar_t beta, scalar_t y[],
          int incY) {
  auto func = blas_system_function<scalar_t>(&cblas_sgemv, &cblas_dgemv);
  const auto trans_ = c_trans(*trans);
  auto a_ = store_converted(a, lda * n);
  const auto x_dim = transposed_dim(trans_, m, n);
  auto x_ = store_converted_vec(x, x_dim, incX);
  const auto y_dim = transposed_dim(trans_, n, m);
  auto y_ = store_converted_vec(y, y_dim, incY);
  auto alpha_ = store_converted(alpha);
  auto beta_ = store_converted(beta);
  func(CblasColMajor, trans_, m, n, alpha_, a_, lda, x_, incX, beta_, y_, incY);
  y_.convert_back(y);
}

template <typename scalar_t>
void ger(int m, int n, scalar_t alpha, const scalar_t a[], int incX,
         const scalar_t x[], int incY, scalar_t y[], int lda) {
  auto func = blas_system_function<scalar_t>(&cblas_sger, &cblas_dger);
  auto alpha_ = store_converted(alpha);
  auto a_ = store_converted(a, lda * n);
  auto x_ = store_converted_vec(x, m, incX);
  auto y_ = store_converted_vec(y, n, incY);
  func(CblasColMajor, m, n, alpha_, a_, incX, x_, incY, y_, lda);
  y_.convert_back(y);
}

template <typename scalar_t>
void trmv(const char *uplo, const char *trans, const char *diag, const int n,
          const scalar_t *a, const int lda, scalar_t *x, const int incX) {
  auto func = blas_system_function<scalar_t>(&cblas_strmv, &cblas_dtrmv);
  auto a_ = store_converted(a, lda * n);
  auto x_ = store_converted_vec(x, n, incX);
  func(CblasColMajor, c_uplo(*uplo), c_trans(*trans), c_diag(*diag), n, a_, lda,
       x_, incX);
  x_.convert_back(x);
}

template <typename scalar_t>
void syr(const char *uplo, const int n, const scalar_t alpha, const scalar_t *x,
         const int incX, scalar_t *a, const int lda) {
  auto func = blas_system_function<scalar_t>(&cblas_ssyr, &cblas_dsyr);
  auto alpha_ = store_converted(alpha);
  auto x_ = store_converted_vec(x, n, incX);
  auto a_ = store_converted(a, lda * n);
  func(CblasColMajor, c_uplo(*uplo), n, alpha_, x_, incX, a_, lda);
  a_.convert_back(a);
}

template <typename scalar_t>
void syr2(const char *uplo, const int n, const scalar_t alpha,
          const scalar_t *x, const int incX, const scalar_t *y, const int incY,
          scalar_t *a, const int lda) {
  auto func = blas_system_function<scalar_t>(&cblas_ssyr2, &cblas_dsyr2);
  auto alpha_ = store_converted(alpha);
  auto x_ = store_converted_vec(x, n, incX);
  auto y_ = store_converted_vec(y, n, incY);
  auto a_ = store_converted(a, lda * n);
  func(CblasColMajor, c_uplo(*uplo), n, alpha_, x_, incX, y_, incY, a_, lda);
  a_.convert_back(a);
}

template <typename scalar_t>
void symv(const char *uplo, const int n, const scalar_t alpha,
          const scalar_t *a, const int lda, const scalar_t *x, const int incX,
          const scalar_t beta, scalar_t *y, const int incY) {
  auto func = blas_system_function<scalar_t>(&cblas_ssymv, &cblas_dsymv);
  auto alpha_ = store_converted(alpha);
  auto a_ = store_converted(a, lda * n);
  auto x_ = store_converted_vec(x, n, incX);
  auto beta_ = store_converted(beta);
  auto y_ = store_converted_vec(y, n, incY);
  func(CblasColMajor, c_uplo(*uplo), n, alpha_, a_, lda, x_, incX, beta_, y_,
       incY);
  y_.convert_back(y);
}

// =======
// Level 3
// =======
template <typename scalar_t>
void gemm(const char *transA, const char *transB, int m, int n, int k,
          scalar_t alpha, const scalar_t a[], int lda, const scalar_t b[],
          int ldb, scalar_t beta, scalar_t c[], int ldc) {
  auto func = blas_system_function<scalar_t>(&cblas_sgemm, &cblas_dgemm);
  auto alpha_ = store_converted(alpha);
  const auto transA_ = c_trans(*transA);
  const auto a_dim = transposed_dim(transA_, m, k);
  auto a_ = store_converted(a, lda * a_dim);
  const auto transB_ = c_trans(*transB);
  const auto b_dim = transposed_dim(transB_, k, n);
  auto b_ = store_converted(b, ldb * b_dim);
  auto beta_ = store_converted(beta);
  auto c_ = store_converted(c, ldc * n);
  func(CblasColMajor, transA_, transB_, m, n, k, alpha_, a_, lda, b_, ldb,
       beta_, c_, ldc);
  c_.convert_back(c);
}

}  // namespace reference_blas

#endif /* end of include guard: SYSTEM_REFERENCE_BLAS_HPP */
