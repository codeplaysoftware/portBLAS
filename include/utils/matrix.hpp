/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2017 Codeplay Software Limited
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
 *  @filename matrix.hpp
 *
 **************************************************************************/

#ifndef UTILS_MATRIX_HPP
#define UTILS_MATRIX_HPP


#include <ostream>
#include <type_traits>


#include <CL/sycl.hpp>


namespace blas {


namespace detail {


template <typename T, typename S>
inline constexpr int min(const T &a, const S &b) {
  return a <= b ? a : b;
}


}


struct span {
  int start;
  int size;

  constexpr span(int st = 0, int sz = 1) : start(st), size(sz) {}

  constexpr span operator [](const span &t) const {
    return span(start + t.start, detail::min(t.size, start + size - t.start));
  }

  constexpr int operator [](int idx) const {
    return start + idx;
  }

  explicit operator bool() const { return size > 0; }
};


inline constexpr span range(int start, int end) {
  return span(start, end - start);
}


#define ENABLE_SPAN_OFFSET_OP(_op) \
inline constexpr span operator _op(const span &s, int ofs) { \
  return span(s.start _op (ofs * s.size), s.size); \
} \
\
inline constexpr span operator _op(int ofs, const span &s) { \
  return span(ofs _op s.start, s.size); \
} \
\
inline span& operator _op##= (span &s, int ofs) { \
  s.start _op##= ofs * s.size; \
  return s; \
}


#define ENABLE_SPAN_SCALE_OP(_op) \
inline constexpr span operator _op(const span &s, int scale) { \
  return span(s.start, s.size _op scale); \
} \
\
inline span& operator _op##= (span &s, int scale) { \
  s.size _op##= scale; \
  return s; \
}


#define ENABLE_SPAN_COMPARE_OP(_op) \
inline constexpr bool operator _op(const span &s, int k) { \
  return s.start + s.size _op k; \
} \
\
inline constexpr bool operator _op(int k, const span &s) { \
  return k _op s.start; \
} \
\
inline constexpr bool operator _op(const span &s, const span &t) { \
  return s.start + s.size _op t.start; \
}


ENABLE_SPAN_OFFSET_OP(+)
ENABLE_SPAN_OFFSET_OP(-)
ENABLE_SPAN_SCALE_OP(*)
ENABLE_SPAN_SCALE_OP(/)
ENABLE_SPAN_SCALE_OP(%)
ENABLE_SPAN_COMPARE_OP(<)
ENABLE_SPAN_COMPARE_OP(<=)
ENABLE_SPAN_COMPARE_OP(>)
ENABLE_SPAN_COMPARE_OP(>=)
ENABLE_SPAN_COMPARE_OP(==)
ENABLE_SPAN_COMPARE_OP(!=)


#undef ENABLE_SPAN_OFFSET_OP
#undef ENABLE_SPAN_SCALE_OP
#undef ENABLE_SPAN_COMPARE_OP



inline constexpr int operator /(const span &s, const span &r) {
  return s.size / r.size;
}


inline span& operator ++(span& s) { return s += 1; }
inline span& operator --(span& s) { return s -= 1; }
inline span operator ++(span& s, int) { auto t = s; s += 1; return t; }
inline span operator --(span& s, int) { auto t = s; s -= 1; return t; }


std::ostream& operator <<(std::ostream &os, const span &s) {
  return os << s.start << ':' << s.start + s.size;
}


enum storage_type {
  cms,
  rms
};


namespace detail {


template <typename T>
struct value_type {
  using type = typename T::value_type;
};


template <size_t n, typename T>
struct value_type<T[n]> {
  using type = T;
};


template <typename T>
struct value_type<T*> {
  using type = T;
};

template <typename T>
struct value_type<T&> {
  using type = typename value_type<T>::type;
};


template <typename T>
struct value_type<cl::sycl::global_ptr<T>> {
  using type = T;
};

template <typename T>
struct value_type<cl::sycl::local_ptr<T>> {
  using type = T;
};

template <typename T>
struct value_type<cl::sycl::constant_ptr<T>> {
  using type = T;
};

template <storage_type>
inline int linearize_idx(int row, int col, int ld) {} // = delete;


template <>
inline int linearize_idx<storage_type::cms>(int row, int col, int ld) {
  return row + col*ld;
}


template <>
inline int linearize_idx<storage_type::rms>(int row, int col, int ld) {
  return row*ld + col;
}


}  // namespace detail



template <storage_type, typename> class matrix;


template<typename MatrixType, typename TernaryFunction>
inline typename std::enable_if<std::is_same<
    matrix<storage_type::cms, typename MatrixType::accessor_type>,
    typename std::remove_const<
      typename std::remove_reference<MatrixType>::type>::type>
  ::value>
::type for_each(MatrixType M, TernaryFunction f) {
  for (int j = 0; j < M.get_num_cols(); ++j) {
    for (int i = 0; i < M.get_num_rows(); ++i) {
      f(i, j, M(i, j));
    }
  }
}


template<typename MatrixType, typename TernaryFunction>
inline typename std::enable_if<std::is_same<
    matrix<storage_type::rms, typename MatrixType::accessor_type>,
    typename std::remove_const<
      typename std::remove_reference<MatrixType>::type>::type>
  ::value>
::type for_each(MatrixType M, TernaryFunction f) {
  for (int i = 0; i < M.get_num_rows(); ++i) {
    for (int j = 0; j < M.get_num_cols(); ++j) {
      f(i, j, M(i, j));
    }
  }
}


template <typename T>
struct ptr_or_ref {
  using type = T&;
};


template <typename T>
struct ptr_or_ref<T*> {
  using type = T*;
};


template <typename T>
struct ptr_or_ref<cl::sycl::global_ptr<T>> {
  using type = cl::sycl::global_ptr<T>;
};


template <typename T>
struct ptr_or_ref<cl::sycl::local_ptr<T>> {
  using type = cl::sycl::local_ptr<T>;
};


template <typename T>
struct ptr_or_ref<cl::sycl::constant_ptr<T>> {
  using type = cl::sycl::constant_ptr<T>;
};


template <typename T>
struct ptr_or_ref<T&> {
  using type = typename ptr_or_ref<T>::type;
};

template <storage_type storage, typename AccessorType>
class matrix {
  public:
    using accessor_type = AccessorType;
    using value_type = typename detail::value_type<AccessorType>::type;
    using accessor_ref = typename ptr_or_ref<AccessorType>::type;

    constexpr matrix(int rows, int cols, accessor_ref data, int ld)
      : row_span_(0, rows), col_span_(0, cols),
        data_(data), ld_(ld) {}

    constexpr matrix(int rows, int cols, accessor_ref data)
      : matrix(rows, cols, data_(data), rows) {}

    // constexpr matrix(const matrix<storage, AccessorType> &other) = default;

    constexpr value_type operator ()(int row, int col) const {
      return data_[detail::linearize_idx<storage>(
          row_span_[row], col_span_[col], ld_)];
    }

    value_type& operator ()(int row, int col) {
      return data_[detail::linearize_idx<storage>(
          row_span_[row], col_span_[col], ld_)];
    }

    constexpr matrix<storage, AccessorType> operator ()
        (const span &rows, const span &cols) const {
      return matrix<storage, AccessorType>
          (row_span_[rows], col_span_[cols], data_, ld_);
    }

    matrix<storage, AccessorType> operator ()
        (const span &rows, const span &cols) {
      return matrix<storage, AccessorType>
          (row_span_[rows], col_span_[cols], data_, ld_);
    }

    constexpr int get_num_rows() const { return row_span_.size; }

    constexpr int get_num_cols() const { return col_span_.size; }

    matrix& operator =(const matrix<storage, AccessorType> &M) {
      using vtype = value_type;
      for_each(M, [=] (int i, int j, vtype val) { (*this)(i, j) = val; });
      return *this;
    }

    template <storage_type storageM, typename AccessorTypeM>
    matrix& operator =(const matrix<storageM, AccessorTypeM> &M) {
      using vtype = typename matrix<storageM, AccessorTypeM>::value_type;
      for_each(M, [=] (int i, int j, vtype val) { (*this)(i, j) = val; });
      return *this;
    }

    matrix& operator =(const value_type &val) {
      for_each(*this, [=] (int, int, value_type &v) { v = val; });
      return *this;
    }

  private:
    constexpr matrix(const span &row_span, const span &col_span,
                     accessor_ref data, int ld)
      : row_span_(row_span), col_span_(col_span), data_(data), ld_(ld) {}

    const span row_span_;
    const span col_span_;
    AccessorType data_;
    const int ld_;
};


template <storage_type storage, typename AccessorType>
constexpr matrix<storage, AccessorType> make_matrix(
    int rows, int cols, AccessorType &&data, int ld) {
  return matrix<storage, AccessorType>(
      rows, cols, std::forward<AccessorType>(data), ld);
}


template <storage_type storage, typename AccessorType>
constexpr matrix<storage, AccessorType> make_matrix(
    int rows, int cols, AccessorType &&data) {
  return make_matrix<storage>(
      rows, cols, std::forward<AccessorType>(data), rows);
}


template <storage_type storage, typename AccessorType>
std::ostream& operator <<(
    std::ostream &os, const matrix<storage, AccessorType> &M) {
  for (int i = 0; i < M.get_num_rows(); ++i) {
    if (i > 0) {
      os << '\n';
    }
    for (int j = 0; j < M.get_num_cols(); ++j) {
      os << M(i, j) << ' ';
    }
  }
  return os;
}


}  // namespace blas


#endif  // UTILS_MATRIX_HPP

