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
 *  @filename vec.hpp
 *
 **************************************************************************/

#ifndef UTILS_VEC_HPP
#define UTILS_VEC_HPP


#include <CL/sycl.hpp>


#include <type_traits>


namespace blas {


#define DEFINE_OPERATOR(_oper, _size) \
friend vec<dataT, _size> operator _oper( \
    const vec<dataT, _size> &lhs, \
    const vec<dataT, _size> &rhs) \
  { return lhs.data_ _oper rhs.data_; } \
friend vec<dataT, _size> operator _oper( \
    const vec<dataT, _size> &lhs, \
    const dataT &rhs) \
  { return lhs.data_ _oper rhs; } \
friend vec<dataT, _size> operator _oper##=( \
    vec<dataT, _size> &lhs, \
    const vec<dataT, _size> &rhs) \
  { return lhs.data_ _oper##= rhs.data_; } \
friend vec<dataT, _size> operator _oper##=( \
    vec<dataT, _size> &lhs, \
    const dataT &rhs) \
  { return lhs.data_ _oper##= rhs; }


/*!
 * @brief A wrapper for the cl::sycl::vec class template, which also enables
 *        handling of single element vectors.
 */
template <typename dataT, int size>
class vec {
  private:
    cl::sycl::vec<dataT, size> data_;

  public:
    using element_type = dataT;

    vec() = default;

    vec(const cl::sycl::vec<dataT, size> &other) : data_(other) {}

    vec(cl::sycl::vec<dataT, size> &&other) : data_(other) {}

    template <typename dataS>
    vec(const cl::sycl::vec<dataS, size> &other) : data_(other) {}

    explicit vec(const dataT &arg) : data_(arg) {}

    template <typename... argTN>
    vec(const argTN&... args) : data_(args...) {}

    size_t get_count() const { return data_.get_count(); }

    size_t get_size() const { return data_.get_size(); }

    template <int... params>
    auto swizzle() const -> decltype(data_.template swizzle<params...>())
    { return data_.template swizzle<params...>(); }

    template <int... params>
    auto swizzle() -> decltype(data_.template swizzle<params...>())
    { return data_.template swizzle<params...>(); }

    template <typename Accessor>
    void load(size_t offset, Accessor acc) { data_.load(offset, acc); }

    template <typename Accessor>
    void store(size_t offset, Accessor acc) { data_.store(offset, acc); }

    DEFINE_OPERATOR(+, size);
    DEFINE_OPERATOR(-, size);
    DEFINE_OPERATOR(*, size);
    DEFINE_OPERATOR(/, size);
    DEFINE_OPERATOR(%, size);
    DEFINE_OPERATOR(>>, size);
    DEFINE_OPERATOR(<<, size);
};


template <typename dataT>
class vec<dataT, 1> {
  private:
    dataT data_;

  public:
    using element_type = dataT;

    vec() = default;

    vec(const cl::sycl::vec<dataT, 1> &other) : data_(other.s0()) {}

    vec(cl::sycl::vec<dataT, 1> &&other) : data_(other.s0()) {}

    template <typename dataS>
    vec(const cl::sycl::vec<dataS, 1> &other) : data_(other.s0()) {}

    explicit vec(const dataT &arg) : data_(arg) {}

    template <typename... argTN>
    vec(const argTN&... args) : data_(args...) {}

    size_t get_count() const { return 1; }

    size_t get_size() const { return sizeof(dataT); }

    template <int k, typename = typename std::enable_if<k == 0>::type>
    const dataT& swizzle() const {
      return data_;
    }

    template <int k, typename = typename std::enable_if<k == 0>::type>
    dataT& swizzle() {
      return data_;
    }

    template <typename Accessor>
    void load(size_t offset, Accessor acc) { data_ = acc[offset]; }

    template <typename Accessor>
    void store(size_t offset, Accessor acc) { acc[offset] = data_; }

    DEFINE_OPERATOR(+, 1);
    DEFINE_OPERATOR(-, 1);
    DEFINE_OPERATOR(*, 1);
    DEFINE_OPERATOR(/, 1);
    DEFINE_OPERATOR(%, 1);
    DEFINE_OPERATOR(>>, 1);
    DEFINE_OPERATOR(<<, 1);
};


#undef DEFINE_OPERATOR


template <int start, int end>
struct for_vec_elem {
  template <typename VecType, typename OpType>
  static void map(const VecType &vec, OpType op) {
    op(start, vec.template swizzle<start>());
    for_vec_elem<start+1, end>::map(vec, op);
  }
  template <typename VecType, typename OpType>
  static void transform(VecType &vec, OpType op) {
    vec.template swizzle<start>() = op(start, vec.template swizzle<start>());
    for_vec_elem<start+1, end>::map(vec, op);
  }
};


template <int end>
struct for_vec_elem<end, end> {
  template <typename VecType, typename OpType>
  static void map(const VecType &vec, OpType op) {}
  template <typename VecType, typename OpType>
  static void transform(VecType &vec, OpType op) {}
};


}


#endif  // UTILS_VEC_HPP
