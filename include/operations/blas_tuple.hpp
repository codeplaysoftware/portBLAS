/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename blas_tuple.hpp
 *
 **************************************************************************/

#ifndef BLAS_TUPLE_HPP
#define BLAS_TUPLE_HPP

#include <cstdlib>
#include <type_traits>
#include <utility>

namespace blas {

namespace detail {

template <typename... Ts> struct typelist{using type=typelist;};
template <typename... Ts> struct Tuple;
template <typename T> struct Tuple<T> {
  static constexpr size_t size = 1;
  T x;
  Tuple(T _x): x(_x) {}
};
template <typename T, typename... Ts> struct Tuple<T, Ts...> {
  static constexpr size_t size = 1 + sizeof...(Ts);
  T x;
  Tuple<Ts...> y;
  template <typename... Args>
  Tuple(T x, Args&... args): x(x), y(args...) {}
};
template <size_t I, typename T> struct GetType;
template <size_t I, typename T, typename... Ts> struct GetType<I, Tuple<T, Ts...>> {
  using type = typename GetType<I-1, Tuple<Ts...>>::type;
  static constexpr type get(const Tuple<T, Ts...> &t) {
    return GetType<I-1, Tuple<Ts...>>::get(t.y);
  }
  static constexpr type &get(Tuple<T, Ts...> &t) {
    return GetType<I-1, Tuple<Ts...>>::get(t.y);
  }
};
template <typename T, typename... Ts>  struct GetType<0, Tuple<T, Ts...>> {
  using type = T;
  static constexpr type get(const Tuple<T, Ts...> &t) {
    return t.x;
  }
  static constexpr type &get(Tuple<T, Ts...> &t) {
    return t.x;
  }
};

} // namespace detail

template <typename... Ts> using Tuple = detail::Tuple<Ts...>;

template <size_t I, typename TupleT>
typename detail::GetType<I, TupleT>::type &get(TupleT &t) {
  return detail::GetType<I, typename std::remove_const<TupleT>::type>::get(t);
}
template <size_t I, typename TupleT>
typename detail::GetType<I, TupleT>::type get(const TupleT &t) {
  return detail::GetType<I, typename std::remove_const<TupleT>::type>::get(t);
}
template <typename... Ts>
Tuple<Ts...> make_tuple(Ts... args) {
  return Tuple<Ts...>(args...);
}

} // namespace blas

#endif /* end of include guard: BLAS_TUPLE_HPP */
