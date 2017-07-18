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
 *  @filename blas_tree_evaluator.hpp
 *
 **************************************************************************/

#ifndef BLAS_TREE_EVALUATOR_HPP
#define BLAS_TREE_EVALUATOR_HPP

#include <evaluators/blas_tree_evaluator_base.hpp>
#include <executors/blas_packet_traits_sycl.hpp>
#include <operations/blas_trees.hpp>

namespace blas {

template <class LHS, class RHS, typename Device>
struct Evaluator<JoinExpr<LHS, RHS>, Device> {
  using Expression = JoinExpr<LHS, RHS>;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<LHS, Device>::cont_type;
  /* constexpr static bool supported = LHS::supported && RHS::supported; */

  Evaluator<LHS, Device> l;
  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr)
      : l(Evaluator<LHS, Device>(expr.l)), r(Evaluator<RHS, Device>(expr.r)) {}
  size_t getSize() const { return r.getSize(); }
  cont_type data() { return l.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    l.eval_subexpr_if_needed(NULL, dev);
    r.eval_subexpr_if_needed(NULL, dev);
    return true;
  }

  value_type eval(size_t i) {
    l.eval(i);
    return r.eval(i);
  }
  value_type eval(cl::sycl::nd_item<1> nditem) {
    return eval(nditem.get_global(0));
  }
};

template <class LHS, class RHS, typename Device>
struct Evaluator<AssignExpr<LHS, RHS>, Device> {
  using Expression = AssignExpr<LHS, RHS>;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<LHS, Device>::cont_type;
  /* constexpr static bool supported = LHS::supported && RHS::supported; */

  Evaluator<LHS, Device> l;
  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr)
      : l(Evaluator<LHS, Device>(expr.l)), r(Evaluator<RHS, Device>(expr.r)) {}
  size_t getSize() const { return r.getSize(); }
  cont_type data() { return l.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    cont_type data = l.data();
    l.eval_subexpr_if_needed(NULL, dev);
    r.eval_subexpr_if_needed(&data, dev);
    return true;
  }

  value_type eval(size_t i) { return l.eval(i) = r.eval(i); }
  value_type eval(cl::sycl::nd_item<1> nditem) {
    return eval(nditem.get_global(0));
  }
};

template <class LHS1, class LHS2, class RHS1, class RHS2, typename Device>
struct Evaluator<DoubleAssignExpr<LHS1, LHS2, RHS1, RHS2>, Device> {
  using Expression = DoubleAssignExpr<LHS1, LHS2, RHS1, RHS2>;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<LHS1, Device>::cont_type;
  /* constexpr static bool supported = LHS1::supported && LHS2::supported &&
   * RHS1::supported && RHS2::supported; */

  Evaluator<LHS1, Device> l1;
  Evaluator<LHS2, Device> l2;
  Evaluator<RHS1, Device> r1;
  Evaluator<RHS2, Device> r2;

  Evaluator(Expression &expr)
      : l1(Evaluator<LHS1, Device>(expr.l1)),
        l2(Evaluator<LHS2, Device>(expr.l2)),
        r1(Evaluator<RHS1, Device>(expr.r1)),
        r2(Evaluator<RHS2, Device>(expr.r2)) {}
  size_t getSize() const { return r1.getSize(); }
  cont_type data() { return l1.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    l1.eval_subexpr_if_needed(NULL, dev);
    l2.eval_subexpr_if_needed(NULL, dev);
    cont_type data1 = l1.data(), data2 = l2.data();
    r1.eval_subexpr_if_needed(&data1, dev);
    r2.eval_subexpr_if_needed(&data2, dev);
    return true;
  }

  value_type eval(size_t i) {
    auto val1 = r1.eval(i);
    auto val2 = r2.eval(i);
    l1.eval(i) = val1;
    l2.eval(i) = val2;
    return val1;
  }
  value_type eval(cl::sycl::nd_item<1> nditem) {
    return eval(nditem.get_global(0));
  }
};

template <class Functor, class SCL, class RHS, typename Device>
struct Evaluator<ScalarExpr<Functor, SCL, RHS>, Device> {
  using Expression = ScalarExpr<Functor, SCL, RHS>;
  using value_type = typename Expression::value_type;
  using dev_functor = functor_traits<Functor, value_type, Device>;
  using cont_type = typename Evaluator<RHS, Device>::cont_type;
  /* constexpr static bool supported = dev_functor::supported && RHS::supported;
   */

  SCL scl;
  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr)
      : scl(expr.scl), r(Evaluator<RHS, Device>(expr.r)) {}
  size_t getSize() const { return r.getSize(); }
  cont_type data() { return r.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    r.eval_subexpr_if_needed(NULL, dev);
    return true;
  }

  value_type eval(size_t i) {
    return dev_functor::eval(internal::get_scalar(scl), r.eval(i));
  }
  value_type eval(cl::sycl::nd_item<1> nditem) {
    return eval(nditem.get_global(0));
  }
};

template <class Functor, class RHS, typename Device>
struct Evaluator<UnaryExpr<Functor, RHS>, Device> {
  using Expression = UnaryExpr<Functor, RHS>;
  using value_type = typename Expression::value_type;
  using dev_functor = functor_traits<Functor, value_type, Device>;
  using cont_type = typename Evaluator<RHS, Device>::cont_type;
  /* constexpr static bool supported = dev_functor::supported && RHS::supported;
   */

  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr) : r(Evaluator<RHS, Device>(expr.r)) {}
  size_t getSize() const { return r.getSize(); }
  cont_type data() { return r.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    r.eval_subexpr_if_needed(NULL, dev);
    return true;
  }

  value_type eval(size_t i) { return dev_functor::eval(r.eval(i)); }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

template <class Functor, class LHS, class RHS, typename Device>
struct Evaluator<BinaryExpr<Functor, LHS, RHS>, Device> {
  using Expression = BinaryExpr<Functor, LHS, RHS>;
  using value_type = typename Expression::value_type;
  using dev_functor = functor_traits<Functor, value_type, Device>;
  using cont_type = typename Evaluator<LHS, Device>::cont_type;
  /* constexpr static bool supported = dev_functor::supported && LHS::supported
   * && RHS::supported; */

  Evaluator<LHS, Device> l;
  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr)
      : l(Evaluator<LHS, Device>(expr.l)), r(Evaluator<RHS, Device>(expr.r)) {}
  size_t getSize() const { return r.getSize(); }
  cont_type data() { return l.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    cont_type data = l.data();
    l.eval_subexpr_if_needed(NULL, dev);
    r.eval_subexpr_if_needed(&data, dev);
    return true;
  }

  value_type eval(size_t i) { return dev_functor::eval(l.eval(i), r.eval(i)); }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

template <class RHS, typename Device>
struct Evaluator<TupleExpr<RHS>, Device> {
  using Expression = TupleExpr<RHS>;
  using value_type = typename Expression::value_type;
  using cont_type = void;
  /* constexpr static bool supported = RHS::supported; */

  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr) : r(Evaluator<RHS, Device>(expr.r)) {}
  size_t getSize() const { return r.getSize(); }
  cont_type data() { return r.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    r.eval_subexpr_if_needed(NULL, dev);
    return true;
  }

  value_type eval(size_t i) { return value_type(i, r.eval(i)); }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

template <class ScalarT, class ContainerT, typename Device>
struct Evaluator<vector_view<ScalarT, ContainerT>, Device> {
  using Expression = vector_view<ScalarT, ContainerT>;
  using value_type = typename Expression::value_type;
  using cont_type = ContainerT;
  /* constexpr static bool supported = true; */

  vector_view<ScalarT, ContainerT> vec;

  Evaluator(Expression &vec) : vec(vec) {}
  size_t getSize() const { return vec.getSize(); }
  cont_type data() { return vec.data_; }
  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) { return true; }
  value_type &eval(size_t i) {
    //  auto eval(size_t i) -> decltype(data_[i]) {
    auto ind = vec.getDisp();
    if (vec.getStrd() == 1) {
      ind += i;
    } else if (vec.getStrd() > 0) {
      ind += vec.getStrd() * i;
    } else {
      ind -= vec.getStrd() * (vec.getSize() - i - 1);
    }
    return vec.getData()[ind];
  }
  value_type &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

template <class ScalarT, class ContainerT, typename Device>
struct Evaluator<matrix_view<ScalarT, ContainerT>, Device> {
  using Expression = matrix_view<ScalarT, ContainerT>;
  using value_type = typename Expression::value_type;
  using cont_type = ContainerT;
  /* constexpr static bool supported = true; */

  matrix_view<ScalarT, ContainerT> mat;

  Evaluator(Expression &mat) : mat(mat) {}
  size_t getSize() const { return mat.getSizeR(); }
  size_t getSizeR() const { return mat.getSizeR(); }
  size_t getSizeC() const { return mat.getSizeC(); }
  int getAccess() const { return mat.getAccess(); }
  int getAccessOpr() const { return mat.getAccessOpr(); }
  cont_type data() { return mat.getData(); }
  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) { return true; }
  value_type &eval(size_t i) { return mat.eval(i); }
  value_type &eval(size_t i, size_t j) { return mat.eval(i, j); }
  value_type &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

}  // namespace blas

#endif
