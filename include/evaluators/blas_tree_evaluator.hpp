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

template <class LHS, class RHS, typename Device_>
struct Evaluator<AssignExpr<LHS, RHS>, Device_> {
  using Expression = AssignExpr<LHS, RHS>;
  using Device = Device_;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<LHS, Device>::cont_type;
  /* constexpr static bool supported = LHS::supported && RHS::supported; */

  Evaluator<LHS, Device> l;
  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr)
      : l(Evaluator<LHS, Device>(expr.l)), r(Evaluator<RHS, Device>(expr.r)) {}
  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return l.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    l.eval_subexpr_if_needed(nullptr, dev);
    r.eval_subexpr_if_needed(l.data(), dev);
    return true;
  }

  value_type eval(size_t i) { return l.evalref(i) = r.eval(i); }
  value_type eval(cl::sycl::nd_item<1> nditem) {
    return eval(nditem.get_global(0));
  }
  value_type &evalref(size_t i) { return l.evalref(i) = r.eval(i); }
  value_type &evalref(cl::sycl::nd_item<1> nditem) {
    return evalref(nditem.get_global(0));
  }
  void cleanup(Device &dev) {
    l.cleanup(dev);
    r.cleanup(dev);
  }
};

template <class LHS1, class LHS2, class RHS1, class RHS2, typename Device_>
struct Evaluator<DoubleAssignExpr<LHS1, LHS2, RHS1, RHS2>, Device_> {
  using Expression = DoubleAssignExpr<LHS1, LHS2, RHS1, RHS2>;
  using Device = Device_;
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
  cont_type *data() { return l1.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    l1.eval_subexpr_if_needed(nullptr, dev);
    l2.eval_subexpr_if_needed(nullptr, dev);
    r1.eval_subexpr_if_needed(l1.data(), dev);
    r2.eval_subexpr_if_needed(l2.data(), dev);
    return true;
  }

  value_type eval(size_t i) {
    auto val1 = r1.eval(i);
    auto val2 = r2.eval(i);
    l1.evalref(i) = val1;
    l2.evalref(i) = val2;
    return val1;
  }
  value_type eval(cl::sycl::nd_item<1> nditem) {
    return eval(nditem.get_global(0));
  }
  void cleanup(Device &dev) {
    l1.cleanup(dev);
    l2.cleanup(dev);
    r1.cleanup(dev);
    r2.cleanup(dev);
  }
};

template <class Functor, class SCL, class RHS, typename Device_>
struct Evaluator<ScalarExpr<Functor, SCL, RHS>, Device_> {
  using Expression = ScalarExpr<Functor, SCL, RHS>;
  using Device = Device_;
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
  cont_type *data() { return r.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    r.eval_subexpr_if_needed(nullptr, dev);
    return true;
  }

  value_type eval(size_t i) {
    return dev_functor::eval(internal::get_scalar(scl), r.eval(i));
  }
  value_type eval(cl::sycl::nd_item<1> nditem) {
    return eval(nditem.get_global(0));
  }

  void cleanup(Device &dev) { r.cleanup(dev); }
};

template <class Functor, class RHS, typename Device_>
struct Evaluator<UnaryExpr<Functor, RHS>, Device_> {
  using Expression = UnaryExpr<Functor, RHS>;
  using Device = Device_;
  using value_type = typename Expression::value_type;
  using dev_functor = functor_traits<Functor, value_type, Device>;
  using cont_type = typename Evaluator<RHS, Device>::cont_type;
  /* constexpr static bool supported = dev_functor::supported && RHS::supported;
   */

  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr) : r(Evaluator<RHS, Device>(expr.r)) {}
  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return r.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    r.eval_subexpr_if_needed(nullptr, dev);
    return true;
  }

  value_type eval(size_t i) { return dev_functor::eval(r.eval(i)); }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  void cleanup(Device &dev) { r.cleanup(dev); }
};

template <class Functor, class LHS, class RHS, typename Device_>
struct Evaluator<BinaryExpr<Functor, LHS, RHS>, Device_> {
  using Expression = BinaryExpr<Functor, LHS, RHS>;
  using Device = Device_;
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
  cont_type *data() { return l.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    l.eval_subexpr_if_needed(nullptr, dev);
    r.eval_subexpr_if_needed(nullptr, dev);
    return true;
  }

  value_type eval(size_t i) { return dev_functor::eval(l.eval(i), r.eval(i)); }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  void cleanup(Device &dev) {
    l.cleanup(dev);
    r.cleanup(dev);
  }
};

template <class RHS, typename Device_>
struct Evaluator<TupleExpr<RHS>, Device_> {
  using Expression = TupleExpr<RHS>;
  using Device = Device_;
  using value_type = typename Expression::value_type;
  using cont_type = void;
  /* constexpr static bool supported = RHS::supported; */

  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr) : r(Evaluator<RHS, Device>(expr.r)) {}
  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return r.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    r.eval_subexpr_if_needed(nullptr, dev);
    return true;
  }

  value_type eval(size_t i) { return value_type(i, r.eval(i)); }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  void cleanup(Device &dev) { r.cleanup(dev); }
};

template <class RHS>
TupleExpr<RHS> make_tplExpr(RHS r) {
  return TupleExpr<RHS>(r);
}

template <typename EvaluatorT>
struct SubExecutor;

template <class RHS, typename Device_>
struct Evaluator<EmptyExpr<RHS>, Device_> {
  using Expression = EmptyExpr<RHS>;
  using Device = Device_;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<RHS, Device>::cont_type;

  Evaluator<RHS, Device> r;
  Evaluator(Expression &expr) : r(Evaluator<RHS, Device>(expr.r)) {}
  Evaluator(Expression &&expr) : Evaluator(expr) {}

  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return r.data(); }
  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    r.eval_subexpr_if_needed(nullptr, dev);
    return true;
  }
  value_type eval(size_t i) { return r.eval(i); }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  void cleanup(Device &dev) { r.cleanup(dev); }
};

template <class RHS, template <class> class MakePointer>
struct Evaluator<BreakExpr<RHS, MakePointer>, SYCLDevice> {
  using Expression = BreakExpr<RHS, MakePointer>;
  using Device = SYCLDevice;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<RHS, Device>::cont_type;
  using Self = Evaluator<Expression, Device>;
  /* constexpr static bool supported = true; */
  bool allocated_result = false;
  bool defined = false;
  typename MakePointer<value_type>::type result;

  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr)
      : r(Evaluator<RHS, Device>(expr.r)),
        result(expr.use_rhs_result ? data() : nullptr) {}

  Evaluator(Expression &&expr) : Evaluator(expr) {}

  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return r.data(); }

  bool eval_subexpr_if_needed(typename MakePointer<value_type>::type cont,
                              Device &dev) {
    r.eval_subexpr_if_needed(nullptr, dev);
    if (!defined) {
      if (cont) {
        result = cont;
        defined = true;
      } else {
        allocated_result = true;
        result = dev.allocate<value_type>(getSize());
        defined = true;
      }
    }
    SubExecutor<Self>::run(*this, dev);
    return true;
  }

  value_type subeval(size_t i) { return result[i] = r.eval(i); }
  value_type subeval(cl::sycl::nd_item<1> ndItem) {
    return subeval(ndItem.get_global(0));
  }
  value_type eval(size_t i) { return result[i]; }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  value_type &evalref(size_t i) { return result[i]; }
  value_type &evalref(cl::sycl::nd_item<1> &ndItem) {
    return evalref(ndItem.get_global(0));
  }
  void cleanup(Device &dev) {
    r.cleanup(dev);
    if (allocated_result) {
      allocated_result = false;
      dev.deallocate<value_type>(result);
    }
  }
};

template <class RHS>
struct Evaluator<BreakExpr<RHS, MakeDevicePointer>, SYCLDevice> {
  using Expression = BreakExpr<RHS, MakeDevicePointer>;
  using Device = SYCLDevice;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<RHS, Device>::cont_type;
  using Self = Evaluator<Expression, Device>;
  /* constexpr static bool supported = true; */
  bool allocated_result = false;
  typename MakeDevicePointer<value_type>::type result;

  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr) : r(Evaluator<RHS, Device>(expr.r)) {}

  Evaluator(Expression &&expr) : Evaluator(expr) {}

  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return r.data(); }
  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    r.eval_subexpr_if_needed(nullptr, dev);
    return true;
  }
  value_type subeval(size_t i) { return result[i] = r.eval(i); }
  value_type subeval(cl::sycl::nd_item<1> ndItem) {
    return subeval(ndItem.get_global(0));
  }
  value_type eval(size_t i) { return result[i]; }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  value_type &evalref(size_t i) { return result[i]; }
  value_type &evalref(cl::sycl::nd_item<1> &ndItem) {
    return evalref(ndItem.get_global(0));
  }
  void cleanup(Device &dev) { r.cleanup(dev); }
};

template <class RHS, template <class> class MakePointer, class Device_>
struct Evaluator<BreakIfExpr<RHS, MakePointer>, Device_> {
  using Expression = BreakIfExpr<RHS, MakePointer>;
  using Device = Device_;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<RHS, Device>::cont_type;

  using RHS_empty = EmptyExpr<RHS>;
  using RHS_break = BreakExpr<RHS, MakePointer>;

  const bool to_break;
  Evaluator<RHS_empty, Device> r_empty;
  Evaluator<RHS_break, Device> r_break;
  size_t N;

  // expr is the expression we want to conditionally isolate from
  Evaluator(Expression &expr)
      : to_break(expr.to_break),
        r_empty(Evaluator<RHS_empty, Device>(expr.r_empty)),
        r_break(Evaluator<RHS_break, Device>(expr.r_break)),
        N(r_empty.getSize()) {}

  Evaluator(Expression &&expr) : Evaluator(expr) {}

  size_t getSize() const { return N; }

  cont_type *data() {
    if (to_break) {
      return r_break.data();
    } else {
      return r_empty.data();
    }
  }
  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    if (to_break) {
      r_break.eval_subexpr_if_needed(cont, dev);
    } else {
      r_empty.eval_subexpr_if_needed(cont, dev);
    }
    return true;
  }
  value_type eval(size_t i) { return r_break.eval(i); }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  value_type eval_alt(size_t i) { return r_empty.eval(i); }
  value_type eval_alt(cl::sycl::nd_item<1> ndItem) {
    return eval_alt(ndItem.get_global(0));
  }
  void cleanup(Device &dev) {
    if (to_break) {
      r_break.cleanup(dev);
    } else {
      r_empty.cleanup(dev);
    }
  }
};

template <class RHS, template <class> class MakePointer>
struct Evaluator<StrideExpr<RHS, MakePointer>, SYCLDevice> {
  using Expression = StrideExpr<RHS, MakePointer>;
  using Device = SYCLDevice;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<RHS, Device>::cont_type;
  /* constexpr static bool supported = true; */

  Evaluator<BreakIfExpr<RHS, MakePointer>, Device> r;
  long offt;
  long strd;
  size_t N;

  // expr is an expression we want to change the stride for
  Evaluator(Expression &expr)
      :  // branch
        r(Evaluator<BreakIfExpr<RHS, MakePointer>, Device>(
            BreakIfExpr<RHS, MakePointer>(expr.r,
                                          !(expr.offt == 0 && expr.strd == 1 &&
                                            expr.N == expr.getSize())))),
        offt(expr.offt),
        strd(expr.strd),
        N(expr.N) {}
  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return r.data(); }
  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    r.eval_subexpr_if_needed(nullptr, dev);
    return true;
  }
  value_type eval(size_t i) {
    return r.eval(offt + (strd >= 0 ? i * strd : (N - i - 1) * strd));
  }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  void cleanup(Device &dev) { return r.cleanup(dev); }
};

template <typename ScalarT, typename ContainerT,
          template <class> class MakePointer>
struct Evaluator<StrideExpr<vector_view<ScalarT, ContainerT>, MakePointer>,
                 SYCLDevice> {
  using RHS = vector_view<ScalarT, ContainerT>;
  using Expression = StrideExpr<RHS, MakePointer>;
  using Device = SYCLDevice;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<RHS, Device>::cont_type;
  /* constexpr static bool supported = true; */

  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr)
      : r(Evaluator<RHS, Device>(
            RHS(expr.r, expr.r.getDisp() + expr.offt, expr.strd, expr.N))) {}
  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return r.data(); }
  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    r.eval_subexpr_if_needed(nullptr, dev);
    return true;
  }
  value_type eval(size_t i) { return r.eval(i); }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  value_type &evalref(size_t i) { return r.evalref(i); }
  value_type &evalref(cl::sycl::nd_item<1> ndItem) {
    return evalref(ndItem.get_global(0));
  }
  void cleanup(Device &dev) { return r.cleanup(dev); }
};

template <class ScalarT, class ContainerT, typename Device_>
struct Evaluator<vector_view<ScalarT, ContainerT>, Device_> {
  using Expression = vector_view<ScalarT, ContainerT>;
  using Device = Device_;
  using value_type = typename Expression::value_type;
  using cont_type = ContainerT;
  /* constexpr static bool supported = true; */

  vector_view<ScalarT, ContainerT> vec;

  Evaluator(Expression vec) : vec(vec) {}
  size_t getSize() const { return vec.getSize(); }
  cont_type *data() { return &vec.getData(); }
  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) { return true; }
  value_type eval(size_t i) { return evalref(i); }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  value_type &evalref(size_t i) {
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
  value_type &evalref(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  void cleanup(Device &dev) {}
};

template <class ScalarT, class ContainerT, typename Device_>
struct Evaluator<matrix_view<ScalarT, ContainerT>, Device_> {
  using Expression = matrix_view<ScalarT, ContainerT>;
  using Device = Device_;
  using value_type = typename Expression::value_type;
  using cont_type = ContainerT;
  /* constexpr static bool supported = true; */

  matrix_view<ScalarT, ContainerT> mat;

  Evaluator(Expression mat) : mat(mat) {}
  size_t getSize() const { return mat.getSizeR(); }
  size_t getSizeR() const { return mat.getSizeR(); }
  size_t getSizeC() const { return mat.getSizeC(); }
  int getAccess() const { return mat.getAccess(); }
  int getAccessOpr() const { return mat.getAccessOpr(); }
  cont_type *data() { return &mat.getData(); }
  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) { return true; }
  value_type eval(size_t i) { return mat.eval(i); }
  value_type eval(size_t i, size_t j) { return mat.eval(i, j); }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  value_type &evalref(size_t i) { return mat.eval(i); }
  value_type &evalref(size_t i, size_t j) { return mat.evalref(i, j); }
  value_type &evalref(cl::sycl::nd_item<1> ndItem) {
    return evalref(ndItem.get_global(0));
  }
  void cleanup(Device &dev) {}
};

}  // namespace blas

#endif
