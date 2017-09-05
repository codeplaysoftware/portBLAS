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
#include <executors/blas_device_sycl.hpp>
#include <operations/blas_trees.hpp>

namespace blas {

namespace detail {

  template <bool NEED_ASSIGN, class Self>
  struct assigner {
    static typename Self::value_type &assign(size_t i, Self self) {
      return self.l.evalref(i) = self.r.eval(i);
    }
  };
  template <class Self>
  struct assigner<false, Self> {
    static typename Self::value_type &assign(size_t i, Self self) {
      return self.l.evalref(i);
    }
  };

} // namespace detail

template <class LHS, class RHS, typename Device_>
struct Evaluator<AssignExpr<LHS, RHS>, Device_> {
  using Expression = AssignExpr<LHS, RHS>;
  using Device = Device_;
  using Self = Evaluator<AssignExpr<LHS, RHS>, Device>;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<LHS, Device>::cont_type;

  static constexpr bool needassign = Evaluator<RHS, Device>::needassign;

  Evaluator<LHS, Device> l;
  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr):
    l(Evaluator<LHS, Device>(expr.l)),
    r(Evaluator<RHS, Device>(expr.r))
  {}
  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return l.data(); }

  template <typename AssignEvaluatorT = void>
  bool eval_subexpr_if_needed(cont_type *cont, AssignEvaluatorT *assign, Device &dev) {
    l.template eval_subexpr_if_needed<void>(nullptr, nullptr, dev);
    return r.template eval_subexpr_if_needed<Self>(l.data(), this, dev);
  }

  template <bool NEED_ASSIGN> value_type eval_(size_t i) {
    return detail::assigner<NEED_ASSIGN, Self>::assign(i, *this);
  }
  template <bool NEED_ASSIGN> value_type eval_(cl::sycl::nd_item<1> ndItem) {
    return eval_<NEED_ASSIGN>(ndItem.get_global(0));
  }
  template <bool NEED_ASSIGN> value_type evalref_(size_t i) {
    return detail::assigner<NEED_ASSIGN, Self>::assign(i, *this);
  }
  template <bool NEED_ASSIGN> value_type evalref_(cl::sycl::nd_item<1> ndItem) {
    return evalref_<NEED_ASSIGN>(ndItem.get_global(0));
  }

  value_type eval(size_t i) { return eval_<needassign>(i); }
  value_type eval(cl::sycl::nd_item<1> nditem) { return eval(nditem.get_global(0)); }
  value_type &evalref(size_t i) { return evalref_<needassign>(i); }
  value_type &evalref(cl::sycl::nd_item<1> nditem) { return evalref(nditem.get_global(0)); }
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

  static constexpr bool needassign = true;

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

  template <typename AssignEvaluatorT = void>
  bool eval_subexpr_if_needed(cont_type *cont, AssignEvaluatorT *assign, Device &dev) {
    l1.template eval_subexpr_if_needed<AssignEvaluatorT>(nullptr, nullptr, dev);
    l2.template eval_subexpr_if_needed<AssignEvaluatorT>(nullptr, nullptr, dev);
    r1.template eval_subexpr_if_needed<AssignEvaluatorT>(l1.data(), nullptr, dev);
    r2.template eval_subexpr_if_needed<AssignEvaluatorT>(l2.data(), nullptr, dev);
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

  static constexpr bool needassign = true;

  SCL scl;
  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr):
    scl(expr.scl), r(Evaluator<RHS, Device>(expr.r))
  {}
  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return r.data(); }
  template <typename AssignEvaluatorT = void>
  bool eval_subexpr_if_needed(cont_type *cont, AssignEvaluatorT *assign, Device &dev) {
    r.template eval_subexpr_if_needed<AssignEvaluatorT>(nullptr, nullptr, dev);
    return true;
  }
  value_type eval(size_t i) {
    return dev_functor::eval(internal::get_scalar(scl), r.eval(i));
  }
  value_type eval(cl::sycl::nd_item<1> nditem) { return eval(nditem.get_global(0)); }
  void cleanup(Device &dev) { r.cleanup(dev); }
};

template <class Functor, class RHS, typename Device_>
struct Evaluator<UnaryExpr<Functor, RHS>, Device_> {
  using Expression = UnaryExpr<Functor, RHS>;
  using Device = Device_;
  using value_type = typename Expression::value_type;
  using dev_functor = functor_traits<Functor, value_type, Device>;
  using cont_type = typename Evaluator<RHS, Device>::cont_type;

  static constexpr bool needassign = Evaluator<RHS, Device>::needassign;

  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr) : r(Evaluator<RHS, Device>(expr.r)) {}
  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return r.data(); }

  template <typename AssignEvaluatorT = void>
  bool eval_subexpr_if_needed(cont_type *cont, AssignEvaluatorT *assign, Device &dev) {
    return r.template eval_subexpr_if_needed<AssignEvaluatorT>(cont, assign, dev);
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

  static constexpr bool needassign = true;

  Evaluator<LHS, Device> l;
  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr):
    l(Evaluator<LHS, Device>(expr.l)),
    r(Evaluator<RHS, Device>(expr.r))
  {}
  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return l.data(); }

  template <typename AssignEvaluatorT = void>
  bool eval_subexpr_if_needed(cont_type *cont, AssignEvaluatorT *assign, Device &dev) {
    l.template eval_subexpr_if_needed<AssignEvaluatorT>(nullptr, nullptr, dev);
    r.template eval_subexpr_if_needed<AssignEvaluatorT>(nullptr, nullptr, dev);
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

  static constexpr bool needassign = true;

  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr) : r(Evaluator<RHS, Device>(expr.r)) {}
  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return r.data(); }

  template <typename AssignEvaluatorT = void>
  bool eval_subexpr_if_needed(cont_type *cont, AssignEvaluatorT *assign, Device &dev) {
    r.template eval_subexpr_if_needed<AssignEvaluatorT>(nullptr, nullptr, dev);
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

template <typename AssignEvaluatorT>
struct SubExecutor;
template <class RHS, template <class> class MakePointer>

struct Evaluator<BreakExpr<RHS, MakePointer>, SYCLDevice> {
  using Expression = BreakExpr<RHS, MakePointer>;
  using Device = SYCLDevice;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<RHS, Device>::cont_type;
  using Self = Evaluator<Expression, Device>;
  bool allocated_result = false;
  bool defined = false;
  typename MakePointer<value_type>::type result;

  static constexpr bool needassign = true;

  Evaluator<RHS, Device> r;
  bool to_break;

  Evaluator(Expression &expr):
    r(Evaluator<RHS, Device>(expr.r)),
    result(expr.use_rhs_result ? data() : nullptr),
    to_break(expr.to_break)
  {}

  Evaluator(Expression &&expr) : Evaluator(expr) {}

  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return r.data(); }

  template <typename AssignEvaluatorT>
  bool eval_subexpr_if_needed(typename MakePointer<value_type>::type cont, AssignEvaluatorT *assign, Device &dev) {
    r.template eval_subexpr_if_needed<AssignEvaluatorT>(nullptr, nullptr, dev);
    if (!defined) {
      if (cont) {
        result = cont;
        defined = true;
      } else {
        allocated_result = true;
        result = dev.allocate<value_type>(to_break ? getSize() : 1);
        defined = true;
      }
    }
    if(to_break) {
      SubExecutor<Self>::run(*this, dev);
    }
    return (cont == nullptr);
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
  typename MakeDevicePointer<value_type>::type result;

  static constexpr bool needassign = true;

  bool to_break;
  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr):
    r(Evaluator<RHS, Device>(expr.r)),
    result(MakeDevicePointer<value_type>::init()),
    to_break(expr.to_break)
  {}

  Evaluator(Expression &&expr):
    Evaluator(expr)
  {}

  size_t getSize() const { return r.getSize(); }
  value_type subeval(size_t i) {
    return result[i] = r.eval(i);
  }
  value_type subeval(cl::sycl::nd_item<1> ndItem) { return subeval(ndItem.get_global(0)); }
  value_type eval(size_t i) {
    if(to_break) {
      return result[i];
    }
    return r.eval(i);
  }
  value_type eval(cl::sycl::nd_item<1> ndItem) { return eval(ndItem.get_global(0)); }
  value_type &evalref(size_t i) {
    return result[i];
  }
  value_type &evalref(cl::sycl::nd_item<1> &ndItem) { return evalref(ndItem.get_global(0)); }
  void cleanup(Device &dev) { r.cleanup(dev); }
};

template <class RHS, template <class> class MakePointer>
struct Evaluator<StrideExpr<RHS, MakePointer>, SYCLDevice> {
  using Expression = StrideExpr<RHS, MakePointer>;
  using Device = SYCLDevice;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<BreakExpr<RHS, MakePointer>, Device>::cont_type;

  static constexpr bool needassign = true;

  Evaluator<RHS, Device> r;
  const long offt;
  const long strd;
  const size_t N;

  Evaluator(Expression &expr):
    r(expr.r),
    offt(expr.offt),
    strd(expr.strd),
    N(expr.N)
  {}

  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return r.data(); }
  template <typename AssignEvaluatorT = void>
  bool eval_subexpr_if_needed(cont_type *cont, AssignEvaluatorT *assign, Device &dev) {
    r.template eval_subexpr_if_needed<AssignEvaluatorT>(nullptr, nullptr, dev);
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

template <class ScalarT, class ContainerT, typename Device_>
struct Evaluator<vector_view<ScalarT, ContainerT>, Device_> {
  using Expression = vector_view<ScalarT, ContainerT>;
  using Device = Device_;
  using value_type = typename Expression::value_type;
  using cont_type = ContainerT;

  static constexpr bool needassign = true;

  vector_view<ScalarT, ContainerT> vec;

  Evaluator(Expression vec) : vec(vec) {}
  size_t getSize() const { return vec.getSize(); }
  cont_type *data() { return &vec.getData(); }
  template <typename AssignEvaluatorT = void>
  bool eval_subexpr_if_needed(cont_type *cont, AssignEvaluatorT *assign, Device &dev) { return true; }
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

  matrix_view<ScalarT, ContainerT> mat;

  Evaluator(Expression mat) : mat(mat) {}
  size_t getSize() const { return mat.getSizeR(); }
  size_t getSizeR() const { return mat.getSizeR(); }
  size_t getSizeC() const { return mat.getSizeC(); }
  int getAccess() const { return mat.getAccess(); }
  int getAccessOpr() const { return mat.getAccessOpr(); }
  cont_type *data() { return &mat.getData(); }
  template <typename AssignEvaluatorT = void>
  bool eval_subexpr_if_needed(cont_type *cont, AssignEvaluatorT *assign, Device &dev) { return true; }
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
