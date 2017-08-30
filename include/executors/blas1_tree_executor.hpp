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
 *  @filename blas1_tree_executor.hpp
 *
 **************************************************************************/

#ifndef BLAS1_TREE_EXECUTOR_HPP
#define BLAS1_TREE_EXECUTOR_HPP

#include <stdexcept>

#include <evaluators/blas1_tree_evaluator.hpp>
#include <evaluators/blas_tree_evaluator.hpp>
#include <executors/blas_device_sycl.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/*! Converter
 * @brief Converter the expression tree passed, converting node types.
 * This set of template specializations is used to convert the buffer
 * types of the expression tree into accessors, suitable to be used on
 * the kernel scope.
 * When using the expression tree on the device, developers call the
 * make_accessor function, which starts processing the tree.
 */
template <typename EvaluatorT> struct Converter;

/*! Converter<AssigExpr<LHS, RHS>, SYCLDevice>
 * @brief See Converter.
 */
template <typename LHS, typename RHS>
struct Converter<Evaluator<AssignExpr<LHS, RHS>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS, SYCLDevice>::value_type;
  using lhs_type = typename Converter<Evaluator<LHS, SYCLDevice>>::out_type;
  using rhs_type = typename Converter<Evaluator<RHS, SYCLDevice>>::out_type;
  using cont_type = typename Converter<Evaluator<LHS, SYCLDevice>>::cont_type;
  using input_type = Evaluator<AssignExpr<LHS, RHS>, SYCLDevice>;
  using out_type = AssignExpr<lhs_type, rhs_type>;

  static out_type convert_to(input_type t, cl::sycl::handler &h) {
    auto lhs = Converter<Evaluator<LHS, SYCLDevice>>::convert_to(t.l, h);
    auto rhs = Converter<Evaluator<RHS, SYCLDevice>>::convert_to(t.r, h);
    return out_type(lhs, rhs);
  }

  static void bind_to(input_type t, Evaluator<out_type, SYCLDevice> ev,
                      cl::sycl::handler &h) {
    Converter<Evaluator<LHS, SYCLDevice>>::bind_to(t.l, ev.l, h);
    Converter<Evaluator<RHS, SYCLDevice>>::bind_to(t.r, ev.r, h);
  }
};

/*! Converter <DoubleAssign<LHS, RHS>, SYCLDevice>
 * @brief See Converter.
 */
template <class LHS1, class LHS2, class RHS1, class RHS2>
struct Converter<Evaluator<DoubleAssignExpr<LHS1, LHS2, RHS1, RHS2>, SYCLDevice>> {
  using lhs1_type = typename Converter<Evaluator<LHS1, SYCLDevice>>::out_type;
  using lhs2_type = typename Converter<Evaluator<LHS2, SYCLDevice>>::out_type;
  using rhs1_type = typename Converter<Evaluator<RHS1, SYCLDevice>>::out_type;
  using rhs2_type = typename Converter<Evaluator<RHS2, SYCLDevice>>::out_type;
  using cont_type = typename Converter<Evaluator<LHS1, SYCLDevice>>::cont_type;
  using input_type = Evaluator<DoubleAssignExpr<LHS1, LHS2, RHS1, RHS2>, SYCLDevice>;
  using out_type = DoubleAssignExpr<lhs1_type, lhs2_type, rhs1_type, rhs2_type>;

  static out_type convert_to(input_type t, cl::sycl::handler &h) {
    auto lhs1 = Converter<Evaluator<LHS1, SYCLDevice>>::convert_to(t.l1, h);
    auto lhs2 = Converter<Evaluator<LHS2, SYCLDevice>>::convert_to(t.l2, h);
    auto rhs1 = Converter<Evaluator<RHS1, SYCLDevice>>::convert_to(t.r1, h);
    auto rhs2 = Converter<Evaluator<RHS2, SYCLDevice>>::convert_to(t.r2, h);
    return out_type(lhs1, lhs2, rhs1, rhs2);
  }

  static void bind_to(input_type t, Evaluator<out_type, SYCLDevice> ev, cl::sycl::handler &h) {
    Converter<Evaluator<LHS1, SYCLDevice>>::bind_to(t.l1, ev.l1, h);
    Converter<Evaluator<LHS2, SYCLDevice>>::bind_to(t.l2, ev.l2, h);
    Converter<Evaluator<RHS1, SYCLDevice>>::bind_to(t.r1, ev.r1, h);
    Converter<Evaluator<RHS2, SYCLDevice>>::bind_to(t.r2, ev.r2, h);
  }
};

/*! Converter<ScalarExpr<Functor, SCL, RHS>, SYCLDevice>
 * @brief See Converter.
 */
template <typename Functor, typename SCL, typename RHS>
struct Converter<Evaluator<ScalarExpr<Functor, SCL, RHS>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS, SYCLDevice>::value_type;
  using scl_type = value_type;
  using rhs_type = typename Converter<Evaluator<RHS, SYCLDevice>>::out_type;
  using cont_type = typename Evaluator<RHS, SYCLDevice>::cont_type;
  using input_type = Evaluator<ScalarExpr<Functor, SCL, RHS>, SYCLDevice>;
  using out_type = ScalarExpr<Functor, scl_type, rhs_type>;

  static out_type convert_to(input_type t, cl::sycl::handler &h) {
    auto rhs = Converter<Evaluator<RHS, SYCLDevice>>::convert_to(t.r, h);
    return out_type(t.scl, rhs);
  }

  static void bind_to(input_type t, Evaluator<out_type, SYCLDevice> ev, cl::sycl::handler &h) {
    Converter<Evaluator<RHS, SYCLDevice>>::bind_to(t.r, ev.r, h);
  }
};

/*! Converter<TupleExpr<Functor, RHS>, SYCLDevice>
 * @brief See Converter.
 */
template <typename RHS>
struct Converter<Evaluator<TupleExpr<RHS>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS, SYCLDevice>::value_type;
  using rhs_type = typename Converter<Evaluator<RHS, SYCLDevice>>::out_type;
  using cont_type = typename Converter<Evaluator<RHS, SYCLDevice>>::cont_type;
  using input_type = Evaluator<TupleExpr<RHS>, SYCLDevice>;
  using out_type = TupleExpr<rhs_type>;

  static out_type convert_to(input_type t, cl::sycl::handler &h) {
    auto rhs = Converter<Evaluator<RHS, SYCLDevice>>::convert_to(t.r, h);
    return out_type(rhs);
  }

  static void bind_to(input_type t, Evaluator<out_type, SYCLDevice> ev,
                      cl::sycl::handler &h) {
    Converter<Evaluator<RHS, SYCLDevice>>::bind_to(t.r, ev.r, h);
  }
};

/*! Converter<UnaryExpr<Functor, RHS>, SYCLDevice>
 * @brief See Converter.
 */
template <typename Functor, typename RHS>
struct Converter<Evaluator<UnaryExpr<Functor, RHS>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS, SYCLDevice>::value_type;
  using rhs_type = typename Converter<Evaluator<RHS, SYCLDevice>>::out_type;
  using cont_type = typename Converter<Evaluator<RHS, SYCLDevice>>::cont_type;
  using input_type = Evaluator<UnaryExpr<Functor, RHS>, SYCLDevice>;
  using out_type = UnaryExpr<Functor, rhs_type>;

  static out_type convert_to(input_type t, cl::sycl::handler &h) {
    auto rhs = Converter<Evaluator<RHS, SYCLDevice>>::convert_to(t.r, h);
    return out_type(rhs);
  }

  static void bind_to(input_type t, Evaluator<out_type, SYCLDevice> ev, cl::sycl::handler &h) {
    Converter<Evaluator<RHS, SYCLDevice>>::bind_to(t.r, ev.r, h);
  }
};

/*! Converter<BinaryExpr<Functor, LHS, RHS>, SYCLDevice>
 * @brief See Converter.
 */
template <typename Functor, typename LHS, typename RHS>
struct Converter<Evaluator<BinaryExpr<Functor, LHS, RHS>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS, SYCLDevice>::value_type;
  using lhs_type = typename Converter<Evaluator<LHS, SYCLDevice>>::out_type;
  using rhs_type = typename Converter<Evaluator<RHS, SYCLDevice>>::out_type;
  using cont_type = typename Evaluator<LHS, SYCLDevice>::cont_type;
  using input_type = Evaluator<BinaryExpr<Functor, LHS, RHS>, SYCLDevice>;
  using out_type = BinaryExpr<Functor, lhs_type, rhs_type>;

  static out_type convert_to(input_type t, cl::sycl::handler &h) {
    auto lhs = Converter<Evaluator<LHS, SYCLDevice>>::convert_to(t.l, h);
    auto rhs = Converter<Evaluator<RHS, SYCLDevice>>::convert_to(t.r, h);
    return out_type(lhs, rhs);
  }

  static void bind_to(input_type t, Evaluator<out_type, SYCLDevice> ev, cl::sycl::handler &h) {
    Converter<Evaluator<LHS, SYCLDevice>>::bind_to(t.l, ev.l, h);
    Converter<Evaluator<RHS, SYCLDevice>>::bind_to(t.r, ev.r, h);
  }
};

/*! Converter<Reduction<Operator, LHS, RHS>, SYCLDevice>
 * @brief See Converter.
 */
template <typename Functor, typename RHS>
struct Converter<Evaluator<ReductionExpr<Functor, RHS, MakeHostPointer>, SYCLDevice>> {
  using Expression = ReductionExpr<Functor, RHS, MakeHostPointer>;
  using value_type = typename Evaluator<Expression, SYCLDevice>::value_type;
  using oper_type = Functor;
  using rhs_type = typename Converter<Evaluator<RHS, SYCLDevice>>::out_type;
  using cont_type = typename Evaluator<RHS, SYCLDevice>::cont_type;
  using input_type = Evaluator<Expression, SYCLDevice>;
  using out_type = ReductionExpr<Functor, rhs_type, MakeDevicePointer>;

  static out_type convert_to(input_type t, cl::sycl::handler &h) {
    auto rhs = Converter<Evaluator<RHS, SYCLDevice>>::convert_to(t.r, h);
    return out_type(rhs);
  }

  static void bind_to(input_type t, Evaluator<out_type, SYCLDevice> ev, cl::sycl::handler &h) {
    Converter<Evaluator<RHS, SYCLDevice>>::bind_to(t.r, ev.r, h);
    h.require(*t.result, ev.result);
  }
};

/*! Converter<BreakExpr<RHS, MakePointer>, SYCLDevice>>
 * @brief See Converter.
 */
template <typename RHS>
struct Converter<Evaluator<BreakExpr<RHS, MakeHostPointer>, SYCLDevice>> {
  using Expression = BreakExpr<RHS, MakeHostPointer>;
  using value_type = typename Evaluator<Expression, SYCLDevice>::value_type;
  using rhs_type = typename Converter<Evaluator<RHS, SYCLDevice>>::out_type;
  using cont_type = typename Evaluator<RHS, SYCLDevice>::cont_type;
  using input_type = Evaluator<Expression, SYCLDevice>;
  using out_type = BreakExpr<rhs_type, MakeDevicePointer>;
  static out_type convert_to(input_type t, cl::sycl::handler &h) {
    auto rhs = Converter<Evaluator<RHS, SYCLDevice>>::convert_to(t.r, h);
    return out_type(rhs, t.to_break);
  }
  static void bind_to(input_type t, Evaluator<out_type, SYCLDevice> ev, cl::sycl::handler &h) {
    Converter<Evaluator<RHS, SYCLDevice>>::bind_to(t.r, ev.r, h);
    /* if(t.to_break) { */
      h.require(*t.result, ev.result);
    /* } */
  }
};

/*! Converter<StrideExpr<RHS, MakePointer>, SYCLDevice>>
 * @brief See Converter.
 */
template <typename RHS>
struct Converter<Evaluator<StrideExpr<RHS, MakeHostPointer>, SYCLDevice>> {
  using Expression = StrideExpr<RHS, MakeHostPointer>;
  using value_type = typename Evaluator<Expression, SYCLDevice>::value_type;
  using rhs_type = typename Converter<Evaluator<RHS, SYCLDevice>>::out_type;
  using cont_type = typename Evaluator<RHS, SYCLDevice>::cont_type;
  using input_type = Evaluator<Expression, SYCLDevice>;
  using out_type = StrideExpr<rhs_type, MakeDevicePointer>;
  static out_type convert_to(input_type t, cl::sycl::handler &h) {
    auto rhs = Converter<Evaluator<RHS, SYCLDevice>>::convert_to(t.r, h);
    return out_type(rhs, t.offt, t.strd, t.N);
  }
  static void bind_to(input_type t, Evaluator<out_type, SYCLDevice> ev, cl::sycl::handler &h) {
    Converter<Evaluator<RHS, SYCLDevice>>::bind_to(t.r, ev.r, h);
  }
};

/*! Converter<vector_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>, SYCLDevice>
 * @brief See Converter.
 */
template <typename ScalarT>
struct Converter<Evaluator<vector_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>, SYCLDevice>> {
  using value_type = ScalarT;
  using cont_type = cl::sycl::buffer<ScalarT, 1>;
  using input_type = Evaluator<vector_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>, SYCLDevice>;
  using nested_type = cl::sycl::buffer<ScalarT, 1>;
  using out_type = vector_view<value_type, cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>>;

  static out_type convert_to(input_type t, cl::sycl::handler &h) {
    auto nested = cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>(t.vec.data_, h);
    return out_type(nested, t.vec.disp_, t.vec.strd_, t.vec.size_);
  }

  static void bind_to(input_type t, Evaluator<out_type, SYCLDevice> ev, cl::sycl::handler &h) {}
};

/*! Converter<matrix_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>, SYCLDevice>
 * @brief See Converter.
 */
template <typename ScalarT>
struct Converter<Evaluator<matrix_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>, SYCLDevice>> {
  using value_type = ScalarT;
  using cont_type = cl::sycl::buffer<ScalarT, 1>;
  using input_type = Evaluator<matrix_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>, SYCLDevice>;
  using nested_type = cl::sycl::buffer<ScalarT, 1>;
  using out_type = matrix_view<ScalarT, cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>>;

  static out_type convert_to(input_type t, cl::sycl::handler &h) {
    auto nested = cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>(t.mat.data_, h);
    return out_type(nested, t.mat.accessDev_, t.mat.sizeR_, t.mat.sizeC_, t.mat.accessOpr_, t.mat.sizeL_, t.mat.disp_);
  }

  static void bind_to(input_type t, Evaluator<out_type, SYCLDevice> ev, cl::sycl::handler &h) {}
};

/*! Converter<matrix_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>, SYCLDevice>
 * @brief See Converter.
 */
template <typename ScalarT, typename ContainerT>
struct Converter<Evaluator<vector_view<ScalarT, ContainerT>, SYCLDevice>> {
  using value_type = ScalarT;
  using cont_type = ContainerT;
  using input_type = Evaluator<vector_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>, SYCLDevice>;
  using nested_type = ContainerT;
  using out_type = vector_view<ScalarT, nested_type>;

  static out_type convert_to(input_type t, cl::sycl::handler &h) = delete;
  static void bind_to(input_type t, Evaluator<out_type, SYCLDevice> ev, cl::sycl::handler &h) {}
};

/*! Converter<matrix_view<ScalarT, ContainerT>, SYCLDevice>
 * @brief See Converter.
 */
template <typename ScalarT, typename ContainerT>
struct Converter<Evaluator<matrix_view<ScalarT, ContainerT>, SYCLDevice>> {
  using value_type = ScalarT;
  using cont_type = ContainerT;
  using input_type = Evaluator<matrix_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>, SYCLDevice>;
  using nested_type = ContainerT;
  using out_type = matrix_view<ScalarT, nested_type>;

  static out_type convert_to(input_type t, cl::sycl::handler &h) = delete;
  static void bind_to(input_type t, Evaluator<out_type, SYCLDevice> ev, cl::sycl::handler &h) {}
};

/** make_accessor.
 * Triggers the conversion of the Expression Tree into the Evaluation
 * tree by calling the convert_to method.
 * @param Tree The Input Expression Tree.
 * @param handler The Command Group Handler used to create the accessors
 */
template <typename EvaluatorT>
Evaluator<typename Converter<EvaluatorT>::out_type, SYCLDevice> make_accessor(EvaluatorT evh, cl::sycl::handler &h) {
  using converter = Converter<EvaluatorT>;
  auto expr = converter::convert_to(evh, h);
  auto evd = Evaluator<decltype(expr), SYCLDevice>(expr);
  converter::bind_to(evh, evd, h);
  return evd;
}

}  // namespace BLAS

#endif  // BLAS1_TREE_EXECUTOR_HPP
