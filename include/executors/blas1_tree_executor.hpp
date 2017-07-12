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

#include <CL/sycl.hpp>

#include <evaluators/blas1_tree_evaluator.hpp>
#include <evaluators/blas2_tree_evaluator.hpp>
#include <evaluators/blas3_tree_evaluator.hpp>
#include <executors/blas_packet_traits_sycl.hpp>
#include <executors/executor_base.hpp>
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

template <typename Evaluator>
struct Converter {
  using value_type = Evaluator;
  using input_type = Evaluator;
  using type = Evaluator;

  /** convert_to.
   * @brief .
   */
  static type convert_to(input_type v, cl::sycl::handler &h) { return v; }
};

/*! Converter <Join<LHS, RHS>>.
 * @brief See Converter.
 */
template <typename LHS, typename RHS>
struct Converter<Evaluator<JoinExpr<LHS, RHS>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS, SYCLDevice>::value_type;
  using lhs_type = typename Converter<Evaluator<LHS, SYCLDevice>>::type;
  using rhs_type = typename Converter<Evaluator<RHS, SYCLDevice>>::type;
  using input_type = Evaluator<JoinExpr<LHS, RHS>, SYCLDevice>;
  using type = Evaluator<JoinExpr<lhs_type, rhs_type>, SYCLDevice>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Converter<Evaluator<LHS, SYCLDevice>>::convert_to(v.l, h);
    auto rhs = Converter<Evaluator<RHS, SYCLDevice>>::convert_to(v.r, h);
    return type(lhs, rhs);
  }
};

/*! Converter <Assign<LHS, RHS>>a
 * @brief See Converter.
 */
template <typename LHS, typename RHS>
struct Converter<Evaluator<AssignExpr<LHS, RHS>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS, SYCLDevice>::value_type;
  using lhs_type = typename Converter<Evaluator<LHS, SYCLDevice>>::type;
  using rhs_type = typename Converter<Evaluator<RHS, SYCLDevice>>::type;
  using input_type = Evaluator<AssignExpr<LHS, RHS>, SYCLDevice>;
  using type = Evaluator<AssignExpr<lhs_type, rhs_type>, SYCLDevice>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Converter<Evaluator<LHS, SYCLDevice>>::convert_to(v.l, h);
    auto rhs = Converter<Evaluator<RHS, SYCLDevice>>::convert_to(v.r, h);
    return type(lhs, rhs);
  }
};

/*! Converter <DoubleAssign<LHS, RHS>>
 * @brief See Converter.
 */
template <class LHS1, class LHS2, class RHS1, class RHS2>
struct Converter<
    Evaluator<DoubleAssignExpr<LHS1, LHS2, RHS1, RHS2>, SYCLDevice>> {
  using lhs1_type = typename Converter<Evaluator<LHS1, SYCLDevice>>::type;
  using lhs2_type = typename Converter<Evaluator<LHS2, SYCLDevice>>::type;
  using rhs1_type = typename Converter<Evaluator<RHS1, SYCLDevice>>::type;
  using rhs2_type = typename Converter<Evaluator<RHS2, SYCLDevice>>::type;
  using input_type = Evaluator<DoubleAssignExpr<LHS1, LHS2, RHS1, RHS2>, SYCLDevice>;
  using type = Evaluator<DoubleAssignExpr<lhs1_type, lhs2_type, rhs1_type, rhs2_type>, SYCLDevice>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs1 = Converter<Evaluator<LHS1, SYCLDevice>>::convert_to(v.l1, h);
    auto lhs2 = Converter<Evaluator<LHS2, SYCLDevice>>::convert_to(v.l2, h);
    auto rhs1 = Converter<Evaluator<RHS1, SYCLDevice>>::convert_to(v.r1, h);
    auto rhs2 = Converter<Evaluator<RHS2, SYCLDevice>>::convert_to(v.r2, h);
    return type(lhs1, lhs2, rhs1, rhs2);
  }
};

/*! Converter<ScalarOp<Operator, SCL, RHS>>
 * @brief See Converter.
 */
template <typename Functor, typename SCL, typename RHS>
struct Converter<Evaluator<ScalarExpr<Functor, SCL, RHS>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS, SYCLDevice>::value_type;
  using scl_type = typename Converter<Evaluator<SCL, SYCLDevice>>::type;
  using rhs_type = typename Converter<Evaluator<RHS, SYCLDevice>>::type;
  using input_type = Evaluator<ScalarExpr<Functor, SCL, RHS>, SYCLDevice>;
  using type = Evaluator<ScalarExpr<Functor, scl_type, rhs_type>, SYCLDevice>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto scl = Converter<Evaluator<SCL, SYCLDevice>>::convert_to(v.scl, h);
    auto rhs = Converter<Evaluator<RHS, SYCLDevice>>::convert_to(v.r, h);
    return type(scl, rhs);
  }
};

/*! Converter<TupleOp<Operator, RHS>
 * @brief See Converter.
 */
template <typename RHS>
struct Converter<Evaluator<TupleExpr<RHS>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS, SYCLDevice>::value_type;
  using rhs_type = typename Converter<Evaluator<RHS, SYCLDevice>>::type;
  using input_type = Evaluator<TupleExpr<RHS>, SYCLDevice>;
  using type = Evaluator<TupleExpr<rhs_type>, SYCLDevice>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs = Converter<Evaluator<RHS, SYCLDevice>>::convert_to(v.r, h);
    return type(rhs);
  }
};

/*! Converter<UnaryOp<Operator, RHS>
 * @brief See Converter.
 */
template <typename Functor, typename RHS>
struct Converter<Evaluator<UnaryExpr<Functor, RHS>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS, SYCLDevice>::value_type;
  using rhs_type = typename Converter<Evaluator<RHS, SYCLDevice>>::type;
  using input_type = Evaluator<UnaryExpr<Functor, RHS>, SYCLDevice>;
  using type = Evaluator<UnaryExpr<Functor, rhs_type>, SYCLDevice>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs = Converter<Evaluator<RHS, SYCLDevice>>::convert_to(v.r, h);
    return type(rhs);
  }
};

/*! Converter<BinaryOp<Operator, LHS, RHS>>
 * @brief See Converter.
 */
template <typename Functor, typename LHS, typename RHS>
struct Converter<Evaluator<BinaryExpr<Functor, LHS, RHS>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS, SYCLDevice>::value_type;
  using lhs_type = typename Converter<Evaluator<LHS, SYCLDevice>>::type;
  using rhs_type = typename Converter<Evaluator<RHS, SYCLDevice>>::type;
  using input_type = Evaluator<BinaryExpr<Functor, LHS, RHS>, SYCLDevice>;
  using type = Evaluator<BinaryExpr<Functor, lhs_type, rhs_type>, SYCLDevice>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Converter<Evaluator<LHS, SYCLDevice>>::convert_to(v.l, h);
    auto rhs = Converter<Evaluator<RHS, SYCLDevice>>::convert_to(v.r, h);
    return type(lhs, rhs);
  }
};

/*! Converter<AssignReduction<Operator, LHS, RHS>>
 * @brief See Converter.
 */
template <typename Functor, typename LHS, typename RHS>
struct Converter<
    Evaluator<AssignReductionExpr<Functor, LHS, RHS>, SYCLDevice>> {
  using value_type = typename LHS::value_type;
  using oper_type = Functor;
  using LHS_type = Evaluator<LHS, SYCLDevice>;
  using cont_type = typename LHS::ContainerT;  // out of row
  using lhs_type = typename Converter<Evaluator<LHS, SYCLDevice>>::type;
  using rhs_type = typename Converter<Evaluator<RHS, SYCLDevice>>::type;
  using input_type = Evaluator<AssignReductionExpr<Functor, LHS, RHS>, SYCLDevice>;
  using type = Evaluator<AssignReductionExpr<Functor, lhs_type, rhs_type>, SYCLDevice>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Converter<Evaluator<LHS, SYCLDevice>>::convert_to(v.l, h);
    auto rhs = Converter<Evaluator<RHS, SYCLDevice>>::convert_to(v.r, h);
    return type(lhs, rhs, v.blqS, v.grdS);
  }
};

/*! Converter<vector_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>>
 * @brief See Converter.
 */
template <typename ScalarT>
struct Converter<Evaluator<vector_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>, SYCLDevice>> {
  using value_type = ScalarT;
  using cont_type = cl::sycl::buffer<ScalarT, 1>;
  using input_type = Evaluator<vector_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>, SYCLDevice>;
  using nested_type = cl::sycl::buffer<ScalarT, 1>;
  using type = Evaluator<vector_view<ScalarT, cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>>, SYCLDevice>;

  static type convert_to(input_type t, cl::sycl::handler &h) {
    auto nested = cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>(t.data_, h);
    return type(nested, t.disp_, t.strd_, t.size_);
  }
};

/*! Converter<matrix_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>>
 * @brief See Converter.
 */
template <typename ScalarT>
struct Converter<Evaluator<matrix_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>, SYCLDevice>> {
  using value_type = ScalarT;
  using cont_type = cl::sycl::buffer<ScalarT, 1>;
  using input_type = Evaluator<matrix_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>, SYCLDevice>;
  using nested_type = cl::sycl::buffer<ScalarT, 1>;
  using type = Evaluator<matrix_view<ScalarT, cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>>, SYCLDevice>;

  static type convert_to(input_type t, cl::sycl::handler &h) {
    auto nested = cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>(t.data_, h);
    return type(nested, t.accessDev_, t.sizeR_, t.sizeC_, t.accessOpr_, t.sizeL_, t.disp_);
  }
};

/*! Converter<matrix_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>>
 * @brief See Converter.
 */
template <typename ScalarT, typename ContainerT>
struct Converter<Evaluator<vector_view<ScalarT, ContainerT>, SYCLDevice>> {
  using value_type = ScalarT;
  using cont_type = ContainerT;
  using input_type = Evaluator<vector_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>, SYCLDevice>;
  using nested_type = ContainerT;
  using type = Evaluator<vector_view<ScalarT, nested_type>, SYCLDevice>;

  static type convert_to(input_type t, cl::sycl::handler &h) = delete;
};

/*! Converter<matrix_view<ScalarT, ContainerT>>
 * @brief See Converter.
 */
template <typename ScalarT, typename ContainerT>
struct Converter<Evaluator<matrix_view<ScalarT, ContainerT>, SYCLDevice>> {
  using value_type = ScalarT;
  using cont_type = ContainerT;
  using input_type = Evaluator<matrix_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>, SYCLDevice>;
  using nested_type = ContainerT;
  using type = Evaluator<matrix_view<ScalarT, nested_type>, SYCLDevice>;

  static type convert_to(input_type t, cl::sycl::handler &h) = delete;
};

/** make_accessor.
 * Triggers the conversion of the Expression Tree into the Evaluation
 * tree by calling the convert_to method.
 * @param Tree The Input Expression Tree.
 * @param handler The Command Group Handler used to create the accessors
 */
template <typename Evaluator>
auto make_accessor(Evaluator e, cl::sycl::handler &h) -> typename Converter<Evaluator>::type {
  return Converter<Evaluator>::convert_to(e, h);
}

}  // namespace blas

#endif  // BLAS1_TREE_EXECUTOR_HPP
