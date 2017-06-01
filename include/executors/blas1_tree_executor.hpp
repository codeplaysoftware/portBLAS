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

#include <executors/executor_base.hpp>
#include <operations/blas1_trees.hpp>
#include <operations/blas2_trees.hpp>
#include <operations/blas3_trees.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/*! Evaluate
 * @brief Evaluate the expression tree passed, converting node types.
 * This set of template specializations is used to convert the buffer
 * types of the expression tree into accessors, suitable to be used on
 * the kernel scope.
 * When using the expression tree on the device, developers call the
 * make_accessor function, which starts processing the tree.
 */
template <typename Tree>
struct Evaluate {
  using value_type = Tree;
  using input_type = Tree;
  using type = Tree;

  /** convert_to.
   * @brief .
   */
  static type convert_to(input_type v, cl::sycl::handler &h) { return v; }
};

/*! Evaluate <Join<LHS, RHS>>.
 * @brief See Evaluate.
 */
template <typename LHS, typename RHS>
struct Evaluate<Join<LHS, RHS>> {
  using value_type = typename RHS::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs_type = typename Evaluate<RHS>::type;
  using input_type = Join<LHS, RHS>;
  using type = Join<lhs_type, rhs_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs = Evaluate<RHS>::convert_to(v.r, h);
    return type(lhs, rhs);
  }
};

/*! Evaluate <Assign<LHS, RHS>>a
 * @brief See Evaluate.
 */
template <typename LHS, typename RHS>
struct Evaluate<Assign<LHS, RHS>> {
  using value_type = typename RHS::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs_type = typename Evaluate<RHS>::type;
  using input_type = Assign<LHS, RHS>;
  using type = Assign<lhs_type, rhs_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs = Evaluate<RHS>::convert_to(v.r, h);
    return type(lhs, rhs);
  }
};

/*! Evaluate <DoubleAssign<LHS, RHS>>
 * @brief See Evaluate.
 */
template <class LHS1, class LHS2, class RHS1, class RHS2>
struct Evaluate<DobleAssign<LHS1, LHS2, RHS1, RHS2>> {
  using lhs1_type = typename Evaluate<LHS1>::type;
  using lhs2_type = typename Evaluate<LHS2>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using input_type = DobleAssign<LHS1, LHS2, RHS1, RHS2>;
  using type = DobleAssign<lhs1_type, lhs2_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs1 = Evaluate<LHS1>::convert_to(v.l1, h);
    auto lhs2 = Evaluate<LHS2>::convert_to(v.l2, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs1, lhs2, rhs1, rhs2);
  }
};

/*! Evaluate<ScalarOp<Operator, SCL, RHS>>
 * @brief See Evaluate.
 */
template <typename Operator, typename SCL, typename RHS>
struct Evaluate<ScalarOp<Operator, SCL, RHS>> {
  using value_type = typename RHS::value_type;
  using scl_type = typename Evaluate<SCL>::type;
  using rhs_type = typename Evaluate<RHS>::type;
  using input_type = ScalarOp<Operator, SCL, RHS>;
  using type = ScalarOp<Operator, scl_type, rhs_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto scl = Evaluate<SCL>::convert_to(v.scl, h);
    auto rhs = Evaluate<RHS>::convert_to(v.r, h);
    return type(scl, rhs);
  }
};

/*! Evaluate<TupleOp<Operator, RHS>
 * @brief See Evaluate.
 */
template <typename RHS>
struct Evaluate<TupleOp<RHS>> {
  using value_type = typename RHS::value_type;
  using rhs_type = typename Evaluate<RHS>::type;
  using input_type = TupleOp<RHS>;
  using type = TupleOp<rhs_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs = Evaluate<RHS>::convert_to(v.r, h);
    return type(rhs);
  }
};

/*! Evaluate<UnaryOp<Operator, RHS>
 * @brief See Evaluate.
 */
template <typename Operator, typename RHS>
struct Evaluate<UnaryOp<Operator, RHS>> {
  using value_type = typename RHS::value_type;
  using rhs_type = typename Evaluate<RHS>::type;
  using input_type = UnaryOp<Operator, RHS>;
  using type = UnaryOp<Operator, rhs_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs = Evaluate<RHS>::convert_to(v.r, h);
    return type(rhs);
  }
};

/*! Evaluate<BinaryOp<Operator, LHS, RHS>>
 * @brief See Evaluate.
 */
template <typename Operator, typename LHS, typename RHS>
struct Evaluate<BinaryOp<Operator, LHS, RHS>> {
  using value_type = typename RHS::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs_type = typename Evaluate<RHS>::type;
  using input_type = BinaryOp<Operator, LHS, RHS>;
  using type = BinaryOp<Operator, lhs_type, rhs_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs = Evaluate<RHS>::convert_to(v.r, h);
    return type(lhs, rhs);
  }
};

/*! Evaluate<AssignReduction<Operator, LHS, RHS>>
 * @brief See Evaluate.
 */
template <typename Operator, typename LHS, typename RHS>
struct Evaluate<AssignReduction<Operator, LHS, RHS>> {
  using value_type = typename LHS::value_type;
  using oper_type = Operator;
  using LHS_type = LHS;
  using cont_type = typename LHS::ContainerT;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs_type = typename Evaluate<RHS>::type;
  using input_type = AssignReduction<Operator, LHS, RHS>;
  using type = AssignReduction<Operator, lhs_type, rhs_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs = Evaluate<RHS>::convert_to(v.r, h);
    return type(lhs, rhs, v.blqS, v.grdS);
  }
};

/*! Evaluate<vector_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>>
 * @brief See Evaluate.
 */
template <typename ScalarT>
struct Evaluate<vector_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>> {
  using value_type = ScalarT;
  using cont_type = cl::sycl::buffer<ScalarT, 1>;
  using input_type = vector_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>;
  using nested_type = cl::sycl::buffer<ScalarT, 1>;
  using type = vector_view<
      ScalarT,
      cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>>;

  static type convert_to(input_type t, cl::sycl::handler &h) {
    auto nested =
        cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::global_buffer>(t.data_, h);
    return type(nested, t.disp_, t.strd_, t.size_);
  }
};

/*! Evaluate<matrix_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>>
 * @brief See Evaluate.
 */
template <typename ScalarT>
struct Evaluate<matrix_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>> {
  using value_type = ScalarT;
  using cont_type = cl::sycl::buffer<ScalarT, 1>;
  using input_type = matrix_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>;
  using nested_type = cl::sycl::buffer<ScalarT, 1>;
  using type = matrix_view<
      ScalarT,
      cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>>;

  static type convert_to(input_type t, cl::sycl::handler &h) {
    auto nested =
        cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::global_buffer>(t.data_, h);
    return type(nested, t.accessDev_, t.sizeR_, t.sizeC_, t.accessOpr_,
                t.sizeL_, t.disp_);
  }
};

/*! Evaluate<matrix_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>>
 * @brief See Evaluate.
 */
template <typename ScalarT, typename ContainerT>
struct Evaluate<vector_view<ScalarT, ContainerT>> {
  using value_type = ScalarT;
  using cont_type = ContainerT;
  using input_type = vector_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>;
  using nested_type = ContainerT;
  using type = vector_view<ScalarT, nested_type>;

  static type convert_to(input_type t, cl::sycl::handler &h) = delete;
};

/*! Evaluate<matrix_view<ScalarT, ContainerT>>
 * @brief See Evaluate.
 */
template <typename ScalarT, typename ContainerT>
struct Evaluate<matrix_view<ScalarT, ContainerT>> {
  using value_type = ScalarT;
  using cont_type = ContainerT;
  using input_type = matrix_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>;
  using nested_type = ContainerT;
  using type = matrix_view<ScalarT, nested_type>;

  static type convert_to(input_type t, cl::sycl::handler &h) = delete;
};

/** make_accessor.
 * Triggers the conversion of the Expression Tree into the Evaluation
 * tree by calling the convert_to method.
 * @param Tree The Input Expression Tree.
 * @param handler The Command Group Handler used to create the accessors
 */
template <typename Tree>
auto make_accessor(Tree t, cl::sycl::handler &h) ->
    typename Evaluate<Tree>::type {
  return Evaluate<Tree>::convert_to(t, h);
}

}  // namespace blas

#endif  // BLAS1_TREE_EXECUTOR_HPP
