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
 *  @filename blas2_tree_executor.hpp
 *
 **************************************************************************/

#ifndef BLAS2_TREE_EXECUTOR_HPP
#define BLAS2_TREE_EXECUTOR_HPP

#include <stdexcept>

#include <CL/sycl.hpp>

#include <views/view_sycl.hpp>
#include <executors/executor_base.hpp>
#include <operations/blas1_trees.hpp>
#include <operations/blas2_trees.hpp>
#include <operations/blas3_trees.hpp>

namespace blas {

/** Evaluate.
 */
template <typename Tree>
struct Evaluate;

/**** CLASSICAL DOT PRODUCT GEMV ****/
/*! Evaluate<PrdRowMatVct>.
 * @brief See Evaluate.
 */
template <typename RHS1, typename RHS2>
struct Evaluate<PrdRowMatVct<RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using input_type = PrdRowMatVct<RHS1, RHS2>;
  using type = PrdRowMatVct<rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(rhs1, rhs2);
  }
};

/*! Evaluate<PrdRowMatVctMult>.
 * @brief See Evaluate.
 */
template <typename LHS, typename RHS1, typename RHS2, typename RHS3>
struct Evaluate<PrdRowMatVctMult<LHS, RHS1, RHS2, RHS3>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using rhs3_type = typename Evaluate<RHS3>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = PrdRowMatVctMult<LHS, RHS1, RHS2, RHS3>;
  using type = PrdRowMatVctMult<lhs_type, rhs1_type, rhs2_type, rhs3_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    auto rhs3 = Evaluate<RHS3>::convert_to(v.r3, h);
    return type(lhs, v.scl, rhs1, rhs2, rhs3, v.nThr);
  }
};

/*! Evaluate<PrdRowMatVctMultShm>.
 * @brief See Evaluate.
 */
template <typename LHS, typename RHS1, typename RHS2>
struct Evaluate<PrdRowMatVctMultShm<LHS, RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = PrdRowMatVctMultShm<LHS, RHS1, RHS2>;
  using type = PrdRowMatVctMultShm<lhs_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs, rhs1, rhs2, v.nThr);
  }
};

/*! Evaluate<AddPrdRowMatVctMultShm>.
 * @brief See Evaluate.
 */
template <typename LHS, typename RHS1, typename RHS2>
struct Evaluate<AddPrdRowMatVctMultShm<LHS, RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using lhs_type = typename Evaluate<LHS>::type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using cont_type = typename Evaluate<LHS>::cont_type;
  using input_type = AddPrdRowMatVctMultShm<LHS, RHS1, RHS2>;
  using type = AddPrdRowMatVctMultShm<lhs_type, rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Evaluate<LHS>::convert_to(v.l, h);
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(lhs, v.scl, rhs1, rhs2);
  }
};

/*! Evaluate<RedRowMatVct>.
 * @brief See Evaluate.
 */
template <typename RHS1, typename RHS2>
struct Evaluate<RedRowMatVct<RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using input_type = RedRowMatVct<RHS1, RHS2>;
  using type = RedRowMatVct<rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(rhs1, rhs2, v.warpSize);
  }
};

/*! Evaluate<ModifRank1>
 * @brief See Evaluate.
 */
template <typename RHS1, typename RHS2, typename RHS3>
struct Evaluate<ModifRank1<RHS1, RHS2, RHS3>> {
  using value_type = typename RHS2::value_type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using rhs3_type = typename Evaluate<RHS3>::type;
  using input_type = ModifRank1<RHS1, RHS2, RHS3>;
  using type = ModifRank1<rhs1_type, rhs2_type, rhs3_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    auto rhs3 = Evaluate<RHS2>::convert_to(v.r3, h);
    return type(rhs1, rhs2, rhs3);
  }
};

}  // namespace blas

#endif  // BLAS2_TREE_EXECUTOR_HPP
