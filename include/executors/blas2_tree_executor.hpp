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

#include <evaluators/blas1_tree_evaluator.hpp>
#include <evaluators/blas2_tree_evaluator.hpp>
#include <evaluators/blas3_tree_evaluator.hpp>
#include <executors/blas_device_sycl.hpp>
#include <executors/executor_base.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/** Converter.
 */
template <typename Tree>
struct Converter;

/**** CLASSICAL DOT PRODUCT GEMV ****/
/*! Converter<PrdRowMatVct>.
 * @brief See Converter.
 */
template <typename RHS1, typename RHS2>
struct Converter<Evaluator<PrdRowMatVctExpr<RHS1, RHS2>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS2, SYCLDevice>::value_type;
  using rhs1_type = typename Converter<Evaluator<RHS1, SYCLDevice>>::type;
  using rhs2_type = typename Converter<Evaluator<RHS2, SYCLDevice>>::type;
  using input_type = Evaluator<PrdRowMatVctExpr<RHS1, RHS2>, SYCLDevice>;
  using type = Evaluator<PrdRowMatVctExpr<rhs1_type, rhs2_type>, SYCLDevice>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs1 = Converter<Evaluator<RHS1, SYCLDevice>>::convert_to(v.r1, h);
    auto rhs2 = Converter<Evaluator<RHS2, SYCLDevice>>::convert_to(v.r2, h);
    return type(rhs1, rhs2);
  }
};

/*! Converter<PrdRowMatVctMult>.
 * @brief See Converter.
 */
template <typename LHS, typename RHS1, typename RHS2, typename RHS3>
struct Converter<
    Evaluator<PrdRowMatVctMultExpr<LHS, RHS1, RHS2, RHS3>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS2, SYCLDevice>::value_type;
  using lhs_type = typename Converter<Evaluator<LHS, SYCLDevice>>::type;
  using rhs1_type = typename Converter<Evaluator<RHS1, SYCLDevice>>::type;
  using rhs2_type = typename Converter<Evaluator<RHS2, SYCLDevice>>::type;
  using rhs3_type = typename Converter<Evaluator<RHS3, SYCLDevice>>::type;
  using cont_type = typename Converter<Evaluator<LHS, SYCLDevice>>::cont_type;
  using input_type =
      Evaluator<PrdRowMatVctMultExpr<LHS, RHS1, RHS2, RHS3>, SYCLDevice>;
  using type =
      Evaluator<PrdRowMatVctMultExpr<lhs_type, rhs1_type, rhs2_type, rhs3_type>,
                SYCLDevice>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Converter<Evaluator<LHS, SYCLDevice>>::convert_to(v.l, h);
    auto rhs1 = Converter<Evaluator<RHS1, SYCLDevice>>::convert_to(v.r1, h);
    auto rhs2 = Converter<Evaluator<RHS2, SYCLDevice>>::convert_to(v.r2, h);
    auto rhs3 = Converter<Evaluator<RHS3, SYCLDevice>>::convert_to(v.r3, h);
    return type(lhs, v.scl, rhs1, rhs2, rhs3, v.nThr);
  }
};

/*! Converter<PrdRowMatVctMultShm>.
 * @brief See Converter.
 */
template <typename LHS, typename RHS1, typename RHS2>
struct Converter<
    Evaluator<PrdRowMatVctMultShmExpr<LHS, RHS1, RHS2>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS2, SYCLDevice>::value_type;
  using lhs_type = typename Converter<Evaluator<LHS, SYCLDevice>>::type;
  using rhs1_type = typename Converter<Evaluator<RHS1, SYCLDevice>>::type;
  using rhs2_type = typename Converter<Evaluator<RHS2, SYCLDevice>>::type;
  using cont_type = typename Converter<Evaluator<LHS, SYCLDevice>>::cont_type;
  using input_type =
      Evaluator<PrdRowMatVctMultShmExpr<LHS, RHS1, RHS2>, SYCLDevice>;
  using type =
      Evaluator<PrdRowMatVctMultShmExpr<lhs_type, rhs1_type, rhs2_type>,
                SYCLDevice>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Converter<Evaluator<LHS, SYCLDevice>>::convert_to(v.l, h);
    auto rhs1 = Converter<Evaluator<RHS1, SYCLDevice>>::convert_to(v.r1, h);
    auto rhs2 = Converter<Evaluator<RHS2, SYCLDevice>>::convert_to(v.r2, h);
    return type(lhs, rhs1, rhs2, v.nThr);
  }
};

/*! Converter<AddPrdRowMatVctMultShm>.
 * @brief See Converter.
 */
template <typename LHS, typename RHS1, typename RHS2>
struct Converter<
    Evaluator<AddPrdRowMatVctMultShmExpr<LHS, RHS1, RHS2>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS2, SYCLDevice>::value_type;
  using lhs_type = typename Converter<Evaluator<LHS, SYCLDevice>>::type;
  using rhs1_type = typename Converter<Evaluator<RHS1, SYCLDevice>>::type;
  using rhs2_type = typename Converter<Evaluator<RHS2, SYCLDevice>>::type;
  using cont_type = typename Converter<Evaluator<LHS, SYCLDevice>>::cont_type;
  using input_type =
      Evaluator<AddPrdRowMatVctMultShmExpr<LHS, RHS1, RHS2>, SYCLDevice>;
  using type =
      Evaluator<AddPrdRowMatVctMultShmExpr<lhs_type, rhs1_type, rhs2_type>,
                SYCLDevice>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto lhs = Converter<Evaluator<LHS, SYCLDevice>>::convert_to(v.l, h);
    auto rhs1 = Converter<Evaluator<RHS1, SYCLDevice>>::convert_to(v.r1, h);
    auto rhs2 = Converter<Evaluator<RHS2, SYCLDevice>>::convert_to(v.r2, h);
    return type(lhs, v.scl, rhs1, rhs2);
  }
};

/*! Converter<RedRowMatVct>.
 * @brief See Converter.
 */
template <typename RHS1, typename RHS2>
struct Converter<Evaluator<RedRowMatVctExpr<RHS1, RHS2>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS2, SYCLDevice>::value_type;
  using rhs1_type = typename Converter<Evaluator<RHS1, SYCLDevice>>::type;
  using rhs2_type = typename Converter<Evaluator<RHS2, SYCLDevice>>::type;
  using input_type = Evaluator<RedRowMatVctExpr<RHS1, RHS2>, SYCLDevice>;
  using type = Evaluator<RedRowMatVctExpr<rhs1_type, rhs2_type>, SYCLDevice>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs1 = Converter<Evaluator<RHS1, SYCLDevice>>::convert_to(v.r1, h);
    auto rhs2 = Converter<Evaluator<RHS2, SYCLDevice>>::convert_to(v.r2, h);
    return type(rhs1, rhs2, v.warpSize);
  }
};

/*! Converter<ModifRank1>
 * @brief See Converter.
 */
template <typename RHS1, typename RHS2, typename RHS3>
struct Converter<Evaluator<ModifRank1Expr<RHS1, RHS2, RHS3>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS2, SYCLDevice>::value_type;
  using rhs1_type = typename Converter<Evaluator<RHS1, SYCLDevice>>::type;
  using rhs2_type = typename Converter<Evaluator<RHS2, SYCLDevice>>::type;
  using rhs3_type = typename Converter<Evaluator<RHS3, SYCLDevice>>::type;
  using input_type = Evaluator<ModifRank1Expr<RHS1, RHS2, RHS3>, SYCLDevice>;
  using type =
      Evaluator<ModifRank1Expr<rhs1_type, rhs2_type, rhs3_type>, SYCLDevice>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs1 = Converter<Evaluator<RHS1, SYCLDevice>>::convert_to(v.r1, h);
    auto rhs2 = Converter<Evaluator<RHS2, SYCLDevice>>::convert_to(v.r2, h);
    auto rhs3 = Converter<Evaluator<RHS2, SYCLDevice>>::convert_to(v.r3, h);
    return type(rhs1, rhs2, rhs3);
  }
};

}  // namespace blas

#endif  // BLAS2_TREE_EXECUTOR_HPP
