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
 *  @filename blas3_tree_executor.hpp
 *
 **************************************************************************/

#ifndef BLAS3_TREE_EXECUTOR_HPP
#define BLAS3_TREE_EXECUTOR_HPP

#include <stdexcept>

#include <CL/sycl.hpp>

#include <executors/executor_base.hpp>
#include <operations/blas1_trees.hpp>
#include <operations/blas2_trees.hpp>
#include <operations/blas3_trees.hpp>
#include <views/view_sycl.hpp>

namespace blas {

template <typename Tree>
struct Evaluate;

/********************************/
/*            BLAS 3            */
/********************************/

template <typename RHS1, typename RHS2>
struct Evaluate<PrdRowMatColMat<RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using input_type = PrdRowMatColMat<RHS1, RHS2>;
  using type = PrdRowMatColMat<rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler& h) {
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(rhs1, rhs2);
  }
};

}  // namespace blas

#endif  // BLAS3_TREE_EXECUTOR_HPP
