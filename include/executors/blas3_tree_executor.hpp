/***************************************************************************
 *
 *  @license
 *  Copyright (C) Codeplay Software Limited
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

template <typename RHS1, typename RHS2, bool DoubleBuffer, bool NbcA, bool NbcB,
          int ClSize, typename TileType, bool TransA, bool TransB, typename T>
struct Evaluate<GemmFactory<RHS1, RHS2, DoubleBuffer, NbcA, NbcB, ClSize,
                            TileType, TransA, TransB, T>> {
  using value_type = typename RHS1::value_type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using input_type = GemmFactory<RHS1, RHS2, DoubleBuffer, NbcA, NbcB, ClSize,
                                 TileType, TransA, TransB, T>;
  using type = GemmFactory<rhs1_type, rhs2_type, DoubleBuffer, NbcA, NbcB,
                           ClSize, TileType, TransA, TransB, T>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs1 = Evaluate<RHS1>::convert_to(v._A, h);
    auto rhs2 = Evaluate<RHS1>::convert_to(v._B, h);
    auto rhs3 = Evaluate<RHS2>::convert_to(v._C, h);
    return type(rhs1, rhs2, rhs3, v.alpha, v.beta);
  }
};
template <typename RHS1, typename RHS2, int WgSize, bool TransA, bool TransB,
          typename T>
struct Evaluate<ReferenceGemmFactory<RHS1, RHS2, WgSize, TransA, TransB, T>> {
  using value_type = typename RHS1::value_type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using input_type =
      ReferenceGemmFactory<RHS1, RHS2, WgSize, TransA, TransB, T>;
  using type =
      ReferenceGemmFactory<rhs1_type, rhs2_type, WgSize, TransA, TransB, T>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs1 = Evaluate<RHS1>::convert_to(v._A, h);
    auto rhs2 = Evaluate<RHS1>::convert_to(v._B, h);
    auto rhs3 = Evaluate<RHS2>::convert_to(v._C, h);
    return type(rhs1, rhs2, rhs3, v.alpha, v.beta);
  }
};

}  // namespace blas

#endif  // BLAS3_TREE_EXECUTOR_HPP
