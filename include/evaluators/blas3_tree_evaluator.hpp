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
 *  @filename blas3_tree_evaluator.hpp
 *
 **************************************************************************/

#ifndef BLAS3_TREE_EVALUATOR_HPP_TQIIPJVC
#define BLAS3_TREE_EVALUATOR_HPP_TQIIPJVC

#include <stdexcept>
#include <vector>

#include <evaluators/blas_tree_evaluator_base.hpp>
#include <executors/blas_packet_traits_sycl.hpp>
#include <operations/blas3_trees.hpp>
#include <operations/blas_operators.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/*! PrdRowMatColMat.
 * @brief CLASSICAL DOT PRODUCT GEMM
 * Each thread computes a dot product
 * If the matrix is column-major the accesses are coalescent.
*/
template <class RHS1, class RHS2, typename Device>
struct Evaluator<PrdRowMatColMatExpr<RHS1, RHS2>, Device> {
  using Expression = PrdRowMatColMatExpr<RHS1, RHS2>;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<RHS1, Device>::cont_type;
  /* static constexpr bool supported = RHS1::supported && RHS2::supported; */

  Evaluator<RHS1, Device> r1;
  Evaluator<RHS2, Device> r2;

  Evaluator(Expression &expr)
      : r1(Evaluator<RHS1, Device>(expr.r1)),
        r2(Evaluator<RHS2, Device>(expr.r2)) {}

  size_t getSize() const { return r1.getSize(); }
  cont_type *data() { return r1.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    r1.eval_subexpr_if_needed(nullptr, dev);
    r2.eval_subexpr_if_needed(nullptr, dev);
    return true;
  }

  value_type eval(size_t k) {
    auto dim1 = (r2.getAccessOpr()) ? r2.getSizeR() : r2.getSizeC();
    auto dim2 = (r2.getAccessOpr()) ? r2.getSizeC() : r2.getSizeR();
    auto row = (r2.getAccess()) ? (k / dim2) : (k % dim2);
    auto col = (r2.getAccess()) ? (k % dim2) : (k / dim2);

    auto val =
        functor_traits<iniAddOp1_struct, value_type, Device>::eval(r1.eval(0));
    for (size_t j = 0; j < dim1; j++) {
      val += r1.eval(row, j) * r2.eval(j, col);
    }
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

  void cleanup(Device &dev) {
    r1.cleanup(dev);
    r2.cleanup(dev);
  }
};

}  // namespace blas

#endif  // BLAS3_TREES_HPP
