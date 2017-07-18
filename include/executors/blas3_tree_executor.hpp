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

#include <evaluators/blas3_tree_evaluator.hpp>
#include <executors/blas_device_sycl.hpp>
#include <views/view_sycl.hpp>

namespace blas {

template <typename Tree>
struct Converter;

/********************************/
/*            BLAS 3            */
/********************************/

template <typename RHS1, typename RHS2>
struct Converter<Evaluator<PrdRowMatColMatExpr<RHS1, RHS2>, SYCLDevice>> {
  using value_type = typename Evaluator<RHS2, SYCLDevice>::value_type;
  using rhs1_type = typename Converter<Evaluator<RHS1, SYCLDevice>>::out_type;
  using rhs2_type = typename Converter<Evaluator<RHS2, SYCLDevice>>::out_type;
  using input_type = Evaluator<PrdRowMatColMatExpr<RHS1, RHS2>, SYCLDevice>;
  using out_type = PrdRowMatColMatExpr<rhs1_type, rhs2_type>;

  static out_type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs1 = Converter<Evaluator<RHS1, SYCLDevice>>::convert_to(v.r1, h);
    auto rhs2 = Converter<Evaluator<RHS2, SYCLDevice>>::convert_to(v.r2, h);
    return out_type(rhs1, rhs2);
  }
};

}  // namespace blas

#endif  // BLAS3_TREE_EXECUTOR_HPP
