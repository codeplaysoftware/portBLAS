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
 *  @filename executor_base.hpp
 *
 **************************************************************************/

#ifndef EXECUTOR_BASE_HPP
#define EXECUTOR_BASE_HPP

namespace blas {

/** Executor.
 * @brief Primary template for the Executor specializations.
 * The Executor represents the object that executes a tree on
 * a specific backend.
 * Executors have state, and they must be instantiated
 * before using them.
 * Only one method is mandatory, the Execute method.
 */
template <class ExecutionPolicy>
class Executor {
 public:
  template <typename Tree>
  void execute(Tree t) = delete;
};

class Sequential {};
class Parallel {};
class SYCL {};

/*! Executor<Sequential>.
 * @brief Template specialization for Sequential Execution.
 * @bug Expression trees are not tested with this Executor.
 */
template <>
class Executor<Sequential> {
 public:
  template <typename Tree>
  void execute(Tree t) {
    size_t _N = t.get_size();
    for (size_t i = 0; i < _N; i++) {
      t.eval(i);
    }
  };
};

/*! Executor<Parallel>.
 * @brief Template specialization for Parallel Execution.
 * @bug Expression trees are not tested with this Executor.
 */
template <>
class Executor<Parallel> {
 public:
  template <typename Tree>
  void execute(Tree t) {
    size_t _N = t.get_size();
#pragma omp parallel for
    for (size_t i = 0; i < _N; i++) {
      t.eval(i);
    }
  };
};

}  // namespace blas

#endif  // EXECUTOR_BASE_HPP
