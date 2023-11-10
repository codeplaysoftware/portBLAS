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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename portblas_handle.hpp
 *
 **************************************************************************/

#ifndef PORTBLAS_HANDLE_HPP
#define PORTBLAS_HANDLE_HPP

#include <algorithm>

#include "blas_meta.h"
#include "operations/blas1_trees.hpp"
#include "operations/blas2_trees.hpp"
#include "operations/blas_operators.hpp"
#include "portblas_helper.h"
#include "sb_handle/kernel_constructor.h"
#include "sb_handle/portblas_handle.h"
#include "sb_handle/temp_memory_pool.hpp"
#include "views/view.h"
namespace blas {

template <helper::AllocType alloc, typename value_t>
typename std::enable_if<
    alloc == helper::AllocType::buffer,
    typename helper::AllocHelper<value_t, alloc>::type>::type
SB_Handle::acquire_temp_mem(size_t size) {
  if (tempMemPool_ != NULL)
    return tempMemPool_->acquire_buff_mem<value_t>(size);
  else
    return make_sycl_iterator_buffer<value_t>(size);
}

template <typename container_t>
typename std::enable_if<
    std::is_same<container_t, typename helper::AllocHelper<
                                  typename ValueType<container_t>::type,
                                  helper::AllocType::buffer>::type>::value,
    typename SB_Handle::event_t>::type
SB_Handle::release_temp_mem(const typename SB_Handle::event_t& dependencies,
                            const container_t& mem) {
  if (tempMemPool_ != NULL)
    return tempMemPool_->release_buff_mem(dependencies, mem);
  else
    return {};
}

#ifdef SB_ENABLE_USM
template <helper::AllocType alloc, typename value_t>
typename std::enable_if<
    alloc == helper::AllocType::usm,
    typename helper::AllocHelper<value_t, alloc>::type>::type
SB_Handle::acquire_temp_mem(size_t size) {
  if (tempMemPool_ != NULL)
    return tempMemPool_->acquire_usm_mem<value_t>(size);
  else
    return cl::sycl::malloc_device<value_t>(size, q_);
}

template <typename container_t>
typename std::enable_if<
    std::is_same<container_t, typename helper::AllocHelper<
                                  typename ValueType<container_t>::type,
                                  helper::AllocType::usm>::type>::value,
    typename SB_Handle::event_t>::type
SB_Handle::release_temp_mem(const typename SB_Handle::event_t& dependencies,
                            const container_t& mem) {
  if (tempMemPool_ != NULL)
    return tempMemPool_->release_usm_mem(dependencies, mem);
  else {
    cl::sycl::context context = q_.get_context();
    return {q_.submit([&](cl::sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.host_task([=]() { cl::sycl::free(mem, context); });
    })};
  }
}
#endif

/*!
 * @brief Executes the tree without defining required shared memory.
 */
template <typename expression_tree_t>
inline typename SB_Handle::event_t SB_Handle::execute(
    expression_tree_t t, const typename SB_Handle::event_t& dependencies) {
  const auto localSize = get_work_group_size();
  auto _N = t.get_size();
  auto nWG = (_N + localSize - 1) / localSize;
  auto globalSize = nWG * localSize;

  return {execute_tree<using_local_memory::disabled>(
      get_queue(), t, localSize, globalSize, 0, dependencies)};
};

/*!
 * @brief Executes the tree fixing the localSize but without defining
 * required shared memory.
 */
template <typename expression_tree_t, typename index_t>
inline typename SB_Handle::event_t SB_Handle::execute(
    expression_tree_t t, index_t localSize,
    const typename SB_Handle::event_t& dependencies) {
  auto _N = t.get_size();
  auto nWG = (_N + localSize - 1) / localSize;
  auto globalSize = nWG * localSize;
  return {execute_tree<using_local_memory::disabled>(
      q_, t, localSize, globalSize, 0, dependencies)};
};

/*!
 * @brief Executes the tree fixing the localSize but without defining
 * required shared memory.
 */
template <typename expression_tree_t, typename index_t>
inline typename SB_Handle::event_t SB_Handle::execute(
    expression_tree_t t, index_t localSize, index_t globalSize,
    const typename SB_Handle::event_t& dependencies) {
  return {execute_tree<using_local_memory::disabled>(
      q_, t, localSize, globalSize, 0, dependencies)};
}

/*!
 * @brief Executes the tree with specific local, global and shared
 * memory values.
 */
template <typename expression_tree_t, typename index_t>
inline typename SB_Handle::event_t SB_Handle::execute(
    expression_tree_t t, index_t localSize, index_t globalSize, index_t shMem,
    const typename SB_Handle::event_t& dependencies) {
  return {execute_tree<using_local_memory::enabled>(
      q_, t, localSize, globalSize, shMem, dependencies)};
}

/*!
 * @brief Applies a reduction to a tree.
 */
template <typename operator_t, typename lhs_t, typename rhs_t>
inline typename SB_Handle::event_t SB_Handle::execute(
    AssignReduction<operator_t, lhs_t, rhs_t> t,
    const typename SB_Handle::event_t& dependencies) {
  using expression_tree_t = AssignReduction<operator_t, lhs_t, rhs_t>;
  auto _N = t.get_size();
  auto localSize = t.local_num_thread_;
  // IF THERE ARE ENOUGH ELEMENTS, EACH BLOCK PROCESS TWO BLOCKS OF
  // ELEMENTS THEREFORE, 2*GLOBALSIZE ELEMENTS ARE PROCESSED IN A STEP
  // MOREOVER, A LOOP ALLOWS TO REPEAT THE PROCESS UNTIL
  // ALL THE ELEMENTS ARE PROCESSED
  auto nWG = (t.global_num_thread_ + (2 * localSize) - 1) / (2 * localSize);
  auto lhs = t.lhs_;
  auto rhs = t.rhs_;

  // Two accessors to local memory
  auto sharedSize = ((nWG < localSize) ? localSize : nWG);
  constexpr bool is_usm = std::is_pointer<typename lhs_t::container_t>::value;
  auto shMem1 = acquire_temp_mem < is_usm ? helper::AllocType::usm
                                          : helper::AllocType::buffer,
       typename lhs_t::value_t > (sharedSize);
  auto shMem2 = acquire_temp_mem < is_usm ? helper::AllocType::usm
                                          : helper::AllocType::buffer,
       typename lhs_t::value_t > (sharedSize);

  auto opShMem1 =
      make_vector_view(shMem1, typename lhs_t::increment_t(1), sharedSize);
  auto opShMem2 =
      make_vector_view(shMem2, typename lhs_t::increment_t(1), sharedSize);
  typename SB_Handle::event_t event;
  bool frst = true;
  bool even = false;
  do {
    auto globalSize = nWG * localSize;
    if (frst) {
      // THE FIRST CASE USES THE ORIGINAL BINARY/TERNARY FUNCTION
      auto localTree = expression_tree_t(((nWG == 1) ? lhs : opShMem1), rhs,
                                         localSize, globalSize);
      event.push_back(execute_tree<using_local_memory::enabled>(
          q_, localTree, localSize, globalSize, sharedSize, dependencies));
    } else {
      // THE OTHER CASES ALWAYS USE THE BINARY FUNCTION
      auto localTree = AssignReduction<operator_t, lhs_t, lhs_t>(
          ((nWG == 1) ? lhs : (even ? opShMem2 : opShMem1)),
          (even ? opShMem1 : opShMem2), localSize, globalSize);
      event.push_back(execute_tree<using_local_memory::enabled>(
          q_, localTree, localSize, globalSize, sharedSize, event));
    }
    _N = nWG;
    nWG = (_N + (2 * localSize) - 1) / (2 * localSize);
    frst = false;
    even = !even;
  } while (_N > 1);

  release_temp_mem({*event.rbegin()}, shMem1);

  release_temp_mem({*event.rbegin()}, shMem2);

  return event;
}

/*!
 * @brief Applies a reduction to a tree, receiving a scratch
 * BufferIterator.
 */
template <typename operator_t, typename lhs_t, typename rhs_t,
          typename local_memory_t>
inline typename SB_Handle::event_t SB_Handle::execute(
    AssignReduction<operator_t, lhs_t, rhs_t> t, local_memory_t scr,
    const typename SB_Handle::event_t& dependencies) {
  using expression_tree_t = AssignReduction<operator_t, lhs_t, rhs_t>;
  auto _N = t.get_size();
  auto localSize = t.local_num_thread_;
  // IF THERE ARE ENOUGH ELEMENTS, EACH BLOCK PROCESS TWO BLOCKS OF
  // ELEMENTS THEREFORE, 2*GLOBALSIZE ELEMENTS ARE PROCESSED IN A STEP
  // MOREOVER, A LOOP ALLOWS TO REPEAT THE PROCESS UNTIL
  // ALL THE ELEMENTS ARE PROCESSED
  auto nWG = (t.global_num_thread_ + (2 * localSize) - 1) / (2 * localSize);
  auto lhs = t.lhs_;
  auto rhs = t.rhs_;
  typename SB_Handle::event_t event;
  // Two accessors to local memory
  auto sharedSize = ((nWG < localSize) ? localSize : nWG);
  auto opShMem1 = lhs_t(scr, 1, sharedSize);
  auto opShMem2 = lhs_t(scr + sharedSize, 1, sharedSize);

  bool frst = true;
  bool even = false;
  do {
    auto globalSize = nWG * localSize;
    if (frst) {
      // THE FIRST CASE USES THE ORIGINAL BINARY/TERNARY FUNCTION
      auto localTree = expression_tree_t(((nWG == 1) ? lhs : opShMem1), rhs,
                                         localSize, globalSize);
      event.push_back(execute_tree<using_local_memory::enabled>(
          q_, localTree, localSize, globalSize, sharedSize, dependencies));
    } else {
      // THE OTHER CASES ALWAYS USE THE BINARY FUNCTION
      auto localTree = AssignReduction<operator_t, lhs_t, lhs_t>(
          ((nWG == 1) ? lhs : (even ? opShMem2 : opShMem1)),
          (even ? opShMem1 : opShMem2), localSize, globalSize);
      event.push_back(execute_tree<using_local_memory::enabled>(
          q_, localTree, localSize, globalSize, sharedSize, dependencies));
    }
    _N = nWG;
    nWG = (_N + (2 * localSize) - 1) / (2 * localSize);
    frst = false;
    even = !even;
  } while (_N > 1);
  return event;
}

template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          bool SymmA, bool SymmB, typename element_t, bool is_beta_zero,
          int GemmMemoryType, int GemmAlgorithm, int GemmVectorization,
          int VectorSize, int BatchType, bool UseJointMatrix>
inline typename SB_Handle::event_t SB_Handle::execute(
    Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
         TransB, SymmA, SymmB, element_t, is_beta_zero, GemmMemoryType,
         GemmAlgorithm, GemmVectorization, VectorSize, BatchType,
         UseJointMatrix>
        gemm_tree,
    const typename SB_Handle::event_t& dependencies) {
  using gemm_t = Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize,
                      tile_type, TransA, TransB, SymmA, SymmB, element_t,
                      is_beta_zero, GemmMemoryType, GemmAlgorithm,
                      GemmVectorization, VectorSize, BatchType, UseJointMatrix>;
  auto rng = gemm_tree.get_nd_range(SB_Handle::get_num_compute_units());
  return {execute_tree<
      Choose<GemmMemoryType == static_cast<int>(gemm_memory_t::local), int,
             using_local_memory::enabled, using_local_memory::disabled>::type>(
      q_, gemm_tree, rng.get_local_range()[0], rng.get_global_range()[0],
      gemm_t::local_memory_size, dependencies)};
}

/* Tall and skinny Gemm */
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          bool SymmA, bool SymmB, typename element_t, bool is_beta_zero,
          int GemmMemoryType, int GemmVectorization, int VectorSize,
          int BatchType>
inline typename SB_Handle::event_t SB_Handle::execute(
    Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
         TransB, SymmA, SymmB, element_t, is_beta_zero, GemmMemoryType,
         static_cast<int>(gemm_algorithm_t::tall_skinny), GemmVectorization,
         VectorSize, BatchType>
        gemm_wrapper,
    const typename SB_Handle::event_t& dependencies) {
  using index_t = typename std::make_signed<typename input_t::index_t>::type;

  const index_t rows = gemm_wrapper.m_;
  const index_t cols = gemm_wrapper.n_;
  const index_t ldc = gemm_wrapper.ldc_;

  /* Depth of the cube buffer */
  const index_t depth = GemmPartial<
      input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
      TransB, false, is_beta_zero, element_t,
      GemmMemoryType>::get_ideal_cube_depth(SB_Handle::get_num_compute_units(),
                                            rows, cols, gemm_wrapper.k_);

  /* In some cases, use the tsgemm kernel as a normal gemm operation */
  if (depth == 1 || gemm_wrapper.k_ <= 2048) {
    GemmPartial<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
                TransA, TransB, true, is_beta_zero, element_t, GemmMemoryType>
        gemm_partial(gemm_wrapper.a_, gemm_wrapper.b_, gemm_wrapper.c_,
                     gemm_wrapper.alpha_, gemm_wrapper.beta_, 1);
    auto events = execute(gemm_partial, dependencies);

    return events;
  }
  /* Else use the tall and skinny algorithm */
  constexpr bool is_usm = std::is_pointer<typename input_t::container_t>::value;

  /* First step: partial gemm */
  /* Create the cube buffer that will hold the output of the partial gemm */
  auto cube_buffer = acquire_temp_mem < is_usm ? helper::AllocType::usm
                                               : helper::AllocType::buffer,
       element_t > (rows * cols * depth);

  /* Create a first matrix view used for the partial gemm */
  auto cube_gemm =
      make_matrix_view<col_major>(cube_buffer, rows, cols * depth, rows);
  /* Execute the partial gemm operation */
  /* Note: we set is_beta_zero to true regardless of the value of beta
   * because this option is meant for use with a simple Gemm only */
  GemmPartial<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
              TransA, TransB, false, true, element_t, GemmMemoryType>
      gemm_partial(gemm_wrapper.a_, gemm_wrapper.b_, cube_gemm,
                   gemm_wrapper.alpha_, gemm_wrapper.beta_, depth);
  auto events = execute(gemm_partial, dependencies);

  /* Create a second view used for the reduction */
  auto cube_reduction =
      make_matrix_view<col_major>(cube_buffer, rows * cols, depth, rows * cols);
  using CubeType = decltype(cube_reduction);
  constexpr auto reductions_per_thread = 64;
  constexpr int work_group_size = tile_type::wg_rows * tile_type::wg_cols;
  using params_t =
      blas::ReductionParams<index_t, element_t, ClSize, work_group_size,
                            reductions_per_thread,
                            static_cast<int>(reduction_dim_t::outer)>;
  /* Second step: reduction */
  /* Best case: we can reduce directly in C */
  if (is_beta_zero && ldc == rows) {
    Reduction<blas::AddOperator, params_t, CubeType, output_t> reduction(
        cube_reduction, gemm_wrapper.c_);
    events = concatenate_vectors(events, execute(reduction, events));
  }
  /* Otherwise we reduce to a temporary buffer */
  else {
    /* Create a temporary buffer to hold alpha * A * B */
    auto temp_buffer = acquire_temp_mem < is_usm ? helper::AllocType::usm
                                                 : helper::AllocType::buffer,
         element_t > (rows * cols);
    auto temp = make_matrix_view<col_major>(temp_buffer, rows, cols, rows);

    /* Execute the reduction */
    Reduction<blas::AddOperator, params_t, CubeType, output_t> reduction(
        cube_reduction, temp);
    events = concatenate_vectors(events, execute(reduction, events));

    /* If beta is zero, simply do a 2D copy from the temp buffer to C */
    if (is_beta_zero) {
      auto assignOp = make_op<Assign>(gemm_wrapper.c_, temp);
      events = concatenate_vectors(events, execute(assignOp, events));
    }
    /* Else add temp and beta * C and then assign to C */
    else {
      auto scalOp = make_op<ScalarOp, ProductOperator>(gemm_wrapper.beta_,
                                                       gemm_wrapper.c_);
      auto addOp = make_op<BinaryOp, AddOperator>(temp, scalOp);
      auto assignOp = make_op<Assign>(gemm_wrapper.c_, addOp);
      events = concatenate_vectors(events, execute(assignOp, events));
    }

    release_temp_mem(events, temp_buffer);
  }

  release_temp_mem(events, cube_buffer);

  return events;
}

/* GemmPartial */
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          bool IsFinal, bool IsBetaZero, typename element_t, int GemmMemoryType>
inline typename SB_Handle::event_t SB_Handle::execute(
    GemmPartial<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
                TransA, TransB, IsFinal, IsBetaZero, element_t, GemmMemoryType>
        gemm_partial,
    const typename SB_Handle::event_t& dependencies) {
  auto gemm_partial_range =
      gemm_partial.get_nd_range(SB_Handle::get_num_compute_units());
  return {execute_tree<
      Choose<GemmMemoryType == static_cast<int>(gemm_memory_t::local), int,
             using_local_memory::enabled, using_local_memory::disabled>::type>(
      q_, gemm_partial, gemm_partial_range.get_local_range()[0],
      gemm_partial_range.get_global_range()[0], gemm_partial.local_memory_size,
      dependencies)};
}

/* ReductionPartial */
template <typename operator_t, typename params_t, typename input_t,
          typename output_t>
inline typename SB_Handle::event_t SB_Handle::execute(
    Reduction<operator_t, params_t, input_t, output_t> reduction,
    const typename SB_Handle::event_t& dependencies) {
  auto step_range = reduction.get_nd_range(SB_Handle::get_num_compute_units());

  return {execute_tree<using_local_memory::enabled>(
      q_, reduction, step_range.get_local_range()[0],
      step_range.get_global_range()[0], params_t::get_local_memory_size(),
      dependencies)};
}

}  // namespace blas

#endif  // PORTBLAS_HANDLE_HPP
